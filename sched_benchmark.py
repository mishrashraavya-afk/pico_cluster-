import subprocess
import csv
import json
import re
import os
import sys
import argparse
import time
import shutil
import tempfile
import traceback
import signal
import socket
import urllib.parse
import urllib.request
import http.server
import threading
import webbrowser
import getpass
import pwd
from functools import partial
from dataclasses import dataclass
from typing import Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGPATH = os.path.join(SCRIPT_DIR, "logs")
CSV_PATH = os.path.join(SCRIPT_DIR, "csv")
TRACE_PATH = os.path.join(SCRIPT_DIR, "traces")
TEMP_PATH = "/sys/class/thermal/thermal_zone1/temp"
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")


def _ensure_output_path(path: str) -> str:
    """Create parent directories and ensure the target file can be written."""
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)

    if os.geteuid() == 0:
        sudo_user = os.environ.get("SUDO_USER") or getpass.getuser()
        try:
            pw = pwd.getpwnam(sudo_user)
            if not os.path.exists(path):
                with open(path, "a", encoding="utf-8"):
                    pass
            os.chown(parent, pw.pw_uid, pw.pw_gid)
            if os.path.exists(path):
                os.chown(path, pw.pw_uid, pw.pw_gid)
        except Exception as exc:
            print(f"Warning: could not adjust ownership for {path}: {exc}", file=sys.stderr)

    return path


def _load_env_file(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


for key, value in _load_env_file(ENV_PATH).items():
    os.environ.setdefault(key, value)

# Defaults come from the local .env file when present; fall back to hardcoded values otherwise.
# Hardcoded per user request:
DEFAULT_ELECTRICITYMAPS_API_KEY = "em_neypzepbdMZzTjp2YJkPBRTuj7zucmuM"
DEFAULT_ELECTRICITYMAPS_ZONE = "US-CAL-CISO"
DEFAULT_ELECTRICITY_PRICES = {
    "US-CAL-CISO": 72.0,
    "US-CA": 72.0,
    "DE": 80.0,
    "FR": 85.0,
    "GB": 90.0,
}

BENCHMARKS = {
    "sysbench": ["cpu", "threads", "mutex", "memory"],
    "wrk": ["nginx"], 
    "stress_ng": ["cpu", "vm", "io", "matrix"],
    "npb": ["bt"],  # add sp, cg, ep, ft, is, lu, mg later if you want

}
SCHEDULERS = ["EEVDF", "scx_bpfland", "scx_rlfifo", "scx_flow", "scx_lavd", "scx_rusty", "scx_simple"]

# --- NPB configuration ---
NPB_BIN_DIR = "."        # directory holding compiled NPB-MPI binaries (set to an
                         # absolute path unless you run this script from the NPB dir)
NPB_CLASS = "A"          # problem size: S, W, A, B, C, D, E
NPB_NPROCS = 4           # MUST match the compiled binary; BT/SP need a perfect square
NPB_MPIRUN = "mpirun"    # or "mpiexec"

# --- scheduler load/verify timing ---
SCHED_ATTACH_WAIT = 3      # seconds to allow a scheduler to attach before benchmarking
SCHED_SETTLE_WAIT = 2      # seconds to allow full detach before loading the next one
SCX_VERIFY_TIMEOUT = 15    # seconds to wait for sysfs to confirm the active scheduler

# Best-effort confirmation that a scheduler actually attached, read from sysfs.
# If this path is wrong for your kernel the check degrades safely (it never
# skips a scheduler it simply can't verify) — but check it once with:
#     cat /sys/kernel/sched_ext/root/ops
VERIFY_SCX_VIA_SYSFS = True
SCX_OPS_PATH = "/sys/kernel/sched_ext/root/ops"

# --- perf configuration ---
PERF_BIN = "perf"

def _perf_available():
    return shutil.which(PERF_BIN) is not None

def parse_perf_stats(shell_output):
    """
    Parse the human-readable `perf stat` table (printed to stderr) using the
    DEFAULT event set. Each field falls back to 0 / 0.0 if perf didn't emit it
    (e.g. '<not counted>' or '<not supported>' on a kernel without PMU access).
    Returns a 10-field tuple in CSV column order.
    """
    def _find_number(pattern):
        match = re.search(pattern, shell_output, re.IGNORECASE)
        if not match:
            return 0.0 if pattern.endswith("f") else 0
        value = match.group(1).replace(",", "")
        try:
            return float(value) if "." in value or "e" in value.lower() else int(float(value))
        except ValueError:
            return 0.0 if pattern.endswith("f") else 0

    task_clock = _find_number(r"([\d,\.]+)\s+msec task-clock")
    context_switches = _find_number(r"([\d,\.]+)\s+context-switches")
    cpu_migrations = _find_number(r"([\d,\.]+)\s+cpu-migrations")
    page_faults = _find_number(r"([\d,\.]+)\s+page-faults")
    cycles = _find_number(r"([\d,\.]+)\s+(?:cpu-)?cycles")
    instructions = _find_number(r"([\d,\.]+)\s+instructions")
    branches = _find_number(r"([\d,\.]+)\s+branches")
    branch_misses = _find_number(r"([\d,\.]+)\s+branch-misses")
    seconds_user = _find_number(r"([\d,\.]+)\s+seconds user")
    seconds_sys = _find_number(r"([\d,\.]+)\s+seconds sys")

    return (
        float(task_clock),
        int(context_switches),
        int(cpu_migrations),
        int(page_faults),
        int(cycles),
        int(instructions),
        int(branches),
        int(branch_misses),
        float(seconds_user),
        float(seconds_sys),
    )

def _npb_with_perf_stat(base_cmd):
    """
    perf stat wraps the NPB run with the DEFAULT event set (task-clock,
    context-switches, cpu-migrations, page-faults, cycles, instructions,
    branches, branch-misses). No tracepoints -> no tracefs access needed.
    Counters cover the mpirun process tree (MPI ranks are inherited children),
    NOT the whole system. perf prints its table to stderr; NPB results land on
    stdout. Overhead is low, so throughput from this run is still valid.
    Returns (npb_output, perf_text, stats_tuple) where stats_tuple is the
    10-field tuple from parse_perf_stats.
    """
    cmd = [PERF_BIN, "stat", "--", *base_cmd]
    result = subprocess.run(cmd, capture_output=True, text=True)
    npb_output = result.stdout            # NPB results
    perf_text  = result.stderr            # perf stat table (+ mpirun stderr)
    return npb_output, perf_text, parse_perf_stats(perf_text)

def _npb_with_perf_sched(base_cmd, label, trial):
    """
    perf sched record -a wraps the NPB run, then `perf sched latency` is parsed
    for scheduling delay (the wakeup-latency signal that feeds the Grafana
    scheduler-detail panels). WARNING: perf sched record has real overhead and
    inflates runtime — do NOT treat the Mop/s from this run as throughput.
    Returns (npb_output, perf_text, sched_metrics_dict).
    """
    with tempfile.NamedTemporaryFile(prefix=f"sched_{label}_t{trial}_",
                                     suffix=".data", delete=False) as tf:
        data_path = tf.name
    try:
        rec = subprocess.run(
            [PERF_BIN, "sched", "record", "-a", "-o", data_path, "--", *base_cmd],
            capture_output=True, text=True,
        )
        npb_output = rec.stdout

        lat = subprocess.run(
            [PERF_BIN, "sched", "latency", "-i", data_path],
            capture_output=True, text=True,
        )
        perf_text = rec.stderr + "\n----- perf sched latency -----\n" + lat.stdout
        metrics = _parse_perf_sched_latency(lat.stdout)
    finally:
        try:
            os.remove(data_path)     # these files get big — don't keep them
        except OSError:
            pass
    return npb_output, perf_text, metrics

def _parse_perf_sched_latency(text):
    """
    Pull avg/max scheduling delay (in ms) out of `perf sched latency`. The
    'Avg delay' column is time the task was runnable but not running — i.e.
    run-queue / wakeup latency. Column layout varies slightly across perf
    versions, so this matches each task row loosely. If your perf prints a
    different layout, run `perf sched latency` by hand once and tweak the regex.
    avg is switch-count-weighted; max is the worst across all tasks.
    """
    row = re.compile(
        r"\|\s*([0-9.]+)\s*ms\s*\|\s*([0-9]+)\s*\|\s*(?:avg[:=]?\s*)?([0-9.]+)\s*ms"
        r"\s*\|\s*(?:max[:=]?\s*)?([0-9.]+)\s*ms"
    )
    wsum, wcount, max_delay = 0.0, 0, 0.0
    for line in text.splitlines():
        m = row.search(line)
        if not m:
            continue
        switches  = int(m.group(2))
        avg_delay = float(m.group(3))
        wsum   += avg_delay * switches
        wcount += switches
        max_delay = max(max_delay, float(m.group(4)))
    avg = (wsum / wcount) if wcount else None
    return {
        "avg_delay_ms": round(avg, 6) if avg is not None else None,
        "max_delay_ms": max_delay if wcount else None,
    }

@dataclass
class SysbenchOptions:
    threads: int = 32
    time: int = 90
    cpu_max_prime: int = 300000
    memory_block_size: int = 64
    memory_total_size: int = 1024*1024*1024*1024
    memory_scope: str = "global"
    memory_hugetlb: str = "off"
    memory_oper: str = "read"
    memory_access_mode: str = "seq"
    thread_yields: int = 1000
    thread_locks: int = 8
    mutex_num: int = 2048
    mutex_locks: int = 100000
    mutex_loops: int = 400000
    file_num: int = 512
    file_block_size: int = 16384
    file_total_size: int = 1024*1024*1024*8
    file_test_mode: str = "rndrd"
    file_io_mode: str = "sync"
    file_async_backlog: int = 128
    # file_extra_flags: str = "sync,dsync,direct"
    file_fsync_freq: int = 10
    file_fsync_all: str = "off"
    file_fsync_end: str = "on"
    file_fsync_mode: str = "fsync"
    file_merged_requests: int = 0
    file_rw_ratio: int = 1

def run_sysbench(sched, benchmark, workers, timeout_seconds=120, use_rapl=False, trials=10):
    print(f"Running {benchmark} sysbench")
    for trial in range(trials):
        print("trial=", trial)
        command =  f"sudo perf stat sysbench {benchmark}"
        command += f" --threads={SysbenchOptions.threads}"
        command += f" --time={SysbenchOptions.time}"
        if benchmark == "cpu":
            command += f" --cpu-max-prime={SysbenchOptions.cpu_max_prime}"
        elif benchmark == "memory":
            #command += f"--memory-block-size={SysbenchOptions.memory_alloc_size}K"
            command += f" --memory-total-size={SysbenchOptions.memory_total_size} \
                    --memory-scope={SysbenchOptions.memory_scope} \
                    --memory-hugetlb={SysbenchOptions.memory_hugetlb} \
                    --memory-oper={SysbenchOptions.memory_oper} \
                    --memory-access_mode={SysbenchOptions.memory_access_mode}"
        elif benchmark == "threads":
            command += f" --thread-yields={SysbenchOptions.thread_yields} \
                    --thread-locks={SysbenchOptions.thread_locks}"
        elif benchmark == "mutex":
            command += f" --mutex-num={SysbenchOptions.mutex_num} \
                    --mutex-locks={SysbenchOptions.mutex_locks} \
                    --mutex-loops={SysbenchOptions.mutex_loops}"
        elif benchmark == "fileio":
            command += f" --file-num={SysbenchOptions.file_num} \
                    --file-block-size={SysbenchOptions.file_block_size} \
                    --file-total-size={SysbenchOptions.file_total_size} \
                    --file-test-mode={SysbenchOptions.file_test_mode} \
                    --file-io-mode={SysbenchOptions.file_io_mode} \
                    --file-async-backlog={SysbenchOptions.file_async_backlog} \
                    --file-fsync-freq={SysbenchOptions.file_fsync_freq} \
                    --file-fsync-all={SysbenchOptions.file_fsync_all} \
                    --file-fsync-end={SysbenchOptions.file_fsync_end} \
                    --file-fsync-mode={SysbenchOptions.file_fsync_mode} \
                    --file-merged={SysbenchOptions.file_merged_requests} \
                    --file-rw-ratio={SysbenchOptions.file_rw_ratio}"

        command += " run"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout
        perf_output = result.stderr

        total_time_match = re.search(r"total time:\s+([0-9.]+)", output)
        total_events_match = re.search(r"total number of events:\s+([0-9.]+)", output)

        total_time = float(total_time_match.group(1)) if total_time_match else 0.0
        total_events = float(total_events_match.group(1)) if total_events_match else 0.0

        latency_min = float(re.search(r"min:\s+([0-9.]+)", output).group(1)) if re.search(r"min:\s+([0-9.]+)", output) else 0.0
        latency_avg = float(re.search(r"avg:\s+([0-9.]+)", output).group(1)) if re.search(r"avg:\s+([0-9.]+)", output) else 0.0
        latency_max = float(re.search(r"max:\s+([0-9.]+)", output).group(1)) if re.search(r"max:\s+([0-9.]+)", output) else 0.0
        latency_p95 = float(re.search(r"95th percentile:\s+([0-9.]+)", output).group(1)) if re.search(r"95th percentile:\s+([0-9.]+)", output) else 0.0
        latency_sum = float(re.search(r"sum:\s+([0-9.]+)", output).group(1)) if re.search(r"sum:\s+([0-9.]+)", output) else 0.0

        events_match = re.search(r"events \(avg/stddev\):\s+([0-9.]+)/([0-9.]+)", output)
        time_match = re.search(r"execution time \(avg/stddev\):\s+([0-9.]+)/([0-9.]+)", output)

        events_avg = float(events_match.group(1)) if events_match else 0.0
        events_stddev = float(events_match.group(2)) if events_match else 0.0

        time_avg = float(time_match.group(1)) if time_match else 0.0
        time_stddev = float(time_match.group(2)) if time_match else 0.0

        task_clock, context_switches, cpu_migrations, page_faults, cycles, instructions, branches, branch_misses, seconds_user, seconds_sys = parse_perf_stats(perf_output)

        fname = f"{CSV_PATH}/sysbench.csv"
        print(f"\tWriting to {fname}")
        with open(fname, "a", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                sched,
                benchmark,
                trial,
                total_time,
                total_events,
                latency_min,
                latency_avg,
                latency_max,
                latency_p95,
                latency_sum,
                events_avg,
                events_stddev,
                time_avg,
                time_stddev,
                task_clock,
                context_switches,
                cpu_migrations,
                page_faults,
                cycles,
                instructions,
                branches,
                branch_misses,
                seconds_user,
                seconds_sys
            ])
            if use_rapl:
                        append_rapl_metrics(
                            sched,
                            "sysbench",
                            benchmark,
                            trial,
                            runtime_s=float(real_time),
                            workers=workers,
                            filename=os.path.join(SCRIPT_DIR, "rapl_sysbench.csv"),
                        )

@dataclass
class WrkOptions:
    threads = 32
    connections = 100
    duration = "90"
    url = "http://localhost"

def run_wrk(sched, benchmark, workers, timeout_seconds=120, use_rapl=False,trials=10):
    fname = f"{CSV_PATH}/wrk.csv"

    for trial in range(trials):
        print("trial=", trial)

        # Start nginx
        subprocess.run(
            "sudo systemctl start nginx",
            shell=True,
            capture_output=True,
            text=True
        )

        command = f"sudo perf stat wrk"
        command += f" -t{WrkOptions.threads}"
        command += f" -c{WrkOptions.connections}"
        command += f" -d{WrkOptions.duration}"
        command += f" {WrkOptions.url}"

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )

        output = result.stdout + result.stderr
        perf_output = result.stderr
        task_clock, context_switches, cpu_migrations, page_faults, cycles, instructions, branches, branch_misses, seconds_user, seconds_sys = parse_perf_stats(perf_output)


        # Stop nginx
        subprocess.run(
            "sudo systemctl stop nginx",
            shell=True,
            capture_output=True,
            text=True
        )

        latency = float(re.search(r"Latency\s+([\d.]+)", output).group(1)) if re.search(r"Latency\s+([\d.]+)", output) else 0.0
        req_sec = float(re.search(r"Requests/sec:\s+([\d.]+)", output).group(1)) if re.search(r"Requests/sec:\s+([\d.]+)", output) else 0.0
        transfer = float(re.search(r"Transfer/sec:\s+([\d.]+)", output).group(1)) if re.search(r"Transfer/sec:\s+([\d.]+)", output) else 0.0
        total = int(re.search(r"([\d,]+)\s+requests in", output).group(1).replace(",", "")) if re.search(r"([\d,]+)\s+requests in", output) else 0


        with open(fname, "a", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                sched,
                benchmark,
                trial,
                WrkOptions.duration,
                WrkOptions.threads,
                WrkOptions.connections,
                latency,
                req_sec,
                transfer,
                total,
                task_clock,
                context_switches,
                cpu_migrations,
                page_faults,
                cycles,
                instructions,
                branches,
                branch_misses,
                seconds_user,
                seconds_sys
            ])
            if use_rapl:
                        append_rapl_metrics(
                            sched,
                            "wrk",
                            benchmark,
                            trial,
                            runtime_s=float(real_time),
                            workers=workers,
                            filename=os.path.join(SCRIPT_DIR, "rapl_wrk.csv"),
                        )

def run_stressng(sched, stressor, workers, trials=10, timeout_seconds=120, use_rapl=False):
    """Run stress-ng for a scheduler and append both benchmark and perf metrics to CSV."""
    print(f"Running {stressor} stress-ng ({sched})")

    os.makedirs(CSV_PATH, exist_ok=True)
    stressng_log_path = os.path.join(LOGPATH, "stress_ng")
    os.makedirs(stressng_log_path, exist_ok=True)

    fname = _ensure_output_path(os.path.join(CSV_PATH, "stress_ng_final.csv"))
    # Keep the CSV schema stable so every row includes both stress-ng summary values
    # and the perf counters collected alongside the run.
    expected_header = [
        "scheduler",
        "benchmark",
        "trial",
        "bogo_ops",
        "real_time",
        "usr_time",
        "sys_time",
        "bogo_ops_per_sec_real",
        "bogo_ops_per_sec_cpu",
        "task_clock",
        "context_switches",
        "cpu_migrations",
        "page_faults",
        "cycles",
        "instructions",
        "branches",
        "branch_misses",
        "seconds_user",
        "seconds_sys",
    ]

    if not os.path.exists(fname):
        with open(fname, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(expected_header)
    else:
        try:
            with open(fname, "r", newline="") as file:
                rows = list(csv.reader(file))
            if rows and rows[0] != expected_header:
                with open(fname, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(expected_header)
                    writer.writerows(rows[1:])
        except FileNotFoundError:
            pass

    for trial in range(1, trials + 1):

        print(f"\tTrial {trial}/{trials}")

        # Wrap the stress-ng command with perf stat when perf is available so the
        # CSV captures CPU and PMU counters for each trial.
        command = [
            "stress-ng",
            f"--{stressor}",
            str(workers),
            "--timeout",
            f"{timeout_seconds}s",
            "--metrics-brief",
        ]
        if _perf_available():
            command = [PERF_BIN, "stat", "--", *command]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        # Save the raw stress-ng output for debugging and later inspection.
        log_name = _ensure_output_path(os.path.join(stressng_log_path, f"{sched}_{stressor}_trial_{trial}.log"))

        try:
            with open(log_name, "w") as log:
                log.write(output)
        except PermissionError:
            fallback_logs = os.path.join("/tmp", f"trial_perfetto_logs_{getpass.getuser()}")
            os.makedirs(fallback_logs, exist_ok=True)
            fallback_log_name = os.path.join(fallback_logs, os.path.basename(log_name))
            print(f"Permission denied writing {log_name}; writing log to {fallback_log_name} instead")
            with open(fallback_log_name, "w") as log:
                log.write(output)

        if result.returncode != 0:
            print("\tStress-ng failed.")
            continue

        metric_match = re.search(
            r"stress-ng:\s+metrc:\s+\[\d+\]\s+"
            r"\w+\s+"
            r"(\d+)\s+"
            r"([\d.]+)\s+"
            r"([\d.]+)\s+"
            r"([\d.]+)\s+"
            r"([\d.]+)\s+"
            r"([\d.]+)",
            output,
        )

        if metric_match is None:
            print("\tCould not parse metrics.")
            continue

        # Parse the stress-ng summary values and combine them with the perf stats.
        bogo_ops = int(metric_match.group(1))
        real_time = float(metric_match.group(2))
        usr_time = float(metric_match.group(3))
        sys_time = float(metric_match.group(4))
        bogo_ops_per_sec_real = float(metric_match.group(5))
        bogo_ops_per_sec_cpu = float(metric_match.group(6))

        perf_text = result.stderr
        if not perf_text and result.stdout:
            perf_text = result.stdout

        task_clock, context_switches, cpu_migrations, page_faults, cycles, instructions, branches, branch_misses, seconds_user, seconds_sys = parse_perf_stats(perf_text)
        if not any([task_clock, context_switches, cpu_migrations, page_faults, cycles, instructions, branches, branch_misses, seconds_user, seconds_sys]):
            perf_text = output
            task_clock, context_switches, cpu_migrations, page_faults, cycles, instructions, branches, branch_misses, seconds_user, seconds_sys = parse_perf_stats(perf_text)

        row = [
            sched,
            stressor,
            trial,
            bogo_ops,
            real_time,
            usr_time,
            sys_time,
            bogo_ops_per_sec_real,
            bogo_ops_per_sec_cpu,
            task_clock,
            context_switches,
            cpu_migrations,
            page_faults,
            cycles,
            instructions,
            branches,
            branch_misses,
            seconds_user,
            seconds_sys,
        ]

        try:
            with open(fname, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)
        except PermissionError:
            fallback_csv_dir = os.path.join("/tmp", f"trial_perfetto_csv_{getpass.getuser()}")
            os.makedirs(fallback_csv_dir, exist_ok=True)
            fallback_fname = os.path.join(fallback_csv_dir, os.path.basename(fname))
            print(f"Permission denied writing {fname}; appending CSV to {fallback_fname} instead")
            with open(fallback_fname, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

        if use_rapl:
            append_rapl_metrics(
                sched,
                "stress_ng",
                stressor,
                trial,
                runtime_s=float(real_time),
                workers=workers,
                filename=os.path.join(SCRIPT_DIR, "rapl_stressng.csv"),
            )

        print("\tDone")


def run_npb(sched, kernel, trials=10, binary_name="", perf_mode="off"):
    print(f"Running NPB {binary_name or kernel} ({sched})"
          + (f" + perf [{perf_mode}]" if perf_mode != "off" else ""))

    npb_log_path = f"{LOGPATH}/npb"
    os.makedirs(npb_log_path, exist_ok=True)

    if binary_name:
        # exact binary you typed, e.g. "bt.A.4" or "subdir/bt.A.4"
        binary = binary_name if os.path.dirname(binary_name) else f"{NPB_BIN_DIR}/{binary_name}"
        label = os.path.basename(binary_name)
        # derive -np from the trailing number so it always matches the build
        last = label.split(".")[-1]
        nprocs = int(last) if last.isdigit() else NPB_NPROCS
    else:
        binary = f"{NPB_BIN_DIR}/{kernel}.{NPB_CLASS}.{NPB_NPROCS}"
        label = kernel
        nprocs = NPB_NPROCS

    if not os.path.exists(binary):
        print(f"\tBinary not found: {binary}")
        print(f"\tBuild it: make {kernel} CLASS={NPB_CLASS} NPROCS={nprocs}")
        return

    if perf_mode != "off" and not _perf_available():
        print("\tperf not found on PATH — running NPB without perf.")
        perf_mode = "off"

    base_cmd = [NPB_MPIRUN, "-np", str(nprocs), binary]
    fname = _ensure_output_path(f"{CSV_PATH}/npb.csv")

    for trial in range(1, trials + 1):
        print(f"\tTrial {trial}/{trials}")

        perf_stats = (None,) * 10   # task_clock..seconds_sys, from parse_perf_stats
        perf_sched = {}             # avg_delay_ms / max_delay_ms
        if perf_mode == "stat":
            output, perf_text, perf_stats = _npb_with_perf_stat(base_cmd)
        elif perf_mode == "sched":
            output, perf_text, perf_sched = _npb_with_perf_sched(base_cmd, label, trial)
        elif perf_mode == "all":
            # Pass 1 — cheap: clean throughput + scheduler counters.
            # `output` (NPB results) comes from THIS pass, so time/Mop/s are valid.
            output, perf_text_stat, perf_stats = _npb_with_perf_stat(base_cmd)
            # Pass 2 — heavy: scheduling/wakeup delay only. Its NPB output is
            # discarded for metrics (overhead-inflated); we keep just the latency.
            _sched_out, perf_text_sched, perf_sched = _npb_with_perf_sched(base_cmd, label, trial)
            perf_text = (perf_text_stat
                         + "\n\n===== perf sched (separate pass — latency only) =====\n"
                         + perf_text_sched)
        else:
            result = subprocess.run(base_cmd, capture_output=True, text=True)
            output = result.stdout + result.stderr
            perf_text = ""

        log_name = f"{npb_log_path}/{sched}_{label}_trial_{trial}.log"
        with open(log_name, "w") as log:
            log.write(output)
            if perf_text:
                log.write(f"\n\n===== perf ({perf_mode}) =====\n")
                log.write(perf_text)

        time_match   = re.search(r"Time in seconds\s*=\s*([0-9.]+)", output)
        mops_match   = re.search(r"Mop/s total\s*=\s*([0-9.]+)", output)
        mopsp_match  = re.search(r"Mop/s/process\s*=\s*([0-9.]+)", output)
        verify_match = re.search(r"Verification\s*=\s*(\w+)", output)
        verification = verify_match.group(1) if verify_match else None

        # A completed NPB run always prints a Verification line. If it's present
        # and not SUCCESSFUL, the numbers are wrong — stop the whole run hard.
        if verification is not None and verification.upper() != "SUCCESSFUL":
            raise RuntimeError(
                f"NPB verification failed for {sched} / {label} trial {trial}: "
                f"Verification = {verification}. See {log_name}."
            )

        # No parseable metrics => the run didn't complete; skip the row (and keep
        # the raw log, which was already saved above, for diagnosis).
        if time_match is None or mops_match is None:
            print(f"\tCould not parse NPB results — see {log_name}, skipping row.")
            continue

        time_seconds = float(time_match.group(1))
        mops_total   = float(mops_match.group(1))
        # mops_process comes ONLY from its own regex — never substituted from
        # another field — so this column can never hold a stray "SUCCESSFUL".
        mops_process = float(mopsp_match.group(1)) if mopsp_match else None

        with open(fname, "a", newline="") as file:
            csv.writer(file).writerow([
                sched, label, trial, time_seconds, mops_total, mops_process,
                *perf_stats,
            ])

        extra = ""
        if perf_mode in ("stat", "all"):
            extra += (f", ctxsw={perf_stats[1]}, mig={perf_stats[2]}, "
                      f"cycles={perf_stats[4]}, instr={perf_stats[5]}")
        if perf_mode in ("sched", "all"):
            extra += (f", avg_delay={perf_sched.get('avg_delay_ms')}ms, "
                      f"max_delay={perf_sched.get('max_delay_ms')}ms")
        print(f"\tDone — {time_seconds}s, {mops_total} Mop/s, {mops_process} Mop/s/proc{extra}")

def collect_temperature(stop_event, sched, suite, benchmark, filename, trials=10):
    print("Temperature thread started")
    temperatures = []

    while not stop_event.is_set():

        try:
            with open("/sys/class/thermal/thermal_zone1/temp") as f:
                temp = int(f.read().strip()) / 1000.0
                print("Current temperature:", temp)
                temperatures.append(temp)

        except Exception as e:
            print("Temperature error:", e)

        time.sleep(1)


    if not temperatures:
        print("No temperature samples collected")
        return


    temp_min = min(temperatures)
    temp_max = max(temperatures)
    temp_avg = sum(temperatures) / len(temperatures)


    print("Temperature results:")
    print("Min:", temp_min)
    print("Max:", temp_max)
    print("Avg:", temp_avg)


    exists = os.path.exists(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow([
                "scheduler",
                "suite",
                "benchmark",
                "trial",
                "temp_min",
                "temp_max",
                "temp_avg"
            ])

        writer.writerow([
            sched,
            suite,
            benchmark,
            trials,
            temp_min,
            temp_max,
            temp_avg
        ])

def run_benchmark(sched: str, suite: str, benchmark: str, trials: int,
                  npb_binary: str = "", perf_mode: str = "off", timeout: int = 120,
                  use_rapl: bool = False, use_temperature: bool = False) -> None:
    
    ''' temp_stop_event = None
    temp_thread = None

    if use_temperature:
        temp_stop_event = threading.Event()
        temperature_file = os.path.join(CSV_PATH, "temperature.csv")

        temp_thread = threading.Thread(
            target=collect_temperature,
            args=(
                temp_stop_event,
                sched,
                suite,
                benchmark,
                os.path.join(CSV_PATH, "temperature.csv"),
                trials
            )
    )
    #temp_thread.start() '''
    

    # TODO: Start power monitor
    if suite == "sysbench":
        run_sysbench(sched, benchmark, trials)

    elif suite == "wrk":
        run_wrk(sched, benchmark, trials)
    elif suite == "stress_ng":
        workers = {
            "cpu": 4,
            "vm": 4,
            "io": 4,
            "matrix": 4,
        }[benchmark]
        run_stressng(sched, benchmark, workers, trials, timeout, use_rapl=use_rapl)
    elif suite == "npb":
        run_npb(sched, benchmark, trials, npb_binary, perf_mode)

    if use_temperature:
        temp_stop_event.set()
        temp_thread.join()

    if suite == "stress_ng":
        return

    if not use_rapl:
        return

    if suite == "sysbench":
        cmd = ["sysbench", benchmark, "run"]
    elif suite == "wrk":
        cmd = ["wrk", "-t32", "-c100", "-d90s", "http://localhost"]
    elif suite == "npb":
        binary = f"{NPB_BIN_DIR}/{benchmark}.{NPB_CLASS}.{NPB_NPROCS}"
        cmd = [NPB_MPIRUN, "-np", str(NPB_NPROCS), binary]
    else:
        return

    collect_rapl(
        sched,
        suite,
        benchmark,
        1,
        cmd,
        timeout=timeout,
        filename=os.path.join(SCRIPT_DIR, "rapl.csv"),
    )


CSV_COLUMNS = [
    "scheduler",
    "suite",
    "benchmark",
    "trial",
    "package_j",
    "core_j",
    "dram_j",
    "runtime_s",
    "power_w",
    "package_kwh",
    "carbon_intensity_g_per_kwh",
    "estimated_co2_g",
    "electricity_price_eur_per_mwh",
    "estimated_cost",
    "renewable_percentage",
    "fossil_percentage",
    "solar_pct",
    "wind_pct",
    "hydro_pct",
    "nuclear_pct",
    "gas_pct",
    "coal_pct",
    "biomass_pct",
    "geothermal_pct",
    "battery_pct",
    "unknown_pct",
    "zone",
    "timestamp",
]

STRESSNG_WORKERS = {
    "cpu": 4,
    "vm": 4,
    "io": 4,
    "matrix": 4,
}


def run_command_directly(cmd_list, timeout: int = 300) -> tuple[str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    elapsed = time.perf_counter() - start
    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        combined += proc.stderr
    return combined, elapsed


def convert_joules_to_kwh(joules: Optional[float]) -> Optional[float]:
    if joules is None:
        return None
    return joules / 3_600_000.0


def estimate_energy_metrics(runtime_s: Optional[float], benchmark: str, workers: int = 4) -> Dict[str, Optional[float]]:
    if runtime_s is None or runtime_s <= 0:
        runtime_s = 1.0

    workload_factor = {
        "cpu": 1.0,
        "vm": 0.9,
        "io": 0.75,
        "matrix": 1.2,
    }.get(benchmark, 1.0)

    estimated_power_w = 45.0 * workload_factor * min(1.0 + (workers / 8.0), 1.6)
    package_j = estimated_power_w * runtime_s
    core_j = package_j * 0.7
    dram_j = package_j * 0.2
    return {
        "package_j": package_j,
        "core_j": core_j,
        "dram_j": dram_j,
        "runtime_s": float(runtime_s),
        "power_w": estimated_power_w,
    }


def _fetch_json(url: str, api_key: str) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"auth-token": api_key})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)
    except Exception as exc:
        print(f"Warning: could not fetch Electricity Maps data from {url}: {exc}", file=sys.stderr)
        return None


def _extract_numeric(data: Optional[dict], keys: tuple[str, ...]) -> Optional[float]:
    if not isinstance(data, dict):
        return None
    for key in keys:
        if key in data and data[key] not in (None, ""):
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return None
    return None


def _extract_generation_mix(data: Optional[dict]) -> Dict[str, Optional[float]]:
    if not isinstance(data, dict):
        return {}
    mix: Dict[str, Optional[float]] = {}
    for key in ("solar", "wind", "hydro", "nuclear", "gas", "coal", "biomass", "geothermal", "battery", "oil", "unknown"):
        mix[key] = _extract_numeric(data, (key,))
    return mix

def get_electricity_maps_data(api_key: Optional[str], zone: Optional[str]) -> Dict[str, Optional[object]]:
    if not api_key or not zone:
        return {
            "carbon_intensity": None,
            "renewable_percentage": None,
            "fossil_percentage": None,
            "solar": None,
            "wind": None,
            "hydro": None,
            "coal": None,
            "gas": None,
            "nuclear": None,
            "battery": None,
            "biomass": None,
            "geothermal": None,
            "unknown": None,
            "price": None,
            "timestamp": None,
            "zone": zone,
        }

    carbon_data = _fetch_json(f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}", api_key)
    power_data = _fetch_json(f"https://api.electricitymap.org/v3/power-breakdown/latest?zone={zone}", api_key)
    price_data = _fetch_json(f"https://api.electricitymap.org/v3/electricity-price/latest?zone={zone}", api_key)

    carbon_intensity = None
    if isinstance(carbon_data, dict):
        carbon_intensity = _extract_numeric(carbon_data, ("carbonIntensity", "carbon_intensity", "value"))

    mix_data = power_data
    if isinstance(power_data, dict):
        for key in ("powerBreakdown", "powerProductionBreakdown", "productionBreakdown", "breakdown"):
            if key in power_data and isinstance(power_data[key], dict):
                mix_data = power_data[key]
                break

    generation_mix = _extract_generation_mix(mix_data)

    if isinstance(power_data, dict):
        print(json.dumps(power_data, indent=2), file=sys.stderr)

    total_generation = sum(value for value in generation_mix.values() if value is not None)
    if total_generation > 0:
        for source in generation_mix:
            if generation_mix[source] is not None:
                generation_mix[source] = generation_mix[source] / total_generation * 100.0

    price = None
    if isinstance(price_data, dict):
        price = _extract_numeric(price_data, ("price", "value", "pricePerMwh"))
    if price is None and zone:
        price = DEFAULT_ELECTRICITY_PRICES.get(zone)
        if price is not None:
            print(f"Warning: using fallback electricity price of {price} €/MWh for zone {zone}", file=sys.stderr)

    renewable_percentage = sum(
        generation_mix.get(src, 0) or 0
        for src in ("solar", "wind", "hydro", "biomass", "geothermal", "battery")
    )
    fossil_percentage = sum(
        generation_mix.get(src, 0) or 0
        for src in ("coal", "gas", "oil")
    )

    timestamp = None
    if isinstance(carbon_data, dict):
        timestamp = carbon_data.get("datetime") or carbon_data.get("timestamp") or carbon_data.get("updatedAt")
    if timestamp is None and isinstance(price_data, dict):
        timestamp = price_data.get("datetime") or price_data.get("timestamp") or price_data.get("updatedAt")

    return {
        "carbon_intensity": carbon_intensity,
        "renewable_percentage": renewable_percentage,
        "fossil_percentage": fossil_percentage,
        "solar": generation_mix.get("solar"),
        "wind": generation_mix.get("wind"),
        "hydro": generation_mix.get("hydro"),
        "coal": generation_mix.get("coal"),
        "gas": generation_mix.get("gas"),
        "nuclear": generation_mix.get("nuclear"),
        "battery": generation_mix.get("battery"),
        "biomass": generation_mix.get("biomass"),
        "geothermal": generation_mix.get("geothermal"),
        "unknown": generation_mix.get("unknown"),
        "price": price,
        "timestamp": timestamp,
        "zone": zone,
    }

def print_summary(scheduler: str, benchmark: str, metrics: Dict[str, Optional[float]], grid_data: Dict[str, Optional[object]], package_kwh: Optional[float]) -> None:
    carbon_intensity = grid_data.get("carbon_intensity")
    estimated_co2_g = grid_data.get("estimated_co2_g")
    price = grid_data.get("price")
    estimated_cost = grid_data.get("estimated_cost")
    renewable_percentage = grid_data.get("renewable_percentage")
    fossil_percentage = grid_data.get("fossil_percentage")

    print(f"Scheduler: {scheduler}")
    print(f"Benchmark: {benchmark}")
    print()
    print(f"Energy:        {package_kwh:.5f} kWh" if package_kwh is not None else "Energy:        n/a")
    print(f"Carbon:        {carbon_intensity} gCO₂/kWh" if carbon_intensity is not None else "Carbon:        n/a")
    print(f"Emissions:     {estimated_co2_g:.5f} gCO₂" if estimated_co2_g is not None else "Emissions:     n/a")
    print()
    print(f"Price:         €{price}/MWh" if price is not None else "Price:         n/a")
    print(f"Cost:          €{estimated_cost:.5f}" if estimated_cost is not None else "Cost:          n/a")
    print()
    print(f"Renewables:    {renewable_percentage:.0f}%" if renewable_percentage is not None else "Renewables:    n/a")
    print(f"Fossil:        {fossil_percentage:.0f}%" if fossil_percentage is not None else "Fossil:        n/a")
    print()
    print(f"Solar:         {grid_data.get('solar')}%" if grid_data.get("solar") is not None else "Solar:         n/a")
    print(f"Wind:          {grid_data.get('wind')}%" if grid_data.get("wind") is not None else "Wind:          n/a")
    print(f"Hydro:         {grid_data.get('hydro')}%" if grid_data.get("hydro") is not None else "Hydro:         n/a")
    print(f"Nuclear:       {grid_data.get('nuclear')}%" if grid_data.get("nuclear") is not None else "Nuclear:       n/a")
    print(f"Gas:           {grid_data.get('gas')}%" if grid_data.get("gas") is not None else "Gas:           n/a")
    print(f"Coal:          {grid_data.get('coal')}%" if grid_data.get("coal") is not None else "Coal:          n/a")
    print()

def append_rapl_metrics(scheduler: str, suite: str, benchmark: str, trial: int,
                        runtime_s: float, workers: Optional[int] = None,
                        filename: str = "rapl.csv"):
    """Estimate energy metrics from a known runtime and append one CSV row."""
    metrics = estimate_energy_metrics(runtime_s, benchmark=benchmark, workers=workers if workers is not None else STRESSNG_WORKERS.get(benchmark, 4))
    package_kwh = convert_joules_to_kwh(metrics.get("package_j"))
    grid_data = get_electricity_maps_data(
        os.environ.get("ELECTRICITYMAPS_API_KEY", DEFAULT_ELECTRICITYMAPS_API_KEY),
        os.environ.get("ELECTRICITYMAPS_ZONE", DEFAULT_ELECTRICITYMAPS_ZONE),
    )

    if package_kwh is not None and grid_data.get("carbon_intensity") is not None:
        grid_data["estimated_co2_g"] = package_kwh * float(grid_data["carbon_intensity"])

    if package_kwh is not None and grid_data.get("price") is not None:
        price_per_kwh = float(grid_data["price"]) / 1000.0
        grid_data["estimated_cost"] = package_kwh * price_per_kwh

    grid_data["package_kwh"] = package_kwh

    filename = _ensure_output_path(filename)
    exists = os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(CSV_COLUMNS)

        pkg = metrics.get("package_j")
        rt = metrics.get("runtime_s")
        power = ""
        try:
            if pkg is not None and rt:
                if float(rt) > 0:
                    power = float(pkg) / float(rt)
        except Exception:
            power = ""

        row = [
            scheduler,
            suite,
            benchmark,
            trial,
            metrics.get("package_j", ""),
            metrics.get("core_j", ""),
            metrics.get("dram_j", ""),
            metrics.get("runtime_s", ""),
            power,
            package_kwh if package_kwh is not None else "",
            grid_data.get("carbon_intensity", "") if grid_data.get("carbon_intensity") is not None else "",
            grid_data.get("estimated_co2_g", "") if grid_data.get("estimated_co2_g") is not None else "",
            grid_data.get("price", "") if grid_data.get("price") is not None else "",
            grid_data.get("estimated_cost", "") if grid_data.get("estimated_cost") is not None else "",
            grid_data.get("renewable_percentage", "") if grid_data.get("renewable_percentage") is not None else "",
            grid_data.get("fossil_percentage", "") if grid_data.get("fossil_percentage") is not None else "",
            grid_data.get("solar", "") if grid_data.get("solar") is not None else "",
            grid_data.get("wind", "") if grid_data.get("wind") is not None else "",
            grid_data.get("hydro", "") if grid_data.get("hydro") is not None else "",
            grid_data.get("nuclear", "") if grid_data.get("nuclear") is not None else "",
            grid_data.get("gas", "") if grid_data.get("gas") is not None else "",
            grid_data.get("coal", "") if grid_data.get("coal") is not None else "",
            grid_data.get("biomass", "") if grid_data.get("biomass") is not None else "",
            grid_data.get("geothermal", "") if grid_data.get("geothermal") is not None else "",
            grid_data.get("battery", "") if grid_data.get("battery") is not None else "",
            grid_data.get("unknown", "") if grid_data.get("unknown") is not None else "",
            grid_data.get("zone", "") if grid_data.get("zone") is not None else "",
            grid_data.get("timestamp", "") if grid_data.get("timestamp") is not None else "",
        ]
        w.writerow(row)

    print_summary(scheduler, benchmark, metrics, grid_data, package_kwh)
    return {
        "package_j": metrics.get("package_j"),
        "core_j": metrics.get("core_j"),
        "dram_j": metrics.get("dram_j"),
        "runtime_s": metrics.get("runtime_s"),
        "power_w": metrics.get("power_w"),
        "package_kwh": package_kwh,
        "carbon_intensity_g_per_kwh": grid_data.get("carbon_intensity"),
        "estimated_co2_g": grid_data.get("estimated_co2_g"),
        "electricity_price_eur_per_mwh": grid_data.get("price"),
        "estimated_cost": grid_data.get("estimated_cost"),
        "renewable_percentage": grid_data.get("renewable_percentage"),
        "fossil_percentage": grid_data.get("fossil_percentage"),
        "solar_pct": grid_data.get("solar"),
        "wind_pct": grid_data.get("wind"),
        "hydro_pct": grid_data.get("hydro"),
        "nuclear_pct": grid_data.get("nuclear"),
        "gas_pct": grid_data.get("gas"),
        "coal_pct": grid_data.get("coal"),
        "biomass_pct": grid_data.get("biomass"),
        "geothermal_pct": grid_data.get("geothermal"),
        "battery_pct": grid_data.get("battery"),
        "unknown_pct": grid_data.get("unknown"),
        "zone": grid_data.get("zone"),
        "timestamp": grid_data.get("timestamp"),
    }


def collect_rapl(scheduler: str, suite: str, benchmark: str, trial: int, cmd_list,
                 timeout: int = 120, sudo_perf: bool = False, allow_missing_energy: bool = False,
                 filename: str = "rapl.csv"):
    """Run the benchmark command, estimate energy metrics, optionally fetch carbon intensity, and append a CSV row."""
    if not cmd_list:
        raise ValueError("cmd_list must not be empty")

    try:
        cmd_to_run = list(cmd_list)
        if sudo_perf:
            cmd_to_run = ["sudo", *cmd_to_run]
        raw, runtime_s = run_command_directly(cmd_to_run, timeout=timeout + 30)
        metrics = estimate_energy_metrics(runtime_s, benchmark=benchmark, workers=STRESSNG_WORKERS.get(benchmark, 4))
    except subprocess.TimeoutExpired:
        print("Benchmark timed out", file=sys.stderr)
        raise
    except Exception as exc:
        print(f"run error: {exc}", file=sys.stderr)
        raise

    return append_rapl_metrics(
        scheduler,
        suite,
        benchmark,
        trial,
        runtime_s=float(metrics.get("runtime_s") or 0.0),
        workers=STRESSNG_WORKERS.get(benchmark, 4),
        filename=filename,
    )
    
def run_perfetto_trace(
    scheduler: str,
    suite: str,
    benchmark: str,
    benchmark_function,
    serve_host: str = None,
    serve_port: int = 9001,
    forward_port: int = None,
    open_browser: bool = True,
) -> None:
    # Run tracebox while executing a benchmark and launch the Perfetto UI afterwards.

    script_dir = SCRIPT_DIR
    trace_dir = TRACE_PATH
    os.makedirs(trace_dir, exist_ok=True)
    trace_file = os.path.join(trace_dir, f"{scheduler}_{suite}_{benchmark}.pftrace")
    tracebox_path = os.path.join(script_dir, "tracebox")
    config_file = os.path.abspath(os.path.join(script_dir, "config.pbtx"))

    print(
        "========================================\n"
        f"Starting tracebox Recording\n"
        f"Scheduler : {scheduler}\n"
        f"Suite : {suite}\n"
        f"Benchmark : {benchmark}\n"
        f"Output : {trace_file}\n"
        "========================================"
    )

    # Configure the tracebox command that will record the benchmark run to a .pftrace file.
    tracebox_cmd = [
        tracebox_path,
        "--txt",
        "-c",
        config_file,
        "-o",
        trace_file,
    ]

    sudo_check = subprocess.run(
        ["sudo", "-n", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    is_root = os.geteuid() == 0
    if not is_root:
        tracebox_cmd = ["sudo"] + tracebox_cmd
        if sudo_check.returncode == 0:
            print("Using passwordless sudo to launch tracebox")
        else:
            print("Using sudo to launch tracebox; you may be prompted for your password")
    else:
        print("Running tracebox as root")

    # Capture stderr when sudo is available so tracebox errors are still visible.
    use_pipe_stderr = is_root or sudo_check.returncode == 0
    try:
        perfetto_proc = subprocess.Popen(
            tracebox_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if use_pipe_stderr else None,
            stdin=sys.stdin if not use_pipe_stderr else subprocess.DEVNULL,
            text=True,
            preexec_fn=os.setsid if not is_root else None,
        )
    except Exception as exc:
        print(f"Failed to start tracebox: {exc}")
        return

    print("tracebox=========================running...")
    time.sleep(2)

    # Run the requested benchmark while the trace is being recorded.
    try:
        benchmark_function()
    finally:
        print("Stopping tracebox recording...")
        if not is_root:
            os.killpg(os.getpgid(perfetto_proc.pid), signal.SIGINT)
        else:
            perfetto_proc.send_signal(signal.SIGINT)
        try:
            perfetto_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("tracebox did not exit in time, killing process...")
            if not is_root:
                os.killpg(os.getpgid(perfetto_proc.pid), signal.SIGKILL)
            else:
                perfetto_proc.kill()
            perfetto_proc.wait(timeout=10)

    stdout, stderr = perfetto_proc.communicate()
    if stdout:
        print(stdout)
    if stderr is not None:
        print(stderr)
    else:
        print("stderr is not captured because tracebox was launched with interactive sudo.")
    print("Return code:", perfetto_proc.returncode)

    print("Trace exists:", os.path.exists(trace_file))
    print("Absolute trace path:", os.path.abspath(trace_file))
    print("Directory contents:", os.listdir(os.path.dirname(trace_file)))

    # If the trace was produced successfully, either serve it through the Perfetto UI
    # or just save it and exit cleanly.
    if os.path.exists(trace_file) and os.path.getsize(trace_file) > 0:
        size = os.path.getsize(trace_file)
        print(
            "========================================\n"
            f"tracebox Trace Saved\n"
            f"Location : {trace_file}\n"
            f"Size : {size}\n"
            "========================================"
        )

        if not open_browser:
            print("Skipping Perfetto UI launch because --perfetto-no-open-browser was requested.")
            return

        try:
            server, ssh_process = launch_perfetto_ui(
                trace_file,
                serve_host=serve_host,
                serve_port=serve_port,
                forward_port=forward_port,
                open_browser=open_browser,
            )
        except Exception as exc:
            print("launch_perfetto_ui failed:", exc)
            traceback.print_exc()
            return
        print("\nPerfetto UI should now be available in your browser.")
        print("Keep this process running while you inspect the trace, then press Ctrl+C to stop the HTTP server.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down Perfetto trace server...")
        finally:
            server.shutdown()
            server.server_close()
            if ssh_process is not None:
                ssh_process.terminate()
    else:
        print(f"tracebox trace missing after stop: {trace_file}")

def launch_perfetto_ui(trace_file, serve_host=None, serve_port=9001, forward_port=None, open_browser=True):
    """Serve a generated Perfetto trace and open it in the Perfetto UI.

    The workflow is intentionally simple:
    1. Start a tiny HTTP server from the trace directory.
    2. Build a URL that points to the trace file.
    3. If a remote host is provided, optionally tunnel the local port to that host.
    4. Open the Perfetto UI with the trace URL.
    """

    # Pick the HTTP port for the trace server and derive the trace file details.
    port = serve_port or 9001
    trace_file = os.path.abspath(trace_file)
    trace_dir = os.path.dirname(trace_file)
    trace_name = os.path.basename(trace_file)

    # Serve the trace from the directory that contains it.
    os.chdir(trace_dir)

    print("Entered launch_perfetto_ui()")
    print("PORT =", port)
    print("trace_dir =", trace_dir)
    print("trace_file exists =", os.path.exists(trace_file))
    print("Trace file:", trace_file)

    ssh_connection = "SSH_CONNECTION" in os.environ
    if serve_host:
        host = serve_host
    elif ssh_connection:
        host = socket.getfqdn()
        try:
            host = socket.gethostbyname(host)
        except Exception:
            host = subprocess.check_output(["hostname", "-I"], text=True).split()[0]
    else:
        host = subprocess.check_output(["hostname", "-I"], text=True).split()[0]

    class LoggingHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            print(self.address_string(), "-", format % args)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
            self.end_headers()

    handler = partial(LoggingHandler, directory=trace_dir)

    # Start a lightweight HTTP server so the trace file can be loaded by Perfetto.
    print("Creating HTTP server...")
    try:
        server = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    except OSError as exc:
        if exc.errno == 98:
            raise RuntimeError(f"Port {port} is already in use. Free it or choose a different --perfetto-port.")
        raise

    print("HTTP server successfully created")
    print("Listening on", server.server_address)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("Thread alive:", thread.is_alive())

    # Build the direct URLs that the Perfetto UI will use.
    local_url = f"http://127.0.0.1:{port}/{urllib.parse.quote(trace_name)}"
    remote_url = f"http://{host}:{port}/{urllib.parse.quote(trace_name)}"
    ssh_process = None

    if serve_host and forward_port and not ssh_connection:
        # When a remote server is used, expose the trace on the local forwarded port.
        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            "-L",
            f"{forward_port}:127.0.0.1:{port}",
            serve_host,
        ]
        print("Starting SSH tunnel:", " ".join(ssh_cmd))
        try:
            ssh_process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(1)
        except Exception as exc:
            print("Failed to start SSH tunnel:", exc)

        accessible_url = f"http://127.0.0.1:{forward_port}/{urllib.parse.quote(trace_name)}"
    elif serve_host or ssh_connection:
        accessible_url = remote_url
    else:
        accessible_url = local_url

    ui_url = f"http://ui.perfetto.dev/#!/?url={urllib.parse.quote(accessible_url, safe=':/?&=')}"

    print("Perfetto trace server URL:", accessible_url)
    print("Open this URL in the Perfetto UI if the browser does not launch automatically.")
    print(f"Trace URL: {local_url}")
    print(f"Remote trace URL: {remote_url}")

    if open_browser:
        print(f"Opening Perfetto UI URL: {ui_url}")
        opened = _open_browser_url(ui_url)
        if not opened:
            print("Unable to open browser automatically. Open this URL manually:")
            print(ui_url)

    return server, ssh_process

def _open_browser_url(url):
    if not url:
        return False

    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        env = os.environ.copy()
        for key in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY"):
            if key in os.environ:
                env[key] = os.environ[key]
        try:
            subprocess.run(
                ["sudo", "-u", sudo_user, "xdg-open", url],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            print(f"Attempted to open browser as original user: {sudo_user}")
            return True
        except Exception as exc:
            print("Failed to open browser as original user:", exc)

    try:
        return webbrowser.open(url, new=2)
    except Exception as exc:
        print("webbrowser.open failed:", exc)
        return False

def create_files(suite):
    os.makedirs(LOGPATH, exist_ok=True)
    os.makedirs(CSV_PATH, exist_ok=True)
    os.makedirs(TRACE_PATH, exist_ok=True)

    if suite == "sysbench":
        fname = f"{CSV_PATH}/sysbench.csv"
        if os.path.exists(fname):
            return  # File already exists, do not overwrite
        with open(fname, "w") as file:
            writer = csv.writer(file)
            writer.writerow([
                "scheduler",
                "benchmark",
                "trial",
                "total_time",
                "total_events",
                "latency_min",
                "latency_avg",
                "latency_max",
                "latency_p95",
                "latency_sum",
                "events_avg",
                "events_stddev",
                "time_avg",
                "time_stddev",
            ])
    elif suite == "wrk":
        fname = f"{CSV_PATH}/wrk.csv"

        if os.path.exists(fname):
            return

        with open(fname, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "scheduler",
                "benchmark",
                "trial",
                "duration",
                "threads",
                "connections",
                "latency_ms",
                "requests_per_sec",
                "transfer_per_sec",
                "total_requests",
                "task_clock",
                "context_switches",
                "cpu_migrations",
                "page_faults",
                "cycles",
                "instructions",
                "branches",
                "branch_misses",
                "seconds_user",
                "seconds_sys"
            ])
    
            
    elif suite == "stress_ng":
        os.makedirs(os.path.join(LOGPATH, "stress_ng"), exist_ok=True)
        fname = os.path.join(CSV_PATH, "stress_ng_final.csv")
        if os.path.exists(fname):
            return
        with open(fname, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "scheduler",
                "benchmark",
                "trial",
                "bogo_ops",
                "real_time",
                "usr_time",
                "sys_time",
                "bogo_ops_per_sec_real",
                "bogo_ops_per_sec_cpu",
            ])
    elif suite == "npb":
        fname = f"{CSV_PATH}/npb.csv"
        if os.path.exists(fname):
            return
        with open(fname, "w") as file:
            csv.writer(file).writerow([
                "scheduler", "benchmark", "trial",
                "time_seconds", "mops_total", "mops_process",
                "task_clock_msec", "context_switches", "cpu_migrations",
                "page_faults", "cycles", "instructions", "branches",
                "branch_misses", "seconds_user", "seconds_sys",
            ])
    elif suite == "temperature":
        fname = f"{CSV_PATH}/temperature.csv"

        if os.path.exists(fname):
            return

        with open(fname, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "scheduler",
                "suite",
                "benchmark",
                "trial",
                "temp_min",
                "temp_max",
                "temp_avg"
            ])

def scx_active_ops():
    """Active sched_ext scheduler name from sysfs, or None if it can't be read."""
    try:
        with open(SCX_OPS_PATH) as f:
            name = f.read().strip()
            return name or None
    except OSError:
        return None

def confirm_scx_attached(sched):
    """
    Best-effort confirmation that `sched` actually became the active scheduler.

    Returns:
        True  - confirmed active via sysfs
        False - sysfs is readable but a different/no scheduler is active (real failure)
        None  - sysfs unavailable, so we can't verify (caller should NOT skip on this)
    """
    if not VERIFY_SCX_VIA_SYSFS:
        return None
    stem = sched.replace("scx_", "")          # scx_bpfland -> "bpfland"
    deadline = time.time() + SCX_VERIFY_TIMEOUT
    saw_sysfs = False
    while time.time() < deadline:
        ops = scx_active_ops()
        if ops is not None:
            saw_sysfs = True
            if stem in ops:
                return True
        time.sleep(0.5)
    return False if saw_sysfs else None

def unload_scheduler(sched, sched_proc):
    """Kill the scheduler and wait for the kernel to fully detach."""
    subprocess.run(["sudo", "pkill", "-f", os.path.basename(sched)])
    sched_proc.wait()
    time.sleep(SCHED_SETTLE_WAIT)

def build_arg_parser():
    parser = argparse.ArgumentParser(prog="benchmark_data_csv.py")
    parser.add_argument("-s", "--scheduler",  default="", help="Specify a scheduler to run benchmarks with. If not provided, all schedulers will be used.")
    parser.add_argument("-S", "--suite",      default="", help="Specify a benchmark suite to run.")
    parser.add_argument("-b", "--benchmark",  default="", help="Specify a benchmark to run within the suite.")
    parser.add_argument("-t", "--trials", type=int, default=10, help="Specify the number of trials to run for each benchmark.")
    parser.add_argument("--timeout", type=int, default=120, help="Specify the stress-ng timeout in seconds.")
    parser.add_argument("-B", "--npb-binary", default="", help="Exact NPB binary to run, e.g. bt.A.4. Overrides the auto-built name; -np is taken from the trailing number.")
    parser.add_argument("-P", "--perf-mode", choices=["off", "stat", "sched", "all"], default="off",
        help="Profile each NPB trial with perf. 'stat': cheap default-event counters, "
             "throughput valid. 'sched': wakeup delay via perf sched, throughput NOT "
             "valid. 'all': both passes per trial — throughput/counters from the stat "
             "pass, latency from the sched pass (doubles runtime per trial).")
    parser.add_argument("--perfetto", action="store_true", help="Wrap benchmark execution with Perfetto recording.")
    parser.add_argument("--perfetto-host", default=None, help="Remote host or IP to use for serving the Perfetto trace to the browser.")
    parser.add_argument("--perfetto-port", type=int, default=9001, help="Fixed port for the Perfetto trace HTTP server.")
    parser.add_argument("--perfetto-forward-port", type=int, default=None, help="Local port to use for SSH forwarding when serving from a remote host.")
    parser.add_argument("mode", nargs="*", help="Optional trailing token: 'rapl' to run the collector and save to rapl.csv")
    parser.add_argument("--perfetto-no-open-browser", action="store_true", help="Save the Perfetto trace files without opening the Perfetto UI in a browser.")
    parser.add_argument("--temperature", action="store_true", help="Record CPU temperature while running benchmarks.")
    parser.add_argument("--rapl", action="store_true", help="Collect energy metrics.")
    return parser

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    use_rapl = args.rapl or any(token.lower() == "rapl" for token in args.mode)
    use_rapl = args.rapl
    sched_list = [args.scheduler] if args.scheduler else SCHEDULERS
    suite_map  = {args.suite: BENCHMARKS[args.suite]} if args.suite else BENCHMARKS

    for suite, benchmarks in suite_map.items():
        create_files(suite)

    for sched in sched_list:
        sched_proc = None

        if sched != "EEVDF":
            sched_proc = subprocess.Popen(["sudo", sched])
            time.sleep(SCHED_ATTACH_WAIT)

            # 1) The process must still be alive. If it exited, the load failed
            #    (not on PATH, sudo prompt, or EBUSY because the previous
            #    scheduler hadn't fully detached yet).
            if sched_proc.poll() is not None:
                print(f"!! {sched} exited immediately (PATH? sudo? EBUSY?) — skipping")
                continue

            # 2) If sysfs verification is available, the active scheduler must
            #    actually be this one. This is what prevents mislabeled rows
            #    where the scheduler "didn't switch".
            confirmed = confirm_scx_attached(sched)
            if confirmed is False:
                print(f"!! {sched} did not become active (sysfs shows: {scx_active_ops()}) "
                      f"— skipping to avoid mislabeled data")
                unload_scheduler(sched, sched_proc)
                continue
            elif confirmed is True:
                print(f">> {sched} confirmed active ({scx_active_ops()})")
            # confirmed is None => couldn't verify via sysfs; proceed on the
            # strength of the liveness check above.
    for suite, benchmarks in suite_map.items():
        create_files(suite)

        if args.temperature:
            create_files("temperature")
        try:
            for suite, benchmarks in suite_map.items():
                bench_list = [args.benchmark] if args.benchmark else benchmarks
                for benchmark in bench_list:
                    if args.perfetto:
                        run_perfetto_trace(
                            sched,
                            suite,
                            benchmark,
                            lambda sched=sched, suite=suite, benchmark=benchmark: run_benchmark(
                                sched,
                                suite,
                                benchmark,
                                args.trials,
                                args.npb_binary,
                                args.perf_mode,
                                args.timeout,
                                use_rapl,
                            ),
                            serve_host=args.perfetto_host,
                            serve_port=args.perfetto_port,
                            forward_port=args.perfetto_forward_port,
                            open_browser=not args.perfetto_no_open_browser,
                        )
                    else:
                        run_benchmark(sched, suite, benchmark, args.trials,
                                      args.npb_binary, args.perf_mode,
                                      args.timeout, use_rapl, args.temperature)
        finally:
            # Always unload, even if a verification RuntimeError propagates,
            # so a failed run never leaves a scheduler stuck on the kernel.
            if sched_proc is not None:
                unload_scheduler(sched, sched_proc)
