"""Measure reusable hardware costs without an application workload.

Usage: python -m compiler.offload.calibrate --output profile.json --threads 4
       --precision 64 --cuda-host-cxx /usr/bin/g++-14
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .profile import (
    LATENCY_RATES,
    SCHEMA_VERSION,
    THROUGHPUT_RATES,
    TRANSFER_KINDS,
    WORKER_RATES,
    ProfileError,
    validate_profile,
)


class CalibrationError(ValueError):
    """Measurements failed or cannot support a finite physical cost model."""


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise CalibrationError(f"{name} must be a finite positive number")
    return float(value)


def _median(record: dict) -> float:
    values = record.get("seconds")
    if not isinstance(values, list) or len(values) < 3:
        raise CalibrationError("every measurement requires at least three duration samples")
    return statistics.median(_positive(value, "duration") for value in values)


def parse_measurements(text: str) -> list[dict]:
    """Accept only native benchmark JSON lines, preserving raw observations."""
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise CalibrationError("native benchmark emitted a non-JSON measurement") from error
        if not isinstance(record, dict) or record.get("kind") not in {"device", "rate", "latency", "transfer"}:
            raise CalibrationError("native benchmark emitted an unknown measurement kind")
        if record["kind"] != "device":
            _median(record)
        records.append(record)
    if not records:
        raise CalibrationError("native benchmark emitted no measurements")
    return records


def fit_transfer(records: list[dict]) -> dict:
    """Fit serialized transfer time = nonnegative latency + bytes/bandwidth.

    Medians suppress individual scheduling outliers. If unconstrained least
    squares yields negative latency, refit the slope through zero instead of
    reporting an unphysical intercept. Raw samples and fit error remain in
    the profile; this is a representative model, not a throughput guarantee.
    """
    points = []
    for record in records:
        size = record.get("bytes")
        if type(size) is not int or size <= 0:
            raise CalibrationError("transfer bytes must be a positive integer")
        points.append((float(size), _median(record)))
    if len(points) < 3 or len({point[0] for point in points}) != len(points):
        raise CalibrationError("transfer fit requires at least three distinct byte sizes")
    mean_x = statistics.mean(point[0] for point in points)
    mean_y = statistics.mean(point[1] for point in points)
    denominator = math.fsum((x - mean_x) ** 2 for x, _ in points)
    slope = math.fsum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    latency = mean_y - slope * mean_x
    if latency < 0:
        latency = 0.0
        slope = math.fsum(x * y for x, y in points) / math.fsum(x * x for x, _ in points)
    _positive(slope, "transfer seconds per byte")
    bandwidth = 1.0 / slope
    return {
        "latency_seconds": latency,
        "bandwidth_bytes_per_second": bandwidth,
        "fit_max_relative_error": max(abs(latency + x * slope - y) / y for x, y in points),
    }


def profile_from_measurements(
    records: list[dict],
    *,
    precision_bits: int,
    cpu_threads: int,
    cpu_name: str,
    nvcc_version: str,
    host_cxx_version: str,
) -> dict:
    """Build a validated profile; missing measurements never get default rates."""
    devices = [record for record in records if record["kind"] == "device"]
    if len(devices) != 1:
        raise CalibrationError("exactly one device identity record is required")
    device = devices[0]
    if type(device.get("cpu_threads")) is not int or device["cpu_threads"] != cpu_threads:
        raise CalibrationError("measured OpenMP team size differs from the requested thread budget")
    rates: dict[str, Any] = {}
    transfers: dict[str, list[dict]] = {name: [] for name in TRANSFER_KINDS}
    for record in records:
        kind = record["kind"]
        if kind == "device":
            continue
        if kind == "transfer":
            name = str(record.get("direction")) + "_" + str(record.get("memory"))
            if name not in transfers:
                raise CalibrationError(f"unknown transfer measurement: {name}")
            transfers[name].append(record)
            continue
        name = record.get("name")
        allowed = LATENCY_RATES if kind == "latency" else THROUGHPUT_RATES + WORKER_RATES
        if name not in allowed or name in rates:
            raise CalibrationError(f"unknown or duplicate {kind} measurement: {name}")
        rates[name] = _median(record) if kind == "latency" else _positive(record.get("work"), "work") / _median(record)
    for name, observations in transfers.items():
        rates[name] = fit_transfer(observations)
    if cpu_threads == 1:
        for name in WORKER_RATES:
            if name in rates:
                raise CalibrationError("single-thread calibration must not report nonexistent CPU workers")
            rates[name] = 0.0
    profile = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {name: device.get(name) for name in ("gpu_uuid", "gpu_name", "compute_capability")}
        | {"cpu_name": cpu_name, "machine": platform.machine(), "device_ordinal": device.get("device_ordinal"),
           "async_engine_count": device.get("async_engine_count")},
        "toolchain": {"nvcc_version": nvcc_version, "host_cxx_version": host_cxx_version,
                      "cuda_runtime_version": device.get("cuda_runtime_version"),
                      "driver_version": device.get("driver_version")},
        "precision_bits": precision_bits,
        "cpu_threads": cpu_threads,
        "rates": rates,
        "measurement_method": {
            "samples": "median of five warmed measurements",
            "transfer": "wall time of serialized cudaMemcpyAsync plus stream synchronization; nonnegative linear fit",
            "launch": "wall time of empty launch plus stream synchronization",
            "pump": "host enqueue time of one pinned 4KiB H2D, empty kernel, and pinned 4KiB D2H; excludes final wait",
            "cpu_compute": "four independent multiply-add recurrences per element, 256 steps, FMA counts as two ops",
            "gpu_compute": "same recurrence workload timed with CUDA events",
            "memory": "array triad; traffic counts two reads and one write",
            "packing": "single-thread gather/scatter at stride four into pinned storage; payload bytes only",
            "cpu_workers": "measured with cpu_threads minus one, reserving one thread for the GPU pump",
            "compatibility": "schema, CPU name, GPU UUID/compute capability, toolchain versions, precision, CPU threads",
        },
        "measurements": records,
    }
    try:
        return validate_profile(profile)
    except ProfileError as error:
        raise CalibrationError(str(error)) from error


def cpu_identity() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _run(argv: list[str], directory: Path, log: Path, *, timeout: int) -> str:
    try:
        result = subprocess.run(argv, cwd=directory, capture_output=True, text=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CalibrationError(f"cannot execute calibration command: {error}") from error
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise CalibrationError(f"calibration command exited {result.returncode}; see {log}: {result.stderr[-2000:]}")
    return result.stdout


def _tool(path: str | None, candidates: tuple[str, ...]) -> str:
    if path:
        resolved = shutil.which(path)
        if resolved:
            return str(Path(resolved).resolve())
        raise CalibrationError(f"executable not found: {path}")
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return str(Path(resolved).resolve())
    raise CalibrationError("no executable found among: " + ", ".join(candidates))


def calibrate(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(args.build_dir).resolve() if args.build_dir else output.with_suffix(".build")
    directory.mkdir(parents=True, exist_ok=True)
    nvcc = _tool(args.nvcc, ("nvcc", "/usr/local/cuda/bin/nvcc"))
    host = _tool(args.cuda_host_cxx, ("g++-14", "g++"))
    nvcc_version = _run([nvcc, "--version"], directory, directory / "nvcc-version.log", timeout=30).strip()
    host_version = _run([host, "--version"], directory, directory / "host-version.log", timeout=30).strip()
    source = Path(__file__).with_name("calibration.cu")
    binary = directory / "calibration"
    command = [nvcc, "-O3", "-std=c++17", "-arch=" + args.arch, "-ccbin", host,
               "-Xcompiler=-fopenmp", "-DCALIBRATION_PRECISION=" + str(args.precision),
               str(source), "-o", str(binary)]
    print("Building standalone CUDA calibration...", file=sys.stderr, flush=True)
    _run(command, directory, directory / "build.log", timeout=180)
    run_command = [str(binary), str(args.threads), str(args.max_mib), str(args.device)]
    print("Measuring offline CPU/GPU costs...", file=sys.stderr, flush=True)
    observations = _run(run_command, directory, directory / "measurements.jsonl", timeout=180)
    profile = profile_from_measurements(parse_measurements(observations), precision_bits=args.precision,
                                        cpu_threads=args.threads, cpu_name=cpu_identity(),
                                        nvcc_version=nvcc_version, host_cxx_version=host_version)
    profile["calibration"] = {"build_command": command, "run_command": run_command,
                              "benchmark_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                              "max_transfer_bytes": args.max_mib * 1024 * 1024,
                              "artifacts": str(directory), "application_profiled": False}
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=output.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(profile, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(output)
    print(f"Hardware profile written to {output}", file=sys.stderr)
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="validated hardware profile JSON destination")
    parser.add_argument("--threads", type=int, default=4, help="total CPU thread budget, including any GPU pump")
    parser.add_argument("--precision", type=int, choices=(32, 64), default=64)
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    parser.add_argument("--max-mib", type=int, default=64, help="largest transfer, from 8 to 1024 MiB")
    parser.add_argument("--nvcc", default=os.environ.get("CUDACXX"))
    parser.add_argument("--cuda-host-cxx", "--host-cxx", dest="cuda_host_cxx", default=None)
    parser.add_argument("--arch", default="native", help="NVCC architecture, normally native for offline calibration")
    parser.add_argument("--build-dir", help="optional directory retaining native benchmark and raw measurements")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.device < 0 or not 8 <= args.max_mib <= 1024:
        parser.error("threads must be positive, device nonnegative, and max-mib between 8 and 1024")
    try:
        calibrate(args)
    except (CalibrationError, OSError) as error:
        print(f"Calibration failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
