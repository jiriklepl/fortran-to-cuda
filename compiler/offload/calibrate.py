"""Measure reusable hardware costs without an application workload.

Usage: python -m compiler.offload.calibrate --output profile.json --threads 4
       --precision 64 --cuda-host-cxx /usr/bin/g++-14

Add --scoped-costs to measure the common runtime's management and planning
operations. Base profiles stay usable for ordinary policies without this
extension; scoped automatic execution requires matching runtime calibration.
Add --collective-costs to measure the original persistent OpenMP team and its
generated protocol. It also identifies the actual Fortran compiler and ordered
semantic flags; defaults are -std=f2018 -O3 -fopenmp.
Add --numerical-costs for independently validated native Fortran, generated CPU
and GPU compute models. Rejected families retain their raw evidence. Version 1
is available explicitly for legacy C++/CUDA consumers.
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

from compiler.emission.common.resources import read_scoped_runtime
from compiler.numerical_contract import (
    cuda_compile_options,
    numerical_build_contract,
    require_explicit_cuda_environment,
)

from .profile import (
    LATENCY_RATES,
    SCHEMA_VERSION,
    SCOPED_BATCH_PAYLOADS,
    SCOPED_COST_NAMES,
    SCOPED_TRANSFER_COST_NAMES,
    THROUGHPUT_RATES,
    TRANSFER_KINDS,
    WORKER_RATES,
    ProfileError,
    require_profile_numerical_contract,
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
    numerical_contract: dict | None = None,
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
    if numerical_contract is not None:
        profile["numerical_contract"] = numerical_contract
    try:
        return validate_profile(profile)
    except ProfileError as error:
        raise CalibrationError(str(error)) from error


def parse_scoped_measurements(text: str) -> list[dict]:
    """Parse the opt-in common-runtime benchmark independently of base costs."""
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise CalibrationError("scoped benchmark emitted a non-JSON measurement") from error
        if not isinstance(record, dict) or record.get("kind") not in {
            "device", "scoped_cost", "scoped_allocation", "scoped_planning", "scoped_transfer",
        }:
            raise CalibrationError("scoped benchmark emitted an unknown measurement kind")
        if record["kind"] != "device":
            _median(record)
        records.append(record)
    if not records:
        raise CalibrationError("scoped benchmark emitted no measurements")
    return records


def profile_with_scoped_measurements(
    profile: dict, records: list[dict], *, runtime_id: str, cold_startup_seconds: list[float],
) -> dict:
    """Attach explicit common-runtime costs, preserving every raw observation.

    Allocation/release constants use the slowest measured median across byte
    sizes and are applicable only within the published maximum byte count.
    Planning costs divide complete candidate simulation time by its reported
    work count; they do not substitute an unrelated host loop throughput.
    """
    devices = [record for record in records if record["kind"] == "device"]
    if len(devices) != 1:
        raise CalibrationError("exactly one scoped device identity record is required")
    device = devices[0]
    for name in ("gpu_uuid", "gpu_name", "compute_capability"):
        if device.get(name) != profile["hardware"][name]:
            raise CalibrationError(f"scoped benchmark hardware {name} mismatch")
    for name in ("cuda_runtime_version", "driver_version"):
        if device.get(name) != profile["toolchain"][name]:
            raise CalibrationError(f"scoped benchmark toolchain {name} mismatch")
    if device.get("cpu_threads") != profile["cpu_threads"]:
        raise CalibrationError("scoped benchmark thread budget mismatch")
    if len(cold_startup_seconds) < 3:
        raise CalibrationError("cold driver startup requires at least three fresh process samples")
    cold = [_positive(value, "cold driver startup") for value in cold_startup_seconds]
    costs = {"cold_driver_startup_seconds": statistics.median(cold)}
    allocations: dict[str, list[tuple[int, float]]] = {"allocation_seconds": [], "release_seconds": []}
    planning = []
    transfer_records = []
    for record in records:
        kind, name = record["kind"], record.get("name")
        if kind == "device":
            continue
        if kind == "scoped_transfer":
            transfer_records.append(record)
            continue
        if kind == "scoped_planning":
            work = record.get("work")
            if type(work) is not int or work <= 0:
                raise CalibrationError("scoped planning work must be a positive integer")
            planning.append(_median(record) / work)
        elif kind == "scoped_allocation":
            size = record.get("bytes")
            if name not in allocations or type(size) is not int or size <= 0:
                raise CalibrationError("invalid scoped allocation measurement")
            if size in {prior[0] for prior in allocations[name]}:
                raise CalibrationError("duplicate scoped allocation byte count")
            allocations[name].append((size, _median(record)))
        elif kind == "scoped_cost":
            if name not in SCOPED_COST_NAMES or name in costs or name in allocations or name == "planning_operation_seconds":
                raise CalibrationError(f"unknown or duplicate scoped cost measurement: {name}")
            costs[name] = _median(record)
        else:
            raise CalibrationError("unknown scoped measurement")
    allocation_sizes = {size for size, _ in allocations["allocation_seconds"]}
    release_sizes = {size for size, _ in allocations["release_seconds"]}
    if len(allocation_sizes) < 3 or allocation_sizes != release_sizes:
        raise CalibrationError("scoped allocation and release require three matching distinct byte sizes")
    if not planning:
        raise CalibrationError("missing scoped candidate planning measurements")
    for name, measurements in allocations.items():
        costs[name] = max(duration for _, duration in measurements)
    costs["planning_operation_seconds"] = max(planning)
    result = dict(profile)
    result["scoped"] = {
        "schema_version": 1, "runtime_id": runtime_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "max_allocation_bytes": max(allocation_sizes), "costs": costs,
        "measurements": records,
        "cold_driver_startup_samples_seconds": cold,
        "measurement_method": {
            "samples": "median of five warmed observations; maximum median across allocation sizes and planning cases",
            "cold_driver_startup": "CUDA set-device and primary-context initialization in five fresh processes; no OS cold-state claim",
            "gpu_setup": "complete warm GPU scope lifecycle: create, first enter/leave, and close including mandatory stream/pool destruction",
            "create": "scope metadata creation; cleanup excluded",
            "register": "three-dimensional whole-root registration with negative lower bounds; unregister excluded",
            "host_access": "read access begin/end to an already-current host rectangle; no transfer",
            "device_access": "access begin/end to an already-allocated device rectangle; no transfer or wait",
            "allocation": "first device begin/end and completion wait on undefined storage; includes access bookkeeping",
            "release": "unregister of allocated undefined storage and completion wait; includes unregister bookkeeping",
            "wait": "common-runtime completion wait after an empty kernel; enqueue excluded",
            "launch_enqueue": "empty kernel enqueue, CUDA error check, and runtime launch notification; final wait excluded",
            "planning": "actual bounded common-runtime candidate simulation; 1/4/16 mixed units with full, opposite-face, and eight-rectangle effects; complete duration divided by reported work count",
            "accounting": "management costs are conservative complete-call costs; allocation/release can include bookkeeping charged elsewhere",
            "application_profiled": False,
        },
    }
    if transfer_records:
        result["scoped"]["transfers"] = scoped_transfers_from_measurements(transfer_records)
    try:
        return validate_profile(result, scoped_runtime_id=runtime_id)
    except ProfileError as error:
        raise CalibrationError(str(error)) from error


def scoped_transfers_from_measurements(records: list[dict]) -> dict:
    """Fit bounded single-coordinator staging costs from explicit observations.

    Slowest medians/rates cover the finite slot payload set and representative
    rectangular packing geometries. These costs remain separate from direct
    copies, launch/compute rates, and allocation costs. Application timing is
    never supplied to this function.
    """
    observations = {name: [] for name in SCOPED_TRANSFER_COST_NAMES if not name.endswith("row_seconds")}
    seen = set()
    for record in records:
        name = record.get("name")
        if record.get("kind") != "scoped_transfer" or name not in observations:
            raise CalibrationError("unknown scoped transfer measurement")
        size, geometry = record.get("bytes"), record.get("geometry", "")
        if size is not None and (type(size) is not int or size <= 0):
            raise CalibrationError("scoped transfer bytes must be a positive integer")
        if not isinstance(geometry, str):
            raise CalibrationError("scoped transfer geometry must be a string")
        units = record.get("units")
        if units is not None and (name != "preparation_operation_seconds" or type(units) is not int or units <= 0):
            raise CalibrationError("batch preparation units must be a positive integer")
        key = (name, size, geometry, units)
        if key in seen:
            raise CalibrationError("duplicate scoped transfer measurement")
        seen.add(key)
        duration = _median(record)
        if name.endswith("bytes_per_second"):
            if size is None or size not in SCOPED_BATCH_PAYLOADS or not geometry:
                raise CalibrationError("packing measurements require a finite payload and geometry")
            rows = record.get("rows")
            if type(rows) is not int or rows <= 0 or size % rows:
                raise CalibrationError("packing measurements require a positive exact physical row count")
            if geometry not in {"contiguous", "thin_rows"} or (geometry == "contiguous" and rows != 1):
                raise CalibrationError("unsupported scoped packing measurement geometry")
            observations[name].append((size, duration, rows, geometry))
        else:
            if name.startswith("staging_") and size not in SCOPED_BATCH_PAYLOADS:
                raise CalibrationError("staging measurements require the finite slot payloads")
            if name == "preparation_operation_seconds":
                work = record.get("work")
                if type(work) is not int or work <= 0:
                    raise CalibrationError("batch preparation requires an actual positive integer work count")
                duration /= work
            observations[name].append((size, duration))
    costs = {}
    for name, values in observations.items():
        if not values:
            raise CalibrationError("missing scoped transfer measurement: " + name)
        if name.startswith("staging_"):
            if {size for size, _ in values} != set(SCOPED_BATCH_PAYLOADS):
                raise CalibrationError("staging measurements must cover all finite payloads: " + name)
            costs[name] = [max(value for size, value in values if size == payload)
                           for payload in SCOPED_BATCH_PAYLOADS]
        elif name.endswith("bytes_per_second"):
            for geometry in ("contiguous", "thin_rows"):
                if {size for size, _, _, observed_geometry in values if observed_geometry == geometry} != set(SCOPED_BATCH_PAYLOADS):
                    raise CalibrationError("packing measurements must cover both geometries and all finite payloads: " + name)
            bandwidth = min(size / duration for size, duration, _, geometry in values if geometry == "contiguous")
            row_cost = max(0.0, max((duration - size / bandwidth) / rows
                                   for size, duration, rows, geometry in values if geometry == "thin_rows"))
            costs[name] = bandwidth
            costs[name.replace("bytes_per_second", "row_seconds")] = row_cost
        else:
            costs[name] = max(value for _, value in values)
    return {
        "schema_version": 1,
        "max_slot_bytes": SCOPED_BATCH_PAYLOADS[-1],
        "batch_payload_bytes": list(SCOPED_BATCH_PAYLOADS),
        "costs": costs,
        "measurements": records,
        "measurement_method": {
            "application_profiled": False,
            "coordinators": 1,
            "staging": "actual shared two-slot acquire/release; cold allocation and cached reuse measured separately",
            "packing": "exact physical rectangular row copies into/from pinned storage; conservative contiguous bandwidth plus nonnegative per-row overhead from thin rows",
            "events": "non-default stream completion event record and completed-event wait measured separately",
            "batch_prepare": "actual metadata-only versioned batch preview divided by its reported preparation work; no callback execution",
            "aggregation": "maximum median fixed costs per slot payload; minimum contiguous bandwidth and maximum nonnegative thin-row residual across bounded observations",
            "pipeline": "model includes preparation, initial immutable uploads, per-batch launches/events, fill and drain; no online timing feedback",
        },
    }


def cpu_identity() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _run(argv: list[str], directory: Path, log: Path, *, timeout: int, env: dict | None = None) -> str:
    try:
        result = subprocess.run(argv, cwd=directory, capture_output=True, text=True, check=False, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired) as error:
        def partial(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        log.write_text(partial(getattr(error, "stdout", "")) + partial(getattr(error, "stderr", ""))
                       + "\n" + str(error) + "\n")
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


def _calibrate_scoped(profile: dict, args: argparse.Namespace, directory: Path, nvcc: str, host: str) -> dict:
    require_explicit_cuda_environment()
    scoped_directory = directory / "scoped"
    scoped_directory.mkdir(exist_ok=True)
    sources, manifest = read_scoped_runtime()
    for name, content in sources.items():
        (scoped_directory / name).write_text(content)
    source = Path(__file__).with_name("scoped_calibration.cu")
    binary = scoped_directory / "scoped-calibration"
    command = [nvcc, "-O3", *cuda_compile_options(), "-std=c++17", "-arch=" + args.arch, "-ccbin", host,
               "-Xcompiler=-fopenmp", "-DCALIBRATION_PRECISION=" + str(args.precision),
               "-I" + str(scoped_directory), str(source), str(scoped_directory / "scoped_runtime.cu"),
               "-o", str(binary)]
    print("Building opt-in common-runtime calibration...", file=sys.stderr, flush=True)
    _run(command, scoped_directory, scoped_directory / "build.log", timeout=180)
    environment = dict(os.environ)
    for name in list(environment):
        if name in {"FORT_RUNTIME_TRACE", "FORT_PHASE_TIMING", "CUDA_LAUNCH_BLOCKING"} or name.startswith("FORT_SCOPE_TEST_"):
            environment.pop(name)
    cold_commands, cold_samples = [], []
    for sample in range(5):
        command_cold = [str(binary), "--cold-startup", str(args.device)]
        text = _run(command_cold, scoped_directory, scoped_directory / f"cold-startup-{sample}.json",
                    timeout=60, env=environment)
        try:
            record = json.loads(text)
            if not isinstance(record, dict) or record.get("kind") != "cold_driver_startup":
                raise ValueError("unknown cold startup measurement")
            cold_samples.append(_positive(record.get("seconds"), "cold driver startup"))
        except (ValueError, TypeError) as error:
            raise CalibrationError(f"invalid cold startup measurement: {error}") from error
        cold_commands.append(command_cold)
    run_command = [str(binary), str(args.threads), str(args.max_mib), str(args.device)]
    print("Measuring common-runtime management and candidate planning costs...", file=sys.stderr, flush=True)
    observations = _run(run_command, scoped_directory, scoped_directory / "measurements.jsonl", timeout=180,
                        env=environment)
    result = profile_with_scoped_measurements(profile, parse_scoped_measurements(observations),
                                              runtime_id=manifest["runtime_id"], cold_startup_seconds=cold_samples)
    result["scoped"]["calibration"] = {
        "build_command": command, "run_command": run_command,
        "cold_startup_commands": cold_commands, "runtime": manifest,
        "benchmark_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "artifacts": str(scoped_directory), "application_profiled": False,
    }
    return result


def calibrate(args: argparse.Namespace) -> dict:
    require_explicit_cuda_environment()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(args.build_dir).resolve() if args.build_dir else output.with_suffix(".build")
    directory.mkdir(parents=True, exist_ok=True)
    nvcc = _tool(args.nvcc, ("nvcc", "/usr/local/cuda/bin/nvcc"))
    host = _tool(args.cuda_host_cxx, ("g++-14", "g++"))
    nvcc_version = _run([nvcc, "--version"], directory, directory / "nvcc-version.log", timeout=30).strip()
    host_version = _run([host, "--version"], directory, directory / "host-version.log", timeout=30).strip()
    refresh = getattr(args, "refresh_scoped", None)
    base_profile_bytes = None
    if refresh:
        if not getattr(args, "scoped_costs", False):
            raise CalibrationError("--refresh-scoped requires --scoped-costs")
        try:
            base_profile_bytes = Path(refresh).read_bytes()
            profile = validate_profile(json.loads(base_profile_bytes), precision_bits=args.precision, cpu_threads=args.threads,
                                       hardware={"cpu_name": cpu_identity()},
                                       toolchain={"nvcc_version": nvcc_version, "host_cxx_version": host_version})
            require_profile_numerical_contract(profile)
        except (ProfileError, ValueError) as error:
            raise CalibrationError(str(error)) from error
    else:
        source = Path(__file__).with_name("calibration.cu")
        binary = directory / "calibration"
        command = [nvcc, "-O3", *cuda_compile_options(), "-std=c++17", "-arch=" + args.arch, "-ccbin", host,
                   "-Xcompiler=-fopenmp", "-DCALIBRATION_PRECISION=" + str(args.precision),
                   str(source), "-o", str(binary)]
        print("Building standalone CUDA calibration...", file=sys.stderr, flush=True)
        _run(command, directory, directory / "build.log", timeout=180)
        run_command = [str(binary), str(args.threads), str(args.max_mib), str(args.device)]
        print("Measuring offline CPU/GPU costs...", file=sys.stderr, flush=True)
        observations = _run(run_command, directory, directory / "measurements.jsonl", timeout=180)
        profile = profile_from_measurements(parse_measurements(observations), precision_bits=args.precision,
                                            cpu_threads=args.threads, cpu_name=cpu_identity(),
                                            nvcc_version=nvcc_version, host_cxx_version=host_version,
                                            numerical_contract=numerical_build_contract())
        profile["calibration"] = {"build_command": command, "run_command": run_command,
                                  "benchmark_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                  "max_transfer_bytes": args.max_mib * 1024 * 1024,
                                  "artifacts": str(directory), "application_profiled": False}
    if getattr(args, "scoped_costs", False):
        profile = _calibrate_scoped(profile, args, directory, nvcc, host)
        if refresh:
            profile["scoped"]["calibration"]["base_profile"] = {
                "path": str(Path(refresh).resolve()),
                "sha256": hashlib.sha256(base_profile_bytes).hexdigest(),
                "base_rates_remeasured": False,
                "identity_checked": "CPU, toolchain, precision and thread budget; current GPU/runtime/driver verified by new scoped measurements",
            }
    if getattr(args, "collective_costs", False):
        if not getattr(args, "scoped_costs", False):
            raise CalibrationError("--collective-costs requires --scoped-costs")
        from .collective_calibration import calibrate_collective
        print("Measuring the original persistent-team and emitted collective protocol...", file=sys.stderr, flush=True)
        try:
            profile = calibrate_collective(profile, args, directory, nvcc, host, run=_run, tool=_tool)
        except ValueError as error:
            raise CalibrationError(str(error)) from error
    if getattr(args, "numerical_costs", False):
        from .numerical_calibration import NumericalCalibrationError, calibrate_numerical, calibrate_numerical_v2
        print("Measuring independent numerical families and backends...", file=sys.stderr, flush=True)
        try:
            if getattr(args, "numerical_version", 2) == 1:
                profile = calibrate_numerical(profile, args, directory, nvcc, host, run=_run)
            else:
                profile = calibrate_numerical_v2(profile, args, directory, nvcc, host, run=_run, tool=_tool)
        except NumericalCalibrationError as error:
            raise CalibrationError(str(error)) from error
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
    parser.add_argument("--scoped-costs", action="store_true",
                        help="also measure common-runtime management, allocation, planning, staging and batch costs")
    parser.add_argument("--refresh-scoped", metavar="BASE_PROFILE",
                        help="refresh scoped costs only, preserving base rates after hardware/toolchain verification; requires --scoped-costs")
    parser.add_argument("--collective-costs", action="store_true",
                        help="also measure the actual generated persistent-team protocol; requires --scoped-costs")
    parser.add_argument("--numerical-costs", action="store_true",
                        help="also validate generic numerical costs for native Fortran, generated CPU and GPU")
    parser.add_argument("--numerical-version", type=int, choices=(1, 2), default=2,
                        help="numerical evidence schema (default 2; version 1 supports legacy C++ consumers only)")
    parser.add_argument("--cpu-affinity", help="fixed comma-separated CPU indices for numerical v2 (default first allowed CPUs matching --threads)")
    parser.add_argument("--fortran", help="Fortran compiler for numerical/collective calibration (default gfortran-15, gfortran-14 or gfortran)")
    parser.add_argument("--fortran-flag", action="append", default=[],
                        help="repeat to replace Fortran defaults in order; include -fopenmp "
                             "(defaults: -std=f2018 -O3 -fopenmp)")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.device < 0 or not 8 <= args.max_mib <= 1024:
        parser.error("threads must be positive, device nonnegative, and max-mib between 8 and 1024")
    if args.refresh_scoped and not args.scoped_costs:
        parser.error("--refresh-scoped requires --scoped-costs")
    if args.collective_costs and (not args.scoped_costs or args.threads > 256):
        parser.error("--collective-costs requires --scoped-costs and at most 256 threads")
    try:
        calibrate(args)
    except (CalibrationError, OSError) as error:
        print(f"Calibration failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
