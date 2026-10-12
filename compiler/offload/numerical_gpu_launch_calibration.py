"""Measure a predeclared nonzero launch control using exact retained GPU objects.

Run once under an external resource guard. Historical parents are readable;
the separate cost query requires a current compatible parent before use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from compiler.numerical_contract import require_explicit_cuda_environment
from compiler.offload import numerical_gpu_execution as gpu
from compiler.offload import numerical_gpu_execution_costs as costs
from compiler.offload.numerical_execution import _hash


def _reference(path, expected=None):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    if not path.is_file() or before.st_size > 256 * 2**20:
        raise gpu.NumericalGpuExecutionError("GPU launch regular artifact exceeds256MiB")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    after = path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or expected is not None and digest.hexdigest() != expected):
        raise gpu.NumericalGpuExecutionError("GPU launch artifact changed/differs: " + str(path))
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": before.st_size}


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _parse(text):
    if len(text) > 2**20 or len(text.encode()) > 2**20:
        raise gpu.NumericalGpuExecutionError("GPU launch output exceeds1MiB")
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise gpu.NumericalGpuExecutionError("GPU launch non-JSON raw output") from error
        if not isinstance(row, dict) or row.get("kind") not in {
                gpu.IDENTITY_KIND, costs.PREFLIGHT_KIND, costs.BATCH_KIND}:
            raise gpu.NumericalGpuExecutionError("GPU launch unknown raw output")
        rows.append(row)
        if len(rows) > costs.MAX_RECORDS:
            raise gpu.NumericalGpuExecutionError("GPU launch output record budget exhausted")
    if not rows or rows[0].get("kind") != gpu.IDENTITY_KIND or any(r["kind"] == gpu.IDENTITY_KIND for r in rows[1:]):
        raise gpu.NumericalGpuExecutionError("GPU launch one initial actual identity required")
    return rows


def calibrate_gpu_launch(profile, args, *, run, tool):
    require_explicit_cuda_environment()
    gpu.validate_gpu_execution_profile(profile)
    if costs.SECTION in profile:
        raise gpu.NumericalGpuExecutionError("preserve existing GPU costs; use an unmodified parent")
    parent, base = profile[gpu.SECTION], deepcopy(profile)
    target = Path(args.build_dir).resolve()
    target.mkdir(parents=True, exist_ok=False)
    commands = []
    try:
        from compiler.offload.calibrate import cpu_identity

        profile_reference = args.profile_reference
        costs._reference(profile_reference)
        _reference(profile_reference["path"], profile_reference["sha256"])
        if cpu_identity() != profile["hardware"]["cpu_name"]:
            raise gpu.NumericalGpuExecutionError("GPU launch enqueue CPU differs from parent")
        affinity = parent["identity"]["sampling_cpu_affinity"]
        if not set(affinity) <= set(os.sched_getaffinity(0)):
            raise gpu.NumericalGpuExecutionError("GPU launch parent CPU placement unavailable")
        original = Path(parent["calibration"]["artifacts"])
        parent_refs = {}
        for name, digest in {**parent["identity"]["emitted_sources"], **parent["identity"]["objects"],
                             "numerical-gpu-execution": parent["identity"]["binary_sha256"]}.items():
            parent_refs[name] = _reference(original / name, digest)
        for source, destination in (("workers.o", "workers.o"), ("gpu_recipes.hpp", "gpu_recipes.hpp"),
                                    ("driver.cu", "parent_driver.cu")):
            shutil.copyfile(parent_refs[source]["path"], target / destination)
        (target / "launch_driver.cu").write_bytes(Path(__file__).with_name("numerical_gpu_launch_driver.cu").read_bytes())
        nvcc = tool(args.nvcc or parent["identity"]["build_paths"]["nvcc"], ("nvcc",))
        host = tool(args.cuda_host_cxx or parent["identity"]["build_paths"]["host"], ("g++-14", "g++"))
        tool_refs = {name: _reference(path) for name, path in (("nvcc", nvcc), ("host", host))}
        versions = {name: run([path, "--version"], target, target / (name + ".log"), timeout=30).strip()
                    for name, path in (("nvcc_version", nvcc), ("host_cxx_version", host))}
        if any(value != parent["identity"][name] for name, value in versions.items()):
            raise gpu.NumericalGpuExecutionError("GPU launch compiler versions differ from parent")
        implementation = costs.cost_sources()
        paths = {"nvcc": nvcc, "host": host, "runtime": parent["identity"]["build_paths"]["runtime"],
                 "artifacts": str(target)}
        architecture = parent["identity"]["architecture"]
        planned = costs.launch_build_commands(paths, architecture)
        emitted = {name: _reference(target / name)["sha256"]
                   for name in ("launch_driver.cu", "parent_driver.cu", "gpu_recipes.hpp")}
        for name, argv in zip(("launch-driver", "link"), planned, strict=True):
            commands.append(argv)
            run(argv, target, target / (name + ".log"), timeout=180)
        objects = {name: _reference(target / name)["sha256"] for name in ("workers.o", "launch_driver.o")}
        binary = _reference(target / "numerical-gpu-launch")
        environment = {k: v for k, v in os.environ.items() if k not in {
            "FORT_RUNTIME_TRACE", "FORT_OFFLOAD_TRACE", "FORT_PHASE_TIMING", "CUDA_LAUNCH_BLOCKING",
            "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCOMPILER_ACC_NOTIFY", "NVCOMPILER_ACC_TIME",
            "PGI_ACC_NOTIFY", "PGI_ACC_TIME"} and not k.startswith("FORT_SCOPE_TEST_")}
        environment["FORT_PHASE_TIMING"] = "0"
        argv = ["taskset", "-c", ",".join(map(str, affinity)), binary["path"], str(parent["identity"]["device_ordinal"])]

        def authenticate():
            if profile != base or implementation != costs.cost_sources() or cpu_identity() != profile["hardware"]["cpu_name"]:
                raise gpu.NumericalGpuExecutionError("GPU launch parent/implementation/CPU changed")
            _reference(profile_reference["path"], profile_reference["sha256"])
            for ref in (*parent_refs.values(), *tool_refs.values(), binary):
                _reference(ref["path"], ref["sha256"])
            for name, digest in {**emitted, **objects}.items():
                _reference(target / name, digest)

        def phase(mode, timeout):
            authenticate()
            rows = _parse(run([*argv, mode], target, target / (mode[2:] + ".jsonl"), timeout=timeout, env=environment))
            authenticate()
            return rows

        raw_identity, = phase("--identity", 30)
        identity = {**raw_identity, **versions, "launch_protocol_id": costs.PROTOCOL_ID,
                    "parent_evidence_sha256": _hash(parent), "cost_sources": implementation,
                    "emitted_sources": emitted, "objects": objects, "binary_sha256": binary["sha256"],
                    "build_commands": commands, "build_paths": paths, "architecture": architecture}
        # Actual device/source/compiler identity must match before payload checks.
        costs._identity(profile, identity)
        checked = phase("--preflight", 30)
        if checked[0] != raw_identity:
            raise gpu.NumericalGpuExecutionError("GPU launch identity changed before preflight")
        costs.validate_launch_preflight(profile, identity, checked[1:])
        _write(target / "preflight-receipt.json", {"identity": identity, "preflight": checked[1:], "sampling_started": False})
        rows = phase("--measure", 120)
        if rows[:len(checked)] != checked:
            raise gpu.NumericalGpuExecutionError("GPU launch identity/preflight changed before sampling")
        raw = _reference(target / "launch-raw-samples.jsonl")
        if raw["bytes"] > 2**20:
            raise gpu.NumericalGpuExecutionError("GPU launch raw byte budget exhausted")
        raw_rows = [json.loads(line) for line in Path(raw["path"]).read_text().splitlines()]
        if raw_rows != rows[len(checked):]:
            raise gpu.NumericalGpuExecutionError("GPU launch raw batch file differs")
        calibration = {"application_profiled": False, "parent_evidence_sha256": _hash(parent),
                       "run_commands": [[*argv, mode] for mode in ("--identity", "--preflight", "--measure")],
                       "environment": {k: environment.get(k) for k in gpu.ENVIRONMENT_KEYS},
                       "raw_samples": raw, "tools": tool_refs, "parent_artifacts": parent_refs,
                       "input_profile": getattr(args, "profile_reference", None)}
        result = costs.profile_from_gpu_launch_measurements(base, [identity, *rows[1:]], calibration)
        authenticate()
        _write(target / "costs.json", result[costs.SECTION])
        return result
    except BaseException as error:
        _write(target / "failure.json", {"error": str(error), "error_type": type(error).__name__,
               "application_profiled": False, "commands": commands})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--nvcc")
    parser.add_argument("--cuda-host-cxx")
    args = parser.parse_args(argv)
    if args.output.exists() or args.build_dir.exists() or args.output.resolve() == args.profile.resolve():
        parser.error("fresh output/build paths required; preserve the parent")
    try:
        from compiler.offload.calibrate import _run, _tool
        from compiler.offload.profile import load_profile

        args.profile_reference = _reference(args.profile)
        result = calibrate_gpu_launch(load_profile(args.profile), args, run=_run, tool=_tool)
        _reference(args.profile, args.profile_reference["sha256"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write(args.output, result)
    except (OSError, ValueError) as error:
        parser.exit(1, "GPU launch costs: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
