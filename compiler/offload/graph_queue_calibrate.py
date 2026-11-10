"""Measure owned graph arguments and stable/changed replay; no placement authority.

Run this on an otherwise idle machine. Raw host-submission and complete-operation
times are distinct observations, not additive phases or application forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from compiler.emission.common.resources import read_scoped_runtime
from compiler.numerical_contract import cuda_compile_options, require_explicit_cuda_environment

ARGUMENT_COUNTS = (2, 8, 32, 64)
TASK_COUNTS = (1, 4, 16, 64)


def benchmark_source():
    source = Path(__file__).with_name("graph_queue_calibration.cu").read_text()
    kernels, runs = [], []
    for arguments in ARGUMENT_COUNTS:
        scalars = arguments - 1
        parameters = ",".join(["double *output", *(f"double p{i}" for i in range(scalars))])
        kernels.append(f"__global__ void probe_{arguments}({parameters}) {{ "
                       f"if(threadIdx.x==0) *output=p0+p{scalars-1}; }}")
        for tasks in TASK_COUNTS:
            for changed in (False, True):
                runs.append(f"measure(device,reinterpret_cast<const void *>(probe_{arguments}),"
                            f"{arguments},{tasks},{str(changed).lower()});")
    return source.replace("// FORT_GRAPH_PROBE_KERNELS", "\n".join(kernels)).replace(
        "// FORT_GRAPH_PROBE_RUNS", "\n".join(runs))


def validate_observations(records):
    identities = [r for r in records if r.get("kind") == "graph_queue_identity"]
    rows = [r for r in records if r.get("kind") == "graph_queue_observation"]
    if len(identities) != 1 or len(records) != len(rows) + 1:
        raise ValueError("Expected exactly one identity and typed graph observations")
    expected = {(a, t, c) for a in ARGUMENT_COUNTS for t in TASK_COUNTS for c in (False, True)}
    seen = set()
    for row in rows:
        key = row["arguments"], row["tasks"], row["changed"]
        if key not in expected or key in seen or type(row["changed"]) is not bool:
            raise ValueError("Duplicate or unexpected graph calibration row")
        seen.add(key)
        if (row["samples"] != 5 or row["repetitions"] != 64 or row["graph_replays"] != 320
                or row["logical_kernels"] != 320 * row["tasks"]
                or row["parameter_updates"] != (row["logical_kernels"] if row["changed"] else 0)
                or row["numerical_result_verified"] is not True):
            raise ValueError("Graph observation does not establish the intended execution")
        for name in ("host_seconds", "complete_seconds"):
            values = row[name]
            if len(values) != 5 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Invalid graph timing samples")
            row[name + "_median"] = statistics.median(values)
        if any(h > c for h, c in zip(row["host_seconds"], row["complete_seconds"], strict=True)):
            raise ValueError("Submission time exceeds the enclosing complete-operation interval")
    if seen != expected:
        raise ValueError("Incomplete graph calibration grid")
    return identities[0], rows


def run(command, root, label, environment, timeout):
    record = {"argv": command, "cwd": str(root), "status": "RUNNING"}
    process = None
    start = time.monotonic()
    try:
        with (root / (label + ".log")).open("x") as output:
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record["returncode"] = process.wait(timeout=timeout)
        if record["returncode"]:
            raise RuntimeError(f"{label} failed; inspect {root / (label + '.log')}")
        record["status"] = "PASS"
    except BaseException as error:
        record.update(status="FAIL", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        record["seconds"] = time.monotonic() - start
        (root / (label + ".json")).write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--cuda-host-cxx", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    require_explicit_cuda_environment()
    if args.threads < 1 or args.device < 0 or (args.prepare_only and args.run_prepared):
        parser.error("Invalid thread/device budget or conflicting phases")
    root = args.build_dir.resolve()
    sources, runtime = read_scoped_runtime()
    sources["graph-queue-protocol.cu"] = benchmark_source()
    digest = lambda data: hashlib.sha256(data).hexdigest()
    if not args.run_prepared:
        root.mkdir(parents=True, exist_ok=False)
        for name, content in sources.items():
            (root / name).write_text(content)
    inputs = {str(root / n): digest(s.encode()) for n, s in sources.items()}
    if any(digest(Path(p).read_bytes()) != h for p, h in inputs.items()):
        raise ValueError("Prepared graph calibration source differs from the current runtime")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("FORT_")}
    environment.pop("CUDA_LAUNCH_BLOCKING", None)
    environment.update(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT=str(args.threads),
                       OMP_MAX_ACTIVE_LEVELS="1", OMP_PROC_BIND="false", OMP_SCHEDULE="static")
    executable = root / "graph-queue-protocol"
    if not args.run_prepared:
        command = [str(args.nvcc.resolve()), "-O3", "-std=c++17", "-arch=" + args.arch,
                   "-ccbin", str(args.cuda_host_cxx.resolve()), *cuda_compile_options(), "-Xcompiler=-fopenmp",
                   str(root / "graph-queue-protocol.cu"), str(root / "scoped_runtime.cu"), "-o", str(executable)]
        run(command, root, "build", environment, 180)
        (root / "prepared.json").write_text(json.dumps({"inputs": inputs, "binary_sha256": digest(executable.read_bytes()),
                                                        "runtime": runtime}, indent=2) + "\n")
    if args.prepare_only:
        return
    prepared = json.loads((root / "prepared.json").read_text())
    if prepared["inputs"] != inputs or prepared["binary_sha256"] != digest(executable.read_bytes()):
        raise ValueError("Prepared graph calibration binary or sources changed")
    if args.output.exists():
        raise FileExistsError(args.output)
    affinity = sorted(os.sched_getaffinity(0))
    run([str(executable), str(args.device), str(args.threads)], root, "measurements", environment, 240)
    if (affinity != sorted(os.sched_getaffinity(0))
            or prepared["binary_sha256"] != digest(executable.read_bytes())
            or any(digest(Path(p).read_bytes()) != h for p, h in inputs.items())):
        raise ValueError("Calibration inputs or CPU affinity changed during measurement")
    identity, rows = validate_observations([json.loads(line) for line in (root / "measurements.log").read_text().splitlines()])
    if identity["device"] != args.device or identity["cpu_threads"] != args.threads:
        raise ValueError("Measured graph device/thread identity differs")
    result = {"schema_version": 1, "kind": "graph_queue_runtime_measurements", "runtime": runtime,
              "hardware": identity, "cpu_affinity": affinity, "observations": rows,
              "inputs": inputs, "binary_sha256": prepared["binary_sha256"], "placement_authority": False,
              "scope": "Steady queued graph submission and complete-operation costs; no ELMM workload or fitted placement model"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Measured", len(rows), "graph queue cases:", args.output, flush=True)


if __name__ == "__main__":
    main()
