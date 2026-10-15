"""One fresh, unchanged full application diagnostic; never qualification.

Run each process sequentially after generation/builds have completed. This
script does not change configurations, choose offload placement or retry work.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "elmm-pipeline"))
from elmm_pipeline.campaigns.run import environment
from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("channel-simple", "channel"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    source = ROOT / "benchmarks/results/checked-plan-application-20261014/cases" / args.case
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("existing run evidence is immutable")
    if len(os.sched_getaffinity(0)) != 4:
        raise ValueError("launch with an explicit four-CPU taskset")
    if args.reference:
        args.reference = args.reference.resolve(strict=True)
        reference = json.loads((args.reference / "run.json").read_text())
        if not reference.get("completed") or reference["case"] != args.case:
            raise ValueError("reference must be a completed run of the same case")
    before = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*")) if p.is_file()}
    shutil.copytree(source, output)
    assert all(digest(output / name) == value for name, value in before.items())
    env = environment(4, traced=args.trace)
    env.pop("OMP_PLACES", None)
    env.pop("GOMP_CPU_AFFINITY", None)
    env.update(OMP_PROC_BIND="false", CUDA_VISIBLE_DEVICES=args.gpu)
    argv = [str(binary), "output_dir='output/'"]
    record = {
        "schema_version": 1, "case": args.case, "mode": args.mode,
        "kind": "full-process diagnostic", "qualification": False,
        "automatic_placement_claim": False, "completed": False,
        "argv": argv, "cwd": str(output), "binary_sha256": digest(binary),
        "configuration_sha256": before, "configuration_overrides": {},
        "affinity": sorted(os.sched_getaffinity(0)), "threads": 4,
        "traced": args.trace, "timeout_seconds": args.timeout,
        "environment": {k: v for k, v in sorted(env.items()) if k.startswith(
            ("OMP_", "GOMP_", "FORT_", "CUDA_", "OPENBLAS_", "MKL_"))},
        "compiler_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "pipeline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "elmm-pipeline", text=True).strip(),
    }
    write(output / "command.json", record)
    print(json.dumps({"started": args.case, "mode": args.mode, "directory": str(output)}), flush=True)
    started = time.perf_counter()
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(argv, cwd=output, env=env, stdout=stdout, stderr=stderr)
        record["pid"] = process.pid
        write(output / "command.json", record)
        try:
            record["exit_code"] = process.wait(timeout=args.timeout)
            record["timed_out"] = False
        except subprocess.TimeoutExpired:
            process.kill()
            record["exit_code"] = process.wait()
            record["timed_out"] = True
    record["diagnostic_complete_process_seconds"] = time.perf_counter() - started
    record["binary_unchanged"] = digest(binary) == record["binary_sha256"]
    record["configuration_unchanged"] = all(digest(source / name) == value and digest(output / name) == value
                                               for name, value in before.items())
    try:
        if record["exit_code"] or not record["binary_unchanged"] or not record["configuration_unchanged"]:
            raise ValueError("application failed or input identity changed")
        expected = 109 if args.case == "channel-simple" else None
        if args.reference:
            expected = reference["completion"]["completed_steps"]
        record["completion"] = validate_log((output / "stdout.log").read_text(), expected_completed_steps=expected)
        if not record["completion"]["natural_end"]:
            raise ValueError("unchanged workload must end naturally")
        if args.reference:
            names = ["out.vtk", "avg.vtk"]
            record["vtk"] = {}
            for name in names:
                native, candidate = args.reference / "output" / name, output / "output" / name
                if native.exists() != candidate.exists() or name == "out.vtk" and not native.exists():
                    raise ValueError("reference/candidate output inventory differs: " + name)
                if native.exists():
                    check = compare_vtk(read_vtk(native), read_vtk(candidate))
                    record["vtk"][name] = check
                    if not check["passed"]:
                        raise ValueError("unchanged VTK tolerances failed: " + name)
        record["output_hashes"] = {str(p.relative_to(output)): digest(p)
                                    for p in sorted((output / "output").rglob("*.vtk"))}
        record["completed"] = True
    except Exception as error:
        record["failure"] = str(error)
    write(output / "run.json", record)
    print(json.dumps({key: record.get(key) for key in (
        "case", "mode", "completed", "exit_code", "diagnostic_complete_process_seconds", "completion", "failure")}), flush=True)
    return 0 if record["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
