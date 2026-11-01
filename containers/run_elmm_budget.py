"""Run one explicitly authorized, private ELMM budget diagnostic; never a timing sample."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(bindings):
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError("diagnostic input changed: " + path)


def fftw_plans(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate FFTW field")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite FFTW value: " + value)

    rows = [json.loads(line, object_pairs_hook=pairs, parse_constant=invalid)
            for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 4:
        raise ValueError("exactly four FFTW plans required")
    return rows


def run(plan_path, index, authorization):
    plan, auth = read(plan_path), read(authorization)
    if (auth.get("authorized_execution") != "private_baseline_phase_a_budget"
            or auth.get("plan_sha256") != digest(plan_path)
            or auth.get("runner_sha256") != digest(__file__)):
        raise ValueError("exact diagnostic plan authorization required")
    if sorted(os.sched_getaffinity(0)) != plan["cpu_affinity"]:
        raise ValueError("launch with exact declared taskset affinity")
    if not 0 <= index < len(plan["runs"]):
        raise ValueError("unknown diagnostic index")
    spec = plan["runs"][index]
    output = Path(plan["output_root"])
    output.mkdir(exist_ok=True)
    bindings = {**plan["inputs_sha256"], str(Path(plan_path).resolve()): digest(plan_path),
                str(Path(authorization).resolve()): digest(authorization), **spec["inputs_sha256"]}
    for previous in plan["runs"][:index]:
        receipt = Path(previous["directory"]) / "run.json"
        row = read(receipt)
        if (row.get("status") != "passed" or row.get("input_postguard_passed") is not True
                or row.get("plan_sha256") != digest(plan_path)):
            raise ValueError("a previous diagnostic failed or did not close")
        bindings[str(receipt)] = digest(receipt)
        bindings.update({str(receipt.parent / name): sha for name, sha in row["artifacts_sha256"].items()})
    verify(bindings)
    frozen = Path(plan["frozen_harness"])
    sys.path[:0] = [str(frozen / "containers"), str(frozen / "elmm-pipeline"), plan["transport_root"]]
    import reproduce_e2e as original
    from capture_transport import separated_command
    from elmm_pipeline.campaigns.halos import compare_fields, fields
    from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log
    from probe_prepared_e2e import probe_environment
    from sgs_experiment import compare_outputs, read_outputs

    # This new analyzer is deliberately distinct from the immutable baseline harness.
    module_spec = importlib.util.spec_from_file_location("diagnostic_cpu_budget", plan["analyzer"])
    analyzer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(analyzer)
    directory = Path(spec["directory"])
    original.run_directory(Path(spec["configuration_directory"]), directory)
    environment = probe_environment(4)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if spec["mode"] == "native":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    if environment.get("LD_PRELOAD") not in (None, "", plan["controller"]):
        raise ValueError("unexpected inherited preload")
    environment.update(LD_PRELOAD=plan["controller"], FORT_FFTW_CONTROL_MODE="verify",
        FORT_FFTW_CONTROL_PREFIX=plan["wisdom_prefix"],
        FORT_FFTW_CONTROL_LOG=str(directory / "fftw-plans.jsonl"),
        FORT_BUDGET_OUTPUT=str(directory / "budget.csv"), FORT_BUDGET_FILTER=spec["filter"])
    selected_environment = {key: value for key, value in environment.items()
        if key.startswith(("FORT_", "OMP_", "GOMP_", "CUDA_"))
        or key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PATH")}
    row = {"schema_version": 1, "status": "running", "diagnostic_only": True,
        "performance_sample": False, "mode": spec["mode"], "case": spec["case"], "kind": spec["kind"],
        "argv": spec["argv"], "cwd": str(directory), "binary_sha256": digest(spec["argv"][0]),
        "plan_sha256": digest(plan_path), "index": index, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": selected_environment,
        "full_environment_sha256": hashlib.sha256(json.dumps(environment, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(), "timeout_seconds": spec["timeout_seconds"],
        "input_postguard_passed": False}
    write(directory / "run.json", row)
    try:
        log, elapsed = separated_command(spec["argv"], cwd=directory, env=environment,
                                        log=directory / "run.log", timeout=spec["timeout_seconds"])
        row.update(process_completed=True, instrumented_process_wall_seconds=elapsed)
        if fftw_plans(directory / "fftw-plans.jsonl") != fftw_plans(plan["expected_plans"]):
            raise ValueError("fixed FFTW plans changed")
        timing = validate_log(log, spec["expected_max_steps"], expected_completed_steps=spec["completed_steps"])
        if timing["natural_end"] != spec["natural_end"] or "FORT_SCOPED " in log or "FORT_OFFLOAD " in log:
            raise ValueError("completion or untraced diagnostic contract differs")
        vtk = compare_outputs(read_outputs(Path(spec["reference_directory"]) / "output", read_vtk),
                              read_outputs(directory / "output", read_vtk), compare_vtk)
        row.update(timing=timing, vtk=vtk, exact_four_fftw_plans=True)
        if not vtk["passed"]:
            raise ValueError("diagnostic VTK disagreement")
        if spec["kind"] == "full20-gate":
            policy = {**original.FIELD_POLICY, "field_count": 20}
            comparison = compare_fields(fields(directory, policy), fields(Path(spec["reference_directory"]), policy), policy)
            row["full20_float64_shapes_bounds_halos"] = comparison
            if not all(item["passed"] for item in comparison.values()):
                raise ValueError("diagnostic complete-field disagreement")
        row["budget"] = analyzer.save_analysis(directory / "budget.csv", spec["symbols"], directory / "analysis")
        row["status"] = "passed"
    except BaseException as error:
        row.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        try:
            verify(bindings)
            row["input_postguard_passed"] = True
        except BaseException as error:
            row.update(status="failed", input_postguard_error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            row["artifacts_sha256"] = {str(path.relative_to(directory)): digest(path)
                for path in directory.rglob("*") if path.is_file() and path.name != "run.json"}
            write(directory / "run.json", row)
    print(index, spec["case"], spec["mode"], spec["kind"], row["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise ValueError("explicit diagnostic execution required")
    run(args.plan, args.index, args.authorization)


if __name__ == "__main__":
    main()
