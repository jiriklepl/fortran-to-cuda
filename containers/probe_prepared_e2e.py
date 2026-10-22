#!/usr/bin/env python3
"""Short development timing probe of three authenticated ELMM builds.

This checks matched output and completed steps, but never grants full-duration
float64 correctness or final performance qualification. Use reproduce_e2e.py
for that qualification.
"""
from __future__ import annotations

import argparse
import datetime
import os
import statistics
import sys
from pathlib import Path

import reproduce_e2e as runner
from sgs_experiment import (
    DEFAULT_CONTRACT,
    compare_outputs,
    digest,
    read,
    read_outputs,
    repository_provenance,
    select_configuration,
    verify_contract,
    write,
)


def probe_environment(threads):
    from elmm_pipeline.common import clean_environment
    environment = clean_environment(threads)
    for key in list(environment):
        if key.startswith(("FORT_", "OMP_", "GOMP_", "NVCOMPILER_", "PGI_")):
            environment.pop(key)
    environment.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
                       OMP_SCHEDULE="static", OMP_PROC_BIND="false")
    return environment


def timing_summary(rows):
    summary = runner.summarize(rows)
    for value in summary.values():
        samples = value["samples_seconds"]
        if len(samples) > 1:
            q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
            value.update(quartile_1_seconds=q1, quartile_3_seconds=q3, iqr_seconds=q3-q1)
        else:
            value.update(quartile_1_seconds=None, quartile_3_seconds=None, iqr_seconds=None)
    return summary


def run_probe(args):
    sys.path.insert(0, str(args.pipeline_root.resolve(strict=True)))
    from elmm_pipeline.common import command, digest_tree
    from elmm_pipeline.gpu_toolchain import discover_toolchain
    from elmm_pipeline.gpu_validation import trace_environment
    from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log

    if args.repeats < 1 or args.warmups < 1 or args.threads < 1 or not 1 <= args.timeout <= 240:
        raise ValueError("require positive repeats, at least one warmup, positive threads and timeout <=240s")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("probe output must be new; existing evidence is immutable")
    prepared = read(args.prepared_builds)
    originals, _ = verify_contract(args.app_root, read(DEFAULT_CONTRACT))
    provenance = repository_provenance(originals, read(DEFAULT_CONTRACT), command)
    if provenance != prepared["original_application"] or digest(DEFAULT_CONTRACT) != prepared["contract_sha256"]:
        raise ValueError("probe original application differs from authenticated builds")
    selected, identities = runner.validate_prepared_builds(
        prepared, prepared, threads=args.threads, build_args=[])
    # Authorities are loaded/verified once, then transported through hashed
    # sidecars; do not serialize the complete proof graph into each probe row.
    del selected
    output.mkdir(parents=True)
    source_copy = output / "probe-source.py"
    source_copy.write_bytes(Path(__file__).read_bytes())
    report_path = output / "report.json"
    protocol = {"threads": args.threads, "warmups": args.warmups, "repetitions": args.repeats,
                "rotate": True, "timeout_seconds": args.timeout,
                "cases": {"channel-simple": "natural109steps", "channel": "matched100-step timing cap"}}
    report = {"schema_version": 1, "kind": "prepared-threeway-development-timing-probe",
              "status": "prepared", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "protocol": protocol, "timings_reused": False,
              "full_natural_correctness_qualified": False, "final_performance_qualified": False,
              "limitation": "Matched production VTK and step checks only; no float64 full-storage gate, CUDA counters or naturalchannel qualification. Final acceptance requires fresh reproduce_e2e.py evidence.",
              "prepared_builds": {"path": str(args.prepared_builds.resolve()), "sha256": digest(args.prepared_builds)},
              "builds": prepared["builds"], "compilers": prepared["compilers"],
              "original_application": provenance,
              "calibration_profile": prepared.get("calibration_profile"),
              "authentication": {"files": len(identities), "before_and_after_all_runs": True},
              "script_sha256": digest(__file__), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "script_source": {"path": str(source_copy), "sha256": digest(source_copy)},
              "cases": {}}
    identities[str(args.prepared_builds.resolve())] = digest(args.prepared_builds)
    identities[str(Path(__file__).resolve())] = digest(__file__)
    identities[str(Path(runner.__file__).resolve())] = digest(runner.__file__)
    pipeline_identity = digest_tree(args.pipeline_root / "elmm_pipeline")
    for case in args.cases:
        source, advection = select_configuration(originals, case)
        config, _, config_dir = runner.prepare_workloads(
            source, output / case, advection=advection, steps=100 if case == "channel" else None)
        hashes = {str(path.relative_to(config_dir)): digest(path)
                  for path in config_dir.rglob("*") if path.is_file()}
        report["cases"][case] = {"configuration": config, "configuration_dir": str(config_dir),
                                "configuration_sha256": hashes, "runs": [], "summary": {}}
    write(report_path, report)
    if args.validate_only:
        return report
    toolchain = discover_toolchain(options={key: value for key, value in {
        "nvcc": args.nvcc, "cuda_host_cxx": args.cuda_host, "architecture": args.arch}.items()
        if value is not None}, languages=("cuda",))
    if not toolchain["device"]["available"]:
        raise RuntimeError("visible GPU required for the matched offloading probe")
    report["gpu_toolchain"] = toolchain
    if report["calibration_profile"]:
        report["calibration_hardware_binding"] = runner.calibration_hardware_binding(
            report["calibration_profile"], toolchain)
    environment = trace_environment(probe_environment(args.threads), "local", enabled=False)
    report["environment"] = {key: environment.get(key) for key in (
        "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_SCHEDULE", "OMP_PROC_BIND", "OMP_PLACES",
        "GOMP_CPU_AFFINITY", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES", "FORT_RUNTIME_TRACE", "FORT_OFFLOAD_TRACE")}
    report["status"] = "in_progress"
    try:
        for case, evidence in report["cases"].items():
            reference, steps = None, None
            for number in range(args.warmups + args.repeats):
                for mode in runner.rotated_order(number):
                    for name, sha in evidence["configuration_sha256"].items():
                        if digest(Path(evidence["configuration_dir"]) / name) != sha:
                            raise ValueError("probe configuration changed")
                    built = prepared["builds"][mode]
                    if digest(built["binary"]) != built["sha256"] or digest(__file__) != report["script_sha256"]:
                        raise ValueError("probe executable or runner changed")
                    directory = runner.run_directory(Path(evidence["configuration_dir"]),
                        output / case / "runs" / mode / f"round-{number:02d}")
                    argv = [built["binary"], *evidence["configuration"]["runtime_arguments"]]
                    print("PROBE", case, mode, f"round-{number:02d}", flush=True)
                    log_path = directory / "run.log"
                    log, elapsed = command(argv, cwd=directory, env=environment,
                                           log=log_path, timeout=args.timeout)
                    timing = validate_log(log, evidence["configuration"]["expected_max_steps"],
                                          expected_completed_steps=100 if case == "channel" else 109)
                    observed = read_outputs(directory / "output", read_vtk)
                    if reference is None:
                        if mode != "native":
                            raise RuntimeError("independent originalnative warmup must establish reference")
                        reference, steps = observed, timing["completed_steps"]
                    validation = compare_outputs(reference, observed, compare_vtk)
                    if "FORT_SCOPED " in log or "FORT_OFFLOAD " in log:
                        raise RuntimeError("production probe unexpectedly emitted runtime tracing")
                    row = {"mode": mode, "round": number, "warmup": number < args.warmups,
                           "traced": False, "full_storage": False, "wall_seconds": elapsed,
                           "argv": argv, "cwd": str(directory), "log": str(log_path),
                           "log_sha256": digest(log_path), "timing": timing,
                           "vtk_validation": validation, "passed": validation["passed"]}
                    evidence["runs"].append(row)
                    write(directory / "run.json", row)
                    write(report_path, report)
                    print("RESULT", case, mode, f"round-{number:02d}",
                          "wall_seconds", elapsed, "vtk_passed", row["passed"], flush=True)
                    if not row["passed"]:
                        raise RuntimeError("matched probe output differs from originalnative warmup")
            evidence["summary"] = timing_summary(evidence["runs"])
            evidence["completed_steps"] = steps
        if digest_tree(args.pipeline_root / "elmm_pipeline") != pipeline_identity:
            raise ValueError("pipeline helpers changed during probe")
        if repository_provenance(originals, read(DEFAULT_CONTRACT), command) != provenance:
            raise ValueError("original application checkout changed during probe")
        for path, sha in identities.items():
            if digest(path) != sha:
                raise ValueError("authenticated build input changed during probe: " + path)
        report["status"] = "completed_development_probe"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-builds", type=Path, required=True)
    parser.add_argument("--app-root", type=Path, default=runner.PIPELINE / ".cache/repos")
    parser.add_argument("--pipeline-root", type=Path, default=runner.PIPELINE)
    parser.add_argument("--case", dest="cases", choices=("channel-simple", "channel"), action="append")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--validate-only", action="store_true", help="Authenticate and prepare only; no application or GPU execution")
    parser.add_argument("--nvcc")
    parser.add_argument("--cuda-host")
    parser.add_argument("--arch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.cases = args.cases or ["channel-simple", "channel"]
    if len(args.cases) != len(set(args.cases)):
        parser.error("each case may be selected only once")
    try:
        run_probe(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, "error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
