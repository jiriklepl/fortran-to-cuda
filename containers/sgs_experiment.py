#!/usr/bin/env python3
"""Build and compare original ELMM with explicit SGS scoped/direct CUDA.

The caller provisions the selected public pipeline and runs this command under
its resource guard. No cached executable, historical measurement, or compiler
internal is imported here. The portable contract retains reviewed caller facts;
it never derives initializedness from an array declaration or procedure name.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "sgs_contract.json"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def application_paths(app_root):
    root = Path(app_root).resolve(strict=True)
    candidates = [root / name for name in ("elmm", "ELMM") if (root / name / "src/SConstruct").is_file()]
    if len(candidates) != 1 or not (root / "PoisFFT/src/SConstruct").is_file():
        raise ValueError("app_root must contain one ELMM (elmm/ or ELMM/) and sibling PoisFFT/")
    return {"elmm": candidates[0], "PoisFFT": root / "PoisFFT"}


def contract_path(paths, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2 or relative.parts[0] not in paths:
        raise ValueError("invalid portable contract source: " + name)
    path = (paths[relative.parts[0]] / Path(*relative.parts[1:])).resolve(strict=True)
    if not path.is_relative_to(paths[relative.parts[0]].resolve()):
        raise ValueError("contract source escapes its application tree: " + name)
    return path


def verify_contract(app_root, contract):
    """Check all retained proof-source/include/build hashes before rebinding."""
    paths = application_paths(app_root)
    if (contract.get("schema_version") != 1 or contract.get("entry") != "subgrid::sgs_sigma_uniform"
            or contract.get("defines") != ["DPREC"] or not contract.get("source_sha256")
            or contract.get("source") not in contract["source_sha256"]):
        raise ValueError("unsupported or incomplete reviewed SGS contract")
    hashes = {}
    for table in ("source_sha256", "include_sha256", "build_sha256"):
        for name, expected in contract[table].items():
            path = contract_path(paths, name)
            if digest(path) != expected:
                raise ValueError("reviewed SGS source hash mismatch: " + name)
            hashes[str(path)] = expected
    return paths, hashes


def repository_provenance(paths, contract, command):
    """Require pinned pristine checkouts, including inputs outside SGS effects."""
    repositories = {}
    for name, root in paths.items():
        def git(*arguments):
            text, _elapsed = command(["git", "-C", root, *arguments], timeout=30)
            return text.rstrip("\n")

        if Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve():
            raise ValueError("application source must be its own Git checkout: " + name)
        revision = git("rev-parse", "--verify", "HEAD")
        if revision != contract["repositories"][name]["revision"]:
            raise ValueError("application repository revision mismatch: " + name)
        if git("status", "--porcelain=v1", "--untracked-files=no", "--ignore-submodules=none"):
            raise ValueError("application repository has changed tracked inputs: " + name)
        # Include ignored files in this query. An ignored Fortran/include/build
        # input must not silently enter the copied application either.
        omitted = []
        for relative in filter(None, git("ls-files", "--others", "-z").split("\0")):
            path = Path(relative)
            if (path.suffix not in {".o", ".mod", ".pyc"}
                    and "__pycache__" not in path.parts and not path.name.startswith(".sconsign")):
                raise ValueError("application repository has untracked input: " + name + "/" + relative)
            omitted.append(relative)
        hashes = {}
        for relative in filter(None, git("ls-files", "-z").split("\0")):
            path = contract_path(paths, name + "/" + relative)
            if not path.is_file():
                raise ValueError("unsupported tracked application input: " + name + "/" + relative)
            hashes[relative] = digest(path)
        repositories[name] = {**contract["repositories"][name], "head": revision, "tracked_clean": True,
                              "tracked_sha256": hashes, "omitted_untracked_artifacts": sorted(omitted)}
    return {"schema_version": 1, "repositories": repositories}


def summarize(runs):
    """Keep diagnostic and warmup rows out of complete-wall statistics."""
    summary = {}
    for mode in ("native", "current"):
        measured = [row for row in runs if row["mode"] == mode and not row["warmup"]]
        if any(row["traced"] for row in measured):
            raise ValueError("diagnostic execution cannot enter timing samples")
        samples = [row["wall_seconds"] for row in measured]
        if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("positive finite complete-wall measurements required")
        summary[mode] = {"median": statistics.median(samples), "min": min(samples), "max": max(samples),
                         "repetitions": len(samples), "samples": samples}
    for row in summary.values():
        row["ratio_to_native"] = row["median"] / summary["native"]["median"]
    return summary


def read_outputs(directory, reader):
    """Retain final fields and averaged fields when the application emits them."""
    directory = Path(directory)
    paths = [directory / "out.vtk"]
    if (directory / "avg.vtk").is_file():
        paths.append(directory / "avg.vtk")
    return {path.name: reader(path) for path in paths}


def compare_outputs(reference, observed, compare):
    if reference.keys() != observed.keys():
        raise ValueError("Final/averaged VTK output files differ from the native reference")
    files = {name: compare(expected, observed[name], atol=1e-6, rtol=1e-5)
             for name, expected in reference.items()}
    return {"schema_version": 2, "passed": all(row["passed"] for row in files.values()),
            "atol": 1e-6, "rtol": 1e-5, "files": files,
            "note": "Final and available averaged VTK fields; float32 output, no float64 halo claim."}


def assess_validation(receipt, record, *, allow_native_variation=False, is_reference=False):
    """Keep failed comparisons visible when explicitly timing a variable baseline."""
    passed = record["validation"]["passed"]
    if record["mode"] == "native" and not is_reference:
        receipt["baseline_repeatability"] = passed and receipt.get("baseline_repeatability") is not False
    variable_baseline = receipt.get("baseline_repeatability") is False
    record["validation"]["qualified"] = passed and not variable_baseline
    if variable_baseline:
        record["validation"]["qualification_reason"] = "Native output is not repeatable at original VTK tolerances"
    if not passed:
        reason = "native repeatability failed" if record["mode"] == "native" else "candidate validation failed"
        message = reason + " at original VTK tolerances: " + record["log"]
        if not (allow_native_variation and variable_baseline):
            raise RuntimeError(message)
        receipt.setdefault("validation_warnings", []).append(message)
        print("WARNING:", message, "Continuing unvalidated measurements by explicit request.", flush=True)


def complete_summary(receipt):
    """Report timing even for an explicitly permitted, unqualified baseline."""
    summary = summarize(receipt["runs"])
    ratio = summary["current"]["ratio_to_native"]
    qualified = receipt.get("baseline_repeatability") is True and all(
        row["validation"]["passed"] for row in receipt["correctness"] + receipt["runs"])
    receipt.update(status="passed" if qualified else "completed_unvalidated", summary=summary,
                   correctness_qualified=qualified, speedup_factor=1 / ratio,
                   current_to_native_ratio=ratio, wall_reduction_percent=100 * (1 - ratio),
                   target_native_ratio=.8, target_wall_seconds=.8 * summary["native"]["median"],
                   target_met=ratio <= .8 if qualified and receipt["performance_eligible"]
                       and receipt["comparison_budget_eligible"] else None,
                   timings_reused=False, automatic_policy_promoted=False)


def run_experiment(*, app_root, compiler_root, pipeline_root, python, scons_python,
                   native_library_dir, output, work_dir=None, config=None,
                   case="channel-simple", gpu_options=None, build_args=(), jobs=8,
                   threads=4, steps=None, timeout=900, repeats=7, warmups=1,
                   allow_native_variation=False, scope_transfers="direct"):
    """Run one fresh paired experiment; return the immutable raw JSON receipt.

    ``output`` and ``work_dir`` must be absent. Default work is output/work.
    The selected ``python`` must provide the compiler/parser requirements;
    this process also needs fparser for public Project/export_analysis_sources.
    """
    if threads < 1 or jobs < 1 or repeats < 1 or warmups < 0 or timeout < 1 or (steps is not None and steps < 1):
        raise ValueError("positive threads and valid jobs/repetition/time budgets required")
    if scope_transfers not in {"direct", "pinned", "pipelined", "auto", "coalesced"}:
        raise ValueError("unsupported compiler scope transfer policy")
    app_root, compiler_root, pipeline_root = (Path(path).resolve(strict=True)
                                              for path in (app_root, compiler_root, pipeline_root))
    # Preserve virtual-environment interpreter paths: resolving their symlinks
    # would invoke the base interpreter without the provisioned parser/SCons.
    python, scons_python = (Path(path).absolute() for path in (python, scons_python))
    if not all(path.is_file() for path in (python, scons_python)):
        raise ValueError("compiler and SCons Python interpreters must exist")
    native_library_dir = Path(native_library_dir).resolve(strict=True)
    output = Path(output).resolve()
    work = Path(work_dir).resolve() if work_dir is not None else output / "work"
    if output.exists() or work.exists() or output == work:
        raise ValueError("experiment output and work directories must be new and distinct")
    originals, source_hashes = verify_contract(app_root, read(DEFAULT_CONTRACT))
    if any(path == source or path.is_relative_to(source) for path in (output, work) for source in originals.values()):
        raise ValueError("experiment output/work must stay outside original application trees")
    if not (compiler_root / "compiler/__main__.py").is_file():
        raise ValueError("compiler_root must contain compiler/__main__.py")
    existing = sys.modules.get("elmm_pipeline")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(pipeline_root):
        raise ValueError("another pipeline checkout is already imported")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(pipeline_root))
    from elmm_pipeline.build import build, preprocessing_defines
    from elmm_pipeline.common import clean_environment, command, digest_tree
    from elmm_pipeline.configuration import prepare_config
    from elmm_pipeline.gpu_toolchain import discover_toolchain
    from elmm_pipeline.gpu_validation import gpu_activity, trace_environment
    from elmm_pipeline.local_adapter import apply_scope_sources, transform_scopes
    from elmm_pipeline.scope_extraction import export_analysis_sources
    from elmm_pipeline.structural import Project
    from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log
    contract = read(DEFAULT_CONTRACT)
    whole_original = repository_provenance(originals, contract, command)
    if preprocessing_defines(list(build_args)) != contract["defines"]:
        raise ValueError("reviewed SGS preset supports only the retained DPREC preprocessing configuration")
    if config is None:
        if case == "channel-simple":
            config = HERE / "cases/channel-simple"
            origin = read(config / "origin.json")
            for name, expected in origin["prepared_files_sha256"].items():
                if digest(config / name) != expected:
                    raise ValueError("reviewed channel-simple configuration changed: " + name)
        elif case == "channel":
            config = originals["elmm"] / "examples/channel"
        else:
            raise ValueError("case must be channel-simple/channel, or supply config")
    config = Path(config).resolve(strict=True)
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    provenance_path = output / "original-sources.json"
    write(provenance_path, whole_original)
    receipt = {"schema_version": 1, "status": "in_progress", "case": case,
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "paths": {"app_root": str(app_root), "compiler_root": str(compiler_root),
                         "pipeline_root": str(pipeline_root), "work_dir": str(work), "output": str(output)},
               "selection": {"entry": contract["entry"], "gpu_policy": "sections", "scope_transfers": scope_transfers,
                             "scope_execution": "bounded", "host_threads": threads},
               "protocol": {"threads": threads, "jobs": jobs, "warmups": warmups, "repetitions": repeats,
                            "rotate": True, "explicit_steps": steps, "timeout_seconds": timeout,
                            "allow_native_variation": allow_native_variation},
               "baseline_repeatability": None, "correctness_qualified": False,
               "performance_eligible": steps is None and repeats >= 3 and warmups >= 1,
               "comparison_budget_eligible": threads == 4,
               "workload_label": "explicit step-cap smoke run" if steps is not None else "original configured duration",
               "field_validation_kind": "VTK", "full_array_halos_validated": False,
               "validation": {"atol": 1e-6, "rtol": 1e-5, "native_repeatability_required": True,
                              "scope": "Complete final and available averaged VTK fields; float32 output, no float64 halo claim"},
               "wall_includes": "process startup, computation, native solver, transfers, output and teardown",
               "contract_sha256": digest(DEFAULT_CONTRACT), "script_sha256": digest(__file__),
               "original_input_sha256": source_hashes, "build_args": list(build_args),
               "whole_original_source_provenance": {"path": str(provenance_path), "sha256": digest(provenance_path),
                   "tracked_files": sum(len(row["tracked_sha256"]) for row in whole_original["repositories"].values())},
               "source_contract_repositories": contract["repositories"], "builds": {}, "runs": [], "correctness": []}
    report_path = output / "report.json"
    write(report_path, receipt)
    original_environment = dict(os.environ)
    try:
        toolchain = discover_toolchain(options=gpu_options, languages=("cuda",))
        if not toolchain["device"]["available"]:
            raise RuntimeError("visible NVIDIA GPU required: " + str(toolchain["device"]["error"]))
        options = {**toolchain, **receipt["selection"]}
        receipt["gpu_toolchain"] = toolchain
        receipt["native_library_sha256"] = {str(path): digest(path)
                                             for path in native_library_dir.iterdir() if path.is_file()}
        receipt["compiler_tree_sha256"] = digest_tree(compiler_root / "compiler")
        helper_paths = [Path(importlib.import_module("elmm_pipeline." + name).__file__)
                        for name in ("build", "common", "configuration", "gpu_toolchain", "gpu_validation",
                                     "local_adapter", "scope_extraction", "structural", "validation", "scoped_provenance")]
        receipt["pipeline_helpers_sha256"] = {str(path): digest(path) for path in helper_paths}
        for mode in ("native", "current"):
            destination = work / mode / "application"
            for name, source in originals.items():
                shutil.copytree(source, destination / name,
                                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.o", "*.mod", ".sconsign*", "SConstruct.pipeline"))
            verify_contract(destination, contract)
        defines = preprocessing_defines(list(build_args))
        binary = build(work / "native/application/elmm", work / "native/application/PoisFFT", work / "native/build",
                       scons_python, native_library_dir, jobs=jobs, threads=threads, extra_args=list(build_args))
        receipt["builds"]["native"] = {"binary": str(binary), "sha256": digest(binary),
                                        "build_receipt": str(binary.parent / "build-command.json")}
        write(report_path, receipt)
        current_app = work / "current/application"
        current_paths = application_paths(current_app)
        source = contract_path(current_paths, contract["source"])
        sources = [source, *(contract_path(current_paths, name) for name in contract["source_sha256"] if name != contract["source"])]
        analysis = export_analysis_sources(Project(current_app, defines), work / "analysis", source_files=sources)
        metadata = transform_scopes(source, compiler_root, python, work / "generated/compiler",
                                    entry=contract["entry"], source_files=sources[1:],
                                    capture_facts=contract["capture_facts"], analysis_sources=analysis, gpu_options=options)
        manifest = metadata["scope_manifest"]
        if (manifest["scope_count"] <= 0 or set(manifest["sources"]) != {str(source)}
                or not metadata["gpu_sources"] or any("numerical_contract" not in row for row in metadata["gpu_sources"])):
            raise RuntimeError("current compiler did not emit strict SGS-only CUDA scopes")
        changed = apply_scope_sources(metadata, {str(path): path for path in sources})
        if set(changed) != {source}:
            raise RuntimeError("source changes extend beyond the explicit SGS source")
        receipt["generation"] = {"analysis_sources": str(analysis), "command": metadata["command"],
                                 "scope_count": manifest["scope_count"], "scope_manifest": str(work / "generated/compiler/scope-manifest.json"),
                                 "gpu_sources": metadata["gpu_sources"], "changed_sources": [str(path) for path in changed]}
        binary = build(current_paths["elmm"], current_paths["PoisFFT"], work / "current/build", scons_python,
                       native_library_dir, jobs=jobs, threads=threads, extra_args=list(build_args),
                       extra_sources=[Path(path) for path in metadata["extra_sources"]], patched_sources=changed,
                       gpu_sources=metadata["gpu_sources"], gpu_options=options)
        receipt["builds"]["current"] = {"binary": str(binary), "sha256": digest(binary),
                                         "build_receipt": str(binary.parent / "build-command.json")}
        configuration = prepare_config(config, output / "config", steps=steps)
        receipt["configuration"] = configuration
        receipt["config_sha256"] = {str(path.relative_to(output / "config")): digest(path)
                                     for path in (output / "config").rglob("*") if path.is_file()}
        binary_hashes = {row["binary"]: row["sha256"] for row in receipt["builds"].values()}
        receipt["artifact_sha256"] = {str(work / "generated/compiler" / name): expected
                                      for name, expected in manifest["artifacts_sha256"].items()}
        receipt["build_receipt_sha256"] = {row["build_receipt"]: digest(row["build_receipt"])
                                           for row in receipt["builds"].values()}
        receipt["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        write(report_path, receipt)
        base_environment = clean_environment(threads)
        for key in list(base_environment):
            if key.startswith(("FORT_", "OMP_", "GOMP_", "NVCOMPILER_", "PGI_")):
                base_environment.pop(key)
        base_environment.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE", OMP_SCHEDULE="static", OMP_PROC_BIND="false")
        reference, reference_steps = None, None

        def check_inputs():
            verify_contract(app_root, contract)
            if repository_provenance(originals, contract, command) != whole_original:
                raise ValueError("whole original application checkout changed during experiment")
            for path, expected in {**binary_hashes, **receipt["artifact_sha256"], **receipt["build_receipt_sha256"],
                                   **receipt["pipeline_helpers_sha256"], **receipt["native_library_sha256"],
                                   str(provenance_path): receipt["whole_original_source_provenance"]["sha256"],
                                   str(DEFAULT_CONTRACT): receipt["contract_sha256"],
                                   str(Path(__file__)): receipt["script_sha256"]}.items():
                if digest(path) != expected:
                    raise ValueError("experiment input changed: " + path)
            if digest_tree(compiler_root / "compiler") != receipt["compiler_tree_sha256"]:
                raise ValueError("compiler checkout changed during experiment")
            for name, expected in receipt["config_sha256"].items():
                if digest(output / "config" / name) != expected:
                    raise ValueError("experiment configuration changed: " + name)

        def execute(mode, label, *, traced, warmup=False, round_number=None):
            nonlocal reference, reference_steps
            check_inputs()
            directory = output / "runs" / mode / label
            shutil.copytree(output / "config", directory)
            if (directory / "output").exists():
                shutil.rmtree(directory / "output")
            argv = [receipt["builds"][mode]["binary"], *configuration["runtime_arguments"]]
            env = trace_environment(base_environment, "local", enabled=traced)
            log_path = directory / "run.log"
            print(("VERIFY" if traced or round_number is None else "TIMING"), mode, label, flush=True)
            log, elapsed = command(argv, cwd=directory, env=env, log=log_path, timeout=timeout)
            timing = validate_log(log, configuration["expected_max_steps"], expected_completed_steps=reference_steps)
            observed = read_outputs(directory / "output", read_vtk)
            is_reference = reference is None
            if reference is None:
                if mode != "native":
                    raise RuntimeError("original native reference must execute first")
                reference, reference_steps = observed, timing["completed_steps"]
            validation = compare_outputs(reference, observed, compare_vtk)
            record = {"mode": mode, "argv": argv, "cwd": str(directory), "log": str(log_path),
                      "log_sha256": digest(log_path), "traced": traced, "warmup": warmup,
                      "round": round_number, "timing": timing, "validation": validation,
                      "environment": {key: env.get(key) for key in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "CUDA_VISIBLE_DEVICES")},
                      "diagnostic_wall_seconds" if round_number is None else "wall_seconds": elapsed}
            try:
                assess_validation(receipt, record, allow_native_variation=allow_native_variation,
                                  is_reference=is_reference)
            except RuntimeError:
                write(directory / "run.json", record)
                raise
            if traced:
                activity = gpu_activity(log, "local", [], scope_manifest=manifest)
                record["gpu_activity"] = activity
                if not activity["verified"] or activity["scoped_runtime"]["kernel_launches"] <= 0:
                    write(directory / "run.json", record)
                    raise RuntimeError("SGS CUDA was not exercised")
                if any(row["status"] != 0 for row in activity["scoped_runtime"].get("modeled_evidence", [])
                       if row["event"] == "definition_validation"):
                    write(directory / "run.json", record)
                    raise RuntimeError("ordered source definition validation failed")
            elif "FORT_SCOPED " in log or "FORT_OFFLOAD " in log:
                raise RuntimeError("timing/reference run unexpectedly emitted GPU diagnostics")
            write(directory / "run.json", record)
            return record

        receipt["correctness"].append(execute("native", "reference", traced=False))
        write(report_path, receipt)
        receipt["correctness"].append(execute("native", "repeatability", traced=False))
        write(report_path, receipt)
        receipt["correctness"].append(execute("current", "gpu-validation", traced=True))
        receipt["gpu_evidence"] = receipt["correctness"][-1]["gpu_activity"]
        write(report_path, receipt)
        for round_number in range(warmups + repeats):
            order = ("native", "current") if round_number % 2 == 0 else ("current", "native")
            for mode in order:
                receipt["runs"].append(execute(mode, "round-%02d" % round_number, traced=False,
                                                warmup=round_number < warmups, round_number=round_number))
                write(report_path, receipt)
        check_inputs()
        complete_summary(receipt)
    except BaseException as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "reason": str(error)})
        raise
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
        receipt["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(report_path, receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "baseline_repeatability", "correctness_qualified",
                                                   "performance_eligible", "current_to_native_ratio",
                                                   "speedup_factor", "wall_reduction_percent", "target_native_ratio", "target_met")}))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("app-root", "compiler-root", "pipeline-root", "python", "scons-python", "native-library-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--case", choices=("channel-simple", "channel"), default="channel-simple")
    parser.add_argument("--nvcc")
    parser.add_argument("--cuda-host")
    parser.add_argument("--cuda-runtime-dir")
    parser.add_argument("--gpu-arch")
    parser.add_argument("--scope-transfers", choices=("direct", "pinned", "pipelined", "auto", "coalesced"), default="direct")
    parser.add_argument("--build-arg", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--allow-native-variation", action="store_true",
                        help="Complete unvalidated timings if original native outputs are not repeatable; no target claim")
    args = vars(parser.parse_args(argv))
    args["gpu_options"] = {key: args.pop(flag) for flag, key in (("nvcc", "nvcc"), ("cuda_host", "cuda_host_cxx"),
                           ("cuda_runtime_dir", "cuda_runtime_dir"), ("gpu_arch", "architecture")) if args[flag] is not None}
    for flag in ("nvcc", "cuda_host", "cuda_runtime_dir", "gpu_arch"):
        args.pop(flag, None)
    args["build_args"] = args.pop("build_arg")
    try:
        run_experiment(**args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, "error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
