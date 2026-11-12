#!/usr/bin/env python3
"""Build and compare the current resident ELMM and its native reference."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

import numpy as np

from application_basis import prepare_application_basis
from sgs_experiment import (application_paths, compare_outputs, digest, read_outputs,
                            select_configuration)
from workflow import COMPILER, PIPELINE, WORKSPACE, doctor, plot_experiment, require_pipeline, source_identity

try:
    require_pipeline()
except RuntimeError as error:
    raise SystemExit(f"error: {error}") from None

from elmm_pipeline.campaigns.halos import compare_fields, fields
from elmm_pipeline.campaigns.run import environment
from elmm_pipeline.configuration import prepare_config
from elmm_pipeline.resident_rk import verify_resident_execution
from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log

REQUIRED_FIELDS = {"fields-" + name + ".bin" for name in
                   "U V W Pr Viscosity U2 V2 W2 Ustar Vstar Wstar".split()}
FIELD_POLICY = {"dtype": "float64", "atol": 1e-6, "rtol": 1e-5,
                "metadata_format": "storage-shape-bounds-v1"}
NATURAL_CHANNEL_FIELDS = {"fields-" + name + ".bin" for name in
                          "Moisture Pr Pr_avg Scalar Scalar_avg TDiff Temperature U U2 U_avg Ustar "
                          "V V2 V_avg Viscosity Vstar W W2 W_avg Wstar".split()}
NATURAL_CHANNEL_OUTPUTS = {"out.vtk", "avg.vtk", "U.vtk", "V.vtk", "W.vtk", "Uavg.vtk", "Vavg.vtk",
                           "Wavg.vtk", "Retau.unf", "times.unf", "frame-y-times.txt", "frame-z-times.txt"} | {
                               f"frame-{axis}-{index}.vtk" for axis in ("y", "z") for index in range(51)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def solver_identity(root):
    root = Path(root).resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--",
         "CMakeLists.txt", "dependencies.lock.json", "src", "include", "cmake", "docs"], text=True)
    return {"root": str(root), "revision": revision, "source_status": status.splitlines()}


def prepare_dependencies(directory, *, application, vkfft):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    specifications = {}
    if application:
        specifications.update(json.loads((PIPELINE / "dependencies.json").read_text()))
    if vkfft:
        specifications["VkFFT"] = json.loads((PIPELINE / "pressure-dependencies.json").read_text())["VkFFT"]
    for name, spec in specifications.items():
        destination = directory / name
        print("Preparing pinned dependency:", name, spec["revision"], flush=True)
        subprocess.run(["git", "clone", "--no-checkout", "--", spec["url"], str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", spec["revision"]], check=True)
        revision = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True)
        if revision != spec["revision"] or status:
            raise ValueError("Pinned dependency checkout is not clean: " + str(destination))
    return directory


def command(argv, *, cwd, env, log, timeout):
    started = time.monotonic()
    process = None
    try:
        with Path(log).open("x", encoding="utf-8") as output:
            process = subprocess.Popen([str(value) for value in argv], cwd=cwd, env=env,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=timeout)
        if code:
            raise RuntimeError(f"Command failed with status {code}; see {log}")
        return time.monotonic() - started
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def application_environment(threads, gpu, cache, *, traced):
    env = environment(threads, traced=traced)
    for key in list(env):
        if key.startswith(("FORT_", "POISSON_", "VKFFT_")) or key == "LD_PRELOAD":
            env.pop(key)
    if traced:
        env.update(FORT_RUNTIME_TRACE="1", FORT_OFFLOAD_TRACE="1")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    if cache is not None:
        env["POISSON_SOLVER_KERNEL_CACHE_DIR"] = str(cache)
    return env


def run_application(binary, config, configuration, directory, env, timeout, expected_steps=None):
    shutil.copytree(config, directory, ignore=shutil.ignore_patterns("output"))
    argv = [str(binary), *configuration["runtime_arguments"]]
    elapsed = command(argv, cwd=directory, env=env, log=directory / "application.log", timeout=timeout)
    log = (directory / "application.log").read_text(errors="replace")
    timing = validate_log(log, configuration["expected_max_steps"], expected_completed_steps=expected_steps)
    vtk = read_outputs(directory / "output", read_vtk)
    return {"directory": str(directory), "argv": argv, "wall_seconds": elapsed,
            "application_timing": timing}, vtk, log


def compare_vtks(reference, actual):
    result = compare_outputs(reference, actual, compare_vtk)
    if not result["passed"]:
        raise ValueError("Final or averaged VTK comparison failed")
    return result


def read_fields(directory, expected_count=None):
    paths = {path.name for path in directory.glob("fields-*.bin")}
    if not REQUIRED_FIELDS.issubset(paths):
        raise ValueError("Missing required full-precision diagnostic fields")
    policy = {**FIELD_POLICY, "field_count": len(paths) if expected_count is None else expected_count}
    return fields(directory, policy)


def compare_long_vectors(reference, actual, *, criterion="relative_l2"):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError("Long-run comparison shape mismatch")
    if criterion not in {"relative_l2", "pointwise", "exact"}:
        raise ValueError("Unknown long-run comparison criterion")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(actual)):
        return {"passed": False, "values": int(reference.size), "criterion": criterion,
                "relative_l2": None, "max_absolute_error": None, "reason": "nonfinite input"}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = actual - reference
        maximum = float(np.max(np.abs(error), initial=0))
        peak = float(np.max(np.abs(reference), initial=0))
    if not math.isfinite(maximum):
        return {"passed": False, "values": int(reference.size), "criterion": criterion,
                "relative_l2": None, "max_absolute_error": None, "reason": "nonfinite error"}
    relative = peak_relative = None
    if peak:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            peak_ratio = float(np.divide(maximum, peak))
            relative_ratio = (peak_ratio * float(np.linalg.norm(error / maximum))
                              / float(np.linalg.norm(reference / peak))) if maximum else 0.0
        peak_relative = peak_ratio if math.isfinite(peak_ratio) else None
        relative = relative_ratio if math.isfinite(relative_ratio) else None
    if criterion == "exact":
        passed = np.array_equal(reference, actual)
    elif criterion == "pointwise":
        passed = np.all(np.isclose(actual, reference, atol=FIELD_POLICY["atol"], rtol=FIELD_POLICY["rtol"]))
    else:
        passed = relative is not None and relative <= 0.01 if peak else maximum <= FIELD_POLICY["atol"]
    return {"passed": bool(passed), "values": int(reference.size), "criterion": criterion,
            "relative_l2": relative, "max_absolute_error": maximum,
            "max_error_relative_to_reference_peak": peak_relative}


def compare_long_fields(actual, reference):
    pointwise = compare_fields(actual, reference, FIELD_POLICY)
    return {name: {**compare_long_vectors(reference[name][0], values), "pointwise": pointwise[name]}
            for name, (values, _layout) in actual.items()}


def compare_long_vtk(reference, actual):
    shape, expected = reference
    actual_shape, observed = actual
    if shape != actual_shape or list(expected) != list(observed):
        raise ValueError("Long-run VTK dimensions or field labels differ")
    result = {}
    count = math.prod(shape)
    for name, values in expected.items():
        candidate = observed[name]
        if values.typecode != candidate.typecode:
            raise ValueError("Long-run VTK field precision differs: " + name)
        exact = name.endswith("_COORDINATES") or name == "ptype"
        if not exact and len(values) == 3 * count:
            original = np.asarray(values).reshape(-1, 3)
            changed = np.asarray(candidate).reshape(-1, 3)
            for component in range(3):
                result[f"{name}[{component}]"] = compare_long_vectors(original[:, component], changed[:, component])
        else:
            result[name] = compare_long_vectors(values, candidate, criterion="exact" if exact else "relative_l2")
    return result


def compare_long_outputs(reference, actual):
    reference, actual = Path(reference), Path(actual)
    expected = {str(path.relative_to(reference)): path for path in reference.rglob("*") if path.is_file()}
    observed = {str(path.relative_to(actual)): path for path in actual.rglob("*") if path.is_file()}
    if expected.keys() != observed.keys():
        missing = sorted(expected.keys() - observed.keys())
        extra = sorted(observed.keys() - expected.keys())
        raise ValueError(f"Long-run output inventory differs: missing={missing[:3]}, extra={extra[:3]}")
    results = {}
    for name, path in sorted(expected.items()):
        candidate = observed[name]
        if path.suffix == ".vtk":
            final = name in {"out.vtk", "avg.vtk"}
            results[name] = compare_long_vtk(read_vtk(path, require_flow_fields=final),
                                             read_vtk(candidate, require_flow_fields=final))
        elif name in {"Retau.unf", "times.unf"}:
            if path.stat().st_size % 8 or candidate.stat().st_size % 8:
                raise ValueError("Incomplete long-run output: " + name)
            original = np.fromfile(path, dtype="<f8")
            changed = np.fromfile(candidate, dtype="<f8")
            if name == "Retau.unf":
                if original.size % 2 or changed.size % 2:
                    raise ValueError("Incomplete long-run output: " + name)
                original, changed = original.reshape(-1, 2), changed.reshape(-1, 2)
                results[name] = {field: compare_long_vectors(original[:, component], changed[:, component])
                                 for component, field in enumerate(("Retau", "friction_velocity"))}
            else:
                results[name] = {"time": compare_long_vectors(original, changed, criterion="pointwise")}
        elif path.suffix == ".txt":
            original, changed = np.loadtxt(path, ndmin=2), np.loadtxt(candidate, ndmin=2)
            if original.shape != changed.shape:
                raise ValueError("Long-run output shape differs: " + name)
            results[name] = {"frame_index": compare_long_vectors(original[:, 0], changed[:, 0], criterion="exact")}
            for component in range(1, original.shape[1]):
                results[name][f"column{component}"] = compare_long_vectors(
                    original[:, component], changed[:, component], criterion="pointwise")
        else:
            results[name] = {"bytes": {"passed": path.read_bytes() == candidate.read_bytes(), "criterion": "exact"}}
    return {"passed": all(row["passed"] for components in results.values() for row in components.values()),
            "output_count": len(results), "files": results}


class FieldComparisonError(ValueError):
    def __init__(self, comparison, *, label=None):
        self.comparison = comparison
        failed = sorted((name, row) for name, row in comparison.items() if not row["passed"])
        details = []
        for name, row in failed[:3]:
            maximum = row["max_absolute_error"]
            error = f"{maximum:.6g}" if maximum is not None else "nonfinite"
            details.append(f"{name} ({row['failed_values']}/{row['values']}, max_abs={error})")
        if len(failed) > 3:
            details.append(f"+{len(failed) - 3} more fields")
        prefix = f"{label}: " if label else ""
        super().__init__(prefix + "Full-precision field comparison failed: " + ", ".join(details))


def checked_fields(actual, reference):
    result = compare_fields(actual, reference, FIELD_POLICY)
    if not all(row["passed"] for row in result.values()):
        raise FieldComparisonError(result)
    return result


def inspect_diagnostics(directory):
    directory = Path(directory)
    reference = read_fields(directory / "native-reference")
    comparisons = {}
    for name in ("native-repeat", "resident"):
        candidate = directory / name
        if not candidate.is_dir():
            continue
        result = compare_fields(read_fields(candidate, len(reference)), reference, FIELD_POLICY)
        failed = {field: row for field, row in result.items() if not row["passed"]}
        comparisons[name] = {"passed": not failed, "failed_fields": failed}
    if not comparisons:
        raise ValueError("No native-repeat or resident diagnostic directory found")
    print(json.dumps({"reference": str(directory / "native-reference"), "comparisons": comparisons},
                     indent=2, allow_nan=False))
    if not all(row["passed"] for row in comparisons.values()):
        raise SystemExit(1)


def uses_natural_channel_policy(case, config, steps):
    return case == "channel" and config is None and steps is None


def require_natural_channel(row, log):
    timing = row["application_timing"]
    if timing["completed_steps"] != 2458 or "Time limit reached." not in log:
        raise ValueError("The approved 1% rule requires natural channel completion at 2458 steps")


def require_long_comparison(row, label):
    failures = [name for name, result in row.get("fields", {}).items() if not result["passed"]]
    failures += [f"{name}:{component}" for name, components in row["outputs"]["files"].items()
                 for component, result in components.items() if not result["passed"]]
    if failures:
        details = ", ".join(failures[:3])
        if len(failures) > 3:
            details += f", +{len(failures) - 3} more"
        raise ValueError(f"{label}: natural channel 1% comparison failed: {details}")


def run(args):
    output = args.output_dir.resolve() / "resident"
    work = args.work_dir.resolve() / "resident"
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report = {"schema_version": 1, "status": "running", "release_qualified": False,
              "diagnostics": [], "runs": [], "work": str(work)}
    write(report_path, report)
    try:
        work.mkdir(parents=True, exist_ok=False)
        hardware = doctor(args, require_gpu=True, fortran=args.fortran)
        gpu = hardware["gpu"]
        architecture = gpu["compute_capability"].replace(".", "")
        fortran = Path(shutil.which(args.fortran) or args.fortran).resolve(strict=True)
        cxx = Path(hardware["toolchain"]["cuda_host_cxx"]).resolve(strict=True)
        nvcc = Path(hardware["toolchain"]["nvcc"]).resolve(strict=True)
        dependencies = None
        if args.app_root is None or args.vkfft is None:
            dependencies = prepare_dependencies(work / "dependencies", application=args.app_root is None,
                                                vkfft=args.vkfft is None)
        app_root = args.app_root if args.app_root is not None else dependencies
        vkfft = args.vkfft if args.vkfft is not None else dependencies / "VkFFT"
        basis = prepare_application_basis(app_root, work / "application-basis")
        application = application_paths(basis)
        source, advection = select_configuration(application, args.case, args.config)
        configuration = prepare_config(source, work / "config", steps=args.steps, advection=advection)
        natural_channel = uses_natural_channel_policy(args.case, args.config, args.steps)
        report.update(hardware=hardware, configuration=configuration, case=args.case,
                      dependencies={"application_root": str(app_root), "vkfft": str(vkfft)},
                      source_identities={"compiler": source_identity(COMPILER / "compiler"),
                                         "pipeline": source_identity(PIPELINE),
                                         "config": source_identity(work / "config"),
                                         "solver": solver_identity(args.solver),
                                         "runner_sha256": digest(__file__)},
                      threads=args.threads, build_jobs=args.jobs, warmups=args.warmups, repeats=args.repeats,
                      cache_policy=("separate initially empty diagnostic and timing caches; "
                                    "warmups populate timing cache"),
                      timing_scope=("complete process including startup, cache lookup or compilation, "
                                    "output and teardown"),
                      validation_policy=("natural-channel-componentwise-relative-l2-1-percent"
                                         if natural_channel else "pointwise-1e-6-1e-5"),
                      validation_scope=("all full-precision fields, bounds and outputs; original time checks"
                                        if natural_channel else
                                        "full-precision allocated fields and bounds; final and available averaged VTK"),
                      poisson_execution_verified=False)
        build = work / "build"
        build_command = [sys.executable, PIPELINE / "pipeline.py", "resident-build",
                         "--elmm", application["elmm"], "--poisfft", application["PoisFFT"],
                         "--compiler", COMPILER, "--solver", args.solver, "--vkfft", vkfft,
                         "--output", build, "--fortran", fortran, "--cxx", cxx, "--nvcc", nvcc,
                         "--architecture", architecture, "--jobs", str(args.jobs)]
        report["build_command"] = list(map(str, build_command))
        write(report_path, report)
        build_env = dict(os.environ)
        build_env.pop("LD_PRELOAD", None)
        report["build_seconds"] = command(
            build_command, cwd=work, env=build_env, log=output / "build.log", timeout=args.build_timeout)
        sources = (("compiler", COMPILER / "compiler"), ("pipeline", PIPELINE),
                   ("config", work / "config"))
        for name, root in sources:
            if source_identity(root) != report["source_identities"][name]:
                raise ValueError(name + " source changed during build")
        if solver_identity(args.solver) != report["source_identities"]["solver"]:
            raise ValueError("Solver checkout changed during build")
        if digest(__file__) != report["source_identities"]["runner_sha256"]:
            raise ValueError("Comparison runner changed during build")
        built = json.loads((build / "built.json").read_text())
        if built["status"] != "PASS":
            raise RuntimeError("Resident build did not pass")
        report["build"] = built
        runtime_env = application_environment(args.threads, gpu["uuid"], None, traced=False)
        report["runtime_environment"] = {
            key: value for key, value in runtime_env.items()
            if key.startswith(("OMP_", "GOMP_", "KMP_", "OPENBLAS_", "MKL_")) or
            key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")}
        binaries = built["binaries"]
        identities = [*binaries.values(), built["provider"]]

        def verify_binaries():
            if any(digest(row["path"]) != row["sha256"] for row in identities):
                raise ValueError("Built executable or provider changed during comparison")

        caches = {}
        for name in ("diagnostic", "timing"):
            caches[name] = work / (name + "-cache")
            caches[name].mkdir(mode=0o700)
        packages = json.loads((build / "packages.json").read_text())
        reference_fields = reference_vtk = None
        reference_directory = None
        expected_steps = None
        for name, mode in (("native-reference", "native"), ("native-repeat", "native"), ("resident", "resident")):
            verify_binaries()
            print("Starting diagnostic:", name, flush=True)
            is_gpu = mode == "resident"
            env = application_environment(args.threads, gpu["uuid"], caches["diagnostic"] if is_gpu else None,
                                          traced=is_gpu)
            binary = binaries["ELMM-full20" if is_gpu else "ELMM-native-full20"]["path"]
            row, vtk, log = run_application(binary, work / "config", configuration, output / "diagnostics" / name,
                                            env, args.timeout, expected_steps)
            row.update(mode=mode, traced=is_gpu, timing_sample=False)
            report["diagnostics"].append(row)
            if natural_channel:
                require_natural_channel(row, log)
            field_count = None if reference_fields is None else len(reference_fields)
            actual_fields = read_fields(Path(row["directory"]), field_count)
            row["field_count"] = len(actual_fields)
            if reference_fields is None:
                reference_fields, reference_vtk = actual_fields, vtk
                reference_directory = Path(row["directory"])
                expected_steps = row["application_timing"]["completed_steps"]
                if natural_channel:
                    outputs = {str(path.relative_to(reference_directory / "output"))
                               for path in (reference_directory / "output").rglob("*") if path.is_file()}
                    if set(reference_fields) != NATURAL_CHANNEL_FIELDS or outputs != NATURAL_CHANNEL_OUTPUTS:
                        raise ValueError("Natural channel reference is missing approved fields or outputs")
            elif natural_channel:
                if reference_directory is None:
                    raise RuntimeError("Natural channel reference directory is missing")
                row["fields"] = compare_long_fields(actual_fields, reference_fields)
                row["vtk_pointwise"] = compare_outputs(reference_vtk, vtk, compare_vtk)
                row["outputs"] = compare_long_outputs(reference_directory / "output", Path(row["directory"]) / "output")
                require_long_comparison(row, name)
            else:
                try:
                    row["fields"] = checked_fields(actual_fields, reference_fields)
                except FieldComparisonError as error:
                    row["fields"] = error.comparison
                    raise FieldComparisonError(error.comparison, label=name) from None
                row["vtk"] = compare_vtks(reference_vtk, vtk)
            if is_gpu:
                row["gpu_activity"] = verify_resident_execution(
                    packages, log, ["subgrid::sgs_sigma_uniform", "dynamics::convection"])
            write(report_path, report)
            print("Completed diagnostic:", name, "steps", row["application_timing"]["completed_steps"], flush=True)
        for round_number in range(args.warmups + args.repeats):
            order = ("native", "resident") if round_number % 2 == 0 else ("resident", "native")
            for mode in order:
                verify_binaries()
                is_gpu = mode == "resident"
                env = application_environment(args.threads, gpu["uuid"], caches["timing"] if is_gpu else None,
                                              traced=False)
                binary = binaries["ELMM" if is_gpu else "ELMM-native"]["path"]
                directory = output / "runs" / f"round-{round_number:02}" / mode
                print("Starting", mode, "round", round_number, flush=True)
                row, vtk, log = run_application(binary, work / "config", configuration, directory,
                                                env, args.timeout, expected_steps)
                row.update(mode=mode, round=round_number, warmup=round_number < args.warmups,
                           traced=False)
                report["runs"].append(row)
                if natural_channel:
                    if reference_directory is None:
                        raise RuntimeError("Natural channel reference directory is missing")
                    require_natural_channel(row, log)
                    row["vtk_pointwise"] = compare_outputs(reference_vtk, vtk, compare_vtk)
                    row["outputs"] = compare_long_outputs(reference_directory / "output", directory / "output")
                    require_long_comparison(row, f"round {round_number} {mode}")
                else:
                    row["vtk"] = compare_vtks(reference_vtk, vtk)
                write(report_path, report)
                print(mode, "round", round_number, "seconds", round(row["wall_seconds"], 3), flush=True)
        verify_binaries()
        summary = {}
        for mode in ("native", "resident"):
            samples = [row["wall_seconds"] for row in report["runs"] if row["mode"] == mode and not row["warmup"]]
            summary[mode] = {"samples_seconds": samples, "median_seconds": statistics.median(samples)}
        summary["speedup"] = summary["native"]["median_seconds"] / summary["resident"]["median_seconds"]
        report.update(status="passed", summary=summary, figures=str(output / "figures"))
        write(report_path, report)
        plot_experiment(report_path, output / "figures")
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write(report_path, report)
    print("Report:", report_path, flush=True)
    print("Figures:", output / "figures", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    deps = Path(os.environ["DEPS_ROOT"]) if os.environ.get("DEPS_ROOT") else None
    parser.add_argument("--app-root", type=Path, default=deps)
    parser.add_argument("--vkfft", type=Path, default=deps / "VkFFT" if deps is not None else None)
    solver = Path(os.environ.get("SOLVER_ROOT", WORKSPACE / "poisson-solver"))
    parser.add_argument("--solver", type=Path, default=solver)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "out")))
    work = Path(os.environ["WORK_DIR"]) if "WORK_DIR" in os.environ else None
    parser.add_argument("--work-dir", type=Path, default=work)
    parser.add_argument("--case", choices=("channel-simple", "channel"), default="channel-simple")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--build-timeout", type=int, default=3600)
    fortran = os.environ.get("FC") or ("gfortran-15" if shutil.which("gfortran-15") else "gfortran")
    cuda_host = os.environ.get("CUDAHOSTCXX") or os.environ.get("CUDA_HOST_CXX") or os.environ.get("CXX")
    parser.add_argument("--fortran", default=fortran)
    parser.add_argument("--cuda-host", default=cuda_host)
    parser.add_argument("--nvcc", default=os.environ.get("NVCC"))
    parser.add_argument("--arch", default="auto")
    parser.add_argument("--gpu")
    parser.add_argument("--inspect-diagnostics", type=Path, metavar="DIR",
                        help="compare saved diagnostic dumps without rerunning the application")
    args = parser.parse_args(argv)
    if args.inspect_diagnostics is not None:
        inspect_diagnostics(args.inspect_diagnostics)
        return
    if args.work_dir is None:
        args.work_dir = args.output_dir / "work"
    if args.timeout is None:
        args.timeout = 3600 if uses_natural_channel_policy(args.case, args.config, args.steps) else 900
    if min(args.threads, args.jobs, args.repeats, args.timeout, args.build_timeout) < 1 or args.warmups < 0:
        parser.error("threads, jobs, repeats and timeouts must be positive; warmups must be nonnegative")
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
