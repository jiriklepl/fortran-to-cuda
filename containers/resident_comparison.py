#!/usr/bin/env python3
"""Build and compare the current resident ELMM and its native reference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from application_basis import prepare_application_basis
from sgs_experiment import (application_paths, compare_outputs, digest, read_outputs,
                            select_configuration)
from workflow import COMPILER, PIPELINE, WORKSPACE, doctor, source_identity
from elmm_pipeline.campaigns.halos import compare_fields, fields
from elmm_pipeline.campaigns.run import environment
from elmm_pipeline.configuration import prepare_config
from elmm_pipeline.resident_rk import verify_resident_execution
from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log

REQUIRED_FIELDS = {"fields-" + name + ".bin" for name in
                   "U V W Pr Viscosity U2 V2 W2 Ustar Vstar Wstar".split()}
FIELD_POLICY = {"dtype": "float64", "atol": 1e-6, "rtol": 1e-5,
                "metadata_format": "storage-shape-bounds-v1"}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def solver_identity(root):
    root = Path(root).resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--",
         "CMakeLists.txt", "dependencies.lock.json", "src", "include", "cmake", "docs"], text=True)
    return {"root": str(root), "revision": revision, "source_status": status.splitlines()}


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


def checked_fields(actual, reference):
    result = compare_fields(actual, reference, FIELD_POLICY)
    if not all(row["passed"] for row in result.values()):
        raise ValueError("Full-precision field comparison failed")
    return result


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
        cxx = Path(shutil.which(args.cuda_host) or args.cuda_host).resolve(strict=True)
        nvcc = Path(shutil.which(args.nvcc) or args.nvcc).resolve(strict=True)
        basis = prepare_application_basis(args.app_root, work / "application-basis")
        application = application_paths(basis)
        source, advection = select_configuration(application, args.case, args.config)
        configuration = prepare_config(source, work / "config", steps=args.steps, advection=advection)
        report.update(hardware=hardware, configuration=configuration,
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
                      validation_scope="full-precision allocated fields and bounds; final and available averaged VTK",
                      poisson_execution_verified=False)
        build = work / "build"
        build_command = [sys.executable, PIPELINE / "pipeline.py", "resident-build",
                         "--elmm", application["elmm"], "--poisfft", application["PoisFFT"],
                         "--compiler", COMPILER, "--solver", args.solver, "--vkfft", args.vkfft,
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
        expected_steps = None
        for name, mode in (("native-reference", "native"), ("native-repeat", "native"), ("resident", "resident")):
            verify_binaries()
            is_gpu = mode == "resident"
            env = application_environment(args.threads, gpu["uuid"], caches["diagnostic"] if is_gpu else None,
                                          traced=is_gpu)
            binary = binaries["ELMM-full20" if is_gpu else "ELMM-native-full20"]["path"]
            row, vtk, log = run_application(binary, work / "config", configuration, output / "diagnostics" / name,
                                            env, args.timeout, expected_steps)
            field_count = None if reference_fields is None else len(reference_fields)
            actual_fields = read_fields(Path(row["directory"]), field_count)
            if reference_fields is None:
                reference_fields, reference_vtk = actual_fields, vtk
                expected_steps = row["application_timing"]["completed_steps"]
            else:
                row["fields"] = checked_fields(actual_fields, reference_fields)
                row["vtk"] = compare_vtks(reference_vtk, vtk)
            if is_gpu:
                row["gpu_activity"] = verify_resident_execution(
                    packages, log, ["subgrid::sgs_sigma_uniform", "dynamics::convection"])
            row.update(mode=mode, traced=is_gpu, timing_sample=False, field_count=len(actual_fields))
            report["diagnostics"].append(row)
            write(report_path, report)
        for round_number in range(args.warmups + args.repeats):
            order = ("native", "resident") if round_number % 2 == 0 else ("resident", "native")
            for mode in order:
                verify_binaries()
                is_gpu = mode == "resident"
                env = application_environment(args.threads, gpu["uuid"], caches["timing"] if is_gpu else None,
                                              traced=False)
                binary = binaries["ELMM" if is_gpu else "ELMM-native"]["path"]
                directory = output / "runs" / f"round-{round_number:02}" / mode
                row, vtk, _log = run_application(binary, work / "config", configuration, directory,
                                                 env, args.timeout, expected_steps)
                row.update(mode=mode, round=round_number, warmup=round_number < args.warmups,
                           traced=False, vtk=compare_vtks(reference_vtk, vtk))
                report["runs"].append(row)
                write(report_path, report)
                print(mode, "round", round_number, "seconds", round(row["wall_seconds"], 3), flush=True)
        verify_binaries()
        summary = {}
        for mode in ("native", "resident"):
            samples = [row["wall_seconds"] for row in report["runs"] if row["mode"] == mode and not row["warmup"]]
            summary[mode] = {"samples_seconds": samples, "median_seconds": statistics.median(samples)}
        summary["speedup"] = summary["native"]["median_seconds"] / summary["resident"]["median_seconds"]
        report.update(status="passed", summary=summary)
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write(report_path, report)
    print("Report:", report_path, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    deps = Path(os.environ.get("DEPS_ROOT", PIPELINE / ".cache/repos"))
    parser.add_argument("--app-root", type=Path, default=deps)
    parser.add_argument("--vkfft", type=Path, default=deps / "VkFFT")
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
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--build-timeout", type=int, default=3600)
    parser.add_argument("--fortran", default=os.environ.get("FC", "gfortran-15"))
    parser.add_argument("--cuda-host", default=os.environ.get("CXX", "g++-14"))
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--arch", default="auto")
    parser.add_argument("--gpu")
    args = parser.parse_args(argv)
    if args.work_dir is None:
        args.work_dir = args.output_dir / "work"
    if min(args.threads, args.jobs, args.repeats, args.timeout, args.build_timeout) < 1 or args.warmups < 0:
        parser.error("threads, jobs, repeats and timeouts must be positive; warmups must be nonnegative")
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
