import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .bootstrap import ensure_repo, setup
from .build import build, preprocessing_defines
from .common import (
    ROOT,
    clean_environment,
    command,
    digest_tree,
    python_environment,
    write_json,
)
from .configuration import prepare_config
from .gpu_validation import gpu_activity, native_policy_exercised, trace_environment
from .instrumentation import execution_coverage
from .report import render, summary
from .validation import compare_vtk, read_vtk, validate_log

TOOLS = ("baseline", "local")


def tool_list(value):
    values = list(dict.fromkeys(value.split(",")))
    if not values or any(v not in TOOLS for v in values):
        raise argparse.ArgumentTypeError("Choose comma-separated baseline,local")
    # Validation always requires a native reference.
    return ["baseline", *[v for v in values if v != "baseline"]]


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def compiler_identity():
    versions = {}
    for env, default in (("FC", "gfortran"), ("CC", "gcc"), ("CXX", "g++")):
        executable = shutil.which(os.environ.get(env, default))
        if not executable:
            raise RuntimeError(f"Compiler {os.environ.get(env, default)} not found")
        versions[env] = {"executable": executable,
                         "version": command([executable, "--version"])[0]}
    return versions


def scoped_configuration(args):
    """Validate explicit scope configuration without inferring capture facts."""
    model = getattr(args, "memory_model", "call")
    transfers = getattr(args, "scope_transfers", "direct")
    execution = getattr(args, "scope_execution", None)
    if getattr(args, "scope_measurement", None) is not None or getattr(args, "scope_cost", None) is not None:
        raise ValueError("Retired public scope measurement/cost options")
    if transfers not in {"direct", "pinned", "pipelined", "auto", "coalesced"}:
        raise ValueError("Unknown scope transfer mode")
    if execution not in {None, "graph"}:
        raise ValueError("Public source scopes require graph execution")
    values = {"entry": getattr(args, "scope_entry", None), "source": getattr(args, "scope_source", None),
              "source_files": getattr(args, "scope_source_file", None) or [],
              "capture_facts": getattr(args, "scope_facts", None),
              "effect_contracts": getattr(args, "scope_effect_contracts", None)}
    if model == "call":
        if transfers != "direct" or execution is not None or any(values[key] for key in values):
            raise ValueError("--scope-* options require --memory-model scoped")
        return None
    if model != "scoped":
        raise ValueError("Unknown memory model")
    if args.target != "gpu" or "local" not in args.tools or any(tool not in {"baseline", "local"} for tool in args.tools):
        raise ValueError("Scoped memory requires --target gpu and --tools local")
    if args.gpu_policy not in {"sections", "auto"}:
        raise ValueError("Scoped memory requires --gpu-policy sections or auto")
    if not values["entry"] or not values["source"] or not values["capture_facts"]:
        raise ValueError("Scoped memory requires --scope-entry, --scope-source and --scope-facts")
    for key in ("capture_facts", "effect_contracts"):
        if values[key]:
            values[key] = Path(values[key]).resolve(strict=True)
    values["transfers"] = transfers
    values["execution"] = "graph"
    return values


def prepare_build(tool, args, setup_state, compilers):
    defines = preprocessing_defines(args.build_arg)
    repositories = ROOT / ".cache" / "repos"
    key_data = {"tool": tool, "dependencies": setup_state["pins"], "compilers": compilers,
                "packages": setup_state["packages"].get(tool), "build_args": args.build_arg,
                "preprocessor_defines": defines,
                "native_libraries": setup_state["native"],
                "source_hashes": {name: digest_tree(repositories / name / "src") for name in ("elmm", "PoisFFT")}}
    paths = [ROOT / "elmm_pipeline" / "build.py"]
    # The build helper imports this module for CPU and GPU builds alike.
    paths.append(ROOT / "elmm_pipeline" / "gpu_toolchain.py")
    if tool == "local":
        key_data["local_compiler"] = digest_tree(repositories / "local-compiler" / "compiler")
        key_data["memory_model"] = getattr(args, "memory_model", "call")
    scope_options = None
    if tool == "local" and getattr(args, "memory_model", "call") == "scoped":
        scope_options = scoped_configuration(args)
        from .scoped_adaptation import scope_inputs
        root, scope_sources = scope_inputs(repositories / "elmm" / "src", scope_options)
        scope_options = {**scope_options, "source": str(scope_sources[0].relative_to(root)),
                         "source_files": [str(path.relative_to(root)) for path in scope_sources[1:]]}
        key_data["scope"] = {"entry": scope_options["entry"], "source": scope_options["source"],
                             "source_files": scope_options["source_files"],
                             "transfers": scope_options["transfers"],
                             "execution": scope_options["execution"],
                             "capture_facts_sha256": hashlib.sha256(scope_options["capture_facts"].read_bytes()).hexdigest(),
                             "effect_contracts_sha256": (hashlib.sha256(scope_options["effect_contracts"].read_bytes()).hexdigest()
                                                          if scope_options["effect_contracts"] else None)}
    if tool != "baseline":
        key_data["target"] = args.target
        paths += [ROOT / "elmm_pipeline" / name for name in (
            "adaptation.py", "discovery.py", "structural.py", "local_adapter.py",
            "instrumentation.py", "common.py", "owner_measurement.py", "application_build.py")]
        if args.target == "gpu":
            paths += [ROOT / "elmm_pipeline" / name for name in ("gpu_discovery.py", "gpu_calls.py", "gpu_backend.py")]
            if scope_options:
                paths += [ROOT / "elmm_pipeline" / name for name in ("scoped_adaptation.py", "scope_extraction.py")]
            key_data["gpu_toolchain"] = args.gpu_toolchain
            key_data["gpu_policy"] = getattr(args, "gpu_policy", "always")
            key_data["host_threads"] = args.threads
            profile_path = getattr(args, "calibration_profile", None)
            key_data["calibration_profile"] = ({"sha256": hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()}
                                                if profile_path else None)
        key_data["workers"] = digest_tree(ROOT / "elmm_pipeline" / "tool_workers")
        key_data["selection"] = {"sources": args.source_pattern, "include": args.select,
                                 "exclude": args.exclude, "marked_only": args.marked_only,
                                 "coverage": not args.no_coverage}
    key_data["implementation"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    # Custom compiled hooks affect build identity even when their argument path is unchanged.
    key_data["custom_inputs"] = {}
    for arg in args.build_arg:
        value = arg.split("=", 1)[-1]
        path = Path(value)
        if path.is_file():
            key_data["custom_inputs"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]
    directory = ROOT / "work" / "builds" / f"{tool}-{key}"
    stamp = directory / "complete.json"
    if stamp.is_file():
        data = json.loads(stamp.read_text())
        binary = directory / "bin" / "ELMM"
        if binary.is_file() and hashlib.sha256(binary.read_bytes()).hexdigest() == data["binary_sha256"]:
            return binary, data
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    gpu_options = dict(args.gpu_toolchain or {})
    if tool == "local" and args.target == "gpu":
        gpu_options.update(gpu_policy=getattr(args, "gpu_policy", "always"), host_threads=args.threads)
        if scope_options:
            gpu_options["scope_transfers"] = scope_options["transfers"]
            gpu_options["scope_execution"] = scope_options["execution"]
        profile_path = getattr(args, "calibration_profile", None)
        if profile_path:
            copied_profile = directory / "calibration.json"
            shutil.copy2(profile_path, copied_profile)
            gpu_options["calibration_profile"] = str(copied_profile)
    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.o", "*.mod", ".sconsign*")
    for name in ("elmm", "PoisFFT"):
        shutil.copytree(repositories / name, directory / name, ignore=ignore)
    source_root = directory / "elmm" / "src"
    if scope_options:
        facts = json.loads(scope_options["capture_facts"].read_text())
        from .local_adapter import LocalAdapterError, remap_scope_facts
        try:
            facts = remap_scope_facts(facts, repositories / "elmm" / "src", source_root)
        except LocalAdapterError as error:
            raise ValueError(str(error)) from error
        copied_facts = directory / "scope-captures.json"
        write_json(copied_facts, facts)
        scope_options = {**scope_options, "capture_facts": copied_facts}
        if scope_options["effect_contracts"]:
            copied_contracts = directory / "scope-effect-contracts.json"
            shutil.copy2(scope_options["effect_contracts"], copied_contracts)
            scope_options["effect_contracts"] = copied_contracts
    adapter = {"tool": "baseline", "coverage": "Unmodified ELMM source"}
    if tool != "baseline":
        from .adaptation import adapt_project
        adapter = adapt_project(tool, source_root, ROOT / ".cache/envs" / tool / "bin/python",
                                directory / "generated", compiler_root=repositories / "local-compiler",
                                source_patterns=args.source_pattern, include=args.select,
                                exclude=args.exclude, marked_only=args.marked_only,
                                defines=defines, coverage=not args.no_coverage,
                                target=args.target, gpu_options=gpu_options,
                                memory_model=getattr(args, "memory_model", "call"), scope_options=scope_options)
        if not adapter["regions"] and not adapter.get("native_success"):
            return None, {"identity": key_data, "adapter": adapter,
                          "reason": "No eligible regions accepted by this backend; native code preserved."}
    binary = build(directory / "elmm", directory / "PoisFFT", directory / "bin",
                   ROOT / ".cache/envs/build/bin/python", ROOT / ".cache/native/lib",
                   args.jobs, args.threads, args.build_arg,
                   extra_sources=[Path(p) for p in adapter.get("extra_sources", [])],
                   patched_sources=[source_root / p for p in adapter.get("changed_files", [])],
                   gpu_sources=adapter.get("gpu_sources", []), gpu_options=args.gpu_toolchain,
                   **({"application_build": adapter["application_build"]} if "application_build" in adapter else {}))
    data = {"identity": key_data, "adapter": adapter,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "changed_source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                                      for name in adapter.get("changed_files", [])},
            "build_command": json.loads((binary.parent / "build-command.json").read_text())}
    # Adapter metadata can contain Paths; serialize these as explicit artifact paths.
    data = json.loads(json.dumps(data, default=str))
    write_json(stamp, data)
    return binary, data


def run(args):
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")
    scope_options = scoped_configuration(args)
    defines = preprocessing_defines(args.build_arg)
    if args.gpu_policy != "always" and (args.target != "gpu" or any(tool not in {"baseline", "local"} for tool in args.tools)):
        raise ValueError("GPU runtime policies require --target gpu and --tools local")
    if args.calibration_profile:
        args.calibration_profile = Path(args.calibration_profile).resolve(strict=True)
    args.gpu_toolchain = None
    if args.target == "gpu":
        from .gpu_toolchain import discover_toolchain
        languages = ["cuda"] if "local" in args.tools else []
        args.gpu_toolchain = discover_toolchain(languages=languages, nvcc=args.nvcc,
                                              cuda_host=args.cuda_host,
                                              arch=args.gpu_arch)
        if not args.gpu_toolchain["device"]["available"]:
            raise ValueError("GPU execution is unavailable in this environment: " +
                             str(args.gpu_toolchain["device"]["error"]))
    if not (ROOT / ".cache/setup.json").exists():
        raise ValueError("Run pipeline.py setup first")
    setup_state = json.loads((ROOT / ".cache/setup.json").read_text())
    pins = json.loads((ROOT / "dependencies.json").read_text())
    if setup_state["pins"] != pins:
        raise ValueError("Dependency pins changed; run setup again")
    for name, spec in pins.items():
        ensure_repo(name, spec)
    for tool in args.tools:
        if tool != "baseline":
            installed = command([ROOT / ".cache/envs" / tool / "bin/python", "-m", "pip", "freeze"],
                                env=python_environment())[0].splitlines()
            if installed != setup_state["packages"].get(tool):
                raise ValueError(f"{tool} environment changed; run setup to record updated packages")
    # SCons executes from src/, so resolve explicitly supplied custom files now.
    normalized_args = []
    for arg in args.build_arg:
        key, separator, value = arg.partition("=")
        if separator and Path(value).is_file():
            value = str(Path(value).resolve())
        normalized_args.append(key + separator + value)
    args.build_arg = normalized_args
    missing = set(args.tools) - set(setup_state["tools"])
    if missing:
        raise ValueError(f"Run setup for missing tools: {', '.join(sorted(missing))}")
    source = Path(args.config).resolve() if args.config else ROOT / ".cache/repos/elmm/examples" / args.example
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = Path(args.output).resolve() if args.output else ROOT / "results" / tag
    if destination.exists():
        raise ValueError(f"Output directory must be new: {destination}")
    destination.mkdir(parents=True)
    configuration = prepare_config(source, destination / "config", steps=args.steps, advection=args.advection)
    configuration["sha256"] = digest_tree(destination / "config")
    configuration["original_sha256"] = digest_tree(source if source.is_dir() else source.parent)
    compilers = compiler_identity()
    report = {"schema_version": 3, "created_utc": tag, "configuration": configuration,
              "settings": {"threads": args.threads, "repeats": args.repeats, "warmups": args.warmups,
                           "target": args.target, "gpu_profile": args.gpu_profile,
                           "gpu_policy": args.gpu_policy,
                           "memory_model": getattr(args, "memory_model", "call"),
                           "scope": ({key: str(value) if isinstance(value, Path) else value
                                      for key, value in scope_options.items()} if scope_options else None),
                           "calibration_profile": ({"source": str(args.calibration_profile),
                                "sha256": hashlib.sha256(args.calibration_profile.read_bytes()).hexdigest()}
                                if args.calibration_profile else None),
                           "timeout_seconds": args.timeout, "atol": args.atol, "rtol": args.rtol,
                           "build_args": args.build_arg,
                           "preprocessor_defines": defines,
                           "source_patterns": args.source_pattern, "selection": args.select,
                           "exclusions": args.exclude, "marked_only": args.marked_only,
                           "coverage_enabled": not args.no_coverage,
                           "environment": {k: v for k, v in clean_environment(args.threads).items()
                                           if k.startswith(("OMP_", "OPENBLAS_", "MKL_"))}},
              "dependencies": setup_state["pins"], "packages": setup_state["packages"],
              "host": {"platform": platform.platform(), "machine": platform.machine(),
                       "processor": platform.processor(), "cpu_count": os.cpu_count()},
              "compilers": compilers, "gpu_toolchain": args.gpu_toolchain,
              "native_libraries": setup_state["native"], "variants": {}}
    compiler_info = ROOT / ".cache/repos/local-compiler/source.json"
    if "local" in args.tools and compiler_info.exists():
        report["local_compiler"] = json.loads(compiler_info.read_text())
        actual = digest_tree(compiler_info.parent / "compiler")
        if actual != report["local_compiler"]["compiler_sha256"]:
            raise ValueError("Cached local compiler changed; repeat setup with its explicit source checkout")
    binaries = {}

    def save():
        write_json(destination / "report.json", report)
        (destination / "report.md").write_text(render(report))

    try:
        # Finish all compilation before starting timed runs.
        for tool in args.tools:
            print(f"Preparing {tool} application build", flush=True)
            report["variants"][tool] = {"status": "pending", "runs": []}
            binary, build_metadata = prepare_build(tool, args, setup_state, compilers)
            if binary is None:
                report["variants"][tool] = {"status": "unsupported", "build": build_metadata,
                                            "reason": build_metadata["reason"]}
                continue
            binaries[tool] = binary
            report["variants"][tool] = {"status": "pending", "build": build_metadata, "runs": []}
            artifact_dir = destination / "sources" / tool
            artifact_dir.mkdir(parents=True)
            for relative in build_metadata["adapter"].get("changed_files", []):
                target = artifact_dir / "native" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(binary.parent.parent / "elmm/src" / relative, target)
            shutil.copy2(binary.parent / "build.log", artifact_dir / "build.log")
            calibration = binary.parent.parent / "calibration.json"
            if calibration.is_file():
                shutil.copy2(calibration, artifact_dir / "calibration.json")
            for name in ("scope-captures.json", "scope-effect-contracts.json"):
                scope_input = binary.parent.parent / name
                if scope_input.is_file():
                    shutil.copy2(scope_input, artifact_dir / name)
            generated = binary.parent.parent / "generated"
            if generated.exists():
                shutil.copytree(generated, artifact_dir / "generated")
            gpu_build = binary.parent / "gpu"
            if gpu_build.exists():
                shutil.copytree(gpu_build, artifact_dir / "gpu-build")
        reference = None
        reference_steps = None
        sequence = list(binaries)

        def execute(tool, label, *, warmup, trace=False):
            nonlocal reference, reference_steps
            variant = report["variants"][tool]
            directory = destination / "runs" / tool / label
            shutil.copytree(destination / "config", directory)
            output = directory / "output"
            if output.exists():
                shutil.rmtree(output)
            print(f"Running {tool} {label}", flush=True)
            environment = trace_environment(clean_environment(args.threads), tool, enabled=trace)
            argv = [binaries[tool], *configuration["runtime_arguments"]]
            profile = None
            if trace and args.gpu_profile == "nsys":
                from .gpu_profiling import profile_command
                log, wall, profile = profile_command(argv, directory / "profile", environment,
                                                     directory, timeout=args.timeout)
                if profile.get("application_executed") is False:
                    log, wall = command(argv, cwd=directory, env=environment,
                                        log=directory / "run.log", timeout=args.timeout)
                else:
                    (directory / "run.log").write_text(log)
            else:
                log, wall = command(argv, cwd=directory, env=environment,
                                    log=directory / "run.log", timeout=args.timeout)
            timings = validate_log(log, configuration["expected_max_steps"],
                                   expected_completed_steps=reference_steps)
            if tool != "baseline":
                adapter = variant["build"]["adapter"]
                timings["execution_coverage"] = execution_coverage(
                    log, adapter["regions"], enabled=adapter["coverage_enabled"])
                if trace:
                    timings["gpu_activity"] = gpu_activity(log, tool, adapter["regions"],
                                                            scope_manifest=adapter.get("scope_manifest"),
                                                            scope_packages=adapter.get("scope_packages"))
                    if profile is not None:
                        timings["gpu_profile"] = profile
            vtk = read_vtk(output / "out.vtk")
            timings["grid_shape"] = list(vtk[0])
            if reference is None:
                if tool != "baseline":
                    raise RuntimeError("First run must be baseline")
                reference, reference_steps = vtk, timings["completed_steps"]
            validation = compare_vtk(reference, vtk, atol=args.atol, rtol=args.rtol)
            record = {"warmup": warmup, "label": label, "wall_seconds": wall, **timings,
                      "validation": validation, "log": str(directory.relative_to(destination) / "run.log")}
            if not validation["passed"]:
                variant["status"] = "failed"
                variant["reason"] = "Final application state differs from baseline beyond tolerance"
                variant["failed_run"] = record
                raise RuntimeError(f"{tool}: numerical validation failed; inspect {destination / 'report.json'}")
            return record

        if args.target == "gpu":
            # Profiles/verbose device traces are never timed benchmark samples.
            # Use the same complete workload and validate their final fields too.
            for tool in sequence:
                record = execute(tool, "gpu-validation", warmup=True, trace=tool != "baseline")
                report["variants"][tool]["gpu_validation"] = record
            save()
        for index in range(args.warmups + args.repeats):
            # Rotate order to reduce systematic thermal/order bias; baseline runs first
            # in round zero to establish the deterministic reference state.
            order = sequence[index % len(sequence):] + sequence[:index % len(sequence)]
            for tool in order:
                variant = report["variants"][tool]
                warmup = index < args.warmups
                label = (f"warmup-{index + 1:03d}" if warmup else f"repeat-{index - args.warmups + 1:03d}")
                variant["runs"].append(execute(tool, label, warmup=warmup))
        for variant in report["variants"].values():
            if variant["status"] == "pending":
                variant["status"] = "ok"
                variant["summary"] = summary(variant["runs"])
                adapter = variant["build"]["adapter"]
                native_policy = native_policy_exercised(
                    variant.get("gpu_validation", {}).get("gpu_activity", {}), adapter.get("regions", [])) or adapter.get("native_success", False)
                if adapter["tool"] != "baseline" and adapter["coverage_enabled"] and not native_policy and not any(
                        r["execution_coverage"]["executed_regions"] for r in variant["runs"]):
                    variant["status"] = "not_exercised"
                    variant["reason"] = "No outlined region executed for this configuration."
                if args.target == "gpu" and adapter["tool"] != "baseline" and not native_policy and not variant["gpu_validation"]["gpu_activity"]["verified"]:
                    variant["status"] = "not_exercised"
                    variant["reason"] = "No computation kernel was verified on the GPU for this configuration."
        save()
    except Exception as error:
        report["error"] = str(error)
        if "tool" in locals() and tool in report["variants"]:
            report["variants"][tool]["status"] = "failed"
            report["variants"][tool]["reason"] = str(error)
        save()
        raise
    print(f"Report: {destination / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Portable ELMM application benchmark pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("setup", help="Fetch pinned sources and provision private Python environments")
    install.add_argument("--tools", type=tool_list, default=list(TOOLS))
    install.add_argument("--local-compiler", type=Path, help="Explicit checkout containing compiler/; copied into cache")
    install.add_argument("--fc", help="GNU Fortran compiler executable (e.g. gfortran-15)")
    analyze = sub.add_parser("analyze", help="Discover eligible regions in any Fortran source tree")
    analyze.add_argument("--source-tree", type=Path, required=True)
    analyze.add_argument("--python", type=Path, required=True, help="Explicit Python environment with fparser")
    analyze.add_argument("--output", type=Path, required=True, help="New analysis artifact directory")
    analyze.add_argument("--source-pattern", action="append", help="Relative glob; default **/*.f90")
    analyze.add_argument("--define", action="append", default=[], help="Active preprocessor symbol")
    analyze.add_argument("--fc", help="GNU Fortran compiler used for preprocessing")
    analyze.add_argument("--target", choices=("cpu", "gpu"), default="cpu")
    analyze.add_argument("--local-compiler", type=Path,
                         help="Explicit compiler checkout for broader GPU discovery using its public CLI")
    measure = sub.add_parser("run", help="Build, execute, validate and report")
    select = measure.add_mutually_exclusive_group()
    select.add_argument("--example", default="channel", help="ELMM examples/ subdirectory (default: channel)")
    select.add_argument("--config", help="Any configuration directory, or a .conf file in it")
    measure.add_argument("--tools", type=tool_list, default=list(TOOLS))
    measure.add_argument("--target", choices=("cpu", "gpu"), default="cpu",
                         help="GPU uses local CUDA; baseline remains CPU/OpenMP")
    measure.add_argument("--nvcc", help="CUDA compiler path")
    measure.add_argument("--cuda-host", help="Host C++ compiler supported by nvcc")
    measure.add_argument("--gpu-arch", help="NVIDIA compute capability, e.g. 86; default queries GPU")
    measure.add_argument("--gpu-policy", choices=("always", "sections", "auto", "chunked", "hybrid"), default="always",
                         help="Local compiler runtime policy; always preserves the existing forced GPU path")
    measure.add_argument("--memory-model", choices=("call", "scoped"), default="call",
                         help="Local CUDA memory ownership; scoped uses compiler-owned source scopes")
    measure.add_argument("--scope-transfers", choices=("direct", "pinned", "pipelined", "auto", "coalesced"), default="direct",
                         help="Pass the scoped transfer mode to the compiler; independent of GPU placement")
    measure.add_argument("--scope-execution", choices=("graph",),
                         help="Source scope execution; graph is implicit with scoped memory")
    measure.add_argument("--scope-entry", help="Explicit procedure or module::procedure for scoped execution")
    measure.add_argument("--scope-source", help="Entry source path relative to the application src/ tree")
    measure.add_argument("--scope-source-file", action="append", default=[],
                         help="Additional original source input relative to src/; repeat for the available closure")
    measure.add_argument("--scope-facts", type=Path,
                         help="Explicit versioned capture initialization/lifetime/participation JSON")
    measure.add_argument("--scope-effect-contracts", type=Path,
                         help="Optional compiler public effect-contract JSON")
    measure.add_argument("--calibration-profile", type=Path,
                         help="Offline hardware calibration JSON for local runtime cost decisions")
    measure.add_argument("--gpu-profile", choices=("nsys", "none"), default="nsys",
                         help="Profile separate GPU validation runs; measured runs never use profiling")
    measure.add_argument("--steps", type=positive, help="Explicit cap; omitted preserves the original duration")
    measure.add_argument("--advection", type=int, choices=(2, 4, 5, 6), help="Explicitly override main.conf method")
    measure.add_argument("--threads", type=positive, default=4)
    measure.add_argument("--jobs", type=positive, default=min(8, os.cpu_count() or 1))
    measure.add_argument("--repeats", type=positive, default=3)
    measure.add_argument("--warmups", type=int, default=1)
    measure.add_argument("--timeout", type=positive, default=3600, help="Maximum seconds per application process")
    measure.add_argument("--atol", type=float, default=1e-6)
    measure.add_argument("--rtol", type=float, default=1e-5)
    measure.add_argument("--output", help="New result directory (default: results/<UTC timestamp>)")
    measure.add_argument("--build-arg", action="append", default=[], help="Repeat SCons name=value options, including custom hooks")
    measure.add_argument("--fc", help="GNU Fortran compiler executable (e.g. gfortran-15)")
    measure.add_argument("--source-pattern", action="append", help="Optional relative source glob; default **/*.f90")
    measure.add_argument("--select", action="append", default=[], help="Optional procedure or module::procedure glob")
    measure.add_argument("--exclude", action="append", default=[], help="Procedure glob to leave native")
    measure.add_argument("--marked-only", action="store_true", help="Restrict to explicitly marked source procedures")
    measure.add_argument("--no-coverage", action="store_true", help="Disable inserted execution markers; coverage is unverified")
    args = parser.parse_args()
    if getattr(args, "fc", None):
        os.environ["FC"] = args.fc
    try:
        if args.command == "setup":
            setup(args.tools, args.local_compiler)
        elif args.command == "analyze":
            from .discovery import discover_project
            if args.output.exists():
                raise ValueError("Analysis output directory must be new")
            data = discover_project(args.source_tree, args.python, args.output,
                                    patterns=args.source_pattern or ["**/*.f90"], defines=args.define,
                                    backend=args.target, compiler_root=args.local_compiler,
                                    compiler_python=args.python if args.local_compiler else None)
            print(f"Discovered {len(data['regions'])} eligible regions; diagnostics in {args.output}")
        else:
            args.source_pattern = args.source_pattern or ["**/*.f90"]
            if any(not math.isfinite(t) or t < 0 for t in (args.atol, args.rtol)):
                raise ValueError("Validation tolerances must be finite and nonnegative")
            run(args)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
