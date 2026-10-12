"""Collect production CUDA observations without enabling automatic placement.

Run sequentially under an external memory/time guard. A new directory retains
every source, command, preflight, raw batch and failure; the base stays intact.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from compiler.numerical_contract import numerical_build_contract, require_explicit_cuda_environment
from compiler.offload.numerical_calibration import require_numerical_profile_contract
from compiler.offload.numerical_execution import _hash
from compiler.offload.numerical_gpu_execution import (
    BATCH_KIND,
    ENVIRONMENT_KEYS,
    IDENTITY_KIND,
    MAX_RECORDS,
    PREFLIGHT_KIND,
    PREFLIGHT_RECORDS,
    SECTION,
    NumericalGpuExecutionError,
    compatibility_identity,
    current_gpu_identities,
    gpu_build_commands,
    profile_from_gpu_execution_measurements,
    validate_gpu_execution_preflight,
)


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def gpu_sources(recipes, generator, source_generator, registry_id):
    """Render exact retained regions; no hand-written numerical expressions."""
    from compiler.emission.common.abi import dimension_name
    from compiler.emission.common.c_family import cpp_type
    from compiler.emission.common.symbols import region_symbols
    from compiler.emission.cuda.kernels import generate_kernel, generate_launch
    from compiler.ir import referenced_symbols
    from compiler.offload.cpu_dependency_workloads import _cpp_header
    from compiler.offload.numerical_execution_workloads import NAMES

    if tuple(r.name for r in recipes) != NAMES:
        raise NumericalGpuExecutionError("complete original seventeen-recipe GPU registry required")
    precision = recipes[0].metadata["precision_bits"]
    real = "double" if precision == 64 else "float"
    workers = ["#include <cuda_runtime.h>", "#include <cstdint>", _cpp_header()]
    header = ["#pragma once", "#include <cuda_runtime.h>", "#include <cstdint>", "#include <cstddef>",
        "using execution_real=" + real + ";",
        "using gpu_worker=int(*)(int,const execution_real*,const execution_real*,execution_real*,cudaStream_t,std::uint64_t*);",
        "struct gpu_recipe { const char *name,*identity; gpu_worker launch; const int *sizes,*fits;",
        "std::size_t size_count,fit_count; const execution_real *training_a,*training_b,*training_reference;",
        "std::size_t training_a_count,training_b_count,training_reference_count;",
        "const execution_real *holdout_a,*holdout_b,*holdout_reference;",
        "std::size_t holdout_a_count,holdout_b_count,holdout_reference_count; };",
        "static constexpr const char *gpu_generator=" + json.dumps(generator) + ";",
        "static constexpr const char *gpu_source_generator=" + json.dumps(source_generator) + ";",
        "static constexpr const char *gpu_registry=" + json.dumps(registry_id) + ";"]
    entries = []
    for recipe in recipes:
        region = recipe.region
        symbols = {s.name: s for s in region_symbols(region)}
        bounds = set().union(*(referenced_symbols(e) for loop in region.loops for e in (loop.lower, loop.upper)))
        n, = (s for s in bounds if s.name == "n")
        arrays = {key: s for key, s in symbols.items() if s.rank}
        if (set(arrays) != {"a", "d", "output"} or set(symbols) - set(arrays) - {"n"} or len(region.loops) != 1
                or any(s.rank != 1 or cpp_type(s) != real for s in arrays.values())
                or recipe.metadata["precision_bits"] != precision):
            raise NumericalGpuExecutionError("GPU recipe does not retain the original three-array ABI")
        name = "execution_gpu_" + recipe.name
        namespace = name + "_detail"
        signature = ("int items,const " + real + "*a,const " + real + "*d," + real
                     + "*output,cudaStream_t stream,std::uint64_t*launches")
        header.append('extern "C" int ' + name + "(" + signature + ");")
        workers += ["namespace " + namespace + " {", *generate_kernel(region), "}",
                    'extern "C" int ' + name + "(" + signature + ") {",
                    "if(items<0||!launches)return int(cudaErrorInvalidValue);", "using namespace " + namespace + ";",
                    "const int " + n.cpp_name + "=items;"]
        for key, symbol in arrays.items():
            workers += [("const " if key != "output" else "") + real + "*" + symbol.cpp_name + "_device=" + key + ";",
                        "const std::size_t " + dimension_name(symbol, 1) + "=static_cast<std::size_t>(items);"]
        workers += [*generate_launch(region, stream="stream", profile=False,
                    error_check="if(auto error=cudaGetLastError();error!=cudaSuccess)return int(error);",
                    launch_record="++*launches;"), "return int(cudaSuccess);", "}"]
        fields = [json.dumps(recipe.name), json.dumps(recipe.metadata["recipe_id"]), name]
        for label, values in (("sizes", recipe.metadata["sizes"]), ("fits", recipe.metadata["fit_sizes"])):
            header.append("static constexpr int " + name + "_" + label + "[]={" + ",".join(map(str, values or [0])) + "};")
        fields += [name + "_sizes", name + "_fits", str(len(recipe.metadata["sizes"])), str(len(recipe.metadata["fit_sizes"]))]
        for split in ("training", "holdout"):
            a, d = recipe.lattices(split)
            values = (a, d, recipe.reference_period(split))
            for label, items in zip(("a", "b", "reference"), values, strict=True):
                var = name + "_" + split + "_" + label
                header.append("static constexpr execution_real " + var + "[]={" +
                              ",".join("execution_real(" + format(v, ".17g") + ")" for v in items) + "};")
                fields.append(var)
            fields.extend(str(len(items)) for items in values)
        entries.append("{" + ",".join(fields) + "}")
    header.append("static const gpu_recipe gpu_recipes[]={" + ",\n".join(entries) + "};")
    return {"workers.cu": "\n".join(workers) + "\n", "gpu_recipes.hpp": "\n".join(header) + "\n"}


def _parse(text):
    if len(text) > 8 * 1024 * 1024 or len(text.encode()) > 8 * 1024 * 1024:
        raise NumericalGpuExecutionError("GPU output byte budget exhausted")
    rows = []
    for line in text.splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError as error:
                raise NumericalGpuExecutionError("non-JSON GPU execution output") from error
            if not isinstance(row, dict) or row.get("kind") not in {IDENTITY_KIND, PREFLIGHT_KIND, BATCH_KIND}:
                raise NumericalGpuExecutionError("unknown GPU execution output")
            rows.append(row)
            if len(rows) > MAX_RECORDS:
                raise NumericalGpuExecutionError("GPU output record budget exhausted")
    if not rows or rows[0].get("kind") != IDENTITY_KIND or any(r["kind"] == IDENTITY_KIND for r in rows[1:]):
        raise NumericalGpuExecutionError("one GPU identity must precede observations")
    return rows


def calibrate_gpu_execution(profile, args, *, run, tool):
    require_explicit_cuda_environment()
    require_numerical_profile_contract(profile)
    if SECTION in profile:
        raise NumericalGpuExecutionError("preserve existing GPU evidence; use an unmodified base")
    from compiler.offload.numerical_execution_workloads import execution_recipes, execution_registry

    target = Path(args.build_dir).resolve()
    target.mkdir(parents=True, exist_ok=False)
    commands = []
    base = deepcopy(profile)
    try:
        from compiler.offload.calibrate import cpu_identity

        if cpu_identity() != profile["hardware"].get("cpu_name"):
            raise NumericalGpuExecutionError("GPU enqueue CPU differs from base profile")
        declared = profile.get("numerical_execution", {}).get("identity", {}).get("cpu_affinity")
        supplied = getattr(args, "cpu_affinity", None)
        try:
            affinity = [int(cpu) for cpu in supplied.split(",")] if supplied else declared
        except ValueError as error:
            raise NumericalGpuExecutionError("GPU CPU affinity must be comma-separated integers") from error
        if (not isinstance(affinity, list) or not affinity or affinity != sorted(set(affinity))
                or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
                or not set(affinity) <= set(os.sched_getaffinity(0))
                or declared is not None and affinity != declared):
            raise NumericalGpuExecutionError("GPU requires available fixed CPU placement matching declared CPU evidence")
        nvcc = tool(args.nvcc, ("nvcc",))
        host = tool(args.cuda_host_cxx, ("g++-14", "g++"))
        versions = {key: run([binary, "--version"], target, target / (key + ".log"), timeout=30).strip()
                    for key, binary in (("nvcc_version", nvcc), ("host_cxx_version", host))}
        if any(profile["toolchain"].get(k) != v for k, v in versions.items()):
            raise NumericalGpuExecutionError("GPU toolchain differs from immutable base profile")
        source, registry_id, generator, renderer = current_gpu_identities(profile["precision_bits"])
        registry = execution_registry(profile["precision_bits"])
        if "numerical_execution" in profile and profile["numerical_execution"].get("registry") != list(registry):
            raise NumericalGpuExecutionError("CPU/GPU frozen recipe registries differ")
        emitted = gpu_sources(execution_recipes(profile["precision_bits"]), generator, source, registry_id)
        emitted["driver.cu"] = Path(__file__).with_name("numerical_gpu_execution_driver.cu").read_text()
        for name, text in emitted.items():
            (target / name).write_text(text)
        hashes = {name: _sha(target / name) for name in emitted}
        _write(target / "registry.json", {"registry": registry, "generator_id": generator,
               "source_generator_id": source, "registry_id": registry_id, "renderer_sources": renderer})
        build_paths = {"nvcc": nvcc, "host": host, "artifacts": str(target),
                       "runtime": str(Path(__file__).parents[1] / "runtime")}
        planned_commands = gpu_build_commands(build_paths, args.arch)
        binary = target / "numerical-gpu-execution"
        for name, argv in zip(("workers", "driver"), planned_commands[:2], strict=True):
            commands.append(argv)
            run(argv, target, target / (name + "-build.log"), timeout=180)
        commands.append(planned_commands[2])
        run(commands[-1], target, target / "link.log", timeout=180)
        objects = {name: _sha(target / name) for name in ("workers.o", "driver.o")}
        binary_hash = _sha(binary)
        environment = {k: v for k, v in os.environ.items() if k not in {
            "FORT_RUNTIME_TRACE", "FORT_PHASE_TIMING", "CUDA_LAUNCH_BLOCKING", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"}
            and not k.startswith("FORT_SCOPE_TEST_")}
        environment["FORT_PHASE_TIMING"] = "0"
        argv = ["taskset", "-c", ",".join(map(str, affinity)), str(binary), str(args.device)]

        def phase(mode, timeout):
            if (current_gpu_identities(profile["precision_bits"]) != (source, registry_id, generator, renderer)
                    or any(_sha(target / name) != digest for name, digest in {**hashes, **objects}.items())
                    or _sha(binary) != binary_hash or profile != base
                    or cpu_identity() != profile["hardware"]["cpu_name"]):
                raise NumericalGpuExecutionError("GPU sources/objects/binary/base changed")
            rows = _parse(run([*argv, mode], target, target / (mode[2:] + ".jsonl"), timeout=timeout, env=environment))
            if (current_gpu_identities(profile["precision_bits"]) != (source, registry_id, generator, renderer)
                    or any(_sha(target / name) != digest for name, digest in {**hashes, **objects}.items())
                    or _sha(binary) != binary_hash or profile != base
                    or cpu_identity() != profile["hardware"]["cpu_name"]):
                raise NumericalGpuExecutionError("GPU sources/objects/binary/base changed during execution")
            return rows

        raw_identity, = phase("--identity", 30)
        identity = {**raw_identity, **versions, "renderer_sources": renderer, "emitted_sources": hashes,
                    "objects": objects, "binary_sha256": binary_hash, "build_commands": commands,
                    "numerical_contract": numerical_build_contract(), "build_paths": build_paths,
                    "architecture": args.arch, "cpu_name": cpu_identity()}
        if raw_identity.get("sampling_cpu_affinity") != affinity:
            raise NumericalGpuExecutionError("GPU enqueue CPU placement differs from producer")
        # Check hardware/build identity before the first payload/correctness run.
        from compiler.offload.numerical_gpu_execution import _identity

        _identity(profile, identity, list(registry))
        preflight = phase("--preflight", 180)
        if preflight[0] != raw_identity or len(preflight) != 1 + PREFLIGHT_RECORDS:
            raise NumericalGpuExecutionError("GPU complete preflight identity/inventory required")
        validate_gpu_execution_preflight(profile, identity, preflight[1:], list(registry))
        _write(target / "preflight-receipt.json", {"identity": identity, "preflight": preflight[1:],
               "sampling_started": False, "automatic_placement_authorized": False})
        if args.preflight_only:
            return deepcopy(base)
        rows = phase("--measure", 1800)
        if rows[:len(preflight)] != preflight:
            raise NumericalGpuExecutionError("GPU identity/preflight changed before sampling")
        raw_path = target / "gpu-raw-samples.jsonl"
        if raw_path.stat().st_size > 8 * 1024 * 1024:
            raise NumericalGpuExecutionError("GPU raw batch byte budget exhausted")
        raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        if raw_rows != rows[len(preflight):]:
            raise NumericalGpuExecutionError("GPU raw batches conflict with final observations")
        calibration = {"application_profiled": False, "automatic_placement_authorized": False,
            "base_profile_identity": _hash(base), "compatibility_identity": compatibility_identity(base),
            "base_profile": getattr(args, "base_profile_reference", None),
            "sampling_cpu_affinity": affinity,
            "run_commands": [[*argv, mode] for mode in ("--identity", "--preflight", "--measure")],
            "environment": {k: environment.get(k) for k in ENVIRONMENT_KEYS},
            "raw_samples": {"path": str(raw_path), "sha256": _sha(raw_path), "bytes": raw_path.stat().st_size},
            "artifacts": str(target), "cpu_evidence_revalidated": False}
        result = profile_from_gpu_execution_measurements(base, [identity, *rows[1:]], calibration)
        _write(target / "evidence.json", result[SECTION])
        return result
    except BaseException as error:
        _write(target / "failure.json", {"error": str(error), "error_type": type(error).__name__,
               "automatic_placement_authorized": False, "application_profiled": False, "commands": commands})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default=os.environ.get("CUDACXX"))
    parser.add_argument("--cuda-host-cxx")
    parser.add_argument("--arch", default="native")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu-affinity", help="fixed comma-separated enqueue CPUs; otherwise use the CPU evidence placement")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.build_dir.exists() or args.output.resolve() == args.profile.resolve() or args.device < 0:
        parser.error("new output/build paths and a nonnegative GPU ordinal required; preserve the base")
    try:
        from compiler.offload.calibrate import _run, _tool
        from compiler.offload.profile import load_profile

        digest = _sha(args.profile)
        args.base_profile_reference = {"path": str(args.profile.resolve()), "sha256": digest,
                                       "bytes": args.profile.stat().st_size}
        profile = load_profile(args.profile)
        result = calibrate_gpu_execution(profile, args, run=_run, tool=_tool)
        if _sha(args.profile) != digest:
            raise NumericalGpuExecutionError("GPU base profile changed during execution")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write(args.output, result)
        _write(args.build_dir / "base-profile-receipt.json", {**args.base_profile_reference,
               "preflight_only": args.preflight_only, "automatic_placement_authorized": False})
    except (OSError, ValueError) as error:
        parser.exit(1, "GPU numerical execution: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
