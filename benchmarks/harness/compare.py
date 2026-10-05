#!/usr/bin/env python3
"""Compare Loki/PSyclone transformations by performance and source-edit effort.

Adapters run in separately pinned Python environments. All native builds and
generated sources remain in a new output directory. Requested failures fail the
comparison; GPU execution requires a positive in-device OpenACC probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .common import (
    ATOL,
    compare_values,
    grid,
    nonnegative,
    positive,
    timing_values,
    values,
)
from .drivers import GPU_PROBE, correctness_driver, timing_driver
from .paths import CASES, ROOT, SOURCES


@dataclass(frozen=True)
class Variant:
    name: str
    source: Path
    backend: str = "gnu"
    implementation: Path | None = None
    kernels_per_call: int | None = None

    @property
    def accelerator(self) -> bool:
        return self.backend in {"openacc", "cuda"}

    @property
    def supports_resident(self) -> bool:
        return self.backend == "openacc" or self.implementation is not None


def tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ValueError(f"Required tool unavailable: {name}")
    return path


def nvhpc_default() -> str:
    return shutil.which("nvfortran") or next(
        (
            str(p)
            for p in sorted(
                Path("/opt/nvidia/hpc_sdk").glob("*/**/compilers/bin/nvfortran"),
                reverse=True,
            )
        ),
        "nvfortran",
    )


def gpu_trace(stderr: str) -> dict[str, int]:
    """Require device execution from each implementation, not just the platform."""
    kernels = len(re.findall(r"^launch CUDA kernel\b", stderr, re.M))
    if not kernels:
        raise ValueError("OpenACC implementation emitted no GPU kernel execution trace")
    return {
        "kernel_launches": kernels,
        "upload_bytes": sum(int(size) for size in re.findall(r"^upload CUDA data .*\bbytes=(\d+)", stderr, re.M)),
        "download_bytes": sum(int(size) for size in re.findall(r"^download CUDA data .*\bbytes=(\d+)", stderr, re.M)),
    }


def cuda_trace(stderr: str) -> dict[str, int]:
    """Read successful local-runtime operations, never infer GPU work from availability."""
    kernels = len(re.findall(r"^FORT_RUNTIME kernel\b", stderr, re.M))
    if not kernels:
        raise ValueError("Local CUDA implementation emitted no GPU kernel execution trace")
    return {
        "kernel_launches": kernels,
        "allocations": len(re.findall(r"^FORT_RUNTIME alloc\b", stderr, re.M)),
        "frees": len(re.findall(r"^FORT_RUNTIME free\b", stderr, re.M)),
        "upload_bytes": sum(int(size) for size in re.findall(r"^FORT_RUNTIME upload bytes=(\d+)", stderr, re.M)),
        "download_bytes": sum(int(size) for size in re.findall(r"^FORT_RUNTIME download bytes=(\d+)", stderr, re.M)),
    }


class Comparison:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output = args.output.resolve()
        self.commands = []
        self.objects = {}
        self.tools = {"fc": tool(args.fc), "cxx": tool(args.cxx)}
        if args.gpu != "off":
            self.tools.update(acc_fc=tool(args.acc_fc), nvcc=tool(args.nvcc))
            if args.cuda_host_cxx:
                self.tools["cuda_host_cxx"] = tool(args.cuda_host_cxx)
        self.report = {
            "status": "running",
            "objective": "good performance with minimal original-source modifications",
            "gpu_mode": args.gpu,
            "compute_capability": args.compute_capability,
            "source_changes": 0,
            "source_sha256": {
                c: hashlib.sha256((CASES / c / "Fortran" / SOURCES[c]).read_bytes()).hexdigest() for c in args.cases
            },
            "compiler_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((ROOT / "compiler").rglob("*"))
                if path.is_file()
                and path.suffix in {".py", ".hpp", ".cuh"}
                and not {"tests", "debugging"}.intersection(path.relative_to(ROOT / "compiler").parts)
            },
            "adapters": [],
            "correctness": [],
            "compilation": [],
            "timings": [],
            "policy": {
                "absolute_tolerance": ATOL,
                "native_optimization": "-O3; default FMA; no fast-math or LTO flags",
                "timing": "synchronous wall time; deterministic inputs; consumed checksum",
                "drop_in": "each call includes device data setup and output return",
                "resident_batch": "OpenACC data region or local session; timer includes setup, final retrieval and destruction",
                "iterations": args.iterations,
                "warmup": args.warmup,
                "rounds": args.rounds,
                "threads": 1,
            },
        }

    def save(self) -> None:
        (self.output / "report.json").write_text(json.dumps(self.report, indent=2) + "\n")
        (self.output / "commands.json").write_text(json.dumps(self.commands, indent=2) + "\n")

    def run(self, command, cwd: Path, *, trace: bool = False) -> str:
        command = list(map(str, command))
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "ACC_DEVICE_TYPE": "nvidia",
        }
        # Notifications are recorded only in validation runs, never timed runs.
        env.pop("NVCOMPILER_ACC_NOTIFY", None)
        env.pop("NVCOMPILER_ACC_TIME", None)
        env.pop("FORT_RUNTIME_TRACE", None)
        if trace:
            env["NVCOMPILER_ACC_NOTIFY"] = "3"
            env["FORT_RUNTIME_TRACE"] = "1"
        record = {"command": command, "cwd": str(cwd), "accelerator_trace": trace}
        self.commands.append(record)
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.args.timeout,
        )
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")
        return result.stdout

    def flags(self, variant: Variant, directory: Path, *, checking: bool) -> tuple[str, list[str]]:
        if variant.backend in {"nvfortran", "openacc"}:
            flags = ["-O3", "-cpp", "-module", str(directory), "-I", str(directory)]
            if variant.backend == "openacc":
                flags += [
                    "-acc=gpu",
                    f"-gpu=cc{self.args.compute_capability}",
                    "-Minfo=accel",
                ]
            return self.tools["acc_fc"], flags
        flags = [
            "-O3",
            "-cpp",
            "-ffree-line-length-none",
            "-J",
            str(directory),
            "-I",
            str(directory),
        ]
        if checking:
            flags.append("-fcheck=bounds")
        return self.tools["fc"], flags

    def build(
        self,
        case: str,
        variant: Variant,
        shape: tuple[int, int, int],
        *,
        timing=False,
        resident=False,
    ) -> Path:
        # Modules depend on the implementation, not on the driver's grid. Reuse
        # them across grids while keeping correctness/timing instrumentation apart.
        kind = "timing" if timing else "correctness"
        directory = self.output / case / "build" / variant.name / kind
        directory.mkdir(parents=True, exist_ok=True)
        fc, flags = self.flags(variant, directory, checking=not timing)
        key = (case, variant.name, kind)
        if key not in self.objects:
            objects = []
            if variant.implementation:
                obj = directory / "implementation.o"
                if variant.backend == "cuda":
                    command = [
                        self.tools["nvcc"],
                        "-O3",
                        "-std=c++17",
                        f"-arch=sm_{self.args.compute_capability}",
                    ]
                    if "cuda_host_cxx" in self.tools:
                        command += ["-ccbin", self.tools["cuda_host_cxx"]]
                else:
                    command = [self.tools["cxx"], "-O3", "-std=c++17"]
                self.run(
                    [
                        *command,
                        "-I",
                        variant.implementation.parent,
                        "-c",
                        variant.implementation,
                        "-o",
                        obj,
                    ],
                    directory,
                )
                objects.append(obj)
            obj = directory / "module.o"
            self.run([fc, *flags, "-c", variant.source, "-o", obj], directory)
            objects.append(obj)
            self.objects[key] = objects
        driver_dir = directory / ("x".join(map(str, shape)) + ("-resident" if resident else "-dropin"))
        driver_dir.mkdir()
        driver = driver_dir / "driver.f90"
        driver.write_text(
            timing_driver(
                case,
                shape,
                self.args.iterations,
                self.args.warmup,
                resident=resident,
                session=variant.implementation is not None,
            )
            if timing
            else correctness_driver(case, resident=resident, session=variant.implementation is not None)
        )
        definitions = [f"-DVAR_N{axis}={size}" for axis, size in zip("XYZ", shape, strict=True)]
        driver_obj = driver_dir / "driver.o"
        self.run([fc, *flags, *definitions, "-c", driver, "-o", driver_obj], directory)
        binary = driver_dir / "benchmark"
        objects = [*self.objects[key], driver_obj]
        if variant.backend == "cuda":
            command = [self.tools["nvcc"]]
            if "cuda_host_cxx" in self.tools:
                command += ["-ccbin", self.tools["cuda_host_cxx"]]
            self.run([*command, *objects, "-lgfortran", "-o", binary], directory)
        else:
            libraries = ["-lstdc++"] if variant.implementation else []
            self.run([fc, *flags, *objects, *libraries, "-o", binary], directory)
        return binary

    def prepare(self, case: str) -> list[Variant]:
        directory = self.output / case
        directory.mkdir()
        source = CASES / case / "Fortran" / SOURCES[case]
        variants = [Variant("Fortran-GNU", source)]
        for level in (1, 0):
            local = directory / f"local-opt{level}"
            self.run(
                [
                    sys.executable,
                    "-m",
                    "compiler",
                    "--input",
                    source,
                    "--kernel",
                    case,
                    "--output-dir",
                    local,
                    "--opt-level",
                    str(level),
                ],
                ROOT,
            )
            suffix = "" if level else "-unoptimized"
            variants.append(
                Variant(
                    "Local-C++" + suffix,
                    local / "generated_interface.f90",
                    implementation=local / "generated_cpp_impl.cpp",
                )
            )
            if self.args.gpu != "off":
                variants.append(
                    Variant(
                        "Local-CUDA" + suffix,
                        local / "generated_interface.f90",
                        "cuda",
                        local / "generated_code.cu",
                        1 if level else 4,
                    )
                )
        if self.args.gpu != "off":
            variants += [
                Variant("Fortran-NVHPC", source, "nvfortran"),
                Variant(
                    "Handwritten-OpenACC",
                    CASES / case / "Fortran-ACC" / SOURCES[case],
                    "openacc",
                ),
            ]
        for name in self.args.tools:
            python = tool(getattr(self.args, f"{name}_python"))
            for target in ["cpu", "openacc"] if self.args.gpu != "off" else ["cpu"]:
                for fused in [True, False] if self.args.include_unfused and target == "openacc" else [True]:
                    label = (
                        name.capitalize() + ("-CPU" if target == "cpu" else "-OpenACC") + ("" if fused else "-unfused")
                    )
                    output = directory / f"{label}.f90"
                    command = [
                        python,
                        "-m",
                        "benchmarks.harness.transform",
                        "--tool",
                        name,
                        "--input",
                        source,
                        "--case",
                        case,
                        "--output",
                        output,
                        "--target",
                        target,
                    ]
                    if not fused:
                        command.append("--no-fuse")
                    self.run(command, ROOT)
                    manifest = json.loads(output.with_suffix(".json").read_text())
                    self.report["adapters"].append({"case": case, "variant": label, "manifest": manifest})
                    variants.append(Variant(label, output, "openacc" if target == "openacc" else "gnu"))
                    if target == "cpu" and self.args.gpu != "off":
                        variants.append(Variant(label + "-NVHPC", output, "nvfortran"))
        return variants

    def validate(self, case: str, variants: list[Variant]) -> None:
        for shape in self.args.grids:
            reference = None
            for variant in variants:
                for resident in [False, True] if variant.supports_resident and self.args.resident else [False]:
                    if variant.accelerator and self.args.gpu == "compile":
                        if shape != self.args.grids[0]:
                            continue
                        binary = self.build(case, variant, shape, resident=resident)
                        self.report["compilation"].append(
                            {
                                "case": case,
                                "variant": variant.name,
                                "resident": resident,
                                "binary": str(binary),
                                "executed": False,
                            }
                        )
                        continue
                    binary = self.build(case, variant, shape, resident=resident)
                    actual = values(
                        self.run([binary], binary.parent, trace=variant.accelerator),
                        math.prod(shape),
                    )
                    trace = None
                    if variant.backend == "openacc":
                        trace = gpu_trace(self.commands[-1]["stderr"])
                    elif variant.backend == "cuda":
                        trace = cuda_trace(self.commands[-1]["stderr"])
                        calls = 2 if resident else 1
                        expected = {
                            "kernel_launches": calls * variant.kernels_per_call,
                            "allocations": 4,
                            "frees": 4,
                            "upload_bytes": 3 * 8 * math.prod(size + 2 for size in shape),
                            "download_bytes": 8 * math.prod(size + 2 for size in shape),
                        }
                        if trace != expected:
                            raise ValueError(
                                f"Unexpected local CUDA operations: {case}/{variant.name}: {trace} != {expected}"
                            )
                    if reference is None:
                        reference = actual
                    difference = compare_values(reference, actual)
                    self.report["correctness"].append(
                        {
                            "case": case,
                            "variant": variant.name,
                            "grid": shape,
                            "resident": resident,
                            "gpu_trace": trace,
                            "max_absolute_error": difference,
                            "values": len(actual),
                            "status": "passed",
                        }
                    )
                    print(
                        f"PASS {case} {shape} {variant.name}{' resident' if resident else ''}: {difference:.3e}",
                        flush=True,
                    )
            self.save()

    def measure(self, case: str, variants: list[Variant]) -> None:
        for shape in self.args.timing_grids:
            reference_checksum = None
            for variant in variants:
                if variant.accelerator and self.args.gpu != "run":
                    continue
                for resident in [False, True] if variant.supports_resident and self.args.resident else [False]:
                    binary = self.build(case, variant, shape, timing=True, resident=resident)
                    samples, checksums = [], []
                    for _ in range(self.args.rounds):
                        elapsed, checksum = timing_values(self.run([binary], binary.parent))
                        if reference_checksum is None:
                            reference_checksum = checksum
                        if abs(checksum - reference_checksum) > ATOL * math.prod(shape):
                            raise ValueError(f"Timing checksum differs from original Fortran: {case}/{variant.name}")
                        samples.append(elapsed)
                        checksums.append(checksum)
                    record = {
                        "case": case,
                        "variant": variant.name,
                        "grid": shape,
                        "mode": "resident-batch" if resident else "drop-in",
                        "total_ms": samples,
                        "checksums": checksums,
                        "mean_ms_per_call": statistics.mean(samples) / self.args.iterations,
                        "stdev_ms_per_call": (statistics.stdev(samples) if len(samples) > 1 else 0)
                        / self.args.iterations,
                    }
                    self.report["timings"].append(record)
                    print(
                        f"TIME {case} {shape} {variant.name} {record['mode']}: {record['mean_ms_per_call']:.6f} ms/call",
                        flush=True,
                    )
                    self.save()

    def execute(self) -> None:
        self.report["toolchain"] = {
            name: {"path": path, "version": self.run([path, "--version"], ROOT).strip()}
            for name, path in self.tools.items()
        }
        if self.args.gpu != "off":
            probe = self.output / "gpu_probe.f90"
            probe.write_text(GPU_PROBE)
            binary = self.output / "gpu_probe"
            self.run(
                [
                    self.tools["acc_fc"],
                    "-O3",
                    "-acc=gpu",
                    f"-gpu=cc{self.args.compute_capability}",
                    probe,
                    "-o",
                    binary,
                ],
                self.output,
            )
            if self.args.gpu == "run":
                result = self.run([binary], self.output, trace=True)
                if "OPENACC_GPU_CONFIRMED" not in result:
                    raise ValueError("OpenACC device execution was not confirmed")
                self.report["device_probe"] = result.strip()
        self.save()
        for case in self.args.cases:
            variants = self.prepare(case)
            self.validate(case, variants)
            self.measure(case, variants)
        self.report["status"] = "passed"
        self.save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tools",
        nargs="*",
        choices=["loki", "psyclone"],
        default=["loki", "psyclone"],
        help="External adapters to compare; pass --tools without names for local implementations only.",
    )
    parser.add_argument("--loki-python", default="python")
    parser.add_argument("--psyclone-python", default="python")
    parser.add_argument("--cases", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument(
        "--grids",
        nargs="+",
        type=grid,
        default=[(5, 4, 3), (1, 1, 1), (16, 16, 16), (259, 7, 5)],
    )
    parser.add_argument("--timing-grids", nargs="*", type=grid, default=[])
    parser.add_argument("--gpu", choices=["off", "compile", "run"], default="off")
    parser.add_argument(
        "--resident",
        action="store_true",
        help="also test OpenACC data regions and local sessions keeping arrays resident across a timed batch",
    )
    parser.add_argument(
        "--include-unfused",
        action="store_true",
        help="also measure straightforward four-pass OpenACC transformations",
    )
    parser.add_argument("--fc", default="gfortran")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--acc-fc", default=nvhpc_default())
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuda-host-cxx", default="g++-14")
    parser.add_argument("--compute-capability", default="86")
    parser.add_argument("--iterations", type=positive, default=50)
    parser.add_argument("--warmup", type=nonnegative, default=5)
    parser.add_argument("--rounds", type=positive, default=5)
    parser.add_argument("--timeout", type=positive, default=180)
    args = parser.parse_args()
    for field in ("cases", "tools", "grids", "timing_grids"):
        setattr(args, field, list(dict.fromkeys(getattr(args, field))))
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    comparison = None
    try:
        comparison = Comparison(args)
        comparison.execute()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        if comparison:
            comparison.report.update(status="failed", error=str(error))
            comparison.save()
        else:
            (args.output / "report.json").write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1
    print(f"Report: {args.output.resolve() / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
