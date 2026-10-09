"""Compare ordinary-call GPU policies using public CLI artifacts and native reference.

Run as ``python -m benchmarks.harness.strategies --calibration-profile FILE``.
Validation dumps, unprofiled timing samples, and Nsight evidence are separate
executions. This harness does not import compiler IR or tune application policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import struct
import subprocess
import sys
import time
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import grid, nonnegative, positive

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "benchmarks" / "strategy_cases"
CASES = {
    "pointwise": ("tidal_fields", "tidal_update"),
    "boundary": ("painted_faces", "paint_faces"),
    "stencil": ("ripple_fields", "ripple_filter"),
    "compute": ("polynomial_fields", "polynomial_steps"),
    "prepared": ("prepared_fields", "prepared_update"),
}
MODES = ("native", "current", "sections", "auto", "chunked", "hybrid")
POLICIES = {
    mode: "always" if mode == "current" else mode for mode in MODES if mode != "native"
}
TRACE_KEYS = (
    "FORT_RUNTIME_TRACE",
    "FORT_OFFLOAD_TRACE",
    "FORT_PROFILE",
    "NVCOMPILER_ACC_TIME",
    "NVCOMPILER_ACC_NOTIFY",
    "CUDA_LAUNCH_BLOCKING",
)


def driver_source(module: str, entry: str, *, query=None, native_module=None) -> str:
    """Same initialization/caller ABI for GNU reference and every generated mode."""
    imports = f"  use {module}, only: kernel => {entry}"
    call = "    call kernel(source,destination,nx,ny,nz)"
    if query is not None:
        imports += f", select_gpu => {query}\n  use {native_module}, only: native_kernel => {entry}"
        call = """    if (select_gpu(source,destination,nx,ny,nz)) then
      call kernel(source,destination,nx,ny,nz)
    else
      call native_kernel(source,destination,nx,ny,nz)
    end if"""
    return f"""program strategy_driver
  use iso_fortran_env, only: real64, int64
  use omp_lib, only: omp_get_wtime
{imports}
  implicit none
  real(real64), allocatable :: source(:,:,:), destination(:,:,:)
  integer :: nx,ny,nz,iterations,warmup,i,j,k,iteration,unit
  real(real64) :: started,elapsed
  character(1024) :: arg,dump
  call get_command_argument(1,arg); read(arg,*) nx
  call get_command_argument(2,arg); read(arg,*) ny
  call get_command_argument(3,arg); read(arg,*) nz
  call get_command_argument(4,arg); read(arg,*) iterations
  call get_command_argument(5,arg); read(arg,*) warmup
  call get_command_argument(6,dump)
  if (min(nx,ny,nz,iterations)<1.or.warmup<0) error stop 1
  allocate(source(0:nx+1,-1:ny,2:nz+3),destination(0:nx+1,-1:ny,2:nz+3))
  do k=2,nz+3
    do j=-1,ny
      do i=0,nx+1
        source(i,j,k)=0.125d0+real(mod(3*i+5*j+7*k+31,97),real64)/128.0d0
        destination(i,j,k)=-11.0d0-real(i+2*j+3*k,real64)/1024.0d0
      end do
    end do
  end do
  do iteration=1,warmup
{call}
  end do
  elapsed=0.0d0
  do iteration=1,iterations
    started=omp_get_wtime()
{call}
    elapsed=elapsed+omp_get_wtime()-started
    if (trim(dump)/='-'.and.iteration<iterations) source=source+0.03125d0
  end do
  write(*,'(a,es24.16)') 'call_seconds ',elapsed
  write(*,'(a,es24.16)') 'checksum ',sum(destination)
  write(*,'(a,i0)') 'calls ',iterations
  if (trim(dump)/='-') then
    open(newunit=unit,file=trim(dump),access='stream',form='unformatted',status='replace',action='write')
    write(unit) 'STRAT001'
    write(unit) shape(source,kind=int64)
    write(unit) source,destination
    close(unit)
  end if
end program
"""


def parse_timing(output: str) -> dict:
    fields = {}
    for line in output.splitlines():
        match = re.fullmatch(r"(call_seconds|checksum|calls)\s+(\S+)\s*", line.strip())
        if not match:
            continue
        name, raw = match.groups()
        if name in fields:
            raise ValueError(f"duplicate timing field {name}")
        fields[name] = (
            int(raw)
            if name == "calls"
            else float(raw.replace("D", "E").replace("d", "e"))
        )
    if set(fields) != {"call_seconds", "checksum", "calls"}:
        raise ValueError("missing driver timing/checksum fields")
    if (
        fields["calls"] < 1
        or fields["call_seconds"] <= 0
        or not all(math.isfinite(value) for value in fields.values())
    ):
        raise ValueError("invalid driver timing/checksum fields")
    fields["seconds_per_call"] = fields["call_seconds"] / fields["calls"]
    return fields


def parse_trace(output: str) -> dict:
    runtime = Counter()
    decisions = []
    intervals = []
    for line in output.splitlines():
        if "FORT_RUNTIME " in line:
            event = line.split("FORT_RUNTIME ", 1)[1].split()
            if event:
                runtime[event[0] + "_count"] += 1
                fields = dict(
                    token.split("=", 1) for token in event[1:] if "=" in token
                )
                if "bytes" in fields:
                    runtime[event[0] + "_bytes"] += int(fields["bytes"])
        if "FORT_OFFLOAD " in line:
            fields = dict(
                token.split("=", 1)
                for token in line.split("FORT_OFFLOAD ", 1)[1].split()
                if "=" in token
            )
            for key in ("gpu_units", "cpu_units", "gpu_points", "cpu_points"):
                if key in fields:
                    fields[key] = int(fields[key])
            decisions.append(fields)
        if "FORT_OFFLOAD_INTERVAL " in line:
            fields = dict(
                token.split("=", 1)
                for token in line.split("FORT_OFFLOAD_INTERVAL ", 1)[1].split()
                if "=" in token
            )
            for key in ("begin", "end", "work", "launches", "upload_bytes", "download_bytes", "volume_valid"):
                if key in fields:
                    fields[key] = int(fields[key])
            if "estimate_seconds" in fields:
                fields["estimate_seconds"] = float(fields["estimate_seconds"])
            intervals.append(fields)
    totals = {}
    for decision in decisions:
        kind = decision.get("unit_kind", "unspecified")
        item = totals.setdefault(kind, {"gpu_units": 0, "cpu_units": 0})
        for key in item:
            item[key] += decision.get(key, 0)
    return {
        "runtime": dict(runtime),
        "decisions": decisions,
        "intervals": intervals,
        "unit_totals_by_kind": totals,
    }


def read_dump(path: Path):
    with path.open("rb") as stream:
        header = stream.read(32)
    if len(header) != 32 or header[:8] != b"STRAT001":
        raise ValueError("invalid validation dump header")
    shape = struct.unpack("=3q", header[8:])
    if min(shape) < 3:
        raise ValueError("invalid validation dump dimensions")
    count = math.prod(shape)
    if path.stat().st_size != 32 + 2 * count * 8:
        raise ValueError("validation dump length does not match complete arrays")
    data = np.fromfile(path, dtype=np.float64, offset=32)
    return shape, tuple(
        data[index * count : (index + 1) * count].reshape(shape, order="F")
        for index in range(2)
    )


def compare_dumps(reference: Path, actual: Path, *, atol=1e-10, rtol=1e-10) -> dict:
    shape, expected = read_dump(reference)
    other_shape, observed = read_dump(actual)
    if shape != other_shape:
        raise ValueError("validation dump shapes differ")
    fields = {}
    for name, baseline, result in zip(
        ("source", "destination"), expected, observed, strict=True
    ):
        if not np.isfinite(result).all() or not np.isfinite(baseline).all():
            raise ValueError("non-finite validation values")
        error = np.abs(result - baseline)
        accepted = error <= atol + rtol * np.abs(baseline)
        halo_max = max(
            float(error.take(face, axis=axis).max())
            for axis in range(3)
            for face in (0, -1)
        )
        fields[name] = {
            "values": int(result.size),
            "max_abs_error": float(error.max()),
            "halo_max_abs_error": halo_max,
        }
        if not accepted.all():
            raise ValueError(
                f"{name} full-array validation failed: max error {error.max():.17g}"
            )
    return {
        "passed": True,
        "shape": list(shape),
        "atol": atol,
        "rtol": rtol,
        "fields": fields,
        "calls": 2,
        "host_inputs_modified_between_calls": True,
        "all_array_elements_compared": True,
    }


def _overlap_seconds(first, second) -> float:
    """Union intersection duration, never a sum of overlapping event pairs."""
    events = []
    for kind, ranges in enumerate((first, second)):
        for start, end in ranges:
            if end > start:
                events.extend(((start, kind, 1), (end, kind, -1)))
    active = [0, 0]
    previous = None
    duration = 0
    for position, kind, delta in sorted(events):
        if previous is not None and all(active):
            duration += position - previous
        active[kind] += delta
        previous = position
    return duration / 1e9


def _stream_overlap_seconds(kernels, copies) -> float:
    events = []
    for kind, rows in enumerate((kernels, copies)):
        for start, end, stream in rows:
            if end > start:
                events.extend(((start, kind, stream, 1), (end, kind, stream, -1)))
    active = [Counter(), Counter()]
    previous = None
    duration = 0
    for position, kind, stream, delta in sorted(events):
        concurrent = any(left != right for left in active[0] for right in active[1])
        if previous is not None and concurrent:
            duration += position - previous
        active[kind][stream] += delta
        if not active[kind][stream]:
            del active[kind][stream]
        previous = position
    return duration / 1e9


def parse_profile(path: Path) -> dict:
    """Read actual CUDA activities plus explicitly named CPU-compute NVTX ranges."""
    with closing(
        sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as database:
        tables = {
            row[0]
            for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        kernel_table = next(
            (
                name
                for name in (
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
                )
                if name in tables
            ),
            None,
        )
        kernels = (
            list(database.execute(f'SELECT start,"end",streamId FROM {kernel_table}'))
            if kernel_table
            else []
        )
        copies = (
            list(
                database.execute(
                    'SELECT start,"end",streamId,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY'
                )
            )
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables
            else []
        )
        ranges = {}
        if "NVTX_EVENTS" in tables:
            columns = {
                row[1] for row in database.execute("PRAGMA table_info(NVTX_EVENTS)")
            }
            strings = (
                dict(database.execute("SELECT id,value FROM StringIds"))
                if "StringIds" in tables
                else {}
            )
            text = '"text"' if "text" in columns else "NULL"
            text_id = "textId" if "textId" in columns else "NULL"
            for start, end, label, identifier in database.execute(
                f'SELECT start,"end",{text},{text_id} FROM NVTX_EVENTS'
            ):
                label = label or strings.get(identifier, "")
                if (
                    label.startswith("FORT ")
                    and start is not None
                    and end is not None
                    and end > start
                ):
                    ranges.setdefault(label, []).append((start, end))
        cpu_ranges = ranges.get("FORT hybrid CPU window", [])
        gpu_ranges = [(start, end) for start, end, _ in kernels]
        copy_ranges = [(start, end) for start, end, *_ in copies]
        return {
            "sqlite": str(path),
            "activity_tables": sorted(tables),
            "kernel_count": len(kernels),
            "kernel_seconds_sum": sum(end - start for start, end, _ in kernels) / 1e9,
            "copy_count": len(copies),
            "copy_seconds_sum": sum(end - start for start, end, *_ in copies) / 1e9,
            "h2d_bytes": sum(size for _, _, _, kind, size in copies if kind == 1),
            "d2h_bytes": sum(size for _, _, _, kind, size in copies if kind == 2),
            "kernel_streams": sorted({stream for _, _, stream in kernels}),
            "copy_streams": sorted({stream for _, _, stream, *_ in copies}),
            "cpu_compute_ranges": len(cpu_ranges),
            "cpu_gpu_overlap_seconds": _overlap_seconds(cpu_ranges, gpu_ranges),
            "cpu_copy_overlap_seconds": _overlap_seconds(cpu_ranges, copy_ranges),
            "kernel_copy_overlap_seconds": _stream_overlap_seconds(
                kernels, [row[:3] for row in copies]
            ),
            "host_ranges": {
                label: {
                    "count": len(intervals),
                    "seconds_sum": sum(end - start for start, end in intervals) / 1e9,
                    "seconds_min": min(end - start for start, end in intervals) / 1e9,
                    "seconds_median": statistics.median(end - start for start, end in intervals) / 1e9,
                    "seconds_max": max(end - start for start, end in intervals) / 1e9,
                }
                for label, intervals in sorted(ranges.items())
            },
            "measurement": "separate profiled execution; activity sums are not elapsed wall time",
        }


def comparison_gates(timings: list[dict], profiles: list[dict]) -> list[dict]:
    """Use complete process time and real overlap, not requested policy names."""
    grouped = {}
    for item in timings:
        grouped.setdefault((item["case"], tuple(item["grid"])), {})[item["mode"]] = item
    activity = {
        (item["case"], tuple(item["grid"]), item["mode"]): item for item in profiles
    }
    gates = []
    for (case, shape), modes in grouped.items():
        if "native" not in modes:
            continue
        native = modes["native"]["process_seconds_median"]
        record = {"case": case, "grid": list(shape), "metric": "complete process wall time"}
        if "auto" in modes:
            ratio = modes["auto"]["process_seconds_median"] / native
            record["auto"] = {"ratio_to_native": ratio, "within_5_percent": ratio <= 1.05}
        if "hybrid" in modes and "sections" in modes:
            alternatives = {
                name: modes[name]["process_seconds_median"]
                for name in ("native", "current", "sections") if name in modes
            }
            best = min(alternatives, key=alternatives.get)
            alternative = alternatives[best]
            ratio = modes["hybrid"]["process_seconds_median"] / alternative
            profile = activity.get((case, shape, "hybrid"), {})
            overlap = profile.get("cpu_gpu_overlap_seconds", 0) > 0 and profile.get(
                "kernel_copy_overlap_seconds", 0
            ) > 0
            mixed = any(
                decision.get("gpu_units", 0) > 0 and decision.get("cpu_units", 0) > 0
                for decision in profile.get("trace", {}).get("decisions", [])
            )
            record["hybrid"] = {
                "best_native_or_synchronous_mode": best,
                "ratio_to_best_native_or_synchronous": ratio,
                "at_least_10_percent_faster": ratio <= 0.9,
                "mixed_decision_recorded": mixed,
                "cpu_gpu_and_kernel_copy_overlap_observed": overlap,
                "qualifies": ratio <= 0.9 and mixed and overlap,
            }
        gates.append(record)
    return gates


class Strategies:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.commands = []
        profile = args.calibration_profile.resolve(strict=True)
        self.profile = self.output / "calibration.json"
        if profile != self.profile:
            shutil.copy2(profile, self.profile)
        self.calibration = json.loads(self.profile.read_text())
        if args.architecture is None:
            args.architecture = str(self.calibration.get("hardware", {}).get("compute_capability", "")).replace(".", "")
        if not re.fullmatch(r"[1-9][0-9]{1,2}[af]?", args.architecture):
            raise ValueError("supply an explicit architecture or a profile with its compute capability")
        if (
            self.calibration["precision_bits"] != 64
            or self.calibration["cpu_threads"] != args.host_threads
        ):
            raise ValueError(
                "calibration precision/thread budget differs from benchmark"
            )
        self.binaries = {}
        self.report = {
            "status": "preparing",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_profile": str(self.profile),
            "calibration_sha256": self.digest(self.profile),
            "compiler_root": str(args.compiler_root.resolve()),
            "compiler_files_sha256": {
                str(path.relative_to(args.compiler_root)): self.digest(path)
                for path in sorted((args.compiler_root / "compiler").rglob("*"))
                if path.is_file()
                and path.suffix in {".py", ".hpp", ".cuh"}
                and "tests" not in path.parts
            },
            "configuration": {
                "cases": args.cases,
                "modes": args.modes,
                "grids": args.grids,
                "host_threads": args.host_threads,
                "precision_bits": 64,
                "architecture": args.architecture,
                "rounds": args.rounds,
                "warmup_rounds": args.warmup_rounds,
                "warmup_calls_per_process": args.warmup_calls,
                "calls_per_sample": args.iterations,
                "application_tuning": False,
                "timings_profiled": False,
                "profiling_enabled": not args.skip_profile,
                "borderline_rounds": max(7, args.rounds),
                "borderline_gate_margin": 0.03,
            },
            "generation": [],
            "validation": [],
            "timings": [],
            "profiles": [],
        }
        self.save()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save(self):
        (self.output / "report.json").write_text(
            json.dumps(self.report, indent=2) + "\n"
        )
        (self.output / "commands.json").write_text(
            json.dumps(self.commands, indent=2) + "\n"
        )

    def run(self, command, cwd, *, trace=False, label="command"):
        environment = dict(os.environ)
        for name in TRACE_KEYS:
            environment.pop(name, None)
        environment.update(
            OMP_NUM_THREADS=str(self.args.host_threads),
            OMP_DYNAMIC="FALSE",
            OMP_MAX_ACTIVE_LEVELS="1",
        )
        if trace:
            environment.update(FORT_RUNTIME_TRACE="1", FORT_OFFLOAD_TRACE="1")
        command = list(map(str, command))
        record = {
            "argv": command,
            "cwd": str(cwd),
            "trace": trace,
            "environment": {
                key: environment[key]
                for key in (
                    "OMP_NUM_THREADS",
                    "OMP_DYNAMIC",
                    "OMP_MAX_ACTIVE_LEVELS",
                    *TRACE_KEYS,
                )
                if key in environment
            },
        }
        self.commands.append(record)
        logs = self.output / "logs"
        logs.mkdir(exist_ok=True)
        stem = logs / f"{len(self.commands):04d}-{label}"
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=self.args.timeout,
            check=False,
        )
        wall = time.perf_counter() - started
        stdout, stderr = (
            stem.with_suffix(".stdout.log"),
            stem.with_suffix(".stderr.log"),
        )
        stdout.write_text(result.stdout)
        stderr.write_text(result.stderr)
        record.update(
            returncode=result.returncode,
            process_seconds=wall,
            stdout=str(stdout),
            stderr=str(stderr),
        )
        self.save()
        if result.returncode:
            raise RuntimeError(f"{label} failed ({result.returncode}); see {stderr}")
        return result.stdout, result.stderr, wall

    def build(self, case, mode):
        if (case, mode) in self.binaries:
            return self.binaries[case, mode]
        directory = self.output / case / "build" / mode
        directory.mkdir(parents=True, exist_ok=True)
        original = FIXTURES / f"{case}.f90"
        source = directory / "input.f90"
        shutil.copy2(original, source)
        module, entry = CASES[case]
        objects = []
        module_sources = []
        query = native_module = None
        if mode != "native":
            command = [
                self.args.python,
                "-m",
                "compiler",
                "--input",
                source,
                "--kernel",
                entry,
                "--output-dir",
                directory,
                "--json",
                "--fallback",
                "error",
                "--gpu-policy",
                POLICIES[mode],
                "--calibration-profile",
                self.profile,
                "--host-threads",
                self.args.host_threads,
            ]
            stdout, _, _ = self.run(
                command, self.args.compiler_root, label=f"generate-{case}-{mode}"
            )
            generated = json.loads(stdout)
            if generated.get("supported") is not True:
                raise ValueError(f"compiler rejected {case}/{mode}")
            (directory / "compiler-report.json").write_text(
                json.dumps(generated, indent=2) + "\n"
            )
            self.report["generation"].append(
                {
                    "case": case,
                    "mode": mode,
                    "response": generated,
                    "source_sha256": self.digest(source),
                }
            )
            query = generated.get("offload", {}).get("native_fallback_query")
            if query is not None:
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", query):
                    raise ValueError("invalid public native-fallback query name")
                native_module = module + "_native"
                fallback = directory / "native_fallback.f90"
                fallback.write_text(original.read_text().replace(
                    f"module {module}\n", f"module {native_module}\n", 1
                ))
                module_sources.append(("native-fallback", fallback))
            implementation = directory / "generated_code.cu"
            obj = directory / "gpu.o"
            self.run(
                [
                    self.args.nvcc,
                    "-O3",
                    "-std=c++17",
                    "-arch=sm_" + self.args.architecture,
                    "-ccbin",
                    self.args.cuda_host_cxx,
                    "-Xcompiler=-fopenmp",
                    "-I",
                    directory,
                    "-c",
                    implementation,
                    "-o",
                    obj,
                ],
                directory,
                label=f"nvcc-{case}-{mode}",
            )
            objects.append(obj)
            source = directory / "generated_interface.f90"
        driver = directory / "driver.f90"
        driver.write_text(driver_source(module, entry, query=query, native_module=native_module))
        for name, path in (*module_sources, ("module", source), ("driver", driver)):
            obj = directory / f"{name}.o"
            self.run(
                [
                    self.args.fc,
                    "-O3",
                    "-fopenmp",
                    "-ffree-line-length-none",
                    "-J",
                    directory,
                    "-I",
                    directory,
                    "-c",
                    path,
                    "-o",
                    obj,
                ],
                directory,
                label=f"gfortran-{case}-{mode}-{name}",
            )
            objects.append(obj)
        binary = directory / "benchmark"
        if mode == "native":
            command = [self.args.fc, "-fopenmp", *objects, "-o", binary]
        else:
            command = [
                self.args.nvcc,
                "-ccbin",
                self.args.cuda_host_cxx,
                "-Xcompiler=-fopenmp",
                *objects,
                "-lgfortran",
                "-o",
                binary,
            ]
        self.run(command, directory, label=f"link-{case}-{mode}")
        self.binaries[case, mode] = binary
        return binary

    def binary_command(self, case, mode, shape, *, dump=None, profile=False):
        return [
            self.build(case, mode),
            *shape,
            2 if dump is not None else 3 if profile else self.args.iterations,
            0 if dump is not None else 1 if profile else self.args.warmup_calls,
            str(dump) if dump else "-",
        ]

    def validate(self, case, shape):
        directory = self.output / case / "validation" / "x".join(map(str, shape))
        directory.mkdir(parents=True, exist_ok=True)
        reference = directory / "native.bin"
        for mode in dict.fromkeys(("native", *self.args.modes)):
            dump = directory / f"{mode}.bin"
            stdout, stderr, _ = self.run(
                self.binary_command(case, mode, shape, dump=dump),
                directory,
                trace=mode != "native",
                label=f"validate-{case}-{mode}",
            )
            parse_timing(stdout)
            result = compare_dumps(reference, dump)
            result.update(
                case=case,
                mode=mode,
                grid=list(shape),
                dump=str(dump),
                trace=parse_trace(stdout + stderr),
            )
            if (
                mode == "current"
                and result["trace"]["runtime"].get("kernel_count", 0) == 0
            ):
                raise ValueError(
                    "current CUDA validation produced no runtime kernel trace"
                )
            self.report["validation"].append(result)
            self.save()

    def time_mode(self, case, mode, shape, *, rounds=None):
        directory = self.output / case / "timing" / "x".join(map(str, shape)) / mode
        directory.mkdir(parents=True, exist_ok=True)
        command = self.binary_command(case, mode, shape)
        previous = next(
            (
                item for item in self.report["timings"]
                if item["case"] == case and item["mode"] == mode and item["grid"] == list(shape)
            ),
            None,
        )
        if previous is None:
            for _ in range(self.args.warmup_rounds):
                self.run(command, directory, label=f"warmup-{case}-{mode}")
        samples = [] if previous is None else list(previous["samples"])
        for _ in range(len(samples), rounds or self.args.rounds):
            stdout, stderr, wall = self.run(
                command, directory, label=f"timing-{case}-{mode}"
            )
            if "FORT_RUNTIME " in stdout + stderr or "FORT_OFFLOAD " in stdout + stderr:
                raise ValueError("tracing leaked into a timing sample")
            sample = parse_timing(stdout)
            sample["process_seconds"] = wall
            samples.append(sample)
        checksums = [sample["checksum"] for sample in samples]
        if max(checksums) != min(checksums):
            raise ValueError("nondeterministic checksum between timing samples")
        record = {
            "case": case,
            "mode": mode,
            "grid": list(shape),
            "rounds": len(samples),
            "samples": samples,
            "call_seconds_median": statistics.median(
                sample["call_seconds"] for sample in samples
            ),
            "seconds_per_call_median": statistics.median(
                sample["seconds_per_call"] for sample in samples
            ),
            "process_seconds_median": statistics.median(
                sample["process_seconds"] for sample in samples
            ),
        }
        if previous is None:
            self.report["timings"].append(record)
        else:
            previous.update(record)
        self.save()

    def extend_borderline(self, case, shape):
        modes = {
            item["mode"]: item for item in self.report["timings"]
            if item["case"] == case and item["grid"] == list(shape)
        }
        if "native" not in modes or self.args.rounds >= 7:
            return
        targets = set()
        native = modes["native"]["process_seconds_median"]
        if "auto" in modes:
            ratio = modes["auto"]["process_seconds_median"] / native
            if abs(ratio - 1.05) <= 0.03:
                targets.update(("native", "auto"))
        if "hybrid" in modes and "sections" in modes:
            baseline = min(
                modes[name]["process_seconds_median"]
                for name in ("native", "current", "sections") if name in modes
            )
            ratio = modes["hybrid"]["process_seconds_median"] / baseline
            if abs(ratio - 0.9) <= 0.03:
                targets.update(("native", "current", "sections", "hybrid"))
        for mode in self.args.modes:
            if mode in targets:
                self.time_mode(case, mode, shape, rounds=7)

    def profile_mode(self, case, mode, shape):
        if mode == "native" or self.args.skip_profile:
            return
        directory = self.output / case / "profiles" / "x".join(map(str, shape)) / mode
        directory.mkdir(parents=True, exist_ok=True)
        prefix = directory / "activity"
        command = [
            self.args.nsys,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--export=sqlite",
            "--force-overwrite=true",
            "--output=" + str(prefix),
            *self.binary_command(case, mode, shape, profile=True),
        ]
        stdout, stderr, wall = self.run(
            command, directory, trace=True, label=f"profile-{case}-{mode}"
        )
        parse_timing(stdout)
        metrics = parse_profile(prefix.with_suffix(".sqlite"))
        if mode == "current" and not metrics["kernel_count"]:
            raise ValueError("Nsight recorded no actual kernels for current CUDA")
        metrics.update(
            case=case,
            mode=mode,
            grid=list(shape),
            profiler_process_seconds=wall,
            trace=parse_trace(stdout + stderr),
        )
        self.report["profiles"].append(metrics)
        self.save()

    def markdown(self):
        gates = self.report.get("gates", [])
        auto = [item["auto"] for item in gates if "auto" in item]
        hybrid = [item["hybrid"] for item in gates if "hybrid" in item]
        lines = [
            "# Generic GPU strategy comparison",
            "",
            "Each mode uses real64 data and the same fixed host-thread budget. Calibration is supplied offline; no application tuning occurs in this run.",
            "",
            "Validation compares both complete arrays, including halos, after two calls with changed host inputs. Timing samples and Nsight profiles are separate processes.",
            "",
            "Call time includes synchronous transfer and completion inside the entry. Process time also includes startup, allocation, initialization, warmup, checksum, and teardown. Activity sums are not a wall-time decomposition.",
            "",
            "| Case | Grid | Mode | Call median, ms | Process median, ms | Call speedup vs native |",
            "|---|---|---|---:|---:|---:|",
        ]
        conclusions = []
        if auto:
            conclusions.append(f"Automatic selection meets the 5% complete-process regression gate in {sum(item['within_5_percent'] for item in auto)}/{len(auto)} generic configurations.")
        if hybrid:
            conclusions.append(f"Hybrid qualifies for promotion in {sum(item['qualifies'] for item in hybrid)}/{len(hybrid)} configurations after requiring a mixed decision, actual CPU/GPU and kernel/copy overlap, and at least 10% improvement over the best CPU or synchronous alternative.")
        if conclusions:
            lines[2:2] = [" ".join(conclusions) + " The separate ELMM results determine the application gate.", ""]
        if (self.output / "complete-wall-relative.svg").exists():
            position = 4 if conclusions else 2
            lines[position:position] = ["![Complete process time relative to native](complete-wall-relative.svg)", ""]
        native = {
            (item["case"], tuple(item["grid"])): item["seconds_per_call_median"]
            for item in self.report["timings"]
            if item["mode"] == "native"
        }
        for item in self.report["timings"]:
            baseline = native.get((item["case"], tuple(item["grid"])))
            speedup = (
                "n/a"
                if baseline is None
                else f"{baseline / item['seconds_per_call_median']:.3f}"
            )
            lines.append(
                f"| {item['case']} | {'×'.join(map(str, item['grid']))} | {item['mode']} | "
                f"{item['seconds_per_call_median'] * 1000:.6f} | {item['process_seconds_median'] * 1000:.3f} | {speedup} |"
            )
        lines.extend(
            [
                "",
                "GPU policies may select native execution. The requested mode alone is not evidence of GPU or hybrid computation; inspect recorded decisions and actual CUDA/NVTX activity in [report.json](report.json).",
                "An apparent speedup when the policy selected native execution is not GPU acceleration; short process timings also include startup variability.",
                "",
                "| Case | Grid | Mode | Kernels | H2D bytes | D2H bytes | Kernel/copy overlap, ms | CPU/GPU overlap, ms |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in self.report["profiles"]:
            lines.append(
                f"| {item['case']} | {'×'.join(map(str, item['grid']))} | {item['mode']} | {item['kernel_count']} | "
                f"{item['h2d_bytes']} | {item['d2h_bytes']} | {item['kernel_copy_overlap_seconds'] * 1000:.6f} | "
                f"{item['cpu_gpu_overlap_seconds'] * 1000:.6f} |"
            )
        lines.extend(
            [
                "",
                "Calibration and raw commands/responses are retained in [calibration.json](calibration.json) and [commands.json](commands.json). Profiles include one warmup call and three measured calls; these counts differ from two-call validation and timing samples.",
                "",
                "Host range counts and total/minimum/median/maximum durations are in each profile's `host_ranges`. The pack range includes host packing and H2D enqueue overhead; unpack excludes waiting for completion. Decision ranges cover public-query evaluation and model selection, including first-use CUDA setup when needed, but exclude OpenMP barriers and some dispatched-entry metadata construction. These traced durations are diagnostic evidence, not additions to unprofiled wall time. An absent range means that activity was not measured.",
                "",
                "Promotion gates use complete process wall time: automatic selection must stay within 5% of native; hybrid requires 10% improvement over the best native, current whole-array CUDA, or section-transfer alternative, a mixed decision, and observed CPU/GPU and kernel/copy overlap. Comparisons within three percentage points of a gate extend to seven samples. Generic cases supplement the separate ELMM application gates.",
                "",
            ]
        )
        for item in self.report.get("gates", []):
            lines.append(f"- {item['case']} {'×'.join(map(str, item['grid']))}: " + "; ".join(
                f"{mode} {'passes' if gate[key] else 'does not pass'}"
                for mode, key in (("auto", "within_5_percent"), ("hybrid", "qualifies"))
                if (gate := item.get(mode)) is not None
            ) + ".")
        (self.output / "report.md").write_text("\n".join(lines))

    def execute(self):
        for case in self.args.cases:
            for mode in dict.fromkeys(("native", *self.args.modes)):
                self.build(case, mode)
        if not self.args.build_only:
            self.report["status"] = "running"
            for case in self.args.cases:
                for shape in self.args.grids:
                    self.validate(case, shape)
                    # Rotate mode order deterministically between sizes/cases.
                    offset = (
                        self.args.cases.index(case) + self.args.grids.index(shape)
                    ) % len(self.args.modes)
                    for mode in self.args.modes[offset:] + self.args.modes[:offset]:
                        self.time_mode(case, mode, shape)
                    self.extend_borderline(case, shape)
                    for mode in self.args.modes:
                        self.profile_mode(case, mode, shape)
        self.report["status"] = "compiled" if self.args.build_only else "passed"
        self.report["gates"] = comparison_gates(self.report["timings"], self.report["profiles"])
        self.save()
        self.markdown()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--calibration-profile", required=True, type=Path)
    result.add_argument("--compiler-root", type=Path, default=ROOT)
    result.add_argument("--python", default=sys.executable)
    result.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/results/gpu-strategies-20261006/generic",
    )
    result.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    result.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    result.add_argument(
        "--grids",
        nargs="+",
        type=grid,
        default=[(32, 24, 16), (96, 64, 48), (160, 128, 96)],
    )
    result.add_argument("--host-threads", type=positive, default=4)
    result.add_argument("--iterations", type=positive, default=10)
    result.add_argument(
        "--rounds",
        type=positive,
        default=3,
        help="timed process samples; use 7 for borderline comparisons",
    )
    result.add_argument("--warmup-rounds", type=nonnegative, default=1)
    result.add_argument("--warmup-calls", type=nonnegative, default=1)
    result.add_argument("--fc", default=shutil.which("gfortran-15") or "gfortran")
    result.add_argument("--nvcc", default=shutil.which("nvcc") or "nvcc")
    result.add_argument("--cuda-host-cxx", default=shutil.which("g++-14") or "g++")
    result.add_argument("--architecture", help="target compute capability; defaults to the explicit calibration profile")
    result.add_argument("--nsys", default=shutil.which("nsys") or "nsys")
    result.add_argument("--timeout", type=positive, default=600)
    result.add_argument("--build-only", action="store_true")
    result.add_argument("--skip-profile", action="store_true")
    return result


def main():
    comparison = Strategies(parser().parse_args())
    try:
        comparison.execute()
    except Exception as error:
        comparison.report.update(status="failed", error=str(error))
        comparison.save()
        raise
    print(comparison.output / "report.md")


if __name__ == "__main__":
    main()
