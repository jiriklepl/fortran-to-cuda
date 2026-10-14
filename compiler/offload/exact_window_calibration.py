"""Independent complete-window calibration for the bounded three-dimensional fixture.

This command builds actual compiler output, executes separate native/resident
processes in rotated order, checks every float64 field, then recompiles AUTO
with the measured evidence and verifies its decision. It never times or replays
a running application's state. Artifacts and rejected observations are retained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from hashlib import sha256
from pathlib import Path

from compiler.driver.options import CompilerOptions
from compiler.numerical_contract import require_explicit_cuda_environment
from compiler.offload.calibrate import cpu_identity
from compiler.offload.collective_calibration import normalize_fortran_options
from compiler.offload.config import OffloadConfig
from compiler.offload.exact_window import ENVIRONMENT, HOLDOUTS, PROTOCOL, SAMPLES, build_contract, evidence_decision
from compiler.scopes.source import ScopeBuilder


def fixture_source():
    arithmetic = "\n".join("v=v*1.000001_8+0.000001_8" for _ in range(192))
    return (
        """module exact_fixture
implicit none
contains
subroutine advance(a,b,d,out,n,audit)
real(8),contiguous,intent(in)::a(0:,0:,0:)
real(8),contiguous,intent(inout)::b(0:,0:,0:),d(0:,0:,0:),out(0:,0:,0:)
integer,intent(in)::n
real(8),intent(out)::audit
integer::i,j,k
real(8)::v
do k=1,n
do j=1,n
do i=1,n
v=a(i,j,k)
"""
        + arithmetic
        + """
b(i,j,k)=v
enddo
enddo
enddo
audit=sum(a(:,1,1))
b(2,2,2)=b(2,2,2)+audit*0.000001_8
do k=2,n-1
do j=2,n-1
do i=2,n-1
d(i,j,k)=b(i-1,j,k)+b(i+1,j,k)+b(i,j-1,k)+b(i,j+1,k)+b(i,j,k-1)+b(i,j,k+1)-6._8*b(i,j,k)+0.25_8*a(i,j,k)
enddo
enddo
enddo
do k=2,n-1
do j=2,n-1
do i=2,n-1
out(i,j,k)=out(i,j,k)+0.125_8*d(i,j,k)
enddo
enddo
enddo
end subroutine
end module
"""
    )


DRIVER = """program measure
use exact_fixture, only:advance
use iso_fortran_env,only:int64,compiler_version,compiler_options
use omp_lib
implicit none
real(8),allocatable::a(:,:,:),b(:,:,:),d(:,:,:),out(:,:,:)
real(8)::audit
integer::n,seed,i,j,k,unit
integer(int64)::start,finish,rate
character(32)::argument
call get_command_argument(1,argument)
read(argument,*)n
call get_command_argument(2,argument)
read(argument,*)seed
allocate(a(0:n+1,0:n+1,0:n+1),b(0:n+1,0:n+1,0:n+1),d(0:n+1,0:n+1,0:n+1),out(0:n+1,0:n+1,0:n+1))
do k=0,n+1
do j=0,n+1
do i=0,n+1
a(i,j,k)=real(mod(i+3*j+7*k+seed,113)-56,8)*0.015625_8
enddo
enddo
enddo
b=-3._8
d=-5._8
out=-7._8
call system_clock(start,rate)
call advance(a,b,d,out,n,audit)
call system_clock(finish)
print '(a,es24.16)', 'WINDOW_SECONDS ',real(finish-start,8)/real(rate,8)
print '(a,a)', 'FORTRAN_VERSION ',compiler_version()
print '(a,a)', 'FORTRAN_OPTIONS ',compiler_options()
print '(a,3(i0,1x))', 'CPU_TEAM ',omp_get_max_threads(),merge(1,0,omp_get_dynamic()),omp_get_proc_bind()
print '(a,25(i0,1x))', 'NATIVE_KEY ',size(a,1),lbound(a,1),size(a,2),lbound(a,2),size(a,3),lbound(a,3), &
size(b,1),lbound(b,1),size(b,2),lbound(b,2),size(b,3),lbound(b,3), &
size(d,1),lbound(d,1),size(d,2),lbound(d,2),size(d,3),lbound(d,3), &
size(out,1),lbound(out,1),size(out,2),lbound(out,2),size(out,3),lbound(out,3),n
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
write(unit)a,b,d,out,audit
close(unit)
end program
"""


def emit(source, request):
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": {str(source): sha256(source.read_bytes()).hexdigest()},
        "captures": {
            "argument::" + name: {
                "storage": "stable",
                "initialized": "whole",
                "escapes": False,
                "allocation_changes": False,
            }
            for name in ("a", "b", "d", "out")
        },
    }
    outputs, report = ScopeBuilder(
        [source],
        "exact_fixture::advance",
        facts=facts,
        options=CompilerOptions(),
        config=OffloadConfig(policy="auto", scope_execution="reached", exact_window=request),
    ).run()
    scopes = [scope for scope in report["scopes"] if "exact_window" in scope]
    if len(scopes) != 1:
        raise RuntimeError("complete-window fixture was not admitted: " + json.dumps(report.get("boundaries")))
    return outputs, report, scopes[0]["exact_window"]


def _run(command, *, cwd, environment):
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError("command failed: " + repr(command) + "\n" + result.stdout + result.stderr)
    return result


def build(directory, outputs, report, environment, *, nvcc, cxx, fortran, architecture):
    directory.mkdir()
    for name, text in outputs.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    commands, objects = [], []
    exact = next(scope["exact_window"] for scope in report["scopes"] if "exact_window" in scope)
    required = build_contract(architecture)
    if exact["contract"]["build_contract"] != required:
        raise RuntimeError("consumer semantic build options differ from calibrated contract")
    flags = [fortran, *required["fortran_options"]]
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, item in enumerate(report["build_sources"]):
            if item["role"] != role:
                continue
            source = directory / item["path"]
            target = directory / ("object-" + str(index) + ".o")
            compiler = (
                [
                    nvcc,
                    *required["cuda_options"],
                    "-ccbin",
                    cxx,
                    *(
                        ["-DFORT_EXACT_WINDOW_BUILD_" + exact["contract_sha256"] + "=1"]
                        if item["path"].startswith("exact/")
                        else []
                    ),
                ]
                if item["language"] == "cuda"
                else flags
            )
            command = (
                [*compiler, "-I", str(directory), "-J" + str(directory)]
                if item["language"] == "fortran"
                else [*compiler, "-I", str(directory)]
            )
            command += ["-c", str(source), "-o", str(target)]
            commands.append(command)
            _run(command, cwd=directory, environment=environment)
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver.write_text(DRIVER)
    driver_object = directory / "driver.o"
    command = [*flags, "-I", str(directory), "-J" + str(directory), "-c", str(driver), "-o", str(driver_object)]
    commands.append(command)
    _run(command, cwd=directory, environment=environment)
    objects.append(str(driver_object))
    binary = directory / "window"
    cuda_lib = str(Path(nvcc).resolve().parents[1] / "lib64")
    command = [*flags, *objects, "-L" + cuda_lib, "-Wl,-rpath," + cuda_lib, "-lcudart", "-lstdc++", "-o", str(binary)]
    commands.append(command)
    _run(command, cwd=directory, environment=environment)
    receipt = {
        "commands": commands,
        "binary": str(binary),
        "binary_sha256": sha256(binary.read_bytes()).hexdigest(),
        "objects": {str(path): sha256(Path(path).read_bytes()).hexdigest() for path in objects},
    }
    (directory / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return binary, receipt


def build_native(directory, source, environment, *, fortran, **_tools):
    """Compile the untouched original source, with no generated selector/CUDA."""
    directory.mkdir()
    driver = directory / "driver.f90"
    driver.write_text(DRIVER)
    binary = directory / "native"
    flags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-ffp-contract=off"]
    # Original and identity-reporting driver share the same compile invocation.
    # Linking is separate: GNU adds link-only -foffload-options to a driver
    # compiled during a link, which must not be stripped from semantic identity.
    command = [*flags, "-J" + str(directory), "-I", str(directory), "-c", str(source), str(driver)]
    _run(command, cwd=directory, environment=environment)
    objects = [directory / (source.stem + ".o"), directory / "driver.o"]
    link = [*flags, *map(str, objects), "-o", str(binary)]
    _run(link, cwd=directory, environment=environment)
    receipt = {
        "commands": [command, link],
        "binary": str(binary),
        "binary_sha256": sha256(binary.read_bytes()).hexdigest(),
        "objects": {str(path): sha256(path.read_bytes()).hexdigest() for path in objects},
        "original_source_sha256": sha256(source.read_bytes()).hexdigest(),
    }
    (directory / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return binary, receipt


def collect(binary, native, directory, environment, size, contract):
    observations = []
    for round_index in range(SAMPLES + HOLDOUTS):
        fields = {}
        for turn in range(2):
            mode = ("native", "resident")[(round_index + turn) % 2]
            selected_binary = native if mode == "native" else binary
            result = _run(
                [str(selected_binary), str(size), str(round_index + 1)],
                cwd=directory,
                environment={**environment, "FORT_EXACT_CALIBRATION_MODE": mode},
            )
            stem = f"{round_index:02d}-{mode}"
            (directory / (stem + ".stdout")).write_text(result.stdout)
            (directory / (stem + ".stderr")).write_text(result.stderr)
            metadata = [
                json.loads(line.removeprefix("FORT_EXACT_OBSERVATION "))
                for line in result.stderr.splitlines()
                if line.startswith("FORT_EXACT_OBSERVATION ")
            ]
            if len(metadata) != (0 if mode == "native" else 1):
                raise RuntimeError("actual selector identity was not recorded")
            match = re.search(r"WINDOW_SECONDS\s+(\S+)", result.stdout)
            if match is None:
                raise RuntimeError("complete invocation timing missing")
            payload = (directory / "fields.bin").read_bytes()
            fields[mode] = payload
            version = re.search(r"^FORTRAN_VERSION (.+)$", result.stdout, re.MULTILINE).group(1)
            options = re.search(r"^FORTRAN_OPTIONS (.+)$", result.stdout, re.MULTILINE).group(1)
            team = list(map(int, re.search(r"^CPU_TEAM (.+)$", result.stdout, re.MULTILINE).group(1).split()))
            native_key = list(map(int, re.search(r"^NATIVE_KEY (.+)$", result.stdout, re.MULTILINE).group(1).split()))
            row = {
                "round": round_index,
                "mode": mode,
                "role": "fit" if round_index < SAMPLES else "holdout",
                "seconds": float(match.group(1)),
                **(metadata[0] if metadata else {}),
                "fields_sha256": sha256(payload).hexdigest(),
                "observed_cpu_identity": {
                    "cpu_name": cpu_identity(),
                    "fortran_version": version,
                    "fortran_options": normalize_fortran_options(options),
                    "team": team,
                    "affinity": sorted(os.sched_getaffinity(0)),
                    "environment": {name: environment.get(name) for name in ENVIRONMENT},
                },
                "observed_key": native_key,
            }
            observations.append(row)
        if fields["native"] != fields["resident"]:
            (directory / "mismatch-native.bin").write_bytes(fields["native"])
            (directory / "mismatch-resident.bin").write_bytes(fields["resident"])
            raise RuntimeError("full float64 native/resident fields differ, including halos")
        resident = next(row for row in observations[-2:] if row["mode"] == "resident")
        expected_cpu = {
            "cpu_name": resident["identity"]["cpu_name"],
            "fortran_version": resident["identity"]["fortran_version"],
            "fortran_options": normalize_fortran_options(resident["identity"]["fortran_options"]),
            "team": [
                resident["identity"]["threads"],
                resident["identity"]["dynamic"],
                resident["identity"]["proc_bind"],
            ],
            "affinity": resident["identity"]["affinity"],
            "environment": resident["identity"]["environment"],
        }
        for row in observations[-2:]:
            if row["observed_cpu_identity"] != expected_cpu or row["observed_key"] != resident["key"]:
                raise RuntimeError("native/resident actual descriptor/CPU/Fortran identity differs")
            row.update(key=resident["key"], identity=resident["identity"])
            row["agreement"] = True
        (directory / "observations.json").write_text(json.dumps(observations, indent=2) + "\n")
    evidence = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "contract": contract,
        "key": observations[0]["key"],
        "identity": observations[0]["identity"],
        "observations": observations,
    }
    (directory / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def diagnostic(binary, directory, environment, size, expected_hash, *, label, resident):
    """Trace separately; never use instrumented times as calibration costs."""
    result = _run(
        [str(binary), str(size), "9"],
        cwd=directory,
        environment={**environment, "FORT_RUNTIME_TRACE": "1", "FORT_EXACT_CALIBRATION_MODE": "resident"},
    )
    (directory / (label + ".stdout")).write_text(result.stdout)
    (directory / (label + ".stderr")).write_text(result.stderr)
    field_hash = sha256((directory / "fields.bin").read_bytes()).hexdigest()
    if field_hash != expected_hash:
        raise RuntimeError(label + " complete float64 fields differ")
    events = re.findall(r"^FORT_SCOPED (upload|download|launch) .*?bytes=(\d+)", result.stderr, re.MULTILINE)
    observed = {
        (kind + "_count" if kind == "launch" else kind + "_trace_events"): sum(event == kind for event, _ in events)
        for kind in ("upload", "download", "launch")
    }
    observed.update(
        {
            kind + "_bytes": sum(int(count) for event, count in events if event == kind)
            for kind in ("upload", "download")
        }
    )
    interior = (size - 2) ** 3
    predicted = {
        "upload_bytes": 8 * (size**3 + interior + 1),
        "download_bytes": 8 * (size**3 + 2 * interior),
        "launch_count": 3,
    }
    if not resident:
        predicted = dict.fromkeys(predicted, 0)
    ledger = {
        "predicted": predicted,
        "observed": observed,
        "fields_sha256": field_hash,
        "trace_sha256": sha256(result.stderr.encode()).hexdigest(),
        "coverage": "all entry, native correction, publication and cleanup transfers",
        "counts": "upload/download trace events count CopyPlan/Box records, not individual CUDA API copies",
        "prediction": "a producer input; one corrected b cell reupload; interior out input; b,d,out publication",
    }
    (directory / (label + "-ledger.json")).write_text(json.dumps(ledger, indent=2) + "\n")
    if any(observed[key] != value for key, value in predicted.items()):
        raise RuntimeError(label + " observed complete transfer/launch ledger differs from prediction")
    return ledger


def validate_production(automatic, native, directory, environment, size, evidence, decision):
    """Measure the emitted AUTO path independently; never revise its decision."""
    if "FORT_RUNTIME_TRACE" in environment or "FORT_EXACT_CALIBRATION_MODE" in environment:
        raise RuntimeError("production validation must be untraced without a calibration override")
    identity = evidence["identity"]
    expected_cpu = {
        "cpu_name": identity["cpu_name"],
        "fortran_version": identity["fortran_version"],
        "fortran_options": normalize_fortran_options(identity["fortran_options"]),
        "team": [identity["threads"], identity["dynamic"], identity["proc_bind"]],
        "affinity": identity["affinity"],
        "environment": identity["environment"],
    }
    rows = []
    values = {"native": [], "automatic": []}
    for round_index in range(SAMPLES + HOLDOUTS):
        expected_hash = next(
            row["fields_sha256"]
            for row in evidence["observations"]
            if row["mode"] == "native" and row["round"] == round_index
        )
        for turn in range(2):
            mode = ("native", "automatic")[(round_index + turn) % 2]
            binary = native if mode == "native" else automatic
            result = _run([str(binary), str(size), str(round_index + 1)], cwd=directory, environment=environment)
            stem = f"production-{round_index:02d}-{mode}"
            (directory / (stem + ".stdout")).write_text(result.stdout)
            (directory / (stem + ".stderr")).write_text(result.stderr)
            field_hash = sha256((directory / "fields.bin").read_bytes()).hexdigest()
            version = re.search(r"^FORTRAN_VERSION (.+)$", result.stdout, re.MULTILINE).group(1)
            options = re.search(r"^FORTRAN_OPTIONS (.+)$", result.stdout, re.MULTILINE).group(1)
            team = list(map(int, re.search(r"^CPU_TEAM (.+)$", result.stdout, re.MULTILINE).group(1).split()))
            key = list(map(int, re.search(r"^NATIVE_KEY (.+)$", result.stdout, re.MULTILINE).group(1).split()))
            cpu = {
                "cpu_name": cpu_identity(),
                "fortran_version": version,
                "fortran_options": normalize_fortran_options(options),
                "team": team,
                "affinity": sorted(os.sched_getaffinity(0)),
                "environment": {name: environment.get(name) for name in ENVIRONMENT},
            }
            elapsed = float(re.search(r"WINDOW_SECONDS\s+(\S+)", result.stdout).group(1))
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("production complete-window time invalid")
            if cpu != expected_cpu or key != evidence["key"] or field_hash != expected_hash:
                raise RuntimeError("production native/AUTO identity or complete float64 fields differ")
            if "FORT_EXACT_OBSERVATION" in result.stderr:
                raise RuntimeError("production binary unexpectedly contains calibration override")
            values[mode].append(elapsed)
            rows.append(
                {
                    "round": round_index,
                    "mode": mode,
                    "role": "fit" if round_index < SAMPLES else "holdout",
                    "seconds": elapsed,
                    "fields_sha256": field_hash,
                    "observed_cpu_identity": cpu,
                    "observed_key": key,
                    "agreement": True,
                }
            )
    profitable = max(values["automatic"]) < 0.95 * min(values["native"])
    report = {
        "protocol": "complete-original-window-cold-production-validation-v1",
        "observations": rows,
        "native_seconds": statistics.median(values["native"][:SAMPLES]),
        "automatic_seconds": statistics.median(values["automatic"][:SAMPLES]),
        "automatic_profitable": profitable,
        "predicted_selection": decision["selection"],
        "model_miss": decision["selection"] == "resident" and not profitable,
        "authority": "independent validation only; never used to revise calibration or placement",
    }
    (directory / "production-observations.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", default="16,192")
    parser.add_argument("--architecture", default="sm_86")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    require_explicit_cuda_environment()
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "FALSE"}
    environment.pop("FORT_RUNTIME_TRACE", None)
    environment.pop("FORT_EXACT_CALIBRATION_MODE", None)
    source = output / "original.f90"
    source.write_text(fixture_source())
    outputs, report, record = emit(source, {"calibration": True, "build_contract": build_contract(args.architecture)})
    tools = {
        "nvcc": shutil.which("nvcc"),
        "cxx": shutil.which("g++-14") or shutil.which("g++"),
        "fortran": shutil.which("gfortran-15") or shutil.which("gfortran"),
        "architecture": args.architecture,
    }
    if not all(tools.values()):
        raise RuntimeError("CUDA/C++/Fortran toolchain required")
    binary, receipt = build(output / "calibration", outputs, report, environment, **tools)
    native, native_receipt = build_native(output / "native", source, environment, **tools)
    results = []
    for size in map(int, args.sizes.split(",")):
        directory = output / ("size-" + str(size))
        directory.mkdir()
        evidence = collect(binary, native, directory, environment, size, record["contract"])
        decision = evidence_decision(evidence, record["contract"])
        auto_outputs, auto_report, auto_record = emit(source, {"evidence": evidence})
        if auto_record["contract"] != record["contract"]:
            raise RuntimeError("AUTO and calibration source/worker contracts differ")
        auto, auto_receipt = build(directory / "automatic", auto_outputs, auto_report, environment, **tools)
        expected = next(
            row["fields_sha256"] for row in evidence["observations"] if row["round"] == 8 and row["mode"] == "native"
        )
        resident_trace = diagnostic(
            binary, directory, environment, size, expected, label="resident-diagnostic", resident=True
        )
        automatic_trace = diagnostic(
            auto,
            directory,
            environment,
            size,
            expected,
            label="automatic",
            resident=decision["selection"] == "resident",
        )
        rejected_trace = diagnostic(
            auto,
            directory,
            {**environment, "OMP_NUM_THREADS": "2"},
            size,
            expected,
            label="automatic-thread-mismatch",
            resident=False,
        )
        production = validate_production(auto, native, directory, environment, size, evidence, decision)
        results.append(
            {
                "size": size,
                "decision": decision,
                "production_validation": production,
                "automatic_execution_verified": True,
                "resident_diagnostic": resident_trace,
                "automatic_diagnostic": automatic_trace,
                "thread_mismatch_diagnostic": rejected_trace,
                "calibration_build": receipt,
                "native_build": native_receipt,
                "automatic_build": auto_receipt,
            }
        )
        (output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "size": size,
                    "selection": decision["selection"],
                    "native_seconds": decision.get("native_seconds"),
                    "resident_seconds": decision.get("resident_seconds"),
                    "production_native_seconds": production["native_seconds"],
                    "production_automatic_seconds": production["automatic_seconds"],
                    "production_model_miss": production["model_miss"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
