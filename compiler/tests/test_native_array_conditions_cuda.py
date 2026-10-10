"""A device-produced shared branch value is coherent before its original team."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.numerical_contract import require_explicit_cuda_environment, require_numerical_build_contract


def source_text(precision):
    def units(face):
        return (f"!$omp do\ndo j=0,0\n"
                f"b({face})=b({face})+sum(a({face}:{face}))\n"
                "enddo\n!$omp end do\n") * 48

    return f"""module uniform_array_owner
implicit none
integer::visits=0
contains
subroutine advance(a,gate,b,out,n)
real({precision}),intent(in)::a(-2:)
real({precision}),intent(inout)::gate(-2:),b(-2:),out(-2:)
integer,intent(in)::n
integer::i,j
visits=visits+1
do i=0,n-1
gate(i)=a(i)
b(i)=2*a(i)+real(i,{precision})
enddo
!$omp parallel private(j)
if(gate(0)>0) then
{units('-2')}else
{units('18')}endif
!$omp end parallel
do i=0,n-1
out(i)=b(i)+b(-2)+b(18)
enddo
end subroutine
end module
"""


@pytest.mark.cuda
@pytest.mark.parametrize("precision", [4, 8])
def test_device_branch_value_preserves_original_native_team_and_resident_interior(tmp_path, precision):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA toolchain required")
    require_explicit_cuda_environment()
    source = tmp_path / "owner.f90"
    source.write_text(source_text(precision))
    reference = tmp_path / "reference.f90"
    reference.write_text(source.read_text().replace("uniform_array_owner", "native_array_owner"))
    stable = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::" + name: stable for name in ("a", "gate", "b", "out")},
             "sources": {str(source): sha256(source.read_bytes()).hexdigest()}}
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts, indent=2) + "\n")
    output, build = tmp_path / "generated", tmp_path / "build"
    build.mkdir()
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    commands = []

    def run(arguments, *, trace=False, disabled=False):
        ordinal = len(commands)
        commands.append({"argv": arguments, "cwd": str(build), "trace": trace, "device_disabled": disabled})
        (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(arguments, cwd=build,
            env={**environment, **({"FORT_RUNTIME_TRACE": "1"} if trace else {}),
                 **({"CUDA_VISIBLE_DEVICES": ""} if disabled else {})},
            capture_output=True, text=True, timeout=180, check=False)
        (tmp_path / f"command-{ordinal:02}.stdout").write_text(result.stdout)
        (tmp_path / f"command-{ordinal:02}.stderr").write_text(result.stderr)
        (tmp_path / f"command-{ordinal:02}.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    report = json.loads(run([sys.executable, "-m", "compiler", "--form-scopes", "--input", str(source),
        "--kernel", "uniform_array_owner::advance", "--scope-facts", str(facts_path),
        "--scope-execution", "reached", "--gpu-policy", "sections", "--memory-model", "scoped",
        "--json", "--output-dir", str(output)]).stdout)
    assert report["supported"], report
    manifest = json.loads((output / "scope-manifest.json").read_text())
    assert manifest == report["scopes"]
    owner, = manifest["scopes"]
    assert len(owner["gpu_leaves"]) == 2
    assert not owner["boundaries"], owner["boundaries"]
    native_team, = [operation for segment in owner["planning_segments"]
                   for operation in segment["operations"]["native_operations"]
                   if operation["kind"] == "joined native OpenMP"]
    assert native_team["sections"]["available"], native_team["sections"]
    fflags = [fc, "-O3", "-fopenmp", "-ffp-contract=off", "-fcheck=all", "-J", str(build), "-I", str(build)]
    objects, compiled = [], set()
    for role in ("common_runtime", "shared_entry"):
        for descriptor in manifest["build_sources"]:
            if descriptor["role"] != role:
                continue
            path = output / descriptor["path"]
            assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][descriptor["path"]]
            target = build / f"artifact-{len(objects):02}.o"
            flags = fflags
            if descriptor["language"] == "cuda":
                contract = descriptor["numerical_contract"]
                require_numerical_build_contract(contract)
                flags = [nvcc, "-O3", "-std=c++17", "-ccbin", host, "-Xcompiler=-fopenmp",
                         "-arch=" + os.environ.get("FORT_TEST_CUDA_ARCH", "native"),
                         *contract["required_cuda_options"],
                         *("-Xcompiler=" + option for option in contract["required_host_options"])]
            run([*flags, "-c", str(path), "-o", str(target)])
            objects.append(str(target))
            compiled.add(descriptor["path"])
    replacement = output / manifest["sources"][str(source)]["replacement"]
    for path in (reference, replacement):
        target = build / f"source-{len(objects):02}.o"
        run([*fflags, "-c", str(path), "-o", str(target)])
        objects.append(str(target))
    assert compiled | {str(replacement.relative_to(output))} == {
        item["path"] for item in manifest["build_sources"]}
    bits = "int32" if precision == 4 else "int64"
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""program verify
use iso_fortran_env,only:{bits},error_unit
use native_array_owner,only:native_advance=>advance,native_visits=>visits
use uniform_array_owner,only:generated_advance=>advance,generated_visits=>visits
implicit none
integer,parameter::widths(4)=[17,0,5,17]
real({precision})::a(-2:20),gate(-2:20),b(-2:20),out(-2:20)
real({precision})::expected_gate(-2:20),expected_b(-2:20),expected_out(-2:20),input_reference(-2:20)
integer::step,i,n,sign_value,native_unit,candidate_unit
open(newunit=native_unit,file='native-fields.bin',access='stream',form='unformatted',status='replace')
open(newunit=candidate_unit,file='candidate-fields.bin',access='stream',form='unformatted',status='replace')
do step=1,4
n=widths(step)
sign_value=merge(1,-1,mod(step,2)==1)
do i=-2,20
a(i)=real(sign_value*(3*i+31+step),{precision})/16._{precision}
enddo
input_reference=a
! Its stale host value selects the opposite arm until the GPU result is published.
gate=-a
b=-113
out=-117
expected_gate=gate
expected_b=b
expected_out=out
native_visits=0
generated_visits=0
call native_advance(a,expected_gate,expected_b,expected_out,n)
write(error_unit,*) 'BEGIN_ARRAY_CONDITION_CALL',step,n
call generated_advance(a,gate,b,out,n)
write(error_unit,*) 'END_ARRAY_CONDITION_CALL',step
if(native_visits/=1.or.generated_visits/=1) error stop 'original source prefix replayed'
if(any(transfer(gate,[0_{bits}],size(gate))/=transfer(expected_gate,[0_{bits}],size(expected_gate)))) &
 error stop 'branch array or halo differs'
if(any(transfer(b,[0_{bits}],size(b))/=transfer(expected_b,[0_{bits}],size(expected_b)))) &
 error stop 'native branch, intermediate or halo differs'
if(any(transfer(out,[0_{bits}],size(out))/=transfer(expected_out,[0_{bits}],size(expected_out)))) &
 error stop 'complete output or halo differs'
if(any(transfer(a,[0_{bits}],size(a))/=transfer(input_reference,[0_{bits}],size(input_reference)))) &
 error stop 'read-only input changed'
write(native_unit)input_reference,expected_gate,expected_b,expected_out
write(candidate_unit)a,gate,b,out
enddo
close(native_unit)
close(candidate_unit)
print *,'FOUR_ARRAY_CONDITION_CALLS_FIELDS_BITWISE_OK'
end program
""")
    target = build / "driver.o"
    run([*fflags, "-c", str(driver), "-o", str(target)])
    executable = build / "verify"
    run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", *objects, str(target), "-lgfortran", "-o", str(executable)])
    observed = run([str(executable)], trace=True)
    assert "FOUR_ARRAY_CONDITION_CALLS_FIELDS_BITWISE_OK" in observed.stdout
    expected = (build / "native-fields.bin").read_bytes()
    assert len(expected) == 4 * 4 * 23 * precision
    assert (build / "candidate-fields.bin").read_bytes() == expected
    calls = observed.stderr.split("BEGIN_ARRAY_CONDITION_CALL")[1:]
    assert len(calls) == 4
    for ordinal, call in enumerate(calls):
        trace = call.split("END_ARRAY_CONDITION_CALL", 1)[0]
        n = (17, 0, 5, 17)[ordinal]
        assert trace.count("FORT_SCOPED launch ") == (2 if n else 0), observed.stderr
        if n:
            assert trace.count("FORT_SCOPED initialize ") == 1
            before_consumer = trace.split("FORT_SCOPED launch ")[1]
            assert re.search(r"FORT_SCOPED download .*bytes=" + str(precision) + r"(?:\s|$)", before_consumer), trace
            uploaded = sum(map(int, re.findall(r"FORT_SCOPED upload .*bytes=(\d+)", trace)))
            downloaded = sum(map(int, re.findall(r"FORT_SCOPED download .*bytes=(\d+)", trace)))
            assert uploaded == (n + 2) * precision, "interior or branch values reuploaded: " + trace
            assert downloaded == 3 * n * precision, "bulk branch publication: " + trace
        else:
            assert not re.search(r"FORT_SCOPED (?:up|down)load ", trace), trace
    fallback = run([str(executable)], trace=True, disabled=True)
    assert "FOUR_ARRAY_CONDITION_CALLS_FIELDS_BITWISE_OK" in fallback.stdout
    assert (build / "native-fields.bin").read_bytes() == expected
    assert (build / "candidate-fields.bin").read_bytes() == expected
    assert "FORT_SCOPED launch " not in fallback.stderr
