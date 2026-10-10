"""Native-only inputs and joined sections preserve a resident numerical prefix."""

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


def boundary_source(precision):
    def sections(sign):
        return f"""!$omp sections
!$omp section
do j=0,0
b(-2)=b(-2){sign}sum(a(-2:-2))+edge_data(-2)
enddo
!$omp section
do j=0,0
b(18)=b(18){sign}sum(a(18:18))+edge_data(1)
enddo
!$omp end sections nowait
"""

    return f"""module native_host_boundary_owner
implicit none
real({precision}),allocatable::edge_data(:)
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
if(n>0) then
!$omp parallel private(j)
if(gate(0)>0) then
{sections('+')}else
{sections('-')}endif
!$omp end parallel
endif
do i=0,n-1
out(i)=b(i)+b(-2)+b(18)
enddo
end subroutine
end module
"""


def boundary_driver(precision):
    bits = "int32" if precision == 4 else "int64"
    return f"""program verify
use iso_fortran_env,only:{bits},error_unit
use reference_boundary_owner,only:native_advance=>advance,native_visits=>visits,native_edges=>edge_data
use native_host_boundary_owner,only:generated_advance=>advance,generated_visits=>visits,edges=>edge_data
implicit none
integer,parameter::widths(4)=[17,0,5,17]
real({precision})::a(-2:20),gate(-2:20),b(-2:20),out(-2:20)
real({precision})::expected_gate(-2:20),expected_b(-2:20),expected_out(-2:20),input_reference(-2:20)
integer::step,i,n,lower,sign_value,native_unit,candidate_unit
open(newunit=native_unit,file='native-fields.bin',access='stream',form='unformatted',status='replace')
open(newunit=candidate_unit,file='candidate-fields.bin',access='stream',form='unformatted',status='replace')
do step=1,4
n=widths(step)
sign_value=merge(1,-1,step==1)
do i=-2,20
a(i)=real(sign_value*(3*i+31+step),{precision})/16._{precision}
enddo
if(allocated(edges)) deallocate(edges)
if(allocated(native_edges)) deallocate(native_edges)
if(n>0) then
lower=merge(-7,-3,step==3)
if(step==4) lower=-2
allocate(edges(lower:lower+22))
allocate(native_edges(lower:lower+22))
do i=lower,lower+22
edges(i)=real(2*i+step,{precision})/8._{precision}
enddo
if(step==4) then
edges=a
endif
native_edges=edges
endif
input_reference=a
gate=-a
b=-113
out=-117
expected_gate=gate
expected_b=b
expected_out=out
native_visits=0
generated_visits=0
call native_advance(a,expected_gate,expected_b,expected_out,n)
write(error_unit,*)'BEGIN_NATIVE_HOST_CALL',step,n
if(step==4) then
! The read-only actual aliases host-only module storage. Registration must
! publish the GPU prefix and continue once before the original native group.
call generated_advance(edges,gate,b,out,n)
else
call generated_advance(a,gate,b,out,n)
endif
write(error_unit,*)'END_NATIVE_HOST_CALL',step
if(native_visits/=1.or.generated_visits/=1) error stop 'source prefix replayed'
if(any(transfer(gate,[0_{bits}],size(gate))/=transfer(expected_gate,[0_{bits}],size(expected_gate)))) &
 error stop 'complete gate or halo differs'
if(any(transfer(b,[0_{bits}],size(b))/=transfer(expected_b,[0_{bits}],size(expected_b)))) &
 error stop 'complete intermediate or native faces differ'
if(any(transfer(out,[0_{bits}],size(out))/=transfer(expected_out,[0_{bits}],size(expected_out)))) &
 error stop 'complete consumer or halo differs'
if(any(transfer(a,[0_{bits}],size(a))/=transfer(input_reference,[0_{bits}],size(input_reference)))) &
 error stop 'read-only actual changed'
if(n>0) then
if(any(transfer(edges,[0_{bits}],size(edges))/=transfer(native_edges,[0_{bits}],size(native_edges)))) &
 error stop 'native-only input changed'
if(lbound(edges,1)/=lower.or.ubound(edges,1)/=lower+22) error stop 'original descriptor changed'
endif
write(native_unit)input_reference,expected_gate,expected_b,expected_out
write(candidate_unit)a,gate,b,out
if(n>0) then
write(native_unit)native_edges
write(candidate_unit)edges
endif
enddo
close(native_unit)
close(candidate_unit)
print *,'NATIVE_HOST_BOUNDARIES_COMPLETE_FIELDS_BITWISE_OK'
end program
"""


@pytest.mark.cuda
@pytest.mark.parametrize("precision", [4, 8])
def test_native_input_sections_and_alias_escape_preserve_complete_fields(tmp_path, precision):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA toolchain required")
    require_explicit_cuda_environment()
    checkout = tmp_path / "independent-compiler"
    shutil.copytree(Path(__file__).resolve().parents[2] / "compiler", checkout / "compiler",
        ignore=shutil.ignore_patterns("__pycache__", ".*cache", "CODE_MAP.md", "*PLAN.md"))
    source = tmp_path / "owner.f90"
    source.write_text(boundary_source(precision))
    reference = tmp_path / "reference.f90"
    reference.write_text(source.read_text().replace("native_host_boundary_owner", "reference_boundary_owner"))
    stable = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::" + name: stable for name in ("a", "gate", "b", "out")},
             "sources": {str(source): sha256(source.read_bytes()).hexdigest()}}
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts, indent=2) + "\n")
    output, build = tmp_path / "generated", tmp_path / "build"
    build.mkdir()
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "PYTHONPATH": str(checkout)}
    environment.pop("FORT_RUNTIME_TRACE", None)
    commands = []

    def run(arguments, *, trace=False, disabled=False):
        ordinal = len(commands)
        commands.append({"argv": arguments, "cwd": str(build), "trace": trace, "device_disabled": disabled})
        (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(arguments, cwd=build, env={**environment,
            **({"FORT_RUNTIME_TRACE": "1"} if trace else {}),
            **({"CUDA_VISIBLE_DEVICES": ""} if disabled else {})},
            capture_output=True, text=True, timeout=180, check=False)
        (tmp_path / f"command-{ordinal:02}.stdout").write_text(result.stdout)
        (tmp_path / f"command-{ordinal:02}.stderr").write_text(result.stderr)
        (tmp_path / f"command-{ordinal:02}.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    report = json.loads(run([sys.executable, "-m", "compiler", "--form-scopes", "--input", str(source),
        "--kernel", "native_host_boundary_owner::advance", "--scope-facts", str(facts_path),
        "--scope-execution", "reached", "--gpu-policy", "sections", "--memory-model", "scoped",
        "--json", "--output-dir", str(output)]).stdout)
    assert report["supported"], report
    manifest = json.loads((output / "scope-manifest.json").read_text())
    assert manifest == report["scopes"]
    owner, = manifest["scopes"]
    assert len(owner["gpu_leaves"]) == 2
    assert not owner["boundaries"], owner["boundaries"]
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
    assert compiled | {str(replacement.relative_to(output))} == {item["path"] for item in manifest["build_sources"]}
    driver = tmp_path / "driver.f90"
    driver.write_text(boundary_driver(precision))
    target = build / "driver.o"
    run([*fflags, "-c", str(driver), "-o", str(target)])
    executable = build / "verify"
    run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", *objects, str(target), "-lgfortran", "-o", str(executable)])
    observed = run([str(executable)], trace=True)
    assert "NATIVE_HOST_BOUNDARIES_COMPLETE_FIELDS_BITWISE_OK" in observed.stdout
    expected = (build / "native-fields.bin").read_bytes()
    assert len(expected) == (4 * 4 * 23 + 3 * 23) * precision
    assert (build / "candidate-fields.bin").read_bytes() == expected
    calls = observed.stderr.split("BEGIN_NATIVE_HOST_CALL")[1:]
    assert len(calls) == 4
    for ordinal, call in enumerate(calls):
        trace = call.split("END_NATIVE_HOST_CALL", 1)[0]
        n = (17, 0, 5, 17)[ordinal]
        expected_launches = 1 if ordinal == 3 else (2 if n else 0)
        assert trace.count("FORT_SCOPED launch ") == expected_launches, trace
        uploaded = sum(map(int, re.findall(r"FORT_SCOPED upload .*bytes=(\d+)", trace)))
        downloaded = sum(map(int, re.findall(r"FORT_SCOPED download .*bytes=(\d+)", trace)))
        if ordinal == 3:
            assert uploaded == n * precision, trace
            assert downloaded == 2 * n * precision, trace
        elif n:
            assert uploaded == (n + 2) * precision, "native-only storage or interior reuploaded: " + trace
            assert downloaded == 3 * n * precision, "bulk native-boundary publication: " + trace
        else:
            assert uploaded == downloaded == 0, trace
    untraced = run([str(executable)])
    assert "FORT_SCOPED " not in untraced.stderr
    assert (build / "candidate-fields.bin").read_bytes() == expected
    fallback = run([str(executable)], trace=True, disabled=True)
    assert "NATIVE_HOST_BOUNDARIES_COMPLETE_FIELDS_BITWISE_OK" in fallback.stdout
    assert (build / "native-fields.bin").read_bytes() == expected
    assert (build / "candidate-fields.bin").read_bytes() == expected
    assert not re.search(r"FORT_SCOPED (?:launch|upload|download) ", fallback.stderr)
