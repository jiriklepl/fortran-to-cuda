"""Large original native teams preserve a GPU prefix and exact boundary writes."""

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
    def units(face):
        return (f"!$omp do\ndo j=0,2\n"
                f"b({face},j)=b({face},j)+sum(a({face}:{face},j))\n"
                "enddo\n!$omp end do\n") * 48

    return f"""module independent_boundary_owner
implicit none
integer::visits=0
contains
subroutine step(a,b,out,n,m,flag,escape)
real({precision}),intent(in)::a(-2:,-1:)
real({precision}),intent(inout)::b(-2:,-1:),out(-2:,-1:)
integer,intent(in)::n,m
logical,intent(in)::flag,escape
integer::i,j
visits=visits+1
do j=0,m-1
do i=0,n-1
b(i,j)=2*a(i,j)+real(i+3*j,{precision})
enddo
enddo
!$omp parallel private(j)
if(flag) then
{units('-2')}else
{units('18')}endif
!$omp end parallel
if(escape) call opaque(b)
do j=0,m-1
do i=0,n-1
out(i,j)=b(i,j)+b(-2,j)+b(18,j)
enddo
enddo
end subroutine
end module
"""


@pytest.mark.cuda
@pytest.mark.parametrize("precision", [4, 8])
def test_deferred_native_team_preserves_complete_fields_and_gpu_prefix(tmp_path, precision):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA toolchain required")
    require_explicit_cuda_environment()
    source = tmp_path / "owner.f90"
    source.write_text(boundary_source(precision))
    reference = tmp_path / "reference.f90"
    reference.write_text(source.read_text().replace("independent_boundary_owner", "native_boundary_owner"))
    callback = tmp_path / "callback.f90"
    callback.write_text(f"""module native_boundary_audit
integer::opaque_visits=0
end module
subroutine opaque(x)
use native_boundary_audit,only:opaque_visits
real({precision})::x(*)
opaque_visits=opaque_visits+1
x(1)=x(1)+41
end subroutine
""")
    stable = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::" + name: stable for name in ("a", "b", "out")},
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
        "--kernel", "independent_boundary_owner::step", "--scope-facts", str(facts_path),
        "--scope-execution", "reached", "--gpu-policy", "sections", "--memory-model", "scoped",
        "--json", "--output-dir", str(output)]).stdout)
    assert report["supported"], report
    manifest = json.loads((output / "scope-manifest.json").read_text())
    assert manifest == report["scopes"]
    owner, = manifest["scopes"]
    assert len(owner["gpu_leaves"]) == 2
    # Only the opaque call after the joined team may close ownership.
    assert len(owner["boundaries"]) == 1, owner["boundaries"]
    native_team, = [operation for segment in owner["planning_segments"]
                   for operation in segment["operations"]["native_operations"]
                   if operation["kind"] == "joined native OpenMP"]
    assert native_team["sections"]["available"], native_team["sections"]
    fflags = [fc, "-O3", "-fopenmp", "-ffp-contract=off", "-fcheck=all", "-J", str(build), "-I", str(build)]
    objects = []
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
    replacement = output / manifest["sources"][str(source)]["replacement"]
    for path in (callback, reference, replacement):
        target = build / f"source-{len(objects):02}.o"
        run([*fflags, "-c", str(path), "-o", str(target)])
        objects.append(str(target))
    bits = "int32" if precision == 4 else "int64"
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""program verify
use iso_fortran_env,only:{bits},error_unit
use native_boundary_audit,only:opaque_visits
use native_boundary_owner,only:native_step=>step,native_visits=>visits
use independent_boundary_owner,only:generated_step=>step,generated_visits=>visits
implicit none
integer,parameter::widths(4)=[17,0,5,17],heights(4)=[3,3,0,3]
real({precision})::a(-2:20,-1:5),b(-2:20,-1:5),out(-2:20,-1:5)
real({precision})::expected_b(-2:20,-1:5),expected_out(-2:20,-1:5),input_reference(-2:20,-1:5)
integer::step,i,j,n,m,expected_opaque
logical::flag,escape
do step=1,4
n=widths(step)
m=heights(step)
flag=mod(step,2)==1
escape=step==4
do j=-1,5
do i=-2,20
a(i,j)=real(3*i+5*j+step,{precision})/16._{precision}
enddo
enddo
input_reference=a
b=-113
out=-117
expected_b=b
expected_out=out
native_visits=0
opaque_visits=0
call native_step(a,expected_b,expected_out,n,m,flag,escape)
expected_opaque=opaque_visits
generated_visits=0
opaque_visits=0
write(error_unit,*) 'BEGIN_BOUNDARY_CALL',step,n,m,flag,escape
call generated_step(a,b,out,n,m,flag,escape)
write(error_unit,*) 'END_BOUNDARY_CALL',step
if(native_visits/=1.or.generated_visits/=1) error stop 'original source prefix replayed'
if(opaque_visits/=expected_opaque.or.opaque_visits/=merge(1,0,escape)) error stop 'native escape replayed'
if(any(transfer(b,[0_{bits}],size(b))/=transfer(expected_b,[0_{bits}],size(expected_b)))) &
 error stop 'complete intermediate or halo differs'
if(any(transfer(out,[0_{bits}],size(out))/=transfer(expected_out,[0_{bits}],size(expected_out)))) &
 error stop 'complete output or halo differs'
if(any(transfer(a,[0_{bits}],size(a))/=transfer(input_reference,[0_{bits}],size(input_reference)))) &
 error stop 'read-only input changed'
enddo
print *,'FOUR_BOUNDARY_CALLS_FIELDS_BITWISE_OK'
end program
""")
    target = build / "driver.o"
    run([*fflags, "-c", str(driver), "-o", str(target)])
    executable = build / "verify"
    run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", *objects, str(target), "-lgfortran", "-o", str(executable)])
    observed = run([str(executable)], trace=True)
    assert "FOUR_BOUNDARY_CALLS_FIELDS_BITWISE_OK" in observed.stdout
    calls = observed.stderr.split("BEGIN_BOUNDARY_CALL")[1:]
    assert len(calls) == 4
    for ordinal, call in enumerate(calls):
        trace = call.split("END_BOUNDARY_CALL", 1)[0]
        assert trace.count("FORT_SCOPED launch ") == (2 if ordinal == 0 else 1 if ordinal == 3 else 0), observed.stderr
        assert trace.count("FORT_SCOPED initialize ") <= 1, "native boundary must not reopen ownership"
        if ordinal == 0:
            # Native face writes must not publish the device-current interior.
            # Only the final intermediate and output interiors are downloaded.
            downloaded = sum(map(int, re.findall(r"FORT_SCOPED download .*bytes=(\d+)", trace)))
            assert downloaded == 2 * 17 * 3 * precision, trace
            uploaded = sum(map(int, re.findall(r"FORT_SCOPED upload .*bytes=(\d+)", trace)))
            assert uploaded == (17 * 3 + 2 * 3) * precision, "intermediate interior uploaded again: " + trace
    fallback = run([str(executable)], trace=True, disabled=True)
    assert "FOUR_BOUNDARY_CALLS_FIELDS_BITWISE_OK" in fallback.stdout
    assert "FORT_SCOPED launch " not in fallback.stderr
