"""Demanded native effects retain a three-dimensional resident computation.

This fixture verifies numerical/coherence behavior, not calibration or speedup.
"""

import json
import os
import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.numerical_contract import require_explicit_cuda_environment, require_numerical_build_contract
from compiler.tests.test_native_demand_owner import demanded_operations, generate

SOURCE = """module demand_owner
implicit none
type profile
real(8),allocatable::flux(:)
integer::visits=0
end type
type(profile)::metadata
real(8)::a(-1:6,-1:6,-1:6),b(-1:6,-1:6,-1:6)
real(8)::star(-1:6,-1:6,-1:6),out(-1:6,-1:6,-1:6)
contains
subroutine advance()
integer::i,j,k
do k=-1,6
do j=-1,6
do i=-1,6
b(i,j,k)=2*a(i,j,k)
enddo
enddo
enddo
call correct(-0.5d0*a(0,0,0))
do k=0,5
do j=0,5
do i=0,5
star(i,j,k)=b(i,j,k)+0.25d0*(a(i-1,j,k)+a(i+1,j,k)-2*a(i,j,k))
enddo
enddo
enddo
do k=0,5
do j=0,5
do i=0,5
out(i,j,k)=out(i,j,k)+0.5d0*star(i,j,k)
enddo
enddo
enddo
end subroutine
subroutine correct(scale)
real(8),intent(in)::scale
if(scale<0.d0)then
metadata%flux(1)=metadata%flux(1)+scale+a(1,1,1)
else
metadata%flux(2)=metadata%flux(2)-scale+a(2,2,2)
endif
metadata%visits=metadata%visits+1
end subroutine
end module
"""

DRIVER = """program verify
use iso_fortran_env,only:int64,error_unit
use demand_owner
use reference_owner,only:native_advance=>advance,na=>a,nb=>b,ns=>star,no=>out,nm=>metadata
implicit none
integer::step,i,j,k,native_unit,candidate_unit
open(newunit=native_unit,file='native-fields.bin',access='stream',form='unformatted',status='replace')
open(newunit=candidate_unit,file='candidate-fields.bin',access='stream',form='unformatted',status='replace')
allocate(metadata%flux(-1:2))
allocate(nm%flux(-1:2))
do step=1,2
do k=-1,6
do j=-1,6
do i=-1,6
a(i,j,k)=real(merge(1,-1,step==1)*(i*i+3*j+7*k+33),8)/16.d0
enddo
enddo
enddo
b=-111.d0
star=-113.d0
out=-117.d0
metadata%flux=[-9.d0,-3.d0,1.d0,7.d0]
metadata%visits=0
na=a
nb=b
ns=star
no=out
nm%flux=metadata%flux
nm%visits=0
call native_advance()
write(error_unit,*)'BEGIN_DEMAND_CALL',step
call advance()
write(error_unit,*)'END_DEMAND_CALL',step
if(metadata%visits/=1.or.nm%visits/=1) error stop 'native call did not execute exactly once'
if(any(lbound(a)/=-1).or.any(ubound(a)/=6)) error stop 'old bounds changed'
if(any(lbound(b)/=-1).or.any(ubound(b)/=6)) error stop 'producer bounds changed'
if(any(lbound(star)/=-1).or.any(ubound(star)/=6)) error stop 'diffusion bounds changed'
if(any(lbound(out)/=-1).or.any(ubound(out)/=6)) error stop 'accumulation bounds changed'
if(lbound(metadata%flux,1)/=-1.or.ubound(metadata%flux,1)/=2) error stop 'metadata bounds changed'
if(any(transfer(a,[0_int64],size(a))/=transfer(na,[0_int64],size(na)))) error stop 'old field differs'
if(any(transfer(b,[0_int64],size(b))/=transfer(nb,[0_int64],size(nb)))) error stop 'producer field differs'
if(any(transfer(star,[0_int64],size(star))/=transfer(ns,[0_int64],size(ns)))) error stop 'diffusion halos differ'
if(any(transfer(out,[0_int64],size(out))/=transfer(no,[0_int64],size(no)))) error stop 'accumulation halos differ'
if(any(transfer(metadata%flux,[0_int64],4)/=transfer(nm%flux,[0_int64],4))) error stop 'metadata differs'
write(native_unit)na,nb,ns,no,nm%flux,nm%visits
write(candidate_unit)a,b,star,out,metadata%flux,metadata%visits
enddo
close(native_unit)
close(candidate_unit)
print *,'NATIVE_DEMAND_COMPLETE_FIELDS_BITWISE_OK'
end program
"""


def fixture_source(variant):
    if variant == "cross_procedure":
        return SOURCE.replace("subroutine advance()\ninteger::i,j,k", """subroutine advance()
call produce()
call momentum()
end subroutine
subroutine produce()
integer::i,j,k""").replace("call correct(-0.5d0*a(0,0,0))", """end subroutine
subroutine momentum()
integer::i,j,k
call correct(-0.5d0*a(0,0,0))""")
    if variant == "managed_correction":
        return SOURCE.replace("if(scale<0.d0)then", "b(0,0,0)=b(0,0,0)+scale+metadata%flux(1)\nif(scale<0.d0)then")
    if variant == "pointer_boundary":
        # Replaying the producer would change b and therefore fail the complete
        # field comparison, even if the original native CALL still ran once.
        return SOURCE.replace("real(8),allocatable::flux(:)", "real(8),pointer::flux(:)").replace(
            "b(i,j,k)=2*a(i,j,k)", "b(i,j,k)=b(i,j,k)+2*a(i,j,k)")
    assert variant == "host_read"
    return SOURCE


def test_3d_native_demand_cost_is_not_inferred_from_numerical_profile(tmp_path):
    path, _builder, outputs, report = generate(tmp_path, SOURCE, policy="auto")
    owner, = report["scopes"]
    operation, = demanded_operations(report)
    assert operation["resource_effects"] == {"demand_owner::a": ["read"]}
    assert not owner["boundaries"]
    assert owner["automatic_preflight"]["selection"] == "native"
    assert not owner["estimate_available"]
    assert "native interval costs require compatible source-bound observations" in owner["planning_reason"]
    assert outputs[report["sources"][str(path)]["replacement"]] == SOURCE
    assert "fort_scope_" not in outputs[report["sources"][str(path)]["replacement"]]
    assert not any(item["role"] == "shared_entry" for item in report["build_sources"])
    # Existing deep AUTO fallback still lists unused common runtime artifacts.
    # The original source has no context creation or runtime calls.


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["host_read", "managed_correction", "pointer_boundary", "cross_procedure"])
def test_3d_native_demand_keeps_current_host_read_and_complete_fields(tmp_path, variant):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA toolchain required")
    require_explicit_cuda_environment()
    original = fixture_source(variant)
    source, _builder, outputs, manifest = generate(tmp_path, original)
    owner, = manifest["scopes"]
    assert owner["checked_execution_plan"]["numerical_invocations"] == 3
    if variant == "cross_procedure":
        assert len(owner["checked_execution_plan"]["procedures"]) >= 2
    if variant == "pointer_boundary":
        assert not demanded_operations(manifest)
        assert len(owner["boundaries"]) == 1, owner["boundaries"]
        assert not owner["boundaries"][0]["reopen"]
    else:
        operation, = demanded_operations(manifest)
        assert not owner["boundaries"], owner["boundaries"]
        expected_effects = {"demand_owner::a": ["read"]}
        if variant == "managed_correction":
            expected_effects["demand_owner::b"] = ["read", "write"]
        assert operation["resource_effects"] == expected_effects
    resources = owner["checked_execution_plan"]["resources"]
    assert {row["resource"] for row in resources} == {
        "demand_owner::a", "demand_owner::b", "demand_owner::star", "demand_owner::out"}
    identities = {row["resource"].split("::")[-1]: row["registration_identity"] for row in resources}
    assert len(set(identities.values())) == 4
    replacement = manifest["sources"][str(source)]["replacement"]
    if variant != "cross_procedure":
        assert outputs[replacement].count("call correct(-0.5d0*a(0,0,0))") == 1
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    environment.pop("FORT_RUNTIME_TRACE", None)
    commands = []

    def run(argv, *, trace=False, disabled=False):
        ordinal = len(commands)
        commands.append({"argv": argv, "cwd": str(build), "trace": trace, "device_disabled": disabled})
        (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(argv, cwd=build, env={**environment,
            **({"FORT_RUNTIME_TRACE": "1"} if trace else {}),
            **({"CUDA_VISIBLE_DEVICES": ""} if disabled else {})},
            capture_output=True, text=True, timeout=180, check=False)
        (tmp_path / f"command-{ordinal:02}.stdout").write_text(result.stdout)
        (tmp_path / f"command-{ordinal:02}.stderr").write_text(result.stderr)
        (tmp_path / f"command-{ordinal:02}.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    fflags = [fc, "-O3", "-fopenmp", "-ffp-contract=off", "-fcheck=all", "-J", str(build), "-I", str(build)]
    objects, compiled = [], set()
    for role in ("common_runtime", "shared_entry", "original_source"):
        for descriptor in manifest["build_sources"]:
            if descriptor["role"] != role:
                continue
            path = build / descriptor["path"]
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
    assert compiled == {item["path"] for item in manifest["build_sources"]}
    for name, text in (("reference", original.replace("demand_owner", "reference_owner")), ("driver", DRIVER)):
        path, target = build / (name + ".f90"), build / (name + ".o")
        path.write_text(text)
        run([*fflags, "-c", str(path), "-o", str(target)])
        objects.append(str(target))
    executable = build / "verify"
    run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", *objects, "-lgfortran", "-o", str(executable)])
    observed = run([str(executable)], trace=True)
    assert "NATIVE_DEMAND_COMPLETE_FIELDS_BITWISE_OK" in observed.stdout
    expected = (build / "native-fields.bin").read_bytes()
    assert len(expected) == 2 * ((4 * 8**3 + 4) * 8 + 4)
    assert (build / "candidate-fields.bin").read_bytes() == expected
    calls = observed.stderr.split("BEGIN_DEMAND_CALL")[1:]
    assert len(calls) == 2
    ledgers = []
    for call in calls:
        trace = call.split("END_DEMAND_CALL", 1)[0]
        events = [dict(re.findall(r"(\w+)=([^ ]+)", line)) | {"kind": line.split()[1]}
                  for line in trace.splitlines() if line.startswith("FORT_SCOPED ") and "provenance_version=1" in line]
        expected_launches = 1 if variant == "pointer_boundary" else 3
        assert sum(item["kind"] == "launch" for item in events) == expected_launches, trace
        assert sum(item["kind"] == "close" for item in events) == 1, trace
        assert len({item["context"] for item in events}) == 1, trace
        transfers = [item for item in events if item["kind"] in {"upload", "download"}]
        assert sum(item["kind"] == "upload" for item in transfers) == (3 if variant == "managed_correction" else 2), trace
        assert sum(item["kind"] == "download" for item in transfers) == (1 if variant == "pointer_boundary" else 3), trace
        uploads = {"a": 8**3 * 8, "b": 0, "star": 0, "out": 6**3 * 8}
        downloads = {"a": 0, "b": 8**3 * 8, "star": 6**3 * 8, "out": 6**3 * 8}
        if variant == "managed_correction":
            uploads["b"] = 6**3 * 8
        elif variant == "pointer_boundary":
            uploads.update(b=8**3 * 8, out=0)
            downloads.update(star=0, out=0)
        for name, identity in identities.items():
            selected = [item for item in transfers if int(item["buffer"]) == identity]
            uploaded = sum(int(item["bytes"]) for item in selected if item["kind"] == "upload")
            downloaded = sum(int(item["bytes"]) for item in selected if item["kind"] == "download")
            assert uploaded == uploads[name], trace
            assert downloaded == downloads[name], trace
            assert len({item["buffer_handle"] for item in selected}) == int(bool(selected)), trace
        launches = [index for index, item in enumerate(events) if item["kind"] == "launch"]
        if variant in {"host_read", "cross_procedure"}:
            assert not any(item["kind"] in {"upload", "download"} for item in events[launches[0]+1:launches[1]]), trace
        elif variant == "managed_correction":
            interrupted = [item for item in events[launches[0]+1:launches[1]] if item["kind"] in {"upload", "download"}]
            assert [(item["kind"], int(item["buffer"]), int(item["bytes"])) for item in interrupted] == [
                ("download", identities["b"], 8**3 * 8), ("upload", identities["b"], 6**3 * 8)], trace
        ledgers.append({"variant": variant, "contexts": 1, "launches": expected_launches, "transfers": transfers})
    (tmp_path / "transfer-ledger.json").write_text(json.dumps(ledgers, indent=2) + "\n")
    untraced = run([str(executable)])
    assert "FORT_SCOPED " not in untraced.stderr
    assert (build / "candidate-fields.bin").read_bytes() == expected
    fallback = run([str(executable)], trace=True, disabled=True)
    assert "NATIVE_DEMAND_COMPLETE_FIELDS_BITWISE_OK" in fallback.stdout
    assert (build / "native-fields.bin").read_bytes() == expected
    assert (build / "candidate-fields.bin").read_bytes() == expected
    assert not re.search(r"FORT_SCOPED (?:launch|upload|download) ", fallback.stderr)
