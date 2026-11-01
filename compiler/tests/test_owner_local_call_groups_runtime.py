"""A local reached CALL group publishes once, without replaying a prefix."""
import json
import os
import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = """module live_groups
integer::prefix_visits=0,suffix_visits=0
contains
subroutine first(b,c,n)
integer,intent(in)::n
real(8),intent(inout)::b(:)
real(8),intent(in)::c(:)
integer::i
do i=1,n
b(i)=b(i)+c(i)
enddo
end subroutine
subroutine second(a,b,c,n)
integer,intent(in)::n
real(8),intent(out)::a(:)
real(8),intent(in)::b(:),c(:)
integer::i
do i=1,n
a(i)=b(i)+c(i)
enddo
end subroutine
subroutine third(a,c,n)
integer,intent(in)::n
real(8),intent(inout)::a(:)
real(8),intent(in)::c(:)
integer::i
do i=1,n
a(i)=a(i)+c(i)
enddo
end subroutine
subroutine run(a,b,c,n,flag)
real(8),allocatable,intent(inout)::a(:),b(:),c(:)
integer,intent(in)::n
logical,intent(in)::flag
call unknown_prefix()
if(flag)then
call first(b,c,n)
call second(a,b,c,n)
call third(a,c,n)
endif
call unknown_suffix()
end subroutine
end module
"""

DRIVER = """program exercise
use live_groups
use iso_fortran_env,only:error_unit
implicit none
integer,parameter::nmax=19
real(8),allocatable::a(:),b(:),c(:),ea(:),eb(:),ec(:)
integer::n,i,repeat,repeats
logical::flag,alias
character(32)::mode
call get_command_argument(1,mode)
flag=trim(mode)/='unallocated'
alias=trim(mode)=='alias'
n=nmax
if(trim(mode)=='zero')n=0
repeats=2
if(trim(mode)=='late'.or.alias)repeats=1
if(flag)then
 allocate(a(-3:nmax+4),b(7:nmax+14),c(-8:nmax-1))
 do i=1,nmax+8
  a(i-4)=real(71-i,8)/16.0_8
  b(i+6)=real(500-i,8)/32.0_8
  c(i-9)=real(17*i-3,8)/64.0_8
 enddo
endif
do repeat=1,repeats
 if(flag)then
  ea=a
  eb=b
  ec=c
  do i=1,n
   if(alias)then
    ea(i-4)=ea(i-4)+3.0_8*c(i-9)
   else
    eb(i+6)=eb(i+6)+c(i-9)
    ea(i-4)=eb(i+6)+2.0_8*c(i-9)
   endif
  enddo
 endif
 write(error_unit,'(A,1X,I0)')'LOCAL_GROUP_BEGIN',repeat
 if(alias)then
  call run(a,a,c,n,flag)
 else
  call run(a,b,c,n,flag)
 endif
 write(error_unit,'(A,1X,I0)')'LOCAL_GROUP_END',repeat
 if(prefix_visits/=repeat.or.suffix_visits/=repeat)error stop 'native prefix or suffix replay'
 if(flag)then
  if(any(a/=ea).or.any(c/=ec))error stop 'full fields or readonly physical halos'
  if(.not.alias.and.any(b/=eb))error stop 'prefix output or physical halos'
  c=c+0.25_8
 endif
enddo
print *, 'LOCAL_GROUP_FULL_FIELDS_AND_NATIVE_ONCE_OK',trim(mode)
end program
subroutine unknown_prefix()
use live_groups,only:prefix_visits
prefix_visits=prefix_visits+1
end subroutine
subroutine unknown_suffix()
use live_groups,only:suffix_visits
suffix_visits=suffix_visits+1
end subroutine
"""


@pytest.fixture(scope="module")
def local_group_binaries(tmp_path_factory):
    return build_local_group_binaries(tmp_path_factory)


def build_local_group_binaries(tmp_path_factory, *, source=SOURCE, driver_source=DRIVER,
                               initialized="none", lifetime="selected_window"):
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/GNU toolchains required")
    directory = tmp_path_factory.mktemp("local_group")
    original = directory / "original.f90"
    original.write_text(source)
    fact = {"storage": "stable", "initialized": initialized, "escapes": False,
            "allocation_changes": False, "lifetime": lifetime}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
             "captures": {"argument::"+name: dict(fact) for name in ("a", "b", "c")}}
    contracts = {"live_groups::run": {"native_ancestor": {"version": 2,
        "lifetime": "owner_local_call_groups", "device_roots": ["live_groups::"+name
            for name in ("first", "second", "third")]}}}
    builder = ScopeBuilder([original], "live_groups::run", facts=facts, contracts=contracts,
                           options=CompilerOptions(), config=OffloadConfig(policy="sections"))
    outputs, manifest = builder.run()
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    build = directory / "build"
    build.mkdir()
    for name, source in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    (directory / "report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    commands = []

    def command(argv):
        index = len(commands)
        commands.append(argv)
        (build / f"command-{index}.json").write_text(json.dumps({"argv": argv, "cwd": str(build)}) + "\n")
        result = subprocess.run(argv, cwd=build, capture_output=True, text=True, timeout=120)
        (build / f"command-{index}.stdout").write_text(result.stdout)
        (build / f"command-{index}.stderr").write_text(result.stderr)
        (build / f"command-{index}.completion.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr

    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            path, target = build / artifact["path"], build / f"object-{index}.o"
            assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            if artifact["language"] == "cuda":
                options = [nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp",
                           "-DFORT_SCOPE_TEST_FAULTS", *artifact["numerical_contract"]["required_cuda_options"],
                           "-I", str(build)]
            else:
                options = [fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build)]
            command([*options, "-c", str(path), "-o", str(target)])
            objects.append(str(target))
    driver = build / "driver.f90"
    driver.write_text(driver_source)
    candidate, native = build / "candidate", build / "native"
    command([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build),
             str(driver), *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64",
             "-lcudart", "-lstdc++", "-o", str(candidate)])
    command([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build),
             str(original), str(driver), "-o", str(native)])
    return build, manifest, candidate, native


@pytest.mark.native
def test_published_runtime_and_original_control_build_without_cuda_execution(local_group_binaries):
    _build, _manifest, candidate, native = local_group_binaries
    assert candidate.is_file()
    assert native.is_file()


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["resident", "late", "alias", "zero", "unallocated", "hidden"])
def test_original_call_group_retains_only_until_exact_branch_exit(local_group_binaries, mode):
    build, manifest, candidate, native = local_group_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER", None)
    if mode == "hidden":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    if mode == "late":
        environment["FORT_SCOPE_TEST_FAIL_ALLOC_AFTER"] = "2"
    results = []
    for name, binary in (("native", native), ("candidate", candidate)):
        argv = [str(binary), mode]
        (build / f"{mode}-{name}.command.json").write_text(json.dumps({"argv": argv,
            "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
            "FORT_SCOPE_TEST_FAIL_ALLOC_AFTER": environment.get("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER")}) + "\n")
        result = subprocess.run(argv, cwd=build, env=environment, capture_output=True, text=True, timeout=30)
        (build / f"{mode}-{name}.stdout").write_text(result.stdout)
        (build / f"{mode}-{name}.stderr").write_text(result.stderr)
        (build / f"{mode}-{name}.completion.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        results.append(result)
    reference, result = results
    assert reference.stdout == result.stdout
    assert "LOCAL_GROUP_FULL_FIELDS_AND_NATIVE_ONCE_OK" in result.stdout
    launches = [line for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED launch ")]
    copies = {kind: [dict(re.findall(r"(\w+)=([^ ]+)", line)) for line in result.stderr.splitlines()
                     if line.startswith("FORT_SCOPED " + kind + " ")] for kind in ("upload", "download")}
    ledger = {"mode": mode, "launches": len(launches),
              "domains": result.stderr.count('\"event\":\"domain_create\"'),
              "upload_bytes": sum(int(item["bytes"]) for item in copies["upload"]),
              "download_bytes": sum(int(item["bytes"]) for item in copies["download"]),
              "copies": copies}
    (build / f"{mode}-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    if mode in {"hidden", "zero", "unallocated"}:
        assert not launches, result.stderr
        assert not re.search(r"FORT_SCOPED (upload|download) ", result.stderr)
    elif mode == "resident":
        assert len(launches) == 6, result.stderr
        assert ledger["domains"] == 2, result.stderr
        assert (ledger["upload_bytes"], ledger["download_bytes"]) == (608, 608), result.stderr
        context = manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["resources"]
        slots = {item["resource"]: item["registration_identity"] for item in context}
        uploads = [line for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED upload ")]
        reads = [line for line in uploads if f"buffer={slots['argument::c']} " in line]
        assert len(reads) == 2, result.stderr
        assert all("bytes=152 " in line for line in reads), result.stderr
    elif mode == "late":
        assert len(launches) == 1, result.stderr
        assert ledger["domains"] == 1, result.stderr
        assert (ledger["upload_bytes"], ledger["download_bytes"]) == (304, 152), result.stderr
        assert result.stderr.count("mode=native-scoped-resource gpu_units=0 cpu_units=1") == 2, result.stderr
    else:
        assert len(launches) == 2, result.stderr
        assert ledger["domains"] == 2, result.stderr
        assert (ledger["upload_bytes"], ledger["download_bytes"]) == (608, 304), result.stderr
