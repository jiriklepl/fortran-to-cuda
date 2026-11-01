"""An original conditional NOWAIT team is one preflighted GPU alternative."""
import json
import os
import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.tests.test_atomic_joined_team_source import SOURCE, emit
from compiler.tests.test_source_scopes import run


@pytest.fixture(scope="module")
def compiled_atomic_team(tmp_path_factory):
    build = tmp_path_factory.mktemp("atomic-generated")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fc)):
        pytest.skip("CUDA and Fortran compilation toolchains required")
    source = (SOURCE.replace("a(:),b(:)", "a(-2:),b(-2:)")
        .replace("integer::i", "integer::i\nreal(8),parameter::c1=9._8/8,c3=1._8/(8*3)")
        .replace("a(i)+p(i-1)", "a(i)+c1*p(i-1)-c3*p(i)"))
    outputs, manifest, _ = emit(build, source=source)
    original = build / "owner.f90"
    (build / "scope-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            path = build / artifact["path"]
            assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            target = build / f"object-{index}.o"
            if artifact["language"] == "cuda":
                command = [nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp",
                           *artifact["numerical_contract"]["required_cuda_options"], "-I", str(build)]
            else:
                command = [fc, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build)]
            result = run([*command, "-c", str(path), "-o", str(target)], cwd=build)
            (build / f"compile-{index}.stdout").write_text(result.stdout)
            (build / f"compile-{index}.stderr").write_text(result.stderr)
            objects.append(str(target))
    driver = build / "driver.f90"
    driver.write_text("""program exercise
use atomic_owner
use iso_fortran_env,only:error_unit
implicit none
real(8),allocatable::a(:),b(:),p(:),ea(:),eb(:),op(:)
real(8),parameter::c1=9._8/8,c3=1._8/(8*3)
integer::shape,n,i,path,repeat,mode
logical::flag
do shape=1,4
 select case(shape)
 case(1)
 n=0
 case(2)
 n=1
 case(3)
 n=7
 case default
 n=19
 end select
 allocate(a(-2:n+4),b(-2:n+4),p(0:n+1),ea(-2:n+4),eb(-2:n+4),op(0:n+1))
 do i=-2,n+4
 a(i)=real(17*i*i-3*i+11,8)/64.0_8
 b(i)=real(500-i,8)/32.0_8
 enddo
 do i=0,n+1
 p(i)=real(3*i-7,8)/16.0_8
 enddo
 op=p
 do path=1,3
 mode=2
 if(path==1)mode=4
 flag=path/=3
 do repeat=1,2
 ea=a
 eb=b
 do i=1,n
 if(mode==4)then
 ea(i)=a(i)+c1*p(i-1)-c3*p(i)
 eb(i)=b(i)+p(i+1)
 else
 ea(i)=a(i)+p(i)
 if(flag)then
 eb(i)=b(i)+p(i)
 else
 eb(i)=b(i)-p(i)
 endif
 endif
 enddo
 write(error_unit,'(A,3(1X,I0))')'ATOMIC_BEGIN',n,path,repeat
 call step(a,b,p,n,mode,flag)
 write(error_unit,'(A,3(1X,I0))')'ATOMIC_END',n,path,repeat
 if(any(a/=ea).or.any(b/=eb))error stop 'full fields or halos changed'
 if(any(p/=op))error stop 'readonly input changed'
 enddo
 enddo
 deallocate(a,b,p,ea,eb,op)
enddo
print *, 'ATOMIC_ORIGINAL_FIELDS_HALOS_AND_BRANCHES_OK'
end program
""")
    candidate, native = build / "candidate", build / "native"
    run([fc, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-I", str(build), str(driver), *objects,
         "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(candidate)], cwd=build)
    run([fc, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build),
         str(original), str(driver), "-o", str(native)], cwd=build)
    wrapper = build / "fault.cpp"
    wrapper.write_text(r'''
#include "scoped_runtime.h"
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" int __real_fort_scope_source_execute_v1(fort_scope_t,const fort_scope_external_call_v1*,
 std::uint64_t,fort_scope_source_worker_v1,void*,fort_scope_external_report_v1*);
struct Wrapped { fort_scope_source_worker_v1 worker; void* user; };
static int after_submission(void* opaque,const fort_scope_external_view_v1* views,std::size_t count,
 void* stream,int* started,std::uint64_t* accepted){
 auto& original=*static_cast<Wrapped*>(opaque);
 int status=original.worker(original.user,views,count,stream,started,accepted);
 if(status==FORT_SCOPE_OK&&*started==1&&*accepted==2){
  std::fprintf(stderr,"ATOMIC_INJECTED_AFTER_TWO_ACCEPTED_LAUNCHES\n");
  return FORT_SCOPE_EXECUTION;
 }
 return status;
}
extern "C" int __wrap_fort_scope_source_execute_v1(fort_scope_t context,const fort_scope_external_call_v1* call,
 std::uint64_t expected,fort_scope_source_worker_v1 worker,void* user,fort_scope_external_report_v1* report){
 const char* mode=std::getenv("FORT_ATOMIC_TEST_FAULT");
 if(mode&&std::strcmp(mode,"prework")==0){
  std::fprintf(stderr,"ATOMIC_INJECTED_PREWORK_RESOURCE\n");
  *report={1,FORT_SCOPE_RESOURCE,0,0};return FORT_SCOPE_RESOURCE;
 }
 if(mode&&std::strcmp(mode,"after_submission")==0){
  Wrapped original{worker,user};
  return __real_fort_scope_source_execute_v1(context,call,expected,after_submission,&original,report);
 }
 return __real_fort_scope_source_execute_v1(context,call,expected,worker,user,report);
}
''')
    wrapper_object = build / "fault.o"
    run([host, "-std=c++17", "-I", str(build), "-c", str(wrapper), "-o", str(wrapper_object)], cwd=build)
    fault = build / "fault-candidate"
    run([fc, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-I", str(build), str(driver), *objects,
         str(wrapper_object), "-Wl,--wrap=fort_scope_source_execute_v1", "-L/usr/local/cuda/lib64",
         "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(fault)], cwd=build)
    return build, candidate, native, fault


def test_generated_atomic_team_compiles_with_original_fortran_abi(compiled_atomic_team):
    build, candidate, native, _fault = compiled_atomic_team
    assert candidate.is_file()
    assert native.is_file()
    manifest = json.loads((build / "scope-manifest.json").read_text())
    assert len(manifest["scopes"][0]["source_device_tasks"]) == 1


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["enabled", "hidden"])
def test_atomic_original_team_fields_haloconservation_and_prework_fallback_once(compiled_atomic_team, hidden):
    build, candidate, native, _fault = compiled_atomic_team
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    mode = "hidden" if hidden else "enabled"
    reference = run([str(native)], cwd=build, env=environment)
    result = run([str(candidate)], cwd=build, env=environment)
    for role, process in (("native", reference), ("candidate", result)):
        (build / f"{mode}-{role}.stdout").write_text(process.stdout)
        (build / f"{mode}-{role}.stderr").write_text(process.stderr)
    assert result.stdout == reference.stdout
    assert "ATOMIC_ORIGINAL_FIELDS_HALOS_AND_BRANCHES_OK" in result.stdout
    cases = re.findall(r"ATOMIC_BEGIN (\d+) (\d+) (\d+)\n(.*?)ATOMIC_END \1 \2 \3", result.stderr, re.S)
    assert len(cases) == 24
    for size, path, _repeat, events in cases:
        size, path = int(size), int(path)
        launches = [line for line in events.splitlines() if line.startswith("FORT_SCOPED launch ")]
        uploads = [line for line in events.splitlines() if line.startswith("FORT_SCOPED upload ")]
        downloads = [line for line in events.splitlines() if line.startswith("FORT_SCOPED download ")]
        if hidden or not size:
            assert not launches, events
            assert not uploads, events
            assert not downloads, events
        else:
            assert len(launches) == 2, events
            uploaded = sum(int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in uploads)
            downloaded = sum(int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in downloads)
            p_cells = size + 2 if path == 1 else size
            assert uploaded == 8 * (2 * size + p_cells), events
            assert downloaded == 16 * size, events


@pytest.mark.cuda
def test_all_unit_prework_decline_replays_only_the_complete_original_team(compiled_atomic_team):
    build, _candidate, native, fault = compiled_atomic_team
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1", "FORT_ATOMIC_TEST_FAULT": "prework"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    reference = run([str(native)], cwd=build, env=environment)
    result = run([str(fault)], cwd=build, env=environment)
    (build / "prework.stdout").write_text(result.stdout)
    (build / "prework.stderr").write_text(result.stderr)
    assert result.stdout == reference.stdout
    assert result.stderr.count("ATOMIC_INJECTED_PREWORK_RESOURCE") == 18
    assert not re.search(r"^FORT_SCOPED (?:launch|upload|download) ", result.stderr, re.M)
    assert result.stderr.count("ATOMIC_END") == 24


@pytest.mark.cuda
def test_failure_after_any_submission_is_fatal_without_original_team_replay(compiled_atomic_team):
    build, _candidate, _native, fault = compiled_atomic_team
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1", "FORT_ATOMIC_TEST_FAULT": "after_submission"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run([str(fault)], cwd=build, env=environment, capture_output=True, text=True, timeout=30)
    (build / "after-submission.stdout").write_text(result.stdout)
    (build / "after-submission.stderr").write_text(result.stderr)
    assert result.returncode != 0
    assert result.stderr.count("ATOMIC_INJECTED_AFTER_TWO_ACCEPTED_LAUNCHES") == 1
    assert "atomic original team attempted work; replay prohibited" in result.stderr
    assert len(re.findall(r"^FORT_SCOPED launch ", result.stderr, re.M)) == 2
    assert "ATOMIC_END 1 1 1" not in result.stderr
    assert "ATOMIC_ORIGINAL_FIELDS_HALOS_AND_BRANCHES_OK" not in result.stdout
