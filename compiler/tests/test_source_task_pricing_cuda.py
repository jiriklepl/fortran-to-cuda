"""Measured-profile AUTO choices retain original root/child numerical work.

This opt-in gate consumes an independently calibrated profile. Its original source
body exercises local choices; it is not application speed evidence.
"""

import json
import os
import re
import shutil
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_source_scopes import run
from compiler.tests.test_tile_private import SOURCE


def priced_source(child):
    # The exact repeated arithmetic belongs to the original program and is
    # lowered by the normal source proof. No cost or rate is overridden.
    terms = ["(tmp(li+1)-tmp(li-1))"] * 32
    expression = " &\n       + ".join(" + ".join(terms[i:i+4]) for i in range(0, len(terms), 4))
    source = SOURCE.replace("b(i)=b(i)+(tmp(li+1)-tmp(li-1))", "b(i)=b(i)+ &\n       " + expression)
    assert source != SOURCE
    if child:
        source = source.rsplit("end module", 1)[0] + """subroutine outer(a,b,n,t)
integer,intent(in)::n,t
real(8),intent(in)::a(-3:)
real(8),intent(inout)::b(-3:)
call advance(a,b,n,t)
end subroutine
end module
"""
    return source


def _compile_priced_source(directory, child):
    profile_path = os.environ.get("FORT_TEST_SOURCE_PROFILE")
    if not profile_path:
        pytest.skip("independently calibrated source-task profile required")
    profile_path = Path(profile_path)
    profile = json.loads(profile_path.read_text())
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA and GNU Fortran toolchains required")
    original = directory / "source.f90"
    original.write_text(priced_source(child))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: deepcopy(FACT) for name in ("a", "b")}}
    builder = ScopeBuilder([original], "stencil::" + ("outer" if child else "advance"), facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy="auto", scope_execution="graph", profile=profile))
    outputs, manifest = builder.run()
    assert manifest["scope_count"] == 1, manifest.get("graph_owner_attempt")
    task, = manifest["scopes"][0]["source_device_tasks"]
    live = "native_transactions" in profile.get("source_tasks", {})
    assert task["version"] == (4 if live else 2)
    assert task["execution_binding"]["provider"]["metadata"]["version"] == (6 if live else 3)
    assert manifest["scopes"][0]["estimate_available"] is False
    build = directory / "build"
    build.mkdir()
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (build / "scope-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (build / "profile-input.json").write_text(json.dumps({"path": str(profile_path),
        "sha256": sha256(profile_path.read_bytes()).hexdigest(), "pricing": task["pricing"]}, indent=2) + "\n")
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            path, target = build / artifact["path"], build / f"object-{index}.o"
            assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            if artifact["language"] == "cuda":
                command = [nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp",
                           *artifact["numerical_contract"]["required_cuda_options"], "-I", str(build)]
            else:
                command = [fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build)]
            argv = [*command, "-c", str(path), "-o", str(target)]
            (build / f"compile-{index}.command.json").write_text(json.dumps(argv) + "\n")
            result = run(argv, cwd=build)
            (build / f"compile-{index}.stdout").write_text(result.stdout)
            (build / f"compile-{index}.stderr").write_text(result.stderr)
            objects.append(str(target))
    reference = build / "reference.f90"
    reference.write_text(original.read_text().replace("module stencil", "module reference_stencil", 1))
    procedure = "outer" if child else "advance"
    driver = build / "driver.f90"
    driver.write_text(f"""program exercise
use stencil
use reference_stencil, only: reference_call=>{procedure}
use iso_fortran_env, only:error_unit
use ieee_exceptions
implicit none
real(8),allocatable::a(:),b(:),expected(:),original_a(:)
integer::shape,n,i,repeat
logical::flags_before(size(ieee_all)),flags_after(size(ieee_all))
do shape=1,3
 select case(shape)
 case(1)
 n=0
 case(2)
 n=7
 case default
 n=262144
 end select
 allocate(a(-3:n+4),b(-3:n+4),expected(-3:n+4),original_a(-3:n+4))
 do i=-3,n+4
  a(i)=real(modulo(i,31)-15,8)/64.0_8
  b(i)=real(modulo(i,17)-8,8)/32.0_8
 enddo
 original_a=a
 do repeat=1,2
  expected=b
  call reference_call(a,expected,n,128)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_divide_by_zero,.true.)
  call ieee_get_flag(ieee_all,flags_before)
  write(error_unit,'(A,1X,I0,1X,I0)')'PRICED_CASE_BEGIN',n,repeat
  call {procedure}(a,b,n,128)
  call ieee_get_flag(ieee_all,flags_after)
  if(any(flags_before.neqv.flags_after))error stop 'source pricing changed original IEEE flags'
  write(error_unit,'(A,1X,I0,1X,I0)')'PRICED_CASE_END',n,repeat
  if(any(a/=original_a))error stop 'readonly input changed'
  if(any(b/=expected))error stop 'full field or halo mismatch / source replay'
 enddo
 deallocate(a,b,expected,original_a)
enddo
print *, 'PRICED_ORIGINAL_TEAM_FULL_FIELDS_AND_HALOS_OK'
end program
""")
    executable = build / "candidate"
    argv = [fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build),
            str(reference), str(driver), *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64",
            "-lcudart", "-lstdc++", "-o", str(executable)]
    (build / "link.command.json").write_text(json.dumps(argv) + "\n")
    result = run(argv, cwd=build)
    (build / "link.stdout").write_text(result.stdout)
    (build / "link.stderr").write_text(result.stderr)
    return build, executable


@pytest.fixture(scope="module")
def priced_source_builds(tmp_path_factory):
    builds = {}

    def get(child):
        if child not in builds:
            builds[child] = _compile_priced_source(tmp_path_factory.mktemp("compiled_priced_source"), child)
        return builds[child]

    return get


@pytest.fixture(scope="module", params=[False, True], ids=["root", "nested"])
def compiled_priced_source(request, priced_source_builds):
    return priced_source_builds(request.param)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["enabled", "hidden", "limited"])
def test_measured_auto_source_choices_preserve_original_fields_and_prework_fallback(compiled_priced_source, mode):
    build, executable = compiled_priced_source
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "OMP_THREAD_LIMIT": "4", "OMP_MAX_ACTIVE_LEVELS": "1", "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if mode == "hidden":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif mode == "limited":
        environment["OMP_THREAD_LIMIT"] = "1"
    result = run([str(executable)], cwd=build, env=environment)
    (build / f"{mode}.stdout").write_text(result.stdout)
    (build / f"{mode}.stderr").write_text(result.stderr)
    assert "PRICED_ORIGINAL_TEAM_FULL_FIELDS_AND_HALOS_OK" in result.stdout
    cases = re.findall(r"PRICED_CASE_BEGIN (\d+) (\d+)\n(.*?)PRICED_CASE_END \1 \2", result.stderr, re.S)
    assert len(cases) == 6
    for size, _repeat, events in cases:
        size = int(size)
        quotes = re.findall(r"FORT_SOURCE_QUOTE .*?available=(\d+) gpu=(\d+) reason=(\d+)", events)
        launches = [line for line in events.splitlines() if line.startswith("FORT_SCOPED launch ")]
        uploads = [line for line in events.splitlines() if line.startswith("FORT_SCOPED upload ")]
        downloads = [line for line in events.splitlines() if line.startswith("FORT_SCOPED download ")]
        if not size:
            assert not quotes, events
            assert not launches, events
            assert not uploads, events
            assert not downloads, events
            continue
        assert len(quotes) == 1, events
        if mode == "limited":
            assert quotes[0][0] == "0", events
        else:
            # Both fresh domains must honestly include the measured cold
            # startup. Size alone cannot waive that cost or force a GPU choice.
            assert quotes[0][:2] == ("1", "0"), events
        assert not launches, events
        assert not uploads, events
        assert not downloads, events


@pytest.mark.cuda
def test_measured_auto_uses_real_resident_replicas_and_process_warmth(priced_source_builds):
    build, _ = priced_source_builds(False)
    manifest = json.loads((build / "scope-manifest.json").read_text())
    task, = manifest["scopes"][0]["source_device_tasks"]
    provider = task["execution_binding"]["provider"]
    parameters = provider["metadata"]["parameters"]
    values = {("scalar", "n"): "&n", ("scalar", "t"): "&t",
              ("external_lower", "a"): "&lower", ("external_lower", "b"): "&lower",
              ("scratch_lower", "tmp"): "&scratch_lower", ("scratch_upper", "tmp"): "&scratch_upper"}
    arguments = ",".join(values[p["role"], p["binding"]["name"]] for p in parameters)
    prototype = ",".join(["fort_scope_t", "fort_buffer_t", "fort_buffer_t",
                           *["const int*" for _ in parameters], "fort_scope_external_report_v1*"])
    source = build / "warm.cu"
    source.write_text(r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cfenv>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
extern "C" int @ENTRY@(@PROTOTYPE@);
extern "C" void reference_bridge(const double*,double*,int,int);
static void check(int s){if(s){std::fprintf(stderr,"%d %s\n",s,fort_scope_error());std::abort();}}
__global__ void producer(double* b,std::size_t count){
 const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)b[i]+=1.0/32;
}
int submit(void*,const fort_scope_external_view_v1* views,std::size_t count,void* stream,int* started,uint64_t* accepted){
 *started=0;*accepted=0;assert(count==1);*started=1;
 producer<<<(views[0].extents[0]+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
  static_cast<double*>(views[0].device),views[0].extents[0]);
 if(cudaGetLastError()!=cudaSuccess)return FORT_SCOPE_EXECUTION;*accepted=1;return FORT_SCOPE_OK;
}
static fort_scope_stats stats(fort_scope_t c){fort_scope_stats s{};check(fort_scope_stats_get(c,&s));return s;}
int main(){
 int n=262144,t=128,lower=-3,scratch_lower=-1,scratch_upper=t+2;
 const std::size_t extent=n+8;const std::int64_t lb=lower;
 std::vector<double>a(extent),b(extent),expected(extent);
 for(std::size_t i=0;i<extent;++i){a[i]=double(int(i%31)-15)/64;b[i]=double(int(i%17)-8)/32;expected[i]=b[i]+1.0/32;}
 const auto initial_a=a;
 reference_bridge(a.data(),expected.data(),n,t);
 fort_scope_domain_t domain=0;fort_scope_t context=0;fort_buffer_t ha=0,hb=0;
 check(fort_scope_domain_create_v1(0,&domain));check(fort_scope_create_in_domain_v1(domain,&context));
 check(fort_scope_set_device_budget(context,64*1024*1024));
 fort_scope_layout la{1,FORT_SCOPE_REAL64,8,a.data(),&extent,&lb,1},bl{1,FORT_SCOPE_REAL64,8,b.data(),&extent,&lb,1};
 check(fort_scope_register(context,1,1,&la,1,&ha));check(fort_scope_register(context,2,1,&bl,1,&hb));
 fort_scope_access read{};read.flags=FORT_SCOPE_READ_ALL;void* pointer=nullptr;
 check(fort_scope_device_begin(context,ha,&read,&pointer));check(fort_scope_device_end(context,ha));
 fort_scope_access rw{};rw.flags=FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL;
 fort_scope_external_binding_v1 binding{hb,FORT_SCOPE_REAL64,1,8,rw};fort_scope_external_call_v1 call{1,0,&binding,1,0,0};
 fort_scope_external_report_v1 report{};check(fort_scope_source_execute_v1(context,&call,1,submit,nullptr,&report));
 assert(report.started==1&&report.committed==1);
 const std::size_t plo=3,phi=4;fort_scope_section point{&plo,&phi};fort_scope_access probe{0,1,&point,0,nullptr,0,nullptr};
 check(fort_scope_host_begin(context,hb,&probe));check(fort_scope_host_end(context,hb));
 const auto before=stats(context);assert(before.uploads==2&&before.upload_bytes==2*extent*8&&before.download_bytes==8);
 std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const int flags=std::fetestexcept(FE_ALL_EXCEPT);
 std::fputs("WARM_CASE_BEGIN\n",stderr);
 report={};check(@ENTRY@(context,ha,hb,@ARGUMENTS@,&report));
 assert(std::fetestexcept(FE_ALL_EXCEPT)==flags);
 std::fputs("WARM_CASE_END\n",stderr);
 assert(report.started==1&&report.committed==1);
 const auto after=stats(context);assert(after.launches==before.launches+1);
 assert(after.uploads==before.uploads&&after.downloads==before.downloads);
 check(fort_scope_host_begin(context,hb,&read));check(fort_scope_host_end(context,hb));
 assert(a==initial_a&&b==expected);
 const auto published=stats(context);assert(published.download_bytes==8+extent*8);
 check(fort_scope_close(context));check(fort_scope_domain_close_v1(domain));
 // A fresh domain in this process retains the observed per-device startup
 // knowledge, without pretending that its own stream/replicas already exist.
 n=7;const std::size_t small_extent=n+8;
 la.extents=bl.extents=&small_extent;
 check(fort_scope_domain_create_v1(0,&domain));check(fort_scope_create_in_domain_v1(domain,&context));
 check(fort_scope_register(context,1,1,&la,1,&ha));check(fort_scope_register(context,2,1,&bl,1,&hb));
 std::fputs("WARM_PROCESS_FRESH_DOMAIN_BEGIN\n",stderr);
 report={};const int declined=@ENTRY@(context,ha,hb,@ARGUMENTS@,&report);
 assert(std::fetestexcept(FE_ALL_EXCEPT)==flags);
 std::fputs("WARM_PROCESS_FRESH_DOMAIN_END\n",stderr);
 assert(declined==FORT_SCOPE_BOUNDARY&&report.started==0&&report.committed==0);
 assert(stats(context).uploads==0&&stats(context).launches==0);
 check(fort_scope_close(context));check(fort_scope_domain_close_v1(domain));
 std::puts("MEASURED_RESIDENT_GPU_FULL_FIELDS_AND_PROCESS_WARMTH_OK");
}
'''.replace("@ENTRY@", provider["entry"]).replace("@PROTOTYPE@", prototype).replace("@ARGUMENTS@", arguments))
    bridge = build / "reference_bridge.f90"
    bridge.write_text("""subroutine reference_bridge(a,b,n,t)bind(C)
use iso_c_binding
use reference_stencil
implicit none
integer(c_int),value::n,t
real(c_double),intent(in)::a(-3:n+4)
real(c_double),intent(inout)::b(-3:n+4)
call advance(a,b,n,t)
end subroutine
""")
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    warm = build / "warm"
    runtime_objects = [str(build / f"object-{i}.o") for i, a in enumerate(manifest["build_sources"])
                       if a["role"] in {"common_runtime", "shared_entry", "original_source"}]
    commands = [[nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", "-I", str(build),
                 "-c", str(source), "-o", str(build / "warm.o")],
                [fortran, "-fopenmp", "-ffp-contract=off", "-J", str(build), "-I", str(build),
                 "-c", str(build / "reference.f90"), "-o", str(build / "reference.o")],
                [fortran, "-fopenmp", "-ffp-contract=off", "-J", str(build), "-I", str(build),
                 "-c", str(bridge), "-o", str(build / "bridge.o")],
                [fortran, "-fopenmp", str(build / "warm.o"), str(build / "reference.o"), str(build / "bridge.o"),
                 *runtime_objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(warm)]]
    for i, argv in enumerate(commands):
        (build / f"warm-compile-{i}.command.json").write_text(json.dumps(argv) + "\n")
        result = run(argv, cwd=build)
        (build / f"warm-compile-{i}.stdout").write_text(result.stdout)
        (build / f"warm-compile-{i}.stderr").write_text(result.stderr)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "OMP_THREAD_LIMIT": "4", "OMP_MAX_ACTIVE_LEVELS": "1", "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    result = run([str(warm)], cwd=build, env=environment)
    (build / "warm.stdout").write_text(result.stdout)
    (build / "warm.stderr").write_text(result.stderr)
    assert "MEASURED_RESIDENT_GPU_FULL_FIELDS_AND_PROCESS_WARMTH_OK" in result.stdout
    resident = result.stderr.split("WARM_CASE_BEGIN\n", 1)[1].split("WARM_CASE_END\n", 1)[0]
    assert re.search(r"FORT_SOURCE_QUOTE .*available=1 gpu=1 reason=0", resident), resident
    fresh = result.stderr.split("WARM_PROCESS_FRESH_DOMAIN_BEGIN\n", 1)[1].split("WARM_PROCESS_FRESH_DOMAIN_END\n", 1)[0]
    quote, = re.findall(r"FORT_SOURCE_QUOTE .*available=1 gpu=0 reason=0.*gpu_seconds=([0-9.e+-]+)", fresh)
    assert float(quote) < .1  # The actual measured cold startup is ~.22s.
