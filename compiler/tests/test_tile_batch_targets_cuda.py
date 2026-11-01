"""Finite batch targets share the real source transaction and Fortran oracle."""
import json
import os
import re
import resource
import shutil
import subprocess

import pytest

from compiler.emission.cuda.tile_task_provider import EXPERIMENTAL_BATCH_TARGET_BYTES, emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.tests.test_tile_private_cuda import RUNTIME, checked
from compiler.tests.test_tile_private_emission import original_source
from compiler.tests.test_tile_private_team import team_fixture


def compile_checked(command, directory, label):
    with (directory / "build-commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"role": label, "command": command}) + "\n")
    return checked(command, directory, label)


def original_oracle_stack():
    """OMP_STACKSIZE covers workers; the original master's VLA needs this too."""
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    requested = 256 * 1024 * 1024
    if hard != resource.RLIM_INFINITY and hard < requested:
        raise RuntimeError("original large automatic scratch requires a 256 MiB main stack allowance")
    if soft != resource.RLIM_INFINITY and soft < requested:
        resource.setrlimit(resource.RLIMIT_STACK, (requested, hard))


@pytest.fixture(scope="module")
def batch_target_executable(tmp_path_factory):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not all((nvcc, host, fc)):
        pytest.skip("CUDA and GNU toolchains required")
    directory = tmp_path_factory.mktemp("batch-target-source")
    source = original_source(8, branch=True).replace("advance(a,b,n,t)", "advance(a,b,n,t,pad)")
    source = source.replace("integer,intent(in)::n,t", "integer,intent(in)::n,t,pad")
    source = source.replace("tmp(-1:t+2)", "tmp(-1:t+2+pad)")
    first = source.index("!$omp do ")
    last = source.index("!$omp end do", first) + len("!$omp end do\n")
    source = source[:last] + source[first:last] + source[last:]
    _, analysis, routine, nodes = team_fixture(directory, source)
    proof = source_device_task(analysis, routine, nodes)
    assert len(proof.team.units) == 2
    providers = [emit_tile_task_provider(proof.team, readable_scalars=proof, access_envelopes=True,
                                       batch_target_bytes=target) for target in EXPERIMENTAL_BATCH_TARGET_BYTES]
    (directory / "source-proof.json").write_text(json.dumps(proof.public(), indent=2))
    prototypes, dispatches = [], []
    objects = []
    includes = ["-I", str(RUNTIME), "-I", str(RUNTIME.parents[0] / "emission/common/templates")]
    for index, provider in enumerate(providers):
        (directory / (provider.entry + ".cu")).write_text(provider.cuda_source)
        (directory / (provider.module + ".f90")).write_text(provider.fortran_source)
        (directory / (provider.entry + ".json")).write_text(json.dumps(provider.public(), indent=2))
        values = {("scalar", "n", None): "&n", ("scalar", "t", None): "&t", ("scalar", "pad", None): "&pad",
                  ("external_lower", "a", 1): "n?&lower:nullptr", ("external_lower", "b", 1): "n?&lower:nullptr",
                  ("scratch_lower", "tmp", 1): "n?&scratch_lower:nullptr", ("scratch_upper", "tmp", 1): "n?&scratch_upper:nullptr"}
        parameters = ",".join(values[p.role, p.binding.name, p.axis] for p in provider.parameters)
        prototype = ",".join(["fort_scope_t", "fort_buffer_t", "fort_buffer_t",
                              *["const int*" for _ in provider.parameters], "fort_scope_external_report_v1*"])
        prototypes.append(f'extern "C" int {provider.entry}({prototype});')
        dispatches.append(f"case {index}:return {provider.entry}(context,ha,hb,{parameters},report);")
        target = directory / f"provider-{index}.o"
        compile_checked([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
                 "--prec-div=true", "--prec-sqrt=true", *includes, "-c", str(directory / (provider.entry + ".cu")),
                 "-o", str(target)], directory, f"provider-{index}")
        objects.append(str(target))
    (directory / "bridge.f90").write_text('''subroutine original_bridge(a,b,n,t,pad)bind(C)
use iso_c_binding
use stencil
implicit none
integer(c_int),value::n,t,pad
real(c_double),intent(in)::a(-3:n+4)
real(c_double),intent(inout)::b(-3:n+4)
call advance(a,b,n,t,pad)
end subroutine
''')
    compile_checked([fc, "-O2", "-fopenmp", "-ffp-contract=off", "-J", str(directory), "-c", str(directory / "source.f90"),
             "-o", str(directory / "original.o")], directory, "original")
    compile_checked([fc, "-O2", "-fopenmp", "-ffp-contract=off", "-I", str(directory), "-c", str(directory / "bridge.f90"),
             "-o", str(directory / "bridge.o")], directory, "bridge")
    compile_checked([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp", "-DFORT_SCOPE_TEST_FAULTS",
             "-I", str(RUNTIME), "-c", str(RUNTIME / "scoped_runtime.cu"), "-o", str(directory / "runtime.o")], directory, "runtime")
    driver = r'''#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
PROTOTYPES
extern "C" void original_bridge(const double*,double*,int,int,int);
static int fault_launch=0,attempted=0,native_calls=0;
extern "C" cudaError_t __real___cudaLaunchKernel(cudaKernel_t,dim3,dim3,void**,std::size_t,cudaStream_t);
extern "C" cudaError_t __wrap___cudaLaunchKernel(cudaKernel_t f,dim3 grid,dim3 block,void**args,std::size_t bytes,cudaStream_t stream){
 ++attempted;if(fault_launch&&attempted==fault_launch)grid.x=0;
 return __real___cudaLaunchKernel(f,grid,block,args,bytes,stream);
}
static void check(int s){if(s){std::fprintf(stderr,"status=%d %s\n",s,fort_scope_error());std::abort();}}
static int call(int index,fort_scope_t context,fort_buffer_t ha,fort_buffer_t hb,int n,int pad,int upper_delta=0,fort_scope_external_report_v1*report=nullptr){
 const int t=3,lower=-3,scratch_lower=-1,scratch_upper=upper_delta?t+1:t+2+pad;
 switch(index){DISPATCHES}std::abort();
}
static void native(const std::vector<double>&a,std::vector<double>&b,int n,int pad){++native_calls;original_bridge(a.data(),b.data(),n,3,pad);}
static void ledger(int index,const char*mode,int shape,int n,int pad,const fort_scope_stats&s,const fort_scope_scratch_stats_v1&ss){
 std::printf("BATCH_TARGET_LEDGER {\"target\":%llu,\"mode\":\"%s\",\"shape\":%d,\"n\":%d,\"pad\":%d,\"native_calls\":%d,\"attempts\":%d,\"stats_available\":%s,\"launches\":%llu,\"uploads\":%llu,\"upload_bytes\":%llu,\"downloads\":%llu,\"download_bytes\":%llu,\"scratch_capacity\":%llu,\"scratch_peak\":%llu,\"scratch_acquisitions\":%llu,\"scratch_releases\":%llu}\n",
 static_cast<unsigned long long>(std::size_t(4194304)<<(index==0?0:index==1?2:4)),mode,shape,n,pad,native_calls,attempted,(!std::strcmp(mode,"fatal")&&n)?"false":"true",
 static_cast<unsigned long long>(s.launches),static_cast<unsigned long long>(s.uploads),static_cast<unsigned long long>(s.upload_bytes),static_cast<unsigned long long>(s.downloads),static_cast<unsigned long long>(s.download_bytes),static_cast<unsigned long long>(ss.capacity_bytes),static_cast<unsigned long long>(ss.peak_scratch_bytes),static_cast<unsigned long long>(ss.acquisitions),static_cast<unsigned long long>(ss.releases));
}
int main(int argc,char**argv){
 assert(argc==3);const int index=std::atoi(argv[1]);assert(index>=0&&index<3);const char*mode=argv[2];
 const bool hidden=!std::strcmp(mode,"hidden"),prework=!std::strcmp(mode,"prework"),fatal=!std::strcmp(mode,"fatal"),late=!std::strcmp(mode,"late");
 const std::size_t target=std::size_t(4194304)<<(index==0?0:index==1?2:4);
 for(int shape=0;shape<5;++shape)for(int reject=0;reject<(prework?2:1);++reject){
  const int n=shape==0?0:shape==1?1:shape==2?7:shape==3?864:7,pad=shape==4?int(target/8)+1:9713;
  const std::size_t extent=n+8,tiles=n?(n-1)/3+1:0,stride=(std::size_t(7+pad)*8+7)&~std::size_t(7),batch=std::min(tiles,std::max<std::size_t>(1,target/stride));
  const std::size_t one_launches=tiles?2*((tiles+batch-1)/batch):0,scratch_bytes=stride*batch;
  std::vector<double>a(extent),b(extent);for(int i=-3;i<=n+4;++i){a[i+3]=double(17*i*i-3*i+11)/64.;b[i+3]=double(500-i)/32.;}
  const auto initial_a=a,initial_b=b;auto expected=b;fort_scope_t context=0;fort_buffer_t ha=0,hb=0;
  if(n){check(fort_scope_create(0,&context));check(fort_scope_set_device_budget(context,prework&&reject==0?1:268435456));
   const std::int64_t lower=-3;fort_scope_layout la{1,FORT_SCOPE_REAL64,8,a.data(),&extent,&lower,1},lb{1,FORT_SCOPE_REAL64,8,b.data(),&extent,&lower,1};
   check(fort_scope_register(context,1,1,&la,1,&ha));check(fort_scope_register(context,2,1,&lb,1,&hb));}
  fault_launch=fatal?2:0;attempted=0;native_calls=0;fort_scope_external_report_v1 report{};
  std::fprintf(stderr,"BATCH_CASE_BEGIN %d %d %d\n",shape,reject,n);
  const int status=call(index,context,ha,hb,n,pad,prework&&reject==1?-1:0,&report);
  assert(report.version==1&&report.status==status);
  fort_scope_stats stats{};fort_scope_scratch_stats_v1 ss{};
  if(!n){assert(status==0&&report.started==0&&report.committed==0);assert(a==initial_a&&b==initial_b);ledger(index,mode,shape,n,pad,stats,ss);std::fprintf(stderr,"BATCH_CASE_END %d %d %d\n",shape,reject,n);continue;}
  if(fatal){assert(status==FORT_SCOPE_EXECUTION&&report.started==1&&report.committed==0&&attempted==2);
   assert(fort_scope_stats_get(context,&stats)==FORT_SCOPE_EXECUTION);assert(native_calls==0&&a==initial_a&&b==initial_b);
   assert(fort_scope_close(context)==FORT_SCOPE_EXECUTION);check(fort_scope_abandon(context));stats={};
   ledger(index,mode,shape,n,pad,stats,ss);std::fprintf(stderr,"BATCH_CASE_END %d %d %d\n",shape,reject,n);continue;}
  if(hidden||prework){assert(status==FORT_SCOPE_RESOURCE&&report.started==0&&report.committed==0&&attempted==0);
   check(fort_scope_stats_get(context,&stats));assert(stats.launches==0&&stats.uploads==0&&stats.downloads==0);
   native(a,b,n,pad);original_bridge(a.data(),expected.data(),n,3,pad);assert(native_calls==1);
  }else{
   check(status);assert(report.started==1&&report.committed==1);original_bridge(a.data(),expected.data(),n,3,pad);
   fort_scope_access read{};read.flags=FORT_SCOPE_READ_ALL;
   const auto before_a=a;check(fort_scope_host_begin(context,ha,&read));check(fort_scope_host_end(context,ha));assert(a==before_a);
   check(fort_scope_stats_get(context,&stats));assert(stats.downloads==0);
   if(late){
    const int grown=int(target/8)+17;const int prior_attempts=attempted;setenv("FORT_SCOPE_TEST_FAIL_SCRATCH_ALLOC","1",1);
    const int rejected=call(index,context,ha,hb,n,grown,0,&report);unsetenv("FORT_SCOPE_TEST_FAIL_SCRATCH_ALLOC");
    assert(rejected==FORT_SCOPE_RESOURCE&&report.started==0&&report.committed==0&&attempted==prior_attempts);
    std::size_t lo=4,hi=n+4;fort_scope_section box{&lo,&hi};fort_scope_access rw{0,1,&box,1,&box,0,nullptr};
    check(fort_scope_host_begin(context,ha,&read));check(fort_scope_host_begin(context,hb,&rw));native(a,b,n,grown);
    check(fort_scope_host_end(context,hb));check(fort_scope_host_end(context,ha));original_bridge(a.data(),expected.data(),n,3,grown);
    assert(native_calls==1);check(call(index,context,ha,hb,n,pad,0,&report));original_bridge(a.data(),expected.data(),n,3,pad);
   }else{
    check(call(index,context,ha,hb,n,pad,0,&report));original_bridge(a.data(),expected.data(),n,3,pad);
    const int logical=n>1?n/2:2;std::size_t lo=logical+3,hi=lo+1;fort_scope_section box{&lo,&hi};fort_scope_access write{0,0,nullptr,1,&box,0,nullptr};
    check(fort_scope_host_begin(context,ha,&write));a[lo]+=0.125;check(fort_scope_host_end(context,ha));
    check(call(index,context,ha,hb,n,pad,0,&report));original_bridge(a.data(),expected.data(),n,3,pad);assert(native_calls==0);
   }
   check(fort_scope_host_begin(context,hb,&read));check(fort_scope_host_end(context,hb));check(fort_scope_stats_get(context,&stats));
   assert(stats.launches==(late?2:3)*one_launches);assert(stats.uploads==3);assert(stats.upload_bytes==8*std::size_t(late?3*n+3:2*n+4));
   assert(stats.downloads==(late?2:1)&&stats.download_bytes==8*std::size_t(n)*(late?2:1));
   check(fort_scope_scratch_stats_get_v1(context,&ss));assert(ss.active==0&&ss.active_bytes==0);
   assert(ss.acquisitions==(late?2:3)&&ss.releases==ss.acquisitions&&ss.peak_scratch_bytes>=scratch_bytes&&ss.peak_total_device_bytes<=268435456);
  }
  check(fort_scope_close(context));assert(std::memcmp(b.data(),expected.data(),extent*8)==0);
  if(late||hidden||prework)assert(a==initial_a);
  else{auto changed=initial_a;changed[(n>1?n/2:2)+3]+=0.125;assert(a==changed);}
  ledger(index,mode,shape,n,pad,stats,ss);std::fprintf(stderr,"BATCH_CASE_END %d %d %d\n",shape,reject,n);
 }
 std::puts("FINITE_BATCH_TARGET_SOURCE_FIELDS_HALOS_COHERENCE_AND_FAILURES_OK");
}
'''.replace("PROTOTYPES", "\n".join(prototypes)).replace("DISPATCHES", "\n".join(dispatches))
    (directory / "driver.cpp").write_text(driver)
    compile_checked([nvcc, "-O2", "-std=c++17", "-ccbin", host, *includes, str(directory / "driver.cpp"), *objects,
             str(directory / "runtime.o"), str(directory / "original.o"), str(directory / "bridge.o"), "-lgfortran", "-lgomp",
             "-Xlinker=--wrap=__cudaLaunchKernel", "-o", str(directory / "run")], directory, "link")
    return directory / "run"


@pytest.mark.cuda
@pytest.mark.parametrize("target_index", range(3), ids=["4MiB", "16MiB", "64MiB"])
@pytest.mark.parametrize("mode", ["full", "hidden", "prework", "fatal", "late"])
def test_finite_batch_targets_preserve_source_fields_and_existing_coherence(batch_target_executable, target_index, mode):
    executable = batch_target_executable
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
           "OMP_STACKSIZE": "128M", "FORT_RUNTIME_TRACE": "1", "FORT_SCOPE_TRACE": "1"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if mode == "hidden":
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([str(executable), str(target_index), mode], capture_output=True, text=True,
                            env=env, timeout=30, preexec_fn=original_oracle_stack)
    (executable.parent / f"target-{target_index}-{mode}.stdout").write_text(result.stdout)
    (executable.parent / f"target-{target_index}-{mode}.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FINITE_BATCH_TARGET_SOURCE_FIELDS_HALOS_COHERENCE_AND_FAILURES_OK" in result.stdout
    rows = [json.loads(line.removeprefix("BATCH_TARGET_LEDGER ")) for line in result.stdout.splitlines()
            if line.startswith("BATCH_TARGET_LEDGER ")]
    assert len(rows) == (10 if mode == "prework" else 5)
    assert all(row["target"] == EXPERIMENTAL_BATCH_TARGET_BYTES[target_index] for row in rows)
    cases = re.findall(r"BATCH_CASE_BEGIN (\d+) (\d+) (\d+)\n(.*?)BATCH_CASE_END \1 \2 \3", result.stderr, re.S)
    assert len(cases) == len(rows)
    for (_shape, _reject, count, body), row in zip(cases, rows, strict=True):
        events = [line.split()[1] for line in body.splitlines() if line.startswith("FORT_SCOPED ")]
        physical = [event for event in events if event in {"upload", "download", "launch", "external_commit"}]
        if not int(count) or mode in {"hidden", "prework"}:
            assert not physical
        elif mode == "fatal":
            assert physical == ["upload", "upload", "launch"]
            assert row["stats_available"] is False
            assert row["attempts"] == 2
            assert row["native_calls"] == 0
        else:
            assert physical.count("launch") == row["launches"]
            assert physical.count("external_commit") == (2 if mode == "late" else 3)
            assert physical.count("upload") == 3
            assert physical.count("download") == (2 if mode == "late" else 1)
