"""Real GPU/native/GPU chains retain readonly and exact partial replicas."""

import json
import os
import shutil
import subprocess

import pytest

from compiler.numerical_contract import cuda_compile_options
from compiler.tests.test_scoped_runtime import RUNTIME

CLIENT = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
void check(int s){if(s){std::fprintf(stderr,"%d %s\n",s,fort_scope_error());std::abort();}}
__global__ void numerical(double* b,std::size_t n,int operation){
 const auto i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>0&&i+1<n){if(operation==0)b[i]+=10;else b[i]*=2;}
}
struct State{int operation;};
int submit(void* raw,const fort_scope_external_view_v1* views,std::size_t count,void* stream,int* started,uint64_t* accepted){
 *started=0;*accepted=0;assert(count==1);*started=1;
 numerical<<<(views[0].extents[0]+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
   static_cast<double*>(views[0].device),views[0].extents[0],static_cast<State*>(raw)->operation);
 if(cudaGetLastError()!=cudaSuccess)return FORT_SCOPE_EXECUTION;*accepted=1;return 0;
}
fort_scope_stats stats(fort_scope_t ctx){fort_scope_stats s{};check(fort_scope_stats_get(ctx,&s));return s;}
int main(int argc,char**argv){
 assert(argc==3);const bool writer=!std::strcmp(argv[1],"writer"),borrowed=!std::strcmp(argv[2],"borrowed");
 const std::size_t n=4099,lo=1,hi=n-1;const std::int64_t lower=-3;
 std::vector<double> host(n);for(std::size_t i=0;i<n;++i)host[i]=i;
 fort_scope_domain_t domain=0;fort_scope_t ctx=0;
 if(borrowed){check(fort_scope_domain_create_v1(0,&domain));check(fort_scope_create_in_domain_v1(domain,&ctx));}
 else check(fort_scope_create(0,&ctx));
 fort_buffer_t buffer=0;fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host.data(),&n,&lower,1};
 check(fort_scope_register(ctx,1,1,&layout,1,&buffer));
 fort_scope_section interior{&lo,&hi};fort_scope_access rw{0,1,&interior,1,&interior,0,nullptr};
 fort_scope_external_binding_v1 binding{buffer,FORT_SCOPE_REAL64,1,8,rw};
 fort_scope_external_call_v1 call{1,0,&binding,1,0,0};fort_scope_external_report_v1 report{};State state{0};
 check(fort_scope_source_execute_v1(ctx,&call,1,submit,&state,&report));assert(report.committed==1);
 const std::size_t part_lo=3,part_hi=8;fort_scope_section part{&part_lo,&part_hi};
 fort_scope_access native_access=writer?fort_scope_access{0,1,&part,1,&part,0,nullptr}:
                                      fort_scope_access{0,1,&interior,0,nullptr,0,nullptr};
 fort_scope_external_binding_v1 native_binding{buffer,FORT_SCOPE_REAL64,1,8,native_access};
 fort_scope_external_call_v1 native_call{1,0,&native_binding,1,0,0};uint64_t token=0;
 check(fort_scope_source_host_prepare_v1(ctx,&native_call,&token));assert(token);
 const auto prepared=stats(ctx);assert(prepared.upload_bytes==(n-2)*8);
 assert(prepared.download_bytes==(writer?40:(n-2)*8));
 check(fort_scope_source_host_start_v1(ctx,token));
 if(writer){for(std::size_t i=part_lo;i<part_hi;++i)host[i]*=3;}
 else{volatile double observation=0;for(unsigned repeat=0;repeat<32;++repeat)
   for(std::size_t i=lo;i<hi;++i)observation+=host[i]*double(repeat+1);
  double expected=0;for(std::size_t i=lo;i<hi;++i)expected+=(double(i)+10)*528;
  assert(observation==expected);}
 check(fort_scope_source_host_complete_v1(ctx,token));
 state.operation=1;report={};check(fort_scope_source_execute_v1(ctx,&call,1,submit,&state,&report));assert(report.committed==1);
 const auto continued=stats(ctx);assert(continued.allocations==prepared.allocations);
 assert(continued.upload_bytes==prepared.upload_bytes+(writer?40:0));
 assert(continued.download_bytes==prepared.download_bytes);
 fort_scope_access read{};read.flags=FORT_SCOPE_READ_ALL;
 check(fort_scope_host_begin(ctx,buffer,&read));check(fort_scope_host_end(ctx,buffer));
 for(std::size_t i=0;i<n;++i){double expected=i;
  if(i>0&&i+1<n){expected+=10;if(writer&&i>=part_lo&&i<part_hi)expected*=3;expected*=2;}
  assert(host[i]==expected);}
 const auto final=stats(ctx);assert(final.download_bytes==prepared.download_bytes+(n-2)*8);
 assert(final.launches==2);
 check(fort_scope_close(ctx));if(borrowed)check(fort_scope_domain_close_v1(domain));
 std::printf("NATIVE_LIVE_GPU_CHAIN_FULL_FIELDS_OK writer=%d borrowed=%d h2d=%llu d2h=%llu\n",writer,borrowed,
  static_cast<unsigned long long>(final.upload_bytes),static_cast<unsigned long long>(final.download_bytes));
}
'''


@pytest.fixture(scope="module")
def native_live_cuda_binary(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit and supported GNU host compiler unavailable")
    directory = tmp_path_factory.mktemp("source_native_live_cuda")
    source, executable = directory / "client.cu", directory / "run"
    source.write_text(CLIENT)
    argv = [nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", "-I", str(RUNTIME),
            *cuda_compile_options(),
            str(RUNTIME / "scoped_runtime.cu"), str(source), "-o", str(executable)]
    (directory / "command.json").write_text(json.dumps(argv) + "\n")
    result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    (directory / "build.stdout").write_text(result.stdout)
    (directory / "build.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["readonly", "writer"])
@pytest.mark.parametrize("ownership", ["private", "borrowed"])
def test_original_native_work_retains_exact_live_replicas(native_live_cuda_binary, mode, ownership):
    environment = {**os.environ, "FORT_RUNTIME_TRACE": "1", "OMP_NUM_THREADS": "4"}
    result = subprocess.run([str(native_live_cuda_binary), mode, ownership], env=environment,
                            capture_output=True, text=True, timeout=30)
    prefix = native_live_cuda_binary.parent / (ownership + "-" + mode)
    prefix.with_suffix(".stdout").write_text(result.stdout)
    prefix.with_suffix(".stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_LIVE_GPU_CHAIN_FULL_FIELDS_OK" in result.stdout
