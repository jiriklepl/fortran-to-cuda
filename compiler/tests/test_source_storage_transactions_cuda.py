"""Physical byte transport retains conditional CUDA reads and native replicas."""
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
void check(int s){if(s){std::fprintf(stderr,"%d %s\n",s,fort_scope_error());std::abort();}}
__global__ void original(double* data,std::size_t n,int phase){
 const auto i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>0&&i+1<n&&i%2==0){if(phase==0)data[i]+=10;else data[i]*=2;}
}
int submit(void* raw,const fort_scope_external_view_v1* views,std::size_t count,void* stream,
           int* started,uint64_t* accepted){
 *started=0;*accepted=0;assert(count==1);*started=1;
 original<<<(views[0].extents[0]+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
  static_cast<double*>(views[0].device),views[0].extents[0],*static_cast<int*>(raw));
 if(cudaGetLastError()!=cudaSuccess)return FORT_SCOPE_EXECUTION;
 *accepted=1;return FORT_SCOPE_OK;
}
fort_scope_stats stats(fort_scope_t ctx){fort_scope_stats s{};check(fort_scope_stats_get(ctx,&s));return s;}
uint64_t bits(double value){uint64_t out;std::memcpy(&out,&value,8);return out;}
int main(int argc,char**argv){
 assert(argc==3);const bool writer=!std::strcmp(argv[1],"writer"),borrowed=!std::strcmp(argv[2],"borrowed");
 const std::size_t n=1031,lo=1,hi=n-1;const int64_t lower=-7;
 std::vector<double> host(n);std::vector<uint64_t> expected(n);
 for(std::size_t i=0;i<n;++i){
  expected[i]=(i%2==0&&i>0&&i+1<n)?bits(double(i)):0x7ff0000000000001ULL+i;
  std::memcpy(&host[i],&expected[i],8);
 }
 fort_scope_domain_t domain=0;fort_scope_t ctx=0;
 if(borrowed){check(fort_scope_domain_create_v1(0,&domain));check(fort_scope_create_in_domain_v1(domain,&ctx));}
 else check(fort_scope_create(0,&ctx));
 fort_buffer_t buffer=0;fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host.data(),&n,&lower,1};
 check(fort_scope_register(ctx,1,1,&layout,0,&buffer));
 fort_scope_section interior{&lo,&hi};fort_scope_access rw{0,1,&interior,1,&interior,0,nullptr};
 fort_scope_external_binding_v1 binding{buffer,FORT_SCOPE_REAL64,1,8,rw};
 fort_scope_external_call_v1 call{1,0,&binding,1,0,0};fort_scope_external_report_v1 report{};int phase=0;
 check(fort_scope_source_execute_v2(ctx,&call,1,submit,&phase,&report));assert(report.started&&report.committed);
 assert(stats(ctx).upload_bytes==(n-2)*8&&stats(ctx).download_bytes==0);
 // The v1 source and numerical APIs remain strict even though storage exists.
 assert(fort_scope_source_execute_v1(ctx,&call,1,submit,&phase,&report)==FORT_SCOPE_UNINITIALIZED);
 void* ptr=nullptr;assert(fort_scope_device_begin(ctx,buffer,&rw,&ptr)==FORT_SCOPE_UNINITIALIZED);
 const std::size_t rlo=2,rhi=7,wlo=4,whi=5;
 fort_scope_section probe{&rlo,&rhi},changed{&wlo,&whi};
 fort_scope_access host_access{0,1,&probe,writer?1u:0u,writer?&changed:nullptr,0,nullptr};
 fort_scope_plan_binding native{buffer,host_access};
 check(fort_scope_original_host_begin_v1(ctx,&native,1));
 assert(host[2]==12&&host[4]==14&&host[6]==16);
 if(writer)host[4]+=7;
 check(fort_scope_host_end(ctx,buffer));
 assert(stats(ctx).download_bytes==40);
 phase=1;check(fort_scope_source_execute_v2(ctx,&call,1,submit,&phase,&report));
 assert(report.started&&report.committed);
 auto final=stats(ctx);assert(final.launches==2&&final.upload_bytes==(n-2)*8+(writer?8:0));
 assert(final.download_bytes==40);
 check(fort_scope_observe_all_v1(ctx));
 assert(stats(ctx).download_bytes==40+(n-2)*8);
 check(fort_scope_observe_v1(ctx,&buffer,1));
 assert(stats(ctx).download_bytes==40+(n-2)*8);
 assert(fort_scope_device_begin(ctx,buffer,&rw,&ptr)==FORT_SCOPE_UNINITIALIZED);
 // An ordinary publication must include representations, not just definitions.
 check(fort_scope_close(ctx));if(borrowed)check(fort_scope_domain_close_v1(domain));
 for(std::size_t i=0;i<n;++i){
  if(i>0&&i+1<n&&i%2==0)expected[i]=bits((double(i)+10+(writer&&i==4?7:0))*2);
  uint64_t actual;std::memcpy(&actual,&host[i],8);assert(actual==expected[i]);
 }
 std::printf("SOURCE_STORAGE_BYTES_OK writer=%d borrowed=%d uploads=%llu intermediate_downloads=%llu\n",
  writer,borrowed,static_cast<unsigned long long>(final.upload_bytes),static_cast<unsigned long long>(final.download_bytes));
}
'''


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit and supported GNU compiler unavailable")
    directory = tmp_path_factory.mktemp("source_storage_cuda")
    source, executable = directory / "client.cu", directory / "run"
    source.write_text(CLIENT)
    argv = [nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", "-I", str(RUNTIME),
            *cuda_compile_options(), str(RUNTIME / "scoped_runtime.cu"), str(source), "-o", str(executable)]
    (directory / "command.json").write_text(json.dumps(argv) + "\n")
    result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    (directory / "build.stdout").write_text(result.stdout)
    (directory / "build.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["readonly", "writer"])
@pytest.mark.parametrize("ownership", ["private", "borrowed"])
def test_original_cuda_and_native_keep_partial_replicas_and_unknown_holes(binary, mode, ownership):
    result = subprocess.run([str(binary), mode, ownership], capture_output=True, text=True, timeout=30,
                            env={**os.environ, "FORT_RUNTIME_TRACE": "1", "OMP_NUM_THREADS": "4"})
    prefix = binary.parent / (ownership + "-" + mode)
    prefix.with_suffix(".stdout").write_text(result.stdout)
    prefix.with_suffix(".stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SOURCE_STORAGE_BYTES_OK" in result.stdout
