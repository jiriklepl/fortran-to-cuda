"""Actual pending work and hidden-device metadata-only source quotes."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_runtime import RUNTIME

CLIENT = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstring>
static unsigned selected_devices=0;
extern "C" cudaError_t __real_cudaSetDevice(int);
extern "C" cudaError_t __wrap_cudaSetDevice(int device) {++selected_devices;return __real_cudaSetDevice(device);}
void check(int status){if(status){std::fprintf(stderr,"%d: %s\n",status,fort_scope_error());std::abort();}}
void cuda(cudaError_t status){if(status!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::abort();}}
fort_scope_plan_costs base(){
 fort_scope_plan_costs c{};c.version=2;c.valid=1;c.max_allocation_bytes=1<<20;
 c.cpu_flops=c.cpu_bandwidth=1e8;c.gpu_flops=c.gpu_bandwidth=1e12;
 c.h2d_bandwidth=c.d2h_bandwidth=1e12;c.h2d_latency=c.d2h_latency=1e-6;
 c.create_seconds=c.register_seconds=c.host_access_seconds=c.device_access_seconds=1e-6;
 c.gpu_setup_seconds=c.cold_driver_startup_seconds=c.allocation_seconds=c.release_seconds=1e-6;
 c.wait_seconds=c.launch_enqueue_seconds=c.planning_operation_seconds=1e-6;return c;
}
fort_scope_source_quote_result_v1 quote(fort_scope_t ctx,fort_buffer_t buffer,fort_scope_access access,
                                      unsigned flags=FORT_SCOPE_SOURCE_QUOTE_NATIVE_CLOSE_DOMAIN,size_t scratch=0){
 fort_scope_external_binding_v1 binding{buffer,FORT_SCOPE_REAL64,1,8,access};
 fort_scope_external_call_v1 call{1,0,&binding,1,0,0};
 fort_scope_source_work_v1 work{1,64,flags,0,2,1000,128,scratch,.05,{2,5,1,0,.01}};
 fort_scope_source_costs_v1 rates{1,1,64,0,FORT_SCOPE_SOURCE_PROTOCOL_ID,8,256,1<<20,.001,.002,.003,.004,.005};
 auto costs=base();fort_scope_source_quote_result_v1 result{};
 check(fort_scope_source_quote_v1(ctx,&call,&work,&costs,&rates,1,&result));return result;
}
__global__ void increment(double* data){if(threadIdx.x>0&&threadIdx.x<9)data[threadIdx.x]+=1;}
int submit(void*,const fort_scope_external_view_v1* views,size_t count,void* raw,int* started,uint64_t* accepted){
 assert(count==1);*started=0;*accepted=0;
 for(unsigned i=0;i<2;++i){*started=1;increment<<<1,32,0,static_cast<cudaStream_t>(raw)>>>(static_cast<double*>(views[0].device));
  if(cudaGetLastError()!=cudaSuccess)return FORT_SCOPE_EXECUTION;++*accepted;}return FORT_SCOPE_OK;
}
fort_scope_stats stats(fort_scope_t ctx){fort_scope_stats result{};check(fort_scope_stats_get(ctx,&result));return result;}
int main(int argc,char**argv){
 assert(argc==2);fort_scope_domain_t domain=0;fort_scope_t ctx=0;
 check(fort_scope_domain_create_v1(0,&domain));check(fort_scope_create_in_domain_v1(domain,&ctx));
 check(fort_scope_set_device_budget(ctx,256));check(fort_scope_task_graph_enable_v1(ctx));
 double host[10];for(unsigned i=0;i<10;++i)host[i]=i;
 size_t extents[]={10},lo[]={1},hi[]={9};int64_t lower[]={-4};
 fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host,extents,lower,7};fort_buffer_t buffer=0;
 check(fort_scope_register(ctx,1,7,&layout,1,&buffer));
 fort_scope_section interior{lo,hi};fort_scope_access rw{0,1,&interior,1,&interior,0,nullptr};
 auto fresh=quote(ctx,buffer,rw);assert(fresh.available&&fresh.gpu.upload_bytes==64&&fresh.gpu_terminal.download_bytes==64);
 assert(!selected_devices);assert(!stats(ctx).allocations);check(fort_scope_set_device_budget(ctx,256));
 if(!std::strcmp(argv[1],"hidden")){check(fort_scope_close(ctx));check(fort_scope_domain_close_v1(domain));
  assert(!selected_devices);std::puts("HIDDEN_METADATA_QUOTE_OK");return 0;}
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
 fort_scope_scratch_lease_v1 scratch{};check(fort_scope_scratch_acquire_v1(ctx,64,&scratch));
 check(fort_scope_scratch_release_v1(ctx,scratch.token));
 fort_scope_external_binding_v1 binding{buffer,FORT_SCOPE_REAL64,1,8,rw};
 fort_scope_external_call_v1 call{1,0,&binding,1,0,0};fort_scope_external_report_v1 report{};
 check(fort_scope_source_execute_v1(ctx,&call,2,submit,nullptr,&report));assert(report.committed);
 const unsigned fence=FORT_SCOPE_SOURCE_QUOTE_NATIVE_CLOSE_DOMAIN|FORT_SCOPE_SOURCE_QUOTE_SCRATCH_LEASE|
                      FORT_SCOPE_SOURCE_QUOTE_WAIT_BEFORE_LEASE;
 auto pending=quote(ctx,buffer,rw,fence,128);assert(pending.available&&pending.gpu.waits==1&&pending.gpu_terminal.waits==1);
 size_t probe_lo[]={3},probe_hi[]={4};fort_scope_section point{probe_lo,probe_hi};
 fort_scope_access probe{0,1,&point,0,nullptr,0,nullptr};
 check(fort_scope_host_begin(ctx,buffer,&probe));check(fort_scope_host_end(ctx,buffer));assert(host[3]==5);
 fort_scope_access read{0,1,&interior,0,nullptr,0,nullptr};
 const auto before=stats(ctx);const auto device_calls=selected_devices;auto resident=quote(ctx,buffer,read);
 const auto after=stats(ctx);assert(!std::memcmp(&before,&after,sizeof(before)));assert(selected_devices==device_calls);
 assert(resident.available&&!resident.gpu.upload_bytes&&!resident.gpu.download_bytes);
 assert(resident.native_terminal.download_bytes==56&&resident.gpu_terminal.download_bytes==56);
 if(!std::strcmp(argv[1],"queued")){
  void* pointer=nullptr;check(fort_scope_device_reserve_v1(ctx,buffer,&pointer));
  void* arguments[]={&pointer};size_t widths[]={sizeof(pointer)};
  fort_scope_plan_binding effect{buffer,rw};fort_scope_graph_task_v1 task{
   {1,0,reinterpret_cast<const void*>(increment),{1,1,1},{32,1,1},0,arguments,1},&effect,1};
  check(fort_scope_graph_enqueue_v1(ctx,9,&task,widths));
  fort_scope_graph_queue_stats_v1 queue_before{},queue_after{};
  check(fort_scope_graph_queue_stats_get_v1(ctx,&queue_before));assert(queue_before.pending_tasks==1);
  auto declined=quote(ctx,buffer,rw);assert(!declined.available&&declined.reason==FORT_SCOPE_SOURCE_QUOTE_LIVE_STATE);
  check(fort_scope_graph_queue_stats_get_v1(ctx,&queue_after));assert(!std::memcmp(&queue_before,&queue_after,sizeof(queue_before)));
 }
 check(fort_scope_close(ctx));check(fort_scope_domain_close_v1(domain));
 const double added=!std::strcmp(argv[1],"queued")?3:2;
 for(unsigned i=0;i<10;++i)assert(host[i]==i+((i>0&&i<9)?added:0));
 std::printf("LIVE_QUOTE_OK mode=%s local_uploads=0 terminal_bytes=56\n",argv[1]);return 0;
}
'''


@pytest.fixture(scope="module")
def quote_cuda_binary(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit and supported GNU host compiler unavailable")
    directory = tmp_path_factory.mktemp("source_quote_cuda")
    source = directory / "client.cu"
    source.write_text(CLIENT)
    target = directory / "run"
    argv = [nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", "-I", str(RUNTIME),
            str(RUNTIME / "scoped_runtime.cu"), str(source), "-Xlinker=--wrap=cudaSetDevice", "-o", str(target)]
    (directory / "build-command.json").write_text(__import__("json").dumps(argv, indent=2))
    result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    (directory / "build.stdout").write_text(result.stdout)
    (directory / "build.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return target


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["hidden", "resident", "queued"])
def test_actual_source_quote_does_not_submit_publish_or_initialize(quote_cuda_binary, mode):
    environment = {**os.environ, "FORT_RUNTIME_TRACE": "1", "FORT_SCOPE_GRAPH_QUEUE": "1"}
    if mode == "hidden":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = subprocess.run([str(quote_cuda_binary), mode], env=environment,
                            capture_output=True, text=True, timeout=30)
    (quote_cuda_binary.parent / (mode + ".stdout")).write_text(result.stdout)
    (quote_cuda_binary.parent / (mode + ".stderr")).write_text(result.stderr)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "QUOTE_OK" in result.stdout
