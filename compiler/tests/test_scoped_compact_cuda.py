"""Real byte/32/64-bit copies through AUTO's calibrated physical backend."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
CLIENT = r'''
#include "scoped_runtime.h"
#include "staging.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
void check(int s){if(s){std::fprintf(stderr,"scope %d: %s\n",s,fort_scope_error());std::abort();}}
void ck(cudaError_t s){if(s!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));std::abort();}}
__global__ void fill(unsigned char *p,size_t n,unsigned *counter){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i==0)atomicAdd(counter,1u);if(i<n)p[i]=i%251+1;}
__global__ void increment(unsigned char *p,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)++p[i];}
void configure(fort_scope_t context,bool uncovered,bool single) {
    // Deliberately synthetic selector coefficients for this correctness test.
    // No observation or hardware profile is created from these values.
    fort_scope_plan_costs b{};b.version=FORT_SCOPE_PLANNING_ABI_VERSION;b.valid=1;b.max_allocation_bytes=1u<<28;
    b.cpu_flops=b.cpu_bandwidth=b.gpu_flops=b.gpu_bandwidth=b.h2d_bandwidth=b.d2h_bandwidth=1e12;
    b.create_seconds=b.register_seconds=b.host_access_seconds=b.device_access_seconds=1e-9;
    b.gpu_setup_seconds=b.cold_driver_startup_seconds=b.allocation_seconds=b.release_seconds=1e-9;
    b.wait_seconds=b.launch_enqueue_seconds=b.planning_operation_seconds=1e-9;
    fort_scope_copy_costs_v3 c{};c.version=FORT_SCOPE_COPY_COST_ABI_VERSION;c.valid=1;c.minimum_relative_saving=.5;
    for(unsigned d=0;d<2;++d)for(auto *rate:{&c.direct[d],&c.compact_cold[d],&c.compact_reuse[d]}) {
        const bool direct=rate==&c.direct[d];
        rate->submission_seconds=direct?1:0;rate->batch_seconds=direct?0:1e-9;
        rate->byte_seconds=direct?1e-9:1e-12;
        rate->minimum_payload_bytes=rate->minimum_rows=rate->minimum_row_bytes=1;
        rate->maximum_payload_bytes=rate->maximum_rows=rate->maximum_row_bytes=16u<<20;
        rate->minimum_pitch_bytes=1;rate->maximum_pitch_bytes=16u<<20;rate->maximum_span_bytes=1u<<30;
        rate->minimum_operations=direct||single?1:2;rate->maximum_operations=256;
        rate->minimum_descriptor_work=1;rate->maximum_descriptor_work=1u<<30;
        rate->minimum_kernel_chunks=1;rate->maximum_kernel_chunks=6;
        rate->word_bytes_mask=7;rate->layout_combinations_mask=254;
    }
    // Four360-byte operations form one complete1440-byte D2H batch.
    if(uncovered)for(auto &rate:c.direct)rate.minimum_payload_bytes=2048;
    check(fort_scope_set_transfers(context,FORT_SCOPE_TRANSFERS_AUTO));
    check(fort_scope_set_copy_costs_v3(context,&c,&b,1));
}
fort_scope_stats stats(fort_scope_t c){fort_scope_stats s{};check(fort_scope_stats_get(c,&s));return s;}
void queued(fort_scope_t c,fort_buffer_t b,unsigned char *device,size_t bytes,unsigned *counter,bool modify) {
    void *args[]={&device,&bytes,&counter};size_t sizes[]={sizeof(device),sizeof(bytes),sizeof(counter)};
    fort_scope_access access{};access.flags=modify?FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL:FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    fort_scope_plan_binding binding{b,access};
    const void *function=modify?reinterpret_cast<const void*>(increment):reinterpret_cast<const void*>(fill);
    fort_scope_graph_task_v1 task{{1,0,function,{unsigned((bytes+255)/256),1,1},{256,1,1},0,args,modify?2u:3u},&binding,1};
    check(fort_scope_graph_enqueue_v1(c,modify?2:1,&task,sizes));
    // Queued argument bytes are owned by the checked runtime.
    bytes=0;device=nullptr;counter=nullptr;
}
int main(int argc,char **argv) {
    assert(argc==2);const std::string mode=argv[1];int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    const size_t element=mode=="bytes"||mode=="single_bytes"?1:mode=="int32"||mode=="single_int32"?4:8;
    const bool single=mode=="oneop"||mode.rfind("single",0)==0;
    const bool queue=mode=="queued"||mode=="single_queued";
    const size_t x=element==8?34:35,y=9,z=5,bytes=x*y*z*element;
    std::vector<unsigned char> allocation(bytes+65,0xe7),got(bytes),original(bytes),expected(bytes);
    auto *host=allocation.data()+(element==1?1:32);
    for(size_t i=0;i<bytes;++i)original[i]=i%251+1;
    auto selected=[&](size_t i){const auto a=(i/element)%x;return a==1||(!single && (a==7||a==13||a==19));};
    fort_scope_t c=0;fort_buffer_t b=0;check(fort_scope_create(0,&c));configure(c,mode=="uncovered",single&&mode!="oneop");
    if(mode=="budget")check(fort_scope_set_device_budget(c,bytes));
    size_t shape[]={x,y,z};int64_t lower[]={-17,2,-9};
    fort_scope_layout layout{3,element==8?FORT_SCOPE_REAL64:element==4?FORT_SCOPE_INTEGER32:FORT_SCOPE_BYTES,
                             element,host,shape,lower,1};check(fort_scope_register(c,31,7,&layout,1,&b));
    int previous=0;void *raw=nullptr;check(fort_scope_gpu_enter(c,&previous,&raw));auto stream=static_cast<cudaStream_t>(raw);
    assert(stream!=nullptr);unsigned char *device=nullptr;unsigned *counter=nullptr;unsigned executions=0;
    ck(cudaMalloc(reinterpret_cast<void**>(&counter),sizeof(unsigned)));ck(cudaMemset(counter,0,sizeof(unsigned)));
    if(queue) {
        check(fort_scope_device_reserve_v1(c,b,reinterpret_cast<void**>(&device)));
        queued(c,b,device,bytes,counter,false);assert(!stats(c).launches);
    } else {
        fort_scope_access overwrite{};overwrite.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
        check(fort_scope_device_begin(c,b,&overwrite,reinterpret_cast<void**>(&device)));
        fill<<<(bytes+255)/256,256,0,stream>>>(device,bytes,counter);ck(cudaGetLastError());
        check(fort_scope_note_launch(c));check(fort_scope_device_end(c,b));
    }
    size_t lo[4][3],hi[4][3];fort_scope_section sections[4];
    for(size_t n=0;n<4;++n){lo[n][0]=1+6*n;lo[n][1]=lo[n][2]=0;hi[n][0]=lo[n][0]+1;hi[n][1]=y;hi[n][2]=z;sections[n]={lo[n],hi[n]};}
    const size_t section_count=single?1:4;
    fort_scope_access read{0,section_count,sections,0,nullptr,0,nullptr};
    if(mode=="oversized_reuse") {
        auto prime=fort_staging::acquire(4u<<20,fort_staging::Role::Compact,false);
        ck(prime.status);assert(prime.slots && prime.slots->capacity==(4u<<20));
        size_t freed=0;ck(fort_staging::release(std::move(prime.slots),freed));
        const auto idle=fort_staging::snapshot();
        assert(idle.idle_role==fort_staging::Role::Compact && idle.idle_capacity==(4u<<20));
        assert(idle.device_reserved==(8u<<20));
    }
    fort_staging::Result occupied;
    if(mode=="pool" || mode=="pinned_pool") {occupied=fort_staging::acquire(fort_staging::pinned_limit/2,mode=="pool"?fort_staging::Role::Compact:fort_staging::Role::Staging,false);assert(occupied.slots);assert(fort_staging::compact_available(fort_staging::snapshot())==0);}
    const std::string fault=mode=="alloc"?"ALLOC":mode=="submit"?"SUBMIT":mode=="record"?"RECORD":mode=="wait"?"WAIT":"";
    if(!fault.empty())setenv(("FORT_SCOPE_TEST_FAIL_COMPACT_"+fault).c_str(),"1",1);
    const auto status=fort_scope_host_begin(c,b,&read);
    if(mode=="submit"||mode=="record"||mode=="wait") {
        assert(status==FORT_SCOPE_EXECUTION);
        unsetenv(("FORT_SCOPE_TEST_FAIL_COMPACT_"+fault).c_str());
        assert(fort_scope_host_begin(c,b,&read)==FORT_SCOPE_EXECUTION);
        ck(cudaMemcpy(got.data(),device,bytes,cudaMemcpyDeviceToHost));assert(got==original);
        ck(cudaMemcpy(&executions,counter,sizeof(unsigned),cudaMemcpyDeviceToHost));assert(executions==1);
        assert(fort_scope_close(c)==FORT_SCOPE_EXECUTION);check(fort_scope_abandon(c));ck(cudaFree(counter));
        size_t freed=0;ck(fort_staging::trim_cache(freed));assert(!fort_staging::usage().reserved);
        std::puts("PASS_FATAL_NO_REPLAY_PRIOR_PRODUCER");return 0;
    }
    check(status);check(fort_scope_host_end(c,b));
    const bool direct=mode=="budget"||mode=="pool"||mode=="pinned_pool"||mode=="alloc"||mode=="uncovered"||mode=="oneop";
    auto after=stats(c);assert(after.launches==1 && after.downloads==(direct?section_count:1u) && after.download_bytes==section_count*y*z*element);
    for(size_t i=0;i<bytes;++i)assert(host[i]==(selected(i)?original[i]:0xe7));
    ck(cudaMemcpy(got.data(),device,bytes,cudaMemcpyDeviceToHost));assert(got==original);
    fort_scope_access write{0,0,nullptr,section_count,sections,section_count,sections};check(fort_scope_host_begin(c,b,&write));
    for(size_t i=0;i<bytes;++i)if(selected(i))host[i]=0xa9;
    check(fort_scope_host_end(c,b));
    fort_scope_access all{};all.flags=FORT_SCOPE_READ_ALL;
    check(fort_scope_device_begin(c,b,&all,reinterpret_cast<void**>(&device)));check(fort_scope_device_end(c,b));check(fort_scope_wait(c));
    after=stats(c);assert(after.uploads==(direct?section_count:1u) && after.upload_bytes==section_count*y*z*element);
    for(size_t i=0;i<bytes;++i)expected[i]=selected(i)?0xa9:original[i];
    ck(cudaMemcpy(got.data(),device,bytes,cudaMemcpyDeviceToHost));assert(got==expected);
    for(int i=0;i<3;++i){check(fort_scope_host_begin(c,b,&read));check(fort_scope_host_end(c,b));check(fort_scope_device_begin(c,b,&all,reinterpret_cast<void**>(&device)));check(fort_scope_device_end(c,b));}
    const auto retained=stats(c);assert(retained.uploads==after.uploads && retained.downloads==after.downloads);
    if(queue) {queued(c,b,device,bytes,counter,true);for(auto &v:expected)++v;}
    check(fort_scope_gpu_leave(c,previous));
    size_t freed=0;
    std::fputs("FINAL_PUBLICATION\n",stderr);
    check(fort_scope_close(c));assert(std::memcmp(host,expected.data(),bytes)==0);
    ck(cudaMemcpy(&executions,counter,sizeof(unsigned),cudaMemcpyDeviceToHost));assert(executions==1);ck(cudaFree(counter));
    if(occupied.slots)ck(fort_staging::release(std::move(occupied.slots),freed));
    const auto offset=size_t(host-allocation.data());for(size_t i=0;i<offset;++i)assert(allocation[i]==0xe7);
    for(size_t i=offset+bytes;i<allocation.size();++i)assert(allocation[i]==0xe7);
    ck(fort_staging::trim_cache(freed));assert(!fort_staging::usage().reserved && !fort_staging::usage().device_reserved);
    std::puts("PASS_COMPLETE_BYTES_CANARIES_REPLICAS_AND_CLEANUP");
}
'''


@pytest.fixture(scope="module")
def compact_executable(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14")
    assert nvcc and host
    directory = tmp_path_factory.mktemp("scoped_compact_cuda")
    client = directory / "client.cu"
    client.write_text(CLIENT)
    target = directory / "run"
    result = subprocess.run([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86",
                             "-DFORT_SCOPE_TEST_FAULTS", "-I", str(RUNTIME), str(client),
                             str(RUNTIME / "scoped_runtime.cu"), "-o", str(target)],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return target


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["normal", "bytes", "int32", "queued", "oversized_reuse", "budget", "pool", "pinned_pool", "alloc",
                                  "submit", "record", "wait", "oneop", "uncovered", "single", "single_queued",
                                  "single_bytes", "single_int32"])
def test_exact_compact_transport_order_budget_and_failures(compact_executable, mode):
    result = subprocess.run([str(compact_executable), mode], capture_output=True, text=True,
                            env={**os.environ, "FORT_RUNTIME_TRACE": "1"}, timeout=30)
    (compact_executable.parent / (mode+".stdout")).write_text(result.stdout)
    (compact_executable.parent / (mode+".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    before_publication = result.stderr.split("FINAL_PUBLICATION", 1)[0]
    events = [json.loads(line.split("evidence ", 1)[1]) for line in before_publication.splitlines()
              if '"event":"compact_transfer"' in line]
    if mode in {"normal", "bytes", "int32", "queued", "oversized_reuse", "single", "single_queued",
                "single_bytes", "single_int32"}:
        expected = 1 if mode.endswith("bytes") else 4 if mode.endswith("int32") else 8
        assert len(events) >= 2
        assert all(row["word_bytes"] == expected for row in events[:2])
        operations = 1 if mode.startswith("single") else 4
        assert [(row["upload"], row["dma_calls"], row["physical_operations"], row["copy_kernel_launches"])
                for row in events[:2]] == [(False, 1, operations, 1), (True, 1, operations, 1)]
        if mode == "oversized_reuse":
            assert all(row["reused"] and row["workspace_bytes"] == 8 << 20 for row in events[:2])
    else:
        assert not events
    if mode in {"submit", "record", "wait"}:
        assert "PASS_FATAL_NO_REPLAY_PRIOR_PRODUCER" in result.stdout
    else:
        assert "PASS_COMPLETE_BYTES_CANARIES_REPLICAS_AND_CLEANUP" in result.stdout
