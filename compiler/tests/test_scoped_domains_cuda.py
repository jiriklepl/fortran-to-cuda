"""Actual stream ordering, exact mirrors and fatal external-call cleanup."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_runtime import RUNTIME

CLIENT = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void check(int status) {
    if(status) { std::fprintf(stderr,"status=%d: %s\n",status,fort_scope_error()); std::abort(); }
}
void gpu(cudaError_t status) {
    if(status!=cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status)); std::abort(); }
}
__global__ void produce(const double *a,double *b,unsigned long long delay) {
    const int i=int(threadIdx.x)+1;
    if(i>=9) return;
    const auto start=clock64(); while(clock64()-start<delay) {}
    b[i]=2*a[i];
}
__global__ void consume(const double *b,double *out,double *workspace,double coefficient) {
    const int i=int(threadIdx.x)+1;
    if(i<9) out[i]=b[i]+coefficient;
    if(i==1) workspace[0]=coefficient;
}
__global__ void increment(double *a) { if(threadIdx.x<8) a[threadIdx.x]+=1000; }
void eviction(fort_scope_domain_t domain,fort_scope_t context) {
    check(fort_scope_set_device_budget(context,128));
    double host[3][8]; fort_buffer_t handles[3]{};
    size_t extents[]={8}; int64_t lower[]={-4};
    for(int j=0;j<3;++j) {
        for(int i=0;i<8;++i) host[j][i]=100*(j+1)+i;
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host[j],extents,lower,7};
        check(fort_scope_register(context,j+1,7,&layout,1,&handles[j]));
    }
    int previous=-1; void *raw=nullptr; check(fort_scope_gpu_enter(context,&previous,&raw));
    const auto stream=static_cast<cudaStream_t>(raw);
    fort_scope_access rw{}; rw.flags=FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL;
    fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
    void *pointer=nullptr; check(fort_scope_device_begin(context,handles[0],&rw,&pointer));
    increment<<<1,8,0,stream>>>(static_cast<double *>(pointer)); gpu(cudaGetLastError());
    check(fort_scope_note_launch(context)); check(fort_scope_device_end(context,handles[0]));
    check(fort_scope_device_begin(context,handles[1],&read,&pointer)); check(fort_scope_device_end(context,handles[1]));
    check(fort_scope_device_reserve_many_v1(context,&handles[2],1,&pointer));
    fort_scope_stats counters{}; check(fort_scope_stats_get(context,&counters));
    assert(counters.upload_bytes==128 && counters.download_bytes==0 && counters.allocations==3);
    fort_scope_residency_stats_v1 resident{}; check(fort_scope_residency_stats_get_v1(context,&resident));
    assert(resident.evictions==1 && resident.dirty_evictions==0);
    for(int i=0;i<8;++i) assert(host[0][i]==100+i);
    check(fort_scope_device_begin(context,handles[0],&read,&pointer));
    // Independent diagnostic readback verifies the dirty allocation survived;
    // it does not publish/alter the runtime's host mirror or its byte counters.
    double observed[8]; gpu(cudaMemcpyAsync(observed,pointer,64,cudaMemcpyDeviceToHost,stream));
    gpu(cudaStreamSynchronize(stream)); check(fort_scope_device_end(context,handles[0]));
    for(int i=0;i<8;++i) assert(observed[i]==1100+i && host[0][i]==100+i);
    check(fort_scope_stats_get(context,&counters));
    assert(counters.upload_bytes==128 && counters.download_bytes==0 && counters.allocations==3);
    fort_buffer_t pinned[]={handles[1],handles[2]}; void *pointers[2]{};
    check(fort_scope_device_reserve_many_v1(context,pinned,2,pointers));
    check(fort_scope_stats_get(context,&counters)); check(fort_scope_residency_stats_get_v1(context,&resident));
    assert(counters.download_bytes==64 && resident.evictions==2 && resident.dirty_evictions==1);
    for(int i=0;i<8;++i) assert(host[0][i]==1100+i && host[1][i]==200+i && host[2][i]==300+i);
    check(fort_scope_gpu_leave(context,previous)); check(fort_scope_close(context));
    check(fort_scope_domain_close_v1(domain)); std::puts("CLEAN_FIRST_EVICTION_OK");
}
struct Provider {
    double *workspace=nullptr;
    double coefficient=0;
    cudaStream_t stream=nullptr;
    int calls=0;
    bool reject=false, fail=false;
};
int external(void *opaque,const fort_scope_external_view_v1 *views,size_t count,void *raw,int *started) {
    auto &p=*static_cast<Provider *>(opaque); ++p.calls;
    assert(count==2 && raw);
    for(size_t i=0;i<count;++i) {
        assert(views[i].type==FORT_SCOPE_REAL64 && views[i].rank==1);
        assert(views[i].generation==7 && views[i].extents[0]==10);
        assert(views[i].element_bytes==8 && views[i].strides[0]==8);
        assert(views[i].bytes==80 && views[i].lower_bounds[0]==-4);
    }
    if(p.reject) { *started=0; return FORT_SCOPE_RESOURCE; }
    p.stream=static_cast<cudaStream_t>(raw);
    *started=1;
    consume<<<1,32,0,p.stream>>>(static_cast<const double *>(views[0].device),
        static_cast<double *>(views[1].device),p.workspace,p.coefficient);
    if(cudaGetLastError()!=cudaSuccess || p.fail) return FORT_SCOPE_EXECUTION;
    return FORT_SCOPE_OK;
}
fort_scope_stats stats(fort_scope_domain_t domain) {
    fort_scope_stats out{}; check(fort_scope_domain_stats_get_v1(domain,&out)); return out;
}
void finish_provider(Provider &provider) {
    if(provider.stream) {
        cudaEvent_t event=nullptr; gpu(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
        gpu(cudaEventRecord(event,provider.stream)); gpu(cudaEventSynchronize(event)); gpu(cudaEventDestroy(event));
    }
    gpu(cudaFree(provider.workspace)); provider.workspace=nullptr;
}
int main(int argc,char **argv) {
    assert(argc==2); const bool empty=!std::strcmp(argv[1],"empty");
    fort_scope_domain_t domain=0; fort_scope_t first=0,second=0;
    check(fort_scope_domain_create_v1(0,&domain));
    check(fort_scope_create_in_domain_v1(domain,&first));
    if(empty) {
        fort_scope_external_call_v1 call{1,0,nullptr,0,0,0}; fort_scope_external_report_v1 report{};
        check(fort_scope_external_execute_v1(first,&call,nullptr,nullptr,&report));
        assert(!report.started && !report.committed);
        check(fort_scope_close(first)); const auto s=stats(domain);
        assert(!s.allocations && !s.upload_bytes && !s.download_bytes);
        check(fort_scope_domain_close_v1(domain)); std::puts("EMPTY_OK"); return 0;
    }
    int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    if(!std::strcmp(argv[1],"eviction")) { eviction(domain,first); return 0; }
    const bool failure=!std::strcmp(argv[1],"poison"), rejection=!std::strcmp(argv[1],"reject");
    const bool swapped=!std::strcmp(argv[1],"swap");
    check(fort_scope_set_device_budget(first,272));
    uint64_t lease=0; check(fort_scope_domain_workspace_reserve_v1(domain,32,&lease));
    Provider provider; gpu(cudaMalloc(&provider.workspace,32));
    provider.fail=failure; provider.reject=rejection;
    double a[10],b[10],out[10];
    for(int i=0;i<10;++i) { a[i]=i+1; b[i]=-100-i; out[i]=-200-i; }
    double *logical_host[3]={a,b,out};
    size_t extents[]={10},lo[]={1},hi[]={9}; int64_t lower[]={-4};
    fort_buffer_t handles[3]{};
    auto register_all=[&](fort_scope_t context) {
        for(size_t i=0;i<3;++i) {
            fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,logical_host[i],extents,lower,7};
            fort_buffer_t result=0; check(fort_scope_register(context,i+1,7,&layout,1,&result));
            if(handles[i]) assert(result==handles[i]); else handles[i]=result;
        }
    };
    register_all(first);
    void *pointers[2]{}; check(fort_scope_device_reserve_many_v1(first,handles,2,pointers));
    auto *pa=static_cast<double *>(pointers[0]),*pb=static_cast<double *>(pointers[1]);
    fort_scope_section interior{lo,hi};
    fort_scope_access read{0,1,&interior,0,nullptr,0,nullptr};
    fort_scope_access overwrite{0,0,nullptr,1,&interior,1,&interior};
    fort_scope_plan_binding bindings[2]={{handles[0],read},{handles[1],overwrite}};
    int khz=0; gpu(cudaDeviceGetAttribute(&khz,cudaDevAttrClockRate,0));
    auto delay=static_cast<unsigned long long>(khz)*20;
    void *arguments[3]={&pa,&pb,&delay}; size_t widths[3]={sizeof(pa),sizeof(pb),sizeof(delay)};
    fort_scope_graph_task_v1 task{{1,0,reinterpret_cast<const void *>(produce),{1,1,1},{32,1,1},0,arguments,3},bindings,2};
    check(fort_scope_graph_enqueue_v1(first,19,&task,widths));
    if(swapped) {
        // The original descriptor-only exchange precedes the checked commit.
        // Its earlier accepted task owns physical handles/argument copies and
        // must still run against the original allocations without replay.
        logical_host[0]=b; logical_host[1]=a;
        fort_scope_layout after_a{1,FORT_SCOPE_REAL64,8,b,extents,lower,7};
        fort_scope_layout after_b{1,FORT_SCOPE_REAL64,8,a,extents,lower,7};
        const auto old_a=handles[0],old_b=handles[1];
        check(fort_scope_identity_swap_v1(first,old_a,old_b,&after_a,&after_b,&handles[0],&handles[1]));
        assert(handles[0]==old_b && handles[1]==old_a);
    }
    fort_scope_graph_queue_stats_v1 queued{}; check(fort_scope_graph_queue_stats_get_v1(first,&queued));
    assert(queued.accepted_tasks==1 && queued.pending_tasks==1);
    assert(!stats(domain).launches && !stats(domain).upload_bytes);
    check(fort_scope_close(first));
    assert(stats(domain).launches==1 && stats(domain).upload_bytes==64 && !stats(domain).download_bytes);
    for(int i=0;i<10;++i) assert(a[i]==i+1 && b[i]==-100-i && out[i]==-200-i);

    check(fort_scope_create_in_domain_v1(domain,&second)); register_all(second);
    check(fort_scope_graph_queue_stats_get_v1(second,&queued));
    assert(!queued.accepted_tasks && !queued.pending_tasks && !queued.submissions);
    // A readonly producer leaves its host mirror valid. B's GPU-written
    // interior needs exactly this reached 8-byte probe, which feeds consume.
    fort_scope_access all_read{}; all_read.flags=FORT_SCOPE_READ_ALL;
    const auto physical_a=handles[swapped?1:0],physical_b=handles[swapped?0:1];
    check(fort_scope_host_begin(second,physical_a,&all_read));
    for(int i=0;i<10;++i) assert(a[i]==i+1);
    check(fort_scope_host_end(second,physical_a)); assert(!stats(domain).download_bytes);
    size_t probe_lo[]={4},probe_hi[]={5}; fort_scope_section probe{probe_lo,probe_hi};
    fort_scope_access probe_read{0,1,&probe,0,nullptr,0,nullptr};
    check(fort_scope_host_begin(second,physical_b,&probe_read));
    assert(b[4]==10); provider.coefficient=a[4]+b[4];
    check(fort_scope_host_end(second,physical_b));
    assert(stats(domain).download_bytes==8 && stats(domain).upload_bytes==64);

    fort_scope_external_binding_v1 external_bindings[2]={
        {physical_b,FORT_SCOPE_REAL64,1,8,read},{handles[2],FORT_SCOPE_REAL64,1,8,overwrite}};
    fort_scope_external_call_v1 call{1,0,external_bindings,2,lease,32};
    fort_scope_external_report_v1 report{};
    const int result=fort_scope_external_execute_v1(second,&call,external,&provider,&report);
    assert(provider.calls==1);
    assert(stats(domain).upload_bytes==64 && stats(domain).download_bytes==8 && stats(domain).allocations==3);
    if(failure) {
        assert(result==FORT_SCOPE_EXECUTION && report.started==1 && !report.committed);
        assert(fort_scope_wait(second)==FORT_SCOPE_EXECUTION);
        assert(fort_scope_external_execute_v1(second,&call,external,&provider,&report)==FORT_SCOPE_EXECUTION);
        assert(provider.calls==1);
        assert(fort_scope_close(second)==FORT_SCOPE_EXECUTION);
        finish_provider(provider);
        check(fort_scope_domain_workspace_release_v1(domain,lease));
        check(fort_scope_abandon(second)); check(fort_scope_domain_abandon_v1(domain));
        for(int i=0;i<10;++i) assert(out[i]==-200-i);
        for(int i=0;i<10;++i) assert(b[i]==(i==4?10:-100-i));
        std::puts("POISON_NO_PUBLICATION_OK"); return 0;
    }
    if(rejection) {
        assert(result==FORT_SCOPE_RESOURCE && !report.started && !report.committed);
        // A checked prework decline permits typed host recovery. Complete
        // only the writes actually performed by this fallback operation.
        check(fort_scope_host_begin(second,physical_b,&read));
        check(fort_scope_host_begin(second,handles[2],&overwrite));
        for(int i=1;i<9;++i) out[i]=b[i]+provider.coefficient;
        check(fort_scope_host_end(second,physical_b)); check(fort_scope_host_end(second,handles[2]));
    } else assert(result==0 && report.started==1 && report.committed==1);
    assert(stats(domain).launches==1); // Opaque provider kernel counts are unknown, never invented.
    check(fort_scope_close(second));
    if(!rejection) for(int i=0;i<10;++i) assert(out[i]==-200-i);
    finish_provider(provider);
    check(fort_scope_domain_workspace_release_v1(domain,lease));
    check(fort_scope_domain_close_v1(domain));
    for(int i=0;i<10;++i) {
        assert(a[i]==i+1);
        assert(b[i]==(i>0 && i<9 ? 2*(i+1) : -100-i));
        assert(out[i]==(i>0 && i<9 ? 2*(i+1)+15 : -200-i));
    }
    std::puts(rejection?"CHECKED_NATIVE_RECOVERY_OK":"DOMAIN_CHAIN_OK");
}
'''


@pytest.fixture(scope="module")
def domain_cuda_executable(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit unavailable")
    directory = tmp_path_factory.mktemp("domain_cuda")
    source = directory / "client.cu"
    source.write_text(CLIENT)
    objects = []
    for name, path in [("runtime", RUNTIME / "scoped_runtime.cu"), ("client", source)]:
        target = directory / (name + ".o")
        result = subprocess.run([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-I", str(RUNTIME),
                                 "-c", str(path), "-o", str(target)], capture_output=True, text=True, timeout=120)
        (directory / (name + ".stdout")).write_text(result.stdout)
        (directory / (name + ".stderr")).write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    target = directory / "run"
    result = subprocess.run([nvcc, "-ccbin", host, *objects, "-o", str(target)],
                            capture_output=True, text=True, timeout=60)
    (directory / "link.stdout").write_text(result.stdout)
    (directory / "link.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    return target


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["chain", "swap", "reject", "poison", "empty", "eviction"])
def test_actual_domain_external_transaction(domain_cuda_executable, mode):
    environment = {**os.environ, "FORT_RUNTIME_TRACE": "1", "FORT_SCOPE_GRAPH_QUEUE": "1"}
    if mode == "empty":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = subprocess.run([str(domain_cuda_executable), mode], env=environment,
                            capture_output=True, text=True, timeout=30)
    directory = domain_cuda_executable.parent
    (directory / (mode + ".stdout")).write_text(result.stdout)
    (directory / (mode + ".stderr")).write_text(result.stderr)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    if mode in {"chain", "swap"}:
        uploads = re.findall(r"FORT_SCOPED upload buffer=(\d+) generation=7 bytes=(\d+)", result.stderr)
        downloads = re.findall(r"FORT_SCOPED download buffer=(\d+) generation=7 bytes=(\d+)", result.stderr)
        source_identity, produced_identity = ("2", "1") if mode == "swap" else ("1", "2")
        assert uploads == [(source_identity, "64")]
        assert sum(int(size) for _identity, size in downloads) == 128
        assert sum(int(size) for identity, size in downloads if identity == produced_identity) == 64
        assert sum(int(size) for identity, size in downloads if identity == "3") == 64
        assert all(identity != source_identity for identity, _size in downloads)
        assert result.stderr.count("FORT_SCOPED detach ") == 2
        assert result.stderr.count("FORT_SCOPED external_commit ") == 1
        lifecycle = [json.loads(line.removeprefix("FORT_SCOPED domain "))
                     for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED domain ")]
        assert [row["event"] for row in lifecycle] == ["domain_create", "domain_attach", "domain_detach",
                                                    "domain_attach", "domain_detach", "domain_retire"]
        assert len({row["domain"] for row in lifecycle}) == 1
        assert lifecycle[-1]["last_context"] == lifecycle[-2]["context"]
        assert result.stderr.count("FORT_SCOPED identity_swap ") == int(mode == "swap")
    if mode == "empty":
        assert "FORT_SCOPED allocate " not in result.stderr
        assert "FORT_SCOPED upload " not in result.stderr
        assert "FORT_SCOPED download " not in result.stderr
