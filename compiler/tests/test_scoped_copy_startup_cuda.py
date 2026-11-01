"""Real CUDA startup credit is a pricing observation, never data authority."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
MODES = ["seed_word64", "seed_int32", "seed_byte", "contexts", "domains", "queued",
         "two_contenders", "model_aba", "stale_reservation", "resource", "incompatible_idle",
         "partial_failure", "seed_submit", "seed_record", "seed_wait", "alloc_miss"]

CLIENT = r'''
#include "scoped_runtime.h"
#include "section_batch_cost.hpp"
#include "staging.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
using namespace fort_physical::compact;
void check(int s){if(s){std::fprintf(stderr,"scope %d: %s\n",s,fort_scope_error());std::abort();}}
void ck(cudaError_t s){if(s!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));std::abort();}}
Pricing prices(bool variant=false) {
    // Explicit synthetic dispatch rates: D=10, cold=9, warm=1. No hardware price.
    Pricing p{};p.copy.version=2;p.copy.valid=1;p.copy.minimum_relative_saving=.5;
    p.base.version=FORT_SCOPE_PLANNING_ABI_VERSION;p.base.valid=1;p.base.max_allocation_bytes=1u<<28;
    p.base.cpu_flops=p.base.cpu_bandwidth=p.base.gpu_flops=p.base.gpu_bandwidth=p.base.h2d_bandwidth=p.base.d2h_bandwidth=1e12;
    p.base.create_seconds=p.base.register_seconds=p.base.host_access_seconds=p.base.device_access_seconds=
      p.base.gpu_setup_seconds=p.base.cold_driver_startup_seconds=p.base.allocation_seconds=p.base.release_seconds=
      p.base.wait_seconds=p.base.launch_enqueue_seconds=p.base.planning_operation_seconds=1e-100;
    for(unsigned d=0;d<2;++d)for(auto *r:{&p.copy.direct[d],&p.copy.compact_cold[d],&p.copy.compact_reuse[d]}) {
        const bool direct=r==&p.copy.direct[d];r->batch_seconds=direct?10:r==&p.copy.compact_cold[d]?9:1;
        r->minimum_payload_bytes=r->minimum_rows=r->minimum_row_bytes=r->minimum_pitch_bytes=1;
        r->maximum_payload_bytes=r->maximum_rows=r->maximum_row_bytes=r->maximum_pitch_bytes=16u<<20;
        r->maximum_span_bytes=1u<<30;r->minimum_operations=direct?1:2;r->maximum_operations=256;
        r->minimum_descriptor_work=1;r->maximum_descriptor_work=1u<<30;
        r->minimum_kernel_chunks=1;r->maximum_kernel_chunks=6;r->word_bytes_mask=7;r->layout_combinations_mask=254;
    }
    if(variant)p.copy.direct[1].maximum_relative_error=1; // Exact different model, same predictions.
    assert(valid_costs(p.copy));return p;
}
__host__ __device__ bool selected(size_t i,size_t element,size_t x) {const auto a=(i/element)%x;return a==1||a==7;}
__host__ __device__ unsigned char value(size_t i,size_t element,size_t x,int round) {
    return static_cast<unsigned char>(((round>0&&selected(i,element,x)?17u*unsigned(round):0u)+i)%251+1);
}
__global__ void inspect(const unsigned char *p,size_t n,size_t element,size_t x,int round,unsigned *counts,int slot) {
    const auto i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(!i)atomicAdd(counts+slot,1u);
    if(i<n && p[i]!=value(i,element,x,round))atomicAdd(counts+2,1u);
}
struct App {
    size_t element,x,y=5,z=3,bytes;std::vector<unsigned char> allocation;unsigned char *host,*device=nullptr;
    unsigned *counts=nullptr;fort_scope_t context=0;fort_scope_domain_t domain=0;fort_buffer_t buffer=0;
    cudaStream_t stream=nullptr;int previous=0;size_t lo[2][3],hi[2][3];fort_scope_section sections[2];
    int last=0,consumers=0;bool ended=false;
    App(size_t word=8,bool borrowed=false,bool variant=false,bool limited=false)
      :element(word),x(word==8?34:35),bytes(x*y*z*element),allocation(bytes+65,0xe7),host(allocation.data()+(word==1?1:32)) {
        if(borrowed){check(fort_scope_domain_create_v1(0,&domain));check(fort_scope_create_in_domain_v1(domain,&context));}
        else check(fort_scope_create(0,&context));
        const auto p=prices(variant);check(fort_scope_set_transfers(context,FORT_SCOPE_TRANSFERS_AUTO));
        check(fort_scope_set_copy_costs_v2(context,&p.copy,&p.base,1));
        if(limited)check(fort_scope_set_device_budget(context,bytes));
        for(size_t i=0;i<bytes;++i)host[i]=value(i,element,x,0);
        const size_t shape[]={x,y,z};const int64_t lower[]={-17,2,-9};
        const fort_scope_layout layout{3,static_cast<uint32_t>(word==8?FORT_SCOPE_REAL64:word==4?FORT_SCOPE_INTEGER32:FORT_SCOPE_BYTES),
            word,host,shape,lower,1};check(fort_scope_register(context,31,7,&layout,1,&buffer));
        void *raw=nullptr;check(fort_scope_gpu_enter(context,&previous,&raw));stream=static_cast<cudaStream_t>(raw);assert(stream);
        ck(cudaMalloc(reinterpret_cast<void**>(&counts),3*sizeof(unsigned)));ck(cudaMemset(counts,0,3*sizeof(unsigned)));
        fort_scope_access all{};all.flags=FORT_SCOPE_READ_ALL;
        check(fort_scope_device_begin(context,buffer,&all,reinterpret_cast<void**>(&device)));
        inspect<<<(bytes+255)/256,256,0,stream>>>(device,bytes,element,x,0,counts,0);ck(cudaGetLastError());
        check(fort_scope_note_launch(context));check(fort_scope_device_end(context,buffer));
        for(size_t k=0;k<2;++k){lo[k][0]=1+6*k;lo[k][1]=lo[k][2]=0;hi[k][0]=lo[k][0]+1;hi[k][1]=y;hi[k][2]=z;sections[k]={lo[k],hi[k]};}
    }
    fort_scope_stats stats(){fort_scope_stats out{};check(fort_scope_stats_get(context,&out));return out;}
    int round(int r,bool queued=false) {
        fort_scope_access write{0,0,nullptr,2,sections,2,sections};check(fort_scope_host_begin(context,buffer,&write));
        for(size_t i=0;i<bytes;++i)if(selected(i,element,x))host[i]=value(i,element,x,r);
        check(fort_scope_host_end(context,buffer));last=r;
        fort_scope_access all{};all.flags=FORT_SCOPE_READ_ALL;
        const auto status=fort_scope_device_begin(context,buffer,&all,reinterpret_cast<void**>(&device));
        if(status)return status;
        if(queued) {
            check(fort_scope_device_end(context,buffer));
            auto ptr=device;auto n=bytes,e=element,xx=x;auto c=counts;int ordinal=r,slot=1;
            void *args[]={&ptr,&n,&e,&xx,&ordinal,&c,&slot};size_t sizes[]={sizeof(ptr),sizeof(n),sizeof(e),sizeof(xx),sizeof(ordinal),sizeof(c),sizeof(slot)};
            const fort_scope_plan_binding binding{buffer,all};
            const fort_scope_graph_task_v1 task{{1,0,reinterpret_cast<const void*>(inspect),{unsigned((bytes+255)/256),1,1},
                {256,1,1},0,args,7},&binding,1};
            check(fort_scope_graph_enqueue_v1(context,100+r,&task,sizes));
            ptr=nullptr;n=e=xx=0;c=nullptr;ordinal=slot=-1; // Submitted argument bytes must be owned.
        } else {
            inspect<<<(bytes+255)/256,256,0,stream>>>(device,bytes,element,x,r,counts,1);ck(cudaGetLastError());
            check(fort_scope_note_launch(context));check(fort_scope_device_end(context,buffer));
        }
        ++consumers;return 0;
    }
    void verify_counts() {
        unsigned observed[3];ck(cudaDeviceSynchronize());ck(cudaMemcpy(observed,counts,sizeof(observed),cudaMemcpyDeviceToHost));
        assert(observed[0]==1 && observed[1]==unsigned(consumers) && observed[2]==0);
    }
    void verify_device(int expected_round) {
        std::vector<unsigned char> actual(bytes);ck(cudaMemcpy(actual.data(),device,bytes,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<bytes;++i)assert(actual[i]==value(i,element,x,expected_round));
    }
    void verify_host() {
        for(size_t i=0;i<bytes;++i)assert(host[i]==value(i,element,x,last));
        const auto offset=size_t(host-allocation.data());for(size_t i=0;i<offset;++i)assert(allocation[i]==0xe7);
        for(size_t i=offset+bytes;i<allocation.size();++i)assert(allocation[i]==0xe7);
    }
    void finish() {
        assert(!ended);check(fort_scope_wait(context));verify_counts();verify_device(last);
        check(fort_scope_gpu_leave(context,previous));check(fort_scope_close(context));
        if(domain)check(fort_scope_domain_close_v1(domain));
        verify_host();
        ck(cudaFree(counts));ended=true;
    }
    void fatal(const std::string &mode) {
        assert(!ended);fort_scope_access all{};all.flags=FORT_SCOPE_READ_ALL;void *again=nullptr;
        assert(fort_scope_device_begin(context,buffer,&all,&again)==FORT_SCOPE_EXECUTION);
        assert(fort_scope_close(context)==FORT_SCOPE_EXECUTION);verify_counts();
        if(mode=="partial_failure") {
            std::vector<unsigned char> actual(bytes);ck(cudaMemcpy(actual.data(),device,bytes,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<bytes;++i)assert(actual[i]==value(i,element,x,(i/element)%x==1?1:0));
        } else verify_device(mode=="seed_submit"?2:3);
        assert(!fort_staging::snapshot().startup.credit[0]);
        verify_host();
        check(fort_scope_abandon(context));ck(cudaFree(counts));ended=true;
    }
};
void training(bool domain=false) {
    for(int i=0;i<2;++i){App app(8,domain);check(app.round(1));assert(app.stats().uploads==3);app.finish();
        assert(fort_staging::snapshot().startup.credit[0]==4.*(i+1));}
}
Plan quote_plan() {return make_plan({{8,8,5,3,34*8,5},{7*8,8,5,3,34*8,5}},34*5*3*8,16u<<20);}
int main(int argc,char **argv) {
    assert(argc==2);const std::string mode=argv[1];int count=0;ck(cudaGetDeviceCount(&count));assert(count==1);
    size_t freed=0;ck(fort_staging::trim_cache(freed));assert(!fort_staging::usage().reserved);
    if(mode=="contexts" || mode=="domains") {
        const bool domain=mode=="domains";training(domain);
        for(int i=0;i<2;++i){App app(8,domain);check(app.round(1));assert(app.stats().uploads==2);app.finish();}
    } else if(mode=="two_contenders") {
        training();std::atomic<int> ready{0};std::atomic<bool> go{false};unsigned long long uploads[2]={};
        auto run=[&](int i){ck(cudaSetDevice(0));App app;check(fort_scope_wait(app.context));++ready;
            while(!go.load())std::this_thread::yield();check(app.round(1));uploads[i]=app.stats().uploads;app.finish();};
        std::thread a(run,0),b(run,1);while(ready.load()!=2)std::this_thread::yield();go=true;a.join();b.join();
        assert(uploads[0]>=2 && uploads[0]<=3 && uploads[1]>=2 && uploads[1]<=3 && uploads[0]+uploads[1]<=5);
        assert(!fort_staging::snapshot().startup.credit[0]);
        assert(fort_staging::usage().device_peak==2*capacities[0]);
    } else if(mode=="model_aba" || mode=="stale_reservation") {
        training();const auto before=fort_staging::snapshot().startup;const auto p=prices();
        const auto offer=startup::choose(quote_plan(),p,true,false,{0,64u<<20,false,true},before);assert(offer.seed);
        if(mode=="model_aba") {
            {App app(8,false,true);check(app.round(1));app.finish();}
            {App app;check(app.round(1));app.finish();}
            const auto now=fort_staging::snapshot().startup;
            assert(now.model==before.model && now.generation!=before.generation && now.credit[0]==4);
        } else {
            auto other=fort_staging::acquire(256,fort_staging::Role::Staging,false);ck(other.status);assert(other.slots);
            ck(fort_staging::release(std::move(other.slots),freed));ck(fort_staging::trim_cache(freed));
        }
        const auto prior=fort_staging::usage();auto stale=fort_staging::acquire(offer.capacity,fort_staging::Role::Compact,false,
            fort_staging::device_limit,true,&before,&offer);
        assert(stale.exhausted && stale.stale_startup && !stale.slots);
        assert(fort_staging::usage().reserved==prior.reserved && fort_staging::usage().device_reserved==prior.device_reserved);
    } else if(mode=="resource" || mode=="incompatible_idle") {
        auto held=fort_staging::acquire(mode=="resource"?fort_staging::pinned_limit/2:4096,fort_staging::Role::Staging,false);
        ck(held.status);assert(held.slots);const auto original=held.slots->slots[0].host;
        if(mode=="incompatible_idle")ck(fort_staging::release(std::move(held.slots),freed));
        {App app;for(int r=1;r<=3;++r)check(app.round(r));assert(app.stats().uploads==7);app.finish();}
        assert(!fort_staging::snapshot().startup.credit[0]);
        if(mode=="incompatible_idle")assert(fort_staging::budget_state().idle->slots[0].host==original);
        if(held.slots)ck(fort_staging::release(std::move(held.slots),freed));
    } else if(mode=="alloc_miss") {
        training();App app;setenv("FORT_SCOPE_TEST_FAIL_COMPACT_ALLOC","1",1);check(app.round(1));
        unsetenv("FORT_SCOPE_TEST_FAIL_COMPACT_ALLOC");assert(app.stats().uploads==3 && fort_staging::snapshot().startup.credit[0]==8);
        check(app.round(2));assert(app.stats().uploads==4);app.finish();
    } else {
        const size_t word=mode=="seed_byte"?1:mode=="seed_int32"?4:8;App app(word);
        if(mode=="partial_failure") {
            setenv("FORT_SCOPE_TEST_FAIL_DIRECT_BATCH_AFTER","1",1);
            assert(app.round(1)==FORT_SCOPE_EXECUTION);unsetenv("FORT_SCOPE_TEST_FAIL_DIRECT_BATCH_AFTER");app.fatal(mode);
        } else {
            check(app.round(1,mode=="queued"));check(app.round(2,mode=="queued"));
            assert(app.stats().uploads==5 && fort_staging::snapshot().startup.credit[0]==8);
            if(mode=="seed_submit" || mode=="seed_record" || mode=="seed_wait") {
                const std::string suffix=mode=="seed_submit"?"SUBMIT":mode=="seed_record"?"RECORD":"WAIT";
                const auto key="FORT_SCOPE_TEST_FAIL_COMPACT_"+suffix;setenv(key.c_str(),"1",1);
                assert(app.round(3)==FORT_SCOPE_EXECUTION);unsetenv(key.c_str());app.fatal(mode);
            } else {
                check(app.round(3,mode=="queued"));assert(app.stats().uploads==6);
                check(app.round(4,mode=="queued"));assert(app.stats().uploads==7);app.finish();
            }
        }
    }
    std::fputs("STARTUP_FINAL_CLEANUP\n",stderr);
    ck(fort_staging::trim_cache(freed));assert(!fort_staging::usage().reserved && !fort_staging::usage().device_reserved);
    std::puts("PASS_STARTUP_FULL_FIELDS_CANARIES_COUNTS_AND_CLEANUP");
}
'''


@pytest.fixture(scope="module")
def startup_executable(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14")
    assert nvcc
    assert host
    folder = tmp_path_factory.mktemp("scoped_startup_cuda")
    (folder / "client.cu").write_text(CLIENT)
    result = subprocess.run([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86",
                             "-DFORT_SCOPE_TEST_FAULTS", "-Xcompiler=-pthread", "-I", str(RUNTIME),
                             str(folder / "client.cu"), str(RUNTIME / "scoped_runtime.cu"), "-o", str(folder / "run")],
                            capture_output=True, text=True, timeout=180)
    (folder / "compile.stdout").write_text(result.stdout)
    (folder / "compile.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return folder / "run"


@pytest.mark.cuda
@pytest.mark.parametrize("mode", MODES)
def test_shared_startup_credit_real_cuda(startup_executable, mode):
    result = subprocess.run([str(startup_executable), mode], capture_output=True, text=True,
                            env={**os.environ, "FORT_RUNTIME_TRACE": "1"}, timeout=40)
    (startup_executable.parent / (mode+".stdout")).write_text(result.stdout)
    (startup_executable.parent / (mode+".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS_STARTUP_FULL_FIELDS_CANARIES_COUNTS_AND_CLEANUP" in result.stdout
    events = [json.loads(line.split("evidence ", 1)[1]) for line in result.stderr.splitlines()
              if '"event":"compact_transfer"' in line]
    cold = [row for row in events if not row["reused"]]
    if mode in {"seed_word64", "seed_int32", "seed_byte", "contexts", "domains", "queued"}:
        assert len(events) == 2
        assert len(cold) == 1
        word = 1 if mode == "seed_byte" else 4 if mode == "seed_int32" else 8
        assert all(row["word_bytes"] == word and row["upload"] and row["physical_operations"] == 2
                   and row["dma_calls"] == row["copy_kernel_launches"] == 1 for row in events)
        assert events[0]["reused"] is False
        assert events[1]["reused"] is True
    elif mode == "two_contenders":
        assert 1 <= len(events) <= 2
        assert len(cold) == 1
    elif mode == "alloc_miss":
        assert len(events) == len(cold) == 1
    else:
        assert not events
