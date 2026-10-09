"""Exact pitched copies and bounded scratch ownership, independent of selection."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.common.resources import read_common_header


def checked(command, path):
    result = subprocess.run(command, cwd=path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.native
@pytest.mark.cuda
def test_pitched_copies_match_independent_coordinates_and_cost_counts(tmp_path):
    nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
    if not nvcc:
        pytest.skip("CUDA compiler required")
    (tmp_path / "common.hpp").write_text(read_common_header())
    (tmp_path / "copies.cu").write_text(r'''
#include <cuda_runtime.h>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>
int copies=0;
cudaError_t counted1(void *d,const void *s,size_t n,cudaMemcpyKind k) {
    ++copies; return cudaMemcpy(d,s,n,k);
}
cudaError_t counted2(void *d,size_t dp,const void *s,size_t sp,size_t w,size_t h,cudaMemcpyKind k) {
    ++copies; return cudaMemcpy2D(d,dp,s,sp,w,h,k);
}
cudaError_t counted3(const cudaMemcpy3DParms *p) { ++copies; return cudaMemcpy3D(p); }
#define cudaMemcpy counted1
#define cudaMemcpy2D counted2
#define cudaMemcpy3D counted3
#include "common.hpp"
#undef cudaMemcpy
#undef cudaMemcpy2D
#undef cudaMemcpy3D
int main() {
    using namespace generated_kernels::offload;
    std::mt19937 random(42);
    for (unsigned rank=1;rank<=4;++rank) {
        std::vector<size_t> dimensions{7,6,5,4}; dimensions.resize(rank);
        size_t n=1; for(auto d:dimensions) n*=d;
        std::vector<double> host(n),expected(n),actual(n),sentinel(n,-123);
        void *device=nullptr;
        CUCH(cudaMalloc(&device,n*sizeof(double)));
        Array a{host.data(),sizeof(double),n*sizeof(double),dimensions};
        for(unsigned sample=0;sample<80;++sample) {
            Box box;
            for(auto d:dimensions) {
                const auto lo=sample==0 ? 0 : random()%d;
                const auto hi=sample==0 ? d-1 : lo+random()%(d-lo);
                box.lower.push_back(lo); box.upper.push_back(hi);
            }
            // Explicit opposite faces, singleton middle axes, and a 3D interior.
            if(sample<=2*rank && sample) {
                box.lower.assign(rank,0); box.upper=dimensions;
                for(auto &v:box.upper) --v;
                const auto axis=(sample-1)/2;
                box.lower[axis]=box.upper[axis]=(sample%2) ? 0 : dimensions[axis]-1;
            }
            if(sample==2*rank+1) {
                box.lower.assign(rank,1); box.upper=dimensions;
                for(auto &v:box.upper) v-=2;
            }
            for(bool upload:{true,false}) {
                std::iota(host.begin(),host.end(),10.0+sample);
                expected=sentinel;
                // Element-by-element oracle uses logical coordinates, not CUDA pitches.
                size_t selected=0;
                for(size_t i=0;i<n;++i) {
                    auto rest=i; bool inside=true;
                    for(unsigned axis=0;axis<rank;++axis) {
                        const auto at=rest%dimensions[axis]; rest/=dimensions[axis];
                        inside &= at>=box.lower[axis] && at<=box.upper[axis];
                    }
                    if(inside) { expected[i]=host[i]; ++selected; }
                }
                CUCH(cudaMemcpy(device,upload ? sentinel.data() : host.data(),a.bytes,cudaMemcpyHostToDevice));
                if(!upload) host=sentinel;
                copies=0;
                copy_box(a,box,device,upload);
                bool valid=true;
                assert(copy_operations(a,box,valid)==static_cast<size_t>(copies) && valid);
                assert(box_bytes(a,box,valid)==selected*sizeof(double));
                if(upload) CUCH(cudaMemcpy(actual.data(),device,a.bytes,cudaMemcpyDeviceToHost));
                else actual=host;
                assert(actual==expected);
            }
        }
        CUCH(cudaFree(device));
    }
}
''')
    command = [nvcc, "-std=c++17", "-O2", "-DFORT_OFFLOAD_ENABLED", "-Xcompiler=-fopenmp"]
    host = shutil.which(os.environ.get("CUDAHOSTCXX", "g++-14"))
    if host:
        command += ["-ccbin", host]
    checked([*command, "copies.cu", "-o", "copies"], tmp_path)
    checked([str(tmp_path / "copies")], tmp_path)


@pytest.mark.native
def test_scratch_reuse_growth_device_eviction_failures_and_concurrent_budget(tmp_path):
    cxx = shutil.which("g++")
    if not cxx:
        pytest.skip("C++ compiler required")
    runtime = Path(__file__).resolve().parents[1] / "runtime/hybrid.hpp"
    (tmp_path / "cache.cpp").write_text(r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <future>
#include <mutex>
#include <unordered_map>
#include <omp.h>
#define __CUDACC__
#define __device__
struct Handle { int device; };
using cudaStream_t=Handle*;
using cudaEvent_t=Handle*;
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=1,cudaErrorNotReady=2,cudaStreamNonBlocking=1,cudaEventDisableTiming=2;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
thread_local int current=0;
std::atomic<int> fail_at{0},allocations{0},handles{0};
std::atomic<int> device_queries{0};
size_t pinned=0,peak=0;
std::mutex memory_mutex;
struct Allocation { int device; size_t bytes; bool host; };
std::unordered_map<void*,Allocation> memory;
bool fails() { return ++allocations==fail_at; }
int cudaGetDevice(int *p) { ++device_queries; *p=current; return 0; }
int cudaSetDevice(int n) { current=n; return 0; }
int cudaGetLastError() { return 0; }
int cudaStreamCreateWithFlags(Handle **p,unsigned) {
    if(fails()) return 1;
    *p=new Handle{current}; ++handles; return 0;
}
int cudaEventCreateWithFlags(Handle **p,unsigned f) { return cudaStreamCreateWithFlags(p,f); }
int cudaStreamDestroy(Handle *p) { assert(p->device==current); delete p; --handles; return 0; }
int cudaEventDestroy(Handle *p) { return cudaStreamDestroy(p); }
int cudaEventSynchronize(Handle *p) { assert(p->device==current); return 0; }
int cudaStreamSynchronize(Handle *p) { assert(p->device==current); return 0; }
int cudaEventRecord(Handle *p,Handle *s) { assert(p->device==current && s->device==current); return 0; }
int allocate(void **p,size_t bytes,bool host) {
    if(fails()) return 1;
    *p=std::malloc(bytes); assert(*p);
    std::lock_guard<std::mutex> lock(memory_mutex);
    memory.emplace(*p,Allocation{current,bytes,host});
    if(host) { pinned+=bytes; peak=std::max(peak,pinned); }
    return 0;
}
int cudaMallocHost(void **p,size_t n) { return allocate(p,n,true); }
int cudaMalloc(void **p,size_t n) { return allocate(p,n,false); }
int cudaFree(void *p) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    const auto allocation=memory.at(p); assert(allocation.device==current);
    if(allocation.host) pinned-=allocation.bytes;
    memory.erase(p); std::free(p); return 0;
}
int cudaFreeHost(void *p) { return cudaFree(p); }
int cudaMemcpyAsync(void*,const void*,size_t,int,Handle*) { return 0; }
#define CUCH(call) do { assert((call)==0); } while(0)
namespace generated_kernels::storage {
std::atomic<int> trace_allocs{0},trace_frees{0};
void trace(const char *operation,size_t=0) {
    if(!std::strcmp(operation,"alloc")) ++trace_allocs;
    if(!std::strcmp(operation,"free")) ++trace_frees;
}
[[noreturn]] void fail(const char*) { std::abort(); }
}
namespace generated_kernels::offload {
void decision_trace(const char*,const char*,size_t,size_t,const char*) {}
}
''' + f'#include "{runtime}"\n' + r'''
int main() {
    using namespace generated_kernels::hybrid;
    auto &budget=budget_state();
    trim_cache(); assert(device_queries==0);
    // Every partial allocation stage must restore both memory and budget.
    for(int point=1;point<=8;++point) {
        allocations=0; fail_at=point;
        assert(!acquire_slots(1024));
        assert(!pinned && !handles && memory.empty() && !budget.reserved);
        assert(generated_kernels::storage::trace_allocs==generated_kernels::storage::trace_frees);
    }
    fail_at=0; allocations=0;
    auto a=acquire_slots(1024); assert(a && a->ready);
    const auto first=a->slots[0].host;
    release_slots(std::move(a));
    a=acquire_slots(512);
    assert(a->slots[0].host==first && a->capacity==1024 && allocations==8);
    release_slots(std::move(a));
    a=acquire_slots(2048); assert(a->capacity==2048 && allocations==16);
    release_slots(std::move(a));
    current=1;
    a=acquire_slots(512); assert(a->device==1 && current==1 && allocations==24);
    release_slots(std::move(a));
    current=0; trim_cache(); assert(pinned==1024 && current==0);
    current=1; trim_cache(); assert(!pinned && !budget.reserved && memory.empty());
    current=0;
    // Two simultaneous leases exhaust the limit. A third waits, then reuses a
    // released pair without evicting active data or exceeding the pinned cap.
    constexpr size_t capacity=pinned_limit/4;
    a=acquire_slots(capacity);
    auto b=acquire_slots(capacity);
    assert(budget.reserved==pinned_limit && pinned==pinned_limit);
    std::promise<void> started;
    auto start=started.get_future();
    auto waiter=std::async(std::launch::async,[&] {
        started.set_value(); return acquire_slots(capacity);
    });
    start.wait();
    assert(waiter.wait_for(std::chrono::milliseconds(40))==std::future_status::timeout);
    const auto count=allocations.load();
    release_slots(std::move(a));
    auto c=waiter.get(); assert(c && allocations==count);
    release_slots(std::move(b)); release_slots(std::move(c));
    assert(budget.reserved==2*capacity && peak<=pinned_limit);
    trim_cache();
    assert(!budget.reserved && !pinned && !handles && memory.empty());
    assert(generated_kernels::storage::trace_allocs==generated_kernels::storage::trace_frees);
}
''')
    checked([cxx, "-std=c++17", "-O2", "-fopenmp", "-pthread", "cache.cpp", "-o", "cache"], tmp_path)
    checked([str(tmp_path / "cache")], tmp_path)
