"""One bounded pinned pool serves ordinary and scoped translation units."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.common.resources import read_common_header

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

MOCK = r'''
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
using cudaError_t=int;
struct Handle { int device; };
using cudaStream_t=Handle*;
using cudaEvent_t=Handle*;
constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=1,cudaErrorNotReady=2,cudaErrorUnknown=3;
constexpr int cudaStreamNonBlocking=1,cudaEventDisableTiming=2;
namespace fake {
inline thread_local int device=0;
inline int fail_at=0,allocations=0,handles=0,streams_drained=0,events_waited=0,queries=0;
inline bool outstanding=false,fail_drain=false;
inline void *outstanding_host=nullptr;
inline size_t pinned=0,peak=0,device_bytes=0;
struct Memory { int device; size_t bytes; bool host; };
inline std::unordered_map<void*,Memory> memory;
inline std::mutex mutex;
inline bool fail() { return ++allocations==fail_at; }
}
inline int cudaGetDevice(int *d) { ++fake::queries; *d=fake::device; return 0; }
inline int cudaSetDevice(int d) { fake::device=d; return 0; }
inline int cudaGetLastError() { return 0; }
inline int cudaStreamCreateWithFlags(Handle **p,unsigned) {
    if(fake::fail()) return 1;
    *p=new Handle{fake::device}; ++fake::handles; return 0;
}
inline int cudaEventCreateWithFlags(Handle **p,unsigned f) { return cudaStreamCreateWithFlags(p,f); }
inline int cudaStreamDestroy(Handle *p) { assert(p->device==fake::device); delete p; --fake::handles; return 0; }
inline int cudaEventDestroy(Handle *p) { return cudaStreamDestroy(p); }
inline int cudaEventSynchronize(Handle*) { ++fake::events_waited; return 0; }
inline int cudaStreamSynchronize(Handle *p) {
    assert(p->device==fake::device); ++fake::streams_drained;
    if(fake::fail_drain) return cudaErrorUnknown;
    fake::outstanding=false; return 0;
}
inline int allocate(void **p,size_t n,bool host) {
    if(fake::fail()) return 1;
    *p=std::malloc(n); assert(*p);
    std::lock_guard<std::mutex> lock(fake::mutex);
    fake::memory.emplace(*p,fake::Memory{fake::device,n,host});
    if(host) { fake::pinned+=n; fake::peak=std::max(fake::peak,fake::pinned); }
    else fake::device_bytes+=n;
    return 0;
}
inline int cudaMallocHost(void **p,size_t n) { return allocate(p,n,true); }
inline int cudaMalloc(void **p,size_t n) { return allocate(p,n,false); }
inline int cudaFree(void *p) {
    assert(p!=fake::outstanding_host || !fake::outstanding);
    std::lock_guard<std::mutex> lock(fake::mutex);
    const auto a=fake::memory.at(p); assert(a.device==fake::device);
    if(a.host) fake::pinned-=a.bytes; else fake::device_bytes-=a.bytes;
    fake::memory.erase(p); std::free(p); return 0;
}
inline int cudaFreeHost(void *p) { return cudaFree(p); }
'''


@pytest.fixture(scope="module")
def pool_program(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("shared_staging")
    (directory / "cuda_runtime.h").write_text(MOCK)
    # The packaging path inlines this exact header text into ordinary support.
    header = (RUNTIME / "staging.hpp").read_text()
    assert header in read_common_header()
    (directory / "ordinary.cuh").write_text(header)
    (directory / "ordinary.cpp").write_text(r'''
#define CUCH(call) ordinary_policy_is_not_part_of_the_shared_pool(call)
#include "ordinary.cuh"
void *ordinary_budget() { return &fort_staging::budget_state(); }
fort_staging::Result ordinary_acquire(size_t n) { return fort_staging::acquire(n,fort_staging::Role::Compact,false); }
''')
    (directory / "scoped.cpp").write_text(f'#include "{RUNTIME / "staging.hpp"}"\n' + r'''
#include <cstring>
void *ordinary_budget();
fort_staging::Result ordinary_acquire(size_t);
int main(int argc,char **argv) {
    using namespace fort_staging;
    auto &budget=budget_state();
    assert(&budget==ordinary_budget());
    size_t freed=0;
    assert(trim_cache(freed)==0 && fake::queries==0);
    if(argc>1 && std::strcmp(argv[1],"undrained")==0) {
        auto a=acquire(256,Role::Staging,false);
        a.slots->slots[0].pending=true; fake::outstanding=true; fake::fail_drain=true;
        fake::outstanding_host=a.slots->slots[0].host;
        a.slots.reset();
        // Completion was not proved: the DMA storage remains charged/alive.
        assert(fake::pinned==256 && budget.reserved==512 && fake::memory.size()==1);
        assert(fake::streams_drained==1 && fake::events_waited==0);
        return 0;
    }
    // Every staging allocation stage rolls back partial resources/reservation.
    for(int point=1;point<=6;++point) {
        fake::allocations=0; fake::fail_at=point;
        auto failed=acquire(1024,Role::Staging,false);
        assert(!failed.slots && failed.status==cudaErrorMemoryAllocation);
        assert(!fake::pinned && !fake::handles && fake::memory.empty() && !budget.reserved);
    }
    fake::fail_at=0;
    auto a=acquire(1024,Role::Staging,false);
    assert(a.slots && !a.slots->slots[0].device && !a.slots->slots[1].device && !fake::device_bytes);
    const auto first=a.slots->slots[0].host;
    assert(release(std::move(a.slots),freed)==0);
    a=acquire(512,Role::Staging,false);
    assert(a.reused && a.slots->slots[0].host==first && a.slots->capacity==1024);
    assert(release(std::move(a.slots),freed)==0);
    fake::device=1;
    a=acquire(512,Role::Staging,false);
    assert(a.slots->device==1 && fake::device==1);
    assert(release(std::move(a.slots),freed)==0);
    fake::device=0;
    assert(trim_cache(freed)==0 && fake::pinned==1024);
    fake::device=1;
    assert(trim_cache(freed)==0 && !fake::pinned && !budget.reserved);
    fake::device=0;
    constexpr size_t capacity=pinned_limit/4;
    a=acquire(capacity,Role::Staging,false);
    auto b=ordinary_acquire(capacity);
    assert(b.slots && budget.reserved==pinned_limit && fake::pinned==pinned_limit);
    assert(fake::device_bytes==2*capacity);
    auto unavailable=acquire(capacity,Role::Staging,false);
    assert(unavailable.exhausted && !unavailable.slots);
    assert(release(std::move(a.slots),freed)==0);
    a=acquire(capacity,Role::Staging,false);
    assert(a.reused && budget.reserved==pinned_limit);
    assert(release(std::move(a.slots),freed)==0);
    assert(release(std::move(b.slots),freed)==0);
    assert(budget.reserved==2*capacity && fake::pinned==2*capacity);
    assert(trim_cache(freed)==0 && !budget.reserved && !fake::pinned && !fake::device_bytes);
    assert(fake::peak<=pinned_limit && budget.peak<=pinned_limit);
    a=acquire(512,Role::Staging,false);
    a.slots->slots[0].pending=true; fake::outstanding=true;
    fake::outstanding_host=a.slots->slots[0].host;
    a.slots.reset(); // An event was never recorded; stream completion is required.
    assert(fake::streams_drained==1 && fake::events_waited==0);
    assert(!budget.reserved && !fake::pinned && !fake::handles && fake::memory.empty());
}
''')
    binary = directory / "pool"
    built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                            "-D__CUDACC__", "-I"+str(directory), str(directory / "ordinary.cpp"),
                            str(directory / "scoped.cpp"), "-o", str(binary)],
                           capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    return binary


def test_shared_pool_roles_budget_reuse_and_unrecorded_event_drain(pool_program):
    result = subprocess.run([str(pool_program)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_unproved_completion_retains_charged_dma_storage(pool_program):
    result = subprocess.run([str(pool_program), "undrained"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
