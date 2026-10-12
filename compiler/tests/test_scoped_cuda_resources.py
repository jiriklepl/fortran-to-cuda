"""Bounded idle CUDA resources; no buffer identity or freshness is cached."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.common.resources import read_scoped_runtime

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
MOCK = r'''
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=1,cudaErrorNotReady=2,cudaErrorUnknown=3,
    cudaErrorInvalidValue=4,cudaStreamNonBlocking=1,cudaDevAttrMemoryPoolsSupported=1,
    cudaMemAllocationTypePinned=1,cudaMemLocationTypeDevice=1,
    cudaMemPoolAttrReleaseThreshold=1,cudaMemPoolAttrUsedMemCurrent=2,cudaMemPoolAttrReservedMemCurrent=3;
struct Stream {int device;};
struct Pool {int device;uint64_t reserved=0,used=0,threshold=0;};
using cudaStream_t=Stream*;
using cudaMemPool_t=Pool*;
struct cudaMemPoolProps {int allocType; struct {int type,id;} location;};
namespace fake {
inline int device=0,streams=0,pools=0,created=0,queries=0,fail_at=0,stage=0,query_status=0;
inline bool supported=true,honor_trim=true;
inline bool fail(){return fail_at && ++stage==fail_at;}
}
inline int cudaStreamCreateWithFlags(cudaStream_t *out,int) {
    if(fake::fail())return cudaErrorMemoryAllocation;
    *out=new Stream{fake::device};++fake::streams;++fake::created;return 0;
}
inline int cudaStreamDestroy(cudaStream_t stream) {
    assert(stream && stream->device==fake::device);--fake::streams;delete stream;return 0;
}
inline int cudaStreamQuery(cudaStream_t stream) {
    assert(stream && stream->device==fake::device);++fake::queries;return fake::query_status;
}
inline int cudaDeviceGetAttribute(int *out,int,int device) {
    assert(device==fake::device);if(fake::fail())return cudaErrorUnknown;*out=fake::supported;return 0;
}
inline int cudaMemPoolCreate(cudaMemPool_t *out,const cudaMemPoolProps *props) {
    assert(props->location.id==fake::device);if(fake::fail())return cudaErrorMemoryAllocation;
    *out=new Pool{fake::device};++fake::pools;return 0;
}
inline int cudaMemPoolSetAttribute(cudaMemPool_t pool,int,const void *value) {
    assert(pool->device==fake::device);if(fake::fail())return cudaErrorUnknown;
    pool->threshold=*static_cast<const uint64_t*>(value);return 0;
}
inline int cudaMemPoolGetAttribute(cudaMemPool_t pool,int attr,void *out) {
    assert(pool->device==fake::device);
    *static_cast<uint64_t*>(out)=attr==cudaMemPoolAttrUsedMemCurrent?pool->used:pool->reserved;return 0;
}
inline int cudaMemPoolTrimTo(cudaMemPool_t pool,uint64_t bytes) {
    assert(pool->device==fake::device);if(fake::honor_trim)pool->reserved=std::min(pool->reserved,bytes);return 0;
}
inline int cudaMemPoolDestroy(cudaMemPool_t pool) {
    assert(pool && pool->device==fake::device);--fake::pools;delete pool;return 0;
}
'''
PROGRAM = r'''
#include "scoped_cuda_resources.hpp"
#include <cstring>
int main(int argc,char **argv) {
    using namespace fort_scoped::cuda_resources;
    assert(argc==2);
    Resources a,b,c;
    assert(!acquire(0,a) && fake::created==0 && fake::queries==0);
    const char *mode=argv[1];
    if(!std::strcmp(mode,"failures")) {
        for(int i=1;i<=4;++i) {
            fake::stage=0;fake::fail_at=i;
            assert(create(0,a)!=cudaSuccess);
            assert(destroy(a)==0 && !a.stream && !a.pool && !fake::streams && !fake::pools);
            assert(!acquire(0,b));
        }
        return 0;
    }
    if(!std::strcmp(mode,"no-pools"))fake::supported=false;
    assert(create(0,a)==0 && a.stream);
#if CUDART_VERSION >= 11020
    assert(bool(a.pool)==fake::supported);
    if(a.pool)assert(a.pool->threshold==retained_limit);
#else
    assert(!a.pool);
#endif
    if(!std::strcmp(mode,"pending") || !std::strcmp(mode,"poisoned")) {
        fake::query_status=!std::strcmp(mode,"pending")?cudaErrorNotReady:cudaErrorUnknown;
        assert(retire(0,a)==fake::query_status && a.stream && !acquire(0,b));
        fake::query_status=0;assert(destroy(a)==0);return 0;
    }
    if(!std::strcmp(mode,"live-allocation")) {
        a.pool->used=16;
        assert(retire(0,a)==cudaErrorInvalidValue && !acquire(0,b) && a.pool->used==16);
        a.pool->used=0;assert(destroy(a)==0);return 0;
    }
    if(!std::strcmp(mode,"oversize")) {
        a.pool->reserved=retained_limit+1;fake::honor_trim=false;
        assert(retire(0,a)==0 && !a.stream && !a.pool && !acquire(0,b));
        assert(!fake::streams && !fake::pools);return 0;
    }
    const auto stream=a.stream;
    if(a.pool)a.pool->reserved=2*retained_limit;
    assert(retire(0,a)==0 && !a.stream && !a.pool);
    assert(acquire(0,b) && b.stream==stream);
    assert(b.retained_bytes<=retained_limit);
    // A second active invocation gets independent resources. On return only
    // one bundle remains cached, irrespective of close order.
    assert(!acquire(0,a) && create(0,a)==0 && a.stream!=b.stream);
    assert(retire(0,b)==0 && retire(0,a)==0 && fake::streams==1);
    fake::device=1;
    assert(!acquire(1,c) && create(1,c)==0 && retire(1,c)==0);
    assert(fake::streams==2);
    fake::device=0;assert(acquire(0,a) && destroy(a)==0);
    fake::device=1;assert(acquire(1,c) && destroy(c)==0);
    assert(!fake::streams && !fake::pools);
}
'''


@pytest.fixture(scope="module")
def resource_programs(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("idle_cuda_resources")
    (directory / "cuda_runtime.h").write_text(MOCK)
    (directory / "main.cpp").write_text(PROGRAM)
    binaries = {}
    for version in (11020, 11000):
        binary = directory / f"resources-{version}"
        result = subprocess.run([
            compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
            f"-DCUDART_VERSION={version}", "-I", str(directory), "-I", str(RUNTIME),
            str(directory / "main.cpp"), "-o", str(binary),
        ], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        binaries[version] = binary
    return binaries


@pytest.mark.parametrize("mode", ["reuse", "failures", "pending", "poisoned", "live-allocation", "oversize", "no-pools"])
def test_completed_resources_are_reused_with_strict_idle_bound(resource_programs, mode):
    result = subprocess.run([str(resource_programs[11020]), mode], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_stream_only_reuse_on_older_cuda(resource_programs):
    result = subprocess.run([str(resource_programs[11000]), "reuse"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_resource_cache_is_published_by_independent_runtime_interface():
    sources, manifest = read_scoped_runtime()
    assert sources["scoped_cuda_resources.hpp"] == (RUNTIME / "scoped_cuda_resources.hpp").read_text()
    assert "scoped_cuda_resources.hpp" in manifest["headers"]
    cache = manifest["idle_execution_resources"]
    assert cache["limit_per_device"] == 1
    assert cache["retained_device_bytes_limit"] == 64 * 1024 * 1024
    assert "invocation-local" in cache["freshness"]
