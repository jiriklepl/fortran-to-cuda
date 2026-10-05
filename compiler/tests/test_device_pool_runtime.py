"""Exercise native-pool ownership with a deterministic, thread-safe CUDA model."""

from __future__ import annotations

import os
import subprocess

import pytest

from compiler.emission.common.resources import read_common_header
from compiler.tests.test_memory_runtime import _run, _tool

FAKE_CUDA_POOL = r"""
#define __CUDACC__
#define __host__
#define __device__
#ifndef CUDART_VERSION
#define CUDART_VERSION 13000
#endif
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
using cudaError_t = int;
using cudaStream_t = int;
constexpr int cudaSuccess = 0, cudaErrorNotSupported = 801, cudaErrorInvalidValue = 1;
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaHostRegisterPortable = 1;
constexpr int cudaDevAttrMemoryPoolsSupported = 115, cudaMemAllocationTypePinned = 1;
constexpr int cudaMemHandleTypeNone = 0, cudaMemLocationTypeDevice = 1;
constexpr int cudaMemPoolAttrReleaseThreshold = 4;
struct cudaMemLocation { int type = 0, id = 0; };
struct cudaMemPoolProps { int allocType = 0, handleTypes = 0; cudaMemLocation location; };
namespace fake {
struct Pool {
    int device;
    std::uint64_t threshold = 0;
    std::size_t active = 0;
    std::map<std::size_t, std::vector<void*>> available;
};
struct Allocation { Pool* pool; std::size_t bytes; int device; };
inline std::mutex mutex;
inline std::unordered_map<void*, Allocation> live;
inline thread_local int current_device = 0, last_error = cudaSuccess;
// Inject an independently pending error, or an API that leaves the error state clean.
inline int reported_error = -1;
inline int error(int status) {
    if (status != cudaSuccess) last_error = reported_error < 0 ? status : reported_error;
    return status;
}
inline int driver_version = 13000, pools_supported = 1, create_error = 0, capability_error = 0;
inline int throw_pool_allocation = 0, allocation_error = 0;
inline bool throw_copy = false;
inline std::atomic<int> allocations{0}, pool_allocations{0}, backing_allocations{0};
inline std::atomic<int> frees{0}, uploads{0}, downloads{0}, syncs{0}, events{0};
inline std::atomic<int> pool_creates{0}, pool_destroys{0}, device_queries{0};
inline std::atomic<int> driver_queries{0}, capability_queries{0}, registrations{0}, unregistrations{0};
inline std::atomic<long long> clock{0};
inline std::atomic<std::uint64_t> last_threshold{0};
inline bool fail_sync = false;
inline void* backing(std::size_t bytes) {
    ++backing_allocations;
    clock += 2;
    void* pointer = std::malloc(bytes);
    assert(pointer);
    return pointer;
}
}
using cudaMemPool_t = fake::Pool*;
struct Event { long long timestamp = 0; };
using cudaEvent_t = Event*;
inline const char* cudaGetErrorString(int) { return "fake CUDA failure"; }
inline int cudaGetLastError() {
    const int result = fake::last_error; fake::last_error = cudaSuccess; return result;
}
inline int cudaDriverGetVersion(int* version) {
    ++fake::driver_queries; *version = fake::driver_version; return 0;
}
inline int cudaSetDevice(int device) { fake::current_device = device; return 0; }
inline int cudaGetDevice(int* device) {
    ++fake::device_queries; *device = fake::current_device; return 0;
}
inline int cudaDeviceGetAttribute(int* value, int attribute, int device) {
    assert(attribute == cudaDevAttrMemoryPoolsSupported && device == fake::current_device);
    ++fake::capability_queries;
    if (fake::capability_error) return fake::error(fake::capability_error);
    *value = fake::pools_supported; return 0;
}
inline int cudaMemPoolCreate(cudaMemPool_t* pool, const cudaMemPoolProps* props) {
    assert(props->allocType == cudaMemAllocationTypePinned);
    assert(props->handleTypes == cudaMemHandleTypeNone);
    assert(props->location.type == cudaMemLocationTypeDevice);
    assert(props->location.id == fake::current_device);
    if (fake::create_error) return fake::error(fake::create_error);
    *pool = new fake::Pool{fake::current_device}; ++fake::pool_creates; return 0;
}
inline int cudaMemPoolSetAttribute(cudaMemPool_t pool, int attribute, void* value) {
    assert(attribute == cudaMemPoolAttrReleaseThreshold);
    pool->threshold = *static_cast<std::uint64_t*>(value);
    fake::last_threshold = pool->threshold; return 0;
}
inline int cudaMemPoolDestroy(cudaMemPool_t pool) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    assert(pool->device == fake::current_device && pool->active == 0);
    for (auto& slot : pool->available) for (void* pointer : slot.second) {
        std::free(pointer); fake::clock += 3;
    }
    delete pool; ++fake::pool_destroys; return 0;
}
inline int cudaMalloc(void** pointer, std::size_t bytes) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    *pointer = fake::backing(bytes);
    assert(fake::live.emplace(*pointer, fake::Allocation{nullptr, bytes, fake::current_device}).second);
    ++fake::allocations; return 0;
}
inline int cudaMallocFromPoolAsync(void** pointer, std::size_t bytes, cudaMemPool_t pool, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    assert(stream == 0 && pool->device == fake::current_device);
    if (fake::allocation_error) return fake::error(fake::allocation_error);
    if (fake::throw_pool_allocation && fake::pool_allocations + 1 == fake::throw_pool_allocation)
        throw std::bad_alloc();
    auto& available = pool->available[bytes];
    if (available.empty()) *pointer = fake::backing(bytes);
    else { *pointer = available.back(); available.pop_back(); }
    assert(fake::live.emplace(*pointer, fake::Allocation{pool, bytes, fake::current_device}).second);
    ++pool->active; ++fake::pool_allocations; return 0;
}
inline int cudaFree(void* pointer) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    const auto found = fake::live.find(pointer);
    assert(found != fake::live.end() && found->second.device == fake::current_device);
    auto allocation = found->second;
    fake::live.erase(found);
    if (allocation.pool) {
        --allocation.pool->active;
        allocation.pool->available[allocation.bytes].push_back(pointer);
    } else { std::free(pointer); fake::clock += 3; }
    ++fake::frees; return 0;
}
inline int cudaMemcpy(void* destination, const void* source, std::size_t bytes, int direction) {
    if (fake::throw_copy) throw std::runtime_error("copy failed");
    if (direction == cudaMemcpyHostToDevice) ++fake::uploads; else ++fake::downloads;
    fake::clock += 5;
    std::memcpy(destination, source, bytes); return 0;
}
inline int cudaHostRegister(void*, std::size_t, unsigned int) { ++fake::registrations; return 0; }
inline int cudaHostUnregister(void*) { ++fake::unregistrations; return 0; }
inline int cudaDeviceSynchronize() { ++fake::syncs; return fake::fail_sync ? 1 : 0; }
inline int cudaEventCreate(Event** event) { ++fake::events; *event = new Event; return 0; }
inline int cudaEventDestroy(Event* event) { delete event; return 0; }
inline int cudaEventRecord(Event* event, int) { event->timestamp = fake::clock.load(); return 0; }
inline int cudaEventSynchronize(Event*) { return 0; }
inline int cudaEventElapsedTime(float* time, Event* start, Event* stop) {
    *time = static_cast<float>(stop->timestamp - start->timestamp); return 0;
}
"""

RUNTIME_MAIN = r"""
#include <string>
#include "common_functions.cuh"
using namespace generated_kernels;
using storage::AllocationPolicy;
void call(int value, std::size_t count = 4, AllocationPolicy policy = AllocationPolicy::pooled) {
    storage::Buffer<int> input({count}, policy), output({count}, policy);
    std::vector<int> source(count, value), destination(count);
    input.update_device(source.data(), {count});
    for (std::size_t index = 0; index < count; ++index) output.device_data()[index] = input.device_data()[index] + 7;
    output.device_written();
    CUCH(cudaGetLastError()); // Generated launch checks must not see a handled capability error.
    storage::synchronize();
    output.update_host(destination.data(), {count});
    for (int result : destination) assert(result == value + 7);
    storage::synchronize();
    input.release_completed(); output.release_completed();
    input.release_completed(); output.release_completed(); // Idempotent; slot destruction does no CUDA work.
}
int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "reuse";
    if (mode == "unsupported") fake::pools_supported = 0;
    if (mode == "old-driver") fake::driver_version = 11010;
    if (mode == "not-supported") fake::create_error = cudaErrorNotSupported;
    if (mode == "create-failure") fake::create_error = cudaErrorInvalidValue;
    if (mode == "attribute-unsupported") fake::capability_error = cudaErrorNotSupported;
    if (mode == "attribute-failure") fake::capability_error = cudaErrorInvalidValue;
    if (mode == "allocation-failure") fake::allocation_error = cudaErrorInvalidValue;
    if (mode == "unsupported-clean-error-state" || mode == "unsupported-pending-error") {
        fake::create_error = cudaErrorNotSupported;
        fake::reported_error = mode == "unsupported-clean-error-state" ? cudaSuccess : cudaErrorInvalidValue;
    }
    if (mode == "unsupported-live") {
        fake::pools_supported = 0;
        storage::Buffer<int> data({4}, AllocationPolicy::pooled);
        storage::trim_cache(); assert(fake::pool_creates == 0 && fake::pool_destroys == 0);
        storage::synchronize(); data.release_completed(); return 0;
    }
    if (mode == "partial-construction") {
        fake::throw_pool_allocation = 2;
        try { call(5); assert(false); } catch (const std::bad_alloc&) {}
        assert(fake::syncs == 1 && fake::frees == 1 && fake::live.empty());
        storage::trim_cache(); assert(fake::pool_destroys == 1); return 0;
    }
    if (mode == "empty") {
        storage::Buffer<int> empty({0}, AllocationPolicy::pooled);
        empty.release_completed(); storage::trim_cache();
        assert(fake::pool_creates == 0 && fake::allocations == 0 && fake::device_queries == 0);
        assert(fake::events == 0 && fake::syncs == 0);
        return 0;
    }
    if (mode == "trim-active") {
        storage::Buffer<int> live({4}, AllocationPolicy::pooled); storage::trim_cache(); return 1;
    }
    if (mode == "device-backstop") {
        {
            storage::Buffer<int> data({4}, AllocationPolicy::pooled);
            fake::current_device = 1;
        }
        assert(fake::current_device == 1 && fake::syncs == 1 && fake::frees == 1);
        fake::current_device = 0; storage::trim_cache(); return 0;
    }
    if (mode == "exception" || mode == "sync-failure") {
        try {
            storage::Buffer<int> data({4}, AllocationPolicy::pooled);
            fake::throw_copy = true;
            fake::fail_sync = mode == "sync-failure";
            int values[] = {1, 2, 3, 4}; data.update_device(values, {4});
        } catch (const std::runtime_error&) {}
        assert(fake::syncs == 1 && fake::frees == 1 && fake::live.empty());
        storage::trim_cache(); assert(fake::pool_destroys == 1); return 0;
    }
    if (mode == "concurrent") {
        constexpr int threads = 8;
        std::atomic<int> ready{0}; std::atomic<bool> finish{false};
        std::vector<std::thread> workers;
        for (int index = 0; index < threads; ++index) workers.emplace_back([&, index]() {
            fake::current_device = index % 2;
            storage::Buffer<int> data({4}, AllocationPolicy::pooled);
            ++ready;
            while (!finish.load()) std::this_thread::yield();
            int values[] = {index, index, index, index}, result[4] = {};
            data.update_device(values, {4}); data.update_host(result, {4});
            assert(result[0] == index);
            storage::synchronize(); data.release_completed();
        });
        while (ready.load() != threads) std::this_thread::yield();
        assert(fake::pool_creates == 2 && fake::backing_allocations == threads);
        finish = true;
        for (auto& worker : workers) worker.join();
        assert(fake::live.empty() && fake::frees == threads && fake::events == 0);
        storage::trim_cache(); fake::current_device = 1; storage::trim_cache();
        assert(fake::pool_destroys == 2); return 0;
    }
    if (mode == "profile") timing::reset_timing_vectors();
    for (int value = 0; value != 3; ++value) {
        timing::ProfiledCallGuard guard;
        timing::record_run();
        call(value);
    }
    assert(fake::uploads == 3 && fake::downloads == 3 && fake::frees == 6);
    const bool pooled = fake::pool_creates != 0;
    assert(fake::backing_allocations == (pooled ? 2 : 6));
    if (pooled) {
        assert(fake::pool_allocations == 6 && fake::driver_queries == 1 && fake::capability_queries == 1);
        const char* configured = std::getenv("FORT_CUDA_POOL_BYTES");
        assert(fake::last_threshold == (configured ? std::stoull(configured) : 268435456ULL));
    }
    if (mode == "profile") timing::print_timing_summary();
    else assert(fake::events == 0 && fake::syncs == 6);
    storage::trim_cache(); storage::trim_cache();
    if (pooled) {
        assert(fake::pool_destroys == 1);
        call(10, 7); assert(fake::pool_creates == 2 && fake::backing_allocations == 4);
        call(20, 7); assert(fake::backing_allocations == 4);
        call(30, 7, AllocationPolicy::dedicated); assert(fake::allocations == 2);
        storage::trim_cache();
    }
    assert(fake::live.empty());
}
"""


@pytest.fixture(scope="module")
def pool_runtime(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pool_runtime")
    (directory / "common_functions.cuh").write_text(read_common_header())
    (directory / "runtime.cpp").write_text(FAKE_CUDA_POOL + RUNTIME_MAIN)
    _run([_tool("g++"), "-std=c++17", "-pthread", "runtime.cpp", "-o", "run"], directory)
    _run([_tool("g++"), "-std=c++17", "-pthread", "-DCUDART_VERSION=11010", "runtime.cpp", "-o", "legacy"], directory)
    return directory


def run_pool(directory, mode="reuse", *, budget=None, executable="run", check=True):
    env = {key: value for key, value in os.environ.items() if key != "FORT_CUDA_POOL_BYTES"}
    env["FORT_RUNTIME_TRACE"] = "1"
    if budget is not None:
        env["FORT_CUDA_POOL_BYTES"] = budget
    return _run([f"./{executable}", mode], directory, env=env, check=check)


@pytest.mark.native
@pytest.mark.parametrize(
    "mode", ["reuse", "empty", "exception", "concurrent", "device-backstop", "partial-construction", "unsupported-live"]
)
def test_pool_ownership_and_lifecycle(pool_runtime, mode):
    run_pool(pool_runtime, mode)


@pytest.mark.native
@pytest.mark.parametrize(
    "mode", ["unsupported", "old-driver", "not-supported", "attribute-unsupported", "unsupported-clean-error-state"]
)
def test_pool_capability_falls_back_to_fresh_allocations(pool_runtime, mode):
    result = run_pool(pool_runtime, mode)
    assert "FORT_RUNTIME pool_create" not in result.stderr
    assert result.stderr.count("FORT_RUNTIME alloc") == 6


@pytest.mark.native
@pytest.mark.parametrize("budget", ["0", "1024", "18446744073709551615"])
def test_pool_retention_configuration(pool_runtime, budget):
    result = run_pool(pool_runtime, budget=budget)
    assert ("FORT_RUNTIME pool_create" in result.stderr) == (budget != "0")


@pytest.mark.native
@pytest.mark.parametrize("budget", ["", "-1", "+1", " 8", "2K", "18446744073709551616"])
def test_pool_rejects_invalid_retention_configuration(pool_runtime, budget):
    result = run_pool(pool_runtime, budget=budget, check=False)
    assert result.returncode != 0
    assert "FORT_CUDA_POOL_BYTES must be an unsigned decimal byte count" in result.stderr


@pytest.mark.native
@pytest.mark.parametrize(
    "mode",
    [
        "trim-active",
        "create-failure",
        "sync-failure",
        "attribute-failure",
        "allocation-failure",
        "unsupported-pending-error",
    ],
)
def test_pool_does_not_hide_lifecycle_or_cuda_failures(pool_runtime, mode):
    result = run_pool(pool_runtime, mode, check=False)
    assert result.returncode != 0
    if mode == "trim-active":
        assert "while ordinary calls are active" in result.stderr
    else:
        assert "CUDA error" in result.stderr
    assert "FORT_RUNTIME free" not in result.stderr


@pytest.mark.native
def test_old_cuda_headers_use_legacy_allocations(pool_runtime):
    result = run_pool(pool_runtime, executable="legacy")
    assert "FORT_RUNTIME pool_create" not in result.stderr


@pytest.mark.native
def test_pool_profiling_keeps_fresh_transfer_counts(pool_runtime):
    result = run_pool(pool_runtime, "profile")
    assert "calls:              3" in result.stdout
    assert "malloc_total_ms:    4" in result.stdout
    assert "h2d_total_ms:       15" in result.stdout
    assert "d2h_total_ms:       15" in result.stdout
    assert "free_total_ms:      0" in result.stdout


@pytest.mark.native
def test_cpu_pool_policy_and_trim_are_noops(tmp_path):
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "cpu.cpp").write_text(r"""
#include <cassert>
#include "common_functions.cuh"
int main() {
    using namespace generated_kernels::storage;
    Buffer<int> data({2}, AllocationPolicy::pooled);
    const int input[] = {3, 7}; int result[2] = {};
    data.update_device(input, {2}); data.update_host(result, {2});
    assert(result[0] == 3 && result[1] == 7);
    data.release_completed(); trim_cache();
}
""")
    _run([_tool("g++"), "-std=c++17", "cpu.cpp", "-o", "cpu"], tmp_path)
    result = subprocess.run(
        ["./cpu"], cwd=tmp_path, capture_output=True, text=True, env={**os.environ, "FORT_CUDA_POOL_BYTES": "invalid"}
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.native
def test_pool_metadata_allocation_failure_leaves_no_partial_registry_entry(tmp_path):
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "failure.cpp").write_text(
        FAKE_CUDA_POOL
        + r"""
#include "common_functions.cuh"
static int fail_after = -1;
void* operator new(std::size_t bytes) {
    if (fail_after >= 0 && fail_after-- == 0) throw std::bad_alloc();
    if (void* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
int main() {
    using namespace generated_kernels::storage;
    auto& registry = allocation_detail::pool_registry();
    // Fail state allocation, map-node allocation, and map-bucket allocation.
    for (int position = 0; position < 3; ++position) {
        fail_after = position;
        try { allocation_detail::PoolLease lease; assert(false); }
        catch (const std::bad_alloc&) {}
        fail_after = -1;
        assert(registry.devices.empty());
        trim_cache();
        assert(fake::pool_creates == 0 && fake::driver_queries == 0);
    }
    {
        Buffer<int> data({4}, AllocationPolicy::pooled);
        synchronize(); data.release_completed();
    }
    trim_cache(); assert(fake::pool_creates == 1 && fake::pool_destroys == 1);
}
"""
    )
    _run([_tool("g++"), "-std=c++17", "-pthread", "failure.cpp", "-o", "failure"], tmp_path)
    _run(["./failure"], tmp_path)
