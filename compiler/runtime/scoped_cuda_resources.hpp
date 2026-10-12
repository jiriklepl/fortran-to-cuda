// Idle execution resources only: buffer identities and coherence never survive
// a scoped invocation. Callers own error classification and device selection.
#ifndef FORT_SCOPED_CUDA_RESOURCES_HPP
#define FORT_SCOPED_CUDA_RESOURCES_HPP
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
namespace fort_scoped::cuda_resources {
constexpr uint64_t retained_limit = 64ULL * 1024 * 1024;
struct Resources {
    cudaStream_t stream = nullptr;
    cudaMemPool_t pool = nullptr;
    uint64_t retained_bytes = 0;
};
struct Cache {
    std::mutex mutex;
    std::unordered_map<int, Resources> idle;
};
inline Cache &cache() {
    // CUDA runtime shutdown owns process teardown. Never call CUDA from static
    // destructors after the driver has shut down (as for pinned staging).
    static auto *state = new Cache;
    return *state;
}
inline bool acquire(int device, Resources &out) {
    auto &state = cache();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto found = state.idle.find(device);
    if (found == state.idle.end()) return false;
    out = found->second;
    state.idle.erase(found);
    return true;
}
inline cudaError_t create(int device, Resources &out) {
    (void)device; // Stream-only CUDA versions have no device-local pool API.
    auto status = cudaStreamCreateWithFlags(&out.stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) return status;
#if CUDART_VERSION >= 11020
    int supported = 0;
    status = cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device);
    if (status != cudaSuccess || !supported) return status;
    cudaMemPoolProps properties{};
    properties.allocType = cudaMemAllocationTypePinned;
    properties.location.type = cudaMemLocationTypeDevice;
    properties.location.id = device;
    status = cudaMemPoolCreate(&out.pool, &properties);
    if (status != cudaSuccess) return status;
    // Keep recently freed pages across normal stream completion. This is only
    // a release threshold; retire() also enforces the actual idle byte cap.
    uint64_t threshold = retained_limit;
    status = cudaMemPoolSetAttribute(out.pool, cudaMemPoolAttrReleaseThreshold, &threshold);
#endif
    return status;
}
inline cudaError_t destroy(Resources &resources) {
    cudaError_t status = cudaSuccess;
#if CUDART_VERSION >= 11020
    if (resources.pool) {
        status = cudaMemPoolDestroy(resources.pool);
        if (status == cudaSuccess) resources.pool = nullptr;
    }
#endif
    if (resources.stream) {
        const auto next = cudaStreamDestroy(resources.stream);
        if (status == cudaSuccess) status = next;
        if (next == cudaSuccess) resources.stream = nullptr;
    }
    return status;
}
inline cudaError_t retire(int device, Resources &resources) {
    // Only completely drained, allocation-free, healthy resources can enter
    // the cache. cudaStreamQuery also surfaces deferred execution failures.
    auto status = cudaStreamQuery(resources.stream);
    if (status != cudaSuccess) return status;
#if CUDART_VERSION >= 11020
    if (resources.pool) {
        uint64_t used = 0;
        status = cudaMemPoolGetAttribute(resources.pool, cudaMemPoolAttrUsedMemCurrent, &used);
        if (status != cudaSuccess) return status;
        if (used) return cudaErrorInvalidValue;
        status = cudaMemPoolTrimTo(resources.pool, retained_limit);
        if (status != cudaSuccess) return status;
        status = cudaMemPoolGetAttribute(resources.pool, cudaMemPoolAttrReservedMemCurrent,
                                         &resources.retained_bytes);
        if (status != cudaSuccess) return status;
        // Driver granularity can prevent trimming to the requested threshold.
        // In that case do not retain this bundle at all.
        if (resources.retained_bytes > retained_limit) return destroy(resources);
    }
#endif
    auto &state = cache();
    std::lock_guard<std::mutex> lock(state.mutex);
    // At most one idle stream/pool per device. Concurrent active invocations
    // keep independent streams; a second returning bundle is simply retired.
    if (state.idle.count(device)) return destroy(resources);
    try { state.idle.emplace(device, resources); }
    catch (...) { return destroy(resources); } // A cache miss cannot fail work.
    resources = {};
    return cudaSuccess;
}
} // namespace fort_scoped::cuda_resources
#endif
