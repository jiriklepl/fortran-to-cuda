// CUDA allocation policy is independent of buffer coherence and execution
// plans.
#ifndef FORT_RUNTIME_ALLOCATION_HPP
#define FORT_RUNTIME_ALLOCATION_HPP
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace generated_kernels::storage {

inline void trace(const char *operation, std::size_t bytes);
[[noreturn]] inline void fail(const char *message);

enum class AllocationPolicy { dedicated, pooled };

#ifdef __CUDACC__
namespace allocation_detail {

inline std::uint64_t retention_bytes() {
    static const std::uint64_t bytes = []() {
        const char *text = std::getenv("FORT_CUDA_POOL_BYTES");
        if (!text)
            return std::uint64_t{268435456};
        if (!*text)
            fail("FORT_CUDA_POOL_BYTES must be an unsigned decimal byte count");
        std::uint64_t value = 0;
        for (; *text; ++text) {
            if (*text < '0' || *text > '9' || value > (std::numeric_limits<std::uint64_t>::max() - (*text - '0')) / 10)
                fail("FORT_CUDA_POOL_BYTES must be an unsigned decimal byte count");
            value = value * 10 + (*text - '0');
        }
        return value;
    }();
    return bytes;
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
struct PoolState {
    explicit PoolState(int ordinal) : device(ordinal) {}
    int device;
    cudaMemPool_t pool = nullptr;
    std::size_t active_leases = 0;
    std::once_flag initialized;
};

struct PoolRegistry {
    std::mutex mutex;
    std::unordered_map<int, std::shared_ptr<PoolState>> devices;
};

inline PoolRegistry &pool_registry() {
    // CUDA teardown order is controlled by the application, never a static
    // destructor.
    static auto *registry = new PoolRegistry;
    return *registry;
}

inline void consume_unsupported_error() {
    // An expected capability failure must not poison the next launch check.
    // Preserve visibility of any different pending error instead of clearing blindly.
    const auto pending = cudaGetLastError();
    if (pending != cudaErrorNotSupported)
        CUCH(pending);
}

inline void initialize_pool(PoolState &state) {
    int driver_version = 0;
    CUCH(cudaDriverGetVersion(&driver_version));
    int supported = 0;
    if (driver_version >= 11020) {
        const auto status = cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, state.device);
        if (status == cudaErrorNotSupported) {
            consume_unsupported_error();
            supported = 0;
        } else
            CUCH(status);
    }
    if (!supported) {
        trace("pool_fallback", 0);
        return;
    }
    cudaMemPoolProps properties{};
    properties.allocType = cudaMemAllocationTypePinned;
    properties.handleTypes = cudaMemHandleTypeNone;
    properties.location.type = cudaMemLocationTypeDevice;
    properties.location.id = state.device;
    const auto status = cudaMemPoolCreate(&state.pool, &properties);
    if (status == cudaErrorNotSupported) {
        consume_unsupported_error();
        state.pool = nullptr;
        trace("pool_fallback", 0);
        return;
    }
    CUCH(status);
    auto threshold = retention_bytes();
    CUCH(cudaMemPoolSetAttribute(state.pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    trace("pool_create", threshold);
}

class PoolLease {
    std::shared_ptr<PoolState> state_;

  public:
    PoolLease() {
        int device = 0;
        CUCH(cudaGetDevice(&device));
        auto &registry = pool_registry();
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            auto found = registry.devices.find(device);
            if (found == registry.devices.end())
                found = registry.devices.emplace(device, std::make_shared<PoolState>(device)).first;
            state_ = found->second;
            ++state_->active_leases;
        }
        try {
            std::call_once(state_->initialized, [&]() { initialize_pool(*state_); });
        } catch (...) {
            release();
            throw;
        }
    }
    PoolLease(const PoolLease &) = delete;
    PoolLease &operator=(const PoolLease &) = delete;
    ~PoolLease() { release(); }
    cudaMemPool_t pool() const { return state_->pool; }
    int device() const { return state_->device; }

  private:
    void release() {
        if (!state_)
            return;
        auto &registry = pool_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        --state_->active_leases;
        state_.reset();
    }
};
#endif

class DeviceAllocation {
    void *pointer_ = nullptr;
    bool pooled_ = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    std::unique_ptr<PoolLease> lease_;
#endif
    void free_pointer() {
        CUCH(cudaFree(pointer_));
        pointer_ = nullptr;
        trace("free", 0);
        if (pooled_)
            trace("pool_free", 0);
    }
    void release_lease() {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        lease_.reset();
#endif
    }
    void cleanup(bool profile_free = false) {
        if (pointer_) {
            // Unlike legacy cudaFree, pooled cudaFree never waits for outstanding
            // work.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
            int previous_device = 0;
            if (pooled_) {
                CUCH(cudaGetDevice(&previous_device));
                if (previous_device != lease_->device())
                    CUCH(cudaSetDevice(lease_->device()));
                CUCH(cudaDeviceSynchronize());
            }
#endif
            if (profile_free)
                timing::measure_free([&]() { free_pointer(); });
            else
                free_pointer();
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
            if (pooled_ && previous_device != lease_->device())
                CUCH(cudaSetDevice(previous_device));
#endif
        }
        release_lease();
    }

  public:
    DeviceAllocation(std::size_t bytes, AllocationPolicy policy) {
        if (!bytes)
            return;
        try {
            timing::measure_alloc([&]() {
                if (policy == AllocationPolicy::pooled && retention_bytes()) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
                    lease_ = std::make_unique<PoolLease>();
                    if (lease_->pool()) {
                        pooled_ = true;
                        CUCH(cudaMallocFromPoolAsync(&pointer_, bytes, lease_->pool(), 0));
                        trace("pool_alloc", bytes);
                    }
#endif
                }
                if (!pooled_) {
                    release_lease();
                    CUCH(cudaMalloc(&pointer_, bytes));
                }
                trace("alloc", bytes);
            });
        } catch (...) {
            cleanup();
            throw;
        }
    }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    ~DeviceAllocation() { cleanup(std::uncaught_exceptions() == 0); }
    void *get() const { return pointer_; }
    void release_completed() {
        // The caller's memory plan has synchronized all accesses before this
        // operation.
        if (pointer_)
            timing::measure_free([&]() { free_pointer(); });
        release_lease();
    }
};
} // namespace allocation_detail
#endif

inline void trim_cache() {
#if defined(__CUDACC__) && defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    auto &registry = allocation_detail::pool_registry();
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.devices.empty())
            return;
    }
    int device = 0;
    CUCH(cudaGetDevice(&device));
    std::shared_ptr<allocation_detail::PoolState> state;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = registry.devices.find(device);
        if (found == registry.devices.end())
            return;
        if (found->second->active_leases)
            fail("cannot trim the CUDA pool while ordinary calls are active");
        state = found->second;
        registry.devices.erase(found);
    }
    if (state->pool) {
        CUCH(cudaMemPoolDestroy(state->pool));
        trace("pool_destroy", 0);
    }
#endif
}
} // namespace generated_kernels::storage
#endif // FORT_RUNTIME_ALLOCATION_HPP
