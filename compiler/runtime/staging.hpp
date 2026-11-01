// Shared pinned storage only. Callers supply tracing and execution-error policy.
#ifndef FORT_STAGING_HPP
#define FORT_STAGING_HPP
#ifdef __CUDACC__
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace fort_staging {
constexpr std::size_t pinned_limit = 64ULL * 1024 * 1024;
// A single shared pool bounds both active leases and its one idle pair.
// The compact device reservation is independent of logical payload ownership.
constexpr std::size_t device_limit = pinned_limit;
enum class Role { Compact, Staging };
struct Budget;
struct Slot {
    unsigned char *host = nullptr, *device = nullptr;
    cudaStream_t stream{};
    cudaEvent_t complete{};
    bool pending = false;
};
struct Slots {
    Slot slots[2];
    const std::size_t capacity;
    const int device;
    const Role role;
    Budget *budget;
    bool charged = false;
    std::size_t allocated_devices = 0;
    Slots(std::size_t bytes, int ordinal, Role kind, Budget &owner)
        : capacity(bytes), device(ordinal), role(kind), budget(&owner) {}
    cudaError_t allocate() {
        for (auto &slot : slots) {
            auto status = cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking);
            if (status != cudaSuccess) return status;
            status = cudaEventCreateWithFlags(&slot.complete, cudaEventDisableTiming);
            if (status != cudaSuccess) return status;
            status = cudaMallocHost(reinterpret_cast<void **>(&slot.host), capacity);
            if (status != cudaSuccess) return status;
            if (role == Role::Compact) {
                status = cudaMalloc(reinterpret_cast<void **>(&slot.device), capacity);
                if (status != cudaSuccess) return status;
                ++allocated_devices;
            }
        }
        return cudaSuccess;
    }
    cudaError_t cleanup(std::size_t &freed_devices) noexcept {
        int previous = -1;
        auto first = cudaGetDevice(&previous);
        if (first != cudaSuccess) return first;
        if (previous != device) {
            first = cudaSetDevice(device);
            if (first != cudaSuccess) return first;
        }
        auto remember = [&](cudaError_t status) { if (first == cudaSuccess) first = status; };
        for (auto &slot : slots) {
            if (slot.pending) {
                // A failed event record may leave an unrecorded event. Drain
                // the stream itself before freeing any pinned allocation.
                const auto status = cudaStreamSynchronize(slot.stream);
                remember(status);
                if (status == cudaSuccess) slot.pending = false;
                // Retain charged storage if completion cannot be established.
                // Never free pinned memory that a copy may still be using.
                if (slot.pending) continue;
            }
            if (slot.device) {
                const auto status = cudaFree(slot.device); remember(status);
                if (status == cudaSuccess) { slot.device = nullptr; ++freed_devices; }
            }
            if (slot.host) {
                const auto status = cudaFreeHost(slot.host); remember(status);
                if (status == cudaSuccess) slot.host = nullptr;
            }
            if (slot.complete) {
                const auto status = cudaEventDestroy(slot.complete); remember(status);
                if (status == cudaSuccess) slot.complete = {};
            }
            if (slot.stream) {
                const auto status = cudaStreamDestroy(slot.stream); remember(status);
                if (status == cudaSuccess) slot.stream = {};
            }
        }
        if (previous != device) remember(cudaSetDevice(previous));
        return first;
    }
    ~Slots();
};
struct Budget {
    std::mutex mutex;
    std::condition_variable released;
    std::size_t reserved = 0, peak = 0, device_reserved = 0, device_peak = 0;
    std::uint64_t revision = 0;
    std::unique_ptr<Slots> idle;
};
// One named inline singleton across generated CUDA units and the scoped object.
// The metadata survives process teardown; CUDA teardown is explicit.
inline Budget &budget_state() {
    static auto *state = new Budget;
    return *state;
}
inline Slots::~Slots() {
    std::size_t ignored = 0;
    const auto status = cleanup(ignored);
    if (charged && status == cudaSuccess) {
        { std::lock_guard<std::mutex> lock(budget->mutex); budget->reserved -= 2 * capacity;
          if(role==Role::Compact)budget->device_reserved-=2*capacity;
          ++budget->revision; }
        budget->released.notify_all();
    }
}
struct Result {
    cudaError_t status = cudaSuccess;
    std::unique_ptr<Slots> slots;
    bool reused = false, exhausted = false, unpriced_idle = false;
    std::size_t freed_devices = 0, evicted_devices = 0, allocated_devices = 0;
};
// mutex held; an idle pair is already complete. Never wait for active work here.
inline cudaError_t evict_idle(Budget &budget, std::size_t &freed_devices) {
    if (!budget.idle) return cudaSuccess;
    const auto status = budget.idle->cleanup(freed_devices);
    if (status != cudaSuccess) return status;
    budget.reserved -= 2 * budget.idle->capacity;
    if(budget.idle->role==Role::Compact)budget.device_reserved-=2*budget.idle->capacity;
    budget.idle->charged = false;
    budget.idle.reset();
    ++budget.revision;
    return cudaSuccess;
}
inline Result acquire(std::size_t bytes, Role role, bool blocking,
                      std::size_t available_device_bytes=device_limit, bool reuse_or_empty_only=false) {
    Result result;
    if (!bytes || bytes > pinned_limit / 2) { result.exhausted = true; return result; }
    const auto device_bytes=role==Role::Compact ? 2*bytes : 0;
    const auto maximum_device_bytes=std::min(device_limit,available_device_bytes);
    if(device_bytes>maximum_device_bytes) { result.exhausted=true; return result; }
    int device = -1;
    result.status = cudaGetDevice(&device);
    if (result.status != cudaSuccess) return result;
    auto &budget = budget_state();
    {
        std::unique_lock<std::mutex> lock(budget.mutex);
        for (;;) {
            if (budget.idle && budget.idle->device == device && budget.idle->role == role &&
                budget.idle->capacity >= bytes &&
                (role!=Role::Compact || budget.device_reserved<=maximum_device_bytes)) {
                result.slots = std::move(budget.idle); result.reused = true; ++budget.revision; return result;
            }
            // Measured cold Compact acquisition covers an empty idle cache.
            // Another caller may install an incompatible pair after pricing.
            if(reuse_or_empty_only && budget.idle) {
                result.exhausted=true;result.unpriced_idle=true;return result;
            }
            result.status = evict_idle(budget, result.freed_devices);
            if (result.status != cudaSuccess) return result;
            if (budget.reserved <= pinned_limit - 2 * bytes &&
                budget.device_reserved<=maximum_device_bytes-device_bytes) break;
            if (!blocking) { result.exhausted = true; return result; }
            budget.released.wait(lock);
        }
        budget.reserved += 2 * bytes;
        budget.peak = std::max(budget.peak, budget.reserved);
        budget.device_reserved+=device_bytes;
        budget.device_peak=std::max(budget.device_peak,budget.device_reserved);
        ++budget.revision;
    }
    try {
        result.slots = std::make_unique<Slots>(bytes, device, role, budget);
    } catch (...) {
        { std::lock_guard<std::mutex> lock(budget.mutex); budget.reserved -= 2 * bytes;
          budget.device_reserved-=device_bytes; ++budget.revision; }
        budget.released.notify_all();
        throw;
    }
    result.slots->charged = true;
    result.evicted_devices = result.freed_devices;
    result.status = result.slots->allocate();
    result.allocated_devices = result.slots->allocated_devices;
    if (result.status != cudaSuccess) {
        const auto cleanup = result.slots->cleanup(result.freed_devices);
        if (cleanup != cudaSuccess) result.status = cleanup;
        result.slots.reset(); // destructor rolls back the reservation after cleanup.
    }
    return result;
}
inline cudaError_t release(std::unique_ptr<Slots> slots, std::size_t &freed_devices) {
    if (!slots) return cudaSuccess;
    for (const auto &slot : slots->slots)
        if (slot.pending) return cudaErrorNotReady;
    auto &budget = budget_state();
    {
        std::lock_guard<std::mutex> lock(budget.mutex);
        const auto status = evict_idle(budget, freed_devices);
        if (status != cudaSuccess) return status;
        budget.idle = std::move(slots);
        ++budget.revision;
    }
    budget.released.notify_all();
    return cudaSuccess;
}
inline cudaError_t trim_cache(std::size_t &freed_devices) {
    auto &budget = budget_state();
    std::lock_guard<std::mutex> lock(budget.mutex);
    if (!budget.idle) return cudaSuccess; // Native-only programs do not query CUDA.
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess && budget.idle->device == device) status = evict_idle(budget, freed_devices);
    budget.released.notify_all();
    return status;
}
struct Usage { std::size_t reserved, peak, device_reserved, device_peak; };
inline Usage usage() {
    auto &budget = budget_state();
    std::lock_guard<std::mutex> lock(budget.mutex);
    return {budget.reserved, budget.peak, budget.device_reserved, budget.device_peak};
}
// Metadata only. The caller must reprice the actual acquired lease before
// submitting work: another context may consume or replace this idle pair.
struct Snapshot {
    std::uint64_t revision;
    std::size_t pinned_reserved, device_reserved, idle_capacity;
    int idle_device;
    Role idle_role;
};
// Idle storage may be evicted before acquisition; active and allocating
// leases remain charged against both quotas regardless of the requesting domain.
inline std::size_t compact_available(const Snapshot &s) {
    const auto pinned_active=s.pinned_reserved-2*s.idle_capacity;
    const auto device_active=s.device_reserved-(s.idle_role==Role::Compact?2*s.idle_capacity:0);
    return std::min(pinned_active>pinned_limit?0:pinned_limit-pinned_active,
                    device_active>device_limit?0:device_limit-device_active);
}
inline Snapshot snapshot() {
    auto &budget=budget_state();
    std::lock_guard<std::mutex> lock(budget.mutex);
    return {budget.revision,budget.reserved,budget.device_reserved,
            budget.idle ? budget.idle->capacity : 0,
            budget.idle ? budget.idle->device : -1,
            budget.idle ? budget.idle->role : Role::Staging};
}
} // namespace fort_staging
#endif
#endif
