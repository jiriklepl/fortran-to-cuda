/* Shared state implementation: compile once with nvcc, or with
 * -DFORT_SCOPE_CPU_TEST for the isolated coherence reference backend. */
#include "scoped_runtime.h"
#include "section_copy.hpp"
#include "scoped_regions.hpp"
#include "scoped_planning.hpp"
#define FORT_SCOPE_TEAM_OBSERVER_IMPLEMENTATION
#include "scoped_team_observer.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef FORT_SCOPE_CPU_TEST
#include <cuda_runtime.h>
#include "staging.hpp"
#endif

namespace {
using namespace fort_scoped::coherence;
struct Error : std::runtime_error {
    int status;
    Error(int status_, const char *message) : std::runtime_error(message), status(status_) {}
};
thread_local char last_error[512]{}; // Diagnostic only; never an implicit context.
void diagnostic(const char *message) noexcept {
    std::strncpy(last_error, message, sizeof(last_error)-1);
    last_error[sizeof(last_error)-1] = '\0';
}
void require(bool condition, int status, const char *message) {
    if (!condition) throw Error(status, message);
}
size_t multiply(size_t a, size_t b) {
    require(!b || a <= std::numeric_limits<size_t>::max() / b, FORT_SCOPE_ARGUMENT, "array byte extent overflow");
    return a * b;
}
struct Buffer {
    fort_buffer_t handle;
    uint64_t identity, generation;
    uint32_t type;
    size_t element_bytes, bytes;
    void *host, *device = nullptr;
    std::vector<size_t> extents, strides;
    std::vector<int64_t> lower;
    Region initialized, host_current, device_current;
    std::optional<Prepared> prepared;
    Box full() const { return {std::vector<size_t>(extents.size(), 0), extents}; }
};
struct Context {
    explicit Context(int ordinal) : device(ordinal) { transfer_stats.version = FORT_SCOPE_TRANSFER_ABI_VERSION; }
    int device;
    bool ready = false, pending = false, poisoned = false, closed = false;
    size_t device_budget = std::numeric_limits<size_t>::max();
    std::mutex mutex;
    std::unordered_map<fort_buffer_t, std::unique_ptr<Buffer>> buffers;
    std::unordered_map<uint64_t, fort_buffer_t> identities;
    fort_scope_stats stats{};
    fort_scope_transfer_stats transfer_stats{};
    std::optional<fort_scope_batch_costs> transfer_costs;
    std::optional<fort_scope_team_costs> team_costs;
    std::optional<fort_scope_batch_report> batch_report;
    bool batch_active = false;
    std::vector<fort_scoped::planning::Operation> plan;
    std::vector<bool> schedule;
    size_t worker_cursor = 0;
    bool plan_recording = false, plan_installed = false;
    uint32_t endpoint_mode = FORT_SCOPE_PLAN_COMPLETE;
    bool owner_create_accounted = false, owner_estimate_available = true;
    bool owner_native_common_compute_excluded = false;
    size_t registrations_incurred = 0;
    uint64_t owner_segments = 0;
    double owner_execution_seconds = 0, owner_terminal_seconds = 0;
    bool segment_cost_pending = false;
    std::optional<fort_scope_plan_report> last_report;
    uint64_t query_generation = 1, state_generation = 1;
    struct Proof {
        uint64_t query_generation, state_generation;
        fort_scoped::planning::DefinitionValidation result;
    };
    std::optional<Proof> definition_proof;
    struct QuerySnapshot {
        uint64_t query_generation, state_generation;
        fort_scoped::planning::Inputs input;
    };
    std::optional<QuerySnapshot> planning_snapshot;
    std::optional<fort_scoped::planning::Result> preview;
    uint64_t preview_query_generation = 0, preview_state_generation = 0;
    std::array<unsigned char, sizeof(fort_scope_plan_costs)> preview_costs{};
#ifndef FORT_SCOPE_CPU_TEST
    cudaStream_t stream = nullptr;
    cudaMemPool_t pool = nullptr;
#endif
};
bool actual_team_matches(const Context &c) {
#ifdef _OPENMP
    return c.team_costs && omp_get_level() == int(c.team_costs->expected_omp_level) &&
        omp_get_num_threads() == int(c.team_costs->cpu_threads);
#else
    (void)c;
    return false;
#endif
}
std::mutex driver_mutex;
std::unordered_set<int> initialized_devices;
std::mutex registry_mutex;
std::mutex trace_mutex;
std::unordered_map<fort_scope_t, std::shared_ptr<Context>> contexts;
uint64_t next_token = 1;
uint64_t token() {
    std::lock_guard<std::mutex> lock(registry_mutex);
    require(next_token < std::numeric_limits<int64_t>::max(), FORT_SCOPE_RESOURCE, "scope token range exhausted");
    return next_token++;
}
std::shared_ptr<Context> lookup(fort_scope_t handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto found = contexts.find(handle);
    require(found != contexts.end(), FORT_SCOPE_STALE, "invalid or stale scope handle");
    return found->second;
}
Buffer &buffer(Context &c, fort_buffer_t handle) {
    auto found = c.buffers.find(handle);
    require(found != c.buffers.end(), FORT_SCOPE_STALE, "invalid, foreign, or stale buffer handle");
    return *found->second;
}
void trace(const char *operation, const Buffer *b = nullptr, size_t bytes = 0) {
    const char *enabled = std::getenv("FORT_RUNTIME_TRACE");
    if (!enabled || std::strcmp(enabled, "1")) return;
    std::lock_guard<std::mutex> lock(trace_mutex);
    std::cerr << "FORT_SCOPED " << operation;
    if (b) std::cerr << " buffer=" << b->identity << " generation=" << b->generation;
    std::cerr << " bytes=" << bytes << '\n';
}
#ifndef FORT_SCOPE_CPU_TEST
void cuda_check(Context &c, cudaError_t status, bool allocation = false) {
    if (status == cudaSuccess) return;
    // No numerical work can have run in a context without an initialized
    // stream. An unavailable device at this point permits native continuation.
    // The same errors after initialization are execution failures: replaying
    // earlier operations would be unsafe.
    if (!c.ready && (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
                     status == cudaErrorInvalidDevice || status == cudaErrorInitializationError)) {
        throw Error(FORT_SCOPE_RESOURCE, cudaGetErrorString(status));
    }
    if (allocation && status == cudaErrorMemoryAllocation)
    {
        const auto pending = cudaGetLastError();
        if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) {
            c.poisoned = true;
            throw Error(FORT_SCOPE_EXECUTION, cudaGetErrorString(pending));
        }
        throw Error(FORT_SCOPE_RESOURCE, cudaGetErrorString(status));
    }
    c.poisoned = true;
    throw Error(FORT_SCOPE_EXECUTION, cudaGetErrorString(status));
}
class DeviceGuard {
    Context &context;
    int previous;
public:
    explicit DeviceGuard(Context &c) : context(c), previous(0) {
        cuda_check(c, cudaGetDevice(&previous));
        if (previous != c.device) cuda_check(c, cudaSetDevice(c.device));
    }
    ~DeviceGuard() {
        if (previous != context.device && cudaSetDevice(previous) != cudaSuccess) context.poisoned = true;
    }
};
#endif
void initialize(Context &c) {
    if (c.ready) return;
#ifndef FORT_SCOPE_CPU_TEST
    DeviceGuard guard(c);
    cuda_check(c, cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking), true);
    c.ready = true; // A partially initialized context still owns this stream.
#if CUDART_VERSION >= 11020
    int supported = 0;
    cuda_check(c, cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, c.device));
    if (supported) {
        cudaMemPoolProps properties{};
        properties.allocType = cudaMemAllocationTypePinned;
        properties.location.type = cudaMemLocationTypeDevice;
        properties.location.id = c.device;
        cuda_check(c, cudaMemPoolCreate(&c.pool, &properties), true);
    }
#endif
#endif
    c.ready = true;
    { std::lock_guard<std::mutex> lock(driver_mutex); initialized_devices.insert(c.device); }
    trace("initialize");
}
void allocate(Context &c, Buffer &b) {
    if (b.device || !b.bytes) return;
    require(c.stats.allocated_bytes <= c.device_budget &&
            b.bytes <= c.device_budget-c.stats.allocated_bytes,
            FORT_SCOPE_RESOURCE, "full-layout device allocation exceeds the declared scope budget");
#ifdef FORT_SCOPE_TEST_FAULTS
    const char *fail_after = std::getenv("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER");
    require(!fail_after || c.stats.allocations < std::strtoull(fail_after, nullptr, 10),
            FORT_SCOPE_RESOURCE, "injected pre-operation allocation failure");
#endif
    initialize(c);
#ifdef FORT_SCOPE_CPU_TEST
    require(!std::getenv("FORT_SCOPE_TEST_FAIL_ALLOC"), FORT_SCOPE_RESOURCE, "injected device allocation failure");
    b.device = std::malloc(b.bytes);
    require(b.device, FORT_SCOPE_RESOURCE, "device reference allocation failed");
#else
    DeviceGuard guard(c);
#if CUDART_VERSION >= 11020
    if (c.pool) cuda_check(c, cudaMallocFromPoolAsync(&b.device, b.bytes, c.pool, c.stream), true);
    else
#endif
        cuda_check(c, cudaMalloc(&b.device, b.bytes), true);
    c.pending = true;
#endif
    ++c.stats.allocations;
    c.stats.allocated_bytes += b.bytes;
    c.stats.peak_device_bytes = std::max(c.stats.allocated_bytes, c.stats.peak_device_bytes);
    trace("allocate", &b, b.bytes);
}
void wait(Context &c) {
    if (!c.pending) return;
#ifdef FORT_SCOPE_CPU_TEST
    if (std::getenv("FORT_SCOPE_TEST_FAIL_WAIT")) {
        c.poisoned = true;
        throw Error(FORT_SCOPE_EXECUTION, "injected execution completion failure");
    }
#else
    DeviceGuard guard(c);
    cuda_check(c, cudaStreamSynchronize(c.stream));
#endif
    c.pending = false;
    ++c.stats.waits;
    trace("wait");
}
const char *transfer_reason(uint32_t reason) noexcept {
    switch (reason) {
        case FORT_SCOPE_TRANSFER_ESTIMATES_UNAVAILABLE: return "transfer_estimates_unavailable";
        case FORT_SCOPE_TRANSFER_PIPELINED_UNAVAILABLE: return "pipelined_not_available";
        case FORT_SCOPE_TRANSFER_BUDGET: return "pinned_budget_exhausted";
        case FORT_SCOPE_TRANSFER_ALLOCATION: return "pinned_allocation_failed";
        default: return "none";
    }
}
void transfer_fallback(Context &c, uint32_t reason) noexcept {
    c.transfer_stats.effective_mode = FORT_SCOPE_TRANSFERS_DIRECT;
    c.transfer_stats.fallback_reason = reason;
    ++c.transfer_stats.fallbacks;
}
fort_scope_transfer_stats transfer_statistics(const Context &c) {
    auto result = c.transfer_stats;
#ifndef FORT_SCOPE_CPU_TEST
    const auto usage = fort_staging::usage();
    result.process_reserved_bytes = usage.reserved;
    result.process_peak_bytes = usage.peak;
#endif
    return result;
}
void transfer_statistics_trace(const Context &c, fort_scope_t handle) noexcept {
    const auto *enabled = std::getenv("FORT_RUNTIME_TRACE");
    if (c.transfer_stats.requested_mode == FORT_SCOPE_TRANSFERS_DIRECT || !enabled || std::strcmp(enabled, "1")) return;
    try {
        const auto stats = transfer_statistics(c);
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << std::setprecision(17) << "FORT_SCOPED evidence {\"schema_version\":1,\"context\":" << handle
                  << ",\"event\":\"transfer_statistics\",\"stats_version\":" << stats.version
                  << ",\"complete\":true,\"requested_mode\":" << stats.requested_mode
                  << ",\"effective_mode\":" << stats.effective_mode
                  << ",\"fallback_reason\":" << stats.fallback_reason
                  << ",\"reason\":\"" << transfer_reason(stats.fallback_reason) << "\"";
        for (const auto &[key, value] : {
            std::pair{"fallbacks", stats.fallbacks}, {"pinned_uploads", stats.pinned_uploads},
            {"pinned_downloads", stats.pinned_downloads}, {"pinned_upload_bytes", stats.pinned_upload_bytes},
            {"pinned_download_bytes", stats.pinned_download_bytes}, {"packed_bytes", stats.packed_bytes},
            {"unpacked_bytes", stats.unpacked_bytes}, {"tiles", stats.tiles}, {"events", stats.events},
            {"event_waits", stats.event_waits}, {"staging_allocations", stats.staging_allocations},
            {"staging_reuses", stats.staging_reuses}, {"slot_capacity", stats.slot_capacity},
            {"staging_device_bytes", stats.staging_device_bytes},
            {"process_reserved_bytes", stats.process_reserved_bytes}, {"process_peak_bytes", stats.process_peak_bytes}})
            std::cerr << ",\"" << key << "\":" << value;
        std::cerr << ",\"packing_seconds\":" << stats.packing_seconds
                  << ",\"unpacking_seconds\":" << stats.unpacking_seconds
                  << ",\"event_wait_seconds\":" << stats.event_wait_seconds << "}\n";
    } catch (...) {} // Reporting cannot change numerical success.
}
size_t staging_capacity(size_t bytes) noexcept {
    for (const size_t candidate : {256ULL*1024, 1024ULL*1024, 4ULL*1024*1024, 16ULL*1024*1024})
        if (bytes <= candidate) return candidate;
    return 16ULL*1024*1024;
}
void pack_tile(unsigned char *packed, char *host, const fort_physical::CopyOperation &op, bool unpack) {
    const size_t slice = op.pitch*op.physical_height;
    for (size_t z=0; z<op.depth; ++z)
        for (size_t y=0; y<op.height; ++y) {
            auto *original = host+op.offset+z*slice+y*op.pitch;
            auto *compact = packed+(z*op.height+y)*op.width;
            if (unpack) std::memcpy(original, compact, op.width);
            else std::memcpy(compact, original, op.width);
        }
}
bool copy_pinned(Context &c, Buffer &b, const fort_physical::CopyPlan &plan, bool upload) {
    // Full-layout allocations and all prior writers belong to c.stream.
    // The synchronous control establishes that ordering before slot streams.
    wait(c);
    const auto capacity = staging_capacity(plan.bytes);
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
    if (std::getenv("FORT_SCOPE_TEST_FAIL_STAGING_ALLOC")) {
        transfer_fallback(c, FORT_SCOPE_TRANSFER_ALLOCATION); return false;
    }
#endif
#ifdef FORT_SCOPE_CPU_TEST
    std::array<std::vector<unsigned char>, 2> storage;
    try { for (auto &slot : storage) slot.resize(capacity); }
    catch (const std::bad_alloc &) { transfer_fallback(c, FORT_SCOPE_TRANSFER_ALLOCATION); return false; }
    auto *packed = storage[0].data();
    ++c.transfer_stats.staging_allocations; ++c.transfer_stats.staging_allocations;
    c.transfer_stats.slot_capacity = capacity;
#else
    DeviceGuard guard(c);
    auto lease = fort_staging::acquire(capacity, fort_staging::Role::Staging, false);
    if (lease.exhausted) { transfer_fallback(c, FORT_SCOPE_TRANSFER_BUDGET); return false; }
    if (lease.status != cudaSuccess) {
        try { cuda_check(c, lease.status, true); }
        catch (const Error &error) {
            if (error.status != FORT_SCOPE_RESOURCE) throw;
            transfer_fallback(c, FORT_SCOPE_TRANSFER_ALLOCATION); return false;
        }
    }
    require(lease.slots != nullptr, FORT_SCOPE_STATE, "missing scoped staging lease");
    auto &slot = lease.slots->slots[0];
    auto *packed = slot.host;
    c.transfer_stats.staging_allocations += lease.reused ? 0 : 2;
    c.transfer_stats.staging_reuses += lease.reused ? 1 : 0;
    c.transfer_stats.slot_capacity = lease.slots->capacity;
#endif
    auto *host = static_cast<char *>(b.host), *device = static_cast<char *>(b.device);
    auto &stats = c.transfer_stats;
    stats.effective_mode = FORT_SCOPE_TRANSFERS_PINNED;
    plan.visit([&](const fort_physical::CopyOperation &operation) {
        return fort_physical::visit_tiles(operation, stats.slot_capacity, [&](const fort_physical::CopyOperation &op) {
            const size_t bytes = op.width*op.height*op.depth;
            if (upload) {
                const auto started = std::chrono::steady_clock::now();
                pack_tile(packed, host, op, false);
                stats.packing_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                stats.packed_bytes += bytes;
            }
#ifdef FORT_SCOPE_CPU_TEST
            pack_tile(packed, device, op, upload);
#else
            const auto direction = upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
            auto *to = upload ? device+op.offset : reinterpret_cast<char *>(packed);
            const auto *from = upload ? reinterpret_cast<const char *>(packed) : device+op.offset;
            const auto to_pitch = upload ? op.pitch : op.width;
            const auto from_pitch = upload ? op.width : op.pitch;
            if (op.depth == 1) {
                if (op.height == 1)
                    cuda_check(c, cudaMemcpyAsync(to, from, op.width, direction, slot.stream));
                else cuda_check(c, cudaMemcpy2DAsync(to, to_pitch, from, from_pitch, op.width, op.height, direction, slot.stream));
            } else {
                cudaMemcpy3DParms parameters{};
                parameters.srcPtr = make_cudaPitchedPtr(const_cast<char *>(from), from_pitch, from_pitch,
                                                       upload ? op.height : op.physical_height);
                parameters.dstPtr = make_cudaPitchedPtr(to, to_pitch, to_pitch,
                                                       upload ? op.physical_height : op.height);
                parameters.extent = make_cudaExtent(op.width, op.height, op.depth);
                parameters.kind = direction;
                cuda_check(c, cudaMemcpy3DAsync(&parameters, slot.stream));
            }
            slot.pending = true;
#endif
            ++stats.tiles;
            if (upload) { ++c.stats.uploads; c.stats.upload_bytes += bytes; ++stats.pinned_uploads; stats.pinned_upload_bytes += bytes; }
            else { ++c.stats.downloads; c.stats.download_bytes += bytes; ++stats.pinned_downloads; stats.pinned_download_bytes += bytes; }
            const auto started = std::chrono::steady_clock::now();
#ifdef FORT_SCOPE_CPU_TEST
            if (std::getenv("FORT_SCOPE_TEST_FAIL_STAGING_EVENT")) {
                c.poisoned = true;
                throw Error(FORT_SCOPE_EXECUTION, "injected staging completion failure; unsafe replay prohibited");
            }
#else
#ifdef FORT_SCOPE_TEST_FAULTS
            if (std::getenv("FORT_SCOPE_TEST_FAIL_STAGING_RECORD")) {
                c.poisoned = true;
                throw Error(FORT_SCOPE_EXECUTION, "injected staging event-record failure; unsafe replay prohibited");
            }
#endif
            cuda_check(c, cudaEventRecord(slot.complete, slot.stream));
#endif
            ++stats.events;
#ifndef FORT_SCOPE_CPU_TEST
            cuda_check(c, cudaEventSynchronize(slot.complete));
            slot.pending = false;
#endif
            ++stats.event_waits;
            stats.event_wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            if (!upload) {
                const auto started = std::chrono::steady_clock::now();
                pack_tile(packed, host, op, true);
                stats.unpacking_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                stats.unpacked_bytes += bytes;
            }
            return true;
        });
    });
#ifndef FORT_SCOPE_CPU_TEST
    size_t freed = 0;
    cuda_check(c, fort_staging::release(std::move(lease.slots), freed));
#endif
    trace(upload ? "upload" : "download", &b, plan.bytes);
    return true;
}
void copy(Context &c, Buffer &b, const Box &box, bool upload) {
    const fort_physical::CopyPlan plan(b.element_bytes, b.extents, box.lo, box.hi);
    require(plan.valid, FORT_SCOPE_ARGUMENT, "invalid physical transfer layout");
    if (!plan.bytes) return;
    allocate(c, b);
    if (c.transfer_stats.requested_mode == FORT_SCOPE_TRANSFERS_PINNED && copy_pinned(c, b, plan, upload)) return;
    auto *host = static_cast<char *>(b.host), *device = static_cast<char *>(b.device);
#ifndef FORT_SCOPE_CPU_TEST
    DeviceGuard guard(c);
    const auto direction = upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
#endif
    plan.visit([&](const fort_physical::CopyOperation &op) {
        auto *to = upload ? device+op.offset : host+op.offset;
        const auto *from = upload ? host+op.offset : device+op.offset;
#ifdef FORT_SCOPE_CPU_TEST
        const size_t slice = op.pitch*op.physical_height;
        for (size_t z=0; z<op.depth; ++z)
            for (size_t y=0; y<op.height; ++y)
                std::memcpy(to+z*slice+y*op.pitch, from+z*slice+y*op.pitch, op.width);
#else
        if (op.depth == 1) {
            if (op.height == 1 || op.pitch == op.width)
                cuda_check(c, cudaMemcpyAsync(to, from, op.width*op.height, direction, c.stream));
            else cuda_check(c, cudaMemcpy2DAsync(to, op.pitch, from, op.pitch, op.width, op.height, direction, c.stream));
        } else {
            cudaMemcpy3DParms parameters{};
            parameters.srcPtr = make_cudaPitchedPtr(const_cast<char *>(from), op.pitch, op.pitch, op.physical_height);
            parameters.dstPtr = make_cudaPitchedPtr(to, op.pitch, op.pitch, op.physical_height);
            parameters.extent = make_cudaExtent(op.width, op.height, op.depth);
            parameters.kind = direction;
            cuda_check(c, cudaMemcpy3DAsync(&parameters, c.stream));
        }
#endif
        if (upload) ++c.stats.uploads; else ++c.stats.downloads;
        return true;
    });
    if (upload) c.stats.upload_bytes += plan.bytes; else c.stats.download_bytes += plan.bytes;
    c.pending = true;
    trace(upload ? "upload" : "download", &b, plan.bytes);
}
Region sections(const Buffer &b, const fort_scope_section *data, size_t count, bool full) {
    require(count <= rectangle_limit && (!count || data), FORT_SCOPE_ARGUMENT, "section list exceeds the bounded interface");
    Region result;
    if (full && !empty(b.full())) result.push_back(b.full());
    Budget budget;
    for (size_t i=0; i<count; ++i) {
        require(data[i].lower && data[i].upper, FORT_SCOPE_ARGUMENT, "missing section coordinates");
        Box box;
        for (size_t k=0; k<b.extents.size(); ++k) {
            auto lo = data[i].lower[k], hi = data[i].upper[k];
            require(lo <= hi && hi <= b.extents[k], FORT_SCOPE_ARGUMENT, "section outside physical array bounds");
            box.lo.push_back(lo); box.hi.push_back(hi);
        }
        if (!empty(box)) result = unite(std::move(result), {box}, budget);
    }
    return result;
}
Effects effects(const Buffer &b, const fort_scope_access *access) {
    require(access && !(access->flags & ~7U), FORT_SCOPE_ARGUMENT, "invalid access descriptor");
    Effects e{sections(b, access->reads, access->read_count, access->flags & FORT_SCOPE_READ_ALL),
              sections(b, access->writes, access->write_count, access->flags & FORT_SCOPE_WRITE_ALL),
              sections(b, access->overwrites, access->overwrite_count, access->flags & FORT_SCOPE_OVERWRITE_ALL)};
    Budget budget;
    require(difference(e.overwrites, e.writes, budget).empty(), FORT_SCOPE_ARGUMENT, "must-overwrite exceeds may-write sections");
    return e;
}
void reconcile(Context &c, Buffer &b, bool device) {
    // Every rectangle copied is current at the source. Copying already-current
    // destination cells is harmless; copying a stale enclosing box is not.
    const Region latest_other = device ? b.host_current : b.device_current;
    for (const auto &part : latest_other) copy(c, b, part, device);
    if (!device) wait(c);
    auto &chosen = device ? b.device_current : b.host_current;
    auto &other = device ? b.host_current : b.device_current;
    chosen = b.initialized;
    other.clear();
    ++c.stats.reconciliations;
    trace(device ? "reconcile_device" : "reconcile_host", &b);
}
void ensure(Context &c, Buffer &b, const Region &requested, bool device) {
    if (requested.empty()) return;
    Budget budget;
    auto &chosen = device ? b.device_current : b.host_current;
    try {
        require(difference(requested, b.initialized, budget).empty(), FORT_SCOPE_UNINITIALIZED,
                "access reads an uninitialized section");
        const auto missing = difference(requested, chosen, budget);
        auto updated = unite(chosen, missing, budget);
        for (const auto &part : missing) copy(c, b, part, device);
        chosen = std::move(updated);
    } catch (const Fragmented &) {
        reconcile(c, b, device);
        Budget retry;
        require(difference(requested, b.initialized, retry).empty(), FORT_SCOPE_UNINITIALIZED,
                "access reads an uninitialized section");
    }
}
void begin(Context &c, Buffer &b, const fort_scope_access *access, bool device) {
    require(!b.prepared, FORT_SCOPE_STATE, "buffer access already prepared");
    const auto e = effects(b, access);
    if (device) allocate(c, b);
    auto prepare_access = [&]() {
        Budget budget;
        const auto preserve = difference(e.writes, e.overwrites, budget);
        ensure(c, b, e.reads, device);
        ensure(c, b, preserve, device);
        return prepare(b, e, device);
    };
    try { b.prepared = prepare_access(); }
    catch (const Fragmented &) {
        reconcile(c, b, device);
        b.prepared = prepare_access();
    }
    // Copies share the kernel stream: queue needed downloads before the one
    // host boundary wait. This also completes uploads reading borrowed storage.
    if (!device) wait(c);
}
void end(Buffer &b, bool device) {
    require(b.prepared && b.prepared->device == device, FORT_SCOPE_STATE, "missing or mismatched access begin");
    b.initialized = std::move(b.prepared->initialized);
    (device ? b.device_current : b.host_current) = std::move(b.prepared->current);
    (device ? b.host_current : b.device_current) = std::move(b.prepared->opposite);
    b.prepared.reset();
    trace(device ? "device_commit" : "host_commit", &b);
}
void publish(Context &c, Buffer &b) {
    require(!b.prepared, FORT_SCOPE_STATE, "unfinished buffer access at scope boundary");
    ensure(c, b, b.initialized, false);
    wait(c);
}
void release(Context &c, Buffer &b) {
    if (!b.device) return;
#ifdef FORT_SCOPE_CPU_TEST
    std::free(b.device);
#else
    DeviceGuard guard(c);
#if CUDART_VERSION >= 11020
    if (c.pool) { cuda_check(c, cudaFreeAsync(b.device, c.stream)); c.pending = true; }
    else
#endif
        cuda_check(c, cudaFree(b.device));
#endif
    b.device = nullptr;
    c.stats.allocated_bytes -= b.bytes;
    trace("release", &b, b.bytes);
}
template<class F> int protect(F &&f) noexcept {
    fort_scoped::TeamObservation observation(FORT_SCOPE_TEAM_OBSERVE_API);
    try { f(); return FORT_SCOPE_OK; }
    catch (const Error &error) { diagnostic(error.what()); return error.status; }
    catch (const Fragmented &) { diagnostic("section fragmentation requires a scope boundary"); return FORT_SCOPE_BOUNDARY; }
    catch (const std::bad_alloc &) { diagnostic("scope metadata allocation failed"); return FORT_SCOPE_RESOURCE; }
    catch (const std::exception &error) { diagnostic(error.what()); return FORT_SCOPE_STATE; }
    catch (...) { diagnostic("unknown scope runtime failure"); return FORT_SCOPE_STATE; }
}
enum class Change { None, Query, State };
void invalidate(Context &c, Change change) noexcept {
    if (change == Change::None) return;
    c.definition_proof.reset(); c.planning_snapshot.reset(); c.preview.reset();
    auto &generation = change == Change::Query ? c.query_generation : c.state_generation;
    // Both cached stamps are discarded before wrapping, so a reused generation
    // can never match a surviving old proof or preview.
    generation = generation == std::numeric_limits<uint64_t>::max() ? 1 : generation+1;
}
void note_numerical_activity(Context &c) noexcept {
    // A complete-owner estimate must cover the entire context, including work
    // performed before its first continuation query or after a completed one.
    if (c.endpoint_mode != FORT_SCOPE_PLAN_CONTINUE || !c.segment_cost_pending)
        c.owner_estimate_available = false;
}
template<class F> int with(fort_scope_t handle, F &&f, Change change = Change::State) noexcept {
    return protect([&]() {
        const auto c = lookup(handle);
        std::lock_guard<std::mutex> lock(c->mutex);
        require(!c->closed, FORT_SCOPE_STALE, "invalid or stale scope handle");
        require(!c->poisoned, FORT_SCOPE_EXECUTION, "scope has an execution failure; unsafe replay prohibited");
        // Invalidate before a possibly partial mutation, including operations
        // that fail. Read-only metadata probes preserve a context-local proof.
        invalidate(*c, change);
        try { f(*c); }
        catch (...) {
            if (c->endpoint_mode == FORT_SCOPE_PLAN_CONTINUE && change != Change::None)
                c->owner_estimate_available = false;
            throw;
        }
    });
}
}

namespace {
bool diagnostics_enabled() noexcept {
    const char *enabled = std::getenv("FORT_RUNTIME_TRACE");
    return enabled && !std::strcmp(enabled, "1");
}
using PlanningClock = std::chrono::steady_clock;
class PlanningTimer {
    Context &context;
    fort_scope_t handle;
    const char *phase;
    bool enabled;
    PlanningClock::time_point started;
public:
    bool cache_hit = false;
    PlanningTimer(Context &c, fort_scope_t h, const char *p)
        : context(c), handle(h), phase(p), enabled(diagnostics_enabled()),
          started(enabled ? PlanningClock::now() : PlanningClock::time_point{}) {}
    double seconds() const noexcept {
        return enabled ? std::chrono::duration<double>(PlanningClock::now()-started).count() : 0;
    }
    ~PlanningTimer() noexcept {
        if (!enabled) return;
        const double elapsed = seconds();
        try {
            std::lock_guard<std::mutex> lock(trace_mutex);
            std::cerr << std::setprecision(17) << "FORT_SCOPED planning_timing context=" << handle
                      << " phase=" << phase << " seconds=" << elapsed
                      << " cache_hit=" << (cache_hit ? 1 : 0)
                      << " query_generation=" << context.query_generation
                      << " state_generation=" << context.state_generation << '\n';
        } catch (...) {}
    }
};
bool driver_initialized(const Context &c) {
    std::lock_guard<std::mutex> lock(driver_mutex);
    return initialized_devices.count(c.device) != 0;
}
void complete_segment_estimate(Context &c) noexcept {
    if (!c.segment_cost_pending || c.pending || c.worker_cursor != c.schedule.size() ||
        std::any_of(c.buffers.begin(), c.buffers.end(), [](const auto &entry) { return entry.second->prepared.has_value(); })) return;
    c.segment_cost_pending = false;
    if (!c.last_report) { c.owner_estimate_available = false; return; }
    const auto &report = *c.last_report;
    if (c.owner_segments == std::numeric_limits<uint64_t>::max()) c.owner_estimate_available = false;
    else ++c.owner_segments;
    const double execution = c.owner_execution_seconds+report.execution_seconds;
    if (!report.available || !std::isfinite(execution) || !std::isfinite(execution+report.terminal.seconds))
        c.owner_estimate_available = false;
    else {
        c.owner_execution_seconds = execution;
        c.owner_terminal_seconds = report.terminal.seconds;
    }
    c.owner_native_common_compute_excluded |= report.native_common_compute_excluded;
}
const fort_scoped::planning::Inputs &planning_inputs(Context &c, fort_scope_t handle) {
    if (c.planning_snapshot && c.planning_snapshot->query_generation == c.query_generation &&
        c.planning_snapshot->state_generation == c.state_generation) {
        auto &input = c.planning_snapshot->input;
        input.driver_initialized = driver_initialized(c);
        input.definitions_validated = c.definition_proof &&
            c.definition_proof->query_generation == c.query_generation &&
            c.definition_proof->state_generation == c.state_generation;
        return input;
    }
    PlanningTimer timing(c, handle, "query_construction");
    require(c.plan.size() <= fort_scoped::planning::detail::operation_limit &&
            c.buffers.size() <= fort_scoped::planning::detail::operation_limit,
            FORT_SCOPE_BOUNDARY, "planning_record_budget_exceeded");
    fort_scoped::planning::Inputs input;
    input.operations = c.plan;
    input.team_costs = c.team_costs;
    input.continuation = c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE;
    if (input.continuation) {
        input.charge_create = !c.owner_create_accounted;
        input.registrations_incurred = c.registrations_incurred;
    }
    input.definitions_validated = c.definition_proof &&
        c.definition_proof->query_generation == c.query_generation &&
        c.definition_proof->state_generation == c.state_generation;
    input.query_construction_operations = 1 + c.plan.size();
    input.device_budget = c.device_budget;
    input.device_ready = c.ready;
    input.pending = c.pending;
    input.transfer_mode = c.transfer_stats.requested_mode == FORT_SCOPE_TRANSFERS_PINNED
        ? FORT_SCOPE_TRANSFERS_PINNED : FORT_SCOPE_TRANSFERS_DIRECT;
    input.transfer_costs = c.transfer_costs;
#ifdef FORT_SCOPE_CPU_TEST
    input.asynchronous_release = false;
#else
#if CUDART_VERSION >= 11020
    // New scopes use the stream-ordered pool on supported devices. Charging
    // its extra completion wait is conservative if the runtime uses cudaFree.
    input.asynchronous_release = !c.ready || c.pool != nullptr;
#else
    input.asynchronous_release = false;
#endif
#endif
    input.driver_initialized = driver_initialized(c);
    for (const auto &entry : c.buffers) {
        const auto &b = *entry.second;
        require(!b.prepared, FORT_SCOPE_STATE, "planning during a prepared buffer access");
        input.resources.push_back({b.handle, b.element_bytes, b.bytes, b.extents,
                                   b.initialized, b.host_current, b.device_current, b.device != nullptr});
    }
    c.planning_snapshot = Context::QuerySnapshot{c.query_generation, c.state_generation, std::move(input)};
    return c.planning_snapshot->input;
}
std::vector<fort_scoped::planning::Binding> planning_bindings(
        Context &c, const fort_scope_plan_binding *bindings, size_t count) {
    require(count <= 256 && (!count || bindings), FORT_SCOPE_ARGUMENT, "invalid planning binding list");
    std::vector<fort_scoped::planning::Binding> result;
    for (size_t i=0; i<count; ++i) {
        require(std::none_of(result.begin(), result.end(), [&](const auto &old) {
            return old.buffer == bindings[i].buffer;
        }), FORT_SCOPE_ALIAS, "planning bindings must use canonical unique buffers");
        auto &b = buffer(c, bindings[i].buffer);
        require(!b.prepared, FORT_SCOPE_STATE, "planning during a prepared buffer access");
        result.push_back({b.handle, effects(b, &bindings[i].access)});
    }
    return result;
}
bool same_region(const Region &a, const Region &b) {
    if (a.size() != b.size()) return false;
    for (size_t k=0; k<a.size(); ++k) if (a[k].lo != b[k].lo || a[k].hi != b[k].hi) return false;
    return true;
}
bool same_bindings(const std::vector<fort_scoped::planning::Binding> &a,
                   const std::vector<fort_scoped::planning::Binding> &b) {
    if (a.size() != b.size()) return false;
    for (size_t k=0; k<a.size(); ++k) {
        if (a[k].buffer != b[k].buffer || !same_region(a[k].effects.reads, b[k].effects.reads) ||
            !same_region(a[k].effects.writes, b[k].effects.writes) ||
            !same_region(a[k].effects.overwrites, b[k].effects.overwrites)) return false;
    }
    return true;
}
struct BatchDecline { uint32_t reason; };
void batch_require(bool condition, uint32_t reason = FORT_SCOPE_BATCH_GEOMETRY) {
    if (!condition) throw BatchDecline{reason};
}
uint64_t batch_add(uint64_t a, uint64_t b) {
    batch_require(b <= std::numeric_limits<uint64_t>::max()-a, FORT_SCOPE_BATCH_ARITHMETIC); return a+b;
}
struct BatchBinding { size_t resource; uint32_t axis; int64_t step; Effects base; };
struct BatchUnit { uint32_t kind; uint64_t unit; std::vector<BatchBinding> bindings; double flops, memory; };
struct BatchResource {
    Buffer *original;
    fort_scoped::planning::Resource final;
    Region incoming, written, exported, prefix;
    uint32_t axis = FORT_SCOPE_BATCH_FIXED_AXIS;
    int64_t step = 0;
    bool mutable_value = false, used = false;
};
struct BatchPlan {
    size_t iterations = 0, workers = 0;
    std::vector<BatchUnit> units;
    std::vector<BatchResource> resources;
    uint64_t operations = 1;
};
Region batch_intersection(const Region &a, const Region &b, uint64_t &operations) {
    Budget budget; Region result;
    for (const auto &left : a) for (const auto &right : b) {
        operations = batch_add(operations, 1);
        if (auto box = intersection(left, right, budget)) result = unite(std::move(result), {*box}, budget);
    }
    return result;
}
Region batch_shift(const Region &base, const BatchBinding &binding, const Buffer &b,
                   size_t begin, size_t count, uint64_t &operations) {
    if (!count || base.empty()) return {};
    if (binding.axis == FORT_SCOPE_BATCH_FIXED_AXIS) return base;
    batch_require(begin <= std::numeric_limits<size_t>::max()-(count-1), FORT_SCOPE_BATCH_ARITHMETIC);
    const uint64_t magnitude = binding.step < 0 ? uint64_t(-(binding.step+1))+1 : uint64_t(binding.step);
    const size_t last = begin+count-1;
    batch_require(!magnitude || last <= std::numeric_limits<size_t>::max()/magnitude, FORT_SCOPE_BATCH_ARITHMETIC);
    const auto first_shift = size_t(magnitude)*begin, last_shift = size_t(magnitude)*last;
    Region result;
    for (auto box : base) {
        operations = batch_add(operations, 1);
        // A sparse ordinal sequence is not its enclosing rectangle. Keep the
        // ordinary worker path until an exact bounded union is supported.
        batch_require(count==1 || box.hi[binding.axis]-box.lo[binding.axis]>=magnitude,
                      FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
        if (binding.step < 0) {
            batch_require(box.lo[binding.axis] >= last_shift && box.hi[binding.axis] >= first_shift);
            box.lo[binding.axis] -= last_shift; box.hi[binding.axis] -= first_shift;
        } else {
            batch_require(last_shift <= b.extents[binding.axis] && box.hi[binding.axis] <= b.extents[binding.axis]-last_shift);
            box.lo[binding.axis] += first_shift; box.hi[binding.axis] += last_shift;
        }
        append(result, std::move(box));
    }
    return result;
}
Effects batch_effects(const BatchBinding &binding, const BatchPlan &plan, size_t begin, size_t count,
                      uint64_t &operations) {
    const auto &b = *plan.resources[binding.resource].original;
    return {batch_shift(binding.base.reads,binding,b,begin,count,operations),
            batch_shift(binding.base.writes,binding,b,begin,count,operations),
            batch_shift(binding.base.overwrites,binding,b,begin,count,operations)};
}
int64_t batch_floor(int64_t value, int64_t divisor) {
    return value/divisor-(value<0 && value%divisor != 0);
}
int64_t batch_ceil(int64_t value, int64_t divisor) {
    return value/divisor+(value>0 && value%divisor != 0);
}
bool batch_crosses(const Box &a, const Box &b, const BatchResource &resource, size_t iterations,
                   uint64_t &operations) {
    operations = batch_add(operations, 1);
    for (size_t k=0; k<a.lo.size(); ++k) if (k != resource.axis)
        if (a.hi[k] <= b.lo[k] || b.hi[k] <= a.lo[k]) return false;
    const auto axis = resource.axis;
    const auto extent = resource.original->extents[axis];
    const auto alo = resource.step > 0 ? a.lo[axis] : extent-a.hi[axis];
    const auto ahi = resource.step > 0 ? a.hi[axis] : extent-a.lo[axis];
    const auto blo = resource.step > 0 ? b.lo[axis] : extent-b.hi[axis];
    const auto bhi = resource.step > 0 ? b.hi[axis] : extent-b.lo[axis];
    const auto step = resource.step > 0 ? resource.step : -resource.step;
    // Strict rectangle intersection for any nonzero ordinal difference, not
    // just neighboring batches. Coordinate bounds make subtraction checked.
    const auto lower = batch_floor(int64_t(alo)-int64_t(bhi),step)+1;
    const auto upper = batch_ceil(int64_t(ahi)-int64_t(blo),step)-1;
    return std::max<int64_t>(1,lower) <= std::min<int64_t>(int64_t(iterations-1),upper);
}
BatchPlan prepare_batch(Context &c, const fort_scope_batch &descriptor) {
    require(descriptor.version == FORT_SCOPE_BATCH_ABI_VERSION &&
            (descriptor.execution_mode == FORT_SCOPE_GPU || descriptor.execution_mode == FORT_SCOPE_AUTO),
            FORT_SCOPE_ARGUMENT, "invalid scoped batch version or execution mode");
    batch_require(descriptor.iterations && descriptor.iterations <= size_t(std::numeric_limits<int64_t>::max()));
    batch_require(descriptor.unit_count && descriptor.unit_count <= fort_scoped::planning::detail::operation_limit && descriptor.units,
                  FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
    BatchPlan plan; plan.iterations = descriptor.iterations;
    std::unordered_map<fort_buffer_t,size_t> indices;
    auto resource = [&](fort_buffer_t handle) {
        auto found = indices.find(handle);
        if (found != indices.end()) return found->second;
        auto &b = buffer(c,handle);
        batch_require(!b.prepared, FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
        for (size_t extent : b.extents) batch_require(extent <= size_t(std::numeric_limits<int64_t>::max()));
        const auto index = plan.resources.size(); indices.emplace(handle,index);
        plan.resources.push_back({&b,{b.handle,b.element_bytes,b.bytes,b.extents,b.initialized,b.host_current,b.device_current,b.device != nullptr},
                                  {},{},{},{}});
        plan.operations = batch_add(plan.operations,1+b.extents.size()+b.initialized.size()+b.host_current.size()+b.device_current.size());
        return index;
    };
    for (size_t i=0; i<descriptor.unit_count; ++i) {
        const auto &source = descriptor.units[i];
        batch_require(source.kind == FORT_SCOPE_PLAN_WORKER || source.kind == FORT_SCOPE_PLAN_FORGET,
                      FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
        batch_require(source.count <= fort_scoped::planning::detail::operation_limit && (!source.count || source.bindings));
        batch_require(std::isfinite(source.flops) && source.flops >= 0 && std::isfinite(source.memory_bytes) && source.memory_bytes >= 0);
        if (source.kind == FORT_SCOPE_PLAN_WORKER) {
            batch_require(source.unit && (source.flops>0 || source.memory_bytes>0));
            batch_require(++plan.workers <= fort_scoped::planning::detail::worker_limit, FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
        }
        BatchUnit unit{source.kind,source.unit,{},source.flops,source.memory_bytes};
        for (size_t j=0; j<source.count; ++j) {
            const auto &item = source.bindings[j]; const auto index = resource(item.buffer);
            batch_require(std::none_of(unit.bindings.begin(),unit.bindings.end(),[&](const auto &old){return old.resource==index;}));
            auto &r = plan.resources[index];
            batch_require(item.axis == FORT_SCOPE_BATCH_FIXED_AXIS || item.axis < r.original->extents.size());
            batch_require(item.axis != FORT_SCOPE_BATCH_FIXED_AXIS || item.step == 0);
            batch_require(item.step != std::numeric_limits<int64_t>::min());
            auto access = effects(*r.original,&item.access);
            if (source.kind == FORT_SCOPE_PLAN_WORKER && (!access.reads.empty() || !access.writes.empty())) r.used=true;
            if (source.kind == FORT_SCOPE_PLAN_WORKER && !access.writes.empty()) {
                Budget budget;
                batch_require(difference(access.writes,access.overwrites,budget).empty(), FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
                batch_require(item.axis != FORT_SCOPE_BATCH_FIXED_AXIS && item.step != 0);
                const auto width = size_t(item.step<0 ? -item.step : item.step);
                for (const auto &box : access.writes) batch_require(box.hi[item.axis]-box.lo[item.axis]==width);
                r.mutable_value = true;
            }
            unit.bindings.push_back({index,item.axis,item.step,std::move(access)});
            plan.operations = batch_add(plan.operations,1+item.access.read_count+item.access.write_count+item.access.overwrite_count);
        }
        plan.units.push_back(std::move(unit));
    }
    batch_require(plan.workers>0, FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
    for (auto &r : plan.resources) if (r.mutable_value) {
        bool found = false;
        for (const auto &unit : plan.units) if (unit.kind == FORT_SCOPE_PLAN_WORKER)
            for (const auto &binding : unit.bindings) if (plan.resources[binding.resource].original == r.original) {
                if (binding.base.reads.empty() && binding.base.writes.empty() && binding.base.overwrites.empty()) continue;
                if (!found) { r.axis=binding.axis; r.step=binding.step; found=true; }
                batch_require(r.axis != FORT_SCOPE_BATCH_FIXED_AXIS && r.step != 0 && binding.axis==r.axis && binding.step==r.step,
                              FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
            }
        if (plan.iterations>1) {
            Region writes, accesses; Budget budget;
            for (const auto &unit : plan.units) if (unit.kind == FORT_SCOPE_PLAN_WORKER)
                for (const auto &binding : unit.bindings) if (plan.resources[binding.resource].original == r.original) {
                    writes=unite(std::move(writes),binding.base.writes,budget);
                    accesses=unite(std::move(accesses),binding.base.reads,budget);
                    accesses=unite(std::move(accesses),binding.base.writes,budget);
                }
            for (const auto &write : writes) for (const auto &access : accesses)
                batch_require(!batch_crosses(write,access,r,plan.iterations,plan.operations) &&
                              !batch_crosses(access,write,r,plan.iterations,plan.operations), FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
        }
    }
    // Compose whole-entry definitions in source order. Earlier GPU outputs
    // satisfy later reads; they never become uploads from stale host storage.
    for (const auto &unit : plan.units) {
        for (const auto &binding : unit.bindings) {
            auto &r = plan.resources[binding.resource]; auto &state = r.final;
            plan.operations = batch_add(plan.operations,1);
            if (unit.kind == FORT_SCOPE_PLAN_FORGET) {
                state.initialized.clear(); state.host_current.clear(); state.device_current.clear(); continue;
            }
            auto access=batch_effects(binding,plan,0,plan.iterations,plan.operations); Budget budget;
            auto needs=unite(access.reads,difference(access.writes,access.overwrites,budget),budget);
            require(difference(needs,state.initialized,budget).empty(),FORT_SCOPE_UNINITIALIZED,"batch reads an undefined section");
            auto missing=difference(needs,state.device_current,budget);
            require(difference(missing,state.host_current,budget).empty(),FORT_SCOPE_UNINITIALIZED,"batch input has no current source");
            r.incoming=unite(std::move(r.incoming),missing,budget);
            state.device_current=unite(std::move(state.device_current),missing,budget);
            auto prepared=prepare(state,access,true);
            state.initialized=std::move(prepared.initialized); state.device_current=std::move(prepared.current);
            state.host_current=std::move(prepared.opposite);
            r.written=unite(std::move(r.written),access.writes,budget);
        }
    }
    batch_require(descriptor.export_count <= plan.resources.size() && (!descriptor.export_count || descriptor.exports));
    for (size_t i=0; i<descriptor.export_count; ++i) {
        const auto &item=descriptor.exports[i]; auto found=indices.find(item.buffer);
        batch_require(found != indices.end()); auto &r=plan.resources[found->second];
        auto access=effects(*r.original,&item.access); Budget budget;
        batch_require(access.writes.empty() && access.overwrites.empty() && difference(access.reads,r.written,budget).empty());
        require(difference(access.reads,r.final.initialized,budget).empty(),FORT_SCOPE_UNINITIALIZED,"batch exports undefined output");
        r.exported=unite(std::move(r.exported),access.reads,budget);
        r.final.host_current=unite(std::move(r.final.host_current),access.reads,budget);
        plan.operations=batch_add(plan.operations,1+access.reads.size());
    }
    // Immutable repeated/halo reads are made current once before any batch.
    for (size_t index=0; index<plan.resources.size(); ++index) {
        auto &r=plan.resources[index]; if (r.mutable_value || r.incoming.empty()) continue;
        bool overlapping=false;
        for (const auto &unit : plan.units) if (unit.kind==FORT_SCOPE_PLAN_WORKER)
            for (const auto &binding : unit.bindings) if (binding.resource==index) {
                if (binding.axis==FORT_SCOPE_BATCH_FIXED_AXIS || binding.step==0) overlapping=true;
                else for (const auto &box : binding.base.reads)
                    if (box.hi[binding.axis]-box.lo[binding.axis] != size_t(binding.step<0 ? -binding.step : binding.step)) overlapping=true;
            }
        if (!overlapping && plan.iterations>1) {
            Region reads; Budget budget; bool first=true;
            for (const auto &unit : plan.units) if (unit.kind==FORT_SCOPE_PLAN_WORKER)
                for (const auto &binding : unit.bindings) if (binding.resource==index) {
                    if (first) { r.axis=binding.axis; r.step=binding.step; first=false; }
                    if (binding.axis!=r.axis || binding.step!=r.step) overlapping=true;
                    reads=unite(std::move(reads),binding.base.reads,budget);
                }
            if (!overlapping) for (const auto &left : reads) for (const auto &right : reads)
                if (batch_crosses(left,right,r,plan.iterations,plan.operations)) overlapping=true;
        }
        if (overlapping) r.prefix=r.incoming;
    }
    return plan;
}
bool equivalent_region(const Region &a, const Region &b) {
    Budget budget; return difference(a,b,budget).empty() && difference(b,a,budget).empty();
}
size_t batch_schedule(const Context &c, const BatchPlan &plan, uint32_t mode, uint64_t &operations) {
    if (!c.plan_installed) {
        batch_require(mode==FORT_SCOPE_GPU,FORT_SCOPE_BATCH_PLACEMENT); return 0;
    }
    size_t worker=0, position=c.plan.size();
    for (size_t i=0; i<c.plan.size(); ++i) if (c.plan[i].kind==FORT_SCOPE_PLAN_WORKER)
        if (worker++==c.worker_cursor) { position=i; break; }
    size_t prefix=0; while (prefix<plan.units.size() && plan.units[prefix].kind==FORT_SCOPE_PLAN_FORGET) ++prefix;
    batch_require(position<c.plan.size() && position>=prefix && plan.units.size()<=c.plan.size()-(position-prefix),FORT_SCOPE_BATCH_PLACEMENT);
    position-=prefix; worker=c.worker_cursor;
    for (size_t i=0; i<plan.units.size(); ++i) {
        const auto &expected=c.plan[position+i]; const auto &unit=plan.units[i];
        operations=batch_add(operations,1);
        batch_require(expected.kind==unit.kind,FORT_SCOPE_BATCH_PLACEMENT);
        if (unit.kind==FORT_SCOPE_PLAN_WORKER) {
            batch_require(expected.unit==unit.unit && worker<c.schedule.size() && c.schedule[worker],FORT_SCOPE_BATCH_PLACEMENT); ++worker;
        }
        for (const auto &binding : unit.bindings) {
            const auto handle=plan.resources[binding.resource].original->handle;
            auto found=std::find_if(expected.bindings.begin(),expected.bindings.end(),[&](const auto &item){return item.buffer==handle;});
            if (unit.kind==FORT_SCOPE_PLAN_FORGET) { batch_require(found!=expected.bindings.end(),FORT_SCOPE_BATCH_PLACEMENT); continue; }
            auto full=batch_effects(binding,plan,0,plan.iterations,operations);
            if (found==expected.bindings.end()) {
                batch_require(full.reads.empty() && full.writes.empty() && full.overwrites.empty(),FORT_SCOPE_BATCH_PLACEMENT); continue;
            }
            batch_require(equivalent_region(found->effects.reads,full.reads) && equivalent_region(found->effects.writes,full.writes) &&
                          equivalent_region(found->effects.overwrites,full.overwrites),FORT_SCOPE_BATCH_PLACEMENT);
        }
        for (const auto &binding : expected.bindings)
            batch_require(std::any_of(unit.bindings.begin(),unit.bindings.end(),[&](const auto &item){
                return plan.resources[item.resource].original->handle==binding.buffer;
            }),FORT_SCOPE_BATCH_PLACEMENT);
    }
    return plan.workers;
}
struct BatchCopy { Buffer *buffer; fort_physical::CopyOperation operation; size_t packed_offset; };
struct BatchCopies {
    std::vector<BatchCopy> uploads, downloads;
    uint64_t upload_bytes=0, download_bytes=0, upload_rows=0, download_rows=0, operations=1;
};
void batch_copies_append(BatchCopies &copies, Buffer &b, const Region &region, bool upload) {
    auto &list=upload ? copies.uploads : copies.downloads;
    auto &bytes=upload ? copies.upload_bytes : copies.download_bytes;
    auto &rows=upload ? copies.upload_rows : copies.download_rows;
    for (const auto &box : region) {
        fort_physical::CopyPlan physical(b.element_bytes,b.extents,box.lo,box.hi);
        batch_require(physical.valid,FORT_SCOPE_BATCH_ARITHMETIC);
        physical.visit([&](const auto &op) {
            batch_require(list.size()<intersection_limit,FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
            const auto size=multiply(multiply(op.width,op.height),op.depth);
            list.push_back({&b,op,size_t(bytes)}); bytes=batch_add(bytes,size);
            rows=batch_add(rows,multiply(op.height,op.depth)); copies.operations=batch_add(copies.operations,1); return true;
        });
    }
}
BatchCopies batch_copies(const BatchPlan &plan, size_t begin, size_t count, bool capacity_bound=false) {
    BatchCopies copies;
    for (size_t index=0; index<plan.resources.size(); ++index) {
        const auto &r=plan.resources[index]; Region needs,writes; Budget budget;
        for (const auto &unit : plan.units) if (unit.kind==FORT_SCOPE_PLAN_WORKER)
            for (const auto &binding : unit.bindings) if (binding.resource==index) {
                auto access=batch_effects(binding,plan,begin,count,copies.operations);
                needs=unite(std::move(needs),access.reads,budget);
                needs=unite(std::move(needs),difference(access.writes,access.overwrites,budget),budget);
                writes=unite(std::move(writes),access.writes,budget);
            }
        if (!r.incoming.empty() && r.prefix.empty()) {
            auto incoming=capacity_bound ? needs : batch_intersection(needs,r.incoming,copies.operations);
            batch_copies_append(copies,*r.original,incoming,true);
        }
        if (!r.exported.empty()) {
            auto outgoing=capacity_bound ? writes : batch_intersection(writes,r.exported,copies.operations);
            batch_copies_append(copies,*r.original,outgoing,false);
        }
    }
    return copies;
}
double batch_copy_cost(uint64_t bytes, uint64_t rows, uint64_t calls, const fort_scope_batch_costs &costs, bool upload) {
    const double value=double(bytes)/(upload ? costs.pinned_h2d_bandwidth : costs.pinned_d2h_bandwidth) +
        double(calls)*(upload ? costs.pinned_h2d_latency : costs.pinned_d2h_latency) +
        double(bytes)/(upload ? costs.pack_bytes_per_second : costs.unpack_bytes_per_second) +
        double(rows)*(upload ? costs.pack_row_seconds : costs.unpack_row_seconds);
    batch_require(std::isfinite(value) && value>=0,FORT_SCOPE_BATCH_ARITHMETIC); return value;
}
fort_scoped::planning::Inputs batch_inputs(const Context &c, const BatchPlan &plan, uint64_t &operations) {
    fort_scoped::planning::Inputs input;
    input.device_budget=c.device_budget; input.device_ready=c.ready; input.driver_initialized=driver_initialized(c);
    input.pending=c.pending; input.continuation=true; input.charge_create=false; input.registrations_incurred=0;
#ifdef FORT_SCOPE_CPU_TEST
    input.asynchronous_release=false;
#endif
    for (const auto &r : plan.resources) {
        const auto &b=*r.original;
        input.resources.push_back({b.handle,b.element_bytes,b.bytes,b.extents,b.initialized,b.host_current,b.device_current,b.device!=nullptr});
    }
    for (const auto &unit : plan.units) {
        fort_scoped::planning::Operation op{unit.kind,unit.unit,{},unit.flops,unit.memory,true};
        for (const auto &binding : unit.bindings) {
            auto access=batch_effects(binding,plan,0,plan.iterations,operations);
            if (unit.kind==FORT_SCOPE_PLAN_WORKER && access.reads.empty() && access.writes.empty() && access.overwrites.empty()) continue;
            op.bindings.push_back({plan.resources[binding.resource].original->handle,std::move(access)});
        }
        input.operations.push_back(std::move(op));
    }
    return input;
}
fort_scope_batch_report price_batch(const Context &c, const BatchPlan &plan, const fort_scope_plan_costs &base,
                                    const fort_scope_batch_costs &costs, uint64_t operations) {
    using namespace fort_scoped::planning;
    fort_scope_batch_report report{}; report.version=FORT_SCOPE_BATCH_ABI_VERSION;
    report.selected_transfers=FORT_SCOPE_TRANSFERS_DIRECT;
    auto input=batch_inputs(c,plan,operations); detail::validate(input,base);
    auto baseline=detail::initial(input,base); uint64_t simulated=0;
    const double entry_terminal=detail::entry_terminal(input,base,simulated).seconds;
    for (const auto &op : input.operations) detail::execute(baseline,op,op.kind==FORT_SCOPE_PLAN_WORKER,input,base,simulated);
    for (size_t k=0; k<plan.resources.size(); ++k) if (!plan.resources[k].exported.empty()) {
        detail::seconds(baseline.time,base.host_access_seconds);
        detail::ensure(baseline,baseline.resources[k],plan.resources[k].exported,false,input,base); detail::wait(baseline,base);
    }
    detail::finish(baseline,input,base,simulated);
    report.baseline_seconds=baseline.execution_time;
    report.terminal_delta_seconds=baseline.terminal.seconds-entry_terminal;
    batch_require(std::isfinite(report.terminal_delta_seconds),FORT_SCOPE_BATCH_ARITHMETIC);
    double lifecycle=0,compute=0;
    if (!c.ready) lifecycle=base.gpu_setup_seconds+(input.driver_initialized ? 0 : base.cold_driver_startup_seconds);
    for (const auto &r : plan.resources) {
        if (r.used && r.original->bytes && !r.original->device) {
            batch_require(r.original->bytes<=base.max_allocation_bytes,FORT_SCOPE_BATCH_GEOMETRY);
            lifecycle+=base.allocation_seconds;
        }
        BatchCopies total; batch_copies_append(total,*r.original,r.incoming,true); batch_copies_append(total,*r.original,r.exported,false);
        report.upload_bytes=batch_add(report.upload_bytes,total.upload_bytes);
        report.download_bytes=batch_add(report.download_bytes,total.download_bytes);
    }
    for (const auto &unit : plan.units) if (unit.kind==FORT_SCOPE_PLAN_WORKER)
        compute+=std::max(unit.flops/base.gpu_flops,unit.memory/base.gpu_bandwidth);
    const auto requested=c.transfer_stats.requested_mode;
    double best=requested==FORT_SCOPE_TRANSFERS_PIPELINED ? std::numeric_limits<double>::infinity() : report.baseline_seconds;
    const auto one=batch_copies(plan,0,1,true);
    const auto two=batch_copies(plan,0,std::min<size_t>(plan.iterations,2),true);
    operations=batch_add(operations,one.operations+two.operations+simulated);
    const uint64_t bytes_one=std::max(one.upload_bytes,one.download_bytes), bytes_two=std::max(two.upload_bytes,two.download_bytes);
    const uint64_t per_iteration=std::max<uint64_t>(1,std::max(bytes_one,bytes_two>bytes_one ? bytes_two-bytes_one : 0));
    for (size_t capacity_index=0; capacity_index<FORT_SCOPE_BATCH_CAPACITIES; ++capacity_index) {
        const auto capacity=detail::batch_capacities[capacity_index];
        if (capacity>costs.max_slot_bytes || per_iteration>capacity) continue;
        size_t count=std::min<size_t>(plan.iterations,std::max<size_t>(1,capacity/per_iteration));
        if ((requested==FORT_SCOPE_TRANSFERS_PIPELINED || requested==FORT_SCOPE_TRANSFERS_AUTO) && plan.iterations>1)
            count=std::min(count,std::max<size_t>(1,plan.iterations/2));
        auto maximum=batch_copies(plan,0,count,true); operations=batch_add(operations,maximum.operations);
        while (std::max(maximum.upload_bytes,maximum.download_bytes)>capacity && count>1) {
            count=std::max<size_t>(1,count/2); maximum=batch_copies(plan,0,count,true); operations=batch_add(operations,maximum.operations);
        }
        if (std::max(maximum.upload_bytes,maximum.download_bytes)>capacity) continue;
        const uint64_t batches=plan.iterations/count+(plan.iterations%count!=0);
        double prefix_seconds=0; uint64_t prefix_bytes=0,prefix_calls=0;
        for (const auto &r : plan.resources) for (const auto &box : r.prefix) {
            fort_physical::CopyPlan physical(r.original->element_bytes,r.original->extents,box.lo,box.hi);
            batch_require(physical.valid,FORT_SCOPE_BATCH_ARITHMETIC);
            uint64_t rows=0,calls=0;
            physical.visit([&](const auto &op) {
                return fort_physical::visit_tiles(op,capacity,[&](const auto &tile) {
                    rows=batch_add(rows,multiply(tile.height,tile.depth)); calls=batch_add(calls,1);
                    operations=batch_add(operations,1); return true;
                });
            });
            prefix_seconds+=batch_copy_cost(physical.bytes,rows,calls,costs,true)+double(calls)*(costs.event_record_seconds+costs.event_wait_seconds);
            prefix_bytes=batch_add(prefix_bytes,physical.bytes); prefix_calls=batch_add(prefix_calls,calls);
        }
        const double h2d=double(maximum.uploads.size())*costs.pinned_h2d_latency+double(maximum.upload_bytes)/costs.pinned_h2d_bandwidth;
        const double d2h=double(maximum.downloads.size())*costs.pinned_d2h_latency+double(maximum.download_bytes)/costs.pinned_d2h_bandwidth;
        const double gpu=compute*double(count)/double(plan.iterations)+double(plan.workers)*base.launch_enqueue_seconds;
        const double pump=double(maximum.upload_bytes)/costs.pack_bytes_per_second+double(maximum.download_bytes)/costs.unpack_bytes_per_second+
            double(maximum.upload_rows)*costs.pack_row_seconds+double(maximum.download_rows)*costs.unpack_row_seconds+
            costs.event_record_seconds+costs.event_wait_seconds+double(maximum.operations)*costs.preparation_operation_seconds;
        const double fixed=lifecycle+costs.staging_cold_seconds[capacity_index]+2*costs.ready_event_seconds+prefix_seconds;
        const double pinned=fixed+double(batches)*(h2d+gpu+d2h+pump);
        const double copies=costs.async_engine_count>1 ? std::max(h2d,d2h) : h2d+d2h;
        const double period=std::max({copies,gpu,pump,(h2d+d2h+gpu)/2});
        const double pipelined=fixed+h2d+gpu+d2h+pump+double(batches-1)*period;
        batch_require(std::isfinite(pinned) && std::isfinite(pipelined),FORT_SCOPE_BATCH_ARITHMETIC);
        if (requested==FORT_SCOPE_TRANSFERS_AUTO && pinned<best) {
            best=pinned; report.selected_transfers=FORT_SCOPE_TRANSFERS_PINNED;
            report.chunk_iterations=count; report.slot_bytes=capacity; report.batches=batches;
            report.prefix_upload_bytes=prefix_bytes; report.prefix_uploads=prefix_calls;
        }
        if ((requested==FORT_SCOPE_TRANSFERS_AUTO || requested==FORT_SCOPE_TRANSFERS_PIPELINED) && costs.async_engine_count &&
            batches>1 && (maximum.upload_bytes || maximum.download_bytes) && pipelined<best) {
            best=pipelined; report.selected_transfers=FORT_SCOPE_TRANSFERS_PIPELINED;
            report.chunk_iterations=count; report.slot_bytes=capacity; report.batches=batches;
            report.prefix_upload_bytes=prefix_bytes; report.prefix_uploads=prefix_calls;
        }
        if (!report.pinned_seconds || pinned<report.pinned_seconds) report.pinned_seconds=pinned;
        if (!report.pipelined_seconds || pipelined<report.pipelined_seconds) report.pipelined_seconds=pipelined;
    }
    report.preparation_operations=operations;
    const double preparation=double(operations)*costs.preparation_operation_seconds;
    report.baseline_seconds+=preparation; best+=preparation;
    if (report.pinned_seconds) report.pinned_seconds+=preparation;
    if (report.pipelined_seconds) report.pipelined_seconds+=preparation;
    report.available=1; report.reason=FORT_SCOPE_BATCH_NONE;
    if (report.selected_transfers==FORT_SCOPE_TRANSFERS_DIRECT ||
        (requested==FORT_SCOPE_TRANSFERS_AUTO && !(best<=.8*report.baseline_seconds))) {
        report.selected_transfers=FORT_SCOPE_TRANSFERS_DIRECT;
        report.reason=requested==FORT_SCOPE_TRANSFERS_PIPELINED ? FORT_SCOPE_BATCH_GEOMETRY : FORT_SCOPE_BATCH_NO_ADVANTAGE;
        report.chunk_iterations=report.batches=report.slot_bytes=report.prefix_upload_bytes=report.prefix_uploads=0; best=report.baseline_seconds;
    }
    report.execution_seconds=best; report.estimated_seconds=best+report.terminal_delta_seconds;
    batch_require(!plan.workers || report.batches<=std::numeric_limits<uint64_t>::max()/plan.workers,FORT_SCOPE_BATCH_ARITHMETIC);
    report.launches=report.selected_transfers==FORT_SCOPE_TRANSFERS_DIRECT ? plan.workers : report.batches*plan.workers;
    batch_require(std::isfinite(report.baseline_seconds) && std::isfinite(report.execution_seconds) &&
                  std::isfinite(report.estimated_seconds) && std::isfinite(report.pinned_seconds) &&
                  std::isfinite(report.pipelined_seconds),FORT_SCOPE_BATCH_ARITHMETIC);
    return report;
}
const char *batch_reason(uint32_t reason) noexcept {
    switch (reason) {
        case FORT_SCOPE_BATCH_MISSING_COSTS: return "batch_transfer_estimates_unavailable";
        case FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN: return "batch_independence_not_proven";
        case FORT_SCOPE_BATCH_PLACEMENT: return "batch_requires_approved_gpu_chain";
        case FORT_SCOPE_BATCH_NO_ADVANTAGE: return "batch_20_percent_margin_not_met";
        case FORT_SCOPE_BATCH_BUDGET: return "pinned_budget_exhausted";
        case FORT_SCOPE_BATCH_ALLOCATION: return "batch_resource_allocation_failed";
        case FORT_SCOPE_BATCH_GEOMETRY: return "batch_geometry_not_supported";
        case FORT_SCOPE_BATCH_ARITHMETIC: return "batch_arithmetic_overflow";
        default: return "none";
    }
}
void batch_trace(fort_scope_t handle, const fort_scope_batch_report &report) noexcept {
    if (!diagnostics_enabled()) return;
    try {
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << std::setprecision(17) << "FORT_SCOPED evidence {\"schema_version\":1,\"event\":\"batch_statistics\",\"stats_version\":1,\"context\":" << handle
                  << ",\"available\":" << report.available << ",\"applied\":" << report.applied
                  << ",\"selected_transfers\":" << report.selected_transfers << ",\"reason\":\"" << batch_reason(report.reason) << "\""
                  << ",\"owner_cost_available\":" << report.owner_cost_available
                  << ",\"owner_cost_reason\":\"" << (report.applied ? "complete_owner_batch_reprice_unavailable" : "unchanged") << "\"";
#define FORT_BATCH_FIELD(name) std::cerr << ",\"" #name "\":" << report.name
        FORT_BATCH_FIELD(preparation_operations); FORT_BATCH_FIELD(batches); FORT_BATCH_FIELD(chunk_iterations); FORT_BATCH_FIELD(slot_bytes);
        FORT_BATCH_FIELD(upload_bytes); FORT_BATCH_FIELD(download_bytes); FORT_BATCH_FIELD(prefix_upload_bytes); FORT_BATCH_FIELD(prefix_uploads);
        FORT_BATCH_FIELD(launches); FORT_BATCH_FIELD(estimated_seconds); FORT_BATCH_FIELD(baseline_seconds); FORT_BATCH_FIELD(pinned_seconds);
        FORT_BATCH_FIELD(pipelined_seconds); FORT_BATCH_FIELD(execution_seconds); FORT_BATCH_FIELD(terminal_delta_seconds);
        FORT_BATCH_FIELD(completed_batches); FORT_BATCH_FIELD(actual_upload_bytes); FORT_BATCH_FIELD(actual_download_bytes); FORT_BATCH_FIELD(actual_launches);
#undef FORT_BATCH_FIELD
        std::cerr << "}\n";
    } catch (...) {}
}
void decision_trace(const fort_scope_plan_decision &d, const std::string &reason) {
    const char *enabled = std::getenv("FORT_RUNTIME_TRACE");
    if (!enabled || std::strcmp(enabled, "1")) return;
    std::lock_guard<std::mutex> lock(trace_mutex);
    std::cerr << "FORT_SCOPED decision available=" << d.available << " gpu_units=" << d.gpu_units
              << " cpu_units=" << d.cpu_units << " candidates=" << d.candidates
              << " simulated_operations=" << d.simulated_operations
              << " upload_bytes=" << d.upload_bytes << " download_bytes=" << d.download_bytes
              << " launches=" << d.launches << " waits=" << d.waits
              << " peak_device_bytes=" << d.peak_device_bytes
              << " estimated_seconds=" << d.estimated_seconds << " native_seconds=" << d.native_seconds
              << " reason=" << (reason.empty() ? "calibrated_selection" : reason) << '\n';
}
void definition_validation_trace(Context &context, fort_scope_t handle,
                                 const fort_scoped::planning::DefinitionValidation &proof,
                                 bool cache_hit, double seconds) noexcept {
    const char *enabled = std::getenv("FORT_RUNTIME_TRACE");
    if (!enabled || std::strcmp(enabled, "1")) return;
    try {
        std::ostringstream out;
        out << std::setprecision(17) << "{\"schema_version\":1,\"context\":" << handle
            << ",\"event\":\"definition_validation\",\"status\":" << proof.status
            << ",\"cache_hit\":" << (cache_hit ? "true" : "false")
            << ",\"seconds\":" << seconds
            << ",\"query_generation\":" << context.query_generation
            << ",\"state_generation\":" << context.state_generation
            << ",\"reason\":" << std::quoted(proof.reason)
            << ",\"evidence\":\"ordered_definition_preflight\",\"mutates_live_state\":false";
        if (proof.operation < context.plan.size()) {
            out << ",\"operation\":" << proof.operation << ",\"unit\":" << context.plan[proof.operation].unit;
            const auto found = context.buffers.find(proof.buffer);
            if (found != context.buffers.end()) {
                const auto &b = *found->second;
                out << ",\"resource\":" << b.handle << ",\"identity\":" << b.identity
                    << ",\"generation\":" << b.generation;
            }
        }
        out << '}';
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << "FORT_SCOPED evidence " << out.str() << '\n';
    } catch (...) { /* Optional diagnostics must not change the proof result. */ }
}
struct EvidenceOutput {
    Context &context;
    fort_scope_t handle;
    size_t rows = 0;
    bool truncated = false;
};
void json_coordinates(std::ostream &out, const std::vector<size_t> &values) {
    out << '[';
    for (size_t k=0; k<values.size(); ++k) { if (k) out << ','; out << values[k]; }
    out << ']';
}
void json_rectangle(std::ostream &out, const Box &box) {
    out << "{\"lower\":"; json_coordinates(out, box.lo);
    out << ",\"upper\":"; json_coordinates(out, box.hi); out << '}';
}
void json_region(std::ostream &out, const Region &region) {
    out << '[';
    for (size_t k=0; k<region.size(); ++k) { if (k) out << ','; json_rectangle(out, region[k]); }
    out << ']';
}
void json_terminal_cost(std::ostream &out, const fort_scope_terminal_cost &cost) {
    out << "{\"seconds\":" << cost.seconds << ",\"download_bytes\":" << cost.download_bytes
        << ",\"downloads\":" << cost.downloads << ",\"waits\":" << cost.waits
        << ",\"releases\":" << cost.releases << '}';
}
void json_continuation(std::ostream &out, const fort_scope_plan_report &report) {
    out << ",\"continuation\":{\"report_version\":" << report.version
        << ",\"endpoint_mode\":" << report.endpoint_mode
        << ",\"available\":" << report.available
        << ",\"execution_counter_estimates_available\":" << report.available
        << ",\"execution_seconds\":" << report.execution_seconds
        << ",\"native_execution_seconds\":" << report.native_execution_seconds
        << ",\"entry_terminal\":"; json_terminal_cost(out, report.entry_terminal);
    out << ",\"terminal\":"; json_terminal_cost(out, report.terminal);
    out << ",\"native_terminal\":"; json_terminal_cost(out, report.native_terminal);
    out << ",\"terminal_hypothetical\":true,\"ranking_seconds\":" << report.ranking_seconds
        << ",\"native_ranking_seconds\":" << report.native_ranking_seconds
        << ",\"owner_available\":" << report.owner_available
        << ",\"owner_totals_completed_only\":true"
        << ",\"owner_segments\":" << report.owner_segments
        << ",\"owner_execution_seconds\":" << report.owner_execution_seconds
        << ",\"owner_terminal_seconds\":" << report.owner_terminal_seconds
        << ",\"owner_complete_seconds\":" << report.owner_complete_seconds
        << ",\"native_common_compute_excluded\":" << report.native_common_compute_excluded << '}';
}
void evidence_row(void *opaque, const fort_scoped::planning::EvidenceEvent &event) {
    auto &output = *static_cast<EvidenceOutput *>(opaque);
    // A trace can be incomplete, but may never retain an unbounded event graph.
    if (output.rows >= 4096) { output.truncated = true; return; }
    std::ostringstream out;
    out << std::setprecision(17) << "{\"schema_version\":1,\"context\":" << output.handle
        << ",\"sequence\":" << output.rows++ << ",\"event\":\"" << event.event << '"';
    if (event.phase) out << ",\"phase\":\"" << event.phase << '"';
    if (event.resource) {
        const auto &resource = *event.resource;
        const auto &registered = buffer(output.context, resource.handle);
        out << ",\"resource\":" << resource.handle << ",\"identity\":" << registered.identity
            << ",\"generation\":" << registered.generation << ",\"element_bytes\":" << resource.element_bytes
            << ",\"allocation_bytes\":" << resource.bytes << ",\"allocated\":" << (resource.allocated ? "true" : "false")
            << ",\"extents\":"; json_coordinates(out, resource.extents);
        out << ",\"initialized\":"; json_region(out, resource.initialized);
        out << ",\"host_current\":"; json_region(out, resource.host_current);
        out << ",\"device_current\":"; json_region(out, resource.device_current);
    }
    if (event.operation) {
        const auto &operation = *event.operation;
        out << ",\"kind\":" << operation.kind << ",\"unit\":" << operation.unit
            << ",\"gpu\":" << (event.gpu ? "true" : "false")
            << ",\"gpu_supported\":" << (operation.gpu_available ? "true" : "false")
            << ",\"flops\":" << operation.flops << ",\"memory_bytes\":" << operation.memory_bytes;
    }
    if (event.binding) {
        out << ",\"reads\":"; json_region(out, event.binding->effects.reads);
        out << ",\"writes\":"; json_region(out, event.binding->effects.writes);
        out << ",\"overwrites\":"; json_region(out, event.binding->effects.overwrites);
    }
    if (event.preservation_reads) {
        out << ",\"preservation_reads\":"; json_region(out, *event.preservation_reads);
    }
    if (event.rectangle) {
        out << ",\"unit\":" << event.unit << ",\"direction\":\"" << (event.upload ? "h2d" : "d2h")
            << "\",\"rectangle\":"; json_rectangle(out, *event.rectangle);
        out << ",\"bytes\":" << event.bytes << ",\"copy_calls\":" << event.copies;
    }
    if (!std::strcmp(event.event, "gate")) {
        out << ",\"first_worker\":" << event.first << ",\"last_worker_exclusive\":" << event.last
            << ",\"estimated_seconds\":" << event.seconds << ",\"counterfactual_seconds\":" << event.counterfactual_seconds
            << ",\"required_saving_seconds\":" << event.required_saving
            << ",\"accepted\":" << (event.accepted ? "true" : "false");
    }
    out << '}';
    std::lock_guard<std::mutex> lock(trace_mutex);
    std::cerr << "FORT_SCOPED evidence " << out.str() << '\n';
}
void planning_evidence(Context &context, fort_scope_t handle, const fort_scope_plan_costs &costs,
                       const fort_scoped::planning::Result &result) noexcept {
    const char *enabled = std::getenv("FORT_RUNTIME_TRACE");
    if (!enabled || std::strcmp(enabled, "1")) return;
    EvidenceOutput output{context, handle};
    bool complete = false;
    try {
        {
            std::lock_guard<std::mutex> lock(trace_mutex);
            std::cerr << std::setprecision(17) << "FORT_SCOPED evidence {\"schema_version\":1,\"context\":" << handle
                  << ",\"event\":\"decision\",\"device\":" << context.device
                  << ",\"available\":" << result.decision.available << ",\"gpu_units\":" << result.decision.gpu_units
                  << ",\"cpu_units\":" << result.decision.cpu_units << ",\"reason\":\"" << result.reason
                  << "\",\"native_common_compute_excluded\":" << (result.native_common_compute_excluded ? "true" : "false")
                  << ",\"estimated_seconds\":" << result.decision.estimated_seconds
                  << ",\"native_seconds\":" << result.decision.native_seconds
                  << ",\"peak_device_bytes\":" << result.decision.peak_device_bytes
                  << ",\"upload_bytes\":" << result.decision.upload_bytes
                  << ",\"download_bytes\":" << result.decision.download_bytes
                  << ",\"launches\":" << result.decision.launches
                  << ",\"simulated_operations\":" << result.decision.simulated_operations
                  << ",\"aggregate_only\":" << (result.native_startup_shortcut ? "true" : "false")
                  << ",\"evidence\":\"" << (result.native_startup_shortcut ? "native_startup_lower_bound" : "modeled_final_schedule")
                  << "\",\"section_coordinates\":\"zero_based_exclusive\"";
            std::cerr << ",\"transfer_requested_mode\":" << context.transfer_stats.requested_mode
                      << ",\"transfer_effective_mode\":" << context.transfer_stats.effective_mode
                      << ",\"transfer_reason\":\"" << transfer_reason(context.transfer_stats.fallback_reason) << "\"";
            if (result.report.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE) json_continuation(std::cerr, result.report);
            std::cerr << "}\n";
        }
        const fort_scoped::planning::EvidenceSink sink{&output, evidence_row};
        auto input = planning_inputs(context, handle);
        input.driver_initialized = result.driver_initialized;
        fort_scoped::planning::evidence(input, costs, result, sink);
        complete = !output.truncated;
    } catch (...) {
        // Diagnostics never convert a valid numerical decision into failure.
    }
    try {
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << "FORT_SCOPED evidence {\"schema_version\":1,\"context\":" << handle
                  << ",\"event\":\"end\",\"rows\":" << output.rows
                  << ",\"complete\":" << (complete ? "true" : "false")
                  << ",\"truncated\":" << (output.truncated ? "true" : "false") << "}\n";
    } catch (...) {}
}
}
extern "C" int fort_scope_plan_host_current(fort_scope_t h, fort_buffer_t handle) {
    return with(h, [&](Context &c) {
        const auto &b = buffer(c, handle);
        require(!b.prepared, FORT_SCOPE_STATE, "planning payload has an unfinished access");
        Budget budget;
        const Region full = empty(b.full()) ? Region{} : Region{b.full()};
        require(difference(full, b.initialized, budget).empty() &&
                difference(full, b.host_current, budget).empty(), FORT_SCOPE_BOUNDARY,
                "planning payload must already be fully initialized and host current");
    }, Change::None);
}
extern "C" int fort_scope_plan_reset_mode(fort_scope_t h, uint32_t endpoint_mode) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "query_construction");
        require(endpoint_mode <= FORT_SCOPE_PLAN_CONTINUE, FORT_SCOPE_ARGUMENT, "invalid planning endpoint mode");
        require(!c.pending, FORT_SCOPE_STATE, "planning requires completed earlier execution");
        for (const auto &entry : c.buffers)
            require(!entry.second->prepared, FORT_SCOPE_STATE, "planning during a prepared buffer access");
        require(!c.plan_installed || c.worker_cursor == c.schedule.size(), FORT_SCOPE_STATE,
                "cannot discard an unfinished execution schedule");
        complete_segment_estimate(c);
        if (c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE && endpoint_mode == FORT_SCOPE_PLAN_COMPLETE)
            c.owner_estimate_available = false;
        c.plan.clear(); c.schedule.clear(); c.worker_cursor = 0;
        c.plan_installed = false; c.plan_recording = true;
        c.endpoint_mode = endpoint_mode; c.last_report.reset();
    }, Change::Query);
}
extern "C" int fort_scope_plan_reset(fort_scope_t h) {
    return fort_scope_plan_reset_mode(h, FORT_SCOPE_PLAN_COMPLETE);
}
extern "C" int fort_scope_plan_report_v2(fort_scope_t h, fort_scope_plan_report *out) {
    return with(h, [&](Context &c) {
        require(out && c.last_report.has_value(), FORT_SCOPE_STATE, "no finalized planning cost report");
        *out = *c.last_report;
        out->owner_segments = c.owner_segments;
        out->owner_available = c.owner_estimate_available && c.owner_segments && !c.segment_cost_pending;
        out->owner_execution_seconds = c.owner_execution_seconds;
        out->owner_terminal_seconds = c.owner_terminal_seconds;
        out->owner_complete_seconds = c.owner_execution_seconds+c.owner_terminal_seconds;
        out->native_common_compute_excluded |= c.owner_native_common_compute_excluded;
    }, Change::None);
}
extern "C" int fort_scope_plan_add(fort_scope_t h, uint32_t kind, uint64_t unit,
                                   const fort_scope_plan_binding *bindings, size_t count,
                                   double flops, double memory_bytes, int gpu_available) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "query_construction");
        require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE, "reset the planning query before recording");
        require(kind <= FORT_SCOPE_PLAN_FORGET && (gpu_available == 0 || gpu_available == 1),
                FORT_SCOPE_ARGUMENT, "invalid planning operation kind or availability");
        require(c.plan.size() < 256, FORT_SCOPE_BOUNDARY, "planning record budget exceeded");
        require(kind != FORT_SCOPE_PLAN_WORKER || unit, FORT_SCOPE_ARGUMENT, "worker planning unit requires an identity");
        c.plan.push_back({kind, unit, planning_bindings(c, bindings, count), flops, memory_bytes, gpu_available != 0});
    }, Change::Query);
}
extern "C" int fort_scope_set_team_costs_v1(fort_scope_t h, const fort_scope_team_costs *costs, int compatible) {
    return with(h, [&](Context &c) {
        require(compatible == 0 || compatible == 1, FORT_SCOPE_ARGUMENT, "invalid collective profile compatibility");
        require(c.buffers.empty() && !c.ready && !c.plan_recording && !c.plan_installed,
                FORT_SCOPE_STATE, "configure collective costs before registration and planning");
        c.team_costs.reset();
        if (!compatible) return;
        require(costs && fort_scoped::planning::detail::valid_team_costs(*costs),
                FORT_SCOPE_BOUNDARY, "collective synchronization calibration is unavailable");
#ifdef _OPENMP
        if (omp_get_level() == int(costs->expected_omp_level) && omp_get_num_threads() == int(costs->cpu_threads))
            c.team_costs = *costs;
#endif
    });
}
extern "C" int fort_scope_team_costs_ready_v1(fort_scope_t h, int *out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing collective readiness output");
        *out = actual_team_matches(c) ? 1 : 0;
    }, Change::None);
}
extern "C" int fort_scope_plan_team_entry_v1(fort_scope_t h) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "query_construction");
        require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE, "reset the planning query before recording");
        require(c.plan.size() < 256, FORT_SCOPE_BOUNDARY, "planning record budget exceeded");
        c.plan.push_back({FORT_SCOPE_PLAN_TEAM_ENTRY, 0, {}, 0, 0, false});
    }, Change::Query);
}
extern "C" int fort_scope_plan_team_native_call_v1(fort_scope_t h) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "query_construction");
        require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE, "reset the planning query before recording");
        require(c.plan.size() < 256, FORT_SCOPE_BOUNDARY, "planning record budget exceeded");
        c.plan.push_back({FORT_SCOPE_PLAN_TEAM_NATIVE_CALL, 0, {}, 0, 0, false});
    }, Change::Query);
}
extern "C" int fort_scope_plan_forget_sections_v1(fort_scope_t h, fort_buffer_t handle,
                                                  const fort_scope_section *items, size_t count) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "query_construction");
        require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE, "reset the planning query before recording");
        require(c.plan.size() < 256, FORT_SCOPE_BOUNDARY, "planning record budget exceeded");
        auto &b = buffer(c, handle);
        auto removed = sections(b, items, count, false);
        fort_scoped::planning::Binding binding{handle, {}};
        binding.effects.writes = std::move(removed);
        c.plan.push_back({FORT_SCOPE_PLAN_DISCARD, 0, {std::move(binding)}, 0, 0, false});
    }, Change::Query);
}
extern "C" int fort_scope_plan_validate(fort_scope_t h) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "validation");
        fort_scoped::planning::DefinitionValidation proof;
        try {
            require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE,
                    "definition validation requires a complete recorded query");
            require(!c.pending, FORT_SCOPE_STATE, "definition validation requires completed earlier execution");
            timing.cache_hit = c.definition_proof &&
                c.definition_proof->query_generation == c.query_generation &&
                c.definition_proof->state_generation == c.state_generation;
            proof = timing.cache_hit ? c.definition_proof->result :
                fort_scoped::planning::validate_definitions(planning_inputs(c, h));
        } catch (const Error &error) {
            c.definition_proof.reset(); c.preview.reset();
            if (c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE) c.owner_estimate_available = false;
            proof.status = error.status; proof.reason = error.what();
            definition_validation_trace(c, h, proof, false, timing.seconds()); throw;
        } catch (const std::bad_alloc &) {
            c.definition_proof.reset(); c.preview.reset();
            if (c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE) c.owner_estimate_available = false;
            proof.status = FORT_SCOPE_RESOURCE; proof.reason = "planning_resource_failure";
            definition_validation_trace(c, h, proof, false, timing.seconds()); throw;
        } catch (const Fragmented &) {
            c.definition_proof.reset(); c.preview.reset();
            if (c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE) c.owner_estimate_available = false;
            proof.status = FORT_SCOPE_BOUNDARY; proof.reason = "region_fragmentation_unavailable";
            definition_validation_trace(c, h, proof, false, timing.seconds()); throw;
        } catch (...) {
            c.definition_proof.reset(); c.preview.reset();
            if (c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE) c.owner_estimate_available = false;
            proof.status = FORT_SCOPE_STATE; proof.reason = "planning_state_unavailable";
            definition_validation_trace(c, h, proof, false, timing.seconds()); throw;
        }
        definition_validation_trace(c, h, proof, timing.cache_hit, timing.seconds());
        if (proof.status != FORT_SCOPE_OK) {
            c.definition_proof.reset(); c.preview.reset();
            if (c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE) c.owner_estimate_available = false;
            std::ostringstream message;
            message << "definition preflight: " << proof.reason;
            if (proof.operation < c.plan.size()) {
                message << " operation=" << proof.operation << " unit=" << c.plan[proof.operation].unit;
                if (proof.buffer) {
                    const auto &b = buffer(c, proof.buffer);
                    message << " resource=" << b.identity << " generation=" << b.generation;
                }
            }
            throw Error(proof.status, message.str().c_str());
        }
        c.definition_proof = Context::Proof{c.query_generation, c.state_generation, proof};
    }, Change::None);
}
extern "C" int fort_scope_plan_select(fort_scope_t h, const fort_scope_plan_costs *costs,
                                      int compatible, fort_scope_plan_decision *out) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "selection");
        require(costs && out && compatible >= -1 && compatible <= 1, FORT_SCOPE_ARGUMENT, "invalid planning selection arguments");
        require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE, "planning selection requires a completed query");
        if (c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE) {
            require(!c.pending, FORT_SCOPE_STATE, "planning requires completed earlier execution");
            if (!(c.definition_proof && c.definition_proof->query_generation == c.query_generation &&
                  c.definition_proof->state_generation == c.state_generation)) {
                const auto proof = fort_scoped::planning::validate_definitions(planning_inputs(c, h));
                if (proof.status != FORT_SCOPE_OK) {
                    c.owner_estimate_available = false;
                    throw Error(proof.status, proof.reason);
                }
                c.definition_proof = Context::Proof{c.query_generation, c.state_generation, proof};
            }
        }
        const bool team_mismatch = c.team_costs && !actual_team_matches(c);
        timing.cache_hit = !team_mismatch && c.preview && c.preview_query_generation == c.query_generation &&
            c.preview_state_generation == c.state_generation &&
            c.preview->driver_initialized == driver_initialized(c) &&
            !std::memcmp(c.preview_costs.data(), costs, sizeof(*costs));
        auto pricing = *costs;
        // The v1 profile describes direct copies. Pinned preparation, tiling
        // and completion costs are not calibrated by that profile.
        const bool pinned_uncalibrated = c.transfer_stats.requested_mode == FORT_SCOPE_TRANSFERS_PINNED &&
            (!c.transfer_costs || !fort_scoped::planning::detail::valid_transfer_costs(*c.transfer_costs));
        if (pinned_uncalibrated || team_mismatch) pricing.valid = 0;
        auto result = timing.cache_hit
            ? *c.preview : fort_scoped::planning::select(planning_inputs(c, h), pricing);
        if (compatible == -1 && !team_mismatch) {
            c.preview = result;
            c.preview_query_generation = c.query_generation;
            c.preview_state_generation = c.state_generation;
            std::memcpy(c.preview_costs.data(), costs, sizeof(*costs));
        }
        if (compatible == 0) {
            result.decision.available = 0;
            result.decision.gpu_units = 0;
            result.decision.cpu_units = static_cast<uint32_t>(result.gpu_workers.size());
            result.decision.upload_bytes = result.decision.download_bytes = 0;
            result.decision.uploads = result.decision.downloads = 0;
            result.decision.launches = result.decision.waits = result.decision.allocations = 0;
            result.decision.peak_device_bytes = c.stats.allocated_bytes;
            result.decision.estimated_seconds = result.decision.native_seconds;
            std::fill(result.gpu_workers.begin(), result.gpu_workers.end(), false);
            result.reason = "hardware_or_calibration_incompatible";
            result.report.available = 0;
        }
        if (pinned_uncalibrated) result.reason = "transfer_estimates_unavailable";
        if (team_mismatch) result.reason = "collective_team_mismatch";
        *out = result.decision;
        if (compatible != -1) {
            if (c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE) {
                c.owner_create_accounted = true; c.registrations_incurred = 0;
                c.segment_cost_pending = true;
                if (!result.report.available) c.owner_estimate_available = false;
            } else c.owner_estimate_available = false; // Legacy native fallback can run outside context hooks.
            result.report.owner_segments = c.owner_segments;
            result.report.owner_available = c.owner_estimate_available && c.owner_segments && !c.segment_cost_pending;
            result.report.owner_execution_seconds = c.owner_execution_seconds;
            result.report.owner_terminal_seconds = c.owner_terminal_seconds;
            result.report.owner_complete_seconds = c.owner_execution_seconds+c.owner_terminal_seconds;
            result.report.native_common_compute_excluded |= c.owner_native_common_compute_excluded;
            c.last_report = result.report;
            planning_evidence(c, h, *costs, result);
            c.schedule = std::move(result.gpu_workers);
            c.worker_cursor = 0;
            // A successful whole-native choice leaves original source fallback
            // free to close this untouched metadata-only context.
            c.plan_installed = c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE ||
                (result.decision.available && result.decision.gpu_units);
            if (!c.plan_installed) c.schedule.clear();
            c.plan_recording = false;
            decision_trace(*out, result.reason);
        }
    }, Change::None);
}
extern "C" int fort_scope_plan_next(fort_scope_t h, uint64_t unit,
                                    const fort_scope_plan_binding *bindings, size_t count, int *gpu) {
    return with(h, [&](Context &c) {
        require(gpu, FORT_SCOPE_ARGUMENT, "missing execution choice output");
        *gpu = 0;
        if (!c.plan_installed) return;
        require(c.worker_cursor < c.schedule.size(), FORT_SCOPE_STATE, "execution has more workers than its planned query");
        size_t worker = 0;
        const fort_scoped::planning::Operation *expected = nullptr;
        for (const auto &op : c.plan) if (op.kind == FORT_SCOPE_PLAN_WORKER) {
            if (worker++ == c.worker_cursor) { expected = &op; break; }
        }
        require(expected && expected->unit == unit &&
                same_bindings(expected->bindings, planning_bindings(c, bindings, count)),
                FORT_SCOPE_STATE, "execution worker order or physical effects changed after planning");
        *gpu = c.schedule[c.worker_cursor++] ? 1 : 0;
    });
}

extern "C" uint32_t fort_scope_abi_version() { return FORT_SCOPE_ABI_VERSION; }
extern "C" const char *fort_scope_error() { return last_error; }
extern "C" int fort_scope_create(int device, fort_scope_t *out) {
    return protect([&]() {
        require(out && device >= 0, FORT_SCOPE_ARGUMENT, "invalid context output or device");
        auto c = std::make_shared<Context>(device);
        auto handle = token();
        std::lock_guard<std::mutex> lock(registry_mutex);
        contexts.emplace(handle, std::move(c)); *out = handle;
    });
}
extern "C" int fort_scope_serial_caller(void) {
#ifdef _OPENMP
    return !omp_in_parallel();
#else
    // Without OpenMP support this runtime cannot determine whether its host
    // is in a team. Unknown participation retains the original native span.
    return 0;
#endif
}
extern "C" int fort_scope_device_get(fort_scope_t h, int *out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing context device output");
        *out = c.device;
    }, Change::None);
}
extern "C" int fort_scope_set_device_budget(fort_scope_t h, size_t bytes) {
    return with(h, [&](Context &c) {
        require(!c.ready, FORT_SCOPE_STATE, "device budget must be set before CUDA initialization");
        c.device_budget = bytes;
    });
}
extern "C" int fort_scope_set_transfers(fort_scope_t h, uint32_t mode) {
    return with(h, [&](Context &c) {
        require(mode <= FORT_SCOPE_TRANSFERS_AUTO, FORT_SCOPE_ARGUMENT, "invalid scoped transfer mode");
        require(c.buffers.empty() && !c.ready && !c.plan_recording && !c.plan_installed,
                FORT_SCOPE_STATE, "configure scoped transfers before registration and planning");
        auto &stats = c.transfer_stats;
        stats.requested_mode = mode;
        stats.effective_mode = mode == FORT_SCOPE_TRANSFERS_PINNED ? mode : uint32_t(FORT_SCOPE_TRANSFERS_DIRECT);
        stats.fallback_reason = FORT_SCOPE_TRANSFER_NONE;
        if (mode == FORT_SCOPE_TRANSFERS_AUTO) transfer_fallback(c, FORT_SCOPE_TRANSFER_ESTIMATES_UNAVAILABLE);
        if (mode == FORT_SCOPE_TRANSFERS_PIPELINED) transfer_fallback(c, FORT_SCOPE_TRANSFER_PIPELINED_UNAVAILABLE);
    }, Change::Query);
}
extern "C" int fort_scope_transfer_stats_get_v1(fort_scope_t h, fort_scope_transfer_stats *out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing scoped transfer statistics output");
        *out = transfer_statistics(c);
    }, Change::None);
}
extern "C" int fort_scope_set_transfer_costs_v1(fort_scope_t h, const fort_scope_batch_costs *costs,
                                               int compatible) {
    return with(h, [&](Context &c) {
        require(compatible == 0 || compatible == 1, FORT_SCOPE_ARGUMENT, "invalid transfer calibration compatibility");
        require(c.buffers.empty() && !c.ready && !c.plan_recording && !c.plan_installed,
                FORT_SCOPE_STATE, "configure scoped transfer pricing before registration and planning");
        c.transfer_costs.reset();
        if (compatible && costs && fort_scoped::planning::detail::valid_transfer_costs(*costs)) {
            c.transfer_costs = *costs;
            if (c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_AUTO || c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_PIPELINED) {
                c.transfer_stats.fallback_reason=FORT_SCOPE_TRANSFER_NONE; c.transfer_stats.fallbacks=0;
            }
        }
    });
}
extern "C" int fort_scope_batch_report_get_v1(fort_scope_t h, fort_scope_batch_report *out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing batch report output");
        *out = c.batch_report.value_or(fort_scope_batch_report{});
        out->version = FORT_SCOPE_BATCH_ABI_VERSION;
    }, Change::None);
}
extern "C" int fort_scope_batch_execute_v1(fort_scope_t h, const fort_scope_batch *descriptor,
        const fort_scope_plan_costs *base, const fort_scope_batch_costs *transfers, int compatible,
        fort_scope_batch_worker worker, void *user, fort_scope_batch_report *out) {
    return with(h,[&](Context &c) {
        require(descriptor && out && compatible>=-1 && compatible<=1,FORT_SCOPE_ARGUMENT,"invalid scoped batch arguments");
        require(!c.batch_active,FORT_SCOPE_STATE,"only one scoped batch may be active");
        *out={}; out->version=FORT_SCOPE_BATCH_ABI_VERSION; out->preparation_operations=1;
        const auto *pricing=transfers ? transfers : c.transfer_costs ? &*c.transfer_costs : nullptr;
        auto observation=[&]() {
            if (compatible!=-1) { c.batch_report=*out; batch_trace(h,*out); }
        };
        if (!compatible || !base || !pricing || !fort_scoped::planning::detail::valid_transfer_costs(*pricing)) {
            out->reason=FORT_SCOPE_BATCH_MISSING_COSTS; observation(); return;
        }
        BatchPlan plan; size_t consume=0;
        try {
            plan=prepare_batch(c,*descriptor);
            consume=batch_schedule(c,plan,descriptor->execution_mode,plan.operations);
            *out=price_batch(c,plan,*base,*pricing,plan.operations);
        } catch (const BatchDecline &decline) {
            out->reason=decline.reason; out->preparation_operations=std::max<uint64_t>(1,plan.operations); observation(); return;
        } catch (const Fragmented &) {
            out->reason=FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN; observation(); return;
        } catch (const fort_scoped::planning::detail::Unavailable &) {
            out->reason=FORT_SCOPE_BATCH_MISSING_COSTS; observation(); return;
        } catch (const std::bad_alloc &) {
            out->reason=FORT_SCOPE_BATCH_ALLOCATION; observation(); return;
        }
        if (compatible==-1) return;
        if (out->selected_transfers==FORT_SCOPE_TRANSFERS_DIRECT) { observation(); return; }
        require(worker,FORT_SCOPE_ARGUMENT,"selected scoped batch requires a numerical callback");
        // All storage is acquired before the first transfer/numerical callback.
        // A short nonblocking staging lease cannot deadlock with another owner.
        invalidate(c,Change::State);
#ifdef FORT_SCOPE_CPU_TEST
        std::array<std::vector<unsigned char>,2> storage;
        std::array<bool,2> pending{};
        try {
            require(!std::getenv("FORT_SCOPE_TEST_FAIL_BATCH_ALLOC"),FORT_SCOPE_RESOURCE,"injected batch resource failure");
            for (auto &slot : storage) slot.resize(out->slot_bytes);
            for (auto &r : plan.resources) if (r.used) allocate(c,*r.original);
        } catch (const std::bad_alloc &) {
            out->reason=FORT_SCOPE_BATCH_ALLOCATION; c.owner_estimate_available=false; observation(); return;
        } catch (const Error &error) {
            if (error.status!=FORT_SCOPE_RESOURCE) throw;
            out->reason=FORT_SCOPE_BATCH_ALLOCATION; c.owner_estimate_available=false; observation(); return;
        }
        const size_t actual_capacity=out->slot_bytes;
        auto packed=[&](size_t slot){return storage[slot].data();};
        auto mark_pending=[&](size_t slot,bool value){pending[slot]=value;};
        auto stream=[&](size_t slot)->void* {return reinterpret_cast<void*>(slot+1);};
        c.transfer_stats.staging_allocations+=2;
#else
        try { initialize(c); }
        catch (const Error &error) {
            if (error.status!=FORT_SCOPE_RESOURCE) throw;
            out->reason=FORT_SCOPE_BATCH_ALLOCATION; c.owner_estimate_available=false; observation(); return;
        }
        DeviceGuard guard(c);
        auto lease=fort_staging::acquire(out->slot_bytes,fort_staging::Role::Staging,false);
        if (lease.exhausted) { out->reason=FORT_SCOPE_BATCH_BUDGET; observation(); return; }
        if (lease.status!=cudaSuccess) {
            try { cuda_check(c,lease.status,true); }
            catch (const Error &error) {
                if (error.status!=FORT_SCOPE_RESOURCE) throw;
                out->reason=FORT_SCOPE_BATCH_ALLOCATION; c.owner_estimate_available=false; observation(); return;
            }
        }
        require(lease.slots!=nullptr,FORT_SCOPE_STATE,"missing batch staging lease");
        const size_t actual_capacity=lease.slots->capacity;
        auto packed=[&](size_t slot){return lease.slots->slots[slot].host;};
        auto mark_pending=[&](size_t slot,bool value){lease.slots->slots[slot].pending=value;};
        auto stream=[&](size_t slot)->void* {return reinterpret_cast<void*>(lease.slots->slots[slot].stream);};
        c.transfer_stats.staging_allocations+=lease.reused ? 0 : 2;
        c.transfer_stats.staging_reuses+=lease.reused ? 1 : 0;
        // Reprice the acquisition actually incurred before numerical work.
        size_t requested_index=0,actual_index=0;
        while (requested_index+1<FORT_SCOPE_BATCH_CAPACITIES && fort_scoped::planning::detail::batch_capacities[requested_index]<out->slot_bytes) ++requested_index;
        while (actual_index+1<FORT_SCOPE_BATCH_CAPACITIES && fort_scoped::planning::detail::batch_capacities[actual_index]<actual_capacity) ++actual_index;
        if (lease.reused) {
            const double change=pricing->staging_reuse_seconds[actual_index]-pricing->staging_cold_seconds[requested_index];
            out->execution_seconds+=change; out->estimated_seconds+=change;
        }
        try { for (auto &r : plan.resources) if (r.used) allocate(c,*r.original); }
        catch (const Error &error) {
            if (error.status!=FORT_SCOPE_RESOURCE) throw;
            out->reason=FORT_SCOPE_BATCH_ALLOCATION; c.owner_estimate_available=false; observation(); return;
        }
#endif
        if (!std::isfinite(out->execution_seconds) || !std::isfinite(out->estimated_seconds) ||
            (c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_AUTO && !(out->execution_seconds<=.8*out->baseline_seconds))) {
            out->reason=FORT_SCOPE_BATCH_NO_ADVANTAGE; observation(); return;
        }
        c.transfer_stats.slot_capacity=actual_capacity;
        std::vector<fort_scope_batch_view> views;
        views.reserve(plan.resources.size());
        for (auto &r : plan.resources) {
            auto &b=*r.original;
            views.push_back({b.handle,b.device,{uint32_t(b.extents.size()),b.type,b.element_bytes,b.host,b.extents.data(),b.lower.data(),b.generation}});
        }
        std::array<BatchCopies,2> copies;
        std::array<bool,2> active{};
        auto enqueue=[&](const BatchCopy &copy,size_t slot,bool upload) {
            const auto &op=copy.operation; const auto bytes=multiply(multiply(op.width,op.height),op.depth);
            auto *compact=packed(slot)+copy.packed_offset;
            auto *device=static_cast<char*>(copy.buffer->device);
#ifdef FORT_SCOPE_CPU_TEST
            pack_tile(compact,device,op,upload);
#else
            const auto direction=upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
            auto *to=upload ? device+op.offset : reinterpret_cast<char*>(compact);
            const auto *from=upload ? reinterpret_cast<const char*>(compact) : device+op.offset;
            const size_t to_pitch=upload ? op.pitch : op.width, from_pitch=upload ? op.width : op.pitch;
            auto cuda_stream=static_cast<cudaStream_t>(stream(slot));
            if (op.depth==1) {
                if (op.height==1) cuda_check(c,cudaMemcpyAsync(to,from,op.width,direction,cuda_stream));
                else cuda_check(c,cudaMemcpy2DAsync(to,to_pitch,from,from_pitch,op.width,op.height,direction,cuda_stream));
            } else {
                cudaMemcpy3DParms parameters{};
                parameters.srcPtr=make_cudaPitchedPtr(const_cast<char*>(from),from_pitch,op.width,upload ? op.height : op.physical_height);
                parameters.dstPtr=make_cudaPitchedPtr(to,to_pitch,op.width,upload ? op.physical_height : op.height);
                parameters.extent=make_cudaExtent(op.width,op.height,op.depth); parameters.kind=direction;
                cuda_check(c,cudaMemcpy3DAsync(&parameters,cuda_stream));
            }
#endif
            mark_pending(slot,true);
            if (upload) {
                ++c.stats.uploads; c.stats.upload_bytes+=bytes;
                ++c.transfer_stats.pinned_uploads; c.transfer_stats.pinned_upload_bytes+=bytes; out->actual_upload_bytes+=bytes;
            } else {
                ++c.stats.downloads; c.stats.download_bytes+=bytes;
                ++c.transfer_stats.pinned_downloads; c.transfer_stats.pinned_download_bytes+=bytes; out->actual_download_bytes+=bytes;
            }
            ++c.transfer_stats.tiles; trace(upload ? "upload" : "download",copy.buffer,bytes);
        };
        auto record=[&](size_t slot) {
#ifdef FORT_SCOPE_CPU_TEST
            (void)slot;
#endif
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
            require(!std::getenv("FORT_SCOPE_TEST_FAIL_BATCH_RECORD"),FORT_SCOPE_EXECUTION,"injected batch completion record failure");
#endif
#ifndef FORT_SCOPE_CPU_TEST
            cuda_check(c,cudaEventRecord(lease.slots->slots[slot].complete,static_cast<cudaStream_t>(stream(slot))));
#endif
            ++c.transfer_stats.events;
        };
        auto wait_slot=[&](size_t slot) {
            const auto started=std::chrono::steady_clock::now();
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
            require(!std::getenv("FORT_SCOPE_TEST_FAIL_BATCH_WAIT"),FORT_SCOPE_EXECUTION,"injected batch completion wait failure");
#endif
#ifndef FORT_SCOPE_CPU_TEST
            cuda_check(c,cudaEventSynchronize(lease.slots->slots[slot].complete));
#endif
            ++c.transfer_stats.event_waits;
            c.transfer_stats.event_wait_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            mark_pending(slot,false);
        };
        auto pack=[&](const BatchCopy &copy,size_t slot,bool unpack) {
            const auto started=std::chrono::steady_clock::now();
            pack_tile(packed(slot)+copy.packed_offset,static_cast<char*>(copy.buffer->host),copy.operation,unpack);
            const auto bytes=multiply(multiply(copy.operation.width,copy.operation.height),copy.operation.depth);
            const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            if (unpack) { c.transfer_stats.unpacked_bytes+=bytes; c.transfer_stats.unpacking_seconds+=seconds; }
            else { c.transfer_stats.packed_bytes+=bytes; c.transfer_stats.packing_seconds+=seconds; }
        };
        auto finish=[&](size_t slot) {
            if (!active[slot]) return;
            wait_slot(slot);
            for (const auto &copy : copies[slot].downloads) pack(copy,slot,true);
            active[slot]=false; ++out->completed_batches;
        };
        try {
            // Ready-event waits capture this record; later slot completion
            // records cannot change those already queued dependencies.
            c.batch_active=true; out->applied=1; c.owner_estimate_available=false;
            if (c.last_report) c.last_report->available=0;
            c.worker_cursor+=consume;
#ifndef FORT_SCOPE_CPU_TEST
            cuda_check(c,cudaEventRecord(lease.slots->slots[0].complete,c.stream)); ++c.transfer_stats.events;
            for (size_t slot=0; slot<2; ++slot) {
                cuda_check(c,cudaStreamWaitEvent(static_cast<cudaStream_t>(stream(slot)),lease.slots->slots[0].complete,0));
            }
#endif
            // Exact immutable unions complete before any batch kernel starts.
            for (const auto &r : plan.resources) for (const auto &box : r.prefix) {
                fort_physical::CopyPlan physical(r.original->element_bytes,r.original->extents,box.lo,box.hi);
                physical.visit([&](const auto &op) {
                    return fort_physical::visit_tiles(op,actual_capacity,[&](const auto &tile) {
                        BatchCopy copy{r.original,tile,0}; pack(copy,0,false); enqueue(copy,0,true); record(0); wait_slot(0); return true;
                    });
                });
            }
            size_t ordinal=0,slot=0;
            while (ordinal<plan.iterations) {
                finish(slot);
                const auto count=std::min<size_t>(out->chunk_iterations,plan.iterations-ordinal);
                copies[slot]=batch_copies(plan,ordinal,count);
                require(std::max(copies[slot].upload_bytes,copies[slot].download_bytes)<=actual_capacity,
                        FORT_SCOPE_EXECUTION,"batch staging capacity changed after execution started");
                for (const auto &copy : copies[slot].uploads) pack(copy,slot,false);
                for (const auto &copy : copies[slot].uploads) enqueue(copy,slot,true);
                fort_scope_batch_window window{FORT_SCOPE_BATCH_ABI_VERSION,ordinal,count,stream(slot),views.data(),views.size()};
                uint64_t launches=0; mark_pending(slot,true);
                const int status=worker(&window,user,&launches);
                c.stats.launches+=launches; out->actual_launches+=launches;
                require(status==FORT_SCOPE_OK,FORT_SCOPE_EXECUTION,"scoped batch numerical callback failed");
                require(launches==plan.workers,FORT_SCOPE_EXECUTION,"scoped batch callback launch sequence changed");
                for (const auto &copy : copies[slot].downloads) enqueue(copy,slot,false);
                record(slot); active[slot]=true;
                if (out->selected_transfers==FORT_SCOPE_TRANSFERS_PINNED) finish(slot);
                ordinal+=count; slot^=1;
            }
            finish(0); finish(1);
            for (auto &r : plan.resources) {
                r.original->initialized=std::move(r.final.initialized);
                r.original->host_current=std::move(r.final.host_current);
                r.original->device_current=std::move(r.final.device_current);
            }
#ifndef FORT_SCOPE_CPU_TEST
            size_t freed=0; cuda_check(c,fort_staging::release(std::move(lease.slots),freed));
#endif
            // Both slot events include the context readiness record. Completed
            // slot work must not manufacture an extra c.stream wait.
            c.pending=false; c.batch_active=false;
            c.transfer_stats.effective_mode=out->selected_transfers;
            c.transfer_stats.fallback_reason=FORT_SCOPE_TRANSFER_NONE;
            out->owner_cost_available=0; observation();
        } catch (...) {
            c.poisoned=true; c.owner_estimate_available=false; c.batch_active=false;
            c.batch_report=*out;
            throw Error(FORT_SCOPE_EXECUTION,"scoped batch execution failed; unsafe replay prohibited");
        }
    },Change::None);
}
static int register_buffer(fort_scope_t h, uint64_t identity, uint64_t generation,
                           const fort_scope_layout *layout, bool full_initialized,
                           const fort_scope_section *defined, size_t defined_count, fort_buffer_t *out) {
    return with(h, [&](Context &c) {
        require(layout && out && identity && generation && layout->rank && layout->extents && layout->lower_bounds &&
                layout->element_bytes && layout->type <= FORT_SCOPE_LOGICAL, FORT_SCOPE_ARGUMENT, "invalid buffer registration");
        const size_t widths[] = {layout->element_bytes, 4, 8, 4, 1};
        require(widths[layout->type] == layout->element_bytes, FORT_SCOPE_ARGUMENT, "element width does not match type");
        auto b = std::make_unique<Buffer>();
        b->identity = identity; b->generation = generation; b->host = layout->host;
        b->type = layout->type; b->element_bytes = layout->element_bytes;
        b->extents.assign(layout->extents, layout->extents+layout->rank);
        b->lower.assign(layout->lower_bounds, layout->lower_bounds+layout->rank);
        // Empty arrays never read storage, including overflowing other extents.
        bool is_empty = std::find(b->extents.begin(), b->extents.end(), 0) != b->extents.end();
        size_t stride = is_empty ? 0 : b->element_bytes;
        for (size_t extent : b->extents) { b->strides.push_back(stride); stride = multiply(stride, extent); }
        b->bytes = stride;
        require(!b->bytes || b->host, FORT_SCOPE_ARGUMENT, "nonempty buffer has no host storage");
        // Validate before the identity lookup too: a duplicate is not an excuse
        // to accept malformed coordinates. Existing freshness remains intact.
        const auto initial = sections(*b, defined, defined_count, full_initialized);
        const auto existing = c.identities.find(identity);
        if (existing != c.identities.end()) {
            auto &old = buffer(c, existing->second);
            require(old.generation == generation, FORT_SCOPE_STALE, "allocation generation changed; unregister before changing storage");
            require(old.host == b->host && old.extents == b->extents && old.lower == b->lower && old.type == b->type &&
                    old.element_bytes == b->element_bytes, FORT_SCOPE_ARGUMENT, "inconsistent registration of a logical buffer");
            *out = old.handle; return;
        }
        const auto first = reinterpret_cast<uintptr_t>(b->host);
        require(b->bytes <= std::numeric_limits<uintptr_t>::max()-first, FORT_SCOPE_ARGUMENT, "host address range overflow");
        for (const auto &entry : c.buffers) {
            const auto &old = *entry.second;
            const auto other = reinterpret_cast<uintptr_t>(old.host);
            require(!b->bytes || !old.bytes || first+b->bytes <= other || other+old.bytes <= first,
                    FORT_SCOPE_ALIAS, "overlapping registrations require one canonical buffer");
        }
        b->initialized = b->host_current = initial;
        require(c.registrations_incurred < std::numeric_limits<size_t>::max(),
                FORT_SCOPE_RESOURCE, "scope registration accounting overflow");
        b->handle = token();
        const auto handle = b->handle;
        c.buffers.emplace(handle, std::move(b));
        try { c.identities.emplace(identity, handle); }
        catch (...) { c.buffers.erase(handle); throw; }
        ++c.registrations_incurred;
        *out = handle;
    });
}
extern "C" int fort_scope_register(fort_scope_t h, uint64_t identity, uint64_t generation,
                                   const fort_scope_layout *layout, int initialized, fort_buffer_t *out) {
    if (initialized != 0 && initialized != 1)
        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "host initialization flag must be zero or one");
    return register_buffer(h, identity, generation, layout, initialized != 0, nullptr, 0, out);
}
extern "C" int fort_scope_register_sections(fort_scope_t h, uint64_t identity, uint64_t generation,
                                            const fort_scope_layout *layout, const fort_scope_section *defined,
                                            size_t count, fort_buffer_t *out) {
    return register_buffer(h, identity, generation, layout, false, defined, count, out);
}
extern "C" int fort_scope_forget_definition(fort_scope_t h, fort_buffer_t handle) {
    return with(h, [&](Context &c) {
        auto &b = buffer(c, handle);
        require(!b.prepared, FORT_SCOPE_STATE, "definition change during a prepared buffer access");
        wait(c);
        b.initialized.clear(); b.host_current.clear(); b.device_current.clear();
        trace("forget_definition", &b);
    });
}
extern "C" int fort_scope_forget_sections_v1(fort_scope_t h, fort_buffer_t handle,
                                             const fort_scope_section *items, size_t count) {
    return with(h, [&](Context &c) {
        auto &b = buffer(c, handle);
        require(!b.prepared, FORT_SCOPE_STATE, "definition change during a prepared buffer access");
        const auto removed = sections(b, items, count, false);
        Budget budget;
        // Do not partially invalidate state if any bounded subtraction fails.
        auto initialized = difference(b.initialized, removed, budget);
        auto host = difference(b.host_current, removed, budget);
        auto device = difference(b.device_current, removed, budget);
        wait(c);
        b.initialized = std::move(initialized);
        b.host_current = std::move(host);
        b.device_current = std::move(device);
        trace("forget_sections", &b);
    });
}
extern "C" int fort_scope_view_get_v1(fort_scope_t h, const fort_scope_view_v1 *view,
                                       fort_scope_view_layout_v1 *out) {
    return with(h, [&](Context &c) {
        require(view && out && view->version == FORT_SCOPE_VIEW_ABI_VERSION && view->rank &&
                view->origins && view->extents && view->lower_bounds,
                FORT_SCOPE_ARGUMENT, "invalid borrowed view descriptor");
        auto &b = buffer(c, view->buffer);
        require(view->generation == b.generation, FORT_SCOPE_STALE, "borrowed view allocation generation changed");
        require(view->rank == b.extents.size(), FORT_SCOPE_ARGUMENT, "borrowed view rank differs from root");
        const bool is_empty = std::find(view->extents, view->extents+view->rank, size_t(0)) != view->extents+view->rank;
        size_t elements = is_empty ? 0 : 1, offset = 0;
        for (size_t k=0; k<view->rank; ++k) {
            require(view->origins[k] <= b.extents[k] && view->extents[k] <= b.extents[k]-view->origins[k],
                    FORT_SCOPE_ARGUMENT, "borrowed view exceeds canonical root");
            const int64_t lower = view->lower_bounds[k];
            require(lower >= INT32_MIN && lower <= INT32_MAX && view->extents[k] <= INT32_MAX,
                    FORT_SCOPE_BOUNDARY, "borrowed view bounds exceed INTEGER ABI");
            if (view->extents[k])
                require(int64_t(view->extents[k]-1) <= INT32_MAX-lower,
                        FORT_SCOPE_BOUNDARY, "borrowed view upper bound exceeds INTEGER ABI");
            // Empty arrays have no address, even if another extent overflows.
            if (!is_empty) {
                elements = multiply(elements, view->extents[k]);
                const size_t part = multiply(view->origins[k], b.strides[k]);
                require(part <= std::numeric_limits<size_t>::max()-offset,
                        FORT_SCOPE_ARGUMENT, "borrowed view byte offset overflow");
                offset += part;
            }
        }
        *out = {{static_cast<uint32_t>(b.extents.size()), b.type, b.element_bytes, b.host,
                 b.extents.data(), b.lower.data(), b.generation}, view->origins, view->extents,
                b.strides.data(), view->lower_bounds, elements, offset};
    }, Change::None);
}
extern "C" int fort_scope_layout_get(fort_scope_t h, fort_buffer_t b, fort_scope_layout *out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing layout output");
        auto &a = buffer(c,b);
        *out = {static_cast<uint32_t>(a.extents.size()), a.type, a.element_bytes, a.host, a.extents.data(), a.lower.data(), a.generation};
    }, Change::None);
}
extern "C" int fort_scope_host_begin(fort_scope_t h, fort_buffer_t b, const fort_scope_access *a) {
    return with(h, [&](Context &c) {
        note_numerical_activity(c);
        begin(c, buffer(c,b), a, false);
    });
}
extern "C" int fort_scope_host_end(fort_scope_t h, fort_buffer_t b) {
    return with(h, [&](Context &c) { end(buffer(c,b), false); });
}
extern "C" int fort_scope_device_begin(fort_scope_t h, fort_buffer_t b, const fort_scope_access *a, void **out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing device pointer output");
        note_numerical_activity(c);
        auto &value = buffer(c,b); begin(c, value, a, true); *out = value.device;
    });
}
extern "C" int fort_scope_device_end(fort_scope_t h, fort_buffer_t b) {
    return with(h, [&](Context &c) { end(buffer(c,b), true); c.pending = c.pending || c.ready; });
}
extern "C" int fort_scope_cancel_access(fort_scope_t h, fort_buffer_t b) {
    return with(h, [&](Context &c) { buffer(c,b).prepared.reset(); });
}
extern "C" int fort_scope_gpu_enter(fort_scope_t h, int *previous, void **stream) {
    return with(h, [&](Context &c) {
        require(previous && stream, FORT_SCOPE_ARGUMENT, "missing GPU entry outputs");
        note_numerical_activity(c);
        initialize(c);
#ifdef FORT_SCOPE_CPU_TEST
        *previous = 0; *stream = nullptr;
#else
        cuda_check(c, cudaGetDevice(previous));
        cuda_check(c, cudaSetDevice(c.device));
        *stream = reinterpret_cast<void *>(c.stream);
#endif
    });
}
extern "C" int fort_scope_gpu_leave(fort_scope_t h, int previous) {
    // Restoration is required even when a numerical error poisoned the scope.
    return protect([&]() {
        const auto owner = lookup(h);
        std::lock_guard<std::mutex> lock(owner->mutex);
        require(!owner->closed, FORT_SCOPE_STALE, "invalid or stale scope handle");
        auto &c = *owner;
        invalidate(c, Change::State);
#ifndef FORT_SCOPE_CPU_TEST
        cuda_check(c, cudaSetDevice(previous));
#else
        (void)c; (void)previous;
#endif
    });
}
extern "C" int fort_scope_note_launch(fort_scope_t h) {
    return with(h, [&](Context &c) {
        require(c.ready, FORT_SCOPE_STATE, "GPU launch recorded without entering the context device");
        note_numerical_activity(c);
        ++c.stats.launches; c.pending = true; trace("launch");
    });
}
extern "C" int fort_scope_execution_error(fort_scope_t h, const char *message) {
    return protect([&]() {
        const auto c = lookup(h);
        std::lock_guard<std::mutex> lock(c->mutex);
        require(!c->closed, FORT_SCOPE_STALE, "invalid or stale scope handle");
        const char *reason = c->poisoned && last_error[0] ? last_error : message;
        invalidate(*c, Change::State);
        c->owner_estimate_available = false;
        c->poisoned = true;
        throw Error(FORT_SCOPE_EXECUTION, reason ? reason : "scoped numerical execution failed");
    });
}
extern "C" int fort_scope_report_error(int status, const char *message) {
    if (status != FORT_SCOPE_OK) diagnostic(message ? message : "scoped entry error");
    return status;
}
extern "C" int fort_scope_wait(fort_scope_t h) {
    return with(h, [&](Context &c) { wait(c); complete_segment_estimate(c); });
}
extern "C" int fort_scope_stats_get(fort_scope_t h, fort_scope_stats *out) {
    return with(h, [&](Context &c) { require(out, FORT_SCOPE_ARGUMENT, "missing stats output"); *out = c.stats; }, Change::None);
}
extern "C" int fort_scope_unregister(fort_scope_t h, fort_buffer_t handle) {
    return with(h, [&](Context &c) {
        auto &b = buffer(c,handle); publish(c,b); release(c,b);
        c.identities.erase(b.identity); c.buffers.erase(handle);
    });
}
extern "C" int fort_scope_close(fort_scope_t h) {
    const auto status = with(h, [&](Context &c) {
        require(!c.plan_installed || c.worker_cursor == c.schedule.size(), FORT_SCOPE_STATE,
                "scope closed before its scheduled workers completed");
        for (auto &entry : c.buffers) publish(c, *entry.second);
        for (auto &entry : c.buffers) release(c, *entry.second);
        wait(c);
#ifndef FORT_SCOPE_CPU_TEST
        if (c.ready) {
            DeviceGuard guard(c);
            if (c.pool) cuda_check(c, cudaMemPoolDestroy(c.pool));
            cuda_check(c, cudaStreamDestroy(c.stream));
        }
#endif
        c.buffers.clear(); c.identities.clear(); c.closed = true;
        transfer_statistics_trace(c, h);
    });
    if (status == FORT_SCOPE_OK) {
        std::lock_guard<std::mutex> lock(registry_mutex); contexts.erase(h);
    }
    return status;
}
extern "C" int fort_scope_abandon(fort_scope_t h) {
    return protect([&]() {
        const auto c = lookup(h);
        std::lock_guard<std::mutex> lock(c->mutex);
        require(!c->closed, FORT_SCOPE_STALE, "invalid or stale scope handle");
        require(c->poisoned, FORT_SCOPE_STATE, "only a failed scope can be abandoned");
#ifdef FORT_SCOPE_CPU_TEST
        for (auto &entry : c->buffers) std::free(entry.second->device);
#else
        // A CUDA execution failure invalidates results. Cleanup is best effort;
        // CUDA can reject further operations after a fatal device fault.
        int previous = 0;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        if (cudaSetDevice(c->device) == cudaSuccess && c->ready) {
            (void)cudaStreamSynchronize(c->stream);
            for (auto &entry : c->buffers) {
                if (entry.second->device) (void)cudaFree(entry.second->device);
            }
            if (c->pool) (void)cudaMemPoolDestroy(c->pool);
            (void)cudaStreamDestroy(c->stream);
        }
        if (restore) (void)cudaSetDevice(previous);
#endif
        c->buffers.clear(); c->identities.clear(); c->closed = true;
        std::lock_guard<std::mutex> registry_lock(registry_mutex); contexts.erase(h);
    });
}
