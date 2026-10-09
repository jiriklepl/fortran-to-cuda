/* Shared state implementation: compile once with nvcc, or with
 * -DFORT_SCOPE_CPU_TEST for the isolated coherence reference backend. */
#include "scoped_runtime.h"
#include "section_copy.hpp"
#include "scoped_regions.hpp"
#include "scoped_planning.hpp"
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
    explicit Context(int ordinal) : device(ordinal) {}
    int device;
    bool ready = false, pending = false, poisoned = false, closed = false;
    size_t device_budget = std::numeric_limits<size_t>::max();
    std::mutex mutex;
    std::unordered_map<fort_buffer_t, std::unique_ptr<Buffer>> buffers;
    std::unordered_map<uint64_t, fort_buffer_t> identities;
    fort_scope_stats stats{};
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
void copy(Context &c, Buffer &b, const Box &box, bool upload) {
    const fort_physical::CopyPlan plan(b.element_bytes, b.extents, box.lo, box.hi);
    require(plan.valid, FORT_SCOPE_ARGUMENT, "invalid physical transfer layout");
    if (!plan.bytes) return;
    allocate(c, b);
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
        timing.cache_hit = c.preview && c.preview_query_generation == c.query_generation &&
            c.preview_state_generation == c.state_generation &&
            c.preview->driver_initialized == driver_initialized(c) &&
            !std::memcmp(c.preview_costs.data(), costs, sizeof(*costs));
        auto result = timing.cache_hit
            ? *c.preview : fort_scoped::planning::select(planning_inputs(c, h), *costs);
        if (compatible == -1) {
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
