/* Shared state implementation: compile once with nvcc, or with
 * -DFORT_SCOPE_CPU_TEST for the isolated coherence reference backend. */
#include "scoped_runtime.h"
#include "section_copy.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#ifndef FORT_SCOPE_CPU_TEST
#include <cuda_runtime.h>
#endif

namespace {
constexpr size_t rectangle_limit = 32, intersection_limit = 1024;
struct Error : std::runtime_error {
    int status;
    Error(int status_, const char *message) : std::runtime_error(message), status(status_) {}
};
struct Fragmented {};
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
struct Box { std::vector<size_t> lo, hi; };
using Region = std::vector<Box>;
struct Budget {
    size_t checks = 0;
    void check() { if (++checks > intersection_limit) throw Fragmented{}; }
};
bool empty(const Box &b) {
    for (size_t k=0; k<b.lo.size(); ++k) if (b.lo[k] == b.hi[k]) return true;
    return false;
}
bool contains(const Box &a, const Box &b) {
    for (size_t k=0; k<a.lo.size(); ++k) if (a.lo[k] > b.lo[k] || a.hi[k] < b.hi[k]) return false;
    return true;
}
std::optional<Box> intersection(const Box &a, const Box &b, Budget &budget) {
    budget.check();
    Box result = a;
    for (size_t k=0; k<a.lo.size(); ++k) {
        result.lo[k] = std::max(a.lo[k], b.lo[k]);
        result.hi[k] = std::min(a.hi[k], b.hi[k]);
        if (result.lo[k] >= result.hi[k]) return {};
    }
    return result;
}
void append(Region &region, Box box) {
    if (empty(box)) return;
    if (region.size() == rectangle_limit) throw Fragmented{};
    region.push_back(std::move(box));
}
Region subtract(const Box &box, const Box &cut, Budget &budget) {
    auto common = intersection(box, cut, budget);
    if (!common) return {box};
    Region result;
    Box middle = box;
    for (size_t k=0; k<box.lo.size(); ++k) {
        if (middle.lo[k] < common->lo[k]) {
            Box part = middle; part.hi[k] = common->lo[k]; append(result, std::move(part));
            middle.lo[k] = common->lo[k];
        }
        if (common->hi[k] < middle.hi[k]) {
            Box part = middle; part.lo[k] = common->hi[k]; append(result, std::move(part));
            middle.hi[k] = common->hi[k];
        }
    }
    return result;
}
Region difference(const Region &a, const Region &b, Budget &budget) {
    Region result;
    for (const auto &box : a) {
        Region pending{box};
        for (const auto &cut : b) {
            Region next;
            for (const auto &part : pending)
                for (auto &piece : subtract(part, cut, budget)) append(next, std::move(piece));
            pending = std::move(next);
            if (pending.empty()) break;
        }
        for (auto &part : pending) append(result, std::move(part));
    }
    return result;
}
Region unite(Region a, const Region &b, Budget &budget) {
    for (const auto &box : b) {
        if (std::any_of(a.begin(), a.end(), [&](const Box &old) { return contains(old, box); })) continue;
        a.erase(std::remove_if(a.begin(), a.end(), [&](const Box &old) { return contains(box, old); }), a.end());
        for (auto &piece : difference({box}, a, budget)) append(a, std::move(piece));
    }
    return a;
}
struct Effects { Region reads, writes, overwrites; };
struct Prepared { bool device; Region initialized, current, opposite; };
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
    std::mutex mutex;
    std::unordered_map<fort_buffer_t, std::unique_ptr<Buffer>> buffers;
    std::unordered_map<uint64_t, fort_buffer_t> identities;
    fort_scope_stats stats{};
#ifndef FORT_SCOPE_CPU_TEST
    cudaStream_t stream = nullptr;
    cudaMemPool_t pool = nullptr;
#endif
};
std::mutex registry_mutex;
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
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
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
    trace("initialize");
}
void allocate(Context &c, Buffer &b) {
    if (b.device || !b.bytes) return;
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
Prepared prepare(Buffer &b, const Effects &e, bool device) {
    Budget budget;
    auto initialized = unite(b.initialized, e.writes, budget);
    auto current = unite(device ? b.device_current : b.host_current, e.writes, budget);
    auto opposite = difference(device ? b.host_current : b.device_current, e.writes, budget);
    return {device, std::move(initialized), std::move(current), std::move(opposite)};
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
template<class F> int with(fort_scope_t handle, F &&f) noexcept {
    return protect([&]() {
        const auto c = lookup(handle);
        std::lock_guard<std::mutex> lock(c->mutex);
        require(!c->closed, FORT_SCOPE_STALE, "invalid or stale scope handle");
        require(!c->poisoned, FORT_SCOPE_EXECUTION, "scope has an execution failure; unsafe replay prohibited");
        f(*c);
    });
}
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
extern "C" int fort_scope_register(fort_scope_t h, uint64_t identity, uint64_t generation,
                                   const fort_scope_layout *layout, int initialized, fort_buffer_t *out) {
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
        if (initialized && !empty(b->full())) b->initialized = b->host_current = {b->full()};
        b->handle = token();
        const auto handle = b->handle;
        c.buffers.emplace(handle, std::move(b));
        try { c.identities.emplace(identity, handle); }
        catch (...) { c.buffers.erase(handle); throw; }
        *out = handle;
    });
}
extern "C" int fort_scope_layout_get(fort_scope_t h, fort_buffer_t b, fort_scope_layout *out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing layout output");
        auto &a = buffer(c,b);
        *out = {static_cast<uint32_t>(a.extents.size()), a.type, a.element_bytes, a.host, a.extents.data(), a.lower.data(), a.generation};
    });
}
extern "C" int fort_scope_host_begin(fort_scope_t h, fort_buffer_t b, const fort_scope_access *a) {
    return with(h, [&](Context &c) { begin(c, buffer(c,b), a, false); });
}
extern "C" int fort_scope_host_end(fort_scope_t h, fort_buffer_t b) {
    return with(h, [&](Context &c) { end(buffer(c,b), false); });
}
extern "C" int fort_scope_device_begin(fort_scope_t h, fort_buffer_t b, const fort_scope_access *a, void **out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing device pointer output");
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
        ++c.stats.launches; c.pending = true; trace("launch");
    });
}
extern "C" int fort_scope_execution_error(fort_scope_t h, const char *message) {
    return protect([&]() {
        const auto c = lookup(h);
        std::lock_guard<std::mutex> lock(c->mutex);
        require(!c->closed, FORT_SCOPE_STALE, "invalid or stale scope handle");
        const char *reason = c->poisoned && last_error[0] ? last_error : message;
        c->poisoned = true;
        throw Error(FORT_SCOPE_EXECUTION, reason ? reason : "scoped numerical execution failed");
    });
}
extern "C" int fort_scope_report_error(int status, const char *message) {
    if (status != FORT_SCOPE_OK) diagnostic(message ? message : "scoped entry error");
    return status;
}
extern "C" int fort_scope_wait(fort_scope_t h) { return with(h, [&](Context &c) { wait(c); }); }
extern "C" int fort_scope_stats_get(fort_scope_t h, fort_scope_stats *out) {
    return with(h, [&](Context &c) { require(out, FORT_SCOPE_ARGUMENT, "missing stats output"); *out = c.stats; });
}
extern "C" int fort_scope_unregister(fort_scope_t h, fort_buffer_t handle) {
    return with(h, [&](Context &c) {
        auto &b = buffer(c,handle); publish(c,b); release(c,b);
        c.identities.erase(b.identity); c.buffers.erase(handle);
    });
}
extern "C" int fort_scope_close(fort_scope_t h) {
    const auto status = with(h, [&](Context &c) {
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
