/* Shared state implementation: compile once with nvcc, or with
 * -DFORT_SCOPE_CPU_TEST for the isolated coherence reference backend. */
#include "scoped_runtime.h"
#include "section_copy.hpp"
#include "section_batch_cost.hpp"
#include "scoped_regions.hpp"
#include "scoped_graphs.hpp"
#include "floating_environment.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
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
#include "scoped_cuda_resources.hpp"
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
struct Completion {
    bool complete = false;
#ifndef FORT_SCOPE_CPU_TEST
    cudaEvent_t event = nullptr;
#endif
};
struct UploadReader {
    Region sections;
    std::shared_ptr<Completion> completion;
};
struct Buffer {
    fort_buffer_t handle;
    uint64_t identity, generation;
    uint32_t type;
    size_t element_bytes, bytes;
    void *host, *device = nullptr;
    uint64_t last_device_use = 0;
    std::vector<size_t> extents, strides;
    std::vector<int64_t> lower;
    Region initialized, host_current, device_current;
    Region unfenced_uploads;
    std::vector<UploadReader> upload_readers;
    std::optional<Prepared> prepared;
    Box full() const { return {Coordinates(extents.size()), Coordinates(extents)}; }
};
struct QueuedBinding {
    fort_buffer_t buffer;
    std::shared_ptr<const Effects> effects;
};
struct BoundEffects { fort_buffer_t buffer; Effects effects; };
struct QueuedTask {
    uint64_t key;
    fort_scope_kernel_v1 kernel;
    // Numerical work may be submitted by a later native access. Retain its
    // original diagnostic position independently of that flush boundary.
    std::unique_ptr<fort_scope_trace_state_v1> provenance;
    fort_scoped::graphs::ArgumentValues values;
    std::vector<void *> arguments;
    std::vector<QueuedBinding> bindings;
    // One bit per buffer handle, modulo 64: tasks with disjoint sets share no buffer.
    uint64_t buffers = 0;
    fort_scope_kernel_v1 launch() {
        kernel.arguments=arguments.data();
        return kernel;
    }
};
// The device threads a launch starts: what the queue counts as its work.
inline uint64_t launch_work(const fort_scope_kernel_v1 &kernel) {
    uint64_t threads = 1;
    for (const uint32_t extent : {kernel.grid[0],kernel.grid[1],kernel.grid[2],
                                  kernel.block[0],kernel.block[1],kernel.block[2]}) {
        if (extent && threads > std::numeric_limits<uint64_t>::max()/extent) return std::numeric_limits<uint64_t>::max();
        threads *= extent;
    }
    return threads;
}
struct KernelTracePositions {
    const fort_scope_trace_state_v1 *const *values=nullptr;
    const fort_scope_trace_state_v1 *fallback=nullptr;
    const fort_scope_trace_state_v1 *at(size_t index) const noexcept {
        return values ? values[index] : fallback;
    }
};
bool trace_allocation_allowed(const char *stage) noexcept {
#ifdef FORT_SCOPE_TEST_FAULTS
    const auto *failure=std::getenv("FORT_SCOPE_TEST_FAIL_TRACE_ALLOC");
    return !failure || std::strcmp(failure,stage);
#else
    (void)stage;
    return true;
#endif
}
// Physical values and their exact completion dependencies have one owner.
// An execution context borrows this state.
struct Residency {
    explicit Residency(int ordinal) : device(ordinal) {
        transfer_stats.version = FORT_SCOPE_TRANSFER_ABI_VERSION;
        scratch_stats.version = FORT_SCOPE_SCRATCH_ABI_VERSION;
        completion_stats.version = FORT_SCOPE_COMPLETION_ABI_VERSION;
        residency_stats.version = FORT_SCOPE_RESIDENCY_ABI_VERSION;
    }
    int device;
    uint64_t domain = 0, active_context = 0;
    uint64_t last_context = 0;
    fort_scope_trace_state_v1 last_provenance{};
    bool ready = false, pending = false, poisoned = false, retired = false;
    bool copy_completion_priced = false, budget_warned = false;
    size_t device_budget = std::numeric_limits<size_t>::max();
    std::mutex mutex;
    std::unordered_map<fort_buffer_t, std::unique_ptr<Buffer>> buffers;
    std::unordered_map<uint64_t, fort_buffer_t> identities;
    fort_scope_stats stats{};
    void *scratch = nullptr;
    size_t scratch_capacity = 0;
    uint64_t scratch_token = 0;
    fort_scope_scratch_stats_v1 scratch_stats{};
    fort_scope_transfer_stats transfer_stats{};
    // Physical transport remains necessary after the last execution detaches.
    std::optional<fort_physical::compact::Pricing> copy_costs;
    fort_scope_completion_stats_v1 completion_stats{};
    fort_scope_residency_stats_v1 residency_stats{};
    uint64_t residency_clock = 0;
    std::vector<Buffer *> unfenced_upload_buffers;
    std::deque<std::shared_ptr<Completion>> completions;
    std::weak_ptr<Completion> tail_completion;
    std::unordered_map<uint64_t, size_t> workspaces;
    size_t workspace_bytes = 0;
#ifdef FORT_SCOPE_CPU_TEST
    size_t compact_reference_capacity = 0;
    fort_physical::compact::startup::State compact_reference_startup;
#endif
#ifndef FORT_SCOPE_CPU_TEST
    std::vector<cudaEvent_t> completion_events, available_completion_events;
    cudaStream_t stream = nullptr;
    cudaMemPool_t pool = nullptr;
#endif
};
struct Context {
    explicit Context(int ordinal) : Context(std::make_shared<Residency>(ordinal), false) {}
    explicit Context(std::shared_ptr<Residency> owner, bool borrowed)
        : residency(std::move(owner)), borrowed_domain(borrowed), device(residency->device),
          ready(residency->ready), pending(residency->pending), poisoned(residency->poisoned),
          device_budget(residency->device_budget), mutex(residency->mutex), buffers(residency->buffers),
          identities(residency->identities), stats(residency->stats), scratch(residency->scratch),
          scratch_capacity(residency->scratch_capacity), scratch_token(residency->scratch_token),
          scratch_stats(residency->scratch_stats), transfer_stats(residency->transfer_stats),
          completion_stats(residency->completion_stats), residency_stats(residency->residency_stats),
          residency_clock(residency->residency_clock), unfenced_upload_buffers(residency->unfenced_upload_buffers),
          completions(residency->completions), tail_completion(residency->tail_completion),
          copy_costs(residency->copy_costs)
#ifndef FORT_SCOPE_CPU_TEST
          , completion_events(residency->completion_events), available_completion_events(residency->available_completion_events),
          stream(residency->stream), pool(residency->pool)
#endif
    {
        graph_stats.version = FORT_SCOPE_GRAPH_ABI_VERSION;
        queue_stats.version = FORT_SCOPE_GRAPH_ABI_VERSION;
        const auto *queue = std::getenv("FORT_SCOPE_GRAPH_QUEUE"), *graphs = std::getenv("FORT_SCOPE_CUDA_GRAPHS");
        queue_stats.queue_enabled = !queue || std::strcmp(queue,"0");
        queue_stats.graphs_enabled = !graphs || std::strcmp(graphs,"0");
        if (const auto *work = std::getenv("FORT_SCOPE_GRAPH_QUEUE_WORK")) {
            char *end = nullptr;
            const auto value = std::strtoull(work,&end,10);
            if (end != work && !*end) queue_work_limit = value;
        }
    }
    std::shared_ptr<Residency> residency;
    bool borrowed_domain = false;
    int device;
    fort_scope_t handle = 0;
    fort_scope_trace_state_v1 provenance{};
    bool &ready, &pending, &poisoned;
    bool closed = false;
    size_t &device_budget;
    std::mutex &mutex;
    std::unordered_map<fort_buffer_t, std::unique_ptr<Buffer>> &buffers;
    std::unordered_map<uint64_t, fort_buffer_t> &identities;
    fort_scope_stats &stats;
    void *&scratch;
    size_t &scratch_capacity;
    uint64_t &scratch_token;
    fort_scope_scratch_stats_v1 &scratch_stats;
    fort_scope_transfer_stats &transfer_stats;
    fort_scope_graph_stats_v1 graph_stats{};
    fort_scope_completion_stats_v1 &completion_stats;
    fort_scope_graph_queue_stats_v1 queue_stats{};
    fort_scope_residency_stats_v1 &residency_stats;
    uint64_t &residency_clock;
    std::vector<std::unique_ptr<QueuedTask>> commands;
    // Device threads of the queued launches, and how many the queue holds back at most. A queue saves
    // host time per launch, but the device waits for what is held back: once that is more than a
    // moment of device work it is submitted. Zero holds back any amount.
    uint64_t queued_work = 0, queue_work_limit = uint64_t{1} << 18;
    // The state each buffer will have once the queued commands have run. An
    // entry is rebuilt on demand, from the commands that touch its buffer.
    struct VirtualBuffer { uint64_t epoch; Buffer state; };
    // Section lists repeat from one call of an entry to the next. Parsing
    // normalizes them with the region algebra, so each distinct list of each
    // array shape is parsed once.
    struct ParsedAccess { std::vector<size_t> key; std::shared_ptr<const Effects> effects; };
    std::unordered_map<uint64_t,std::vector<ParsedAccess>> parsed_accesses;
    size_t parsed_access_count = 0;
    std::vector<size_t> parsed_access_key;
    std::unordered_map<fort_buffer_t,VirtualBuffer> virtual_buffers;
    uint64_t virtual_epoch = 1, virtual_checked = 0;
    std::vector<Buffer *> &unfenced_upload_buffers;
    std::deque<std::shared_ptr<Completion>> &completions;
    std::weak_ptr<Completion> &tail_completion;
    std::optional<fort_physical::compact::Pricing> &copy_costs;
#ifndef FORT_SCOPE_CPU_TEST
    fort_scoped::graphs::Cache graphs;
    std::vector<cudaEvent_t> &completion_events, &available_completion_events;
    cudaStream_t &stream;
    cudaMemPool_t &pool;
#endif
};
enum class Change { None, State };
void invalidate(Context &c, Change change) noexcept;
void drain_accesses(Context &c,const fort_scope_plan_binding *bindings,size_t count);
void drain_resources(Context &c,const fort_buffer_t *buffers,size_t count);
void flush_commands(Context &c,const std::vector<bool> *selected=nullptr,bool conflict=false);
void reserve_devices(Context &c,const fort_buffer_t *handles,size_t count,void **devices=nullptr);
void touch_device(Context &c,Buffer &b) noexcept {
    if (c.residency_clock==std::numeric_limits<uint64_t>::max()) {
        for (auto &entry : c.buffers) entry.second->last_device_use=0;
        c.residency_clock=0;
    }
    b.last_device_use=++c.residency_clock;
}
std::mutex driver_mutex;
std::unordered_set<int> initialized_devices;
std::mutex registry_mutex;
std::mutex trace_mutex;
std::unordered_map<fort_scope_t, std::shared_ptr<Context>> contexts;
std::unordered_map<fort_scope_domain_t, std::shared_ptr<Residency>> domains;
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
std::shared_ptr<Residency> lookup_domain(fort_scope_domain_t handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto found = domains.find(handle);
    require(found != domains.end(), FORT_SCOPE_STALE, "invalid or stale residency domain");
    return found->second;
}
Buffer &buffer(Context &c, fort_buffer_t handle) {
    auto found = c.buffers.find(handle);
    require(found != c.buffers.end(), FORT_SCOPE_STALE, "invalid, foreign, or stale buffer handle");
    return *found->second;
}
uint64_t buffer_bit(fort_buffer_t handle) noexcept {
    return uint64_t(1) << (handle%64);
}
bool trace_enabled() noexcept {
    const char *enabled = std::getenv("FORT_RUNTIME_TRACE");
    return enabled && !std::strcmp(enabled, "1");
}
void trace_id(char (&destination)[65], const char *source) noexcept {
    if (!source) { destination[0] = '\0'; return; }
    bool valid = true;
    size_t length = 0;
    while (length < 65 && source[length]) {
        const char c = source[length++];
        valid = valid && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    }
    destination[0] = '\0';
    if (valid && length == 64) {
        std::memcpy(destination, source, 64);
        destination[64] = '\0';
    }
}
struct IdentitySwapTrace {
    uint64_t first_identity,second_identity,first_handle,second_handle,first_generation,second_generation;
};
void trace(const Context &c, const char *operation, const Buffer *b = nullptr, size_t bytes = 0,
           const IdentitySwapTrace *exchange=nullptr, const fort_scope_trace_state_v1 *position=nullptr) noexcept {
    if (!trace_enabled()) return;
    try {
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << "FORT_SCOPED " << operation;
        if (b) std::cerr << " buffer=" << b->identity << " generation=" << b->generation;
        std::cerr << " bytes=" << bytes << " provenance_version=1 context=" << c.handle;
        if(c.residency->domain)
            std::cerr << " domain_version=1 domain=" << c.residency->domain
                      << " domain_role=" << (c.handle==c.residency->domain ? "storage" : "execution");
        if (b) std::cerr << " buffer_handle=" << b->handle;
        const auto &p = position ? *position : c.provenance;
        for (const auto &[name, value] : {
            std::pair{"owner", p.owner}, {"procedure", p.procedure}, {"segment", p.segment},
            {"operation", p.operation}, {"implementation", p.implementation}, {"boundary", p.boundary}})
            std::cerr << ' ' << name << '=' << (value[0] ? value : "unknown");
        if(exchange)
            std::cerr << " swap_version=1 first_identity=" << exchange->first_identity
                      << " second_identity=" << exchange->second_identity
                      << " first_before_handle=" << exchange->first_handle
                      << " second_before_handle=" << exchange->second_handle
                      << " first_after_handle=" << exchange->second_handle
                      << " second_after_handle=" << exchange->first_handle
                      << " first_before_generation=" << exchange->first_generation
                      << " second_before_generation=" << exchange->second_generation
                      << " first_after_generation=" << exchange->second_generation
                      << " second_after_generation=" << exchange->first_generation;
        std::cerr << '\n';
    } catch (...) {} // Reporting must never change numerical success.
}
// Domain metadata links physical ownership to context/source provenance. It
// never supplies definition or source-lifetime authority.
void domain_trace(const Residency &owner,const char *event,const Context *context=nullptr) noexcept {
    if(!trace_enabled()) return;
    try {
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << "FORT_SCOPED domain {\"schema_version\":1,\"event\":\"" << event
                  << "\",\"domain\":" << owner.domain
                  << ",\"context\":" << (context ? context->handle : 0)
                  << ",\"physical_context\":" << owner.domain
                  << ",\"last_context\":" << owner.last_context << ",\"device\":" << owner.device
                  << ",\"failed\":" << (owner.poisoned ? "true" : "false");
        const auto &p=context ? context->provenance : owner.last_provenance;
        for(const auto &[name,value] : {
            std::pair{"owner",p.owner},{"procedure",p.procedure},{"segment",p.segment},
            {"operation",p.operation},{"implementation",p.implementation},{"boundary",p.boundary}})
            std::cerr << ",\"" << name << "\":\"" << (value[0] ? value : "unknown") << '"';
        std::cerr << "}\n";
    } catch(...) {} // Optional attribution cannot change runtime status.
}
struct DomainJson { const Context &context; };
std::ostream &operator<<(std::ostream &out,DomainJson attribution) {
    const auto &c=attribution.context;
    if(c.residency->domain)
        out << ",\"domain_version\":1,\"domain\":" << c.residency->domain
            << ",\"domain_role\":\"" << (c.handle==c.residency->domain ? "storage" : "execution") << '"';
    return out;
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
    fort_scoped::cuda_resources::Resources resources;
    const bool reused = fort_scoped::cuda_resources::acquire(c.device, resources);
    const auto status = reused ? cudaSuccess : fort_scoped::cuda_resources::create(c.device, resources);
    c.stream = resources.stream;
    c.pool = resources.pool;
    c.ready = c.stream != nullptr; // Own partial initialization even on failure.
    cuda_check(c, status, true);
    if (reused) trace(c, "initialize_reuse", nullptr, resources.retained_bytes);
#endif
    c.ready = true;
    { std::lock_guard<std::mutex> lock(driver_mutex); initialized_devices.insert(c.device); }
    trace(c, "initialize");
}
void mark_context_pending(Context &c) noexcept {
    c.residency->copy_completion_priced=false;
    c.pending = true;
    c.tail_completion.reset();
}
void complete_through(Context &c, std::shared_ptr<Completion> target) noexcept {
    if (c.tail_completion.lock() == target) {
        c.pending = false;c.residency->copy_completion_priced=false;c.tail_completion.reset();
    }
    while (!c.completions.empty()) {
        auto completion = c.completions.front();
        c.completions.pop_front();
        completion->complete = true;
#ifndef FORT_SCOPE_CPU_TEST
        if (completion->event) {
            // Capacity is reserved before event creation, so completion never
            // allocates while reclaiming a borrowed-memory dependency.
            c.available_completion_events.push_back(completion->event);
            completion->event = nullptr;
        }
#endif
        if (completion == target) break;
    }
}
void complete_all(Context &c) noexcept {
    c.residency->copy_completion_priced=false;
    if (!c.completions.empty()) complete_through(c,c.completions.back());
    for (auto *b : c.unfenced_upload_buffers) b->unfenced_uploads.clear();
    c.unfenced_upload_buffers.clear();
    c.tail_completion.reset();
}
bool completion_ready(Context &c, std::shared_ptr<Completion> completion) {
    if (completion->complete) return true;
#ifdef FORT_SCOPE_CPU_TEST
    (void)c;
    return false; // Deterministic reference: completion is explicit.
#else
    DeviceGuard guard(c);
    ++c.completion_stats.event_queries;
    const auto status = cudaEventQuery(completion->event);
    if (status == cudaErrorNotReady) return false;
    cuda_check(c,status);
    complete_through(c,completion);
    return true;
#endif
}
std::shared_ptr<Completion> record_completion(Context &c) {
    auto completion = std::make_shared<Completion>();
    c.completions.push_back(completion);
    try {
#ifndef FORT_SCOPE_CPU_TEST
        DeviceGuard guard(c);
        if (c.available_completion_events.empty()) {
            c.completion_events.reserve(c.completion_events.size()+1);
            c.available_completion_events.reserve(c.completion_events.size()+1);
            cuda_check(c,cudaEventCreateWithFlags(&completion->event,cudaEventDisableTiming),true);
            c.completion_events.push_back(completion->event);
            ++c.completion_stats.created_events;
        } else {
            completion->event = c.available_completion_events.back();
            c.available_completion_events.pop_back();
            ++c.completion_stats.reused_events;
        }
        cuda_check(c,cudaEventRecord(completion->event,c.stream));
#endif
        ++c.completion_stats.event_records;
        c.tail_completion = completion;
        return completion;
    } catch (...) {
        c.completions.pop_back();
        throw;
    }
}
void note_upload_reader(Context &c, Buffer &b, const Box &box) {
    Budget budget;
    Region updated;
    try { updated = unite(b.unfenced_uploads,{box},budget); }
    catch (const Fragmented &) { updated = {b.full()}; }
    if (b.unfenced_uploads.empty()) c.unfenced_upload_buffers.push_back(&b);
    b.unfenced_uploads = std::move(updated);
}
void wait(Context &c);
void seal_uploads(Context &c) {
    if (c.unfenced_upload_buffers.empty()) return;
    // Reclaim completed prefix events; later completion on this stream also
    // proves earlier DMA readers have finished.
    while (!c.completions.empty() && completion_ready(c,c.completions.front())) {}
    for (auto *b : c.unfenced_upload_buffers) {
        b->upload_readers.erase(std::remove_if(b->upload_readers.begin(),b->upload_readers.end(),
            [](const auto &reader) { return reader.completion->complete; }),b->upload_readers.end());
        b->upload_readers.reserve(b->upload_readers.size()+1);
    }
    std::shared_ptr<Completion> completion;
    try { completion = record_completion(c); }
    catch (const Error &error) {
        if (error.status != FORT_SCOPE_RESOURCE) throw;
        wait(c); // A missing event cannot leave borrowed-memory uses untracked.
        return;
    } catch (const std::bad_alloc &) { wait(c); return; }
    for (auto *b : c.unfenced_upload_buffers) {
        b->upload_readers.push_back({std::move(b->unfenced_uploads),completion});
        b->unfenced_uploads.clear();
    }
    c.unfenced_upload_buffers.clear();
}
void wait_completion(Context &c, std::shared_ptr<Completion> completion) {
    if (completion->complete) return;
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
    if (std::getenv("FORT_SCOPE_TEST_FAIL_WAIT")) {
        c.poisoned = true;
        throw Error(FORT_SCOPE_EXECUTION,"injected execution completion failure");
    }
#endif
    if (completion_ready(c,completion)) return;
#ifndef FORT_SCOPE_CPU_TEST
    DeviceGuard guard(c);
    cuda_check(c,cudaEventSynchronize(completion->event));
#endif
    ++c.stats.waits; ++c.completion_stats.event_waits;
    complete_through(c,completion);
    trace(c,"wait");
}
// Said once per domain: from here on the run pays for a budget that is too small.
void budget_warning(Context &c,const char *consequence) noexcept {
    if (!c.borrowed_domain || c.residency->budget_warned) return;
    c.residency->budget_warned=true;
    try {
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << "Resident domain: the arrays in use exceed the device budget of " << c.device_budget
                  << " bytes; " << consequence << '.' << std::endl;
    } catch (...) {}
}
size_t field_budget(const Context &c) {
    require(c.scratch_capacity <= c.device_budget &&
            c.residency->workspace_bytes <= c.device_budget-c.scratch_capacity,
            FORT_SCOPE_STATE, "scratch/workspace capacity exceeds scope budget");
    return c.device_budget-c.scratch_capacity-c.residency->workspace_bytes;
}
void payload_peak(Context &c) noexcept {
    // Every allocation has already passed the inclusive, overflow-safe budget
    // check. Preserve the original field-only peak in fort_scope_stats.
    const auto total = c.stats.allocated_bytes+c.scratch_capacity+c.residency->workspace_bytes;
    c.scratch_stats.peak_total_device_bytes = std::max<uint64_t>(
        total, c.scratch_stats.peak_total_device_bytes);
}
void allocate(Context &c, Buffer &b) {
    if (b.device || !b.bytes) return;
    const auto budget = field_budget(c);
    require(c.stats.allocated_bytes <= budget && b.bytes <= budget-c.stats.allocated_bytes,
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
    mark_context_pending(c);
#endif
    ++c.stats.allocations;
    c.stats.allocated_bytes += b.bytes;
    c.stats.peak_device_bytes = std::max(c.stats.allocated_bytes, c.stats.peak_device_bytes);
    payload_peak(c);
    trace(c, "allocate", &b, b.bytes);
}
void wait(Context &c) {
    if (!c.pending) { complete_all(c); return; }
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
    complete_all(c);
    ++c.stats.waits; ++c.completion_stats.stream_waits;
    trace(c, "wait");
}
void release_scratch(Context &c) {
    if (!c.scratch) return;
#ifdef FORT_SCOPE_CPU_TEST
    std::free(c.scratch);
#else
    DeviceGuard guard(c);
#if CUDART_VERSION >= 11020
    if (c.pool) { cuda_check(c, cudaFreeAsync(c.scratch, c.stream)); mark_context_pending(c); }
    else
#endif
        cuda_check(c, cudaFree(c.scratch));
#endif
    trace(c, "scratch_free", nullptr, c.scratch_capacity);
    c.scratch = nullptr;
    c.scratch_capacity = 0;
    c.scratch_stats.capacity_bytes = 0;
}
void grow_scratch(Context &c, size_t bytes) {
    require(c.residency->workspace_bytes <= c.device_budget &&
            c.stats.allocated_bytes <= c.device_budget-c.residency->workspace_bytes &&
            bytes <= c.device_budget-c.residency->workspace_bytes-c.stats.allocated_bytes,
            FORT_SCOPE_RESOURCE, "scratch allocation exceeds the declared scope budget");
    const bool growing = c.scratch != nullptr;
    // An idle lease can still have pending kernels on this stream. Retire its
    // storage only after completion; do not temporarily reserve two arenas.
    if (growing) {
        wait(c);
        release_scratch(c);
        wait(c);
    }
#ifdef FORT_SCOPE_TEST_FAULTS
    require(!std::getenv("FORT_SCOPE_TEST_FAIL_SCRATCH_ALLOC"), FORT_SCOPE_RESOURCE,
            "injected scratch allocation failure");
#endif
    initialize(c);
    void *device = nullptr;
#ifdef FORT_SCOPE_CPU_TEST
    require(!std::getenv("FORT_SCOPE_TEST_FAIL_ALLOC"), FORT_SCOPE_RESOURCE, "injected device allocation failure");
    device = std::malloc(bytes);
    require(device, FORT_SCOPE_RESOURCE, "scratch reference allocation failed");
#else
    // The reference runtime reserves exactly the requested growth too.
    // Extra capacity here can consume the budget for subsequent fields.
    DeviceGuard guard(c);
#if CUDART_VERSION >= 11020
    if (c.pool) cuda_check(c, cudaMallocFromPoolAsync(&device, bytes, c.pool, c.stream), true);
    else
#endif
        cuda_check(c, cudaMalloc(&device, bytes), true);
    mark_context_pending(c);
#endif
    c.scratch = device;
    c.scratch_capacity = bytes;
    ++c.scratch_stats.allocations;
    if (growing) ++c.scratch_stats.grows;
    c.scratch_stats.capacity_bytes = bytes;
    c.scratch_stats.peak_scratch_bytes = std::max<uint64_t>(bytes, c.scratch_stats.peak_scratch_bytes);
    payload_peak(c);
    trace(c, "scratch_allocate", nullptr, bytes);
}
const char *transfer_reason(uint32_t reason) noexcept {
    switch (reason) {
        case FORT_SCOPE_TRANSFER_ESTIMATES_UNAVAILABLE: return "transfer_estimates_unavailable";
        case FORT_SCOPE_TRANSFER_PIPELINED_UNAVAILABLE: return "pipelined_not_available";
        case FORT_SCOPE_TRANSFER_BUDGET: return "pinned_budget_exhausted";
        case FORT_SCOPE_TRANSFER_ALLOCATION: return "pinned_allocation_failed";
        case FORT_SCOPE_TRANSFER_COALESCED_INELIGIBLE: return "coalesced_upload_ineligible";
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
        std::cerr << std::setprecision(17) << "FORT_SCOPED evidence {\"schema_version\":1,\"context\":" << handle << DomainJson{c}
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
void graph_statistics_trace(const Context &c, bool complete) noexcept {
    if (!trace_enabled() || (!c.graph_stats.creations && !c.queue_stats.direct_kernels)) return;
    try {
        const auto &stats = c.graph_stats;
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << "FORT_SCOPED evidence {\"schema_version\":1,\"context\":" << c.handle << DomainJson{c}
                  << ",\"event\":\"graph_statistics\",\"stats_version\":" << stats.version
                  << ",\"complete\":" << (complete ? "true" : "false");
        for (const auto &[key,value] : {
            std::pair{"creations",stats.creations}, {"instantiations",stats.instantiations},
            {"rebuilds",stats.rebuilds}, {"submissions",stats.submissions}, {"replays",stats.replays},
            {"parameter_updates",stats.parameter_updates}, {"logical_kernels",stats.logical_kernels},
            {"live_graphs",stats.live_graphs}, {"destructions",stats.destructions}})
            std::cerr << ",\"" << key << "\":" << value;
        std::cerr << "}\n";
    } catch (...) {}
}
void completion_statistics_trace(const Context &c, bool complete) noexcept {
    if (!trace_enabled() || (!c.completion_stats.event_records && !c.completion_stats.independent_host_tasks)) return;
    try {
        const auto &stats = c.completion_stats;
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << "FORT_SCOPED evidence {\"schema_version\":1,\"context\":" << c.handle << DomainJson{c}
                  << ",\"event\":\"completion_statistics\",\"stats_version\":" << stats.version
                  << ",\"complete\":" << (complete ? "true" : "false");
        for (const auto &[key,value] : {
            std::pair{"event_records",stats.event_records}, {"event_queries",stats.event_queries},
            {"event_waits",stats.event_waits}, {"stream_waits",stats.stream_waits},
            {"created_events",stats.created_events}, {"reused_events",stats.reused_events},
            {"independent_host_tasks",stats.independent_host_tasks}})
            std::cerr << ",\"" << key << "\":" << value;
        std::cerr << "}\n";
    } catch (...) {}
}
void queue_statistics_trace(const Context &c,bool complete) noexcept {
    const auto &stats=c.queue_stats;
    if (!trace_enabled() || (!stats.accepted_tasks && !stats.submissions && !stats.direct_submissions &&
        !stats.direct_kernels && stats.queue_enabled && stats.graphs_enabled)) return;
    try {
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << "FORT_SCOPED evidence {\"schema_version\":1,\"context\":" << c.handle << DomainJson{c}
                  << ",\"event\":\"command_queue_statistics\",\"stats_version\":" << stats.version
                  << ",\"complete\":" << (complete?"true":"false")
                  << ",\"queue_enabled\":" << stats.queue_enabled << ",\"graphs_enabled\":" << stats.graphs_enabled;
        for (const auto &[key,value] : {
            std::pair{"accepted_tasks",stats.accepted_tasks}, {"submissions",stats.submissions},
            {"conflict_flushes",stats.conflict_flushes}, {"explicit_flushes",stats.explicit_flushes},
            {"pending_tasks",stats.pending_tasks}, {"peak_pending_tasks",stats.peak_pending_tasks},
            {"copied_argument_bytes",stats.copied_argument_bytes}, {"direct_submissions",stats.direct_submissions},
            {"direct_kernels",stats.direct_kernels}, {"graph_resource_fallbacks",stats.graph_resource_fallbacks}})
            std::cerr << ",\"" << key << "\":" << value;
        std::cerr << "}\n";
    } catch (...) {}
}
void residency_statistics_trace(const Context &c,bool complete) noexcept {
    const auto &stats=c.residency_stats;
    if (!trace_enabled() || !stats.evictions) return;
    try {
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << "FORT_SCOPED evidence {\"schema_version\":1,\"context\":" << c.handle << DomainJson{c}
                  << ",\"event\":\"residency_statistics\",\"stats_version\":" << stats.version
                  << ",\"complete\":" << (complete?"true":"false") << ",\"device_budget\":" << c.device_budget;
        for (const auto &[key,value] : {
            std::pair{"reserve_batches",stats.reserve_batches}, {"requested_resources",stats.requested_resources},
            {"reuse_hits",stats.reuse_hits}, {"evictions",stats.evictions}, {"evicted_bytes",stats.evicted_bytes},
            {"dirty_evictions",stats.dirty_evictions}, {"dirty_download_bytes",stats.dirty_download_bytes},
            {"pressure_flushes",stats.pressure_flushes}})
            std::cerr << ",\"" << key << "\":" << value;
        std::cerr << "}\n";
    } catch (...) {}
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
    trace(c, upload ? "upload" : "download", &b, plan.bytes);
    return true;
}
void copy(Context &c, Buffer &b, const Box &box, bool upload) {
    const fort_physical::CopyPlan plan(b.element_bytes, b.extents, box.lo, box.hi);
    require(plan.valid, FORT_SCOPE_ARGUMENT, "invalid physical transfer layout");
    if (!plan.bytes) return;
    allocate(c, b);
    if (c.transfer_stats.requested_mode == FORT_SCOPE_TRANSFERS_PINNED && copy_pinned(c, b, plan, upload)) return;
    if (upload) note_upload_reader(c,b,box);
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
        } else if (op.pitch == op.width && op.height == op.physical_height) {
            cuda_check(c, cudaMemcpyAsync(to, from, op.width*op.height*op.depth, direction, c.stream));
        } else if (op.pitch == op.width) {
            cuda_check(c, cudaMemcpy2DAsync(to, op.pitch*op.physical_height, from, op.pitch*op.physical_height, op.width*op.height, op.depth, direction, c.stream));
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
    mark_context_pending(c);
    trace(c, upload ? "upload" : "download", &b, plan.bytes);
}
void compact_trace(const Context &c,const Buffer &b,const fort_physical::compact::Plan &plan,
                   const fort_physical::compact::Price &price,bool upload,size_t boxes,bool reused) noexcept {
    if(!trace_enabled())return;
    try {
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr<<"FORT_SCOPED evidence {\"schema_version\":1,\"context\":"<<c.handle<<DomainJson{c}
                 <<",\"event\":\"compact_transfer\",\"version\":1,\"buffer\":"<<b.handle
                 <<",\"identity\":"<<b.identity<<",\"generation\":"<<b.generation
                 <<",\"upload\":"<<(upload?"true":"false")<<",\"bytes\":"<<plan.bytes
                 <<",\"logical_rectangles\":"<<boxes<<",\"physical_operations\":"<<price.operations
                 <<",\"dma_calls\":1,\"copy_kernel_launches\":"<<price.chunks
                 <<",\"word_bytes\":"<<plan.word_bytes<<",\"workspace_bytes\":"<<price.workspace_bytes
                 <<",\"reused\":"<<(reused?"true":"false");
        const auto &p=c.provenance;
        for(const auto &[key,value]:{std::pair{"owner",p.owner},{"procedure",p.procedure},{"segment",p.segment},
                {"operation",p.operation},{"implementation",p.implementation},{"boundary",p.boundary}})
            std::cerr<<",\""<<key<<"\":\""<<(value[0]?value:"unknown")<<'"';
        std::cerr<<"}\n";
    } catch(...) {}
}
// Pricing is deliberately separate from completed physical-copy evidence.
// This row can survive a later wait/submission failure and grants no execution credit.
void copy_pricing_trace(const Context &c,const Buffer &b,const fort_physical::compact::Plan &plan,
                        const fort_physical::compact::startup::Decision &decision,
                        const fort_physical::compact::startup::State &before,
                        bool upload,bool reused,bool pending) noexcept {
    if(!trace_enabled())return;
    try {
        using namespace fort_physical::compact;
        const auto &selected=decision.price;
        const PoolState warm{selected.capacity,selected.workspace_bytes,true,false};
        const auto reuse=price(plan,c.copy_costs->copy,c.copy_costs->base,upload,pending,warm);
        const PoolState empty{0,selected.workspace_bytes,false,true};
        const auto cold=price(plan,c.copy_costs->copy,c.copy_costs->base,upload,pending,empty);
        // Cold measurements cover the minimal capacity bucket only. In particular,
        // an oversized compatible reused pair does not price an unmeasured cold allocation.
        const bool cold_known=cold.compact_available && cold.capacity==selected.capacity;
        const bool reuse_known=reuse.compact_available;
        unsigned bucket=0;while(bucket<4 && capacities[bucket]!=selected.capacity)++bucket;
        std::ostringstream out;out<<std::setprecision(17);
        out<<"FORT_SCOPED evidence {\"schema_version\":1,\"context\":"<<c.handle<<DomainJson{c}
           <<",\"event\":\"copy_pricing\",\"version\":1,\"evidence\":\"MODELED\""
           <<",\"phase\":\"before_submission\",\"confirms_execution\":false"
           <<",\"buffer\":"<<b.handle<<",\"identity\":"<<b.identity<<",\"generation\":"<<b.generation
           <<",\"device\":"<<c.device<<",\"upload\":"<<(upload?"true":"false")
           <<",\"decision\":\""<<(decision.seed?"credit_funded_cold":reused?"reuse":"locally_profitable_cold")
           <<"\",\"capacity_bytes\":"<<selected.capacity
           <<",\"direct_seconds\":"<<selected.direct_seconds<<",\"selected_seconds\":"<<selected.compact_seconds;
        auto optional=[&](const char *name,bool known,double value) {
            out<<",\""<<name<<"\":";if(known)out<<value;else out<<"null";
        };
        optional("cold_seconds",cold_known,cold.compact_seconds);
        optional("reuse_seconds",reuse_known,reuse.compact_seconds);
        optional("prior_credit_seconds",!reused&&bucket<4,bucket<4?before.credit[bucket]:0);
        optional("startup_premium_seconds",cold_known&&reuse_known,std::max(0.,cold.compact_seconds-reuse.compact_seconds));
        out<<",\"state_revision\":"<<before.revision<<",\"state_generation\":"<<before.generation<<",\"model_key\":[";
        for(size_t i=0;i<before.model.words.size();++i)out<<(i?",":"")<<before.model.words[i];
        out<<']';
        const auto &p=c.provenance;
        for(const auto &[key,value]:{std::pair{"owner",p.owner},{"procedure",p.procedure},{"segment",p.segment},
                {"operation",p.operation},{"implementation",p.implementation},{"boundary",p.boundary}})
            out<<",\""<<key<<"\":\""<<(value[0]?value:"unknown")<<'"';
        out<<"}\n";
        std::lock_guard<std::mutex> lock(trace_mutex);std::cerr<<out.str();
    } catch(...) {}
}
struct StartupObservation {
    fort_physical::compact::startup::State before;
    fort_physical::compact::startup::Decision decision;
    bool valid=false,priced=false;
};
void forget_copy_startup(Context &c) noexcept {
    if(!c.copy_costs)return;
#ifdef FORT_SCOPE_CPU_TEST
    fort_physical::compact::startup::ownership_changed(c.residency->compact_reference_startup);
#else
    fort_staging::forget_startup(fort_physical::compact::startup::key(*c.copy_costs,c.device));
#endif
}
bool copy_compact(Context &c,Buffer &b,const Region &parts,bool upload,StartupObservation &observation) {
    namespace compact=fort_physical::compact;
    if(c.transfer_stats.requested_mode!=FORT_SCOPE_TRANSFERS_AUTO || !c.copy_costs ||
       parts.empty())return false;
    std::vector<fort_physical::CopyOperation> operations;
    compact::Plan plan;
    try {
        for(const auto &part:parts) {
            const fort_physical::CopyPlan original(b.element_bytes,b.extents,part.lo,part.hi);
            require(original.valid,FORT_SCOPE_ARGUMENT,"invalid physical transfer layout");
            if(!original.visit([&](const auto &op) {
                if(operations.size()==compact::operation_limit)return false;
                operations.push_back(op);return true;
            })) return false;
        }
        plan=compact::make_plan(operations,b.bytes,std::numeric_limits<size_t>::max());
    } catch(const compact::Ineligible &) { return false; }
      catch(const std::bad_alloc &) { return false; }
    const auto own=field_budget(c);
    if(c.stats.allocated_bytes>own || (!b.device && b.bytes>own-c.stats.allocated_bytes))return false;
    const auto available=own-c.stats.allocated_bytes-(b.device?0:b.bytes);
    compact::PoolState pool;pool.available_device_bytes=available;
    const auto &pricing=*c.copy_costs;
#ifdef FORT_SCOPE_CPU_TEST
    compact::startup::bind(c.residency->compact_reference_startup,pricing,c.device);
    const auto before=c.residency->compact_reference_startup;
    pool.capacity=c.residency->compact_reference_capacity;
    pool.cold_available=pool.capacity==0;
    pool.reused=pool.capacity>=plan.bytes && pool.capacity<=available/2;
#else
    const auto snapshot=fort_staging::startup_snapshot([&](auto &state){compact::startup::bind(state,pricing,c.device);});
    const auto before=snapshot.startup;
    const auto idle=snapshot.idle_role==fort_staging::Role::Compact ? 2*snapshot.idle_capacity : 0;
    const auto active=snapshot.device_reserved-idle;
    pool.available_device_bytes=std::min(active>available?0:available-active,fort_staging::compact_available(snapshot));
    pool.capacity=snapshot.idle_capacity;
    pool.cold_available=snapshot.idle_capacity==0;
    pool.reused=snapshot.idle_device==c.device && snapshot.idle_role==fort_staging::Role::Compact &&
                pool.capacity>=plan.bytes && snapshot.device_reserved<=available &&
                pool.capacity<=pool.available_device_bytes/2;
#endif
    auto decision=compact::startup::choose(plan,pricing,upload,c.pending&&!c.residency->copy_completion_priced,pool,before);
    auto price=decision.price;
    if(!price.available) return false;
    observation.priced=true;
    if(!price.selected) {
        if(decision.observation>0)observation={before,decision,true,true};
        return false;
    }
    allocate(c,b);
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
    if(std::getenv("FORT_SCOPE_TEST_FAIL_COMPACT_ALLOC")) {
        transfer_fallback(c,FORT_SCOPE_TRANSFER_ALLOCATION);return false;
    }
#endif
#ifdef FORT_SCOPE_CPU_TEST
    std::vector<unsigned char> packed_storage,device_storage;
    try {packed_storage.resize(price.capacity);device_storage.resize(price.capacity);}
    catch(const std::bad_alloc &) {transfer_fallback(c,FORT_SCOPE_TRANSFER_ALLOCATION);return false;}
    if(decision.seed)require(compact::startup::reserve(c.residency->compact_reference_startup,before,decision),
                            FORT_SCOPE_STATE,"compact startup state changed");
    else compact::startup::ownership_changed(c.residency->compact_reference_startup);
    auto *packed=packed_storage.data(),*compact_device=device_storage.data();
    const bool reused=pool.reused;
#else
    DeviceGuard guard(c);
    fort_staging::Result lease;
    try {lease=fort_staging::acquire(price.capacity,fort_staging::Role::Compact,false,available,true,
                                   decision.seed?&before:nullptr,decision.seed?&decision:nullptr);}
    catch(const std::bad_alloc &) {transfer_fallback(c,FORT_SCOPE_TRANSFER_ALLOCATION);return false;}
    if(lease.exhausted) {transfer_fallback(c,(lease.unpriced_idle||lease.stale_startup)?FORT_SCOPE_TRANSFER_ESTIMATES_UNAVAILABLE:FORT_SCOPE_TRANSFER_BUDGET);return false;}
    if(lease.status!=cudaSuccess) {
        try {cuda_check(c,lease.status,true);}catch(const Error &error) {
            if(error.status!=FORT_SCOPE_RESOURCE)throw;
            transfer_fallback(c,FORT_SCOPE_TRANSFER_ALLOCATION);return false;
        }
    }
    require(bool(lease.slots),FORT_SCOPE_STATE,"missing compact staging lease");
    // Cache state can change after metadata pricing. Reprice the actual lease
    // before payload submission; a changed choice never replays source work.
    const bool reused=lease.reused;
    pool={lease.slots->capacity,available,reused};
    decision=compact::startup::choose(plan,pricing,upload,c.pending&&!c.residency->copy_completion_priced,pool,before);
    price=decision.price;
    if(!price.available || !price.selected || (decision.seed&&!lease.startup_paid)) {
        size_t freed=0;cuda_check(c,fort_staging::release(std::move(lease.slots),freed));
        return false;
    }
    auto &slot=lease.slots->slots[0];auto *packed=slot.host,*compact_device=slot.device;
#endif
    const auto workspace=price.workspace_bytes;
    require(workspace<=available,FORT_SCOPE_RESOURCE,"compact workspace exceeds inclusive device budget");
    c.residency->workspace_bytes+=workspace;payload_peak(c);
    bool charged=true,submitted=false;
    auto &stats=c.transfer_stats;
    copy_pricing_trace(c,b,plan,decision,before,upload,reused,c.pending&&!c.residency->copy_completion_priced);
    try {
        wait(c); // Full-layout allocations and prior writers use the domain stream.
        stats.staging_allocations+=reused?0:2;stats.staging_reuses+=reused?1:0;
        stats.slot_capacity=price.capacity;stats.staging_device_bytes=workspace;
        if(upload) {
            const auto start=std::chrono::steady_clock::now();size_t offset=0;
            for(const auto &op:operations) {pack_tile(packed+offset,static_cast<char*>(b.host),op,false);offset+=op.width*op.height*op.depth;}
            stats.packing_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            stats.packed_bytes+=plan.bytes;
        }
        auto kernels=[&](bool scatter) {
            for(const auto &chunk:plan.chunks) {
#ifdef FORT_SCOPE_CPU_TEST
                for(size_t i=0;i<chunk.count;++i) {
                    const auto &d=chunk.ops[i];
                    const fort_physical::CopyOperation op{d.offset,d.width,d.height,d.depth,d.pitch,d.physical_height};
                    pack_tile(compact_device+d.packed_offset,static_cast<char*>(b.device),op,scatter);
                }
#else
                auto *physical=static_cast<unsigned char*>(b.device);
                if(plan.word_bytes==8) {
                    if(scatter)compact::launch<uint64_t,true>(physical,compact_device,chunk,slot.stream);
                    else compact::launch<uint64_t,false>(physical,compact_device,chunk,slot.stream);
                } else if(plan.word_bytes==4) {
                    if(scatter)compact::launch<uint32_t,true>(physical,compact_device,chunk,slot.stream);
                    else compact::launch<uint32_t,false>(physical,compact_device,chunk,slot.stream);
                } else {
                    if(scatter)compact::launch<unsigned char,true>(physical,compact_device,chunk,slot.stream);
                    else compact::launch<unsigned char,false>(physical,compact_device,chunk,slot.stream);
                }
                cuda_check(c,cudaGetLastError());
#endif
            }
        };
        submitted=true;
#ifndef FORT_SCOPE_CPU_TEST
        slot.pending=true; // Also covers launch/API failure before event record.
#endif
        if(!upload)kernels(false);
#ifdef FORT_SCOPE_CPU_TEST
        if(upload)std::memcpy(compact_device,packed,plan.bytes);else std::memcpy(packed,compact_device,plan.bytes);
#else
        cuda_check(c,cudaMemcpyAsync(upload?compact_device:packed,upload?packed:compact_device,plan.bytes,
                                   upload?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,slot.stream));
#endif
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
        if(std::getenv("FORT_SCOPE_TEST_FAIL_COMPACT_SUBMIT"))throw Error(FORT_SCOPE_EXECUTION,"injected compact post-submission failure");
#endif
        if(upload)kernels(true);
        if(upload) {++c.stats.uploads;c.stats.upload_bytes+=plan.bytes;++stats.pinned_uploads;stats.pinned_upload_bytes+=plan.bytes;}
        else {++c.stats.downloads;c.stats.download_bytes+=plan.bytes;++stats.pinned_downloads;stats.pinned_download_bytes+=plan.bytes;}
        const auto start=std::chrono::steady_clock::now();
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
        if(std::getenv("FORT_SCOPE_TEST_FAIL_COMPACT_RECORD"))throw Error(FORT_SCOPE_EXECUTION,"injected compact event-record failure");
#endif
#ifndef FORT_SCOPE_CPU_TEST
        cuda_check(c,cudaEventRecord(slot.complete,slot.stream));
        cuda_check(c,cudaEventSynchronize(slot.complete));slot.pending=false;
#endif
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
        if(std::getenv("FORT_SCOPE_TEST_FAIL_COMPACT_WAIT"))throw Error(FORT_SCOPE_EXECUTION,"injected compact completion failure");
#endif
        ++stats.events;++stats.event_waits;
        stats.event_wait_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(!upload) {
            const auto unpack_start=std::chrono::steady_clock::now();size_t offset=0;
            for(const auto &op:operations) {pack_tile(packed+offset,static_cast<char*>(b.host),op,true);offset+=op.width*op.height*op.depth;}
            stats.unpacking_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-unpack_start).count();
            stats.unpacked_bytes+=plan.bytes;
        }
#ifdef FORT_SCOPE_CPU_TEST
        c.residency->compact_reference_capacity=price.capacity;
        compact::startup::ownership_changed(c.residency->compact_reference_startup);
#else
        size_t freed=0;cuda_check(c,fort_staging::release(std::move(lease.slots),freed));
#endif
        c.residency->workspace_bytes-=workspace;charged=false;
        stats.effective_mode=FORT_SCOPE_TRANSFERS_AUTO;
        compact_trace(c,b,plan,price,upload,parts.size(),reused);
        trace(c,upload?"upload":"download",&b,plan.bytes);
        return true;
    } catch(...) {
        if(submitted) {c.poisoned=true;forget_copy_startup(c);}
#ifndef FORT_SCOPE_CPU_TEST
        if(lease.slots) {
            size_t freed=0;
            if(lease.slots->cleanup(freed)!=cudaSuccess) {c.poisoned=true;charged=false;}
        }
#endif
        if(charged)c.residency->workspace_bytes-=workspace;
        throw;
    }
}
void copy_region(Context &c, Buffer &b, const Region &parts, bool upload) {
    StartupObservation observation;
    if(copy_compact(c,b,parts,upload,observation))return;
    // Experimental upload-only transport. Logical coverage remains the exact
    // requested union: extra physical bytes never establish device freshness.
    // Full source freshness is deliberately stronger than span freshness, and
    // prevents overwriting a newer device value in a gap between rectangles.
    if (upload && !parts.empty() && c.transfer_stats.requested_mode == FORT_SCOPE_TRANSFERS_COALESCED) {
        const auto full = b.full();
        const bool source_full = b.initialized.size()==1 && contains(b.initialized.front(),full) &&
            b.host_current.size()==1 && contains(b.host_current.front(),full);
        size_t first=b.bytes, last=0, required_bytes=0, physical_copies=0;
        if (source_full) {
            for (const auto &part : parts) {
                const fort_physical::CopyPlan plan(b.element_bytes,b.extents,part.lo,part.hi);
                require(plan.valid,FORT_SCOPE_ARGUMENT,"invalid physical transfer layout");
                if (!plan.bytes) continue;
                require(plan.bytes<=b.bytes-required_bytes &&
                        plan.copies<=std::numeric_limits<size_t>::max()-physical_copies,
                        FORT_SCOPE_ARGUMENT,"coalesced transfer extent overflow");
                required_bytes+=plan.bytes; physical_copies+=plan.copies;
                // Valid nonempty physical coordinates and the checked full
                // allocation product bound both flat offsets by b.bytes.
                size_t begin=0, end=b.element_bytes;
                for (size_t k=0;k<b.extents.size();++k) {
                    begin+=part.lo[k]*b.strides[k];
                    end+=(part.hi[k]-1)*b.strides[k];
                }
                first=std::min(first,begin); last=std::max(last,end);
            }
        }
        if (physical_copies>=2 && first<=last && last<=b.bytes &&
            last-first>=required_bytes && last-first-required_bytes<=required_bytes/8) {
            allocate(c,b);
            // This transport deliberately copies bytes between rectangles.
            // Hazard coverage must include those physical gaps as well.
            note_upload_reader(c,b,b.full());
            auto *host=static_cast<char *>(b.host)+first, *device=static_cast<char *>(b.device)+first;
            const auto bytes=last-first;
#ifdef FORT_SCOPE_CPU_TEST
            std::memcpy(device,host,bytes);
#else
            DeviceGuard guard(c);
            cuda_check(c,cudaMemcpyAsync(device,host,bytes,cudaMemcpyHostToDevice,c.stream));
#endif
            ++c.stats.uploads; c.stats.upload_bytes+=bytes; mark_context_pending(c);
            c.transfer_stats.effective_mode=FORT_SCOPE_TRANSFERS_COALESCED;
            trace(c,"upload_coalesced",&b,bytes);
            trace(c,"upload_requested",&b,required_bytes);
            trace(c,"upload",&b,bytes);
            return;
        }
        c.transfer_stats.fallback_reason = FORT_SCOPE_TRANSFER_COALESCED_INELIGIBLE;
        ++c.transfer_stats.fallbacks;
    }
    [[maybe_unused]] size_t submitted_parts=0;
    for (const auto &part : parts) {
        copy(c,b,part,upload);++submitted_parts;
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
        const auto fail_after=std::getenv("FORT_SCOPE_TEST_FAIL_DIRECT_BATCH_AFTER");
        if(fail_after && submitted_parts==std::strtoull(fail_after,nullptr,10)) {
            c.poisoned=true;forget_copy_startup(c);
            throw Error(FORT_SCOPE_EXECUTION,"injected direct batch post-submission failure");
        }
#endif
    }
    if(observation.priced)c.residency->copy_completion_priced=true;
    // Reached direct APIs have all returned success. No completion/readability
    // inference or new synchronization follows from this pricing observation.
    if(observation.valid) {
#ifdef FORT_SCOPE_CPU_TEST
        fort_physical::compact::startup::observe(c.residency->compact_reference_startup,observation.before,observation.decision);
#else
        bool changed=false;
        fort_staging::observe_startup(observation.before,observation.decision,changed);
#endif
    }
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
        if (empty(box)) continue;
        // The first rectangle is its own normalized union.
        if (result.empty()) result.push_back(box);
        else result = unite(std::move(result), {box}, budget);
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
std::shared_ptr<const Effects> parsed_effects(Context &c, const Buffer &b, const fort_scope_access *access) {
    auto &key = c.parsed_access_key;
    key.clear();
    // A list that cannot be read as a key takes the parser's own diagnostics.
    bool readable = access && !(access->flags & ~7U);
    if (readable) {
        key.push_back(b.extents.size());
        key.insert(key.end(),b.extents.begin(),b.extents.end());
        key.push_back(access->flags);
        const std::pair<const fort_scope_section *,size_t> lists[] = {
            {access->reads,access->read_count},{access->writes,access->write_count},
            {access->overwrites,access->overwrite_count}};
        for (const auto &list : lists) {
            readable = readable && list.second <= rectangle_limit && (!list.second || list.first);
            if (!readable) break;
            key.push_back(list.second);
            for (size_t i=0; i<list.second && readable; ++i) {
                readable = list.first[i].lower && list.first[i].upper;
                if (!readable) break;
                key.insert(key.end(),list.first[i].lower,list.first[i].lower+b.extents.size());
                key.insert(key.end(),list.first[i].upper,list.first[i].upper+b.extents.size());
            }
        }
    }
    if (!readable) return std::make_shared<const Effects>(effects(b,access));
    uint64_t hash = 1469598103934665603ULL;
    for (const auto word : key) hash = (hash^word)*1099511628211ULL;
    auto &bucket = c.parsed_accesses[hash];
    for (const auto &entry : bucket) if (entry.key == key) return entry.effects;
    auto parsed = std::make_shared<const Effects>(effects(b,access));
    if (c.parsed_access_count >= 4096) {
        // Queued commands keep the lists they use alive.
        c.parsed_accesses.clear(); c.parsed_access_count = 0;
        c.parsed_accesses[hash].push_back({key,parsed});
    } else bucket.push_back({key,parsed});
    ++c.parsed_access_count;
    return parsed;
}
Region physical_values(const Buffer &b) {
    return fort_scoped::coherence::physical_values(b);
}
void reconcile(Context &c, Buffer &b, bool device) {
    // Every rectangle copied is current at the source. Copying already-current
    // destination cells is harmless; copying a stale enclosing box is not.
    const Region latest_other = device ? b.host_current : b.device_current;
    const auto physical = physical_values(b);
    copy_region(c, b, latest_other, device);
    if (!device) wait(c);
    auto &chosen = device ? b.device_current : b.host_current;
    chosen = physical;
    ++c.stats.reconciliations;
    trace(c, device ? "reconcile_device" : "reconcile_host", &b);
}
void ensure_physical(Context &c, Buffer &b, const Region &requested, bool device) {
    if (requested.empty()) return;
    Budget budget;
    auto &chosen = device ? b.device_current : b.host_current;
    try {
        const auto missing = difference(requested, chosen, budget);
        auto updated = unite(chosen, missing, budget);
        copy_region(c, b, missing, device);
        chosen = std::move(updated);
    } catch (const Fragmented &) {
        reconcile(c, b, device);
    }
}
struct AccessState {
    Region initialized, host_current, device_current;
};
struct AccessPreparation {
    Buffer *resource;
    AccessState state;
    Prepared prepared;
    std::vector<Region> copies;
    Region writes;
    bool reconciled = false;
};
enum class AccessContract { Defined, OriginalHost };
// `b` is any coherence state: a buffer, or private metadata that simulates one.
template<class State>
AccessPreparation prepare_state(Buffer *resource, const State &b, const Effects &e, bool device,
                                AccessContract contract=AccessContract::Defined) {
    AccessPreparation result{resource, {b.initialized, b.host_current, b.device_current}, {}, {}, e.writes, false};
    // Simulate the complete acquisition against private metadata. In
    // particular, a later undefined read or invalid binding cannot publish an
    // earlier buffer or leave a write prepared for work that never ran.
    auto ensure_state = [&](Region requested) {
        if (requested.empty()) return;
        Budget budget;
        if (contract==AccessContract::OriginalHost) {
            // Synchronize only tracked physical values. A may-read envelope
            // is not evidence that every cell is read or even initialized.
            // Keep physical versions surviving a semantic definition kill.
            requested = fort_scoped::coherence::tracked_accesses(b, requested);
        } else {
            require(difference(requested, result.state.initialized, budget).empty(), FORT_SCOPE_UNINITIALIZED,
                    "access reads an uninitialized section");
        }
        auto &chosen = device ? result.state.device_current : result.state.host_current;
        auto missing = difference(requested, chosen, budget);
        // Uniting nothing only normalizes; do that in place.
        if (missing.empty()) { coalesce(chosen); return; }
        auto updated = unite(chosen, missing, budget);
        result.copies.push_back(std::move(missing));
        chosen = std::move(updated);
    };
    auto compute = [&]() {
        Budget budget;
        const auto preserve = difference(e.writes, e.overwrites, budget);
        ensure_state(e.reads);
        ensure_state(preserve);
        if (contract==AccessContract::OriginalHost)
            return fort_scoped::coherence::prepare_original_host(result.state, e);
        return prepare(result.state, e, device);
    };
    try { result.prepared = compute(); }
    catch (const Fragmented &) {
        result.state = {b.initialized, b.host_current, b.device_current};
        result.copies.clear();
        const auto &other = device ? b.host_current : b.device_current;
        if (!other.empty()) result.copies.push_back(other);
        (device ? result.state.device_current : result.state.host_current) = fort_scoped::coherence::physical_values(b);
        result.reconciled = true;
        try { result.prepared = compute(); }
        catch (const Fragmented &) {
            // A write can split the opposite mirror beyond its metadata
            // budget. Only that case may drop the redundant mirror; a
            // read-only reconciliation always retains its original source.
            if (e.writes.empty()) throw;
            (device ? result.state.host_current : result.state.device_current).clear();
            result.prepared = compute();
        }
    }
    return result;
}
AccessPreparation prepare_effects(Buffer &b, const Effects &e, bool device,
                                  AccessContract contract=AccessContract::Defined) {
    require(contract!=AccessContract::OriginalHost || !device, FORT_SCOPE_ARGUMENT,
            "original host access cannot acquire device pointers");
    require(!b.prepared, FORT_SCOPE_STATE, "buffer access already prepared");
    return prepare_state(&b,b,e,device,contract);
}
AccessPreparation prepare_access(Buffer &b, const fort_scope_access *access, bool device,
                                 AccessContract contract=AccessContract::Defined) {
    return prepare_effects(b,effects(b,access),device,contract);
}
void acquire_prepared_accesses(Context &c, std::vector<AccessPreparation> &prepared,
                               bool device, void **devices) {
    const size_t count=prepared.size();
    for (auto &item : prepared)
        for (const auto &parts : item.copies) copy_region(c, *item.resource, parts, device);
    seal_uploads(c);
    if (!device && count) {
        const bool downloads = std::any_of(prepared.begin(),prepared.end(),[](const auto &item) {
            return !item.copies.empty();
        });
        if (downloads && c.pending) {
            // All downloads for this host task are queued before its one
            // completion. Later unrelated GPU work is outside this event.
            try { wait_completion(c,record_completion(c)); }
            catch (const Error &error) { if (error.status != FORT_SCOPE_RESOURCE) throw; wait(c); }
            catch (const std::bad_alloc &) { wait(c); }
        } else {
            std::unordered_set<const Completion *> needed;
            for (auto &item : prepared) {
                if (item.writes.empty()) continue;
                for (const auto &reader : item.resource->upload_readers)
                    if (overlaps(item.writes,reader.sections) && !completion_ready(c,reader.completion))
                        needed.insert(reader.completion.get());
            }
            std::shared_ptr<Completion> latest;
            for (auto i=c.completions.rbegin(); i!=c.completions.rend(); ++i)
                if (needed.count(i->get())) { latest=*i; break; }
            if (latest) wait_completion(c,std::move(latest));
            else if (!downloads) ++c.completion_stats.independent_host_tasks;
        }
    }
    for (size_t i=0; i<count; ++i) {
        auto &item = prepared[i];
        auto &b = *item.resource;
        b.host_current = std::move(item.state.host_current);
        b.device_current = std::move(item.state.device_current);
        b.prepared = std::move(item.prepared);
        if (item.reconciled) {
            ++c.stats.reconciliations;
            trace(c, device ? "reconcile_device" : "reconcile_host", &b);
        }
        if (device) devices[i] = b.device;
    }
}
void begin_accesses(Context &c, const fort_scope_plan_binding *bindings, size_t count,
                    bool device, void **devices, AccessContract contract=AccessContract::Defined) {
    require(!count || bindings, FORT_SCOPE_ARGUMENT, "missing task access bindings");
    require(!device || !count || devices, FORT_SCOPE_ARGUMENT, "missing device pointer outputs");
    drain_accesses(c,bindings,count);
    std::vector<AccessPreparation> prepared;
    prepared.reserve(count);
    std::unordered_set<fort_buffer_t> seen;
    for (size_t i=0; i<count; ++i) {
        require(seen.insert(bindings[i].buffer).second, FORT_SCOPE_ALIAS,
                "task access bindings repeat a canonical buffer");
        prepared.push_back(prepare_access(buffer(c, bindings[i].buffer), &bindings[i].access, device, contract));
    }
    if (device) {
        std::vector<fort_buffer_t> handles;
        handles.reserve(count);
        for (const auto &item : prepared) handles.push_back(item.resource->handle);
        const auto queued = c.commands.size();
        reserve_devices(c,handles.data(),handles.size());
        // Pressure may submit otherwise independent queued sections of a
        // requested root. Preserve those newly materialized versions too.
        if (queued!=c.commands.size())
            for (size_t i=0;i<count;++i)
                prepared[i]=prepare_access(buffer(c,bindings[i].buffer),&bindings[i].access,true,contract);
    }
    acquire_prepared_accesses(c,prepared,device,devices);
}
#ifndef FORT_SCOPE_CPU_TEST
void submit_graph(Context &c, uint64_t key, const fort_scope_kernel_v1 *kernels, size_t count,bool resource_fallback=false,
                  KernelTracePositions positions={},const fort_scoped::graphs::ArgumentValues *const *values=nullptr) {
    seal_uploads(c);
    initialize(c);
    DeviceGuard guard(c);
    bool launch_attempted = false;
    if (c.queue_stats.graphs_enabled) {
        bool warmup = false;
        auto status = c.graphs.observe(key,kernels,count,c.graph_stats,warmup);
        if (status == cudaSuccess && !warmup)
            status = c.graphs.submit(key,kernels,count,c.stream,c.graph_stats,launch_attempted,values);
        if (status == cudaSuccess && !warmup) {
            c.stats.launches += count; mark_context_pending(c);
            trace(c,"graph_submit",nullptr,count);
            // These are submitted logical graph members, not individual CUDA
            // launch API calls or evidence of device completion.
            if (trace_enabled())
                for (size_t i=0;i<count;++i)
                    trace(c,"graph_kernel",nullptr,0,nullptr,positions.at(i));
            return;
        }
        if (status == cudaSuccess) trace(c,"graph_warmup_direct",nullptr,count);
        else {
            if (launch_attempted || (status != cudaErrorMemoryAllocation && status != cudaErrorNotSupported))
                cuda_check(c,status);
            const auto pending = cudaGetLastError();
            if (pending != cudaSuccess && pending != status) cuda_check(c,pending);
            if (!resource_fallback) throw Error(FORT_SCOPE_RESOURCE,cudaGetErrorString(status));
            ++c.queue_stats.graph_resource_fallbacks;
        }
    }
    for (size_t i=0; i<count; ++i) {
        const auto &kernel=kernels[i];
        cuda_check(c,cudaLaunchKernel(kernel.function,dim3(kernel.grid[0],kernel.grid[1],kernel.grid[2]),
            dim3(kernel.block[0],kernel.block[1],kernel.block[2]),const_cast<void **>(kernel.arguments),kernel.shared_bytes,c.stream));
        ++c.stats.launches; ++c.queue_stats.direct_kernels; mark_context_pending(c);
        trace(c,"launch",nullptr,0,nullptr,positions.at(i));
    }
    ++c.queue_stats.direct_submissions;
}
#endif
// Bindings names, for member `i` of a graph, the buffers it touches and its
// effects on each: `size(i)`, `buffer(i,j)` and `effects(i,j,actual)`.
template<class Bindings>
void execute_graph(Context &c, uint64_t key, const fort_scope_kernel_v1 *kernels, size_t count,
                   Bindings &&bindings, bool resource_fallback=false, KernelTracePositions positions={},
                   const fort_scoped::graphs::ArgumentValues *const *values=nullptr) {
    struct Resource {
        Buffer *actual;
        AccessState state;
        std::vector<Region> uploads;
        size_t reconciliations = 0;
    };
    std::vector<Resource> resources;
    // Each step sees preceding writes, but none of those definitions become
    // externally visible before the entire group has been submitted safely.
    for (size_t i=0; i<count; ++i) {
        const size_t binding_count = bindings.size(i);
        for (size_t j=0; j<binding_count; ++j) {
            const auto handle = bindings.buffer(i,j);
            for (size_t k=0; k<j; ++k)
                require(bindings.buffer(i,k) != handle,FORT_SCOPE_ALIAS,"graph task repeats a canonical buffer");
            auto found = std::find_if(resources.begin(),resources.end(),[&](const Resource &resource) {
                return resource.actual->handle == handle;
            });
            if (found == resources.end()) {
                auto &b = buffer(c,handle);
                require(!b.prepared,FORT_SCOPE_STATE,"graph resource access already prepared");
                resources.push_back({&b,{b.initialized,b.host_current,b.device_current},{},0});
                found = resources.end()-1;
            }
            auto &resource = *found;
            const Effects &e = bindings.effects(i,j,*resource.actual);
            if (device_effects_keep_state(resource.state,e)) continue;
            auto prepared = prepare_state(resource.actual,resource.state,e,true);
            for (auto &parts : prepared.copies) {
                Budget budget;
                require(difference(parts,resource.actual->host_current,budget).empty(),FORT_SCOPE_STATE,
                        "graph external requirement has no current host source");
                resource.uploads.push_back(std::move(parts));
            }
            resource.state.initialized = std::move(prepared.prepared.initialized);
            resource.state.device_current = std::move(prepared.prepared.current);
            resource.state.host_current = std::move(prepared.prepared.opposite);
            if (prepared.reconciled) ++resource.reconciliations;
        }
    }
#ifdef FORT_SCOPE_CPU_TEST
    (void)key;
    (void)kernels;
    (void)resource_fallback;
    (void)positions;
    (void)values;
    throw Error(FORT_SCOPE_RESOURCE,"CUDA graphs unavailable in the CPU reference backend");
#else
    std::vector<fort_buffer_t> handles;
    handles.reserve(resources.size());
    for (const auto &resource : resources) handles.push_back(resource.actual->handle);
    reserve_devices(c,handles.data(),handles.size());
    for (auto &resource : resources)
        for (const auto &parts : resource.uploads) copy_region(c,*resource.actual,parts,true);
    submit_graph(c,key,kernels,count,resource_fallback,positions,values);
    for (auto &resource : resources) {
        auto &b = *resource.actual;
        b.initialized = std::move(resource.state.initialized);
        b.host_current = std::move(resource.state.host_current);
        b.device_current = std::move(resource.state.device_current);
        c.stats.reconciliations += resource.reconciliations;
        trace(c,"device_commit",&b);
    }
#endif
}
void execute_graph(Context &c, uint64_t key, const fort_scope_graph_task_v1 *tasks, size_t count) {
    require(!count || (key && tasks),FORT_SCOPE_ARGUMENT,"missing graph tasks or stable group key");
    if (!count) return;
    std::vector<fort_scope_kernel_v1> kernels;
    kernels.reserve(count);
    for (size_t i=0; i<count; ++i) {
        kernels.push_back(tasks[i].kernel);
        const auto invalid = fort_scoped::graphs::invalid(&kernels.back(),1);
        require(!invalid,FORT_SCOPE_ARGUMENT,invalid ? invalid : "invalid graph task kernel");
        require(!tasks[i].binding_count || tasks[i].bindings,FORT_SCOPE_ARGUMENT,"missing graph task effects");
    }
    struct Described {
        const fort_scope_graph_task_v1 *tasks;
        Effects parsed;
        size_t size(size_t i) const { return tasks[i].binding_count; }
        fort_buffer_t buffer(size_t i,size_t j) const { return tasks[i].bindings[j].buffer; }
        const Effects &effects(size_t i,size_t j,const Buffer &actual) {
            return parsed = ::effects(actual,&tasks[i].bindings[j].access);
        }
    };
    execute_graph(c,key,kernels.data(),count,Described{tasks,{}});
}
void apply_queued(Buffer &state,const QueuedBinding &binding) {
    if (device_effects_keep_state(state,*binding.effects)) return;
    auto prepared=prepare_state(&state,state,*binding.effects,true);
    state.initialized=std::move(prepared.prepared.initialized);
    state.device_current=std::move(prepared.prepared.current);
    state.host_current=std::move(prepared.prepared.opposite);
}
void forget_virtual(Context &c) noexcept {
    ++c.virtual_epoch;
}
// A queued command never coexists with an unfinished access to its buffers.
void refresh_virtual(Context &c) {
    if (c.virtual_checked==c.virtual_epoch) return;
    for (const auto &task : c.commands) for (const auto &binding : task->bindings)
        require(!buffer(c,binding.buffer).prepared,FORT_SCOPE_STATE,"queued resource already prepared");
    c.virtual_checked=c.virtual_epoch;
}
// The state of one buffer after the queued commands, or nullptr when none of
// them touches it and `create` is false.
Buffer *virtual_state(Context &c,fort_buffer_t handle,bool create=false) {
    auto found=c.virtual_buffers.find(handle);
    if (found!=c.virtual_buffers.end() && found->second.epoch==c.virtual_epoch) return &found->second.state;
    const auto touches=[&](const std::unique_ptr<QueuedTask> &task) {
        return (task->buffers&buffer_bit(handle)) &&
            std::any_of(task->bindings.begin(),task->bindings.end(),[&](const QueuedBinding &binding) {
                return binding.buffer==handle;
            });
    };
    if (!create && std::none_of(c.commands.begin(),c.commands.end(),touches)) return nullptr;
    const auto &actual=buffer(c,handle);
    require(!actual.prepared,FORT_SCOPE_STATE,"queued resource already prepared");
    // A stale entry is overwritten in place, which reuses its storage. It has
    // no valid epoch until every queued effect has been applied.
    if (found==c.virtual_buffers.end()) found=c.virtual_buffers.emplace(handle,Context::VirtualBuffer{0,actual}).first;
    else { found->second.epoch=0; found->second.state=actual; }
    auto &state=found->second.state;
    const uint64_t bit=buffer_bit(handle);
    for (const auto &task : c.commands) if (task->buffers&bit)
        for (const auto &binding : task->bindings) if (binding.buffer==handle) apply_queued(state,binding);
    found->second.epoch=c.virtual_epoch;
    return &state;
}
void apply_queued(Context &c,const QueuedTask &task) {
    for (const auto &binding : task.bindings) apply_queued(*virtual_state(c,binding.buffer,true),binding);
}
bool conflicts(const Effects &earlier,const Effects &later) {
    return overlaps(earlier.writes,later.reads) || overlaps(earlier.writes,later.writes) ||
           overlaps(earlier.reads,later.writes);
}
bool conflicts(const QueuedTask &earlier,const QueuedTask &later) {
    if (!(earlier.buffers&later.buffers)) return false;
    for (const auto &a : earlier.bindings) for (const auto &b : later.bindings)
        if (a.buffer==b.buffer && conflicts(*a.effects,*b.effects)) return true;
    return false;
}
void flush_commands(Context &c,const std::vector<bool> *selected,bool conflict) {
    if (c.commands.empty()) return;
    std::vector<fort_scope_kernel_v1> kernels;
    kernels.reserve(c.commands.size());
    std::vector<const QueuedTask *> tasks;
    tasks.reserve(c.commands.size());
    std::vector<const fort_scoped::graphs::ArgumentValues *> values;
    values.reserve(c.commands.size());
    const bool tracing=trace_enabled();
    const fort_scope_trace_state_v1 unknown{};
    std::unique_ptr<const fort_scope_trace_state_v1 *[]> positions;
    if (tracing && trace_allocation_allowed("flush"))
        positions.reset(new(std::nothrow) const fort_scope_trace_state_v1 *[c.commands.size()]);
    uint64_t key=1469598103934665603ULL;
    for (size_t i=0; i<c.commands.size(); ++i) if (!selected || (*selected)[i]) {
        kernels.push_back(c.commands[i]->launch());
        tasks.push_back(c.commands[i].get());
        values.push_back(&c.commands[i]->values);
        if (positions) positions[tasks.size()-1]=c.commands[i]->provenance ? c.commands[i]->provenance.get() : &unknown;
        key=(key^c.commands[i]->key)*1099511628211ULL;
    }
    if (tasks.empty()) return;
    // Diagnostic allocation failure loses attribution, never numerical work.
    struct Queued {
        const QueuedTask *const *tasks;
        size_t size(size_t i) const { return tasks[i]->bindings.size(); }
        fort_buffer_t buffer(size_t i,size_t j) const { return tasks[i]->bindings[j].buffer; }
        const Effects &effects(size_t i,size_t j,const Buffer &) const { return *tasks[i]->bindings[j].effects; }
    };
    execute_graph(c,key?key:1,kernels.data(),kernels.size(),Queued{tasks.data()},true,
                  {positions.get(),&unknown},values.data());
    // No numerical work is replayable once submit has succeeded. Remove it
    // before any later source operation can query virtual definitions.
    if (!selected) c.commands.clear();
    else {
        size_t index=0;
        c.commands.erase(std::remove_if(c.commands.begin(),c.commands.end(),[&](const auto &) {
            return (*selected)[index++];
        }),c.commands.end());
    }
    ++c.queue_stats.submissions;
    if (conflict) ++c.queue_stats.conflict_flushes;
    else ++c.queue_stats.explicit_flushes;
    c.queue_stats.pending_tasks=c.commands.size();
    c.queued_work=0;
    for (const auto &command : c.commands) {
        const auto work=launch_work(command->kernel);
        c.queued_work=work>std::numeric_limits<uint64_t>::max()-c.queued_work ? std::numeric_limits<uint64_t>::max() : c.queued_work+work;
    }
    invalidate(c,Change::State);
    trace(c,"queue_flush",nullptr,tasks.size());
}
void drain_effects(Context &c,const std::vector<BoundEffects> &accesses) {
    if (c.commands.empty()) return;
    std::vector<bool> selected(c.commands.size(),false);
    uint64_t wanted=0;
    for (const auto &access : accesses) wanted|=buffer_bit(access.buffer);
    for (size_t i=0; i<c.commands.size(); ++i) if (c.commands[i]->buffers&wanted)
        for (const auto &queued : c.commands[i]->bindings) for (const auto &access : accesses)
            if (queued.buffer==access.buffer && conflicts(*queued.effects,access.effects)) selected[i]=true;
    // Include dependency ancestors while preserving original relative order.
    // Unrelated earlier tasks remain queued across this host boundary.
    for (size_t j=c.commands.size(); j-->0;) if (selected[j])
        for (size_t i=0; i<j; ++i) if (conflicts(*c.commands[i],*c.commands[j])) selected[i]=true;
    flush_commands(c,&selected,true);
}
void drain_accesses(Context &c,const fort_scope_plan_binding *bindings,size_t count) {
    if (c.commands.empty()) return;
    require(!count || bindings,FORT_SCOPE_ARGUMENT,"missing queue boundary accesses");
    std::vector<BoundEffects> accesses;
    accesses.reserve(count);
    for (size_t i=0; i<count; ++i)
        accesses.push_back({bindings[i].buffer,*parsed_effects(c,buffer(c,bindings[i].buffer),&bindings[i].access)});
    drain_effects(c,accesses);
}
void drain_resources(Context &c,const fort_buffer_t *handles,size_t count) {
    if (c.commands.empty()) return;
    require(!count || handles,FORT_SCOPE_ARGUMENT,"missing queue boundary resources");
    std::vector<BoundEffects> accesses;
    for (size_t i=0; i<count; ++i) {
        auto &b=buffer(c,handles[i]);
        const Region full=empty(b.full())?Region{}:Region{b.full()};
        accesses.push_back({handles[i],{full,full,full}});
    }
    drain_effects(c,accesses);
}
// A queued launch runs its members concurrently: each member is a part of the
// kernel with its own effects. The launch is accepted only when no member
// depends on another, judged on the actual sections, and then carries the
// union of their effects. One member is the ordinary single task.
void enqueue_graph(Context &c,uint64_t key,const fort_scope_kernel_v1 &kernel,const size_t *sizes,
                   const fort_scope_graph_member_v1 *members,size_t member_count) {
    require(key && members && member_count,FORT_SCOPE_ARGUMENT,"missing queued graph task or code key");
    const auto invalid=fort_scoped::graphs::invalid(&kernel,1);
    require(!invalid,FORT_SCOPE_ARGUMENT,invalid?invalid:"invalid queued kernel");
    require(!kernel.argument_count || sizes,FORT_SCOPE_ARGUMENT,"missing queued kernel argument sizes");
    for (size_t m=0; m<member_count; ++m)
        require(!members[m].binding_count || members[m].bindings,FORT_SCOPE_ARGUMENT,"missing queued task effects");
    auto owned=std::make_unique<QueuedTask>(); owned->key=key; owned->kernel=kernel;
    if (trace_enabled() && trace_allocation_allowed("enqueue"))
        owned->provenance.reset(new(std::nothrow) fort_scope_trace_state_v1(c.provenance));
    owned->arguments.reserve(kernel.argument_count);
    uint64_t bytes=0;
    for (size_t i=0; i<kernel.argument_count; ++i) {
        require(sizes[i] && sizes[i]<=std::numeric_limits<uint64_t>::max()-bytes,FORT_SCOPE_ARGUMENT,"invalid queued argument size");
        bytes+=sizes[i];
    }
    require(owned->values.capture(kernel,sizes),FORT_SCOPE_ARGUMENT,"queued argument padding overflows size_t");
    for (size_t i=0;i<kernel.argument_count;++i) owned->arguments.push_back(owned->values.argument(i));
    for (size_t m=0; m<member_count; ++m) for (size_t i=0; i<members[m].binding_count; ++i) {
        const auto &binding=members[m].bindings[i]; auto &b=buffer(c,binding.buffer);
        for (size_t k=0; k<i; ++k)
            require(members[m].bindings[k].buffer!=binding.buffer,FORT_SCOPE_ALIAS,"queued task repeats a canonical buffer");
        require(!b.prepared,FORT_SCOPE_STATE,"queued resource already prepared");
        require(!b.bytes || b.device,FORT_SCOPE_STATE,"reserve queued device storage before enqueue");
        auto parsed=parsed_effects(c,b,&binding.access);
        const auto shared=std::find_if(owned->bindings.begin(),owned->bindings.end(),[&](const QueuedBinding &earlier) {
            return earlier.buffer==binding.buffer;
        });
        if (shared==owned->bindings.end()) {
            owned->bindings.push_back({binding.buffer,std::move(parsed)});
            owned->buffers|=buffer_bit(binding.buffer);
            continue;
        }
        // The same buffer under two members, including one array passed for
        // two arguments. The caller runs such members one after another.
        if (conflicts(*shared->effects,*parsed))
            throw Error(FORT_SCOPE_BOUNDARY,"members of a queued launch depend on each other");
        Budget budget;
        shared->effects=std::make_shared<const Effects>(Effects{
            unite(shared->effects->reads,parsed->reads,budget),
            unite(shared->effects->writes,parsed->writes,budget),
            unite(shared->effects->overwrites,parsed->overwrites,budget)});
    }
#ifndef FORT_SCOPE_CPU_TEST
    initialize(c); // Device availability is checked before accepting work.
#endif
    refresh_virtual(c);
    try {
        apply_queued(c,*owned);
        c.commands.push_back(std::move(owned));
    } catch (...) { forget_virtual(c); throw; }
    c.queue_stats.pending_tasks=c.commands.size();
    const auto accepted_size=c.commands.size();
    const auto work=launch_work(kernel), held=c.queued_work;
    c.queued_work=work>std::numeric_limits<uint64_t>::max()-held ? std::numeric_limits<uint64_t>::max() : held+work;
    if (!c.queue_stats.queue_enabled || (c.queue_work_limit && c.queued_work>=c.queue_work_limit)) {
        try { flush_commands(c); }
        catch (const Error &error) {
            if (error.status==FORT_SCOPE_RESOURCE) {
                c.commands.pop_back(); forget_virtual(c); c.queue_stats.pending_tasks=c.commands.size();
                c.queued_work=held;
            }
            throw;
        }
    }
    ++c.queue_stats.accepted_tasks; c.queue_stats.copied_argument_bytes+=bytes;
    c.queue_stats.peak_pending_tasks=std::max<uint64_t>(c.queue_stats.peak_pending_tasks,accepted_size);
    trace(c,"queue_enqueue",nullptr,bytes);
}
void end(Context &c, Buffer &b, bool device) {
    require(b.prepared && b.prepared->device == device, FORT_SCOPE_STATE, "missing or mismatched access begin");
    b.initialized = std::move(b.prepared->initialized);
    (device ? b.device_current : b.host_current) = std::move(b.prepared->current);
    (device ? b.host_current : b.device_current) = std::move(b.prepared->opposite);
    b.prepared.reset();
    trace(c, device ? "device_commit" : "host_commit", &b);
}
void queue_publication(Context &c, Buffer &b) {
    require(!b.prepared, FORT_SCOPE_STATE, "unfinished buffer access at scope boundary");
    ensure_physical(c,b,physical_values(b),false);
}
void publish(Context &c, Buffer &b) {
    queue_publication(c, b);
    wait(c);
}
void release(Context &c, Buffer &b) {
    if (!b.device) return;
#ifdef FORT_SCOPE_CPU_TEST
    std::free(b.device);
#else
    DeviceGuard guard(c);
#if CUDART_VERSION >= 11020
    if (c.pool) { cuda_check(c, cudaFreeAsync(b.device, c.stream)); mark_context_pending(c); }
    else
#endif
        cuda_check(c, cudaFree(b.device));
#endif
    b.device = nullptr;
    c.stats.allocated_bytes -= b.bytes;
    trace(c, "release", &b, b.bytes);
}
void reserve_devices(Context &c,const fort_buffer_t *handles,size_t count,void **devices) {
    require(!count || handles,FORT_SCOPE_ARGUMENT,"missing task reservation handles");
    if (!count) return;
    const auto budget=field_budget(c);
    std::vector<Buffer *> requested;
    requested.reserve(count);
    std::unordered_set<fort_buffer_t> pinned;
    size_t footprint=0,missing=0,reused=0;
    for (size_t i=0;i<count;++i) {
        auto &b=buffer(c,handles[i]);
        require(!b.prepared,FORT_SCOPE_STATE,"storage reservation during a prepared resource access");
        if (!pinned.insert(handles[i]).second) continue;
        require(b.bytes<=budget-footprint,FORT_SCOPE_RESOURCE,"task device footprint exceeds the declared scope budget");
        footprint+=b.bytes;
        if (b.device) ++reused;
        else missing+=b.bytes; // bounded by the checked unique footprint
        requested.push_back(&b);
    }
    require(c.stats.allocated_bytes<=budget,FORT_SCOPE_STATE,"resident allocations exceed the declared scope budget");
    auto available=budget-static_cast<size_t>(c.stats.allocated_bytes);
    if (missing>available && !c.commands.empty()) {
        // Kernel arguments in accepted commands contain allocation pointers.
        // Submit that queue before any possible victim can be retired.
        flush_commands(c);
        ++c.residency_stats.pressure_flushes;
        available=budget-static_cast<size_t>(c.stats.allocated_bytes);
    }
    std::vector<Buffer *> victims;
    if (missing>available) {
        struct Candidate { Buffer *buffer; bool dirty; };
        std::vector<Candidate> candidates;
        candidates.reserve(c.buffers.size());
        for (auto &entry : c.buffers) {
            auto &b=*entry.second;
            if (b.device && !b.prepared && !pinned.count(b.handle)) {
                // A readonly replica can be dropped without publication.
                // Include physically current bytes even when their logical
                // definition was killed: retirement must preserve storage,
                // and a partial host probe does not make a dirty buffer clean.
                Budget region_budget;
                candidates.push_back({&b,!difference(b.device_current,b.host_current,region_budget).empty()});
            }
        }
        std::sort(candidates.begin(),candidates.end(),[](const Candidate &a,const Candidate &b) {
            if(a.dirty!=b.dirty)return !a.dirty;
            return a.buffer->last_device_use!=b.buffer->last_device_use ?
                a.buffer->last_device_use<b.buffer->last_device_use : a.buffer->handle<b.buffer->handle;
        });
        auto remaining=missing-available;
        for (const auto &candidate : candidates) {
            auto *b=candidate.buffer;
            victims.push_back(b);
            if (b->bytes>=remaining) { remaining=0; break; }
            remaining-=b->bytes;
        }
        require(!remaining,FORT_SCOPE_RESOURCE,"prepared resources pin the remaining device budget");
    }
    if (missing || !victims.empty()) invalidate(c,Change::State);
    const auto downloads=c.stats.download_bytes;
    for (auto *b : victims) {
        const auto before=c.stats.download_bytes;
        queue_publication(c,*b);
        const auto bytes=c.stats.download_bytes-before;
        if (bytes) { ++c.residency_stats.dirty_evictions; c.residency_stats.dirty_download_bytes+=bytes; }
    }
    // Publications borrow caller host memory. Complete the whole batch before
    // returning a host mirror as current or retiring its source allocation.
    if (c.stats.download_bytes!=downloads) wait(c);
    if (!victims.empty()) budget_warning(c,"the least recently used ones now move back to the host");
    for (auto *b : victims) {
        release(c,*b);
        b->device_current.clear();
        ++c.residency_stats.evictions; c.residency_stats.evicted_bytes+=b->bytes;
        trace(c,"evict",b,b->bytes);
    }
    for (auto *b : requested) allocate(c,*b);
    for (auto *b : requested) touch_device(c,*b);
    ++c.residency_stats.reserve_batches;
    c.residency_stats.requested_resources+=requested.size(); c.residency_stats.reuse_hits+=reused;
    if (devices) for (size_t i=0;i<count;++i) devices[i]=buffer(c,handles[i]).device;
}
template<class F> int protect(F &&f) noexcept {
    fort_runtime::HostFloatingEnvironment floating_environment;
    if (!floating_environment.valid()) {
        diagnostic("cannot preserve the caller floating-point environment");
        return FORT_SCOPE_RESOURCE;
    }
    try { f(); return FORT_SCOPE_OK; }
    catch (const Error &error) { diagnostic(error.what()); return error.status; }
    catch (const Fragmented &) { diagnostic("section fragmentation requires a scope boundary"); return FORT_SCOPE_BOUNDARY; }
    catch (const std::bad_alloc &) { diagnostic("scope metadata allocation failed"); return FORT_SCOPE_RESOURCE; }
    catch (const std::exception &error) { diagnostic(error.what()); return FORT_SCOPE_STATE; }
    catch (...) { diagnostic("unknown scope runtime failure"); return FORT_SCOPE_STATE; }
}
void invalidate(Context &c, Change change) noexcept {
    // Queued tasks are simulated on virtual buffer states; a change of the real state makes them stale.
    if (change == Change::State) ++c.virtual_epoch;
}
template<class F> int with(fort_scope_t handle, F &&f, Change change = Change::State) noexcept {
    const int status=protect([&]() {
        const auto c = lookup(handle);
        std::lock_guard<std::mutex> lock(c->mutex);
        require(!c->closed && !c->residency->retired, FORT_SCOPE_STALE, "invalid or stale scope handle");
        require(!c->poisoned, FORT_SCOPE_EXECUTION, "scope has an execution failure; unsafe replay prohibited");
        // Invalidate before a possibly partial mutation, including operations that fail.
        invalidate(*c, change);
        f(*c);
    });
    if(status!=FORT_SCOPE_OK) {
        try {
            const auto c=lookup(handle);std::lock_guard<std::mutex> lock(c->mutex);
            if(c->poisoned)forget_copy_startup(*c);
        } catch(...) {} // Preserve the original failure, including stale handles.
    }
    return status;
}
}
extern "C" void fort_scope_trace_set_v1(fort_scope_t h, uint32_t mask, const char *owner, const char *procedure,
                                          const char *segment, const char *operation, const char *implementation,
                                          const char *boundary, fort_scope_trace_state_v1 *previous) {
    if (previous) previous->version = 0;
    if (!trace_enabled()) return;
    if (mask & ~uint32_t(FORT_SCOPE_TRACE_ALL)) return;
    try {
        const auto c = lookup(h);
        std::lock_guard<std::mutex> lock(c->mutex);
        if (c->closed) return;
        if (previous) { *previous = c->provenance; previous->version = FORT_SCOPE_TRACE_ABI_VERSION; }
        auto &p = c->provenance;
        p.version = FORT_SCOPE_TRACE_ABI_VERSION;
        if (mask & FORT_SCOPE_TRACE_OWNER) trace_id(p.owner, owner);
        if (mask & FORT_SCOPE_TRACE_PROCEDURE) trace_id(p.procedure, procedure);
        if (mask & FORT_SCOPE_TRACE_SEGMENT) trace_id(p.segment, segment);
        if (mask & FORT_SCOPE_TRACE_OPERATION) trace_id(p.operation, operation);
        if (mask & FORT_SCOPE_TRACE_IMPLEMENTATION) trace_id(p.implementation, implementation);
        if (mask & FORT_SCOPE_TRACE_BOUNDARY) trace_id(p.boundary, boundary);
        if (mask) trace(*c, "position");
    } catch (...) {} // Even a stale/foreign diagnostic handle cannot fail work.
}
extern "C" void fort_scope_trace_restore_v1(fort_scope_t h, const fort_scope_trace_state_v1 *previous) {
    if (!trace_enabled()) return;
    if (!previous || previous->version != FORT_SCOPE_TRACE_ABI_VERSION || previous->reserved) return;
    try {
        const auto c = lookup(h);
        std::lock_guard<std::mutex> lock(c->mutex);
        if (c->closed) return;
        auto &p = c->provenance;
        p = {};
        p.version = FORT_SCOPE_TRACE_ABI_VERSION;
        trace_id(p.owner, previous->owner); trace_id(p.procedure, previous->procedure);
        trace_id(p.segment, previous->segment); trace_id(p.operation, previous->operation);
        trace_id(p.implementation, previous->implementation); trace_id(p.boundary, previous->boundary);
        trace(*c, "restore");
    } catch (...) {}
}
extern "C" uint32_t fort_scope_abi_version() { return FORT_SCOPE_ABI_VERSION; }
extern "C" const char *fort_scope_error() { return last_error; }
extern "C" int fort_scope_create(int device, fort_scope_t *out) {
    return protect([&]() {
        require(out && device >= 0, FORT_SCOPE_ARGUMENT, "invalid context output or device");
        auto c = std::make_shared<Context>(device);
        auto handle = token();
        c->handle = handle;
        {
            std::lock_guard<std::mutex> lock(registry_mutex);
            contexts.emplace(handle, c);
        }
        *out = handle;
        trace(*c, "create");
    });
}
extern "C" int fort_scope_domain_create_v1(int device, fort_scope_domain_t *out) {
    return protect([&]() {
        require(out && device>=0,FORT_SCOPE_ARGUMENT,"invalid residency domain output or device");
        auto owner=std::make_shared<Residency>(device);
        owner->domain=token();
        { std::lock_guard<std::mutex> lock(registry_mutex); domains.emplace(owner->domain,owner); }
        *out=owner->domain;
        domain_trace(*owner,"domain_create");
    });
}
extern "C" int fort_scope_create_in_domain_v1(fort_scope_domain_t domain,fort_scope_t *out) {
    return protect([&]() {
        require(out,FORT_SCOPE_ARGUMENT,"missing borrowed context output");
        auto owner=lookup_domain(domain);
        std::lock_guard<std::mutex> lock(owner->mutex);
        require(!owner->retired,FORT_SCOPE_STALE,"residency domain is retired");
        require(!owner->poisoned,FORT_SCOPE_EXECUTION,"residency domain failed; unsafe replay prohibited");
        require(!owner->active_context,FORT_SCOPE_STATE,"residency domain already has an active context");
        auto context=std::make_shared<Context>(owner,true);
        context->handle=token();
        { std::lock_guard<std::mutex> registry(registry_mutex); contexts.emplace(context->handle,context); }
        owner->active_context=context->handle;
        *out=context->handle;
        trace(*context,"create");
        domain_trace(*owner,"domain_attach",context.get());
    });
}
extern "C" int fort_scope_domain_get_v1(fort_scope_t h,fort_scope_domain_t *out) {
    return protect([&]() {
        require(out,FORT_SCOPE_ARGUMENT,"missing residency domain output");
        auto context=lookup(h);
        std::lock_guard<std::mutex> lock(context->mutex);
        require(!context->closed && !context->residency->retired,FORT_SCOPE_STALE,"invalid or stale context");
        *out=context->borrowed_domain ? context->residency->domain : 0;
    });
}
extern "C" int fort_scope_domain_stats_get_v1(fort_scope_domain_t domain,fort_scope_stats *out) {
    return protect([&]() {
        require(out,FORT_SCOPE_ARGUMENT,"missing residency statistics output");
        auto owner=lookup_domain(domain);
        std::lock_guard<std::mutex> lock(owner->mutex);
        require(!owner->retired,FORT_SCOPE_STALE,"retired residency domain");
        *out=owner->stats;
    });
}
extern "C" int fort_scope_domain_device_get_v1(fort_scope_domain_t domain,int *out) {
    return protect([&]() {
        require(out,FORT_SCOPE_ARGUMENT,"missing residency device output");
        auto owner=lookup_domain(domain);
        std::lock_guard<std::mutex> lock(owner->mutex);
        require(!owner->retired,FORT_SCOPE_STALE,"retired residency domain");
        *out=owner->device;
    });
}
namespace {
void check_workspace_capacity(const Residency &owner,size_t bytes) {
    require(bytes<=owner.device_budget && owner.scratch_capacity<=owner.device_budget-bytes &&
            owner.stats.allocated_bytes<=owner.device_budget-bytes-owner.scratch_capacity,
            FORT_SCOPE_RESOURCE,"external workspace exceeds residency budget");
}
void workspace_changed(Residency &owner) {
    const auto total=owner.workspace_bytes+owner.scratch_capacity+owner.stats.allocated_bytes;
    owner.scratch_stats.peak_total_device_bytes=std::max<uint64_t>(
        owner.scratch_stats.peak_total_device_bytes,total);
    if(owner.active_context) {
        const auto context=lookup(owner.active_context);
        invalidate(*context,Change::State);
    }
}
}
extern "C" int fort_scope_domain_workspace_reserve_v1(fort_scope_domain_t domain,size_t bytes,uint64_t *out) {
    return protect([&]() {
        require(out,FORT_SCOPE_ARGUMENT,"missing external workspace lease output");
        auto owner=lookup_domain(domain);
        std::lock_guard<std::mutex> lock(owner->mutex);
        require(!owner->retired,FORT_SCOPE_STALE,"retired residency domain");
        require(!owner->poisoned,FORT_SCOPE_EXECUTION,"residency domain has failed");
        require(bytes<=std::numeric_limits<size_t>::max()-owner->workspace_bytes,FORT_SCOPE_RESOURCE,"workspace byte overflow");
        check_workspace_capacity(*owner,owner->workspace_bytes+bytes);
        const auto lease=token();
        owner->workspaces.emplace(lease,bytes);
        owner->workspace_bytes+=bytes;
        workspace_changed(*owner);
        *out=lease;
    });
}
extern "C" int fort_scope_domain_workspace_resize_v1(fort_scope_domain_t domain,uint64_t lease,size_t bytes) {
    return protect([&]() {
        auto owner=lookup_domain(domain);
        std::lock_guard<std::mutex> lock(owner->mutex);
        require(!owner->retired,FORT_SCOPE_STALE,"retired residency domain");
        require(!owner->poisoned,FORT_SCOPE_EXECUTION,"residency domain has failed");
        auto found=owner->workspaces.find(lease);
        require(found!=owner->workspaces.end(),FORT_SCOPE_STALE,"foreign or stale external workspace lease");
        const auto other=owner->workspace_bytes-found->second;
        require(bytes<=std::numeric_limits<size_t>::max()-other,FORT_SCOPE_RESOURCE,"workspace byte overflow");
        check_workspace_capacity(*owner,other+bytes);
        owner->workspace_bytes=other+bytes; found->second=bytes;
        workspace_changed(*owner);
    });
}
extern "C" int fort_scope_domain_workspace_release_v1(fort_scope_domain_t domain,uint64_t lease) {
    return protect([&]() {
        auto owner=lookup_domain(domain);
        std::lock_guard<std::mutex> lock(owner->mutex);
        require(!owner->retired,FORT_SCOPE_STALE,"retired residency domain");
        auto found=owner->workspaces.find(lease);
        require(found!=owner->workspaces.end(),FORT_SCOPE_STALE,"foreign or stale external workspace lease");
        owner->workspace_bytes-=found->second; owner->workspaces.erase(found);
        workspace_changed(*owner);
    });
}
extern "C" int fort_scope_numerical_environment_supported(void) {
    return fort_runtime::numerical_environment_supported() ? 1 : 0;
}
extern "C" int fort_scope_set_device_budget(fort_scope_t h, size_t bytes) {
    return with(h, [&](Context &c) {
        if(c.borrowed_domain && c.ready) {
            require(bytes==c.device_budget,FORT_SCOPE_STATE,"cannot change a live residency budget"); return;
        }
        require(!c.ready && bytes>=c.residency->workspace_bytes,FORT_SCOPE_STATE,
                "device budget must precede initialization and cover external workspaces");
        c.device_budget = bytes;
    });
}
extern "C" int fort_scope_device_budget_default_v1(int device, size_t *out) {
    return protect([&]() {
        require(out && device>=0,FORT_SCOPE_ARGUMENT,"invalid device or missing budget output");
        const char *stated=std::getenv("FORT_RESIDENT_DEVICE_BUDGET");
        if(stated && stated[0]) {
            // A count of bytes, or a count followed by K, M or G (powers of 1024).
            char *end=nullptr; errno=0;
            const unsigned long long count=std::strtoull(stated,&end,10);
            unsigned shift=0;
            if(end && (*end=='K' || *end=='k')) { shift=10; ++end; }
            else if(end && (*end=='M' || *end=='m')) { shift=20; ++end; }
            else if(end && (*end=='G' || *end=='g')) { shift=30; ++end; }
            require(end!=stated && !*end && !errno && count && stated[0]!='-' &&
                    count<=(std::numeric_limits<size_t>::max()>>shift),FORT_SCOPE_ARGUMENT,
                    "FORT_RESIDENT_DEVICE_BUDGET must be a positive count of bytes, optionally with K, M or G");
            *out=size_t(count)<<shift;
            return;
        }
#ifdef FORT_SCOPE_CPU_TEST
        *out=std::numeric_limits<size_t>::max();
#else
        int previous=0;
        size_t free_bytes=0,total_bytes=0;
        auto failed=[](cudaError_t status) {
            if(status!=cudaSuccess) throw Error(FORT_SCOPE_RESOURCE,cudaGetErrorString(status));
        };
        failed(cudaGetDevice(&previous));
        if(previous!=device) failed(cudaSetDevice(device));
        const auto status=cudaMemGetInfo(&free_bytes,&total_bytes);
        if(previous!=device) failed(cudaSetDevice(previous));
        failed(status);
        // The driver, CUDA graphs and library plans allocate outside the
        // budget: leave them a sixteenth of the device, at least 512 MiB.
        const size_t reserve=std::max<size_t>(total_bytes/16,size_t(512)<<20);
        require(free_bytes>reserve,FORT_SCOPE_RESOURCE,"the device has no free memory for a residency domain");
        *out=free_bytes-reserve;
#endif
    });
}
extern "C" int fort_scope_domain_budget_get_v1(fort_scope_domain_t domain,size_t *budget,size_t *unclaimed) {
    return protect([&]() {
        require(budget && unclaimed,FORT_SCOPE_ARGUMENT,"missing residency budget output");
        auto owner=lookup_domain(domain);
        std::lock_guard<std::mutex> lock(owner->mutex);
        require(!owner->retired,FORT_SCOPE_STALE,"retired residency domain");
        *budget=owner->device_budget;
        const size_t claimed=owner->workspace_bytes+owner->scratch_capacity+owner->stats.allocated_bytes;
        *unclaimed=claimed<owner->device_budget ? owner->device_budget-claimed : 0;
    });
}
extern "C" int fort_scope_set_transfers(fort_scope_t h, uint32_t mode) {
    return with(h, [&](Context &c) {
        require(mode <= FORT_SCOPE_TRANSFERS_COALESCED, FORT_SCOPE_ARGUMENT, "invalid scoped transfer mode");
        if(c.borrowed_domain && (!c.buffers.empty() || c.ready)) {
            require(mode==c.transfer_stats.requested_mode,
                    FORT_SCOPE_STATE,"cannot change a live residency transfer mode"); return;
        }
        require(c.buffers.empty() && !c.ready,
                FORT_SCOPE_STATE, "configure scoped transfers before registration");
        auto &stats = c.transfer_stats;
        stats.requested_mode = mode;
        stats.effective_mode = mode == FORT_SCOPE_TRANSFERS_PINNED ? mode : uint32_t(FORT_SCOPE_TRANSFERS_DIRECT);
        stats.fallback_reason = FORT_SCOPE_TRANSFER_NONE;
        if (mode == FORT_SCOPE_TRANSFERS_AUTO) transfer_fallback(c, FORT_SCOPE_TRANSFER_ESTIMATES_UNAVAILABLE);
        if (mode == FORT_SCOPE_TRANSFERS_PIPELINED) transfer_fallback(c, FORT_SCOPE_TRANSFER_PIPELINED_UNAVAILABLE);
    }, Change::None);
}
extern "C" int fort_scope_transfer_stats_get_v1(fort_scope_t h, fort_scope_transfer_stats *out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing scoped transfer statistics output");
        *out = transfer_statistics(c);
    }, Change::None);
}
extern "C" int fort_scope_set_copy_costs_v3(fort_scope_t h,const fort_scope_copy_costs_v3 *costs,
        const fort_scope_plan_costs *base,int compatible) {
    return with(h,[&](Context &c) {
        require(compatible==0 || compatible==1,FORT_SCOPE_ARGUMENT,"invalid physical copy calibration compatibility");
        require(c.borrowed_domain || (c.buffers.empty() && !c.ready),
                FORT_SCOPE_STATE,"configure physical copy calibration before registration");
        c.copy_costs.reset();
        if(compatible && costs)require(costs->version==FORT_SCOPE_COPY_COST_ABI_VERSION,
            FORT_SCOPE_ARGUMENT,"unsupported physical-copy cost ABI; expected version3");
        if(!compatible || !costs || !base || !fort_physical::compact::valid_costs(*costs) ||
           !fort_physical::compact::valid_base(*base))return;
        c.copy_costs=fort_physical::compact::Pricing{*costs,*base};
        if(c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_AUTO) {
            c.transfer_stats.fallback_reason=FORT_SCOPE_TRANSFER_NONE;c.transfer_stats.fallbacks=0;
        }
    });
}
static int register_buffer(fort_scope_t h, uint64_t identity, uint64_t generation,
                           const fort_scope_layout *layout, bool full_initialized,
                           const fort_scope_section *defined, size_t defined_count, fort_buffer_t *out) {
    return with(h, [&](Context &c) {
        require(layout && out && identity && generation && layout->rank && layout->rank <= rank_limit &&
                layout->extents && layout->lower_bounds &&
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
        b->handle = token();
        const auto handle = b->handle;
        c.buffers.emplace(handle, std::move(b));
        try { c.identities.emplace(identity, handle); }
        catch (...) { c.buffers.erase(handle); throw; }
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
extern "C" int fort_scope_identity_swap_v1(fort_scope_t h,fort_buffer_t first,fort_buffer_t second,
                                            const fort_scope_layout *first_after,const fort_scope_layout *second_after,
                                            fort_buffer_t *first_current,fort_buffer_t *second_current) {
    return with(h,[&](Context &c) {
        // The original descriptor mutation has already happened. No rejection
        // from this point can authorize replay or publication through old names.
        try {
            require(first!=second && first_current && second_current && first_current!=second_current,
                    FORT_SCOPE_ARGUMENT,"identity exchange needs distinct physical handles and output slots");
            auto &a=buffer(c,first), &b=buffer(c,second);
            require(!a.prepared && !b.prepared,FORT_SCOPE_STATE,"identity exchange during prepared access");
            const auto ai=c.identities.find(a.identity),bi=c.identities.find(b.identity);
            require(a.identity!=b.identity && ai!=c.identities.end() && bi!=c.identities.end() &&
                    ai->second==first && bi->second==second,FORT_SCOPE_STALE,"identity exchange has stale canonical slots");
            auto matches=[](const fort_scope_layout *after,const Buffer &physical) {
                if(!after || after->rank!=physical.extents.size() || !after->extents || !after->lower_bounds ||
                   after->host!=physical.host || after->type!=physical.type ||
                   after->element_bytes!=physical.element_bytes || after->generation!=physical.generation) return false;
                return std::equal(physical.extents.begin(),physical.extents.end(),after->extents) &&
                       std::equal(physical.lower.begin(),physical.lower.end(),after->lower_bounds);
            };
            require(matches(first_after,b) && matches(second_after,a),FORT_SCOPE_STALE,
                    "post-exchange descriptor does not match opposite physical allocation");
            // Existing map slots and scalar fields commit without allocation.
            // Queue bindings/upload readers still refer to the same Buffer and
            // physical handle; no pending read or numerical work is replayed.
            const IdentitySwapTrace exchange{a.identity,b.identity,first,second,a.generation,b.generation};
            ai->second=second; bi->second=first;
            std::swap(a.identity,b.identity);
            *first_current=second; *second_current=first;
            trace(c,"identity_swap",nullptr,0,&exchange);
        } catch(Error &error) {
            c.poisoned=true;
            error.status=FORT_SCOPE_EXECUTION;
            throw; // Preserve the exact failed validation without reallocating.
        } catch(...) {
            c.poisoned=true;
            throw Error(FORT_SCOPE_EXECUTION,"post-mutation identity exchange failed; unsafe replay prohibited");
        }
    });
}
extern "C" int fort_scope_admit_source_reads_v1(fort_scope_t h, fort_buffer_t handle,
                                                const fort_scope_section *items, size_t count) {
    return with(h, [&](Context &c) {
        auto &b = buffer(c, handle);
        require(!b.prepared, FORT_SCOPE_STATE, "source read admission during a prepared buffer access");
        const auto reads = sections(b, items, count, false);
        Budget budget;
        if (difference(reads,b.initialized,budget).empty()) return;
        auto admitted = fort_scoped::coherence::admit_source_reads(b, reads);
        invalidate(c,Change::State);
        b.initialized = std::move(admitted.initialized);
        b.host_current = std::move(admitted.host_current);
        trace(c, "admit_source_reads", &b);
    },Change::None);
}
extern "C" int fort_scope_forget_definition(fort_scope_t h, fort_buffer_t handle) {
    return with(h, [&](Context &c) {
        drain_resources(c,&handle,1);
        auto &b = buffer(c, handle);
        require(!b.prepared, FORT_SCOPE_STATE, "definition change during a prepared buffer access");
        b.initialized.clear();
        trace(c, "forget_definition", &b);
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
        require(a, FORT_SCOPE_ARGUMENT, "invalid access descriptor");
        const fort_scope_plan_binding binding{b, *a};
        begin_accesses(c, &binding, 1, false, nullptr);
    });
}
extern "C" int fort_scope_host_end(fort_scope_t h, fort_buffer_t b) {
    return with(h, [&](Context &c) { end(c, buffer(c,b), false); });
}
extern "C" int fort_scope_device_begin(fort_scope_t h, fort_buffer_t b, const fort_scope_access *a, void **out) {
    return with(h, [&](Context &c) {
        require(out && a, FORT_SCOPE_ARGUMENT, "missing device pointer output or access descriptor");
        const fort_scope_plan_binding binding{b, *a};
        begin_accesses(c, &binding, 1, true, out);
    });
}
extern "C" int fort_scope_access_begin_v1(fort_scope_t h, const fort_scope_plan_binding *bindings,
                                           size_t count, int device, void **devices) {
    return with(h, [&](Context &c) {
        require(device == 0 || device == 1, FORT_SCOPE_ARGUMENT, "invalid task access execution mode");
        try { begin_accesses(c, bindings, count, device != 0, devices); }
        catch (const Error &error) {
            if (device && error.status==FORT_SCOPE_RESOURCE)
                budget_warning(c,"loop nests whose arrays do not fit together run on the host");
            throw;
        }
    });
}
extern "C" int fort_scope_original_host_begin_v1(fort_scope_t h, const fort_scope_plan_binding *bindings,
                                                  size_t count) {
    return with(h, [&](Context &c) {
        begin_accesses(c, bindings, count, false, nullptr, AccessContract::OriginalHost);
    });
}
extern "C" int fort_scope_external_execute_v1(fort_scope_t h,const fort_scope_external_call_v1 *call,
                                               fort_scope_external_worker_v1 worker,void *user,
                                               fort_scope_external_report_v1 *out) {
    if(out) { *out={}; out->version=FORT_SCOPE_EXTERNAL_ABI_VERSION; }
    const int status=with(h,[&](Context &c) {
        require(call && out && call->version==FORT_SCOPE_EXTERNAL_ABI_VERSION && !call->reserved &&
                (!call->binding_count || (call->bindings && worker)),
                FORT_SCOPE_ARGUMENT,"invalid external transaction");
        if(!call->binding_count) return;
        if(call->workspace_lease) {
            const auto found=c.residency->workspaces.find(call->workspace_lease);
            require(found!=c.residency->workspaces.end(),FORT_SCOPE_STALE,"foreign or stale external workspace lease");
            require(call->workspace_bytes<=found->second,FORT_SCOPE_RESOURCE,"external workspace claim is too small");
        } else require(!call->workspace_bytes,FORT_SCOPE_ARGUMENT,"external workspace requires a residency lease");
        std::vector<fort_scope_plan_binding> bindings;
        std::vector<fort_scope_external_view_v1> views;
        std::vector<void *> devices(call->binding_count);
        std::unordered_set<fort_buffer_t> seen;
        bindings.reserve(call->binding_count); views.reserve(call->binding_count);
        for(size_t i=0;i<call->binding_count;++i) {
            const auto &binding=call->bindings[i]; auto &b=buffer(c,binding.buffer);
            require(seen.insert(binding.buffer).second,FORT_SCOPE_ALIAS,"external transaction repeats a canonical resource");
            require(binding.type==b.type && binding.rank==b.extents.size() && binding.element_bytes==b.element_bytes,
                    FORT_SCOPE_ARGUMENT,"external transaction type or rank mismatch");
            require(!b.prepared,FORT_SCOPE_STATE,"external transaction overlaps a prepared access");
            bindings.push_back({binding.buffer,binding.access});
            views.push_back({b.handle,b.generation,b.type,static_cast<uint32_t>(b.extents.size()),b.element_bytes,b.bytes,
                             b.extents.data(),b.strides.data(),b.lower.data(),nullptr});
        }
        bool submission_uncertain=false;
        try {
            begin_accesses(c,bindings.data(),bindings.size(),true,devices.data());
            initialize(c);
            for(size_t i=0;i<views.size();++i) views[i].device=devices[i];
            int started=-1, result=FORT_SCOPE_EXECUTION;
            try {
#ifndef FORT_SCOPE_CPU_TEST
                DeviceGuard guard(c);
                void *stream=reinterpret_cast<void *>(c.stream);
#else
                void *stream=nullptr;
#endif
                result=worker(user,views.data(),views.size(),stream,&started);
            } catch(...) {
                submission_uncertain=true;
                out->started=1; c.poisoned=true;
                throw Error(FORT_SCOPE_EXECUTION,"external callback exception; unsafe replay prohibited");
            }
            submission_uncertain=started!=0;
            out->started=started==0 ? 0u : 1u;
            require(!c.poisoned,FORT_SCOPE_EXECUTION,"external device restoration failed");
            if((started!=0 && started!=1) || (result==FORT_SCOPE_OK && started!=1) ||
                    (result!=FORT_SCOPE_OK && (started || result==FORT_SCOPE_EXECUTION ||
                                              result<FORT_SCOPE_ARGUMENT || result>FORT_SCOPE_STATE))) {
                c.poisoned=true;
                throw Error(FORT_SCOPE_EXECUTION,"external execution failed or submission state missing; unsafe replay prohibited");
            }
            if(result!=FORT_SCOPE_OK) throw Error(result,"external callback rejected before submission");
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
            if(std::getenv("FORT_SCOPE_TEST_FAIL_EXTERNAL_COMMIT")) throw std::bad_alloc();
#endif
            mark_context_pending(c);
            for(const auto &binding : bindings) end(c,buffer(c,binding.buffer),true);
            out->committed=1;
            trace(c,"external_commit",nullptr,bindings.size());
        } catch(...) {
            if(submission_uncertain) c.poisoned=true;
            for(const auto &binding : bindings) {
                const auto found=c.buffers.find(binding.buffer);
                if(found!=c.buffers.end()) found->second->prepared.reset();
            }
            if(submission_uncertain) {
                out->started=1;
                throw Error(FORT_SCOPE_EXECUTION,"external submission or commit failed; unsafe replay prohibited");
            }
            throw;
        }
    });
    if(out) out->status=status;
    return status;
}
static void observe_resources(Context &c, const fort_buffer_t *handles, size_t count) {
    require(!count || handles, FORT_SCOPE_ARGUMENT, "missing observed buffers");
    drain_resources(c,handles,count);
    std::vector<Region> physical(count);
    std::vector<std::vector<fort_scope_section>> regions(count);
    std::vector<fort_scope_plan_binding> bindings(count);
    for (size_t i=0; i<count; ++i) {
        auto &b = buffer(c, handles[i]);
        // Original entry expressions may observe newer physical bytes whose
        // logical definition is unknown or was killed. Publication must not
        // lose those versions, and does not turn them into value-read proofs.
        physical[i] = physical_values(b);
        for (const auto &part : physical[i])
            regions[i].push_back({part.lo.data(),part.hi.data()});
        bindings[i] = {handles[i],{0,regions[i].size(),regions[i].data(),0,nullptr,0,nullptr}};
    }
    begin_accesses(c,bindings.data(),count,false,nullptr,AccessContract::OriginalHost);
    for (size_t i=0; i<count; ++i) end(c,buffer(c,handles[i]),false);
    trace(c,"observe",nullptr,count);
}
extern "C" int fort_scope_observe_v1(fort_scope_t h, const fort_buffer_t *handles, size_t count) {
    return with(h, [&](Context &c) { observe_resources(c,handles,count); });
}
extern "C" int fort_scope_observe_all_v1(fort_scope_t h) {
    return with(h, [&](Context &c) {
        // Original specification expressions run before the callee body can
        // acquire anything. Enumerate the live domain, never source descriptors
        // or variable names; hidden aliases therefore see the current value.
        std::vector<fort_buffer_t> handles;
        handles.reserve(c.buffers.size());
        for(const auto &entry:c.buffers) handles.push_back(entry.first);
        std::sort(handles.begin(),handles.end());
        observe_resources(c,handles.data(),handles.size());
        trace(c,"observe_all",nullptr,handles.size());
    });
}
extern "C" int fort_scope_discard_v1(fort_scope_t h, fort_buffer_t handle) {
    return with(h, [&](Context &c) {
        drain_resources(c,&handle,1);
        auto &b = buffer(c,handle);
        require(!b.prepared, FORT_SCOPE_STATE, "discard during a prepared buffer access");
        // Storage remains live and future kernels share the stream. Forgetting
        // dead values needs neither publication nor completion of old writes.
        b.initialized.clear(); b.host_current.clear(); b.device_current.clear();
        trace(c,"discard",&b);
    });
}
// The owner is about to define the buffer again and its values are dead.
// Forgetting them has to follow every queued user, which would submit the
// queue at each reuse. A buffer that queued work still touches keeps its
// values instead: the new definition replaces them in queue order, and the
// only thing given up is the check that nothing reads them first.
extern "C" int fort_scope_recycle_v1(fort_scope_t h, fort_buffer_t handle) {
    return with(h, [&](Context &c) {
        auto &b = buffer(c,handle);
        require(!b.prepared, FORT_SCOPE_STATE, "recycle during a prepared buffer access");
        if (virtual_state(c,handle)) { trace(c,"recycle_queued",&b); return; }
        b.initialized.clear(); b.host_current.clear(); b.device_current.clear();
        trace(c,"discard",&b);
    });
}
extern "C" int fort_scope_retire_v1(fort_scope_t h, fort_buffer_t handle) {
    return with(h, [&](Context &c) {
        drain_resources(c,&handle,1);
        auto &b = buffer(c,handle);
        require(!b.prepared, FORT_SCOPE_STATE, "retirement during a prepared buffer access");
        // No value is observable, but returning borrowed host storage while an
        // earlier upload still reads it would violate its lifetime contract.
        release(c,b);
        wait(c);
        trace(c,"retire",&b);
        c.identities.erase(b.identity); c.buffers.erase(handle);
    });
}
extern "C" int fort_scope_device_end(fort_scope_t h, fort_buffer_t b) {
    return with(h, [&](Context &c) { end(c, buffer(c,b), true); if (c.ready) mark_context_pending(c); });
}
extern "C" int fort_scope_device_reserve_v1(fort_scope_t h, fort_buffer_t handle, void **out) {
    return fort_scope_device_reserve_many_v1(h,&handle,1,out);
}
extern "C" int fort_scope_device_reserve_many_v1(fort_scope_t h,const fort_buffer_t *handles,size_t count,void **out) {
    return with(h, [&](Context &c) {
        require(!count || out,FORT_SCOPE_ARGUMENT,"missing reserved device pointer outputs");
        reserve_devices(c,handles,count,out);
    },Change::None);
}
extern "C" int fort_scope_residency_stats_get_v1(fort_scope_t h,fort_scope_residency_stats_v1 *out) {
    return with(h,[&](Context &c) {
        require(out,FORT_SCOPE_ARGUMENT,"missing residency statistics output");
        *out=c.residency_stats;
    },Change::None);
}
extern "C" int fort_scope_buffer_valid_v1(fort_scope_t h,fort_buffer_t handle,int *out) {
    return with(h,[&](Context &c) {
        require(out,FORT_SCOPE_ARGUMENT,"missing buffer membership output");
        *out=c.buffers.count(handle)!=0;
    },Change::None);
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
        ++c.stats.launches; mark_context_pending(c); trace(c, "launch");
    });
}
extern "C" int fort_scope_graph_submit_v1(fort_scope_t h, uint64_t key,
                                           const fort_scope_kernel_v1 *kernels, size_t count) {
    return with(h, [&](Context &c) {
        require(!count || key, FORT_SCOPE_ARGUMENT, "nonempty kernel group requires a stable nonzero key");
        const char *invalid = fort_scoped::graphs::invalid(kernels,count);
        require(!invalid, FORT_SCOPE_ARGUMENT, invalid ? invalid : "invalid kernel group");
        if (!count) return;
        flush_commands(c);
#ifdef FORT_SCOPE_CPU_TEST
        (void)c;
        throw Error(FORT_SCOPE_RESOURCE,"CUDA graphs unavailable in the CPU reference backend");
#else
        submit_graph(c,key,kernels,count);
#endif
    });
}
extern "C" int fort_scope_graph_execute_v1(fort_scope_t h, uint64_t key,
                                            const fort_scope_graph_task_v1 *tasks, size_t count) {
    return with(h, [&](Context &c) {
        if (count) flush_commands(c);
        execute_graph(c,key,tasks,count);
    });
}
extern "C" int fort_scope_graph_enqueue_v1(fort_scope_t h,uint64_t key,
                                            const fort_scope_graph_task_v1 *task,const size_t *sizes) {
    return with(h,[&](Context &c) {
        require(task,FORT_SCOPE_ARGUMENT,"missing queued graph task or code key");
        const fort_scope_graph_member_v1 member{task->bindings,task->binding_count};
        enqueue_graph(c,key,task->kernel,sizes,&member,1);
    },Change::None);
}
extern "C" int fort_scope_graph_enqueue_members_v1(fort_scope_t h,uint64_t key,const fort_scope_kernel_v1 *kernel,
                                                    const size_t *sizes,const fort_scope_graph_member_v1 *members,
                                                    size_t member_count) {
    return with(h,[&](Context &c) {
        require(kernel,FORT_SCOPE_ARGUMENT,"missing queued graph task or code key");
        enqueue_graph(c,key,*kernel,sizes,members,member_count);
    },Change::None);
}
extern "C" int fort_scope_graph_flush_v1(fort_scope_t h) {
    return with(h,[&](Context &c) { flush_commands(c); });
}
extern "C" int fort_scope_graph_queue_cancel_v1(fort_scope_t h) {
    return with(h,[&](Context &c) {
        c.commands.clear(); c.virtual_buffers.clear(); ++c.virtual_epoch; c.queue_stats.pending_tasks=0;
        c.queued_work=0;
        trace(c,"queue_cancel");
    },Change::None);
}
extern "C" int fort_scope_graph_queue_stats_get_v1(fort_scope_t h,fort_scope_graph_queue_stats_v1 *out) {
    return with(h,[&](Context &c) {
        require(out,FORT_SCOPE_ARGUMENT,"missing queued graph statistics output"); *out=c.queue_stats;
    },Change::None);
}
extern "C" int fort_scope_graph_stats_get_v1(fort_scope_t h, fort_scope_graph_stats_v1 *out) {
    return with(h, [&](Context &c) {
        require(out,FORT_SCOPE_ARGUMENT,"missing CUDA graph statistics output");
        *out = c.graph_stats;
    },Change::None);
}
extern "C" int fort_scope_completion_stats_get_v1(fort_scope_t h, fort_scope_completion_stats_v1 *out) {
    return with(h, [&](Context &c) {
        require(out,FORT_SCOPE_ARGUMENT,"missing completion statistics output");
        *out = c.completion_stats;
    },Change::None);
}
extern "C" int fort_scope_execution_error(fort_scope_t h, const char *message) {
    return protect([&]() {
        const auto c = lookup(h);
        std::lock_guard<std::mutex> lock(c->mutex);
        require(!c->closed, FORT_SCOPE_STALE, "invalid or stale scope handle");
        const char *reason = c->poisoned && last_error[0] ? last_error : message;
        invalidate(*c, Change::State);
        c->poisoned = true;
        throw Error(FORT_SCOPE_EXECUTION, reason ? reason : "scoped numerical execution failed");
    });
}
extern "C" int fort_scope_report_error(int status, const char *message) {
    if (status != FORT_SCOPE_OK) diagnostic(message ? message : "scoped entry error");
    return status;
}
extern "C" int fort_scope_wait(fort_scope_t h) {
    return with(h, [&](Context &c) { flush_commands(c); wait(c); });
}
extern "C" int fort_scope_stats_get(fort_scope_t h, fort_scope_stats *out) {
    return with(h, [&](Context &c) { require(out, FORT_SCOPE_ARGUMENT, "missing stats output"); *out = c.stats; }, Change::None);
}
extern "C" int fort_scope_scratch_acquire_v1(fort_scope_t h, size_t bytes, fort_scope_scratch_lease_v1 *out) {
    if (out) *out = {};
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing scratch lease output");
        require(!c.scratch_token, FORT_SCOPE_STATE, "only one scratch lease may be active");
        // Reserve a globally unique token before allocating. A failed request
        // never exposes a pointer or an active lease, even after arena growth.
        const auto lease_token = token();
        if (bytes) {
            if (bytes > c.scratch_capacity) grow_scratch(c, bytes);
            else ++c.scratch_stats.reuses;
        }
        c.scratch_token = lease_token;
        ++c.scratch_stats.acquisitions;
        c.scratch_stats.active = 1;
        c.scratch_stats.active_bytes = bytes;
        *out = {FORT_SCOPE_SCRATCH_ABI_VERSION, 0, lease_token,
                bytes ? c.scratch : nullptr, bytes, c.scratch_capacity};
        trace(c, "scratch_acquire", nullptr, bytes);
    });
}
extern "C" int fort_scope_scratch_release_v1(fort_scope_t h, uint64_t lease_token) {
    return with(h, [&](Context &c) {
        require(lease_token && c.scratch_token == lease_token, FORT_SCOPE_STALE,
                "invalid, foreign, or stale scratch lease");
        flush_commands(c); // A later lease may reuse or grow this same arena.
        c.scratch_token = 0;
        c.scratch_stats.active = 0;
        c.scratch_stats.active_bytes = 0;
        ++c.scratch_stats.releases;
        trace(c, "scratch_release");
    });
}
extern "C" int fort_scope_scratch_stats_get_v1(fort_scope_t h, fort_scope_scratch_stats_v1 *out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing scratch stats output");
        *out = c.scratch_stats;
    }, Change::None);
}
extern "C" int fort_scope_unregister(fort_scope_t h, fort_buffer_t handle) {
    return with(h, [&](Context &c) {
        drain_resources(c,&handle,1);
        auto &b = buffer(c,handle); publish(c,b); release(c,b);
        c.identities.erase(b.identity); c.buffers.erase(handle);
    });
}
namespace {
void close_storage(Context &c) {
        bool ordered_release = false;
#ifndef FORT_SCOPE_CPU_TEST
#if CUDART_VERSION >= 11020
        ordered_release = c.pool != nullptr;
#endif
#endif
        // Close has no intervening host consumer. Queue every publication and
        // retirement on the same stream before its one completion wait. Keep
        // immediate publication for the synchronous allocator/reference path.
        for (auto &entry : c.buffers) {
            if (ordered_release) queue_publication(c, *entry.second);
            else publish(c, *entry.second);
        }
        for (auto &entry : c.buffers) release(c, *entry.second);
        // Complete scratch uses before retiring the idle arena. Scratch is
        // never registered, defined, reconciled, or published to host storage.
        if (ordered_release) release_scratch(c);
        wait(c);
        if (!ordered_release) { release_scratch(c); wait(c); }
#ifndef FORT_SCOPE_CPU_TEST
        if (c.ready) {
            DeviceGuard guard(c);
            for (auto &event : c.completion_events) {
                if (event) cuda_check(c,cudaEventDestroy(event));
                event = nullptr;
            }
            c.completion_events.clear(); c.available_completion_events.clear();
            fort_scoped::cuda_resources::Resources resources{c.stream, c.pool};
            const auto status = fort_scoped::cuda_resources::retire(c.device, resources);
            c.stream = resources.stream; c.pool = resources.pool;
            cuda_check(c, status);
            c.ready = false;
        }
#endif
        c.buffers.clear(); c.identities.clear(); c.closed = true;
        trace(c, "close");
        transfer_statistics_trace(c, c.handle);
        if(!c.residency->domain) graph_statistics_trace(c,true);
        completion_statistics_trace(c,true);
        if(!c.residency->domain) queue_statistics_trace(c,true);
        residency_statistics_trace(c,true);
        c.residency->retired=true;
}
}
extern "C" int fort_scope_close(fort_scope_t h) {
    const auto status = with(h, [&](Context &c) {
        require(!c.scratch_token, FORT_SCOPE_STATE, "scope closed with an active scratch lease");
        for(const auto &entry:c.buffers) require(!entry.second->prepared,FORT_SCOPE_STATE,"unfinished buffer access at context close");
        flush_commands(c);
#ifndef FORT_SCOPE_CPU_TEST
        if (c.graph_stats.live_graphs) {
            DeviceGuard guard(c);
            cuda_check(c,c.graphs.clear(c.graph_stats));
        }
#endif
        if(c.borrowed_domain) {
            c.residency->last_context=c.handle; c.residency->last_provenance=c.provenance;
            c.closed=true; c.residency->active_context=0;
            trace(c,"detach");
            domain_trace(*c.residency,"domain_detach",&c);
            graph_statistics_trace(c,true); queue_statistics_trace(c,true);
            return;
        }
        close_storage(c);
    });
    if (status == FORT_SCOPE_OK) {
        std::lock_guard<std::mutex> lock(registry_mutex); contexts.erase(h);
    }
    return status;
}
namespace {
void abandon_storage(Context &c) {
    forget_copy_startup(c);
#ifdef FORT_SCOPE_CPU_TEST
        for (auto &entry : c.buffers) std::free(entry.second->device);
        std::free(c.scratch);
#else
        // A CUDA execution failure invalidates results. Cleanup is best effort;
        // CUDA can reject further operations after a fatal device fault.
        int previous = 0;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        if (cudaSetDevice(c.device) == cudaSuccess && c.ready) {
            (void)cudaStreamSynchronize(c.stream);
            (void)c.graphs.clear(c.graph_stats);
            for (auto event : c.completion_events) if (event) (void)cudaEventDestroy(event);
            c.completion_events.clear(); c.available_completion_events.clear();
            for (auto &entry : c.buffers) {
                if (entry.second->device) (void)cudaFree(entry.second->device);
            }
            if (c.scratch) (void)cudaFree(c.scratch);
            if (c.pool) (void)cudaMemPoolDestroy(c.pool);
            (void)cudaStreamDestroy(c.stream);
        }
        if (restore) (void)cudaSetDevice(previous);
#endif
        c.scratch = nullptr; c.scratch_capacity = 0; c.scratch_token = 0;
        c.buffers.clear(); c.identities.clear(); c.closed = true;
        c.residency->workspaces.clear(); c.residency->workspace_bytes=0; c.residency->retired=true;

}
}
extern "C" int fort_scope_abandon(fort_scope_t h) {
    return protect([&]() {
        const auto c = lookup(h);
        std::lock_guard<std::mutex> lock(c->mutex);
        require(!c->closed, FORT_SCOPE_STALE, "invalid or stale scope handle");
        require(c->poisoned, FORT_SCOPE_STATE, "only a failed scope can be abandoned");
        forget_copy_startup(*c);
        if(c->borrowed_domain) {
#ifndef FORT_SCOPE_CPU_TEST
            int previous=0;
            const bool restore=cudaGetDevice(&previous)==cudaSuccess;
            if(cudaSetDevice(c->device)==cudaSuccess) (void)c->graphs.clear(c->graph_stats);
            if(restore) (void)cudaSetDevice(previous);
#endif
            for(auto &entry:c->buffers) entry.second->prepared.reset();
            c->commands.clear(); c->virtual_buffers.clear();
            c->residency->last_context=c->handle; c->residency->last_provenance=c->provenance;
            c->closed=true; c->residency->active_context=0;
            domain_trace(*c->residency,"domain_detach",c.get());
        } else abandon_storage(*c);
        graph_statistics_trace(*c,false);
        queue_statistics_trace(*c,false);
        if (!c->borrowed_domain) {
            completion_statistics_trace(*c,false);
            residency_statistics_trace(*c,false);
        }
        std::lock_guard<std::mutex> registry_lock(registry_mutex); contexts.erase(h);
    });
}
extern "C" int fort_scope_domain_close_v1(fort_scope_domain_t domain) {
    const auto status=protect([&]() {
        auto owner=lookup_domain(domain);
        std::lock_guard<std::mutex> lock(owner->mutex);
        require(!owner->retired,FORT_SCOPE_STALE,"residency domain is retired");
        require(!owner->poisoned,FORT_SCOPE_EXECUTION,"failed domain cannot publish; abandon required");
        require(!owner->active_context,FORT_SCOPE_STATE,"residency domain still has an active context");
        require(owner->workspaces.empty() && !owner->scratch_token,FORT_SCOPE_STATE,"residency domain has outstanding workspace or scratch lease");
        Context observer(owner,false); observer.handle=domain;
        observer.provenance=owner->last_provenance;
        close_storage(observer);
        domain_trace(*owner,"domain_retire");
    });
    if(status==FORT_SCOPE_OK) { std::lock_guard<std::mutex> lock(registry_mutex); domains.erase(domain); }
    return status;
}
extern "C" int fort_scope_domain_abandon_v1(fort_scope_domain_t domain) {
    const auto status=protect([&]() {
        auto owner=lookup_domain(domain);
        std::lock_guard<std::mutex> lock(owner->mutex);
        require(!owner->retired,FORT_SCOPE_STALE,"residency domain is retired");
        require(!owner->active_context,FORT_SCOPE_STATE,"abandon failed context before its residency domain");
        require(owner->poisoned,FORT_SCOPE_STATE,"only a failed residency domain can be abandoned");
        require(owner->workspaces.empty(),FORT_SCOPE_STATE,"destroy provider and release its workspace leases before domain abandonment");
        Context observer(owner,false); observer.handle=domain;
        observer.provenance=owner->last_provenance;
        abandon_storage(observer);
        completion_statistics_trace(observer,false); residency_statistics_trace(observer,false);
        domain_trace(*owner,"domain_abandon");
    });
    if(status==FORT_SCOPE_OK) { std::lock_guard<std::mutex> lock(registry_mutex); domains.erase(domain); }
    return status;
}
