/* Shared state implementation: compile once with nvcc, or with
 * -DFORT_SCOPE_CPU_TEST for the isolated coherence reference backend. */
#include "scoped_runtime.h"
#include "section_copy.hpp"
#include "section_batch_cost.hpp"
#include "scoped_regions.hpp"
#include "scoped_planning.hpp"
#include "scoped_source_quote.hpp"
#include "scoped_graphs.hpp"
#include "scoped_host_metadata.hpp"
#include "floating_environment.hpp"
#define FORT_SCOPE_TEAM_OBSERVER_IMPLEMENTATION
#include "scoped_team_observer.hpp"
#include <algorithm>
#include <array>
#include <chrono>
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
#include "scoped_memo.hpp"
#ifdef __linux__
#include <sched.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__GLIBC__) && (defined(__i386__) || defined(__x86_64__))
#include <fpu_control.h>
#endif
#ifdef __SSE__
#include <xmmintrin.h>
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
    Box full() const { return {std::vector<size_t>(extents.size(), 0), extents}; }
};
struct QueuedBinding {
    fort_buffer_t buffer;
    Effects effects;
    std::vector<fort_scope_section> reads,writes,overwrites;
    fort_scope_plan_binding view() {
        auto pack=[](const Region &region,std::vector<fort_scope_section> &out) {
            out.clear(); out.reserve(region.size());
            for (const auto &box : region) out.push_back({box.lo.data(),box.hi.data()});
        };
        pack(effects.reads,reads); pack(effects.writes,writes); pack(effects.overwrites,overwrites);
        return {buffer,{0,reads.size(),reads.data(),writes.size(),writes.data(),overwrites.size(),overwrites.data()}};
    }
};
struct QueuedTask {
    uint64_t key;
    fort_scope_kernel_v1 kernel;
    // Numerical work may be submitted by a later native access. Retain its
    // original diagnostic position independently of that flush boundary.
    std::unique_ptr<fort_scope_trace_state_v1> provenance;
    std::vector<std::vector<unsigned char>> values;
    std::vector<void *> arguments;
    std::vector<QueuedBinding> bindings;
    std::vector<fort_scope_plan_binding> binding_views;
    fort_scope_graph_task_v1 view() {
        binding_views.clear(); binding_views.reserve(bindings.size());
        for (auto &binding : bindings) binding_views.push_back(binding.view());
        kernel.arguments=arguments.data();
        return {kernel,binding_views.data(),binding_views.size()};
    }
};
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
// An execution context borrows this state; its plans/proofs never do.
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
    bool copy_completion_priced = false;
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
          completions(residency->completions), tail_completion(residency->tail_completion)
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
    std::unordered_map<fort_buffer_t,Buffer> virtual_buffers;
    uint64_t virtual_generation = 0;
    std::vector<Buffer *> &unfenced_upload_buffers;
    std::deque<std::shared_ptr<Completion>> &completions;
    std::weak_ptr<Completion> &tail_completion;
    std::optional<fort_scope_batch_costs> transfer_costs;
    std::optional<fort_physical::compact::Pricing> copy_costs;
    std::optional<fort_scope_team_costs> team_costs;
    std::optional<fort_scope_batch_report> batch_report;
    bool batch_active = false;
    struct SourceHostTask {
        uint64_t token;
        bool started = false;
        std::vector<fort_buffer_t> buffers;
    };
    std::optional<SourceHostTask> source_host_task;
    std::vector<fort_scoped::planning::Operation> plan;
    std::vector<bool> schedule;
    size_t worker_cursor = 0;
    bool plan_recording = false, plan_installed = false, plan_activity_started = false;
    bool graph_planning = false;
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
    fort_scoped::graphs::Cache graphs;
    std::vector<cudaEvent_t> &completion_events, &available_completion_events;
    cudaStream_t &stream;
    cudaMemPool_t &pool;
#endif
};
enum class Change { None, Query, State };
void invalidate(Context &c, Change change) noexcept;
void note_numerical_activity(Context &c) noexcept;
void drain_accesses(Context &c,const fort_scope_plan_binding *bindings,size_t count);
void drain_resources(Context &c,const fort_buffer_t *buffers,size_t count);
void flush_commands(Context &c,const std::vector<bool> *selected=nullptr,bool conflict=false);
Buffer &virtual_buffer(Context &c,fort_buffer_t handle);
void reserve_devices(Context &c,const fort_buffer_t *handles,size_t count,void **devices=nullptr);
void touch_device(Context &c,Buffer &b) noexcept {
    if (c.residency_clock==std::numeric_limits<uint64_t>::max()) {
        for (auto &entry : c.buffers) entry.second->last_device_use=0;
        c.residency_clock=0;
    }
    b.last_device_use=++c.residency_clock;
}
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
std::unordered_map<fort_scope_domain_t, std::shared_ptr<Residency>> domains;
fort_scoped::native_hint::Generation native_hint_gpu_generation, native_hint_error_generation;
fort_scoped::native_hint::Cache native_hint_cache; // Protected by registry_mutex.
void note_hint_error(int status) noexcept {
    if(status!=FORT_SCOPE_OK && status!=FORT_SCOPE_UNINITIALIZED)
        native_hint_error_generation.advance();
}
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
// never supplies placement, definition or source-lifetime authority.
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
    native_hint_gpu_generation.advance(); // A direct GPU attempt vetoes native learning.
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
bool overlaps(const Region &a, const Region &b) {
    for (const auto &left : a) for (const auto &right : b) {
        bool common = true;
        for (size_t axis=0; axis<left.lo.size(); ++axis)
            if (left.lo[axis] >= right.hi[axis] || right.lo[axis] >= left.hi[axis]) { common = false; break; }
        if (common) return true;
    }
    return false;
}
Region pending_host_reads(const Buffer &b) {
    Region result = b.unfenced_uploads;
    try {
        Budget budget;
        for (const auto &reader : b.upload_readers) if (!reader.completion->complete)
            result = unite(std::move(result),reader.sections,budget);
    } catch (const Fragmented &) { return {b.full()}; }
    return result;
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
    // Source quotes and the reference runtime reserve exactly the requested
    // growth. Extra capacity here can consume the budget for subsequent fields.
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
void compact_estimate_unavailable(Context &c) noexcept {
    c.owner_estimate_available=false;
    if(c.last_report) {c.last_report->available=0;c.last_report->owner_available=0;}
}
void compact_fallback(Context &c,uint32_t reason) noexcept {
    compact_estimate_unavailable(c);transfer_fallback(c,reason);
}
[[maybe_unused]] void compact_lease_observation(Context &c,size_t expected_capacity,bool expected_reused,
                               size_t actual_capacity,bool actual_reused) noexcept {
    if(expected_capacity!=actual_capacity || expected_reused!=actual_reused)
        compact_estimate_unavailable(c);
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
       parts.empty() || c.batch_active)return false;
    std::vector<fort_physical::CopyOperation> operations;
    compact::Plan plan;
    try {
        for(const auto &part:parts) {
            const fort_physical::CopyPlan original(b.element_bytes,b.extents,part.lo,part.hi);
            require(original.valid,FORT_SCOPE_ARGUMENT,"invalid physical transfer layout");
            if(!original.visit([&](const auto &op) {
                if(operations.size()==compact::operation_limit)return false;
                operations.push_back(op);return true;
            })) {compact_estimate_unavailable(c);return false;}
        }
        plan=compact::make_plan(operations,b.bytes,std::numeric_limits<size_t>::max());
    } catch(const compact::Ineligible &) { compact_estimate_unavailable(c);return false; }
      catch(const std::bad_alloc &) { compact_estimate_unavailable(c);return false; }
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
    if(!price.available) {compact_estimate_unavailable(c);return false;}
    observation.priced=true;
    if(!price.selected) {
        if(decision.observation>0)observation={before,decision,true,true};
        return false;
    }
    allocate(c,b);
#if defined(FORT_SCOPE_CPU_TEST) || defined(FORT_SCOPE_TEST_FAULTS)
    if(std::getenv("FORT_SCOPE_TEST_FAIL_COMPACT_ALLOC")) {
        compact_fallback(c,FORT_SCOPE_TRANSFER_ALLOCATION);return false;
    }
#endif
#ifdef FORT_SCOPE_CPU_TEST
    std::vector<unsigned char> packed_storage,device_storage;
    try {packed_storage.resize(price.capacity);device_storage.resize(price.capacity);}
    catch(const std::bad_alloc &) {compact_fallback(c,FORT_SCOPE_TRANSFER_ALLOCATION);return false;}
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
    catch(const std::bad_alloc &) {compact_fallback(c,FORT_SCOPE_TRANSFER_ALLOCATION);return false;}
    if(lease.exhausted) {compact_fallback(c,(lease.unpriced_idle||lease.stale_startup)?FORT_SCOPE_TRANSFER_ESTIMATES_UNAVAILABLE:FORT_SCOPE_TRANSFER_BUDGET);return false;}
    if(lease.status!=cudaSuccess) {
        try {cuda_check(c,lease.status,true);}catch(const Error &error) {
            if(error.status!=FORT_SCOPE_RESOURCE)throw;
            compact_fallback(c,FORT_SCOPE_TRANSFER_ALLOCATION);return false;
        }
    }
    require(bool(lease.slots),FORT_SCOPE_STATE,"missing compact staging lease");
    // Cache state can change after metadata pricing. Reprice the actual lease
    // before payload submission; a changed choice never replays source work.
    const bool reused=lease.reused;
    compact_lease_observation(c,price.capacity,pool.reused,lease.slots->capacity,reused);
    pool={lease.slots->capacity,available,reused};
    decision=compact::startup::choose(plan,pricing,upload,c.pending&&!c.residency->copy_completion_priced,pool,before);
    price=decision.price;
    if(!price.available || !price.selected || (decision.seed&&!lease.startup_paid)) {
        size_t freed=0;cuda_check(c,fort_staging::release(std::move(lease.slots),freed));
        compact_estimate_unavailable(c);
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
        if (source_full && !c.last_report && !c.batch_active) {
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
        c.transfer_stats.fallback_reason = c.last_report ? FORT_SCOPE_TRANSFER_ESTIMATES_UNAVAILABLE
                                                       : FORT_SCOPE_TRANSFER_COALESCED_INELIGIBLE;
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
        bool changed=false,booked=false;
#ifdef FORT_SCOPE_CPU_TEST
        auto &state=c.residency->compact_reference_startup;
        changed=!fort_physical::compact::startup::same(state,observation.before);
        booked=fort_physical::compact::startup::observe(state,observation.before,observation.decision);
#else
        booked=fort_staging::observe_startup(observation.before,observation.decision,changed);
#endif
        if(changed||!booked)compact_estimate_unavailable(c);
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
enum class AccessContract { Defined, OriginalHost, OriginalDevice };
AccessPreparation prepare_effects(Buffer &b, const Effects &e, bool device,
                                  AccessContract contract=AccessContract::Defined) {
    require(contract!=AccessContract::OriginalHost || !device, FORT_SCOPE_ARGUMENT,
            "original host access cannot acquire device pointers");
    require(contract!=AccessContract::OriginalDevice || device, FORT_SCOPE_ARGUMENT,
            "original device access requires device pointers");
    require(!b.prepared, FORT_SCOPE_STATE, "buffer access already prepared");
    AccessPreparation result{&b, {b.initialized, b.host_current, b.device_current}, {}, {}, e.writes, false};
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
        } else if (contract==AccessContract::OriginalDevice) {
            // Copy object representations, never evaluate a value or promote
            // may-reads to definitions. Untracked storage still belongs to the
            // original host allocation. Newer tracked device values win even
            // after a semantic definition kill.
            auto unknown = difference(requested, fort_scoped::coherence::physical_values(result.state), budget);
            result.state.host_current = unite(result.state.host_current, unknown, budget);
        } else {
            require(difference(requested, result.state.initialized, budget).empty(), FORT_SCOPE_UNINITIALIZED,
                    "access reads an uninitialized section");
        }
        auto &chosen = device ? result.state.device_current : result.state.host_current;
        auto missing = difference(requested, chosen, budget);
        auto updated = unite(chosen, missing, budget);
        if (!missing.empty()) result.copies.push_back(std::move(missing));
        chosen = std::move(updated);
    };
    auto compute = [&]() {
        Budget budget;
        const auto preserve = difference(e.writes, e.overwrites, budget);
        ensure_state(e.reads);
        ensure_state(preserve);
        if (contract==AccessContract::OriginalHost)
            return fort_scoped::coherence::prepare_original_host(result.state, e);
        if (contract==AccessContract::OriginalDevice)
            return fort_scoped::coherence::prepare_original_device(result.state, e);
        return prepare(result.state, e, device);
    };
    try { result.prepared = compute(); }
    catch (const Fragmented &) {
        result.state = {b.initialized, b.host_current, b.device_current};
        result.copies.clear();
        const auto &other = device ? b.host_current : b.device_current;
        if (!other.empty()) result.copies.push_back(other);
        (device ? result.state.device_current : result.state.host_current) = physical_values(b);
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
                  KernelTracePositions positions={}) {
    seal_uploads(c);
    initialize(c);
    DeviceGuard guard(c);
    bool launch_attempted = false;
    if (c.queue_stats.graphs_enabled) {
        bool warmup = false;
        auto status = c.graphs.observe(key,kernels,count,c.graph_stats,warmup);
        if (status == cudaSuccess && !warmup)
            status = c.graphs.submit(key,kernels,count,c.stream,c.graph_stats,launch_attempted);
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
void execute_graph(Context &c, uint64_t key, const fort_scope_graph_task_v1 *tasks, size_t count,bool resource_fallback=false,
                   KernelTracePositions positions={}) {
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
    struct Resource {
        Buffer *actual;
        Buffer state;
        std::vector<Region> uploads;
        size_t reconciliations = 0;
    };
    std::vector<Resource> resources;
    std::unordered_map<fort_buffer_t,size_t> indices;
    // Each step sees preceding writes, but none of those definitions become
    // externally visible before the entire group has been submitted safely.
    for (size_t i=0; i<count; ++i) {
        std::unordered_set<fort_buffer_t> seen;
        for (size_t j=0; j<tasks[i].binding_count; ++j) {
            const auto &binding = tasks[i].bindings[j];
            require(seen.insert(binding.buffer).second,FORT_SCOPE_ALIAS,"graph task repeats a canonical buffer");
            auto found = indices.find(binding.buffer);
            if (found == indices.end()) {
                auto &b = buffer(c,binding.buffer);
                require(!b.prepared,FORT_SCOPE_STATE,"graph resource access already prepared");
                const auto index = resources.size();
                resources.push_back({&b,b,{ },0});
                found = indices.emplace(binding.buffer,index).first;
            }
            auto &resource = resources[found->second];
            auto prepared = prepare_access(resource.state,&binding.access,true);
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
    (void)resource_fallback;
    (void)positions;
    throw Error(FORT_SCOPE_RESOURCE,"CUDA graphs unavailable in the CPU reference backend");
#else
    std::vector<fort_buffer_t> handles;
    handles.reserve(resources.size());
    for (const auto &resource : resources) handles.push_back(resource.actual->handle);
    reserve_devices(c,handles.data(),handles.size());
    for (auto &resource : resources)
        for (const auto &parts : resource.uploads) copy_region(c,*resource.actual,parts,true);
    submit_graph(c,key,kernels.data(),count,resource_fallback,positions);
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
void apply_queued(Context &c,const QueuedTask &task) {
    for (const auto &binding : task.bindings) {
        auto &actual=buffer(c,binding.buffer);
        require(!actual.prepared,FORT_SCOPE_STATE,"queued resource already prepared");
        auto found=c.virtual_buffers.find(binding.buffer);
        if (found==c.virtual_buffers.end()) found=c.virtual_buffers.emplace(binding.buffer,actual).first;
        auto prepared=prepare_effects(found->second,binding.effects,true);
        found->second.initialized=std::move(prepared.prepared.initialized);
        found->second.device_current=std::move(prepared.prepared.current);
        found->second.host_current=std::move(prepared.prepared.opposite);
    }
}
void refresh_virtual(Context &c) {
    if (c.virtual_generation==c.state_generation) return;
    c.virtual_buffers.clear();
    try {
        for (const auto &task : c.commands) apply_queued(c,*task);
        c.virtual_generation=c.state_generation;
    } catch (...) { c.virtual_generation=0; throw; }
}
Buffer &virtual_buffer(Context &c,fort_buffer_t handle) {
    if (c.commands.empty()) return buffer(c,handle);
    refresh_virtual(c);
    const auto found=c.virtual_buffers.find(handle);
    return found==c.virtual_buffers.end()?buffer(c,handle):found->second;
}
bool conflicts(const Effects &earlier,const Effects &later) {
    return overlaps(earlier.writes,later.reads) || overlaps(earlier.writes,later.writes) ||
           overlaps(earlier.reads,later.writes);
}
bool conflicts(const QueuedTask &earlier,const QueuedTask &later) {
    for (const auto &a : earlier.bindings) for (const auto &b : later.bindings)
        if (a.buffer==b.buffer && conflicts(a.effects,b.effects)) return true;
    return false;
}
void flush_commands(Context &c,const std::vector<bool> *selected,bool conflict) {
    if (c.commands.empty()) return;
    std::vector<fort_scope_graph_task_v1> tasks;
    tasks.reserve(c.commands.size());
    const bool tracing=trace_enabled();
    const fort_scope_trace_state_v1 unknown{};
    std::unique_ptr<const fort_scope_trace_state_v1 *[]> positions;
    if (tracing && trace_allocation_allowed("flush"))
        positions.reset(new(std::nothrow) const fort_scope_trace_state_v1 *[c.commands.size()]);
    uint64_t key=1469598103934665603ULL;
    for (size_t i=0; i<c.commands.size(); ++i) if (!selected || (*selected)[i]) {
        tasks.push_back(c.commands[i]->view());
        if (positions) positions[tasks.size()-1]=c.commands[i]->provenance ? c.commands[i]->provenance.get() : &unknown;
        key=(key^c.commands[i]->key)*1099511628211ULL;
    }
    if (tasks.empty()) return;
    // Diagnostic allocation failure loses attribution, never numerical work.
    execute_graph(c,key?key:1,tasks.data(),tasks.size(),true,{positions.get(),&unknown});
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
    invalidate(c,Change::State);
    trace(c,"queue_flush",nullptr,tasks.size());
}
void drain_effects(Context &c,const std::vector<fort_scoped::planning::Binding> &accesses) {
    if (c.commands.empty()) return;
    std::vector<bool> selected(c.commands.size(),false);
    for (size_t i=0; i<c.commands.size(); ++i) for (const auto &queued : c.commands[i]->bindings)
        for (const auto &access : accesses)
            if (queued.buffer==access.buffer && conflicts(queued.effects,access.effects)) selected[i]=true;
    // Include dependency ancestors while preserving original relative order.
    // Unrelated earlier tasks remain queued across this host boundary.
    for (size_t j=c.commands.size(); j-->0;) if (selected[j])
        for (size_t i=0; i<j; ++i) if (conflicts(*c.commands[i],*c.commands[j])) selected[i]=true;
    flush_commands(c,&selected,true);
}
void drain_accesses(Context &c,const fort_scope_plan_binding *bindings,size_t count) {
    if (c.commands.empty()) return;
    require(!count || bindings,FORT_SCOPE_ARGUMENT,"missing queue boundary accesses");
    std::vector<fort_scoped::planning::Binding> accesses;
    accesses.reserve(count);
    for (size_t i=0; i<count; ++i) accesses.push_back({bindings[i].buffer,effects(buffer(c,bindings[i].buffer),&bindings[i].access)});
    drain_effects(c,accesses);
}
void drain_resources(Context &c,const fort_buffer_t *handles,size_t count) {
    if (c.commands.empty()) return;
    require(!count || handles,FORT_SCOPE_ARGUMENT,"missing queue boundary resources");
    std::vector<fort_scoped::planning::Binding> accesses;
    for (size_t i=0; i<count; ++i) {
        auto &b=buffer(c,handles[i]);
        const Region full=empty(b.full())?Region{}:Region{b.full()};
        accesses.push_back({handles[i],{full,full,full}});
    }
    drain_effects(c,accesses);
}
void enqueue_graph(Context &c,uint64_t key,const fort_scope_graph_task_v1 *task,const size_t *sizes) {
    require(key && task,FORT_SCOPE_ARGUMENT,"missing queued graph task or code key");
    const auto invalid=fort_scoped::graphs::invalid(&task->kernel,1);
    require(!invalid,FORT_SCOPE_ARGUMENT,invalid?invalid:"invalid queued kernel");
    require(!task->kernel.argument_count || sizes,FORT_SCOPE_ARGUMENT,"missing queued kernel argument sizes");
    require(!task->binding_count || task->bindings,FORT_SCOPE_ARGUMENT,"missing queued task effects");
    auto owned=std::make_unique<QueuedTask>(); owned->key=key; owned->kernel=task->kernel;
    if (trace_enabled() && trace_allocation_allowed("enqueue"))
        owned->provenance.reset(new(std::nothrow) fort_scope_trace_state_v1(c.provenance));
    owned->values.reserve(task->kernel.argument_count); owned->arguments.reserve(task->kernel.argument_count);
    uint64_t bytes=0;
    for (size_t i=0; i<task->kernel.argument_count; ++i) {
        require(sizes[i] && sizes[i]<=std::numeric_limits<uint64_t>::max()-bytes,FORT_SCOPE_ARGUMENT,"invalid queued argument size");
        owned->values.emplace_back(sizes[i]);
        std::memcpy(owned->values.back().data(),task->kernel.arguments[i],sizes[i]);
        owned->arguments.push_back(owned->values.back().data()); bytes+=sizes[i];
    }
    std::unordered_set<fort_buffer_t> seen;
    for (size_t i=0; i<task->binding_count; ++i) {
        const auto &binding=task->bindings[i]; auto &b=buffer(c,binding.buffer);
        require(seen.insert(binding.buffer).second,FORT_SCOPE_ALIAS,"queued task repeats a canonical buffer");
        require(!b.prepared,FORT_SCOPE_STATE,"queued resource already prepared");
        require(!b.bytes || b.device,FORT_SCOPE_STATE,"reserve queued device storage before enqueue");
        owned->bindings.push_back({binding.buffer,effects(b,&binding.access),{},{},{}});
    }
#ifndef FORT_SCOPE_CPU_TEST
    initialize(c); // Device availability is checked before accepting work.
#endif
    refresh_virtual(c);
    try {
        apply_queued(c,*owned);
        c.commands.push_back(std::move(owned));
    } catch (...) { c.virtual_generation=0; throw; }
    c.virtual_generation=c.state_generation;
    c.queue_stats.pending_tasks=c.commands.size();
    const auto accepted_size=c.commands.size();
    if (!c.queue_stats.queue_enabled) {
        try { flush_commands(c); }
        catch (const Error &error) {
            if (error.status==FORT_SCOPE_RESOURCE) {
                c.commands.pop_back(); c.virtual_generation=0; c.queue_stats.pending_tasks=c.commands.size();
            }
            throw;
        }
    }
    ++c.queue_stats.accepted_tasks; c.queue_stats.copied_argument_bytes+=bytes;
    native_hint_gpu_generation.advance();
    note_numerical_activity(c);
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
    else {
        // LRU reuse changes placement costs without changing definitions or
        // virtual queued values. Keep their proofs and discard pricing only.
        c.planning_snapshot.reset(); c.preview.reset();
    }
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
        note_hint_error(FORT_SCOPE_RESOURCE);
        return FORT_SCOPE_RESOURCE;
    }
    fort_scoped::TeamObservation observation(FORT_SCOPE_TEAM_OBSERVE_API);
    try { f(); return FORT_SCOPE_OK; }
    catch (const Error &error) { note_hint_error(error.status); diagnostic(error.what()); return error.status; }
    catch (const Fragmented &) { note_hint_error(FORT_SCOPE_BOUNDARY); diagnostic("section fragmentation requires a scope boundary"); return FORT_SCOPE_BOUNDARY; }
    catch (const std::bad_alloc &) { note_hint_error(FORT_SCOPE_RESOURCE); diagnostic("scope metadata allocation failed"); return FORT_SCOPE_RESOURCE; }
    catch (const std::exception &error) { note_hint_error(FORT_SCOPE_STATE); diagnostic(error.what()); return FORT_SCOPE_STATE; }
    catch (...) { note_hint_error(FORT_SCOPE_STATE); diagnostic("unknown scope runtime failure"); return FORT_SCOPE_STATE; }
}
void invalidate(Context &c, Change change) noexcept {
    if (change == Change::None) return;
    c.definition_proof.reset(); c.planning_snapshot.reset(); c.preview.reset();
    auto &generation = change == Change::Query ? c.query_generation : c.state_generation;
    if (change == Change::State) c.virtual_generation=0;
    // Both cached stamps are discarded before wrapping, so a reused generation
    // can never match a surviving old proof or preview.
    generation = generation == std::numeric_limits<uint64_t>::max() ? 1 : generation+1;
}
void note_numerical_activity(Context &c) noexcept {
    if (c.plan_installed) c.plan_activity_started = true;
    // A complete-owner estimate must cover the entire context, including work
    // performed before its first continuation query or after a completed one.
    if (c.endpoint_mode != FORT_SCOPE_PLAN_CONTINUE || !c.segment_cost_pending)
        c.owner_estimate_available = false;
}
template<class F> int with(fort_scope_t handle, F &&f, Change change = Change::State,
                           bool source_host_control = false) noexcept {
    const int status=protect([&]() {
        const auto c = lookup(handle);
        std::lock_guard<std::mutex> lock(c->mutex);
        require(!c->closed && !c->residency->retired, FORT_SCOPE_STALE, "invalid or stale scope handle");
        require(!c->poisoned, FORT_SCOPE_EXECUTION, "scope has an execution failure; unsafe replay prohibited");
        if(c->source_host_task && !source_host_control && change!=Change::None) {
            if(c->source_host_task->started) c->poisoned=true;
            throw Error(c->poisoned ? FORT_SCOPE_EXECUTION : FORT_SCOPE_STATE,
                        "scope mutation during original CPU source transaction");
        }
        // Invalidate before a possibly partial mutation, including operations
        // that fail. Read-only metadata probes preserve a context-local proof.
        invalidate(*c, change);
        try { f(*c); }
        catch (...) {
            if(source_host_control && c->source_host_task && c->source_host_task->started) {
                c->poisoned=true;
                throw Error(FORT_SCOPE_EXECUTION,"original CPU source work started; unsafe replay prohibited");
            }
            if (c->endpoint_mode == FORT_SCOPE_PLAN_CONTINUE && change != Change::None)
                c->owner_estimate_available = false;
            throw;
        }
    });
    // Even a protective-environment/setup failure before entering the lambda
    // is fatal if the original CPU span has already been declared started.
    if(source_host_control && status!=FORT_SCOPE_OK) {
        try {
            const auto c=lookup(handle);
            std::lock_guard<std::mutex> lock(c->mutex);
            if(c->source_host_task && c->source_host_task->started) {
                c->poisoned=true;
                forget_copy_startup(*c);
                return FORT_SCOPE_EXECUTION;
            }
        } catch(...) {} // Preserve the original failure for stale handles.
    }
    if(status!=FORT_SCOPE_OK) {
        try {
            const auto c=lookup(handle);std::lock_guard<std::mutex> lock(c->mutex);
            if(c->poisoned)forget_copy_startup(*c);
        } catch(...) {} // Preserve the original failure, including stale handles.
    }
    return status;
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
            std::cerr << std::setprecision(17) << "FORT_SCOPED planning_timing context=" << handle;
            if(context.residency->domain)
                std::cerr << " domain_version=1 domain=" << context.residency->domain << " domain_role=execution";
            std::cerr << " phase=" << phase << " seconds=" << elapsed
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
    if (!c.segment_cost_pending || c.pending || !c.commands.empty() || c.worker_cursor != c.schedule.size() ||
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
void abandon_planning_metadata(Context &c) noexcept {
    invalidate(c, Change::Query);
    c.plan.clear(); c.schedule.clear(); c.worker_cursor = 0;
    c.plan_installed = false; c.plan_recording = false; c.plan_activity_started = false;
    c.segment_cost_pending = false; c.last_report.reset();
    c.owner_estimate_available = false;
}
void copy_pool_inputs(Context &c,fort_scoped::planning::Inputs &input) {
    input.copy_completion_priced=c.residency->copy_completion_priced;
    input.copy_device=c.device;input.copy_pool_capacity=0;input.copy_pool_active_device_bytes=0;
    input.copy_pool_available_bytes=std::numeric_limits<size_t>::max();
#ifdef FORT_SCOPE_CPU_TEST
    input.copy_pool_capacity=c.residency->compact_reference_capacity;
    input.copy_pool_cold_available=!input.copy_pool_capacity;
    input.copy_startup=c.residency->compact_reference_startup;
#else
    const auto copy_pool=fort_staging::snapshot();
    input.copy_pool_revision=copy_pool.revision;input.copy_startup=copy_pool.startup;
    input.copy_pool_available_bytes=fort_staging::compact_available(copy_pool);
    input.copy_pool_cold_available=copy_pool.idle_capacity==0;
    const bool idle_compact=copy_pool.idle_role==fort_staging::Role::Compact;
    input.copy_pool_active_device_bytes=copy_pool.device_reserved-(idle_compact?2*copy_pool.idle_capacity:0);
    if(idle_compact && copy_pool.idle_device==c.device)input.copy_pool_capacity=copy_pool.idle_capacity;
#endif
}
const fort_scoped::planning::Inputs &planning_inputs(Context &c, fort_scope_t handle) {
    if (c.planning_snapshot && c.planning_snapshot->query_generation == c.query_generation &&
        c.planning_snapshot->state_generation == c.state_generation) {
        auto &input = c.planning_snapshot->input;
        const auto previous=input.copy_startup;const auto previous_pool=input.copy_pool_revision;
        copy_pool_inputs(c,input);
        if(c.copy_costs && (previous_pool!=input.copy_pool_revision || !fort_physical::compact::startup::same(previous,input.copy_startup))) {
            c.preview.reset();compact_estimate_unavailable(c);
        }
        input.driver_initialized = driver_initialized(c);
        input.definitions_validated = c.definition_proof &&
            c.definition_proof->query_generation == c.query_generation &&
            c.definition_proof->state_generation == c.state_generation;
        return input;
    }
    PlanningTimer timing(c, handle, "query_construction");
    require(c.graph_planning || (c.plan.size() <= fort_scoped::planning::detail::operation_limit &&
            c.buffers.size() <= fort_scoped::planning::detail::operation_limit),
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
    input.device_budget = field_budget(c);
    input.device_ready = c.ready;
    input.pending = c.pending || !c.commands.empty();
    input.graph_mode = c.graph_planning;
    input.transfer_mode = c.transfer_stats.requested_mode == FORT_SCOPE_TRANSFERS_PINNED
        ? FORT_SCOPE_TRANSFERS_PINNED : c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_AUTO && c.copy_costs
        ? FORT_SCOPE_TRANSFERS_AUTO : FORT_SCOPE_TRANSFERS_DIRECT;
    input.direct_transfer_fit_applicable = c.transfer_stats.requested_mode == FORT_SCOPE_TRANSFERS_DIRECT;
    input.transfer_costs = c.transfer_costs;
    input.copy_costs=c.copy_costs;
    copy_pool_inputs(c,input);
#ifdef FORT_SCOPE_CPU_TEST
    input.asynchronous_release = false;
#else
#if CUDART_VERSION >= 11020
    // Stream-ordered pools combine terminal publication and retirement into
    // one wait. Until initialization, use the conservative synchronous model
    // because devices without pool support retain per-buffer publication.
    input.asynchronous_release = c.ready && c.pool != nullptr;
#else
    input.asynchronous_release = false;
#endif
#endif
    input.driver_initialized = driver_initialized(c);
    for (const auto &entry : c.buffers) {
        const auto &b = virtual_buffer(c,entry.first);
        require(!b.prepared, FORT_SCOPE_STATE, "planning during a prepared buffer access");
        input.resources.push_back({b.handle, b.element_bytes, b.bytes, b.extents,
                                   b.initialized, b.host_current, b.device_current, b.device != nullptr,
                                   pending_host_reads(b),entry.second->last_device_use});
    }
    c.planning_snapshot = Context::QuerySnapshot{c.query_generation, c.state_generation, std::move(input)};
    return c.planning_snapshot->input;
}
std::vector<fort_scoped::planning::Binding> planning_bindings(
        Context &c, const fort_scope_plan_binding *bindings, size_t count) {
    require((c.graph_planning || count <= 256) && (!count || bindings), FORT_SCOPE_ARGUMENT, "invalid planning binding list");
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
                state.initialized.clear(); continue;
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
    input.device_budget=field_budget(c); input.device_ready=c.ready; input.driver_initialized=driver_initialized(c);
    input.pending=c.pending; input.continuation=true; input.charge_create=false; input.registrations_incurred=0;
#ifdef FORT_SCOPE_CPU_TEST
    input.asynchronous_release=false;
#endif
    for (const auto &r : plan.resources) {
        const auto &b=*r.original;
        input.resources.push_back({b.handle,b.element_bytes,b.bytes,b.extents,b.initialized,b.host_current,b.device_current,
                                   b.device!=nullptr,pending_host_reads(b)});
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
void batch_trace(const Context &c,fort_scope_t handle, const fort_scope_batch_report &report) noexcept {
    if (!diagnostics_enabled()) return;
    try {
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::cerr << std::setprecision(17) << "FORT_SCOPED evidence {\"schema_version\":1,\"event\":\"batch_statistics\",\"stats_version\":1,\"context\":" << handle << DomainJson{c}
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
void decision_trace(const Context &c,const fort_scope_plan_decision &d, const std::string &reason) {
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
              << " reason=" << (reason.empty() ? "calibrated_selection" : reason);
    if(c.residency->domain)
        std::cerr << " context=" << c.handle << " domain_version=1 domain=" << c.residency->domain << " domain_role=execution";
    std::cerr << '\n';
}
void definition_validation_trace(Context &context, fort_scope_t handle,
                                 const fort_scoped::planning::DefinitionValidation &proof,
                                 bool cache_hit, double seconds) noexcept {
    const char *enabled = std::getenv("FORT_RUNTIME_TRACE");
    if (!enabled || std::strcmp(enabled, "1")) return;
    try {
        std::ostringstream out;
        out << std::setprecision(17) << "{\"schema_version\":1,\"context\":" << handle << DomainJson{context}
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
    out << std::setprecision(17) << "{\"schema_version\":1,\"context\":" << output.handle << DomainJson{output.context}
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
        if(event.compact_copy)out<<",\"transport\":\"compact\",\"physical_operations\":"<<event.physical_operations
                                  <<",\"copy_kernel_launches\":"<<event.copy_kernels;
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
            std::cerr << std::setprecision(17) << "FORT_SCOPED evidence {\"schema_version\":1,\"context\":" << handle << DomainJson{context}
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
                  << ",\"aggregate_only\":" << (result.native_startup_shortcut || result.native_analytic_shortcut ? "true" : "false")
                  << ",\"evidence\":\"" << (result.native_startup_shortcut ? "native_startup_lower_bound" :
                      result.native_analytic_shortcut ? "native_stationary_single_candidate" : "modeled_final_schedule")
                  << "\",\"section_coordinates\":\"zero_based_exclusive\"";
            if (context.graph_planning)
                std::cerr << ",\"ranking_policy\":\"warm_graph_heuristic\""
                          << ",\"terminal_debt_adjusted_ranking_seconds\":" << result.report.ranking_seconds
                          << ",\"warm_gate_seconds\":" << result.graph_ranking_seconds
                          << ",\"warm_gate_counterfactual_seconds\":" << result.graph_counterfactual_seconds
                          << ",\"warm_gate_required_saving_seconds\":" << result.graph_required_saving_seconds
                          << ",\"warm_gate_transfer_dispersion_seconds\":" << result.graph_transfer_dispersion_seconds
                          << ",\"cold_driver_startup_debt_seconds\":" << result.cold_driver_startup_debt_seconds
                          << ",\"startup_debt_is_prediction\":true"
                          << ",\"unpriced_team_overhead\":" << (result.unpriced_team_overhead ? "true" : "false");
            std::cerr << ",\"unpriced_memo_overhead\":" << (result.unpriced_memo_overhead ? "true" : "false")
                      << ",\"unpriced_native_analytic_overhead\":" << (result.unpriced_native_analytic_overhead ? "true" : "false")
                      << ",\"placement_memo_hit\":" << (result.memo_hit ? "true" : "false")
                      << ",\"placement_memo_key_seconds\":" << result.memo_key_seconds;
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
        std::cerr << "FORT_SCOPED evidence {\"schema_version\":1,\"context\":" << handle << DomainJson{context}
                  << ",\"event\":\"end\",\"rows\":" << output.rows
                  << ",\"complete\":" << (complete ? "true" : "false")
                  << ",\"truncated\":" << (output.truncated ? "true" : "false") << "}\n";
    } catch (...) {}
}
}
namespace {
void require_planning_host_current(Context &c, fort_buffer_t handle, const Region &reads) {
    const auto &actual = buffer(c, handle);
    require(!actual.prepared, FORT_SCOPE_STATE, "planning payload has an unfinished access");
    // Queued writes have not changed actual host coverage yet. Their virtual
    // effects must also exclude stale snapshots; neither query drains work.
    const auto &current = virtual_buffer(c, handle);
    Budget budget;
    require(difference(reads, actual.initialized, budget).empty() &&
            difference(reads, actual.host_current, budget).empty() &&
            difference(reads, current.initialized, budget).empty() &&
            difference(reads, current.host_current, budget).empty(), FORT_SCOPE_BOUNDARY,
            "planning payload must already be initialized and host current in every requested region");
}
}
extern "C" int fort_scope_plan_host_current(fort_scope_t h, fort_buffer_t handle) {
    return with(h, [&](Context &c) {
        const auto &b = buffer(c, handle);
        require_planning_host_current(c, handle, empty(b.full()) ? Region{} : Region{b.full()});
    }, Change::None);
}
extern "C" int fort_scope_plan_host_sections_current_v1(fort_scope_t h, fort_buffer_t handle,
                                                        const fort_scope_section *items, size_t count) {
    return with(h, [&](Context &c) {
        require_planning_host_current(c, handle, sections(buffer(c, handle), items, count, false));
    }, Change::None);
}
extern "C" int fort_scope_plan_reset_mode(fort_scope_t h, uint32_t endpoint_mode) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "query_construction");
        require(endpoint_mode <= FORT_SCOPE_PLAN_CONTINUE, FORT_SCOPE_ARGUMENT, "invalid planning endpoint mode");
        for (const auto &entry : c.buffers)
            require(!entry.second->prepared, FORT_SCOPE_STATE, "planning during a prepared buffer access");
        require(!c.plan_installed || c.worker_cursor == c.schedule.size(), FORT_SCOPE_STATE,
                "cannot discard an unfinished execution schedule");
        complete_segment_estimate(c);
        // Pending work may span source queries. Its previous prediction cannot
        // be counted as completed, or silently dropped from an available sum.
        if (c.segment_cost_pending) c.owner_estimate_available = false;
        if (c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE && endpoint_mode == FORT_SCOPE_PLAN_COMPLETE)
            c.owner_estimate_available = false;
        c.plan.clear(); c.schedule.clear(); c.worker_cursor = 0;
        c.plan_installed = false; c.plan_recording = true; c.plan_activity_started = false;
        c.endpoint_mode = endpoint_mode; c.last_report.reset();
    }, Change::Query);
}
extern "C" int fort_scope_plan_reset(fort_scope_t h) {
    return fort_scope_plan_reset_mode(h, FORT_SCOPE_PLAN_COMPLETE);
}
extern "C" int fort_scope_plan_cancel_native_v1(fort_scope_t h) {
    return with(h, [&](Context &c) {
        require(c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE && c.plan_installed,
                FORT_SCOPE_STATE, "native handoff requires an installed continuation schedule");
        for (const auto &entry : c.buffers)
            require(!entry.second->prepared, FORT_SCOPE_STATE, "native handoff during a prepared buffer access");
        require(!c.worker_cursor && !c.plan_activity_started, FORT_SCOPE_STATE,
                "native handoff requires an unstarted execution schedule");
        require(std::none_of(c.schedule.begin(), c.schedule.end(), [](bool gpu) { return gpu; }),
                FORT_SCOPE_STATE, "native handoff cannot discard GPU choices");
        // This is a handoff to independently checked original source, not
        // execution of the predicted CPU workers. Preserve physical/virtual
        // definitions, mirrors and any earlier accepted GPU command queue.
        abandon_planning_metadata(c);
        trace(c, "plan_cancel_native");
    }, Change::None);
}
extern "C" int fort_scope_plan_abandon_remaining_v1(fort_scope_t h) {
    return with(h, [&](Context &c) {
        require(c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE && c.plan_installed,
                FORT_SCOPE_STATE, "suffix handoff requires an installed continuation schedule");
        require(!c.batch_active && !c.scratch_token, FORT_SCOPE_STATE,
                "suffix handoff during an active batch or scratch lease");
        for (const auto &entry : c.buffers)
            require(!entry.second->prepared, FORT_SCOPE_STATE, "suffix handoff during a prepared buffer access");
        // The compiler invokes this only at an independently proved boundary
        // before executing the fallback suffix. Accepted prefix commands and
        // their virtual definitions remain owned and are never replayed.
        abandon_planning_metadata(c);
        trace(c, "plan_abandon_remaining");
    }, Change::None);
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
    return fort_scope_plan_add_costs_v2(h, kind, unit, bindings, count, flops, memory_bytes,
                                      gpu_available, 0, 0);
}
extern "C" int fort_scope_plan_add_costs_v2(fort_scope_t h, uint32_t kind, uint64_t unit,
                                   const fort_scope_plan_binding *bindings, size_t count,
                                   double flops, double memory_bytes, int gpu_available,
                                   double cpu_numerical_seconds, double gpu_numerical_seconds) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "query_construction");
        require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE, "reset the planning query before recording");
        require((kind <= FORT_SCOPE_PLAN_FORGET || kind == FORT_SCOPE_PLAN_ORIGINAL_HOST) &&
                (gpu_available == 0 || gpu_available == 1) &&
                (kind != FORT_SCOPE_PLAN_ORIGINAL_HOST || !gpu_available),
                FORT_SCOPE_ARGUMENT, "invalid planning operation kind or availability");
        require(c.graph_planning || c.plan.size() < 256, FORT_SCOPE_BOUNDARY, "planning record budget exceeded");
        require(kind != FORT_SCOPE_PLAN_WORKER || unit, FORT_SCOPE_ARGUMENT, "worker planning unit requires an identity");
        require(std::isfinite(cpu_numerical_seconds) && cpu_numerical_seconds >= 0 &&
                std::isfinite(gpu_numerical_seconds) && gpu_numerical_seconds >= 0,
                FORT_SCOPE_ARGUMENT, "invalid offline numerical costs");
        c.plan.push_back({kind, unit, planning_bindings(c, bindings, count), flops, memory_bytes,
                          gpu_available != 0, cpu_numerical_seconds, gpu_numerical_seconds});
    }, Change::Query);
}
extern "C" int fort_scope_plan_add_compute_costs_v3(fort_scope_t h, uint32_t kind, uint64_t unit,
                                   const fort_scope_plan_binding *bindings, size_t count,
                                   double flops, double memory_bytes, int gpu_available,
                                   const fort_scope_compute_costs_v1 *compute) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "query_construction");
        require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE, "reset the planning query before recording");
        require((kind <= FORT_SCOPE_PLAN_FORGET || kind == FORT_SCOPE_PLAN_ORIGINAL_HOST) &&
                (gpu_available == 0 || gpu_available == 1) &&
                (kind == FORT_SCOPE_PLAN_WORKER || !gpu_available),
                FORT_SCOPE_ARGUMENT, "invalid planning operation kind or availability");
        require(c.graph_planning || c.plan.size() < 256, FORT_SCOPE_BOUNDARY, "planning record budget exceeded");
        require(kind != FORT_SCOPE_PLAN_WORKER || unit, FORT_SCOPE_ARGUMENT, "worker planning unit requires an identity");
        require(compute && fort_scoped::planning::detail::valid_compute_costs(*compute),
                FORT_SCOPE_ARGUMENT, "invalid offline backend compute costs");
        require(std::isfinite(flops) && flops >= 0 && std::isfinite(memory_bytes) && memory_bytes >= 0,
                FORT_SCOPE_ARGUMENT, "invalid planning work");
        fort_scoped::planning::Operation operation{kind, unit, planning_bindings(c, bindings, count), flops,
                                                  memory_bytes, gpu_available != 0};
        operation.compute_costs = *compute;
        c.plan.push_back(std::move(operation));
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
        require(c.graph_planning || c.plan.size() < 256, FORT_SCOPE_BOUNDARY, "planning record budget exceeded");
        c.plan.push_back({FORT_SCOPE_PLAN_TEAM_ENTRY, 0, {}, 0, 0, false});
    }, Change::Query);
}
extern "C" int fort_scope_plan_team_native_call_v1(fort_scope_t h) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "query_construction");
        require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE, "reset the planning query before recording");
        require(c.graph_planning || c.plan.size() < 256, FORT_SCOPE_BOUNDARY, "planning record budget exceeded");
        c.plan.push_back({FORT_SCOPE_PLAN_TEAM_NATIVE_CALL, 0, {}, 0, 0, false});
    }, Change::Query);
}
extern "C" int fort_scope_plan_forget_sections_v1(fort_scope_t h, fort_buffer_t handle,
                                                  const fort_scope_section *items, size_t count) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "query_construction");
        require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE, "reset the planning query before recording");
        require(c.graph_planning || c.plan.size() < 256, FORT_SCOPE_BOUNDARY, "planning record budget exceeded");
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
            require(c.graph_planning || !c.pending, FORT_SCOPE_STATE, "definition validation requires completed earlier execution");
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
static fort_scoped::planning::Result select_with_memo(Context &c,fort_scope_t h,
                                                     const fort_scope_plan_costs &costs) {
    const auto &input=planning_inputs(c,h);
    // A query without any GPU alternative cannot benefit from candidate reuse.
    // The normal selector still checks all metadata, costs and definitions.
    const bool gpu_supported=std::any_of(input.operations.begin(),input.operations.end(),[](const auto &op) {
        return fort_scoped::planning::detail::gpu_estimate_available(op);
    });
    if (!gpu_supported || !fort_scoped::memo::eligible(input) || c.scratch || c.scratch_capacity || c.scratch_token ||
            c.batch_active || !c.commands.empty()) return fort_scoped::planning::select(input,costs);
    static fort_scoped::memo::Cache cache;
    const auto started=PlanningClock::now();
    // Preview performs real, currently unpriced key work too. An older owner
    // report cannot remain qualified until the later installation call.
    c.owner_estimate_available=false;
    if(c.last_report){c.last_report->available=0;c.last_report->owner_available=0;}
    std::optional<fort_scoped::memo::Key> key;
    std::optional<std::vector<bool>> choices;
    try {
        std::vector<uint64_t> environment{static_cast<uint64_t>(c.device),
            c.queue_stats.queue_enabled,c.queue_stats.graphs_enabled,c.transfer_stats.requested_mode};
        auto append=[&](uint64_t value) {
            if(environment.size()>=fort_scoped::memo::byte_limit/sizeof(uint64_t))
                throw std::length_error("memo environment budget");
            environment.push_back(value);
        };
        auto remaining=[&] {return fort_scoped::memo::byte_limit/sizeof(uint64_t)-environment.size();};
        fort_scoped::metadata::execution_environment(append,remaining);
        key=fort_scoped::memo::make_key(input,costs,environment);
        if(key)choices=cache.find(*key);
    } catch(...) { } // Metadata memoization must remain optional.
    const double key_seconds=std::chrono::duration<double>(PlanningClock::now()-started).count();
    auto result=fort_scoped::planning::select(input,costs,choices ? &*choices : nullptr);
    result.unpriced_memo_overhead=true;result.memo_hit=choices.has_value();result.memo_key_seconds=key_seconds;
    result.report.available=0;result.report.owner_available=0;
    // A complete original-native fallback may become profitable with lower
    // future search debt. Do not pin that negative answer in the first version.
    if(key && result.decision.available && (input.continuation || result.decision.gpu_units))
        cache.insert(std::move(*key),result.gpu_workers);
    return result;
}
extern "C" int fort_scope_plan_select(fort_scope_t h, const fort_scope_plan_costs *costs,
                                      int compatible, fort_scope_plan_decision *out) {
    return with(h, [&](Context &c) {
        PlanningTimer timing(c, h, "selection");
        require(costs && out && compatible >= -1 && compatible <= 1, FORT_SCOPE_ARGUMENT, "invalid planning selection arguments");
        // Reject an older, shorter caller before copying or comparing the v2
        // payload. Rebuilding generated clients is required for the new ABI.
        const fort_scope_plan_costs absent_costs{};
        if (!costs->version) costs = &absent_costs;
        else require(costs->version == FORT_SCOPE_PLANNING_ABI_VERSION, FORT_SCOPE_ARGUMENT,
                     "incompatible planning cost ABI");
        require(c.plan_recording && !c.plan_installed, FORT_SCOPE_STATE, "planning selection requires a completed query");
        if (c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE) {
            require(c.graph_planning || !c.pending, FORT_SCOPE_STATE, "planning requires completed earlier execution");
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
        if(c.copy_costs)(void)planning_inputs(c,h); // Refresh shared credit/cache before a preview hit.
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
        auto result = timing.cache_hit ? *c.preview
            : compatible!=0 && !team_mismatch && !pinned_uncalibrated
                ? select_with_memo(c,h,pricing)
                : fort_scoped::planning::select(planning_inputs(c,h),pricing);
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
        // Includes preview: selecting GPU work anywhere in an observation
        // prevents learning a root-native hint, even if later fallback occurs.
        if(result.decision.gpu_units)native_hint_gpu_generation.advance();
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
            c.worker_cursor = 0; c.plan_activity_started = false;
            // COMPLETE native fallback owns no worker schedule. CONTINUE
            // retains ordered CPU choices until consumed or explicitly handed
            // to independently checked original source.
            c.plan_installed = c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE ||
                (result.decision.available && result.decision.gpu_units);
            if (!c.plan_installed) c.schedule.clear();
            c.plan_recording = false;
            decision_trace(c,*out, result.reason);
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
                FORT_SCOPE_PLAN_MISMATCH, "execution worker order or physical effects changed after planning");
        *gpu = c.schedule[c.worker_cursor++] ? 1 : 0;
    });
}
extern "C" int fort_scope_plan_native_begin_v1(fort_scope_t h,
        const fort_scope_native_worker_v1 *workers, size_t worker_count,
        const fort_scope_plan_binding *native_bindings, size_t native_count, int *applied) {
    if (applied) *applied = 0;
    return with(h, [&](Context &c) {
        const bool reuse_preparation=c.commands.empty();
        require(applied && (!worker_count || workers), FORT_SCOPE_ARGUMENT,
                "missing original native handoff descriptors");
        if (!worker_count) return;
        require(c.endpoint_mode == FORT_SCOPE_PLAN_CONTINUE && c.plan_installed,
                FORT_SCOPE_STATE, "original native handoff requires an installed continuation schedule");
        require(!c.batch_active && !c.scratch_token, FORT_SCOPE_STATE,
                "original native handoff during an active batch or scratch lease");
        for (const auto &entry : c.buffers)
            require(!entry.second->prepared, FORT_SCOPE_STATE,
                    "original native handoff during a prepared buffer access");
        require(c.worker_cursor <= c.schedule.size() &&
                worker_count <= c.schedule.size()-c.worker_cursor,
                FORT_SCOPE_PLAN_MISMATCH, "original native worker range exceeds the planned query");
        size_t worker = 0, matched = 0;
        bool all_cpu = true;
        for (const auto &op : c.plan) {
            if (op.kind != FORT_SCOPE_PLAN_WORKER) continue;
            if (worker++ < c.worker_cursor) continue;
            const auto &candidate = workers[matched];
            require(op.unit == candidate.unit && same_bindings(op.bindings,
                    planning_bindings(c,candidate.bindings,candidate.binding_count)),
                    FORT_SCOPE_PLAN_MISMATCH,
                    "original native worker order or physical effects changed after planning");
            all_cpu &= !c.schedule[c.worker_cursor+matched];
            if (++matched == worker_count) break;
        }
        require(matched == worker_count, FORT_SCOPE_PLAN_MISMATCH,
                "original native worker range is incomplete");
        if (!all_cpu) return;
        // Reject every bad/undefined native binding before draining an earlier
        // accepted producer. Native internal intermediates use their separately
        // proved aggregate effects, not a fabricated union of overwrite claims.
        const auto native = planning_bindings(c,native_bindings,native_count);
        std::vector<AccessPreparation> prepared;
        if (reuse_preparation) prepared.reserve(native.size());
        for (const auto &binding : native) {
            auto preparation=prepare_effects(virtual_buffer(c,binding.buffer),binding.effects,false);
            // With no queued commands on entry, virtual buffers are the real
            // buffers. Under this context mutex no drain can change them.
            if (reuse_preparation) prepared.push_back(std::move(preparation));
        }
        invalidate(c,Change::State);
        if (!reuse_preparation) {
            // A partial conflict drain need not materialize unrelated future
            // writes to this root. Rebuild from actual, never virtual, state.
            drain_effects(c,native);
            prepared.reserve(native.size());
            for (const auto &binding : native)
                prepared.push_back(prepare_effects(buffer(c,binding.buffer),binding.effects,false));
        }
        acquire_prepared_accesses(c,prepared,false,nullptr);
        c.worker_cursor += worker_count;
        note_numerical_activity(c);
        c.owner_estimate_available = false;
        c.segment_cost_pending = false;
        if (c.last_report) {
            c.last_report->available = 0;
            c.last_report->owner_available = 0;
        }
        *applied = 1;
        trace(c,"plan_native_begin",nullptr,worker_count);
    }, Change::None);
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
struct NativeHintFP {
    bool supported=false;
    std::array<uint64_t,4> controls{};
};
NativeHintFP native_hint_original_fp() noexcept {
    NativeHintFP result;
    // Check the original caller, not protect()'s temporarily masked traps.
#if defined(__GLIBC__) && defined(__SSE__) && (defined(__i386__) || defined(__x86_64__))
    if(!fort_runtime::numerical_environment_supported())return result;
    fpu_control_t control;_FPU_GETCW(control);
    result.supported=true;
    result.controls={uint64_t(std::fegetround()),uint64_t(fegetexcept()),
                     uint64_t(_mm_getcsr()&0xffc0U),uint64_t(control)};
#endif
    return result; // Unknown control representations quietly use ordinary execution.
}
bool native_hint_serial() noexcept {
#ifdef _OPENMP
    return omp_get_level()==0 && !omp_in_parallel();
#else
    return false;
#endif
}
void native_hint_trace(const char *phase,bool hit,bool eligible,bool learned,
                       const fort_scoped::native_hint::Key &key) noexcept {
    if(!trace_enabled())return;
    try {
        std::ostringstream line;
        line << "FORT_SCOPED native_hint {\"schema_version\":1,\"event\":\"native_hint\",\"phase\":\""
             << phase << "\",\"reason\":\"conservative_native_owner_hint\",\"hit\":" << (hit?"true":"false")
             << ",\"observation_eligible\":" << (eligible?"true":"false")
             << ",\"learned\":" << (learned?"true":"false")
             << ",\"estimates_available\":false,\"key_words\":" << key.count
             << ",\"key_hash\":" << key.hash << "}\n";
        std::lock_guard<std::mutex> lock(trace_mutex);std::cerr << line.str();
    } catch(...) {} // Optional evidence never changes original CPU semantics.
}
extern "C" int fort_scope_native_hint_begin_v1(const uint64_t *metadata_words,size_t word_count,
        fort_scope_native_hint_observation_v1 *observation,int *hit) {
    if(!observation || !hit)
        return fort_scope_report_error(FORT_SCOPE_ARGUMENT,"missing native hint outputs");
    observation->version=FORT_SCOPE_NATIVE_HINT_ABI_VERSION;
    observation->eligible=0;observation->word_count=0;
    observation->gpu_generation=0;observation->error_generation=0;*hit=0;
    const auto fp=native_hint_original_fp();
    if(!fp.supported || !native_hint_serial() || !word_count ||
            word_count>FORT_SCOPE_NATIVE_HINT_WORD_LIMIT || !metadata_words)return FORT_SCOPE_OK;
    fort_runtime::HostFloatingEnvironment floating_environment;
    if(!floating_environment.valid())return FORT_SCOPE_OK;
    try {
        uint64_t gpu,error;
        {
            std::lock_guard<std::mutex> lock(registry_mutex);
            if(!contexts.empty() || !domains.empty())return FORT_SCOPE_OK;
            gpu=native_hint_gpu_generation.value();error=native_hint_error_generation.value();
        }
        if(gpu==std::numeric_limits<uint64_t>::max() || error==std::numeric_limits<uint64_t>::max())
            return FORT_SCOPE_OK;
        fort_scoped::native_hint::Key key;
        key.integer(1);key.integer(word_count);
        for(size_t i=0;i<word_count;++i)key.integer(metadata_words[i]);
        for(auto value:fp.controls)key.integer(value);
        auto append=[&](uint64_t value){key.integer(value);};
        auto remaining=[&]{return fort_scoped::native_hint::word_limit-key.count;};
        fort_scoped::metadata::execution_environment(append,remaining);
        for(const char *name:{"FORT_SCOPE_GRAPH_QUEUE","FORT_SCOPE_CUDA_GRAPHS",
                              "CUDA_VISIBLE_DEVICES","CUDA_DEVICE_ORDER"})
            fort_scoped::metadata::string(append,remaining,std::getenv(name));
        bool found=false;
        {
            std::lock_guard<std::mutex> lock(registry_mutex);
            if(!contexts.empty() || !domains.empty() || native_hint_gpu_generation.value()!=gpu ||
                    native_hint_error_generation.value()!=error)return FORT_SCOPE_OK;
            found=native_hint_cache.find(key);
            if(!found) {
                observation->word_count=key.count;
                std::copy_n(key.words.begin(),key.count,observation->words);
                observation->gpu_generation=gpu;observation->error_generation=error;
                observation->eligible=1;
            }
        }
        *hit=found?1:0;
        native_hint_trace("lookup",found,!found,false,key);
    }catch(...) {observation->eligible=0;} // Optional key/cache failure is a quiet miss.
    return FORT_SCOPE_OK;
}
extern "C" int fort_scope_native_hint_finish_v1(fort_scope_native_hint_observation_v1 *observation) {
    if(!observation)return fort_scope_report_error(FORT_SCOPE_ARGUMENT,"missing native hint observation");
    if(!observation->version)return FORT_SCOPE_OK; // Inert default; inspect no later fields.
    if(observation->version!=FORT_SCOPE_NATIVE_HINT_ABI_VERSION)
        return fort_scope_report_error(FORT_SCOPE_ARGUMENT,"invalid native hint version");
    const auto eligible=observation->eligible;observation->eligible=0;
    if(!eligible)return FORT_SCOPE_OK;
    if(eligible!=1 ||
            !observation->word_count || observation->word_count>FORT_SCOPE_NATIVE_HINT_WORD_LIMIT)
        return fort_scope_report_error(FORT_SCOPE_ARGUMENT,"invalid native hint observation");
    if(!native_hint_serial() || !native_hint_original_fp().supported)return FORT_SCOPE_OK;
    fort_runtime::HostFloatingEnvironment floating_environment;
    if(!floating_environment.valid())return FORT_SCOPE_OK;
    try {
        fort_scoped::native_hint::Key key;
        for(size_t i=0;i<observation->word_count;++i)key.integer(observation->words[i]);
        bool learned=false;
        {
            std::lock_guard<std::mutex> lock(registry_mutex);
            const auto gpu=native_hint_gpu_generation.value(),error=native_hint_error_generation.value();
            if(contexts.empty() && domains.empty() && gpu!=std::numeric_limits<uint64_t>::max() &&
                    error!=std::numeric_limits<uint64_t>::max() &&
                    gpu==observation->gpu_generation && error==observation->error_generation) {
                learned=native_hint_cache.insert(key);
            }
        }
        native_hint_trace("finish",false,false,learned,key);
    }catch(...) {} // No plan, definition, data or status is changed on a failed observation.
    return FORT_SCOPE_OK;
}
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
        context->owner_estimate_available=false;
        if(context->last_report) {
            context->last_report->available=0;
            context->last_report->owner_available=0;
        }
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
extern "C" int fort_scope_serial_caller(void) {
#ifdef _OPENMP
    return !omp_in_parallel();
#else
    // Without OpenMP support this runtime cannot determine whether its host
    // is in a team. Unknown participation retains the original native span.
    return 0;
#endif
}
extern "C" int fort_scope_numerical_environment_supported(void) {
    return fort_runtime::numerical_environment_supported() ? 1 : 0;
}
extern "C" int fort_scope_device_get(fort_scope_t h, int *out) {
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing context device output");
        *out = c.device;
    }, Change::None);
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
extern "C" int fort_scope_set_transfers(fort_scope_t h, uint32_t mode) {
    return with(h, [&](Context &c) {
        require(mode <= FORT_SCOPE_TRANSFERS_COALESCED, FORT_SCOPE_ARGUMENT, "invalid scoped transfer mode");
        if(c.borrowed_domain && (!c.buffers.empty() || c.ready)) {
            require(!c.plan_recording && !c.plan_installed && mode==c.transfer_stats.requested_mode,
                    FORT_SCOPE_STATE,"cannot change a live residency transfer mode"); return;
        }
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
extern "C" int fort_scope_set_copy_costs_v2(fort_scope_t h,const fort_scope_copy_costs_v2 *costs,
        const fort_scope_plan_costs *base,int compatible) {
    return with(h,[&](Context &c) {
        require(compatible==0 || compatible==1,FORT_SCOPE_ARGUMENT,"invalid physical copy calibration compatibility");
        require((c.borrowed_domain || (c.buffers.empty() && !c.ready)) && !c.plan_recording && !c.plan_installed,
                FORT_SCOPE_STATE,"configure physical copy calibration before registration/planning");
        c.copy_costs.reset();
        if(compatible && costs)require(costs->version==FORT_SCOPE_COPY_COST_ABI_VERSION,
            FORT_SCOPE_ARGUMENT,"unsupported physical-copy cost ABI; expected version2");
        if(!compatible || !costs || !base || !fort_physical::compact::valid_costs(*costs))return;
        try {fort_scoped::planning::detail::validate(fort_scoped::planning::Inputs{},*base);}
        catch(const fort_scoped::planning::detail::Unavailable &) {return;}
        c.copy_costs=fort_physical::compact::Pricing{*costs,*base};
        if(c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_AUTO) {
            c.transfer_stats.fallback_reason=FORT_SCOPE_TRANSFER_NONE;c.transfer_stats.fallbacks=0;
        }
    });
}
extern "C" int fort_scope_set_transfer_costs_v1(fort_scope_t h, const fort_scope_batch_costs *costs,
                                               int compatible) {
    return with(h, [&](Context &c) {
        require(compatible == 0 || compatible == 1, FORT_SCOPE_ARGUMENT, "invalid transfer calibration compatibility");
        require((c.borrowed_domain || (c.buffers.empty() && !c.ready)) && !c.plan_recording && !c.plan_installed,
                FORT_SCOPE_STATE, "configure scoped transfer pricing before registration and planning");
        c.transfer_costs.reset();
        if (compatible && costs && fort_scoped::planning::detail::valid_transfer_costs(*costs)) {
            c.transfer_costs = *costs;
            if (!c.borrowed_domain && (c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_AUTO || c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_PIPELINED)) {
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
            if (compatible!=-1) { c.batch_report=*out; batch_trace(c,h,*out); }
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
            ++c.transfer_stats.tiles; trace(c, upload ? "upload" : "download",copy.buffer,bytes);
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
            c.pending=false; complete_all(c); c.batch_active=false;
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
extern "C" int fort_scope_identity_swap_v1(fort_scope_t h,fort_buffer_t first,fort_buffer_t second,
                                            const fort_scope_layout *first_after,const fort_scope_layout *second_after,
                                            fort_buffer_t *first_current,fort_buffer_t *second_current) {
    return with(h,[&](Context &c) {
        // The original descriptor mutation has already happened. No rejection
        // from this point can authorize replay or publication through old names.
        try {
            require(first!=second && first_current && second_current && first_current!=second_current,
                    FORT_SCOPE_ARGUMENT,"identity exchange needs distinct physical handles and output slots");
            require(!c.batch_active && (!c.plan_installed || c.worker_cursor==c.schedule.size()),
                    FORT_SCOPE_STATE,"identity exchange crosses unfinished execution planning");
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
            // A validated but uninstalled query is metadata, not an execution
            // obligation (forced sections/native fallback can leave one).
            // It cannot span the source-proved descriptor exchange frontier.
            abandon_planning_metadata(c);
            // Existing map slots and scalar fields commit without allocation.
            // Queue bindings/upload readers still refer to the same Buffer and
            // physical handle; no pending read or numerical work is replayed.
            const IdentitySwapTrace exchange{a.identity,b.identity,first,second,a.generation,b.generation};
            ai->second=second; bi->second=first;
            std::swap(a.identity,b.identity);
            *first_current=second; *second_current=first;
            c.owner_estimate_available=false;
            if(c.last_report) { c.last_report->available=0; c.last_report->owner_available=0; }
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
extern "C" int fort_scope_forget_sections_v1(fort_scope_t h, fort_buffer_t handle,
                                             const fort_scope_section *items, size_t count) {
    return with(h, [&](Context &c) {
        drain_resources(c,&handle,1);
        auto &b = buffer(c, handle);
        require(!b.prepared, FORT_SCOPE_STATE, "definition change during a prepared buffer access");
        const auto removed = sections(b, items, count, false);
        Budget budget;
        // Do not partially invalidate state if any bounded subtraction fails.
        auto initialized = difference(b.initialized, removed, budget);
        b.initialized = std::move(initialized);
        trace(c, "forget_sections", &b);
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
extern "C" int fort_scope_view_get_v2(fort_scope_t h, const fort_scope_view_v2 *view,
                                       fort_scope_view_layout_v2 *out) {
    return with(h, [&](Context &c) {
        require(view && out && view->version == FORT_SCOPE_VIEW_ABI_VERSION_V2 && !view->reserved &&
                view->rank && view->root_rank && view->rank <= view->root_rank &&
                view->origins && view->extents && view->lower_bounds && view->axes,
                FORT_SCOPE_ARGUMENT, "invalid rank-reduced borrowed view descriptor");
        auto &b = buffer(c, view->buffer);
        require(view->generation == b.generation, FORT_SCOPE_STALE, "borrowed view allocation generation changed");
        require(view->root_rank == b.extents.size(), FORT_SCOPE_ARGUMENT, "borrowed view root rank differs from allocation");
        const bool is_empty = std::find(view->extents, view->extents+view->rank, size_t(0)) != view->extents+view->rank;
        size_t elements = is_empty ? 0 : 1, offset = 0;
        for (size_t axis=0; axis<view->root_rank; ++axis) {
            require(view->origins[axis] <= b.extents[axis], FORT_SCOPE_ARGUMENT, "borrowed view origin exceeds allocation");
            bool retained = false;
            for (size_t k=0; k<view->rank; ++k) retained |= view->axes[k] == axis;
            if (!retained && !is_empty)
                require(view->origins[axis] < b.extents[axis], FORT_SCOPE_ARGUMENT, "borrowed view fixed coordinate exceeds allocation");
            if (!is_empty) {
                const size_t part = multiply(view->origins[axis], b.strides[axis]);
                require(part <= std::numeric_limits<size_t>::max()-offset,
                        FORT_SCOPE_ARGUMENT, "borrowed view byte offset overflow");
                offset += part;
            }
        }
        for (size_t k=0; k<view->rank; ++k) {
            const size_t axis=view->axes[k];
            require(axis < view->root_rank, FORT_SCOPE_ARGUMENT, "borrowed view axis exceeds root rank");
            for (size_t previous=0; previous<k; ++previous)
                require(view->axes[previous] != axis, FORT_SCOPE_ARGUMENT, "borrowed view repeats a root axis");
            require(view->extents[k] <= b.extents[axis]-view->origins[axis],
                    FORT_SCOPE_ARGUMENT, "borrowed view extent exceeds allocation");
            const int64_t lower=view->lower_bounds[k];
            require(lower >= INT32_MIN && lower <= INT32_MAX && view->extents[k] <= INT32_MAX,
                    FORT_SCOPE_BOUNDARY, "borrowed view bounds exceed INTEGER ABI");
            if (view->extents[k])
                require(int64_t(view->extents[k]-1) <= INT32_MAX-lower,
                        FORT_SCOPE_BOUNDARY, "borrowed view upper bound exceeds INTEGER ABI");
            if (!is_empty) elements=multiply(elements,view->extents[k]);
        }
        *out = {{static_cast<uint32_t>(b.extents.size()), b.type, b.element_bytes, b.host,
                 b.extents.data(), b.lower.data(), b.generation}, view->rank, 0,
                view->origins, view->extents, b.strides.data(), view->lower_bounds, view->axes, elements, offset};
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
        require(a, FORT_SCOPE_ARGUMENT, "invalid access descriptor");
        note_numerical_activity(c);
        const fort_scope_plan_binding binding{b, *a};
        begin_accesses(c, &binding, 1, false, nullptr);
    });
}
extern "C" int fort_scope_host_end(fort_scope_t h, fort_buffer_t b) {
    return with(h, [&](Context &c) { end(c, buffer(c,b), false); });
}
extern "C" int fort_scope_device_begin(fort_scope_t h, fort_buffer_t b, const fort_scope_access *a, void **out) {
    native_hint_gpu_generation.advance();
    return with(h, [&](Context &c) {
        require(out && a, FORT_SCOPE_ARGUMENT, "missing device pointer output or access descriptor");
        note_numerical_activity(c);
        const fort_scope_plan_binding binding{b, *a};
        begin_accesses(c, &binding, 1, true, out);
    });
}
extern "C" int fort_scope_access_begin_v1(fort_scope_t h, const fort_scope_plan_binding *bindings,
                                           size_t count, int device, void **devices) {
    if(count && device==1)native_hint_gpu_generation.advance();
    return with(h, [&](Context &c) {
        require(device == 0 || device == 1, FORT_SCOPE_ARGUMENT, "invalid task access execution mode");
        if (count) note_numerical_activity(c);
        begin_accesses(c, bindings, count, device != 0, devices);
    });
}
extern "C" int fort_scope_original_host_begin_v1(fort_scope_t h, const fort_scope_plan_binding *bindings,
                                                  size_t count) {
    return with(h, [&](Context &c) {
        if (count) note_numerical_activity(c);
        begin_accesses(c, bindings, count, false, nullptr, AccessContract::OriginalHost);
    });
}
static int execute_external_transaction(fort_scope_t h,const fort_scope_external_call_v1 *call,
                                        fort_scope_external_worker_v1 worker,
                                        fort_scope_source_worker_v1 source_worker, bool source_task,
                                        uint64_t expected_launches, void *user,
                                        fort_scope_external_report_v1 *out,
                                        AccessContract contract=AccessContract::Defined) {
    if(out) { *out={}; out->version=FORT_SCOPE_EXTERNAL_ABI_VERSION; }
    if(call && call->binding_count) native_hint_gpu_generation.advance();
    const int status=with(h,[&](Context &c) {
        require(call && out && call->version==FORT_SCOPE_EXTERNAL_ABI_VERSION && !call->reserved &&
                (!call->binding_count || (call->bindings && (source_task ? bool(source_worker) : bool(worker)))),
                FORT_SCOPE_ARGUMENT,"invalid external transaction");
        require(!source_task || (call->binding_count ? expected_launches>0 : expected_launches==0),
                FORT_SCOPE_ARGUMENT,"source task requires its checked nonempty launch count");
        if(!call->binding_count) return;
        if(source_task) c.owner_estimate_available=false;
        require(!c.batch_active,FORT_SCOPE_STATE,"external transaction during active batch");
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
            begin_accesses(c,bindings.data(),bindings.size(),true,devices.data(),contract);
            initialize(c);
            for(size_t i=0;i<views.size();++i) views[i].device=devices[i];
            note_numerical_activity(c);
            int started=-1, result=FORT_SCOPE_EXECUTION;
            uint64_t accepted_launches=0;
            auto record_source_launches=[&]() {
                if(!source_task) return;
                require(accepted_launches<=expected_launches &&
                        accepted_launches<=std::numeric_limits<decltype(c.stats.launches)>::max()-c.stats.launches &&
                        accepted_launches<=std::numeric_limits<decltype(c.queue_stats.direct_kernels)>::max()-c.queue_stats.direct_kernels,
                        FORT_SCOPE_EXECUTION,"source task accepted launch count exceeds its checked limit");
                c.stats.launches+=accepted_launches;
                c.queue_stats.direct_kernels+=accepted_launches;
                if(accepted_launches) mark_context_pending(c);
                for(uint64_t i=0;i<accepted_launches;++i) trace(c,"launch");
            };
            try {
#ifndef FORT_SCOPE_CPU_TEST
                DeviceGuard guard(c);
                void *stream=reinterpret_cast<void *>(c.stream);
#else
                void *stream=nullptr;
#endif
                result=source_task ? source_worker(user,views.data(),views.size(),stream,&started,&accepted_launches)
                                   : worker(user,views.data(),views.size(),stream,&started);
            } catch(...) {
                submission_uncertain=true;
                out->started=1; c.poisoned=true;
                record_source_launches();
                throw Error(FORT_SCOPE_EXECUTION,"external callback exception; unsafe replay prohibited");
            }
            submission_uncertain=started!=0 || accepted_launches!=0;
            out->started=started==0 ? 0u : 1u;
            record_source_launches();
            if(source_task && ((accepted_launches && started!=1) ||
                               (result==FORT_SCOPE_OK && accepted_launches!=expected_launches))) {
                c.poisoned=true;
                throw Error(FORT_SCOPE_EXECUTION,"source task submission count differs; unsafe replay prohibited");
            }
            require(!c.poisoned,FORT_SCOPE_EXECUTION,"external device restoration failed");
            if((started!=0 && started!=1) || (result==FORT_SCOPE_OK && started!=1) ||
                    (result!=FORT_SCOPE_OK && (started || result==FORT_SCOPE_EXECUTION ||
                                              result<FORT_SCOPE_ARGUMENT || result>FORT_SCOPE_PLAN_MISMATCH))) {
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
extern "C" int fort_scope_external_execute_v1(fort_scope_t h,const fort_scope_external_call_v1 *call,
                                               fort_scope_external_worker_v1 worker,void *user,
                                               fort_scope_external_report_v1 *out) {
    return execute_external_transaction(h,call,worker,nullptr,false,0,user,out);
}
extern "C" int fort_scope_source_execute_v1(fort_scope_t h,const fort_scope_external_call_v1 *call,
                                             uint64_t expected_launches,fort_scope_source_worker_v1 worker,
                                             void *user,fort_scope_external_report_v1 *out) {
    return execute_external_transaction(h,call,nullptr,worker,true,expected_launches,user,out);
}
extern "C" int fort_scope_source_execute_v2(fort_scope_t h,const fort_scope_external_call_v1 *call,
                                             uint64_t expected_launches,fort_scope_source_worker_v1 worker,
                                             void *user,fort_scope_external_report_v1 *out) {
    return execute_external_transaction(h,call,nullptr,worker,true,expected_launches,user,out,
                                        AccessContract::OriginalDevice);
}
extern "C" int fort_scope_source_host_prepare_v1(fort_scope_t h,const fort_scope_external_call_v1 *call,
                                                 uint64_t *out) {
    if(out) *out=0;
    return with(h,[&](Context &c) {
        require(out && call && call->version==FORT_SCOPE_EXTERNAL_ABI_VERSION && !call->reserved &&
                call->binding_count && call->bindings && !call->workspace_lease && !call->workspace_bytes,
                FORT_SCOPE_ARGUMENT,"invalid original CPU source transaction");
        require(!c.source_host_task && !c.batch_active && !c.scratch_token &&
                (!c.plan_installed || c.worker_cursor==c.schedule.size()),
                FORT_SCOPE_STATE,"original CPU source transaction overlaps active execution");
        for(const auto &entry:c.buffers)
            require(!entry.second->prepared,FORT_SCOPE_STATE,"original CPU source transaction overlaps prepared access");
        Context::SourceHostTask task{token(),false,{}};
        task.buffers.reserve(call->binding_count);
        std::vector<fort_scope_plan_binding> bindings;bindings.reserve(call->binding_count);
        std::unordered_set<fort_buffer_t> seen;
        for(size_t i=0;i<call->binding_count;++i) {
            const auto &binding=call->bindings[i];auto &b=buffer(c,binding.buffer);
            require(seen.insert(binding.buffer).second,FORT_SCOPE_ALIAS,"CPU source repeats a canonical resource");
            require(binding.type==b.type && binding.rank==b.extents.size() && binding.element_bytes==b.element_bytes,
                    FORT_SCOPE_ARGUMENT,"CPU source type or layout mismatch");
            bindings.push_back({b.handle,binding.access});task.buffers.push_back(b.handle);
        }
        // Validate the entire declared read/preservation set against queued
        // definitions too, before any physical acquisition/submission occurs.
        for(const auto &binding:bindings)
            (void)prepare_access(virtual_buffer(c,binding.buffer),&binding.access,false);
        try {
            begin_accesses(c,bindings.data(),bindings.size(),false,nullptr);
            c.source_host_task=std::move(task);
            c.owner_estimate_available=false;
            *out=c.source_host_task->token;
            trace(c,"source_native_prepare",nullptr,bindings.size());
        } catch(...) {
            for(const auto &binding:bindings) buffer(c,binding.buffer).prepared.reset();
            throw;
        }
    });
}
extern "C" int fort_scope_source_host_start_v1(fort_scope_t h,uint64_t token) {
    return with(h,[&](Context &c) {
        require(c.source_host_task && c.source_host_task->token==token,FORT_SCOPE_STALE,"foreign or stale CPU source token");
        require(!c.source_host_task->started,FORT_SCOPE_STATE,"CPU source work was already started");
        c.source_host_task->started=true;
        note_numerical_activity(c);
        trace(c,"source_native_start",nullptr,c.source_host_task->buffers.size());
    },Change::State,true);
}
extern "C" int fort_scope_source_host_complete_v1(fort_scope_t h,uint64_t token) {
    return with(h,[&](Context &c) {
        require(c.source_host_task && c.source_host_task->token==token,FORT_SCOPE_STALE,"foreign or stale CPU source token");
        require(c.source_host_task->started,FORT_SCOPE_STATE,"CPU source completion before original work");
        for(auto handle:c.source_host_task->buffers) {
            const auto &b=buffer(c,handle);
            require(b.prepared && !b.prepared->device,FORT_SCOPE_EXECUTION,"CPU source lost prepared access");
        }
        for(auto handle:c.source_host_task->buffers) end(c,buffer(c,handle),false);
        trace(c,"source_native_commit",nullptr,c.source_host_task->buffers.size());
        c.source_host_task.reset();
    },Change::State,true);
}
extern "C" int fort_scope_source_host_cancel_v1(fort_scope_t h,uint64_t token) {
    return with(h,[&](Context &c) {
        require(c.source_host_task && c.source_host_task->token==token,FORT_SCOPE_STALE,"foreign or stale CPU source token");
        require(!c.source_host_task->started,FORT_SCOPE_EXECUTION,"CPU source cancellation after original work started");
        for(auto handle:c.source_host_task->buffers) buffer(c,handle).prepared.reset();
        trace(c,"source_native_cancel",nullptr,c.source_host_task->buffers.size());
        c.source_host_task.reset();
    },Change::State,true);
}
static int source_quote(fort_scope_t h,const fort_scope_external_call_v1 *call,
        const fort_scope_source_work_v1 *work,const fort_scope_plan_costs *base,
        const fort_scope_source_costs_v1 *protocol,const fort_scope_source_host_costs_v1 *native_costs,
        int compatible,fort_scope_source_quote_result_v1 *out) {
    if(out) { *out={};out->version=native_costs ? FORT_SCOPE_SOURCE_QUOTE_LIVE_ABI_VERSION : FORT_SCOPE_SOURCE_QUOTE_ABI_VERSION; }
    return with(h,[&](Context &c) {
        require(call && work && base && protocol && out,FORT_SCOPE_ARGUMENT,"missing source quote metadata");
        require(call->version==FORT_SCOPE_EXTERNAL_ABI_VERSION && !call->reserved &&
                (!call->binding_count || call->bindings),FORT_SCOPE_ARGUMENT,"invalid source quote binding list");
        fort_scoped::source_quote::Snapshot snapshot;
        snapshot.device_budget=c.device_budget;snapshot.workspace_bytes=c.residency->workspace_bytes;
        snapshot.scratch_capacity=c.scratch_capacity;snapshot.scratch_active=c.scratch_token!=0 || c.batch_active ||
            (c.plan_installed && c.worker_cursor!=c.schedule.size());
        snapshot.queue_pending=!c.commands.empty();
        // A whole-domain native fallback destroys opaque providers first.
        // Their cleanup is outside this source/scratch protocol calibration.
        snapshot.opaque_workspace=!c.residency->workspaces.empty() || call->workspace_lease || call->workspace_bytes;
        snapshot.opaque_workspace_claim=call->workspace_lease || call->workspace_bytes;
        auto &input=snapshot.input;
        input.device_ready=c.ready;input.pending=c.pending;input.driver_initialized=driver_initialized(c);
        input.transfer_mode=c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_PINNED ?
            FORT_SCOPE_TRANSFERS_PINNED : c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_AUTO && c.copy_costs ?
            FORT_SCOPE_TRANSFERS_AUTO : FORT_SCOPE_TRANSFERS_DIRECT;
        input.direct_transfer_fit_applicable=c.transfer_stats.requested_mode==FORT_SCOPE_TRANSFERS_DIRECT;
        input.transfer_costs=c.transfer_costs;
        input.copy_costs=c.copy_costs;
        copy_pool_inputs(c,input);
#ifndef FORT_SCOPE_CPU_TEST
#if CUDART_VERSION >= 11020
        input.asynchronous_release=c.ready && c.pool!=nullptr;
#else
        input.asynchronous_release=false;
#endif
#else
        input.asynchronous_release=false;
#endif
        for(const auto &entry:c.buffers) {
            const auto &b=*entry.second;
            if(b.prepared) snapshot.scratch_active=true;
            input.resources.push_back({b.handle,b.element_bytes,b.bytes,b.extents,b.initialized,b.host_current,
                                       b.device_current,b.device!=nullptr,pending_host_reads(b),b.last_device_use});
        }
        std::vector<fort_scoped::planning::Binding> bindings;
        bindings.reserve(call->binding_count);
        std::unordered_set<fort_buffer_t> seen;
        for(size_t index=0;index<call->binding_count;++index) {
            const auto &binding=call->bindings[index];const auto &b=buffer(c,binding.buffer);
            require(seen.insert(binding.buffer).second,FORT_SCOPE_ALIAS,"source quote repeats a canonical resource");
            require(binding.type==b.type && binding.rank==b.extents.size() && binding.element_bytes==b.element_bytes,
                    FORT_SCOPE_ARGUMENT,"source quote type or rank mismatch");
            bindings.push_back({b.handle,effects(b,&binding.access)});
        }
        *out=fort_scoped::source_quote::price(std::move(snapshot),bindings,*work,*base,*protocol,compatible!=0,native_costs);
    },Change::None);
}
extern "C" int fort_scope_source_quote_v1(fort_scope_t h,const fort_scope_external_call_v1 *call,
        const fort_scope_source_work_v1 *work,const fort_scope_plan_costs *base,
        const fort_scope_source_costs_v1 *protocol,int compatible,fort_scope_source_quote_result_v1 *out) {
    return source_quote(h,call,work,base,protocol,nullptr,compatible,out);
}
extern "C" int fort_scope_source_quote_v2(fort_scope_t h,const fort_scope_external_call_v1 *call,
        const fort_scope_source_work_v1 *work,const fort_scope_plan_costs *base,
        const fort_scope_source_costs_v1 *protocol,const fort_scope_source_host_costs_v1 *native_costs,
        int compatible,fort_scope_source_quote_result_v1 *out) {
    if(out) {*out={};out->version=FORT_SCOPE_SOURCE_QUOTE_LIVE_ABI_VERSION;}
    if(!native_costs) return protect([&] {throw Error(FORT_SCOPE_ARGUMENT,"missing native-live protocol costs");});
    return source_quote(h,call,work,base,protocol,native_costs,compatible,out);
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
    if (count) note_numerical_activity(c);
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
    if(count)native_hint_gpu_generation.advance();
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
        ++c.stats.launches; mark_context_pending(c); trace(c, "launch");
    });
}
extern "C" int fort_scope_graph_submit_v1(fort_scope_t h, uint64_t key,
                                           const fort_scope_kernel_v1 *kernels, size_t count) {
    if(count)native_hint_gpu_generation.advance();
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
        note_numerical_activity(c);
        submit_graph(c,key,kernels,count);
#endif
    });
}
extern "C" int fort_scope_graph_execute_v1(fort_scope_t h, uint64_t key,
                                            const fort_scope_graph_task_v1 *tasks, size_t count) {
    if(count)native_hint_gpu_generation.advance();
    return with(h, [&](Context &c) {
        if (count) flush_commands(c);
        if (count) note_numerical_activity(c);
        execute_graph(c,key,tasks,count);
    });
}
extern "C" int fort_scope_graph_enqueue_v1(fort_scope_t h,uint64_t key,
                                            const fort_scope_graph_task_v1 *task,const size_t *sizes) {
    native_hint_gpu_generation.advance();
    return with(h,[&](Context &c) { enqueue_graph(c,key,task,sizes); },Change::Query);
}
extern "C" int fort_scope_graph_flush_v1(fort_scope_t h) {
    return with(h,[&](Context &c) { flush_commands(c); });
}
extern "C" int fort_scope_graph_queue_cancel_v1(fort_scope_t h) {
    return with(h,[&](Context &c) {
        c.commands.clear(); c.virtual_buffers.clear(); c.virtual_generation=0; c.queue_stats.pending_tasks=0;
        trace(c,"queue_cancel");
    },Change::Query);
}
extern "C" int fort_scope_graph_queue_stats_get_v1(fort_scope_t h,fort_scope_graph_queue_stats_v1 *out) {
    return with(h,[&](Context &c) {
        require(out,FORT_SCOPE_ARGUMENT,"missing queued graph statistics output"); *out=c.queue_stats;
    },Change::None);
}
extern "C" int fort_scope_task_graph_enable_v1(fort_scope_t h) {
    return with(h,[&](Context &c) { c.graph_planning=true; },Change::Query);
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
        c->owner_estimate_available = false;
        c->poisoned = true;
        throw Error(FORT_SCOPE_EXECUTION, reason ? reason : "scoped numerical execution failed");
    });
}
extern "C" int fort_scope_report_error(int status, const char *message) {
    note_hint_error(status);
    if (status != FORT_SCOPE_OK) diagnostic(message ? message : "scoped entry error");
    return status;
}
extern "C" int fort_scope_wait(fort_scope_t h) {
    return with(h, [&](Context &c) { flush_commands(c); wait(c); complete_segment_estimate(c); });
}
extern "C" int fort_scope_stats_get(fort_scope_t h, fort_scope_stats *out) {
    return with(h, [&](Context &c) { require(out, FORT_SCOPE_ARGUMENT, "missing stats output"); *out = c.stats; }, Change::None);
}
extern "C" int fort_scope_scratch_acquire_v1(fort_scope_t h, size_t bytes, fort_scope_scratch_lease_v1 *out) {
    if (out) *out = {};
    return with(h, [&](Context &c) {
        require(out, FORT_SCOPE_ARGUMENT, "missing scratch lease output");
        require(!c.scratch_token, FORT_SCOPE_STATE, "only one scratch lease may be active");
        require(!c.batch_active, FORT_SCOPE_STATE, "scratch acquisition during an active batch");
        // Reserve a globally unique token before allocating. A failed request
        // never exposes a pointer or an active lease, even after arena growth.
        const auto lease_token = token();
        if (bytes) {
            // Existing planning costs do not include scratch lifetime costs.
            // Do not present a complete-owner estimate for unpriced resources.
            c.owner_estimate_available = false;
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
        require(!c.plan_installed || c.worker_cursor == c.schedule.size(), FORT_SCOPE_STATE,
                "scope closed before its scheduled workers completed");
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
