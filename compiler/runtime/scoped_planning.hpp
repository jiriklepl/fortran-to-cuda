/* Pure, bounded complete-scope placement. This code never reads array values,
 * initializes CUDA, or changes the execution context. Region transitions and
 * physical copy operations are shared with scoped_runtime.cu. */
#ifndef FORT_SCOPED_PLANNING_HPP
#define FORT_SCOPED_PLANNING_HPP
#include "scoped_runtime.h"
#include "scoped_regions.hpp"
#include "section_copy.hpp"
#include <cmath>
#include <limits>
#include <new>
#include <string>
#include <unordered_map>

namespace fort_scoped { namespace planning {
using coherence::Box;
using coherence::Region;
using coherence::Effects;
struct Resource {
    fort_buffer_t handle = 0;
    size_t element_bytes = 0, bytes = 0;
    std::vector<size_t> extents;
    Region initialized, host_current, device_current;
    bool allocated = false;
};
struct Binding { fort_buffer_t buffer; Effects effects; };
struct Operation {
    uint32_t kind = FORT_SCOPE_PLAN_NATIVE;
    uint64_t unit = 0;
    std::vector<Binding> bindings;
    double flops = 0, memory_bytes = 0;
    bool gpu_available = false;
};
struct Inputs {
    // Preserve the runtime's publication order: each publication can wait.
    std::vector<Resource> resources;
    std::vector<Operation> operations;
    size_t device_budget = std::numeric_limits<size_t>::max();
    bool device_ready = false, driver_initialized = false, pending = false;
    uint64_t query_construction_operations = 0;
    // The CUDA pool backend queues frees on its stream. A synchronous backend
    // may set this false; it has no final free-completion wait.
    bool asynchronous_release = true;
    // Internal authority from the context's exact query/state generation.
    // Standalone clients must leave this false unless preflight succeeded.
    bool definitions_validated = false;
    bool continuation = false, charge_create = true;
    size_t registrations_incurred = std::numeric_limits<size_t>::max();
    uint32_t transfer_mode = FORT_SCOPE_TRANSFERS_DIRECT;
    std::optional<fort_scope_batch_costs> transfer_costs;
    std::optional<fort_scope_team_costs> team_costs;
};
struct Result {
    fort_scope_plan_decision decision{};
    // Only active WORKER records consume a choice, in their source order.
    std::vector<bool> gpu_workers;
    std::string reason;
    // Unknown fixed native helper computation is an identical common term in
    // every schedule. It is excluded from both time estimates, never guessed.
    bool native_common_compute_excluded = false;
    // Other contexts may initialize CUDA between a preview and its final trace.
    bool driver_initialized = false;
    // A fresh host-only native choice needs no coherence-state simulation.
    bool native_startup_shortcut = false;
    // Fresh collective owners can close metadata and execute original calls.
    bool collective_whole_owner_margin = false;
    fort_scope_plan_report report{};
};
struct DefinitionValidation {
    int status = FORT_SCOPE_BOUNDARY;
    const char *reason = "definition_plan_unavailable";
    // An operation is identified only after structural metadata validation.
    size_t operation = std::numeric_limits<size_t>::max();
    fort_buffer_t buffer = 0;
};

// Optional, synchronous evidence from one final schedule. Search never installs
// a sink, and events borrow existing state rather than retaining snapshots.
struct EvidenceEvent {
    const char *event = nullptr, *phase = nullptr;
    const Resource *resource = nullptr;
    const Operation *operation = nullptr;
    const Binding *binding = nullptr;
    const Region *preservation_reads = nullptr;
    const Box *rectangle = nullptr;
    uint64_t unit = 0, bytes = 0, copies = 0;
    size_t first = 0, last = 0;
    bool gpu = false, upload = false, accepted = false;
    double seconds = 0, counterfactual_seconds = 0, required_saving = 0;
};
struct EvidenceSink {
    void *context;
    void (*emit)(void *, const EvidenceEvent &);
    void operator()(const EvidenceEvent &event) const { emit(context, event); }
};

namespace detail {
constexpr size_t operation_limit = 256, worker_limit = 64;
constexpr size_t frontier_limit = 16, candidate_limit = 128;
constexpr size_t batch_capacities[] = {256*1024, 1024*1024, 4*1024*1024, 16*1024*1024};
struct Unavailable { const char *reason; };
inline void require(bool test, const char *reason) { if (!test) throw Unavailable{reason}; }
inline uint64_t add(uint64_t a, uint64_t b) {
    require(b <= std::numeric_limits<uint64_t>::max()-a, "arithmetic_overflow"); return a+b;
}
inline size_t product(size_t a, size_t b) {
    require(!b || a <= std::numeric_limits<size_t>::max()/b, "arithmetic_overflow"); return a*b;
}
inline void seconds(double &value, double delta) {
    require(std::isfinite(delta) && delta >= 0, "invalid_cost_estimate");
    value += delta; require(std::isfinite(value), "arithmetic_overflow");
}
inline double compute(const Operation &op, const fort_scope_plan_costs &costs, bool gpu) {
    return std::max(op.flops/(gpu ? costs.gpu_flops : costs.cpu_flops),
                    op.memory_bytes/(gpu ? costs.gpu_bandwidth : costs.cpu_bandwidth));
}
inline bool valid_team_costs(const fort_scope_team_costs &costs) {
    if (costs.version != FORT_SCOPE_TEAM_ABI_VERSION || !costs.valid || !costs.cpu_threads ||
        costs.expected_omp_level != 1 || costs.protocol_id != FORT_SCOPE_TEAM_PROTOCOL_ID) return false;
    if (!std::isfinite(costs.native_cpu_flops) || costs.native_cpu_flops <= 0 ||
        !std::isfinite(costs.native_cpu_bandwidth) || costs.native_cpu_bandwidth <= 0) return false;
    for (double value : {costs.owner_seconds, costs.descriptor_seconds, costs.entry_seconds,
                         costs.cpu_worker_seconds, costs.gpu_worker_seconds,
                         costs.native_call_seconds, costs.native_worker_seconds})
        if (!std::isfinite(value) || value < 0) return false;
    return true;
}
inline bool definition_only(const Operation &op) {
    return op.kind == FORT_SCOPE_PLAN_FORGET || op.kind == FORT_SCOPE_PLAN_DISCARD ||
        op.kind == FORT_SCOPE_PLAN_TEAM_ENTRY || op.kind == FORT_SCOPE_PLAN_TEAM_NATIVE_CALL;
}
inline double native_compute(const Operation &op, const Inputs &input, const fort_scope_plan_costs &costs) {
    if (!input.team_costs) return compute(op, costs, false);
    if (definition_only(op)) return 0;
    const auto &team = *input.team_costs;
    return std::max(op.flops/team.native_cpu_flops, op.memory_bytes/team.native_cpu_bandwidth) +
        (op.kind == FORT_SCOPE_PLAN_WORKER ? team.native_worker_seconds : 0);
}
inline bool valid_transfer_costs(const fort_scope_batch_costs &costs) {
    if (costs.version != FORT_SCOPE_BATCH_ABI_VERSION || !costs.valid ||
        costs.max_slot_bytes != batch_capacities[3]) return false;
    for (double value : {costs.event_record_seconds, costs.event_wait_seconds, costs.ready_event_seconds,
                         costs.preparation_operation_seconds, costs.pack_bytes_per_second,
                         costs.unpack_bytes_per_second, costs.pinned_h2d_bandwidth, costs.pinned_d2h_bandwidth})
        if (!std::isfinite(value) || value <= 0) return false;
    for (double value : {costs.pack_row_seconds, costs.unpack_row_seconds,
                         costs.pinned_h2d_latency, costs.pinned_d2h_latency})
        if (!std::isfinite(value) || value < 0) return false;
    for (size_t k=0; k<FORT_SCOPE_BATCH_CAPACITIES; ++k)
        if (!std::isfinite(costs.staging_cold_seconds[k]) || costs.staging_cold_seconds[k] <= 0 ||
            !std::isfinite(costs.staging_reuse_seconds[k]) || costs.staging_reuse_seconds[k] <= 0) return false;
    return true;
}
inline void validate_region(const Region &region, const Resource &b) {
    require(region.size() <= coherence::rectangle_limit, "region_budget_exceeded");
    for (const auto &box : region) {
        require(box.lo.size() == b.extents.size() && box.hi.size() == b.extents.size(), "invalid_resource_region");
        for (size_t k=0; k<b.extents.size(); ++k)
            require(box.lo[k] <= box.hi[k] && box.hi[k] <= b.extents[k], "invalid_resource_region");
    }
}
inline void validate_metadata(const Inputs &input, bool check_work, bool check_coherence) {
    require(input.operations.size() <= operation_limit && input.resources.size() <= operation_limit,
            "planning_record_budget_exceeded");
    std::unordered_map<fort_buffer_t, const Resource *> resources;
    for (const auto &b : input.resources) {
        require(b.handle && b.element_bytes && !b.extents.empty(), "invalid_resource_layout");
        require(resources.emplace(b.handle, &b).second, "duplicate_resource_handle");
        const bool empty = std::find(b.extents.begin(), b.extents.end(), size_t(0)) != b.extents.end();
        size_t bytes = empty ? 0 : b.element_bytes;
        if (!empty) for (size_t extent : b.extents) bytes = product(bytes, extent);
        require(bytes == b.bytes && (!check_coherence || !b.allocated || bytes), "invalid_resource_layout");
        if (check_coherence)
            require(b.device_current.empty() || b.allocated, "invalid_resource_coherence");
        validate_region(b.initialized, b);
        if (check_coherence) {
            validate_region(b.host_current, b); validate_region(b.device_current, b);
            coherence::Budget budget;
            require(coherence::difference(b.host_current, b.initialized, budget).empty() &&
                    coherence::difference(b.device_current, b.initialized, budget).empty(), "invalid_resource_coherence");
            auto current = coherence::unite(b.host_current, b.device_current, budget);
            require(coherence::difference(b.initialized, current, budget).empty(), "invalid_resource_coherence");
        }
    }
    size_t workers = 0;
    for (const auto &op : input.operations) {
        require(op.kind <= FORT_SCOPE_PLAN_TEAM_NATIVE_CALL, "unknown_planning_operation");
        if (op.kind == FORT_SCOPE_PLAN_TEAM_ENTRY || op.kind == FORT_SCOPE_PLAN_TEAM_NATIVE_CALL) {
            require(op.bindings.empty() && !op.unit && !op.gpu_available && !op.flops && !op.memory_bytes,
                    op.kind == FORT_SCOPE_PLAN_TEAM_ENTRY ? "invalid_collective_entry_marker" :
                    "invalid_collective_native_call_marker");
            if (check_work) require(input.team_costs.has_value(), "collective_synchronization_calibration_unavailable");
        }
        if (check_work)
            require(std::isfinite(op.flops) && std::isfinite(op.memory_bytes) && op.flops >= 0 && op.memory_bytes >= 0 &&
                    op.flops < double(std::numeric_limits<uint64_t>::max()) &&
                    op.memory_bytes < double(std::numeric_limits<uint64_t>::max()), "unknown_or_overflowed_work");
        if (op.kind == FORT_SCOPE_PLAN_WORKER) {
            require(op.unit && (!check_work || op.flops > 0 || op.memory_bytes > 0), "unknown_worker_work");
            require(++workers <= worker_limit, "planning_worker_budget_exceeded");
        }
        std::vector<fort_buffer_t> bound;
        for (const auto &binding : op.bindings) {
            const auto found = resources.find(binding.buffer);
            require(found != resources.end(), "unknown_resource_handle");
            require(std::find(bound.begin(), bound.end(), binding.buffer) == bound.end(), "duplicate_operation_binding");
            bound.push_back(binding.buffer);
            const auto &b = *found->second;
            validate_region(binding.effects.reads, b); validate_region(binding.effects.writes, b);
            validate_region(binding.effects.overwrites, b);
            if (op.kind == FORT_SCOPE_PLAN_DISCARD)
                require(binding.effects.reads.empty() && binding.effects.overwrites.empty() &&
                        !op.gpu_available && !op.unit, "invalid_partial_definition_event");
            coherence::Budget budget;
            require(coherence::difference(binding.effects.overwrites, binding.effects.writes, budget).empty(),
                    "overwrite_exceeds_write_region");
        }
    }
}
inline void validate(const Inputs &input, const fort_scope_plan_costs &costs) {
    require(costs.version == FORT_SCOPE_PLANNING_ABI_VERSION && costs.valid && costs.max_allocation_bytes,
            "missing_or_incompatible_calibration");
    const double values[] = {
        costs.cpu_flops, costs.cpu_bandwidth, costs.gpu_flops, costs.gpu_bandwidth,
        costs.h2d_bandwidth, costs.d2h_bandwidth,
        costs.create_seconds, costs.register_seconds, costs.host_access_seconds, costs.device_access_seconds,
        costs.gpu_setup_seconds, costs.cold_driver_startup_seconds, costs.allocation_seconds,
        costs.release_seconds, costs.wait_seconds, costs.launch_enqueue_seconds, costs.planning_operation_seconds
    };
    for (double value : values) require(std::isfinite(value) && value > 0, "invalid_calibration_cost");
    // A fitted transfer intercept can legitimately be zero; throughput and
    // measured lifecycle costs still require strictly positive values.
    for (double latency : {costs.h2d_latency, costs.d2h_latency})
        require(std::isfinite(latency) && latency >= 0, "invalid_calibration_cost");
    if (input.transfer_mode == FORT_SCOPE_TRANSFERS_PINNED)
        require(input.transfer_costs && valid_transfer_costs(*input.transfer_costs), "transfer_estimates_unavailable");
    if (input.team_costs) {
        require(valid_team_costs(*input.team_costs) && !input.continuation,
                "collective_synchronization_calibration_unavailable");
    }
    validate_metadata(input, true, true);
}
struct State {
    std::vector<Resource> resources;
    std::unordered_map<fort_buffer_t, size_t> index;
    fort_scope_plan_decision count{};
    std::vector<bool> choices;
    std::vector<std::pair<size_t, size_t>> intervals;
    size_t allocated = 0;
    // Short leases retain one cached pair between proved straight-line GPU
    // operations. First use is conservative; native calls can replace it.
    size_t staging_capacity = 0;
    bool ready = false, driver_initialized = false, pending = false;
    double time = 0;
    double execution_time = 0;
    fort_scope_terminal_cost terminal{};
};
inline void discard_coherence(State &state) {
    // Completed schedules need only their choices, costs, and counters. Do not
    // retain a full physical snapshot for every candidate.
    std::vector<Resource>{}.swap(state.resources);
    std::unordered_map<fort_buffer_t, size_t>{}.swap(state.index);
}
inline State initial(const Inputs &input, const fort_scope_plan_costs &costs) {
    State s; s.resources = input.resources; s.ready = input.device_ready;
    s.driver_initialized = input.driver_initialized; s.pending = input.pending;
    for (size_t k=0; k<s.resources.size(); ++k) {
        s.index.emplace(s.resources[k].handle, k);
        if (s.resources[k].allocated) s.allocated = add(s.allocated, s.resources[k].bytes);
    }
    require(s.allocated <= input.device_budget, "device_budget_exceeded");
    s.count.peak_device_bytes = s.allocated;
    const size_t registrations = input.registrations_incurred == std::numeric_limits<size_t>::max()
        ? s.resources.size() : input.registrations_incurred;
    seconds(s.time, (input.charge_create ? costs.create_seconds : 0) + double(registrations)*costs.register_seconds);
    if (input.team_costs)
        seconds(s.time, input.team_costs->owner_seconds + double(s.resources.size())*input.team_costs->descriptor_seconds);
    return s;
}
inline void wait(State &s, const fort_scope_plan_costs &costs) {
    if (!s.pending) return;
    seconds(s.time, costs.wait_seconds); s.count.waits = add(s.count.waits, 1); s.pending = false;
}
inline void initialize(State &s, const fort_scope_plan_costs &costs) {
    if (s.ready) return;
    if (!s.driver_initialized) seconds(s.time, costs.cold_driver_startup_seconds);
    seconds(s.time, costs.gpu_setup_seconds); s.ready = s.driver_initialized = true;
}
inline void allocate(State &s, Resource &b, const Inputs &input, const fort_scope_plan_costs &costs) {
    if (b.allocated || !b.bytes) return;
    require(b.bytes <= costs.max_allocation_bytes, "allocation_outside_calibrated_range");
    require(s.allocated <= input.device_budget && b.bytes <= input.device_budget-s.allocated, "device_budget_exceeded");
    initialize(s, costs); seconds(s.time, costs.allocation_seconds); b.allocated = true; s.pending = true;
    s.allocated = add(s.allocated, b.bytes); s.count.allocations = add(s.count.allocations, 1);
    s.count.peak_device_bytes = std::max<uint64_t>(s.count.peak_device_bytes, s.allocated);
}
inline void copy(State &s, Resource &b, const Box &box, bool upload,
                 const Inputs &input, const fort_scope_plan_costs &costs,
                 const EvidenceSink *sink = nullptr, uint64_t unit = 0, const char *phase = "operation") {
    const fort_physical::CopyPlan plan(b.element_bytes, b.extents, box.lo, box.hi);
    require(plan.valid, "physical_transfer_overflow");
    if (!plan.bytes) return;
    allocate(s, b, input, costs);
    uint64_t calls = plan.copies;
    if (input.transfer_mode == FORT_SCOPE_TRANSFERS_PINNED) {
        const auto &transfer = *input.transfer_costs;
        size_t capacity_index = 0;
        while (capacity_index+1<FORT_SCOPE_BATCH_CAPACITIES && batch_capacities[capacity_index]<plan.bytes) ++capacity_index;
        wait(s, costs); // The synchronous control orders full-layout storage first.
        const bool reused=s.staging_capacity>=batch_capacities[capacity_index];
        if (reused) while (capacity_index+1<FORT_SCOPE_BATCH_CAPACITIES && batch_capacities[capacity_index]<s.staging_capacity) ++capacity_index;
        else s.staging_capacity=batch_capacities[capacity_index];
        seconds(s.time, reused ? transfer.staging_reuse_seconds[capacity_index] : transfer.staging_cold_seconds[capacity_index]);
        calls = 0; uint64_t rows = 0;
        plan.visit([&](const fort_physical::CopyOperation &op) {
            return fort_physical::visit_tiles(op, batch_capacities[capacity_index], [&](const fort_physical::CopyOperation &tile) {
                calls = add(calls, 1); rows = add(rows, product(tile.height, tile.depth)); return true;
            });
        });
        seconds(s.time, double(plan.bytes)/(upload ? transfer.pack_bytes_per_second : transfer.unpack_bytes_per_second) +
                        double(rows)*(upload ? transfer.pack_row_seconds : transfer.unpack_row_seconds));
        seconds(s.time, double(calls)*(upload ? transfer.pinned_h2d_latency : transfer.pinned_d2h_latency) +
                        double(plan.bytes)/(upload ? transfer.pinned_h2d_bandwidth : transfer.pinned_d2h_bandwidth));
        seconds(s.time, double(calls)*(transfer.event_record_seconds+transfer.event_wait_seconds));
    } else {
        seconds(s.time, double(plan.copies)*(upload ? costs.h2d_latency : costs.d2h_latency) +
                        double(plan.bytes)/(upload ? costs.h2d_bandwidth : costs.d2h_bandwidth));
    }
    if (sink) {
        EvidenceEvent event; event.event = "copy"; event.phase = phase; event.resource = &b;
        event.rectangle = &box; event.unit = unit; event.upload = upload;
        event.bytes = plan.bytes; event.copies = calls; (*sink)(event);
    }
    auto &bytes = upload ? s.count.upload_bytes : s.count.download_bytes;
    auto &copies = upload ? s.count.uploads : s.count.downloads;
    bytes = add(bytes, plan.bytes); copies = add(copies, calls);
    s.pending = input.transfer_mode != FORT_SCOPE_TRANSFERS_PINNED;
}
inline void ensure(State &s, Resource &b, const Region &requested, bool device,
                   const Inputs &input, const fort_scope_plan_costs &costs,
                   const EvidenceSink *sink = nullptr, uint64_t unit = 0, const char *phase = "operation") {
    if (requested.empty()) return;
    coherence::Budget budget;
    require(coherence::difference(requested, b.initialized, budget).empty(), "uninitialized_read");
    auto &current = device ? b.device_current : b.host_current;
    auto missing = coherence::difference(requested, current, budget);
    auto updated = coherence::unite(current, missing, budget);
    for (const auto &box : missing) copy(s, b, box, device, input, costs, sink, unit, phase);
    current = std::move(updated);
}
inline void execute(State &s, const Operation &op, bool gpu,
                    const Inputs &input, const fort_scope_plan_costs &costs, uint64_t &work,
                    const EvidenceSink *sink = nullptr) {
    work = add(work, 1);
    if (op.kind == FORT_SCOPE_PLAN_TEAM_ENTRY || op.kind == FORT_SCOPE_PLAN_TEAM_NATIVE_CALL) {
        if (input.team_costs) seconds(s.time, op.kind == FORT_SCOPE_PLAN_TEAM_ENTRY ?
                input.team_costs->entry_seconds : input.team_costs->native_call_seconds);
        return;
    }
    if (sink) {
        EvidenceEvent event; event.event = "operation"; event.operation = &op; event.unit = op.unit; event.gpu = gpu;
        (*sink)(event);
        if ((op.kind == FORT_SCOPE_PLAN_FORGET || op.kind == FORT_SCOPE_PLAN_DISCARD)) for (const auto &binding : op.bindings) {
            event.event = "requirement"; event.binding = &binding; event.resource = &s.resources[s.index.at(binding.buffer)]; (*sink)(event);
        }
    }
    if ((op.kind == FORT_SCOPE_PLAN_FORGET || op.kind == FORT_SCOPE_PLAN_DISCARD)) {
        for (const auto &binding : op.bindings) {
            auto &b = s.resources[s.index.at(binding.buffer)]; wait(s, costs);
            if (op.kind == FORT_SCOPE_PLAN_FORGET) {
                b.initialized.clear(); b.host_current.clear(); b.device_current.clear();
            } else {
                coherence::Budget budget;
                b.initialized = coherence::difference(b.initialized, binding.effects.writes, budget);
                b.host_current = coherence::difference(b.host_current, binding.effects.writes, budget);
                b.device_current = coherence::difference(b.device_current, binding.effects.writes, budget);
            }
        }
        return; // Definition changes retain full-layout allocations.
    }
    if (gpu) { require(op.kind == FORT_SCOPE_PLAN_WORKER && op.gpu_available, "unsupported_gpu_worker"); initialize(s, costs); }
    std::vector<coherence::Prepared> prepared;
    for (const auto &binding : op.bindings) {
        auto &b = s.resources[s.index.at(binding.buffer)];
        seconds(s.time, gpu ? costs.device_access_seconds : costs.host_access_seconds);
        if (gpu) allocate(s, b, input, costs);
        coherence::Budget budget;
        auto preserve = coherence::difference(binding.effects.writes, binding.effects.overwrites, budget);
        if (sink) {
            EvidenceEvent event; event.event = "requirement"; event.operation = &op; event.unit = op.unit;
            event.gpu = gpu; event.resource = &b; event.binding = &binding; event.preservation_reads = &preserve; (*sink)(event);
        }
        ensure(s, b, binding.effects.reads, gpu, input, costs, sink, op.unit, "read");
        ensure(s, b, preserve, gpu, input, costs, sink, op.unit, "preserve");
        prepared.push_back(coherence::prepare(b, binding.effects, gpu));
        if (!gpu) wait(s, costs); // Actual host_begin waits after each buffer.
    }
    seconds(s.time, input.team_costs && op.kind == FORT_SCOPE_PLAN_NATIVE
            ? native_compute(op, input, costs) : compute(op, costs, gpu));
    if (input.team_costs && op.kind == FORT_SCOPE_PLAN_WORKER) {
        const auto &team = *input.team_costs;
        seconds(s.time, gpu ? team.gpu_worker_seconds : team.cpu_worker_seconds);
    }
    if (gpu) {
        seconds(s.time, costs.launch_enqueue_seconds); s.count.launches = add(s.count.launches, 1); s.pending = true;
    }
    for (size_t k=0; k<op.bindings.size(); ++k) {
        auto &b = s.resources[s.index.at(op.bindings[k].buffer)]; auto &p = prepared[k];
        b.initialized = std::move(p.initialized);
        (gpu ? b.device_current : b.host_current) = std::move(p.current);
        (gpu ? b.host_current : b.device_current) = std::move(p.opposite);
    }
    if (op.kind == FORT_SCOPE_PLAN_WORKER) s.choices.push_back(gpu);
    if (!gpu) s.staging_capacity=0; // Native computation may enter another staging user.
}
inline void close(State &s, const Inputs &input, const fort_scope_plan_costs &costs, uint64_t &work,
                  const EvidenceSink *sink = nullptr) {
    for (auto &b : s.resources) {
        if (sink) {
            EvidenceEvent event; event.event = "export"; event.phase = "scope_close"; event.resource = &b; (*sink)(event);
        }
        work = add(work, 1); ensure(s, b, b.initialized, false, input, costs, sink, 0, "scope_close"); wait(s, costs);
    }
    for (auto &b : s.resources) if (b.allocated) {
        seconds(s.time, costs.release_seconds); b.allocated = false;
        s.allocated -= b.bytes;
        if (input.asynchronous_release) s.pending = true;
    }
    wait(s, costs);
}
inline void finish(State &s, const Inputs &input, const fort_scope_plan_costs &costs, uint64_t &work,
                   const EvidenceSink *sink = nullptr) {
    if (input.continuation) wait(s, costs); // Actual synchronous segment boundary.
    s.execution_time = s.time;
    const auto count = s.count;
    for (const auto &resource : s.resources) if (resource.allocated) ++s.terminal.releases;
    // Continuation evidence describes executed prefix operations only. Projected
    // close costs appear separately in the decision/report, never as transfers.
    close(s, input, costs, work, input.continuation ? nullptr : sink);
    s.terminal.seconds = s.time-s.execution_time;
    s.terminal.download_bytes = s.count.download_bytes-count.download_bytes;
    s.terminal.downloads = s.count.downloads-count.downloads;
    s.terminal.waits = s.count.waits-count.waits;
    if (input.continuation) s.count = count;
}
inline fort_scope_terminal_cost entry_terminal(const Inputs &input, const fort_scope_plan_costs &costs,
                                               uint64_t &work) {
    auto state = initial(input, costs);
    finish(state, input, costs, work);
    return state.terminal;
}
inline State simulate(const Inputs &input, const fort_scope_plan_costs &costs,
                      const std::vector<bool> &choices, uint64_t &work) {
    State s = initial(input, costs); size_t worker = 0;
    for (const auto &op : input.operations) {
        const bool gpu = op.kind == FORT_SCOPE_PLAN_WORKER ? choices.at(worker++) : false;
        execute(s, op, gpu, input, costs, work);
    }
    require(worker == choices.size(), "invalid_worker_schedule"); finish(s, input, costs, work); return s;
}
struct Gate { double counterfactual, native_compute; bool other_gpu; };
struct Candidate { State state; std::vector<Gate> gates; };
inline void retain(std::vector<State> &front, State state) {
    front.push_back(std::move(state));
    std::stable_sort(front.begin(), front.end(), [](const State &a, const State &b) { return a.time < b.time; });
    if (front.size() > frontier_limit) front.resize(frontier_limit);
}
} // namespace detail

// Placement-independent definition proof. No current-copy locations, device
// budget, estimates, or calibration are consulted. Every operation is checked
// against its entry state before any of that operation's writes are committed.
inline DefinitionValidation validate_definitions(const Inputs &input) noexcept {
    DefinitionValidation result;
    try {
        detail::validate_metadata(input, false, false);
        struct DefinitionState { Region initialized, host_current, device_current; };
        std::vector<DefinitionState> states;
        std::unordered_map<fort_buffer_t, size_t> index;
        for (const auto &resource : input.resources) {
            index.emplace(resource.handle, states.size());
            states.push_back({resource.initialized, {}, {}});
        }
        for (size_t k=0; k<input.operations.size(); ++k) {
            result.operation = k; result.buffer = 0;
            const auto &op = input.operations[k];
            if (op.kind == FORT_SCOPE_PLAN_TEAM_ENTRY || op.kind == FORT_SCOPE_PLAN_TEAM_NATIVE_CALL) continue;
            if ((op.kind == FORT_SCOPE_PLAN_FORGET || op.kind == FORT_SCOPE_PLAN_DISCARD)) {
                for (const auto &binding : op.bindings) {
                    auto &defined = states[index.at(binding.buffer)].initialized;
                    if (op.kind == FORT_SCOPE_PLAN_FORGET) defined.clear();
                    else {
                        coherence::Budget budget;
                        defined = coherence::difference(defined, binding.effects.writes, budget);
                    }
                }
                continue;
            }
            std::vector<coherence::Prepared> prepared;
            for (const auto &binding : op.bindings) {
                result.buffer = binding.buffer;
                auto &state = states[index.at(binding.buffer)];
                coherence::Budget budget;
                const auto preservation = coherence::difference(binding.effects.writes, binding.effects.overwrites, budget);
                const auto required = coherence::unite(binding.effects.reads, preservation, budget);
                if (!coherence::difference(required, state.initialized, budget).empty()) {
                    result.status = FORT_SCOPE_UNINITIALIZED;
                    result.reason = "uninitialized_read";
                    return result;
                }
                // The existing execution transition can now define may-writes:
                // preservation proved every potentially unwritten hole defined.
                prepared.push_back(coherence::prepare(state, binding.effects, false));
            }
            for (size_t j=0; j<op.bindings.size(); ++j)
                states[index.at(op.bindings[j].buffer)].initialized = std::move(prepared[j].initialized);
        }
        result.status = FORT_SCOPE_OK; result.reason = "definition_plan_valid";
        result.operation = std::numeric_limits<size_t>::max(); result.buffer = 0;
    } catch (const detail::Unavailable &error) { result.reason = error.reason; }
      catch (const coherence::Fragmented &) { result.reason = "region_fragmentation_unavailable"; }
      catch (const std::bad_alloc &) { result.status = FORT_SCOPE_RESOURCE; result.reason = "planning_resource_failure"; }
      catch (...) { result.reason = "planning_state_unavailable"; }
    return result;
}

inline Result select(const Inputs &input, const fort_scope_plan_costs &costs) {
    Result result;
    result.report.version = FORT_SCOPE_PLANNING_REPORT_VERSION;
    result.report.endpoint_mode = input.continuation ? FORT_SCOPE_PLAN_CONTINUE : FORT_SCOPE_PLAN_COMPLETE;
    result.driver_initialized = input.driver_initialized;
    size_t workers = 0;
    for (const auto &op : input.operations) if (op.kind == FORT_SCOPE_PLAN_WORKER) ++workers;
    result.gpu_workers.assign(workers, false);
    result.decision.cpu_units = uint32_t(std::min<size_t>(workers, std::numeric_limits<uint32_t>::max()));
    uint64_t work = 0;
    try {
        detail::validate(input, costs);
        double native = 0;
        for (const auto &op : input.operations) if (!detail::definition_only(op)) {
            detail::seconds(native, detail::native_compute(op, input, costs));
            if (op.kind == FORT_SCOPE_PLAN_NATIVE && op.flops == 0 && op.memory_bytes == 0)
                result.native_common_compute_excluded = true;
        }
        result.decision.native_seconds = native;
        std::vector<detail::State> complete;
        std::string unavailable_alternative;
        auto unavailable = [&](const char *reason) {
            if (unavailable_alternative.empty()) unavailable_alternative = reason;
        };
        auto save = [&](detail::State state) {
            if (std::none_of(complete.begin(), complete.end(), [&](const detail::State &old) { return old.choices == state.choices; })) {
                detail::require(complete.size() < detail::candidate_limit, "planning_candidate_budget_exceeded");
                detail::discard_coherence(state);
                complete.push_back(std::move(state));
            }
        };
        const bool gpu_supported = std::any_of(input.operations.begin(), input.operations.end(), [](const Operation &op) {
            return op.kind == FORT_SCOPE_PLAN_WORKER && op.gpu_available;
        });
        const bool fresh = !input.device_ready && !input.pending &&
            std::none_of(input.resources.begin(), input.resources.end(), [](const Resource &b) {
                return b.allocated || !b.device_current.empty();
            });
        result.collective_whole_owner_margin = input.team_costs && fresh && !input.continuation;
        // A fresh scope with any GPU work must pay at least setup and one
        // enqueue, plus cold driver startup if it is not already initialized.
        // Even zero GPU compute/transfers cannot beat this lower bound. Prove
        // the native result before constructing expensive candidate states.
        const double gpu_lower_bound = costs.gpu_setup_seconds + costs.launch_enqueue_seconds +
            (input.driver_initialized ? 0 : costs.cold_driver_startup_seconds);
        double native_startup_comparison = native;
        if (input.continuation) for (const auto &op : input.operations) {
            if (!detail::definition_only(op))
                detail::seconds(native_startup_comparison, double(op.bindings.size())*costs.host_access_seconds);
        }
        if (gpu_supported && fresh && native_startup_comparison <= gpu_lower_bound) {
            // Fresh, host-only coverage stays host current under every proven
            // native definition/write. Reuse source preflight when available;
            // raw planning clients must establish it before this shortcut.
            if (!input.definitions_validated) {
                const auto proof = validate_definitions(input);
                detail::require(proof.status == FORT_SCOPE_OK, proof.reason);
            }
            double estimate = 0;
            const size_t registrations = input.registrations_incurred == std::numeric_limits<size_t>::max()
                ? input.resources.size() : input.registrations_incurred;
            detail::seconds(estimate, (input.charge_create ? costs.create_seconds : 0) +
                double(registrations)*costs.register_seconds);
            if (input.team_costs)
                detail::seconds(estimate, input.team_costs->owner_seconds +
                                double(input.resources.size())*input.team_costs->descriptor_seconds);
            for (const auto &op : input.operations) {
                if (detail::definition_only(op)) continue;
                detail::seconds(estimate, detail::native_compute(op, input, costs));
                if (!input.team_costs) detail::seconds(estimate, double(op.bindings.size())*costs.host_access_seconds);
            }
            // Retain the existing logical accounting for native operations and
            // final publication checks, without allocating their snapshots.
            work = detail::add(input.operations.size(), input.resources.size());
            work = detail::add(work, input.query_construction_operations);
            result.decision.cpu_units = uint32_t(workers); result.decision.available = 1;
            result.decision.candidates = 1; result.decision.simulated_operations = work;
            result.decision.native_seconds = native; result.decision.estimated_seconds = estimate;
            detail::seconds(result.decision.estimated_seconds, double(work)*costs.planning_operation_seconds);
            result.native_startup_shortcut = true;
            result.reason = "native_gpu_startup_lower_bound";
            result.report.available = 1;
            result.report.execution_seconds = result.report.native_execution_seconds = result.decision.estimated_seconds;
            result.report.ranking_seconds = result.report.native_ranking_seconds = result.decision.estimated_seconds;
            result.report.native_common_compute_excluded = result.native_common_compute_excluded;
            if (input.continuation) result.decision.native_seconds = result.report.native_execution_seconds;
            return result;
        }
        const auto initial_terminal = input.continuation ? detail::entry_terminal(input, costs, work)
                                                        : fort_scope_terminal_cost{};
        auto all_native = detail::simulate(input, costs, result.gpu_workers, work);
        if (result.collective_whole_owner_margin) {
            // A whole-native choice executes the untouched original procedures
            // after the fresh context is closed. It does not call generated CPU
            // workers, their entry protocols, or their per-array hooks.
            auto fallback = detail::initial(input, costs);
            detail::seconds(fallback.time, native);
            all_native.time = all_native.execution_time = fallback.time;
            all_native.terminal = {};
        }
        auto report = [&](const detail::State &state, double planning) {
            result.report.available = 1;
            result.report.execution_seconds = state.execution_time + planning;
            result.report.native_execution_seconds = all_native.execution_time + planning;
            result.report.entry_terminal = initial_terminal;
            result.report.terminal = state.terminal;
            result.report.native_terminal = all_native.terminal;
            result.report.ranking_seconds = state.time + planning-initial_terminal.seconds;
            result.report.native_ranking_seconds = all_native.time + planning-initial_terminal.seconds;
            detail::require(std::isfinite(result.report.ranking_seconds) &&
                            std::isfinite(result.report.native_ranking_seconds), "arithmetic_overflow");
            result.report.native_common_compute_excluded = result.native_common_compute_excluded;
            if (input.continuation) {
                result.decision.estimated_seconds = result.report.execution_seconds;
                result.decision.native_seconds = result.report.native_execution_seconds;
            }
        };
        if (!gpu_supported) {
            work = detail::add(work, input.query_construction_operations);
            result.decision = all_native.count;
            result.decision.cpu_units = uint32_t(workers); result.decision.available = 1;
            result.decision.candidates = 1; result.decision.simulated_operations = work;
            result.decision.native_seconds = native; result.decision.estimated_seconds = all_native.time;
            detail::seconds(result.decision.estimated_seconds, double(work)*costs.planning_operation_seconds);
            result.reason = "native_no_supported_gpu_workers";
            report(all_native, double(work)*costs.planning_operation_seconds);
            return result;
        }
        detail::discard_coherence(all_native);
        save(all_native);
        // Guaranteed complete-block alternatives survive prefix pruning. Fixed
        // native/definition operations and ineligible workers split blocks.
        size_t worker = 0;
        for (size_t k=0; k<input.operations.size();) {
            if (input.operations[k].kind != FORT_SCOPE_PLAN_WORKER) { ++k; continue; }
            if (!input.operations[k].gpu_available) { ++worker; ++k; continue; }
            const size_t first = worker;
            while (k<input.operations.size() && input.operations[k].kind == FORT_SCOPE_PLAN_WORKER && input.operations[k].gpu_available) { ++worker; ++k; }
            auto choices = result.gpu_workers;
            for (size_t j=first; j<worker; ++j) choices[j] = true;
            try {
                auto s = detail::simulate(input, costs, choices, work); s.intervals.push_back({first, worker}); save(std::move(s));
            } catch (const detail::Unavailable &error) { unavailable(error.reason); }
              catch (const coherence::Fragmented &) { unavailable("region_fragmentation_unavailable"); }
        }
        // Each position retains distinct full coherence states. The bounded
        // frontier is a deterministic search budget, not a position-only DP.
        std::vector<std::vector<detail::State>> fronts(input.operations.size()+1);
        fronts[0].push_back(detail::initial(input, costs));
        for (size_t k=0; k<input.operations.size(); ++k) {
            for (const auto &prefix : fronts[k]) {
                const auto &op = input.operations[k];
                try {
                    auto s = prefix; detail::execute(s, op, false, input, costs, work); detail::retain(fronts[k+1], std::move(s));
                } catch (const detail::Unavailable &error) { unavailable(error.reason); }
                  catch (const coherence::Fragmented &) { unavailable("region_fragmentation_unavailable"); }
                if (op.kind != FORT_SCOPE_PLAN_WORKER || !op.gpu_available) continue;
                // Adjacent GPU transitions would silently create an interval
                // longer than four. Admit the complete block explicitly.
                if (k && input.operations[k-1].kind == FORT_SCOPE_PLAN_WORKER && !prefix.choices.empty() && prefix.choices.back()) continue;
                size_t end = k;
                while (end<input.operations.size() && input.operations[end].kind == FORT_SCOPE_PLAN_WORKER && input.operations[end].gpu_available) ++end;
                std::vector<size_t> lengths;
                for (size_t n=1; n<=4 && k+n<=end; ++n) lengths.push_back(n);
                const size_t length = end-k;
                if (length > 4 && (!k || input.operations[k-1].kind != FORT_SCOPE_PLAN_WORKER || !input.operations[k-1].gpu_available)) lengths.push_back(length);
                for (size_t n : lengths) try {
                    auto s = prefix; const size_t first = s.choices.size();
                    for (size_t j=k; j<k+n; ++j) detail::execute(s, input.operations[j], true, input, costs, work);
                    s.intervals.push_back({first, first+n}); detail::retain(fronts[k+n], std::move(s));
                } catch (const detail::Unavailable &error) { unavailable(error.reason); }
                  catch (const coherence::Fragmented &) { unavailable("region_fragmentation_unavailable"); }
            }
            // All transitions go strictly forward. Releasing consumed fronts
            // bounds live snapshots to the small frontier/lookahead budget,
            // rather than retaining sixteen snapshots per source operation.
            std::vector<detail::State>{}.swap(fronts[k]);
        }
        for (auto &s : fronts.back()) try { detail::finish(s, input, costs, work); save(std::move(s)); }
            catch (const detail::Unavailable &error) { unavailable(error.reason); }
            catch (const coherence::Fragmented &) { unavailable("region_fragmentation_unavailable"); }
        std::vector<detail::Candidate> candidates;
        std::vector<const Operation *> worker_ops;
        for (const auto &op : input.operations) if (op.kind == FORT_SCOPE_PLAN_WORKER) worker_ops.push_back(&op);
        for (auto &s : complete) {
            if (std::none_of(s.choices.begin(), s.choices.end(), [](bool gpu) { return gpu; })) continue;
            detail::Candidate c{std::move(s), {}}; bool valid = true;
            for (const auto &interval : c.state.intervals) {
                auto choices = c.state.choices; double cpu = 0;
                for (size_t j=interval.first; j<interval.second; ++j) { choices[j] = false; detail::seconds(cpu, detail::native_compute(*worker_ops[j], input, costs)); }
                const bool other_gpu = std::any_of(choices.begin(), choices.end(), [](bool gpu) { return gpu; });
                try {
                    const double time = input.continuation || other_gpu
                        ? detail::simulate(input, costs, choices, work).time : native;
                    c.gates.push_back({time, cpu, other_gpu});
                } catch (const detail::Unavailable &error) { unavailable(error.reason); valid = false; break; }
                  catch (const coherence::Fragmented &) { unavailable("region_fragmentation_unavailable"); valid = false; break; }
            }
            if (valid && !c.gates.empty()) candidates.push_back(std::move(c));
        }
        work = detail::add(work, input.query_construction_operations);
        const double planning = double(work)*costs.planning_operation_seconds;
        detail::require(std::isfinite(planning), "arithmetic_overflow");
        result.decision = all_native.count;
        result.decision.cpu_units = uint32_t(workers);
        result.decision.available = 1; result.decision.candidates = uint32_t(complete.size());
        result.decision.simulated_operations = work;
        result.decision.native_seconds = native;
        result.decision.estimated_seconds = all_native.time;
        detail::seconds(result.decision.estimated_seconds, planning);
        result.reason = "native_is_cheaper";
        report(all_native, planning);
        if (candidates.empty() && !unavailable_alternative.empty()) result.reason = "native_" + unavailable_alternative;
        double best = input.continuation ? all_native.time + planning : native;
        bool margin_rejected = false;
        for (auto &candidate : candidates) {
            const double time = candidate.state.time + planning;
            if (!(time < best)) continue;
            // Local interval counterfactuals may retain other GPU workers and
            // substitute slower generated CPU workers. The complete collective
            // schedule must also beat the untouched original whole-native
            // alternative by 20%, with common unmodeled native work excluded
            // from both sides just as it is in native_seconds.
            if (result.collective_whole_owner_margin && !(time <= native - 0.20*native)) {
                margin_rejected = true;
                continue;
            }
            bool accepted = true;
            for (const auto &gate : candidate.gates) {
                const double alternative = gate.counterfactual + (input.continuation || gate.other_gpu ? planning : 0);
                if (!(time <= alternative - 0.20*gate.native_compute)) { accepted = false; break; }
            }
            if (!accepted) { margin_rejected = true; continue; }
            best = time; result.gpu_workers = candidate.state.choices;
            result.decision = candidate.state.count;
            result.decision.available = 1; result.decision.candidates = uint32_t(complete.size());
            result.decision.simulated_operations = work; result.decision.native_seconds = native;
            result.decision.estimated_seconds = time;
            result.decision.gpu_units = uint32_t(std::count(result.gpu_workers.begin(), result.gpu_workers.end(), true));
            result.decision.cpu_units = uint32_t(workers)-result.decision.gpu_units;
            result.reason = "coherent_gpu_schedule_selected";
            report(candidate.state, planning);
        }
        if (!result.decision.gpu_units && margin_rejected) result.reason = "native_20_percent_margin_not_met";
        return result;
    } catch (const detail::Unavailable &error) { result.reason = error.reason; }
      catch (const coherence::Fragmented &) { result.reason = "region_fragmentation_unavailable"; }
      catch (const std::bad_alloc &) { result.reason = "planning_resource_failure"; }
      catch (...) { result.reason = "planning_state_unavailable"; }
    result.decision.available = 0; result.decision.simulated_operations = work;
    return result;
}

// Re-simulate only the installed final alternative for diagnostics. This does
// not change a decision, context, work counter, or numerical execution. It uses
// the very same region transitions and physical copy plan as selection.
inline void evidence(const Inputs &input, const fort_scope_plan_costs &costs,
                     const Result &result, const EvidenceSink &sink) {
    for (const auto &resource : input.resources) {
        EvidenceEvent event; event.event = "snapshot"; event.phase = "initial"; event.resource = &resource; sink(event);
    }
    if (!result.decision.available || result.native_startup_shortcut) return;
    uint64_t ignored_work = 0;
    auto state = detail::initial(input, costs); size_t worker = 0;
    for (const auto &op : input.operations) {
        const bool gpu = op.kind == FORT_SCOPE_PLAN_WORKER ? result.gpu_workers.at(worker++) : false;
        detail::execute(state, op, gpu, input, costs, ignored_work, &sink);
    }
    detail::finish(state, input, costs, ignored_work, &sink);
    if (result.collective_whole_owner_margin && result.decision.gpu_units) {
        EvidenceEvent event; event.event = "gate"; event.phase = "whole_owner";
        event.first = 0; event.last = result.gpu_workers.size();
        event.seconds = result.decision.estimated_seconds;
        event.counterfactual_seconds = result.decision.native_seconds;
        event.required_saving = .20*result.decision.native_seconds;
        event.accepted = event.seconds <= event.counterfactual_seconds-event.required_saving;
        sink(event);
    }
    // Reconstruct admitted intervals from their exact contiguous worker runs.
    // Native/forget records always split an interval even when they retain data.
    worker = 0;
    for (size_t k=0; k<input.operations.size();) {
        if (input.operations[k].kind != FORT_SCOPE_PLAN_WORKER) { ++k; continue; }
        if (!result.gpu_workers.at(worker)) { ++worker; ++k; continue; }
        const size_t first = worker; double native_compute = 0;
        auto counterfactual = result.gpu_workers;
        while (k<input.operations.size() && input.operations[k].kind == FORT_SCOPE_PLAN_WORKER && result.gpu_workers.at(worker)) {
            counterfactual[worker] = false;
            detail::seconds(native_compute, detail::native_compute(input.operations[k], input, costs)); ++worker; ++k;
        }
        const bool other_gpu = std::any_of(counterfactual.begin(), counterfactual.end(), [](bool value) { return value; });
        const double planning = double(result.decision.simulated_operations)*costs.planning_operation_seconds;
        const double alternative = input.continuation || other_gpu
            ? detail::simulate(input, costs, counterfactual, ignored_work).time + planning
            : result.decision.native_seconds;
        EvidenceEvent event; event.event = "gate"; event.first = first; event.last = worker;
        event.seconds = input.continuation
            ? result.report.execution_seconds + result.report.terminal.seconds : result.decision.estimated_seconds;
        event.counterfactual_seconds = alternative;
        event.required_saving = .20*native_compute;
        event.accepted = event.seconds <= alternative-event.required_saving; sink(event);
    }
}
}} // namespace fort_scoped::planning
#endif
