/* Side-effect-free source-task pricing on the ordinary coherence simulator. */
#ifndef FORT_SCOPED_SOURCE_QUOTE_HPP
#define FORT_SCOPED_SOURCE_QUOTE_HPP
#include "scoped_planning.hpp"

namespace fort_scoped { namespace source_quote {
struct Snapshot {
    planning::Inputs input;
    size_t device_budget=0, workspace_bytes=0, scratch_capacity=0;
    bool scratch_active=false, queue_pending=false, opaque_workspace=false, opaque_workspace_claim=false;
};
inline bool valid_protocol(const fort_scope_source_costs_v1 &costs) {
    if(costs.version!=FORT_SCOPE_SOURCE_QUOTE_ABI_VERSION || !costs.valid || costs.reserved ||
       costs.protocol_id!=FORT_SCOPE_SOURCE_PROTOCOL_ID ||
       (costs.precision_bits!=32 && costs.precision_bits!=64) ||
       !costs.max_bindings || !costs.max_launches || !costs.max_scratch_bytes) return false;
    for(double value:{costs.source_submit_seconds,costs.scratch_cold_acquire_seconds,
                      costs.scratch_reuse_acquire_seconds,costs.scratch_idle_grow_seconds,
                      costs.scratch_release_seconds})
        if(!std::isfinite(value) || value<=0) return false;
    return true;
}
inline fort_scope_source_quote_result_v1 price(Snapshot snapshot,
        const std::vector<planning::Binding> &bindings, const fort_scope_source_work_v1 &work,
        const fort_scope_plan_costs &base, const fort_scope_source_costs_v1 &protocol, bool compatible,
        const fort_scope_source_host_costs_v1 *native_costs=nullptr) {
    namespace detail=planning::detail;
    fort_scope_source_quote_result_v1 out{};
    const bool live=native_costs!=nullptr;
    out.version=live ? FORT_SCOPE_SOURCE_QUOTE_LIVE_ABI_VERSION : FORT_SCOPE_SOURCE_QUOTE_ABI_VERSION;
    out.native_endpoint=out.gpu_endpoint=live ? FORT_SCOPE_SOURCE_QUOTE_NATIVE_LIVE : FORT_SCOPE_SOURCE_QUOTE_NATIVE_CLOSE_DOMAIN;
    auto unavailable=[&](uint32_t reason) { out.available=out.gpu_preferred=0;out.reason=reason;return out; };
    if(!compatible || !valid_protocol(protocol)) return unavailable(FORT_SCOPE_SOURCE_QUOTE_CALIBRATION);
    if(live && (native_costs->version!=1 || native_costs->valid!=1 ||
            native_costs->protocol_id!=FORT_SCOPE_SOURCE_HOST_PROTOCOL_ID || !native_costs->max_bindings ||
            !std::isfinite(native_costs->prepare_start_complete_seconds) || native_costs->prepare_start_complete_seconds<=0))
        return unavailable(FORT_SCOPE_SOURCE_QUOTE_CALIBRATION);
    if(work.version!=(live ? FORT_SCOPE_SOURCE_QUOTE_LIVE_ABI_VERSION : FORT_SCOPE_SOURCE_QUOTE_ABI_VERSION) || work.reserved ||
       work.flags&~uint32_t(FORT_SCOPE_SOURCE_QUOTE_SCRATCH_LEASE|FORT_SCOPE_SOURCE_QUOTE_NATIVE_CLOSE_DOMAIN|
                           FORT_SCOPE_SOURCE_QUOTE_WAIT_BEFORE_LEASE|FORT_SCOPE_SOURCE_QUOTE_NATIVE_LIVE) ||
       (work.flags&(FORT_SCOPE_SOURCE_QUOTE_NATIVE_CLOSE_DOMAIN|FORT_SCOPE_SOURCE_QUOTE_NATIVE_LIVE))!=out.native_endpoint ||
       work.precision_bits!=protocol.precision_bits || !work.expected_launches || bindings.empty() ||
       (!work.real_work_proxy && !work.memory_bytes) ||
       !std::isfinite(work.minimum_relative_saving) || work.minimum_relative_saving<0 || work.minimum_relative_saving>=1 ||
       !detail::valid_compute_costs(work.compute) ||
       (work.compute.known_backends&(FORT_SCOPE_COMPUTE_NATIVE_FORTRAN|FORT_SCOPE_COMPUTE_GPU))!=
            (FORT_SCOPE_COMPUTE_NATIVE_FORTRAN|FORT_SCOPE_COMPUTE_GPU) ||
       work.compute.known_backends&FORT_SCOPE_COMPUTE_GPU_LAUNCH_INCLUDED ||
       (!(work.flags&FORT_SCOPE_SOURCE_QUOTE_SCRATCH_LEASE) &&
        (work.scratch_bytes || work.flags&FORT_SCOPE_SOURCE_QUOTE_WAIT_BEFORE_LEASE)))
        return unavailable(FORT_SCOPE_SOURCE_QUOTE_WORK);
    if(bindings.size()>protocol.max_bindings || work.expected_launches>protocol.max_launches ||
       work.scratch_bytes>protocol.max_scratch_bytes ||
       (live && bindings.size()>native_costs->max_bindings)) return unavailable(FORT_SCOPE_SOURCE_QUOTE_RANGE);
    const bool lease=work.flags&FORT_SCOPE_SOURCE_QUOTE_SCRATCH_LEASE;
    const bool fence=work.flags&FORT_SCOPE_SOURCE_QUOTE_WAIT_BEFORE_LEASE;
    const size_t capacity=lease ? std::max(snapshot.scratch_capacity,work.scratch_bytes) : snapshot.scratch_capacity;
    const bool grow=capacity>snapshot.scratch_capacity;
    if(snapshot.scratch_active || snapshot.queue_pending || snapshot.opaque_workspace_claim ||
       (!live && snapshot.opaque_workspace) ||
       (grow && snapshot.scratch_capacity && snapshot.input.pending && !fence))
        return unavailable(FORT_SCOPE_SOURCE_QUOTE_LIVE_STATE);
    out.scratch_capacity_before=snapshot.scratch_capacity;out.scratch_capacity_after=capacity;
    if(snapshot.workspace_bytes>snapshot.device_budget || capacity>snapshot.device_budget-snapshot.workspace_bytes)
        return unavailable(FORT_SCOPE_SOURCE_QUOTE_BUDGET);
    try {
        auto input=std::move(snapshot.input);
        input.continuation=true;input.charge_create=false;input.registrations_incurred=0;
        input.graph_mode=true;input.team_costs.reset();input.definitions_validated=false;
        input.operations.clear();
        planning::Operation operation{FORT_SCOPE_PLAN_WORKER,1,bindings,double(work.real_work_proxy),double(work.memory_bytes),true};
        operation.compute_costs=work.compute;input.operations.push_back(operation);
        input.device_budget=snapshot.device_budget-snapshot.workspace_bytes-snapshot.scratch_capacity;
        input.terminal_reserved_allocations=snapshot.scratch_capacity ? 1 : 0;
        detail::validate(input,base);
        uint64_t operations=0;
        auto native=detail::initial(input,base);
        if(live) {
            // Simulate acquisition/commit only. Actual original-native compute
            // is added below from its separately known mask; no generated-CPU
            // backend or invented zero-cost body is asserted.
            auto acquisition=operation;acquisition.kind=FORT_SCOPE_PLAN_NATIVE;
            acquisition.compute_costs.reset();acquisition.flops=acquisition.memory_bytes=0;
            auto host_base=base;host_base.host_access_seconds=0;
            detail::execute(native,acquisition,false,input,host_base,operations);
            detail::seconds(native.time,native_costs->prepare_start_complete_seconds);
        } else detail::finish(native,input,base,operations);
        out.native_terminal=native.terminal;
        out.native_seconds=native.time+work.compute.native_fortran_seconds;
        input.device_budget=snapshot.device_budget-snapshot.workspace_bytes-capacity;
        input.terminal_reserved_allocations=capacity ? 1 : 0;
        // The real scratch acquire precedes field reservation and cannot
        // evict fields to grow its arena. Preserve that prework rejection.
        size_t allocated=0;
        for(const auto &resource:input.resources) if(resource.allocated) allocated=detail::add(allocated,resource.bytes);
        if(allocated>input.device_budget) return unavailable(FORT_SCOPE_SOURCE_QUOTE_BUDGET);
        auto gpu=detail::initial(input,base);
        gpu.warm_ranking=false; // Cold driver work is part of this complete quote.
        if(fence) detail::wait(gpu,base);
        if(lease) {
            if(grow) {detail::initialize(gpu,base);gpu.pending=true;}
            detail::seconds(gpu.time,grow ? (snapshot.scratch_capacity ? protocol.scratch_idle_grow_seconds :
                protocol.scratch_cold_acquire_seconds) : protocol.scratch_reuse_acquire_seconds);
        }
        auto submit=base;
        // The measured complete submit replaces these old protocol terms.
        submit.device_access_seconds=submit.launch_enqueue_seconds=0;
        detail::execute(gpu,operation,true,input,submit,operations);
        detail::seconds(gpu.time,protocol.source_submit_seconds);
        if(lease) detail::seconds(gpu.time,protocol.scratch_release_seconds);
        gpu.count.launches=detail::add(gpu.count.launches,work.expected_launches-1);
        if(grow) gpu.count.allocations=detail::add(gpu.count.allocations,1);
        const auto peak=detail::add(detail::add(gpu.count.peak_device_bytes,capacity),snapshot.workspace_bytes);
        if(live) gpu.execution_time=gpu.time;
        else detail::finish(gpu,input,base,operations);
        out.gpu_terminal=gpu.terminal;out.gpu_execution_seconds=gpu.execution_time;
        out.gpu_seconds=gpu.time;
        out.transfer_dispersion_debit_seconds=detail::transfer_dispersion_debit(gpu,native,base);
        detail::require(std::isfinite(out.native_seconds) && std::isfinite(out.gpu_seconds),"arithmetic_overflow");
        out.gpu=gpu.count;out.gpu.available=1;out.gpu.gpu_units=out.gpu.candidates=1;
        out.gpu.simulated_operations=operations;
        out.gpu.estimated_seconds=out.gpu_seconds;out.gpu.native_seconds=out.native_seconds;
        out.peak_total_device_bytes=peak;
        out.available=1;out.reason=FORT_SCOPE_SOURCE_QUOTE_READY;
        out.gpu_preferred=out.gpu_seconds+out.transfer_dispersion_debit_seconds<
            out.native_seconds*(1-work.minimum_relative_saving);
        return out;
    } catch(const detail::Unavailable &) {
        return unavailable(FORT_SCOPE_SOURCE_QUOTE_SIMULATION);
    } catch(const coherence::Fragmented &) {
        return unavailable(FORT_SCOPE_SOURCE_QUOTE_SIMULATION);
    }
}
}} // namespace fort_scoped::source_quote
#endif
