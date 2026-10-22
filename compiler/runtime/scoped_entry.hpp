// Per-entry glue contains no shared registry or implicit current context.
#ifndef FORT_SCOPED_ENTRY_HPP
#define FORT_SCOPED_ENTRY_HPP
#include "scoped_runtime.h"
#include "scoped_regions.hpp"
#include <array>
#include <new>
#include <utility>

namespace fort_scoped {
// Fixed caller-owned diagnostic state; no lookup or allocation when disabled.
// The manifest identifies a worker, while the source coordinator supplies its
// owner/segment/operation. An independent generic call leaves these unknown.
class TracePosition {
    fort_scope_t context_;
    fort_scope_trace_state_v1 previous_;
public:
    TracePosition(fort_scope_t context, const char *procedure, const char *implementation) noexcept
        : context_(context) {
        previous_.version = 0;
        fort_scope_trace_set_v1(context, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &previous_);
        if (previous_.version != FORT_SCOPE_TRACE_ABI_VERSION) return;
        const uint32_t mask = FORT_SCOPE_TRACE_IMPLEMENTATION | FORT_SCOPE_TRACE_BOUNDARY |
            (previous_.procedure[0] ? 0 : FORT_SCOPE_TRACE_PROCEDURE);
        fort_scope_trace_set_v1(context, mask, nullptr, procedure, nullptr, nullptr, implementation, "", nullptr);
    }
    TracePosition(const TracePosition &) = delete;
    void implementation(const char *identity) noexcept {
        if (previous_.version == FORT_SCOPE_TRACE_ABI_VERSION)
            fort_scope_trace_set_v1(context_, FORT_SCOPE_TRACE_IMPLEMENTATION,
                                    nullptr, nullptr, nullptr, nullptr, identity, nullptr, nullptr);
    }
    ~TracePosition() {
        if (previous_.version == FORT_SCOPE_TRACE_ABI_VERSION)
            fort_scope_trace_restore_v1(context_, &previous_);
    }
};
// Window launchers borrow validated full-root views. No context call is legal
// inside the batch callback, whose executor owns the context mutex/coherence.
inline const fort_scope_batch_view *batch_view(const fort_scope_batch_window &window, fort_buffer_t handle) noexcept {
    if (window.version!=FORT_SCOPE_BATCH_ABI_VERSION || (window.view_count && !window.views)) return nullptr;
    for (size_t i=0; i<window.view_count; ++i) if (window.views[i].buffer==handle) return &window.views[i];
    return nullptr;
}
template<size_t Capacity> class AccessBatch {
    struct Binding {
        fort_buffer_t handle = 0;
        fort_scope_access effects{};
        void *device = nullptr;
        bool prepared = false;
    };
    fort_scope_t context_;
    std::array<Binding, Capacity> bindings_{};
    size_t size_ = 0;
    bool device_ = false, executing_ = false, begun_ = false;
    static bool writes(const fort_scope_access &a) {
        return (a.flags & FORT_SCOPE_WRITE_ALL) || a.write_count;
    }
public:
    static constexpr size_t binding_capacity = Capacity;
    explicit AccessBatch(fort_scope_t context) : context_(context) {}
    AccessBatch(const AccessBatch &) = delete;
    ~AccessBatch() {
        if (executing_) (void)fort_scope_execution_error(context_, "numerical entry exited during an operation");
        else cancel();
    }
    int add(fort_buffer_t handle, const fort_scope_access &access) {
        if (begun_ || executing_)
            return fort_scope_report_error(FORT_SCOPE_STATE, "cannot add bindings to a prepared numerical task");
        for (size_t i=0; i<size_; ++i) {
            if (bindings_[i].handle != handle) continue;
            if (writes(access) || writes(bindings_[i].effects))
                return fort_scope_report_error(FORT_SCOPE_ALIAS, "writable numerical arguments alias");
            // Read-only aliases use a conservative full-root requirement.
            // This keeps one begin/end and never invents writable alias support.
            bindings_[i].effects.flags = FORT_SCOPE_READ_ALL;
            bindings_[i].effects.read_count = 0;
            bindings_[i].effects.reads = nullptr;
            return FORT_SCOPE_OK;
        }
        if (size_ == Capacity)
            return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "numerical binding capacity exceeded");
        bindings_[size_].handle = handle;
        bindings_[size_++].effects = access;
        return FORT_SCOPE_OK;
    }
    // Descriptors borrow their original section-coordinate storage. Keep this
    // batch and that storage alive through graph_execute; export executes no
    // numerical work and prepares no coherence state.
    int export_bindings(fort_scope_plan_binding *out, size_t capacity, size_t *count) const {
        if (!count || capacity < size_ || (size_ && !out))
            return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "insufficient numerical binding output capacity");
        for (size_t i=0; i<size_; ++i) out[i] = {bindings_[i].handle, bindings_[i].effects};
        *count = size_;
        return FORT_SCOPE_OK;
    }
    int record(uint32_t kind, uint64_t unit, double flops, double memory_bytes, int gpu_available,
               double cpu_numerical_seconds = 0, double gpu_numerical_seconds = 0) const {
        std::array<fort_scope_plan_binding, Capacity> values{};
        for (size_t i=0; i<size_; ++i) values[i] = {bindings_[i].handle, bindings_[i].effects};
        return fort_scope_plan_add_costs_v2(context_, kind, unit, values.data(), size_, flops, memory_bytes,
                                           gpu_available, cpu_numerical_seconds, gpu_numerical_seconds);
    }
    int record_compute(uint32_t kind, uint64_t unit, double flops, double memory_bytes, int gpu_available,
                       const fort_scope_compute_costs_v1 &compute) const {
        std::array<fort_scope_plan_binding, Capacity> values{};
        for (size_t i=0; i<size_; ++i) values[i] = {bindings_[i].handle, bindings_[i].effects};
        return fort_scope_plan_add_compute_costs_v3(context_, kind, unit, values.data(), size_, flops,
                                                  memory_bytes, gpu_available, &compute);
    }
    int decision(uint64_t unit, bool &gpu) const {
        std::array<fort_scope_plan_binding, Capacity> values{};
        for (size_t i=0; i<size_; ++i) values[i] = {bindings_[i].handle, bindings_[i].effects};
        int selected = 0;
        const int status = fort_scope_plan_next(context_, unit, values.data(), size_, &selected);
        gpu = selected != 0;
        return status;
    }
    int begin(bool device) {
        if (begun_ || executing_)
            return fort_scope_report_error(FORT_SCOPE_STATE, "numerical task access already prepared");
        std::array<fort_scope_plan_binding, Capacity> values{};
        std::array<void *, Capacity> pointers{};
        for (size_t i=0; i<size_; ++i) values[i] = {bindings_[i].handle, bindings_[i].effects};
        const int status = fort_scope_access_begin_v1(context_, values.data(), size_, device, pointers.data());
        if (status) return status;
        device_ = device;
        begun_ = true;
        for (size_t i=0; i<size_; ++i) {
            bindings_[i].device = pointers[i];
            bindings_[i].prepared = true;
        }
        return FORT_SCOPE_OK;
    }
    void cancel() noexcept {
        for (size_t i=0; i<size_; ++i) {
            if (bindings_[i].prepared) {
                (void)fort_scope_cancel_access(context_, bindings_[i].handle);
                bindings_[i].prepared = false;
            }
        }
        begun_ = false;
    }
    void executing() { executing_ = true; }
    int finish() {
        if (!begun_)
            return fort_scope_report_error(FORT_SCOPE_STATE, "numerical task access was not prepared");
        for (size_t i=0; i<size_; ++i) {
            auto &b = bindings_[i];
            const int status = device_ ? fort_scope_device_end(context_, b.handle)
                                       : fort_scope_host_end(context_, b.handle);
            if (status) return fort_scope_execution_error(context_, "cannot commit numerical operation");
            b.prepared = false;
        }
        executing_ = false;
        begun_ = false;
        return FORT_SCOPE_OK;
    }
    void *device(fort_buffer_t handle) const {
        for (size_t i=0; i<size_; ++i) if (bindings_[i].handle == handle) return bindings_[i].device;
        return nullptr;
    }
};
// A query projection can construct exact worker footprints in lexical blocks.
// This collector only owns copies of that existing metadata until one atomic
// native handoff; it records no query and executes no numerical work.
class NativeWorkerCollector {
    struct Binding {
        fort_buffer_t buffer;
        uint32_t flags;
        coherence::Effects effects;
        std::vector<fort_scope_section> reads, writes, overwrites;
        static void views(const coherence::Region &parts, std::vector<fort_scope_section> &out) {
            out.clear(); out.reserve(parts.size());
            for (const auto &part : parts) out.push_back({part.lo.data(),part.hi.data()});
        }
        fort_scope_plan_binding view() {
            views(effects.reads,reads); views(effects.writes,writes); views(effects.overwrites,overwrites);
            return {buffer,{flags,reads.size(),reads.data(),writes.size(),writes.data(),
                            overwrites.size(),overwrites.data()}};
        }
    };
    struct Worker {
        uint64_t unit;
        std::vector<Binding> bindings;
        std::vector<fort_scope_plan_binding> views;
        fort_scope_native_worker_v1 view() {
            views.clear(); views.reserve(bindings.size());
            for (auto &binding : bindings) views.push_back(binding.view());
            return {unit,views.data(),views.size()};
        }
    };
    fort_scope_t context_;
    std::vector<Worker> workers_;
    bool begun_ = false;
    static coherence::Region copy(const fort_scope_section *parts,size_t count,size_t rank) {
        if (count>coherence::rectangle_limit || (count && !parts)) throw int(FORT_SCOPE_ARGUMENT);
        coherence::Region result;
        result.reserve(count);
        for (size_t i=0;i<count;++i) {
            if (!parts[i].lower || !parts[i].upper) throw int(FORT_SCOPE_ARGUMENT);
            result.push_back({std::vector<size_t>(parts[i].lower,parts[i].lower+rank),
                              std::vector<size_t>(parts[i].upper,parts[i].upper+rank)});
        }
        return result;
    }
public:
    explicit NativeWorkerCollector(fort_scope_t context) : context_(context) {}
    NativeWorkerCollector(const NativeWorkerCollector &) = delete;
    int add(uint64_t unit,const fort_scope_plan_binding *bindings,size_t count) {
        if (begun_ || (count && !bindings))
            return fort_scope_report_error(FORT_SCOPE_ARGUMENT,"invalid native worker collection");
        try {
            Worker worker{unit,{},{}};
            worker.bindings.reserve(count);
            for (size_t i=0;i<count;++i) {
                fort_scope_layout layout{};
                const auto &binding=bindings[i];
                const int status=fort_scope_layout_get(context_,binding.buffer,&layout);
                if (status) return status;
                const auto &access=binding.access;
                worker.bindings.push_back({binding.buffer,access.flags,
                    {copy(access.reads,access.read_count,layout.rank),
                     copy(access.writes,access.write_count,layout.rank),
                     copy(access.overwrites,access.overwrite_count,layout.rank)},{},{},{}});
            }
            workers_.push_back(std::move(worker));
            return FORT_SCOPE_OK;
        } catch (int status) {
            return fort_scope_report_error(status,"invalid native worker footprint storage");
        } catch (const std::bad_alloc &) {
            return fort_scope_report_error(FORT_SCOPE_RESOURCE,"native worker metadata allocation failed");
        }
    }
    template<class Batch> int add(uint64_t unit,Batch &batch) {
        std::array<fort_scope_plan_binding,Batch::binding_capacity> bindings{};
        size_t count=0;
        const int status=batch.export_bindings(bindings.data(),bindings.size(),&count);
        return status ? status : add(unit,bindings.data(),count);
    }
    int begin(const fort_scope_plan_binding *native_bindings,size_t count,int *applied) {
        if (applied) *applied=0;
        if (begun_)
            return fort_scope_report_error(FORT_SCOPE_STATE,"native worker collection already applied");
        try {
            std::vector<fort_scope_native_worker_v1> workers;
            workers.reserve(workers_.size());
            for (auto &worker : workers_) workers.push_back(worker.view());
            const int status=fort_scope_plan_native_begin_v1(context_,workers.data(),workers.size(),
                                                           native_bindings,count,applied);
            if (!status && applied && *applied) begun_=true;
            return status;
        } catch (const std::bad_alloc &) {
            return fort_scope_report_error(FORT_SCOPE_RESOURCE,"native worker metadata allocation failed");
        }
    }
};
class GPUCall {
    fort_scope_t context_;
    int previous_ = 0;
    bool entered_ = false;
public:
    void *stream = nullptr;
    explicit GPUCall(fort_scope_t context) : context_(context) {}
    GPUCall(const GPUCall &) = delete;
    int begin() {
        const int status = fort_scope_gpu_enter(context_, &previous_, &stream);
        entered_ = status == FORT_SCOPE_OK;
        return status;
    }
    ~GPUCall() { if (entered_) (void)fort_scope_gpu_leave(context_, previous_); }
};
} // namespace fort_scoped
#endif
