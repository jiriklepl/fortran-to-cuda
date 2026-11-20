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
