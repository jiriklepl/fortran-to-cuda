// Per-entry glue contains no shared registry or implicit current context.
#ifndef FORT_SCOPED_ENTRY_HPP
#define FORT_SCOPED_ENTRY_HPP
#include "scoped_runtime.h"
#include <array>
#include <utility>

namespace fort_scoped {
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
    bool device_ = false, executing_ = false;
    static bool writes(const fort_scope_access &a) {
        return (a.flags & FORT_SCOPE_WRITE_ALL) || a.write_count;
    }
public:
    explicit AccessBatch(fort_scope_t context) : context_(context) {}
    AccessBatch(const AccessBatch &) = delete;
    ~AccessBatch() {
        if (executing_) (void)fort_scope_execution_error(context_, "numerical entry exited during an operation");
        else cancel();
    }
    int add(fort_buffer_t handle, const fort_scope_access &access) {
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
    int begin(bool device) {
        device_ = device;
        for (size_t i=0; i<size_; ++i) {
            auto &b = bindings_[i];
            const int status = device
                ? fort_scope_device_begin(context_, b.handle, &b.effects, &b.device)
                : fort_scope_host_begin(context_, b.handle, &b.effects);
            if (status) { cancel(); return status; }
            b.prepared = true;
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
    }
    void executing() { executing_ = true; }
    int finish() {
        for (size_t i=0; i<size_; ++i) {
            auto &b = bindings_[i];
            const int status = device_ ? fort_scope_device_end(context_, b.handle)
                                       : fort_scope_host_end(context_, b.handle);
            if (status) return fort_scope_execution_error(context_, "cannot commit numerical operation");
            b.prepared = false;
        }
        executing_ = false;
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
