"""Scoped tile submit glue over existing access, stream and scratch contracts.

This adds no field coherence registry. The owning scoped emitter supplies its
ordinary AccessBatch/PreparedAccess, source-bound descriptor/size preparation,
original work-started state, and admitted kernel arguments. Scratch is never a
registered field and has no host publication path.
"""


def support_code():
    """Emit once beside scoped_entry.hpp, after including scoped_runtime.h.

    prepare(total, per_batch, stride) performs reached descriptor inquiries and
    checked size arithmetic only. submit(first,count,scratch,stream) issues the
    entire original tile producer/consumer body on that exact stream and records
    each accepted launch normally. No queued task may run these direct batches
    out of order or reuse their scratch concurrently.
    """
    return r'''
namespace fort_scoped {
class TilePrivateLease {
    fort_scope_t context_;
    bool active_ = false;
public:
    fort_scope_scratch_lease_v1 value{};
    explicit TilePrivateLease(fort_scope_t context) : context_(context) {}
    TilePrivateLease(const TilePrivateLease&) = delete;
    ~TilePrivateLease() { if (active_) (void)fort_scope_scratch_release_v1(context_,value.token); }
    int acquire(std::size_t bytes) {
        const int status=fort_scope_scratch_acquire_v1(context_,bytes,&value);
        active_=status==FORT_SCOPE_OK;
        return status;
    }
    int finish() {
        if (!active_) return FORT_SCOPE_STATE;
        active_=false;
        return fort_scope_scratch_release_v1(context_,value.token);
    }
};

template<class Access, class Prepare, class OrderedSubmit, class MarkStarted>
static int tile_private_submit(fort_scope_t context, Access& access,
    std::size_t total, bool prior_work_started, Prepare prepare,
    OrderedSubmit submit, MarkStarted mark_started) {
    if (total==0) return FORT_SCOPE_OK; // No descriptor read, queue drain or GPU initialization.
    auto prework=[&](int status) {
        if (status!=FORT_SCOPE_OK && prior_work_started)
            return fort_scope_execution_error(context,"tile preparation failed after an accepted prefix; replay prohibited");
        return status;
    };
    // Earlier deferred commands must be issued/completed before direct stream
    // use. Waiting does not export or publish a field to the CPU.
    int status=fort_scope_wait(context);
    if (status) return fort_scope_execution_error(context,"cannot drain work before tile scratch submission");
    std::size_t per_batch=0,stride=0;
    status=prepare(total,per_batch,stride);
    if (status) return prework(status);
    if (!per_batch || per_batch>total || !stride || per_batch>static_cast<std::size_t>(-1)/stride)
        return prework(FORT_SCOPE_RESOURCE);
    TilePrivateLease scratch(context);
    status=scratch.acquire(per_batch*stride);
    if (status) return prework(status);
    status=access.begin(true);
    if (status) { access.cancel(); return prework(status); }
    GPUCall gpu(context);
    status=gpu.begin();
    if (status) { access.cancel(); return prework(status); }
    // All resource/descriptor setup has succeeded. Any callback failure from
    // here may follow accepted work; it is terminal, never a native fallback.
    access.executing();
    mark_started();
    for (std::size_t first=0;first<total;) {
        const std::size_t count=total-first<per_batch?total-first:per_batch;
        status=submit(first,count,scratch.value.device,gpu.stream);
        if (status) return fort_scope_execution_error(context,"tile execution failed; replay prohibited");
        first+=count;
    }
    status=access.finish();
    if (status) return fort_scope_execution_error(context,"cannot commit tile field accesses");
    status=scratch.finish();
    if (status) return fort_scope_execution_error(context,"cannot release tile scratch lease");
    return FORT_SCOPE_OK;
}
} // namespace fort_scoped
'''.strip().splitlines()
