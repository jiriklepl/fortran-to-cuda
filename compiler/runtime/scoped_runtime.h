/* Versioned shared runtime ABI. Link one scoped_runtime object per executable.
 * All coordinates are zero-based physical offsets with exclusive upper bounds.
 * Host allocations are borrowed and must remain stable until unregister,
 * retirement, private-context close or owning-domain close. Detaching a
 * borrowed context does not end its domain's storage lifetime.
 * One caller at a time may execute work in a context. Different contexts may
 * execute concurrently if their host accesses do not conflict. */
#ifndef FORT_SCOPED_RUNTIME_H
#define FORT_SCOPED_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define FORT_SCOPE_ABI_VERSION 1
typedef uint64_t fort_scope_t;
typedef uint64_t fort_buffer_t;
typedef uint64_t fort_scope_domain_t;
/* Optional submission provenance. Full SHA256 identities resolve through the
 * compiler manifest. Unbound fields are explicitly unknown. Diagnostic calls
 * never change numerical status, coherence or scheduling; tracing disabled
 * returns before a context lookup. Masked fields are updated; null/empty IDs
 * clear them. Unmasked fields are preserved. Mask zero only saves state.
 * Strings must be NUL-terminated within 65 bytes; invalid IDs become unknown.
 * Saved state is caller-owned and restores borrowed-call attribution. */
#define FORT_SCOPE_TRACE_ABI_VERSION 1
enum fort_scope_trace_field {
    FORT_SCOPE_TRACE_OWNER = 1, FORT_SCOPE_TRACE_PROCEDURE = 2,
    FORT_SCOPE_TRACE_SEGMENT = 4, FORT_SCOPE_TRACE_OPERATION = 8,
    FORT_SCOPE_TRACE_IMPLEMENTATION = 16, FORT_SCOPE_TRACE_BOUNDARY = 32,
    FORT_SCOPE_TRACE_ALL = 63
};
typedef struct fort_scope_trace_state_v1 {
    uint32_t version, reserved;
    char owner[65], procedure[65], segment[65], operation[65], implementation[65], boundary[65];
} fort_scope_trace_state_v1;
void fort_scope_trace_set_v1(fort_scope_t context, uint32_t update_mask, const char *owner, const char *procedure,
                            const char *segment, const char *operation, const char *implementation,
                            const char *boundary, fort_scope_trace_state_v1 *previous);
void fort_scope_trace_restore_v1(fort_scope_t context, const fort_scope_trace_state_v1 *previous);
enum fort_scope_status {
    FORT_SCOPE_OK = 0, FORT_SCOPE_ARGUMENT = 1, FORT_SCOPE_STALE = 2,
    FORT_SCOPE_ALIAS = 3, FORT_SCOPE_RESOURCE = 4, FORT_SCOPE_BOUNDARY = 5,
    FORT_SCOPE_EXECUTION = 6, FORT_SCOPE_UNINITIALIZED = 7, FORT_SCOPE_STATE = 8,
    // Not an error: an entry was compiled for values of its scalars under which its iterations are
    // independent, the values it was called with are others, and nothing was done.
    FORT_SCOPE_DECLINED = 9
};
enum fort_scope_type {
    FORT_SCOPE_BYTES = 0, FORT_SCOPE_REAL32 = 1, FORT_SCOPE_REAL64 = 2,
    FORT_SCOPE_INTEGER32 = 3, FORT_SCOPE_LOGICAL = 4
};
enum fort_scope_access_flags {
    FORT_SCOPE_READ_ALL = 1, FORT_SCOPE_WRITE_ALL = 2, FORT_SCOPE_OVERWRITE_ALL = 4
};
enum fort_scope_execution_mode { FORT_SCOPE_NATIVE = 0, FORT_SCOPE_GPU = 1 };
#define FORT_SCOPE_TRANSFER_ABI_VERSION 1
enum fort_scope_transfer_mode {
    FORT_SCOPE_TRANSFERS_DIRECT = 0, FORT_SCOPE_TRANSFERS_PINNED = 1,
    FORT_SCOPE_TRANSFERS_PIPELINED = 2, FORT_SCOPE_TRANSFERS_AUTO = 3,
    FORT_SCOPE_TRANSFERS_COALESCED = 4
};
enum fort_scope_transfer_fallback {
    FORT_SCOPE_TRANSFER_NONE = 0, FORT_SCOPE_TRANSFER_ESTIMATES_UNAVAILABLE = 1,
    FORT_SCOPE_TRANSFER_PIPELINED_UNAVAILABLE = 2, FORT_SCOPE_TRANSFER_BUDGET = 3,
    FORT_SCOPE_TRANSFER_ALLOCATION = 4, FORT_SCOPE_TRANSFER_COALESCED_INELIGIBLE = 5
};
/* Additive actual-transfer diagnostics. Legacy allocation statistics continue
 * to count full-layout device payloads only. Timings never select a policy. */
typedef struct fort_scope_transfer_stats {
    uint32_t version, requested_mode, effective_mode, fallback_reason;
    uint64_t fallbacks, pinned_uploads, pinned_downloads, pinned_upload_bytes, pinned_download_bytes;
    uint64_t packed_bytes, unpacked_bytes, tiles, events, event_waits;
    uint64_t staging_allocations, staging_reuses, slot_capacity, staging_device_bytes;
    uint64_t process_reserved_bytes, process_peak_bytes;
    double packing_seconds, unpacking_seconds, event_wait_seconds;
} fort_scope_transfer_stats;
#define FORT_SCOPE_COMPLETION_ABI_VERSION 1
typedef struct fort_scope_completion_stats_v1 {
    uint32_t version, reserved;
    uint64_t event_records, event_queries, event_waits, stream_waits;
    uint64_t created_events, reused_events, independent_host_tasks;
} fort_scope_completion_stats_v1;
typedef struct fort_scope_section {
    const size_t *lower;
    const size_t *upper;
} fort_scope_section;
typedef struct fort_scope_access {
    uint32_t flags;
    size_t read_count;
    const fort_scope_section *reads;
    size_t write_count;
    const fort_scope_section *writes;
    size_t overwrite_count;
    const fort_scope_section *overwrites;
} fort_scope_access;
typedef struct fort_scope_layout {
    uint32_t rank, type;
    size_t element_bytes;
    void *host;
    const size_t *extents;
    const int64_t *lower_bounds;
    uint64_t generation;
} fort_scope_layout;
/* Additive borrowed views. Origins are physical root coordinates; extents and
 * lower bounds belong to the original child dummy. No compact allocation or
 * pointer registration is created. Descriptor pointers remain caller-owned. */
#define FORT_SCOPE_VIEW_ABI_VERSION 1
typedef struct fort_scope_view_v1 {
    uint32_t version, rank;
    fort_buffer_t buffer;
    uint64_t generation;
    const size_t *origins, *extents;
    const int64_t *lower_bounds;
} fort_scope_view_v1;
typedef struct fort_scope_view_layout_v1 {
    fort_scope_layout root;
    const size_t *origins, *extents, *byte_strides;
    const int64_t *lower_bounds;
    size_t elements, byte_offset;
} fort_scope_view_layout_v1;
/* Pure descriptor validation, including checked existing INTEGER ABI bounds.
 * Empty views perform no address arithmetic. No CUDA initialization occurs. */
int fort_scope_view_get_v1(fort_scope_t context, const fort_scope_view_v1 *view,
                            fort_scope_view_layout_v1 *layout);
/* Rank-reduced rectangular views. Origins have root_rank elements, while axes,
 * extents and lower_bounds have rank elements. Each retained dummy dimension
 * maps to one distinct zero-based root axis; other axes select one element.
 * Root pitches and allocation identity remain unchanged. Version 1 remains
 * available for existing independently generated companions. */
#define FORT_SCOPE_VIEW_ABI_VERSION_V2 2
typedef struct fort_scope_view_v2 {
    uint32_t version, rank, root_rank, reserved;
    fort_buffer_t buffer;
    uint64_t generation;
    const size_t *origins, *extents;
    const int64_t *lower_bounds;
    const uint32_t *axes;
} fort_scope_view_v2;
typedef struct fort_scope_view_layout_v2 {
    fort_scope_layout root;
    uint32_t rank, reserved;
    const size_t *origins, *extents, *root_byte_strides;
    const int64_t *lower_bounds;
    const uint32_t *axes;
    size_t elements, byte_offset;
} fort_scope_view_layout_v2;
int fort_scope_view_get_v2(fort_scope_t context, const fort_scope_view_v2 *view,
                            fort_scope_view_layout_v2 *layout);
/* Partial INTENT(OUT) changes definitions only inside the supplied root boxes.
 * All three coverage sets are prepared before committing; fragmentation leaves
 * them unchanged. Existing allocation and unrelated current values survive. */
int fort_scope_forget_sections_v1(fort_scope_t context, fort_buffer_t buffer,
                                  const fort_scope_section *sections, size_t count);
/* Compiler-established exact upward reads at the reached original operation,
 * after its guards and original descriptor checks. This is a precondition of
 * well-defined original execution, never a prefix/halo or alias proof. Only
 * requested values absent from initialized coverage gain a current host copy;
 * existing host/device freshness is untouched. No transfers or waits occur.
 * All region algebra succeeds before committing either coverage set. */
int fort_scope_admit_source_reads_v1(fort_scope_t context, fort_buffer_t buffer,
                                     const fort_scope_section *sections, size_t count);
typedef struct fort_scope_stats {
    uint64_t uploads, downloads, upload_bytes, download_bytes;
    uint64_t allocations, allocated_bytes, peak_device_bytes;
    uint64_t launches, waits, reconciliations;
} fort_scope_stats;

/* Temporary device storage, independent of registered arrays.
 * A lease is valid only in its owning context until its matching release.
 * A borrowing context may reuse its domain's idle cached arena.
 * Scratch has no host mirror, definition coverage, or publication operation.
 * All uses must share the context stream and record launches normally. */
#define FORT_SCOPE_SCRATCH_ABI_VERSION 1
typedef struct fort_scope_scratch_lease_v1 {
    uint32_t version, reserved;
    uint64_t token;
    void *device;
    size_t bytes, capacity;
} fort_scope_scratch_lease_v1;
/* Legacy fort_scope_stats remains field-only. This additive record includes
 * cached scratch capacity and the peak combined field/scratch/workspace payload. */
typedef struct fort_scope_scratch_stats_v1 {
    uint32_t version, active;
    uint64_t acquisitions, allocations, reuses, releases, grows;
    uint64_t capacity_bytes, active_bytes, peak_scratch_bytes, peak_total_device_bytes;
} fort_scope_scratch_stats_v1;

/* One buffer of a numerical task with the sections the task reads and writes. */
typedef struct fort_scope_plan_binding {
    fort_buffer_t buffer;
    fort_scope_access access;
} fort_scope_plan_binding;
/* Base rates of the device a physical-copy calibration was measured on. */
#define FORT_SCOPE_PLANNING_ABI_VERSION 2
typedef struct fort_scope_plan_costs {
    uint32_t version, valid;
    size_t max_allocation_bytes;
    double cpu_flops, cpu_bandwidth, gpu_flops, gpu_bandwidth;
    double h2d_latency, h2d_bandwidth, d2h_latency, d2h_bandwidth;
    double create_seconds, register_seconds, host_access_seconds, device_access_seconds;
    double gpu_setup_seconds, cold_driver_startup_seconds, allocation_seconds, release_seconds;
    double wait_seconds, launch_enqueue_seconds, planning_operation_seconds;
    /* Observed direct/pageable fit dispersion. Zero leaves nominal pricing
     * unchanged. Values must be finite, nonnegative and less than one. */
    double h2d_fit_relative_error, d2h_fit_relative_error;
} fort_scope_plan_costs;
/* Optional exact physical-copy calibration.
 * AUTO may use compact copies only inside the observed geometry envelope.
 * Directions are H2D/D2H; each rate predicts the complete physical batch.
 * Submission terms count original operations or compact kernel chunks.
 * Empirical errors are diagnostics, not global profitability bounds.
 * No field grants memory validity, execution, source motion or lifetime. */
#define FORT_SCOPE_COPY_COST_ABI_VERSION 3
typedef struct fort_scope_copy_rate_v3 {
    double batch_seconds, submission_seconds, byte_seconds, row_seconds, descriptor_seconds;
    double median_relative_error, p90_relative_error, maximum_relative_error;
    size_t minimum_payload_bytes, maximum_payload_bytes, minimum_rows, maximum_rows;
    size_t minimum_row_bytes, maximum_row_bytes, minimum_pitch_bytes, maximum_pitch_bytes, maximum_span_bytes;
    size_t minimum_operations, maximum_operations, minimum_descriptor_work, maximum_descriptor_work;
    size_t minimum_kernel_chunks, maximum_kernel_chunks;
    uint32_t word_bytes_mask, layout_combinations_mask;
} fort_scope_copy_rate_v3;
typedef struct fort_scope_copy_costs_v3 {
    uint32_t version, valid;
    double minimum_relative_saving;
    fort_scope_copy_rate_v3 direct[2], compact_cold[2], compact_reuse[2];
} fort_scope_copy_costs_v3;
/* Configure before registration in a private context, or at any time in a
 * borrowed context. The caller's hardware/runtime guard supplies compatible.
 * The domain owns this physical transport calibration: it survives context
 * detach and applies to final publication. A later child may replace it;
 * missing/invalid/incompatible costs clear the domain's model and cannot
 * select compact. */
int fort_scope_set_copy_costs_v3(fort_scope_t context, const fort_scope_copy_costs_v3 *costs,
                                const fort_scope_plan_costs *base, int compatible);

uint32_t fort_scope_abi_version(void);
const char *fort_scope_error(void);
int fort_scope_create(int device, fort_scope_t *context);
/* Caller-owned physical residency, independent of the contexts that execute in it.
 * At most one context borrows a domain at a time. Borrowed context close drains
 * its accepted task queue and detaches WITHOUT publishing or retiring buffers.
 * Domain close is the explicit host/publication and storage-lifetime frontier;
 * it requires no active context, prepared access, scratch or workspace lease.
 * All canonical registrations, exact mirrors and completion dependencies remain
 * owned by the domain until then. Private create/close semantics are unchanged.
 * Physical statistics queried through borrowed contexts are DOMAIN aggregates;
 * graph and queue state is fresh per context. No implicit domain lookup by
 * host address, thread, procedure or allocation identity is performed. */
#define FORT_SCOPE_DOMAIN_ABI_VERSION 1
int fort_scope_domain_create_v1(int device, fort_scope_domain_t *domain);
int fort_scope_create_in_domain_v1(fort_scope_domain_t domain, fort_scope_t *context);
/* Returns zero for a private context. Also usable on a poisoned live context
 * to arrange nonpublishing cleanup; stale contexts remain errors. */
int fort_scope_domain_get_v1(fort_scope_t context, fort_scope_domain_t *domain);
/* Metadata only: query the ordinal before creating a provider's device
 * workspace. Does not select/initialize CUDA; allowed on a failed domain. */
int fort_scope_domain_device_get_v1(fort_scope_domain_t domain, int *device);
/* Physical field-copy/allocation totals, including work done by prior detached
 * contexts. Read-only and available for diagnostics even after a domain fails;
 * external workspace claims are separately charged, not counted as fields. */
int fort_scope_domain_stats_get_v1(fort_scope_domain_t domain, fort_scope_stats *stats);
int fort_scope_domain_close_v1(fort_scope_domain_t domain);
/* Fatal cleanup only, after its failed context has been abandoned/detached.
 * Does not publish or replay work. Provider-owned external memory must first
 * be destroyed by its owner and every workspace lease released; outstanding
 * provider leases reject abandonment. This call retires runtime storage. */
int fort_scope_domain_abandon_v1(fort_scope_domain_t domain);
/* Account externally allocated persistent/peak workspace, WITHOUT allocating
 * another buffer. Reserve a conservative cap before provider construction,
 * shrink after its measured/queryable footprint is known, release only after
 * provider destruction/completion. Lease tokens are domain-specific. Claims
 * count against the SAME payload+scratch budget across detached contexts.
 * Resize/growth is transactional; reservation never evicts a prepared buffer.
 * These calls may not be invoked from an external worker callback. */
int fort_scope_domain_workspace_reserve_v1(fort_scope_domain_t domain, size_t bytes, uint64_t *lease);
int fort_scope_domain_workspace_resize_v1(fort_scope_domain_t domain, uint64_t lease, size_t bytes);
int fort_scope_domain_workspace_release_v1(fort_scope_domain_t domain, uint64_t lease);
/* No CUDA initialization or descriptor access. Source scopes retain native
 * collective execution in active teams or if runtime OpenMP support is absent. */
int fort_scope_serial_caller(void);
/* Metadata-only check of the ORIGINAL calling thread's IEEE control state. */
int fort_scope_numerical_environment_supported(void);
/* Configure before CUDA initialization. Budget counts full-layout payloads
 * plus cached context scratch, independently of the sections transferred. */
int fort_scope_set_device_budget(fort_scope_t context, size_t bytes);
/* The budget for a domain whose caller states none: FORT_RESIDENT_DEVICE_BUDGET
 * (bytes, or a count with K, M or G) when set, otherwise the memory free on the
 * device now less a reserve of one sixteenth of the device, at least 512 MiB,
 * for what the driver and libraries allocate outside the budget. Selects the
 * device to ask it and restores the caller's selection. */
int fort_scope_device_budget_default_v1(int device, size_t *bytes);
/* The domain's budget and the part of it that no field copy, scratch arena or
 * external workspace claims at this moment. Fields that are not on the device
 * yet claim nothing, so this is a cap for a workspace, not a forecast. */
int fort_scope_domain_budget_get_v1(fort_scope_domain_t domain, size_t *budget, size_t *unclaimed);
/* At most one active lease per context. Empty requests acquire a token with a
 * null device pointer and do not initialize CUDA. Positive requests reuse the
 * arena or grow it after completing earlier uses; old capacity is retired
 * before replacement. Resource failure acquires no lease. Release retains
 * capacity without waiting, copying, or publishing. Close rejects an active
 * lease before publication; abandon retires it after execution failure.
 * Output records are runtime-written; callers need not initialize them. */
int fort_scope_scratch_acquire_v1(fort_scope_t context, size_t bytes,
                                  fort_scope_scratch_lease_v1 *lease);
int fort_scope_scratch_release_v1(fort_scope_t context, uint64_t token);
int fort_scope_scratch_stats_get_v1(fort_scope_t context, fort_scope_scratch_stats_v1 *stats);
/* Metadata-only. Configure before registering numerical work. PINNED is
 * synchronous. AUTO uses compact copies where a physical-copy calibration
 * covers them, and direct copies otherwise; PIPELINED is direct. */
int fort_scope_set_transfers(fort_scope_t context, uint32_t mode);
int fort_scope_transfer_stats_get_v1(fort_scope_t context, fort_scope_transfer_stats *stats);
int fort_scope_completion_stats_get_v1(fort_scope_t context, fort_scope_completion_stats_v1 *stats);
int fort_scope_register(fort_scope_t context, uint64_t identity, uint64_t generation,
                        const fort_scope_layout *layout, int host_initialized, fort_buffer_t *buffer);
/* Register only source-proven initialized host sections. Re-registering an
 * existing identity never resets its definition or coherence state. */
int fort_scope_register_sections(fort_scope_t context, uint64_t identity, uint64_t generation,
                                const fort_scope_layout *layout, const fort_scope_section *initialized,
                                size_t initialized_count, fort_buffer_t *buffer);
/* Commit an independently proved, already executed descriptor-only exchange
 * (for example a complete MOVE_ALLOC swap). Registered identity slots must be
 * distinct. Each post-layout must exactly describe the opposite existing
 * physical allocation, including allocation generation and lower bounds.
 * Physical handles/storage/definitions/mirrors/completions are never moved,
 * copied or reinitialized; only the logical identity map and returned caller
 * bindings change. Accepted queued tasks retain their physical handles.
 * There must be no prepared access.
 * This is a POST-mutation contract: every validation/commit failure in a live
 * context poisons its domain and returns EXECUTION, never authorizes replay.
 * Caller compiler registration epochs are separate from allocation generation. */
int fort_scope_identity_swap_v1(fort_scope_t context, fort_buffer_t first, fort_buffer_t second,
                                const fort_scope_layout *first_after, const fort_scope_layout *second_after,
                                fort_buffer_t *first_current, fort_buffer_t *second_current);
/* A source definition event, such as entry to a plain numeric INTENT(OUT)
 * dummy. Forget semantic definitions while retaining physical mirror versions
 * for native continuation/publication. No transfer or completion wait occurs.
 * Explicit discard_v1 additionally discards the physical values. No access
 * may be prepared on the affected buffer. */
int fort_scope_forget_definition(fort_scope_t context, fort_buffer_t buffer);
int fort_scope_layout_get(fort_scope_t context, fort_buffer_t buffer, fort_scope_layout *layout);
/* Acquire all canonical resources of one numerical task transactionally.
 * Every handle, effect, required definition and resulting coherence state is
 * checked before any allocation or transfer. Duplicate handles are rejected;
 * callers combine read-only aliases before acquisition. On success all
 * accesses are prepared. A host task waits only for required downloads and
 * overlapping uploads that borrow its written host sections. Current host
 * read mirrors and disjoint host writes execute without waiting on unrelated
 * GPU work. Completion events are shared across each transfer batch.
 * On failure no access is prepared and no numerical write is committed;
 * successful allocation/transfer caches may remain. Device pointer outputs
 * are written only on success and are required for nonempty device batches.
 * An empty batch does not initialize CUDA or wait for unrelated device work.
 * Existing end/cancel calls complete each acquired resource. */
int fort_scope_access_begin_v1(fort_scope_t context, const fort_scope_plan_binding *bindings,
                               size_t count, int device, void **devices);
/* Acquire conservative may-access envelopes of unchanged original CPU work.
 * Caller must prove complete effects, stable canonical storage and synchronous
 * completion. All tracked physical values in read/preservation regions reach
 * the host; unknown holes are neither copied nor declared initialized. Pending
 * conflicting device work and overlapping DMA readers obey ordinary task
 * ordering. READ retains device replicas. Successful host_end invalidates
 * possible writes on the device, but only proved overwrites add definitions.
 * The source body executes exactly once after successful acquisition. Complete
 * every binding with host_end; cancel only before that body starts. Failures
 * after source execution must never cause replay. Numerical/device access uses
 * access_begin_v1 and retains its strict initialization requirements. */
int fort_scope_original_host_begin_v1(fort_scope_t context, const fort_scope_plan_binding *bindings,
                                       size_t count);
/* Checked opaque device transaction. Bindings describe physical canonical
 * roots, using the same exact read/write/overwrite rules as generated tasks.
 * Types/ranks/element sizes are checked before any access or provider call.
 * Views and their descriptor arrays are borrowed ONLY until callback return.
 * Device pointers address physical allocation origins (not interior sections).
 * All provider work must use the supplied context stream and retain no view.
 * No scoped/domain API may be called inside the callback (domain mutex held).
 * started is initialized to -1: the worker MUST write 0 on a checked rejection
 * before enqueue, and write 1 BEFORE its first enqueue attempt. Success with
 * nonempty bindings requires started=1; unreported state, exceptions, and any
 * failure after an enqueue attempt poison the domain/context. Native replay
 * is then forbidden. Explicit prework rejection cancels every prepared write;
 * cached allocations/read mirrors may remain. Success commits exact effects
 * and may be asynchronous; normal host access/wait/domain close observes its
 * completion. A zero binding count invokes no callback and initializes no CUDA.
 * This is not a definition proof or a claim that one opaque operation is
 * exactly one CUDA kernel. */
#define FORT_SCOPE_EXTERNAL_ABI_VERSION 1
typedef struct fort_scope_external_binding_v1 {
    fort_buffer_t buffer;
    uint32_t type, rank;
    size_t element_bytes;
    fort_scope_access access;
} fort_scope_external_binding_v1;
typedef struct fort_scope_external_view_v1 {
    fort_buffer_t buffer;
    uint64_t generation;
    uint32_t type, rank;
    size_t element_bytes, bytes;
    const size_t *extents, *strides; /* strides are in BYTES */
    const int64_t *lower_bounds;
    void *device;
} fort_scope_external_view_v1;
typedef struct fort_scope_external_call_v1 {
    uint32_t version, reserved;
    const fort_scope_external_binding_v1 *bindings;
    size_t binding_count;
    uint64_t workspace_lease;
    size_t workspace_bytes;
} fort_scope_external_call_v1;
typedef struct fort_scope_external_report_v1 {
    uint32_t version;
    int32_t status;
    uint32_t started, committed;
} fort_scope_external_report_v1;
typedef int (*fort_scope_external_worker_v1)(void *user, const fort_scope_external_view_v1 *views,
                                             size_t count, void *stream, int *started);
int fort_scope_external_execute_v1(fort_scope_t context, const fort_scope_external_call_v1 *call,
                                   fort_scope_external_worker_v1 worker, void *user,
                                   fort_scope_external_report_v1 *report);
/* Explicit observation publishes every tracked physical section of the selected
 * canonical resources with one host wait. Unknown definitions remain unknown,
 * including current representations surviving a semantic definition kill.
 * Discard forgets values without publication and retains reusable storage.
 * Retire destroys a dead resource without publication and completes borrowed
 * host-memory uses before returning; its handle becomes stale. */
int fort_scope_observe_v1(fort_scope_t context, const fort_buffer_t *buffers, size_t count);
/* Conservative pre-entry observation of all currently registered resources.
 * Drains their queued users and publishes tracked physical regions. Preserves
 * device replicas, allocations and generations; leaves no prepared access.
 * Does not inspect unregistered descriptors or infer any definitions. A caller
 * must stop on error, before original argument/specification evaluation. */
int fort_scope_observe_all_v1(fort_scope_t context);
int fort_scope_discard_v1(fort_scope_t context, fort_buffer_t buffer);
/* Dead values of a buffer its owner is about to define again. Like discard when
 * no queued task touches the buffer; otherwise the values stay until the new
 * definition replaces them, and the queue is not submitted. */
int fort_scope_recycle_v1(fort_scope_t context, fort_buffer_t buffer);
int fort_scope_retire_v1(fort_scope_t context, fort_buffer_t buffer);
int fort_scope_host_begin(fort_scope_t context, fort_buffer_t buffer, const fort_scope_access *access);
int fort_scope_host_end(fort_scope_t context, fort_buffer_t buffer);
int fort_scope_device_begin(fort_scope_t context, fort_buffer_t buffer, const fort_scope_access *access, void **device);
int fort_scope_device_end(fort_scope_t context, fort_buffer_t buffer);
/* Allocate/reuse storage only, without admitting definitions, transferring
 * values, preparing an access or claiming that numerical work has executed.
 * Intended for constructing graph argument values before graph_execute. */
int fort_scope_device_reserve_v1(fort_scope_t context, fort_buffer_t buffer, void **device);
/* Reserve an entire task footprint before constructing its kernel arguments.
 * Duplicate handles return the same pointer. An impossible unique footprint
 * fails before eviction, transfer or output mutation. Under memory pressure,
 * accepted queued work is submitted before unrequested LRU allocations are
 * evicted. Exact dirty values are published; semantic definitions and host
 * mirrors survive. Prepared accesses cannot be evicted. */
int fort_scope_device_reserve_many_v1(fort_scope_t context, const fort_buffer_t *buffers,
                                      size_t count, void **devices);
/* Metadata-only membership probe for borrowed handles across context epochs.
 * A foreign/stale buffer is reported false; a stale context is still an error. */
int fort_scope_buffer_valid_v1(fort_scope_t context, fort_buffer_t buffer, int *valid);
#define FORT_SCOPE_RESIDENCY_ABI_VERSION 1
typedef struct fort_scope_residency_stats_v1 {
    uint32_t version, reserved;
    uint64_t reserve_batches, requested_resources, reuse_hits;
    uint64_t evictions, evicted_bytes, dirty_evictions, dirty_download_bytes, pressure_flushes;
} fort_scope_residency_stats_v1;
int fort_scope_residency_stats_get_v1(fort_scope_t context, fort_scope_residency_stats_v1 *stats);
/* Cancel only before the associated numerical operation has started. */
int fort_scope_cancel_access(fort_scope_t context, fort_buffer_t buffer);
/* gpu_enter selects the context device; gpu_leave restores the caller's device.
 * The returned stream is used by ALL generated launches in this context. */
int fort_scope_gpu_enter(fort_scope_t context, int *previous_device, void **stream);
int fort_scope_gpu_leave(fort_scope_t context, int previous_device);
int fort_scope_note_launch(fort_scope_t context);
/* Ordered kernel groups share the context stream and resource authority.
 * Each node depends on its predecessor, preserving global kernel boundaries.
 * A stable nonzero group key reuses an executable graph. Topology/function or
 * argument-count changes rebuild it; every replay updates kernel parameters,
 * dimensions and shared memory. CUDA copies argument values during this call;
 * descriptors, argument arrays and pointed-to values are borrowed only until
 * return. argument_count must match the compiled kernel signature. A kernel
 * argument containing a device pointer must refer to a currently acquired
 * context resource or scratch lease. Call device_end after successful submit;
 * do not also call note_launch. A failed launch poisons the context and must
 * never be replayed as native work. No-op groups initialize no CUDA resources.
 * The CPU reference backend returns RESOURCE for a nonempty group. */
#define FORT_SCOPE_GRAPH_ABI_VERSION 1
typedef struct fort_scope_kernel_v1 {
    uint32_t version, reserved;
    const void *function;
    uint32_t grid[3], block[3];
    size_t shared_bytes;
    void *const *arguments;
    size_t argument_count;
} fort_scope_kernel_v1;
typedef struct fort_scope_graph_stats_v1 {
    uint32_t version, reserved;
    uint64_t creations, instantiations, rebuilds, submissions, replays;
    uint64_t parameter_updates, logical_kernels, live_graphs, destructions;
} fort_scope_graph_stats_v1;
typedef struct fort_scope_graph_task_v1 {
    fort_scope_kernel_v1 kernel;
    const fort_scope_plan_binding *bindings;
    size_t binding_count;
} fort_scope_graph_task_v1;
typedef struct fort_scope_graph_queue_stats_v1 {
    uint32_t version, queue_enabled, graphs_enabled, reserved;
    uint64_t accepted_tasks, submissions, conflict_flushes, explicit_flushes;
    uint64_t pending_tasks, peak_pending_tasks, copied_argument_bytes;
    uint64_t direct_submissions, direct_kernels, graph_resource_fallbacks;
} fort_scope_graph_queue_stats_v1;
/* Defer one checked task across generated worker boundaries. The runtime owns
 * copies of all argument values and effect coordinates after success. Sizes
 * match the compiled kernel parameter types. Host accesses drain conflicting
 * queued tasks and dependency ancestors; independent work remains queued.
 * Explicit flush permits callers to trade submission batching for overlap.
 * Cancel discards only work that has never been submitted and is explicit.
 * The queue also submits itself once the launches it holds start 2^18 device
 * threads or more, so that the device does not wait for work of that size;
 * FORT_SCOPE_GRAPH_QUEUE_WORK sets another count, and 0 none.
 * Context policies are captured at create: FORT_SCOPE_GRAPH_QUEUE=0 submits
 * each accepted task immediately; FORT_SCOPE_CUDA_GRAPHS=0 uses ordered direct
 * launches. Recoverable prelaunch graph resource failures also use direct
 * launches. Numerical launch/execution failure poisons the context. */
int fort_scope_graph_enqueue_v1(fort_scope_t context, uint64_t code_key,
                                const fort_scope_graph_task_v1 *task, const size_t *argument_sizes);
/* Defer one launch whose kernel runs several members concurrently. Each member
 * has its own effects. The launch is accepted only when no member reads or
 * writes a section that another member writes, judged on the actual sections;
 * otherwise this returns BOUNDARY and queues nothing, and the caller runs the
 * members one after another. An accepted launch carries the union of the
 * members' effects. Other statuses are those of graph_enqueue. */
typedef struct fort_scope_graph_member_v1 {
    const fort_scope_plan_binding *bindings;
    size_t binding_count;
} fort_scope_graph_member_v1;
int fort_scope_graph_enqueue_members_v1(fort_scope_t context, uint64_t code_key,
                                        const fort_scope_kernel_v1 *kernel, const size_t *argument_sizes,
                                        const fort_scope_graph_member_v1 *members, size_t member_count);
int fort_scope_graph_flush_v1(fort_scope_t context);
int fort_scope_graph_queue_cancel_v1(fort_scope_t context);
int fort_scope_graph_queue_stats_get_v1(fort_scope_t context, fort_scope_graph_queue_stats_v1 *stats);
int fort_scope_graph_submit_v1(fort_scope_t context, uint64_t group_key,
                               const fort_scope_kernel_v1 *kernels, size_t count);
/* Checked graph execution simulates per-task effects in order on private
 * states, so a producer can define an intermediate consumed later in the
 * group. It uploads only the external requirements before submission and
 * commits final definitions/coherence only after successful submission. No
 * resource may already be prepared. Reserve device storage before building
 * kernel argument values. No device_end calls follow graph_execute. */
int fort_scope_graph_execute_v1(fort_scope_t context, uint64_t group_key,
                                const fort_scope_graph_task_v1 *tasks, size_t count);
int fort_scope_graph_stats_get_v1(fort_scope_t context, fort_scope_graph_stats_v1 *stats);
/* Numerical clients report execution errors after work has started. This
 * poisons the context; report_error alone only preserves a diagnostic. */
int fort_scope_execution_error(fort_scope_t context, const char *message);
int fort_scope_report_error(int status, const char *message);
int fort_scope_wait(fort_scope_t context);
int fort_scope_stats_get(fort_scope_t context, fort_scope_stats *stats);
int fort_scope_unregister(fort_scope_t context, fort_buffer_t buffer);
int fort_scope_close(fort_scope_t context);
/* Release a failed context WITHOUT publication. Only valid after execution
 * failure: the numerical result is invalid and must never be replayed. */
int fort_scope_abandon(fort_scope_t context);

#ifdef __cplusplus
}
#endif
#endif
