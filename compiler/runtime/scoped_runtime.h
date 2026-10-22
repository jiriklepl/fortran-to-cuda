/* Versioned shared runtime ABI. Link one scoped_runtime object per executable.
 * All coordinates are zero-based physical offsets with exclusive upper bounds.
 * Host allocations are borrowed and must remain stable until unregister/close.
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
/* Optional submission provenance. Full SHA256 identities resolve through the
 * compiler manifest. Unbound fields are explicitly unknown. Diagnostic calls
 * never change numerical status, coherence or scheduling; tracing disabled
 * returns before a context lookup. Masked fields are updated; null/empty IDs
 * clear them. Unmasked fields are preserved. Mask zero only saves state.
 * Strings must be NUL-terminated within 65 bytes; invalid IDs become unknown.
 * Saved state is caller-owned and restores borrowed-call attribution. Like
 * other context APIs, neither call is legal inside a batch-execute callback,
 * whose executor already holds the nonrecursive context mutex. */
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
    FORT_SCOPE_PLAN_MISMATCH = 9
};
#define FORT_SCOPE_NATIVE_HINT_ABI_VERSION 1
#define FORT_SCOPE_NATIVE_HINT_WORD_LIMIT 512
/* Optional original-native scheduling hint. It owns copied metadata only,
 * never data, resource handles, execution plans or definition authority.
 * Begin/finish belong to the same serial owning invocation. Finish is called
 * only on normal successful completion after every context has been closed.
 * A hit has eligible=0; a qualified miss may return an eligible observation.
 * The 512-word budget includes the runtime's copied entry environment. */
typedef struct fort_scope_native_hint_observation_v1 {
    uint32_t version, eligible;
    size_t word_count;
    uint64_t gpu_generation, error_generation;
    uint64_t words[FORT_SCOPE_NATIVE_HINT_WORD_LIMIT];
} fort_scope_native_hint_observation_v1;
int fort_scope_native_hint_begin_v1(const uint64_t *metadata_words,size_t word_count,
                                   fort_scope_native_hint_observation_v1 *observation,int *hit);
int fort_scope_native_hint_finish_v1(fort_scope_native_hint_observation_v1 *observation);
enum fort_scope_type {
    FORT_SCOPE_BYTES = 0, FORT_SCOPE_REAL32 = 1, FORT_SCOPE_REAL64 = 2,
    FORT_SCOPE_INTEGER32 = 3, FORT_SCOPE_LOGICAL = 4
};
enum fort_scope_access_flags {
    FORT_SCOPE_READ_ALL = 1, FORT_SCOPE_WRITE_ALL = 2, FORT_SCOPE_OVERWRITE_ALL = 4
};
enum fort_scope_execution_mode { FORT_SCOPE_NATIVE = 0, FORT_SCOPE_GPU = 1, FORT_SCOPE_AUTO = 2 };
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

/* Context-owned temporary device storage, independent of registered arrays.
 * A lease is valid only in its owning context until its matching release.
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
 * cached scratch capacity and the peak combined field/scratch payload. */
typedef struct fort_scope_scratch_stats_v1 {
    uint32_t version, active;
    uint64_t acquisitions, allocations, reuses, releases, grows;
    uint64_t capacity_bytes, active_bytes, peak_scratch_bytes, peak_total_device_bytes;
} fort_scope_scratch_stats_v1;

/* Optional planning API. Queries record effects without executing source
 * work or changing coherence. Coordinates and access descriptors match the
 * execution API. Selection compares complete ordered schedules and exports. */
#define FORT_SCOPE_PLANNING_ABI_VERSION 2
/* Planning costs use v2; decisions retain their layout. Reports have their
 * own version, independent of the cost payload. */
#define FORT_SCOPE_PLANNING_REPORT_VERSION 2
enum fort_scope_plan_endpoint { FORT_SCOPE_PLAN_COMPLETE = 0, FORT_SCOPE_PLAN_CONTINUE = 1 };
enum fort_scope_plan_kind { FORT_SCOPE_PLAN_NATIVE = 0, FORT_SCOPE_PLAN_WORKER = 1,
                           FORT_SCOPE_PLAN_FORGET = 2, FORT_SCOPE_PLAN_DISCARD = 3,
                           FORT_SCOPE_PLAN_TEAM_ENTRY = 4, FORT_SCOPE_PLAN_TEAM_NATIVE_CALL = 5 };
int fort_scope_plan_forget_sections_v1(fort_scope_t context, fort_buffer_t buffer,
                                       const fort_scope_section *sections, size_t count);
typedef struct fort_scope_plan_binding {
    fort_buffer_t buffer;
    fort_scope_access access;
} fort_scope_plan_binding;
/* Exact compiled numerical worker metadata, borrowed only during native_begin.
 * The original native body's effects have separate compiler proof authority. */
typedef struct fort_scope_native_worker_v1 {
    uint64_t unit;
    const fort_scope_plan_binding *bindings;
    size_t binding_count;
} fort_scope_native_worker_v1;
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
/* Additive, per-operation computation evidence. Each total includes the
 * backend's arithmetic, numerical primitives, memory throughput and any fixed
 * execution cost not charged by the scope/team protocol. Transfers, scope
 * lifetime and separately calibrated team costs remain outside. Version two
 * can include GPU enqueue in a measured launch total; version one always
 * charges the scope's enqueue estimate separately. The payload layout is
 * unchanged, and version-one callers remain supported.
 * A zero total is known only when its bit is set; missing bits never borrow
 * another backend's estimate. Existing plan_costs/v2 callers remain unchanged. */
#define FORT_SCOPE_COMPUTE_ABI_VERSION 2
enum fort_scope_compute_backend {
    FORT_SCOPE_COMPUTE_NATIVE_FORTRAN = 1,
    FORT_SCOPE_COMPUTE_GENERATED_CPU = 2,
    FORT_SCOPE_COMPUTE_GPU = 4,
    FORT_SCOPE_COMPUTE_ALL = 7,
    FORT_SCOPE_COMPUTE_GPU_LAUNCH_INCLUDED = 8
};
typedef struct fort_scope_compute_costs_v1 {
    uint32_t version, known_backends;
    double native_fortran_seconds, generated_cpu_seconds, gpu_seconds;
} fort_scope_compute_costs_v1;
/* Existing-team costs are a distinct optional contract: serial fork/join rates
 * cannot justify collective placement. Rates in plan_costs describe generated
 * cyclic workers; these native rates describe the original orphaned workers. */
#define FORT_SCOPE_TEAM_ABI_VERSION 1
#define FORT_SCOPE_TEAM_PROTOCOL_ID UINT64_C(0x4654434f4c4c0001)
typedef struct fort_scope_team_costs {
    uint32_t version, valid, cpu_threads, expected_omp_level;
    uint64_t protocol_id;
    double native_cpu_flops, native_cpu_bandwidth;
    double owner_seconds, descriptor_seconds, entry_seconds;
    double cpu_worker_seconds, gpu_worker_seconds, native_call_seconds, native_worker_seconds;
} fort_scope_team_costs;
int fort_scope_set_team_costs_v1(fort_scope_t context, const fort_scope_team_costs *costs, int compatible);
int fort_scope_team_costs_ready_v1(fort_scope_t context, int *ready);
int fort_scope_plan_team_entry_v1(fort_scope_t context);
/* Protocol-only marker for an original native call in a persistent team.
 * Internal host preparation remains PLAN_NATIVE and does not pay this cost. */
int fort_scope_plan_team_native_call_v1(fort_scope_t context);
typedef struct fort_scope_plan_decision {
    uint32_t available, gpu_units, cpu_units, candidates;
    uint64_t simulated_operations, upload_bytes, download_bytes, uploads, downloads;
    uint64_t launches, waits, allocations, peak_device_bytes;
    double estimated_seconds, native_seconds;
} fort_scope_plan_decision;
typedef struct fort_scope_terminal_cost {
    double seconds;
    uint64_t download_bytes, downloads, waits, releases;
} fort_scope_terminal_cost;
typedef struct fort_scope_plan_report {
    uint32_t version, endpoint_mode, available, owner_available;
    double execution_seconds, native_execution_seconds;
    fort_scope_terminal_cost entry_terminal, terminal, native_terminal;
    double ranking_seconds, native_ranking_seconds;
    uint64_t owner_segments;
    double owner_execution_seconds, owner_terminal_seconds, owner_complete_seconds;
    uint32_t native_common_compute_excluded;
} fort_scope_plan_report;
/* Optional calibrated transfer costs. The four capacities are, in order,
 * 256 KiB, 1 MiB, 4 MiB and 16 MiB. Existing planning/profile ABIs stay v1. */
#define FORT_SCOPE_BATCH_ABI_VERSION 1
#define FORT_SCOPE_BATCH_CAPACITIES 4
#define FORT_SCOPE_BATCH_FIXED_AXIS UINT32_MAX
typedef struct fort_scope_batch_costs {
    uint32_t version, valid, async_engine_count;
    size_t max_slot_bytes;
    double staging_cold_seconds[FORT_SCOPE_BATCH_CAPACITIES];
    double staging_reuse_seconds[FORT_SCOPE_BATCH_CAPACITIES];
    double event_record_seconds, event_wait_seconds, ready_event_seconds;
    double preparation_operation_seconds;
    double pack_bytes_per_second, unpack_bytes_per_second, pack_row_seconds, unpack_row_seconds;
    double pinned_h2d_latency, pinned_h2d_bandwidth, pinned_d2h_latency, pinned_d2h_bandwidth;
} fort_scope_batch_costs;
/* Access rectangles describe ordinal zero. For begin/count, shift the chosen
 * axis by the minimum/maximum of step*begin and step*(begin+count-1).
 * FIXED_AXIS permits immutable input sections independent of the partition.
 * Unit records preserve source order, including whole-root FORGET events. */
typedef struct fort_scope_batch_binding {
    fort_buffer_t buffer;
    uint32_t axis;
    int64_t step;
    fort_scope_access access;
} fort_scope_batch_binding;
typedef struct fort_scope_batch_unit {
    uint32_t kind;
    uint64_t unit;
    const fort_scope_batch_binding *bindings;
    size_t count;
    double flops, memory_bytes;
} fort_scope_batch_unit;
typedef struct fort_scope_batch {
    uint32_t version, execution_mode;
    size_t iterations;
    const fort_scope_batch_unit *units;
    size_t unit_count;
    /* Optional host requirements after the subchain: only READ descriptors.
     * Unrelated current device values remain resident. */
    const fort_scope_plan_binding *exports;
    size_t export_count;
} fort_scope_batch;
typedef struct fort_scope_batch_view {
    fort_buffer_t buffer;
    void *device;
    fort_scope_layout layout;
} fort_scope_batch_view;
typedef struct fort_scope_batch_window {
    uint32_t version;
    size_t begin, count;
    void *stream;
    const fort_scope_batch_view *views;
    size_t view_count;
} fort_scope_batch_window;
/* The callback launches the complete admitted subchain in order. It must not
 * call context APIs: the executor owns the context mutex and all coherence.
 * Increment *launches after each actual enqueue, and return execution status.
 * Original extents/lower bounds and full-root pointers are supplied unchanged. */
typedef int (*fort_scope_batch_worker)(const fort_scope_batch_window *window,
                                      void *user, uint64_t *launches);
enum fort_scope_batch_reason {
    FORT_SCOPE_BATCH_NONE = 0, FORT_SCOPE_BATCH_MISSING_COSTS = 1,
    FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN = 2, FORT_SCOPE_BATCH_PLACEMENT = 3,
    FORT_SCOPE_BATCH_NO_ADVANTAGE = 4, FORT_SCOPE_BATCH_BUDGET = 5,
    FORT_SCOPE_BATCH_ALLOCATION = 6, FORT_SCOPE_BATCH_GEOMETRY = 7,
    FORT_SCOPE_BATCH_ARITHMETIC = 8
};
typedef struct fort_scope_batch_report {
    uint32_t version, available, applied, selected_transfers, reason, owner_cost_available;
    uint64_t preparation_operations, batches, chunk_iterations, slot_bytes;
    uint64_t upload_bytes, download_bytes, prefix_upload_bytes, prefix_uploads, launches;
    double estimated_seconds, baseline_seconds, pinned_seconds, pipelined_seconds;
    double execution_seconds, terminal_delta_seconds;
    uint64_t completed_batches, actual_upload_bytes, actual_download_bytes, actual_launches;
} fort_scope_batch_report;
/* Metadata-only transfer pricing, configured before registration/planning.
 * compatible=0 retains unavailable estimates; raw pinned rates are insufficient.
 * Missing costs preserve the existing direct placement/transfer behavior. */
int fort_scope_set_transfer_costs_v1(fort_scope_t context, const fort_scope_batch_costs *costs,
                                    int compatible);
/* One active batch per context; synchronous on return. compatible=-1 previews
 * bounded metadata/cost selection without CUDA, callbacks, resources, schedule
 * consumption or coherence changes. An unapplied result leaves ordinary
 * execution available. Once transfer/numerical work starts, errors poison the
 * context and replay is prohibited. Only all-GPU approved chains are admitted. */
int fort_scope_batch_execute_v1(fort_scope_t context, const fort_scope_batch *batch,
                                const fort_scope_plan_costs *costs,
                                const fort_scope_batch_costs *transfer_costs, int compatible,
                                fort_scope_batch_worker worker, void *user,
                                fort_scope_batch_report *report);
/* Read the last completed batch observation without CUDA initialization. */
int fort_scope_batch_report_get_v1(fort_scope_t context, fort_scope_batch_report *report);
/* Pure check for INTEGER payloads used by planning controls. No transfers or
 * definition changes; the complete payload must already be host current. */
int fort_scope_plan_host_current(fort_scope_t context, fort_buffer_t buffer);
int fort_scope_plan_reset(fort_scope_t context);
/* CONTINUE retains ownership: select installs ordered CPU choices even when
 * GPU estimates are unavailable. Complete prior workers and wait before the
 * next reset. Terminal costs are hypothetical and do not publish or release. */
int fort_scope_plan_reset_mode(fort_scope_t context, uint32_t endpoint_mode);
/* Hand an unstarted, entirely CPU CONTINUE choice to independently checked
 * original native source. Reject prepared accesses, consumed workers or any
 * started numerical activity, and every schedule containing a GPU choice.
 * Only planning metadata is discarded: no predicted definitions are committed,
 * and mirrors, allocations and earlier queued GPU work remain owned. Performs
 * no CUDA work or waits. The pending report and complete-owner estimate become
 * unavailable; native effects must use their own checked coherence hooks. */
int fort_scope_plan_cancel_native_v1(fort_scope_t context);
/* Retire hypothetical choices for an unexecuted continuation suffix. The
 * compiler must prove a boundary before that suffix starts and execute its
 * original source exactly once with independent checked coherence hooks.
 * A completed/accepted prefix is never replayed: its queued commands, virtual
 * definitions, physical versions and storage remain owned. Reject prepared
 * accesses and active batch/scratch leases; discard planning metadata only,
 * without CUDA work or waits.
 * Any pending segment/complete-owner estimate becomes unavailable. Ordinary
 * reset and close retain their unfinished-schedule guards. This is not a
 * recovery API after a partially executed source unit or an execution error. */
int fort_scope_plan_abandon_remaining_v1(fort_scope_t context);
/* Read the last finalized selection's modeled costs without touching storage.
 * Owner totals sum reached continuation execution plus one projected close;
 * unavailable/fallback execution invalidates the complete-owner estimate. */
int fort_scope_plan_report_v2(fort_scope_t context, fort_scope_plan_report *report);
int fort_scope_plan_add(fort_scope_t context, uint32_t kind, uint64_t unit,
                        const fort_scope_plan_binding *bindings, size_t count,
                        double flops, double memory_bytes, int gpu_available);
/* Additional, statically counted numerical costs from compatible offline
 * primitive calibration. Old callers retain zero additional costs. */
int fort_scope_plan_add_costs_v2(fort_scope_t context, uint32_t kind, uint64_t unit,
                        const fort_scope_plan_binding *bindings, size_t count,
                        double flops, double memory_bytes, int gpu_available,
                        double cpu_numerical_seconds, double gpu_numerical_seconds);
/* Native Fortran is the original, uninstrumented counterfactual. Generated
 * CPU is the actual host operation inside the coordinated owner: for WORKER,
 * its CPU implementation; for NATIVE, its original body plus any generated
 * inspection/preparation. Unknown preparation must leave that bit unset,
 * never be excluded as common original-native work. All supplied numbers must
 * be finite and nonnegative. Missing estimates do not prevent independent
 * definition validation. Automatic placement requires the native baseline
 * and every implementation cost used by the candidate and its margin checks;
 * unused alternative estimates may remain unavailable. */
int fort_scope_plan_add_compute_costs_v3(fort_scope_t context, uint32_t kind, uint64_t unit,
                        const fort_scope_plan_binding *bindings, size_t count,
                        double flops, double memory_bytes, int gpu_available,
                        const fort_scope_compute_costs_v1 *compute);
/* Validate a complete, successfully recorded query's ordered definitions.
 * No calibration, estimates, CUDA, transfers, or live coverage changes. Keeps
 * recording open for selection or further queries. Undefined requirements
 * return UNINITIALIZED; bounded metadata failures return BOUNDARY, with an
 * explanation from fort_scope_error. Call before executing source work. */
int fort_scope_plan_validate(fort_scope_t context);
/* compatible=0 never chooses GPU. Numerical clients validate calibration
 * identity against their compiled toolchain and current hardware. It may be
 * checked after compatible=-1 previews a potential GPU advantage without
 * installing a schedule. compatible=1 installs the verified choice. */
int fort_scope_plan_select(fort_scope_t context, const fort_scope_plan_costs *costs,
                           int compatible, fort_scope_plan_decision *decision);
/* Consume only active numerical workers, in recorded source order. Changed
 * unit/effects return PLAN_MISMATCH before advancing the cursor or starting
 * work. Exhausted schedules retain STATE. A compiler may retire the remaining
 * plan only at a proved prework boundary; no completed source work is replayed.
 * Generated entries convert any error after work starts into EXECUTION. */
int fort_scope_plan_next(fort_scope_t context, uint64_t unit,
                         const fort_scope_plan_binding *bindings, size_t count, int *gpu);
/* Atomically hand the current source unit's CPU worker choices to independently
 * checked original native source, retaining later choices and earlier queued
 * GPU work. First match every compiled unit/effect against the current worker
 * range. A mismatch returns PLAN_MISMATCH without consuming or acquiring any
 * access. If a matched worker is GPU-selected, return OK with applied=0; an
 * empty range also declines. Nonworker/team-marker records consume no cursor.
 * For an all-CPU range, validate all native bindings against virtual queued
 * definitions, acquire them as one host batch, then consume exactly that range
 * and return applied=1. Failures never consume choices or leave partial access
 * preparation. Conflicting queued producers may execute during acquisition;
 * execution failures poison the context and prohibit numerical replay.
 * Native bindings must be canonical and independently proved for the original
 * body; they need not equal the individual worker effects. Complete them with
 * the ordinary host_end calls after the original body executes exactly once.
 * Requires an installed CONTINUE schedule and no prepared access/batch/lease.
 * Uncalibrated original-native substitution invalidates pending/owner cost
 * reports; it never claims the generated CPU worker estimate was executed. */
int fort_scope_plan_native_begin_v1(fort_scope_t context,
                                   const fort_scope_native_worker_v1 *workers, size_t worker_count,
                                   const fort_scope_plan_binding *native_bindings, size_t native_count,
                                   int *applied);

uint32_t fort_scope_abi_version(void);
const char *fort_scope_error(void);
int fort_scope_create(int device, fort_scope_t *context);
/* No CUDA initialization or descriptor access. Source scopes retain native
 * collective execution in active teams or if runtime OpenMP support is absent. */
int fort_scope_serial_caller(void);
/* Metadata-only check of the ORIGINAL calling thread's IEEE control state. */
int fort_scope_numerical_environment_supported(void);
/* Read the context ordinal without initializing or selecting a CUDA device. */
int fort_scope_device_get(fort_scope_t context, int *device);
/* Configure before CUDA initialization. Budget counts full-layout payloads
 * plus cached context scratch, independently of the sections transferred. */
int fort_scope_set_device_budget(fort_scope_t context, size_t bytes);
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
/* Metadata-only. Configure before registering/querying numerical work.
 * PINNED is synchronous and uses complete costs from the additive setter.
 * AUTO/PIPELINED retain direct placement estimates; approved independent GPU
 * chains may use the versioned batch executor. Missing costs/proofs/resources
 * leave ordinary direct execution available with an explicit reason. */
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
/* Explicit observation publishes every defined section of the selected
 * canonical resources with one host wait. Undefined holes remain undefined.
 * Discard forgets values without publication and retains reusable storage.
 * Retire destroys a dead resource without publication and completes borrowed
 * host-memory uses before returning; its handle becomes stale. */
int fort_scope_observe_v1(fort_scope_t context, const fort_buffer_t *buffers, size_t count);
int fort_scope_discard_v1(fort_scope_t context, fort_buffer_t buffer);
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
 * Context policies are captured at create: FORT_SCOPE_GRAPH_QUEUE=0 submits
 * each accepted task immediately; FORT_SCOPE_CUDA_GRAPHS=0 uses ordered direct
 * launches. Recoverable prelaunch graph resource failures also use direct
 * launches. Numerical launch/execution failure poisons the context. */
int fort_scope_graph_enqueue_v1(fort_scope_t context, uint64_t code_key,
                                const fort_scope_graph_task_v1 *task, const size_t *argument_sizes);
int fort_scope_graph_flush_v1(fort_scope_t context);
int fort_scope_graph_queue_cancel_v1(fort_scope_t context);
int fort_scope_graph_queue_stats_get_v1(fort_scope_t context, fort_scope_graph_queue_stats_v1 *stats);
/* Opt into graph-oriented placement/search budgets for this execution context. */
int fort_scope_task_graph_enable_v1(fort_scope_t context);
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
