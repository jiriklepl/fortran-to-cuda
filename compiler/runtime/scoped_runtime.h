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
enum fort_scope_status {
    FORT_SCOPE_OK = 0, FORT_SCOPE_ARGUMENT = 1, FORT_SCOPE_STALE = 2,
    FORT_SCOPE_ALIAS = 3, FORT_SCOPE_RESOURCE = 4, FORT_SCOPE_BOUNDARY = 5,
    FORT_SCOPE_EXECUTION = 6, FORT_SCOPE_UNINITIALIZED = 7, FORT_SCOPE_STATE = 8
};
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
    FORT_SCOPE_TRANSFERS_PIPELINED = 2, FORT_SCOPE_TRANSFERS_AUTO = 3
};
enum fort_scope_transfer_fallback {
    FORT_SCOPE_TRANSFER_NONE = 0, FORT_SCOPE_TRANSFER_ESTIMATES_UNAVAILABLE = 1,
    FORT_SCOPE_TRANSFER_PIPELINED_UNAVAILABLE = 2, FORT_SCOPE_TRANSFER_BUDGET = 3,
    FORT_SCOPE_TRANSFER_ALLOCATION = 4
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
typedef struct fort_scope_stats {
    uint64_t uploads, downloads, upload_bytes, download_bytes;
    uint64_t allocations, allocated_bytes, peak_device_bytes;
    uint64_t launches, waits, reconciliations;
} fort_scope_stats;

/* Optional planning API v1. Queries record effects without executing source
 * work or changing coherence. Coordinates and access descriptors match the
 * execution API. Selection compares complete ordered schedules and exports. */
#define FORT_SCOPE_PLANNING_ABI_VERSION 1
/* Additive endpoint/report API; existing planning costs and decisions stay v1. */
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
typedef struct fort_scope_plan_costs {
    uint32_t version, valid;
    size_t max_allocation_bytes;
    double cpu_flops, cpu_bandwidth, gpu_flops, gpu_bandwidth;
    double h2d_latency, h2d_bandwidth, d2h_latency, d2h_bandwidth;
    double create_seconds, register_seconds, host_access_seconds, device_access_seconds;
    double gpu_setup_seconds, cold_driver_startup_seconds, allocation_seconds, release_seconds;
    double wait_seconds, launch_enqueue_seconds, planning_operation_seconds;
} fort_scope_plan_costs;
/* Additive, per-operation computation evidence. Each total includes the
 * backend's arithmetic, numerical primitives, memory throughput and any fixed
 * execution cost not charged by the scope/team protocol. Transfers, scope
 * lifetime, GPU enqueue and separately calibrated team costs remain outside.
 * A zero total is known only when its bit is set; missing bits never borrow
 * another backend's estimate. Existing plan_costs/v2 callers remain unchanged. */
#define FORT_SCOPE_COMPUTE_ABI_VERSION 1
enum fort_scope_compute_backend {
    FORT_SCOPE_COMPUTE_NATIVE_FORTRAN = 1,
    FORT_SCOPE_COMPUTE_GENERATED_CPU = 2,
    FORT_SCOPE_COMPUTE_GPU = 4,
    FORT_SCOPE_COMPUTE_ALL = 7
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
 * definition validation, but automatic placement remains unavailable. */
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
/* Consume only active numerical workers, in recorded source order. A sequence
 * mismatch diagnoses an error; it never replays completed source work. */
int fort_scope_plan_next(fort_scope_t context, uint64_t unit,
                         const fort_scope_plan_binding *bindings, size_t count, int *gpu);

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
/* Configure before CUDA initialization. Budget counts live payload bytes of
 * full-layout allocations, independently of the sections transferred. */
int fort_scope_set_device_budget(fort_scope_t context, size_t bytes);
/* Metadata-only. Configure before registering/querying numerical work.
 * PINNED is synchronous and uses complete costs from the additive setter.
 * AUTO/PIPELINED retain direct placement estimates; approved independent GPU
 * chains may use the versioned batch executor. Missing costs/proofs/resources
 * leave ordinary direct execution available with an explicit reason. */
int fort_scope_set_transfers(fort_scope_t context, uint32_t mode);
int fort_scope_transfer_stats_get_v1(fort_scope_t context, fort_scope_transfer_stats *stats);
int fort_scope_register(fort_scope_t context, uint64_t identity, uint64_t generation,
                        const fort_scope_layout *layout, int host_initialized, fort_buffer_t *buffer);
/* Register only source-proven initialized host sections. Re-registering an
 * existing identity never resets its definition or coherence state. */
int fort_scope_register_sections(fort_scope_t context, uint64_t identity, uint64_t generation,
                                const fort_scope_layout *layout, const fort_scope_section *initialized,
                                size_t initialized_count, fort_buffer_t *buffer);
/* A source definition event, such as entry to a plain numeric INTENT(OUT)
 * dummy. Discard old values without publishing them; retain the allocation.
 * No buffer access may be prepared, and earlier context work completes first. */
int fort_scope_forget_definition(fort_scope_t context, fort_buffer_t buffer);
int fort_scope_layout_get(fort_scope_t context, fort_buffer_t buffer, fort_scope_layout *layout);
int fort_scope_host_begin(fort_scope_t context, fort_buffer_t buffer, const fort_scope_access *access);
int fort_scope_host_end(fort_scope_t context, fort_buffer_t buffer);
int fort_scope_device_begin(fort_scope_t context, fort_buffer_t buffer, const fort_scope_access *access, void **device);
int fort_scope_device_end(fort_scope_t context, fort_buffer_t buffer);
/* Cancel only before the associated numerical operation has started. */
int fort_scope_cancel_access(fort_scope_t context, fort_buffer_t buffer);
/* gpu_enter selects the context device; gpu_leave restores the caller's device.
 * The returned stream is used by ALL generated launches in this context. */
int fort_scope_gpu_enter(fort_scope_t context, int *previous_device, void **stream);
int fort_scope_gpu_leave(fort_scope_t context, int previous_device);
int fort_scope_note_launch(fort_scope_t context);
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
