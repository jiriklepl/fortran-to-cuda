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
                           FORT_SCOPE_PLAN_FORGET = 2 };
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
/* Read the context ordinal without initializing or selecting a CUDA device. */
int fort_scope_device_get(fort_scope_t context, int *device);
/* Configure before CUDA initialization. Budget counts live payload bytes of
 * full-layout allocations, independently of the sections transferred. */
int fort_scope_set_device_budget(fort_scope_t context, size_t bytes);
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
