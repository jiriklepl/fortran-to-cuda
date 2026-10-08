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

uint32_t fort_scope_abi_version(void);
const char *fort_scope_error(void);
int fort_scope_create(int device, fort_scope_t *context);
int fort_scope_register(fort_scope_t context, uint64_t identity, uint64_t generation,
                        const fort_scope_layout *layout, int host_initialized, fort_buffer_t *buffer);
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
