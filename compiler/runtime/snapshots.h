/* Two immutable output slots. One serial caller owns the queue; a single
 * writer consumes successful captures in submission order. The writer may not
 * call this queue or the scoped runtime. It must retain no payload pointer.
 * create charges both device slots to the domain workspace budget and checks
 * both pinned slots against pinned_budget. acquire applies backpressure; a slot
 * remains unavailable through DMA AND writer completion. All acquired slots
 * must be submitted or cancelled. destroy joins even after a writer failure,
 * frees storage and releases the lease before domain teardown. No source work
 * may be replayed after submit starts; callers must propagate every error.
 * These additive interfaces have their own version, independent of entry ABI. */
#ifndef FORT_SNAPSHOTS_H
#define FORT_SNAPSHOTS_H
#include "scoped_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef int (*fort_snapshot_writer_v1)(void *user, const void *payload, size_t bytes);
int fort_snapshot_create_v1(fort_scope_domain_t domain, size_t slot_bytes,
                            size_t pinned_budget, void **queue);
int fort_snapshot_acquire_v1(void *queue, uint32_t *slot);
int fort_snapshot_cancel_v1(void *queue, uint32_t slot);
/* Dense complete allocations concatenated in binding order. Checked sizes
 * must sum to slot_bytes. Reads preserve both live replicas. Capture uses the
 * execution stream; DMA uses an event-ordered separate stream and reads only
 * immutable snapshot storage. Borrowed runtime views never escape callback. */
int fort_snapshot_submit_v1(void *queue, uint32_t slot, fort_scope_t context,
    const fort_scope_external_binding_v1 *bindings, const size_t *bytes, size_t count,
    fort_snapshot_writer_v1 writer, void *user);
int fort_snapshot_wait_v1(void *queue);
int fort_snapshot_destroy_v1(void *queue);
#ifdef __cplusplus
}
#endif
#endif
