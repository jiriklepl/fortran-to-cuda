# Context scratch ABI v1

`scoped_runtime.h` and `scoped_memory.f90` expose temporary device storage through
`fort_scope_scratch_acquire_v1`, `fort_scope_scratch_release_v1`, and
`fort_scope_scratch_stats_get_v1`. The interfaces are additive; existing array,
session, planning and statistics layouts remain unchanged.

Each context owns one reusable arena and permits one active lease. Acquisition
returns a versioned record containing a fresh token, device pointer, requested
bytes and arena capacity. A zero-byte request returns a null pointer without
initializing CUDA, but still acquires a token that must be released. Records are
outputs; callers need not initialize their version fields. Failed acquisition
clears the output record and creates no lease.

All scratch accesses must execute on the stream returned by
`fort_scope_gpu_enter` for the same context. Record launches and report execution
errors through the existing runtime APIs. Other streams, simultaneous callers
and use after release are unsupported. Scratch contents are unspecified on
acquisition, including reuse; their persistence is not a numerical contract.

Release validates the owning context and exact active token. It invalidates that
lease and retains the arena without a copy or synchronization. Earlier and later
uses are ordered by the shared context stream. Reacquisition within capacity
does not wait. Growth completes earlier uses, frees the old arena, completes its
stream-ordered free, then allocates the replacement. This avoids temporarily
charging both arenas. A failed budget check retains the old idle arena; a failed
replacement allocation can leave the context with no arena. Neither creates an
active lease. Callers must reserve resources before numerical work and must not
replay earlier work after a later failure.

Scratch is separate from registered arrays: it has no borrowed host allocation,
definition coverage, coherence state, or publication operation. Closing a
context with an active lease returns `FORT_SCOPE_STATE` before any array
publication. With no active lease, close publishes defined managed arrays and
retires scratch after completion. Execution failure poisons the context; normal
release, acquisition and close then fail. `fort_scope_abandon` performs best
effort device cleanup, including active or idle scratch, without publishing
invalid results. Stale and foreign lease tokens return `FORT_SCOPE_STALE`.

The device budget includes cached scratch capacity and all full-layout field
allocations. Ordinary and batch planning subtract cached scratch from the field
budget. Scratch capacity is exact rather than rounded to a growth factor. The
existing `fort_scope_stats` still counts only field allocation bytes and peaks.
The additive scratch statistics report acquisitions, allocations, reuse,
releases, successful growth, cached and active bytes, peak scratch bytes, and
peak combined field/scratch payload. Allocator pool overhead and reserved pool
pages are not payload statistics.

These operations provide storage, not a numerical implementation or cost proof.
Existing planning models do not price scratch lifetime costs; nonempty scratch
acquisition marks the complete-owner estimate unavailable. Generated snapshot
implementations must establish their own legality and calibrated allocation,
traffic, launch and teardown costs before automatic promotion. Runtime source
changes also invalidate calibration identity.
