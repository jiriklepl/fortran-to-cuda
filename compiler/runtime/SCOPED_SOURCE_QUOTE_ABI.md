# Local source-task quotes

`fort_scope_source_quote_v1` compares one checked source task against the exact
native fallback used by the first source-team emitter. Version 1 requires
`FORT_SCOPE_SOURCE_QUOTE_NATIVE_CLOSE_DOMAIN`: the CPU path publishes all dirty
storage, retires the current context and domain, and then runs the original
complete native team once. The GPU path executes the source task and projects
the same terminal retirement. Both endpoint fields report this requirement.
A native path that keeps a domain live needs a different endpoint contract.

The returned numbers are estimates. Independent native and GPU compute totals
have separate known-backend masks; an unavailable backend is never a zero-cost
backend. Static roofline inputs remain estimates. The quote's own snapshot and
simulation overhead is a common, currently unmeasured term excluded from both
alternatives. The API neither marks the whole owner priced nor replaces its
existing unavailable-cost report.

The initial source-submit protocol uses a conservative maximum-median constant
over its declared binding/launch range, rather than fitted launch scaling.
Cold driver startup is charged only when the runtime has not observed device
initialization in this process. Local choices cannot amortize that startup
across a connected future task group. The projected GPU retirement is a local
comparison endpoint; a successful source task leaves the domain live and does
not establish a profitable whole-owner schedule.

The quote copies live canonical regions, allocation metadata and outstanding
host readers into the existing coherence simulator. It initializes no CUDA
state and does not submit, wait, publish, acquire scratch, install a schedule,
advance a cursor or change definitions. GPU replica reuse and exact CPU
publication follow the same region transitions as ordinary workers. Budget
eviction prefers clean replicas and then least recently used eligible replicas.
The fields needed by the task are pinned during the simulation.

The complete source-submit measurement replaces per-field device-access and
per-kernel enqueue charges. GPU computation must therefore exclude launch
enqueue. Cold scratch acquisition, idle reuse, idle arena growth and release
have separate measurements. Scratch shares the actual device budget with
fields and provider workspace, but carries no array definitions or host
publication. Arena growth cannot evict field allocations before acquiring the
new arena, matching the execution API. Its terminal allocator retirement is
reported separately from local source execution.

Version 1 declines unsubmitted queued work, active leases/prepared accesses,
unfinished installed schedules, pending arena growth and opaque provider
cleanup. Submitted pending field users remain part of the normal completion
and publication simulation. Missing, incompatible, nonfinite or out-of-range
evidence also yields an unavailable quote. This result grants no authority to
execute, initialize storage, validate descriptors or replay attempted work.

`WAIT_BEFORE_LEASE` declares the provider's actual pre-scratch completion fence.
It charges a pending prefix wait before source work, independently of the new
source task's terminal wait. Once that prefix completes, idle arena growth can
use its measured protocol. A quote neither executes that fence nor treats the
prefix as already complete in the real context.

`--scoped-costs --source-task-costs` measures the new runtime once using generic
source-submit/scratch probes and independent original-team protocols. The
submit fit uses a predetermined binding/launch grid; failed holdouts retain
their raw evidence and disable the optional protocol. Original-team rows bind
rank, collapse, unit count, thread budget and actual schedule; they exclude
body computation and PRIVATE allocation. Calibration includes neither an
application workload nor a speedup guarantee. A changed runtime requires fresh
calibration and regenerated artifacts before production AUTO qualification.

`fort_scope_source_quote_v2` uses the additive `NATIVE_LIVE` endpoint. Both
alternatives retain the caller's domain; neither projects a terminal close.
The CPU path uses the same typed bindings as the source device transaction.
It prices only their required publication and pending-user waits, followed by
separately measured complete host transaction management and independently
known original-team/body estimates. Unrelated dirty resources remain live.
The GPU-only pre-scratch fence follows the quote, so a native decision does
not execute that fence. Read-only native accesses retain valid device replicas.

The matching host API separates prepare, start, complete and pre-start cancel.
Prepare validates all typed bindings, aliases, layouts and required definitions
before exposing original CPU work. Start marks the attempt immediately before
the original team. Every later failure poisons the context and prohibits replay.
Complete commits the declared writes through ordinary region coherence. The
initial source emitter uses conservative whole READ/READWRITE footprints and
therefore requires those whole fields already initialized; it grants no new
initialization credit from a partial original write. The runtime API itself
accepts exact partial regions for independently checked callers.

Native management calibration has metadata-only and resident READ/READWRITE
rows with 1/4/8 fit bindings and independent 2/6 holdouts. These measure the
complete prepare/start/complete host interval with no copies or original team
inside it. Failed holdouts disable the native protocol without fabricating a
zero cost or changing the separate source-submit measurements.

Diagnostic trace units depend on the event. `upload` and `download` report
physical payload bytes. The existing `external_commit` event uses its `bytes`
field for the number of buffer bindings, and `graph_submit` uses that field for
the logical kernel count. Neither is a payload-byte total. Source scratch
batching must be checked against actual launch or graph-node records; a source
commit does not report its number of kernel batches.
