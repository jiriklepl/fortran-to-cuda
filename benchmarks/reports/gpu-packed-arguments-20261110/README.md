# Packed graph arguments: candidate milestone, 2026-11-10

The generic runtime now stores each queued kernel's argument payload in one
aligned buffer. CUDA graph replay compares complete owned argument bytes,
argument sizes, grid/block dimensions and shared-memory size against the
installed node. Unchanged nodes need no parameter-update API call. Function
identity remains part of the existing graph topology check. Scalar values,
physical pointers and geometry changes still update the affected nodes.

The three-round pilot lowers median complete-process time by **3.37% on
simple109 and 6.42% on channel100**. It wins 3/3 simple pairs and 2/3 channel
pairs; the other channel pair is 0.62% slower. **The predeclared all-pairs pilot
gate fails. This is an unpromoted candidate, not a robust speedup claim.** The
retained source-built GPU control remains the selected benchmark baseline.

## Isolated complete-process comparison

One excluded warmup and three rotated measurements per mode and case; four
threads, affinity `0,2,4,6`, original configurations and output. Startup, solver
plan creation, computation, output and teardown are included. No builds, tests
or profiles ran alongside timing. All 16 timing output comparisons pass; maximum
external CPU is 0.197 cores, below the 0.5-core bound. All owned groups are reaped.
No samples were retried, discarded or trimmed.

| Case | Retained GPU median | Packed candidate median | Median change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| simple109 | 4.339995 s | 4.193638 s | −3.37% | 3/3 |
| channel100 | 5.014085 s | 4.692143 s | −6.42% | 2/3 |

| Case | Variant | All measured samples (s) |
| --- | --- | --- |
| simple109 | retained | 4.634282 / 4.275482 / 4.339995 |
| simple109 | packed | 4.557158 / 4.126320 / 4.193638 |
| channel100 | retained | 5.165416 / 5.014085 / 4.717477 |
| channel100 | packed | 5.197473 / 4.673913 / 4.692143 |

The two binaries use the same qualified source-build numerical objects, local
GPU Poisson provider and native application code. Only the single shared runtime
object is rebuilt for this bounded pilot. Its exact link commands and all
inputs are preserved. These numbers are not a new comparison against native,
be58 or 7a612e4. The prior fresh source-build comparison measured 3.02×/2.74×
native; ratios from different campaigns must not be multiplied.

## Memory, execution and application milestone

47 graph-cache and real-CUDA queue cases pass. They cover aligned mixed-size
payloads, ownership after caller mutation, size/padding overflow, unchanged
replay, changed scalars/pointers/launch geometry, raw graph calls, partial
parameter-update failure, queued lifetime, host/device mirrors and terminal
execution failure. Raw graph calls without complete argument sizes continue
to update every node; pointer equality is never used as value evidence.

Argument snapshots are committed only after a successful CUDA parameter update.
All graph resources and owned host payloads are released with the existing
context. No second coherence engine, public runtime ABI or configuration switch
is introduced. Captured bytes use an explicitly aligned byte-word type so their
copies never interpret arbitrary payload bits as floating-point values.

Both two-step cases, natural simple109 and channel100 pass all 20 fields,
bounds, halos and outputs at unchanged `atol=1e-6`, `rtol=1e-5`. These candidate
runs compare against the immutable, hash-checked approved native reference
outputs. All 40 field-data/bounds files per case are bit-identical to the
retained GPU build. The deterministic pressure reference and four-thread native
configuration are unchanged. Actual GPU stage audits pass; candidate CPU FFTW
plan count remains zero.

The cache-only predecessor reduced short-run parameter updates from 451→259
(simple) and 637→294 (channel), with unchanged kernel and transfer counts.
Its median times improved 6.22%/0.28%, with 3/3 and 1/3 paired wins, so that
version was not promoted. Packing also removes per-argument payload allocations
and permits contiguous byte comparisons. No timing decomposition claims these
mechanisms explain a specific fraction of wall time.

## Source build and remaining qualification

A clean build through `pipeline.py resident-build --build-prepared` **passes**
at `/tmp/fortran-packed-source-build-20261110-01`: all 1,083 build steps complete,
including 248 numerical CUDA entries and one shared runtime. Every object is
compiled afresh, reusing only the immutable prepared source from the prior
source-build milestone. No historical objects or Fortran modules are imported.
The provider source is unchanged and remains based on the pinned local Poisson
revision; GPU binaries resolve the freshly built provider, native binaries do
not load it. Required strict CUDA arithmetic flags are verified.

Eight fresh native/candidate diagnostic processes pass both two-step cases,
natural simple109 and channel100 at the existing tolerances. All 40 field-data
and bounds files per process are bit-identical to the corresponding retained
native/GPU build. The reached GPU stage audit passes. Short-run physical-copy
ledgers and kernel counts are unchanged; parameter updates remain 451→259 and
637→294. `qualification.json` binds these checks to the new binary/source hashes.

This clean build is an integration check; it does not erase or replace the
failed pilot gate and has no additional untraced timing campaign. The candidate
remains unpromoted. Release work must refresh runtime cost calibration and the
complete generated artifact manifests as well as run the formal protocol.

The natural channel2458 discrepancy remains unresolved. Full maintained suites,
fresh four-control/seven-round qualification, generic scheduling migration,
remaining legacy-policy deletion and the 20× research objective remain open.
The complete supported numerical hot path remains on GPU; startup, scheduling,
required scalar host work and output are not claimed GPU-only.

## Evidence

Pilot source, compilation, complete outputs, samples and controller receipts:
`/tmp/fortran-packed-graph-20261110-02`. The cache-only predecessor is preserved
at `/tmp/fortran-graph-parameters-20261110-01`. An initial packed build was
superseded before any application run to make byte-storage copy semantics
explicit; its build and decision remain in `/tmp/fortran-packed-graph-20261110-01`.
The [rejected index experiments](../gpu-index32-rejected-20261110/README.md)
are archived and removed from the compiler emitter.

## Subsequent full-path comparison

The [fresh four-control campaign](../gpu-resident-four-controls-20261110/README.md)
now measures the fully regenerated source build with seven rounds and all four
fresh variants. It passes the primary ≥2× native and ≥10% below 7a timing gates.
This compares the complete resident GPU path with the original controls; it does
not replace this failed isolated runtime pilot or establish its all-pairs gain.
The refreshed 32-case graph-queue calibration records raw costs and parameter
update counts without automatic-placement authority. Natural channel2458 and
full release qualification remain unresolved.
