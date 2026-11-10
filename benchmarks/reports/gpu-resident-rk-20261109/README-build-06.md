# Resident RK milestone, 2026-11-09

The current integration candidate executes SGS, second/fourth-order advection,
channel filtering, FD2 diffusion loops, velocity faces/edges, pressure-gradient
work and the local `poisson-solver` pressure projection on GPU. One caller-owned
coherent domain now survives all RK stages and the original allocation swaps.
The complete timestep is not yet resident: native wall-model work, timestep
control and output remain, with publication at timestep return.

## Complete-process comparison

| Case | Four-thread native | be58 | Candidate | Lower time vs native | Lower time vs be58 |
| --- | ---: | ---: | ---: | ---: | ---: |
| simple109 | 11.643 s | 9.680 s | 9.442 s | 18.91% | 2.46% |
| channel100 | 13.461 s | 11.215 s | 10.315 s | 23.37% | 8.03% |

This is **one measurement after one excluded warmup per case/mode**, 12 processes
in rotated order, on CPUs 0,2,4,6. It includes startup, GPU solver compilation and
plans, all output and teardown. Every process/output check passes. No builds,
tests or profiles ran concurrently, and every measured external CPU load is below
the predeclared 0.5-core threshold. All warmups and measurements are retained.
These are fresh measurements of existing control binaries, not fresh release
builds. One paired win per case does not establish a robust improvement. The
seven-round protocol, `7a612e4` comparison and 2×/20× checkpoints remain open.

## Changes that produced this result

- `compiler/runtime/resident_domain.f90` separates storage lifetime from the
  checked numerical entries, using the existing coherence engine. Read-only
  host observation preserves device replicas; host writes use access transactions.
  Allocation identity follows the original three-MOVE_ALLOC exchange. Dead scratch
  uses the existing retire operation and waits for queued users without publication.
- `compiler/scopes/resident_entries.py` exports source-backed numerical loops.
  `elmm_pipeline/resident_rk.py` prepares the configured source, removes selected
  CPU tiling/line scratch, dispatches the generated entries and owns the reached
  RK lifetime. The channel filter keeps its original weights and accumulation
  order. Integer-valued REAL-array fills share the managed numeric implementation.
- A generated CUDA file was insufficient evidence. The first broad candidate
  had 42 entries able to fall back to CPU because descriptor-origin and grid
  scalars appeared only under predicates. The compiler now supports read-only
  scalar VALUE dummies; the adapter uses that contract for already-defined
  geometry. Shared-entry ABI 3 reports the actual value/reference signature.
  Ordinary guarded reference inputs remain protected.
- The pipeline rejects native-only regions from a requested resident GPU entry.
  `verify_resident_execution` matches actual launch/graph-kernel trace events to
  compiler implementation identities. Both latest traces show all required stages
  and zero generated-CPU events in the managed numerical entries. The pressure
  adapter records seven GPU solves per two-step-plus-initialization run, with zero
  CPU FFTW plans. Native controls retain CPU PoisFFT and four verified plans.
- The current candidate excludes the old source-owner/placement wrappers. This
  is still an explicit ELMM integration candidate, not completed general-purpose
  graph-policy migration. Remaining legacy removal follows that migration.

## Milestone checks

Both configurations pass all 20 fields, bounds, halos and all outputs at two
steps, using the approved deterministic pressure reference, four native threads,
`atol=1e-6`, `rtol=1e-5`, unchanged physics and output settings.
Maximum absolute field differences: simple `4.36638367e-09`, channel `2.72425087e-08`.

The focused compiler milestone suites pass 149 tests plus 21 transfer-source
checks. Full maintained-suite and natural 109/2458 full20 qualification are still
required. The previously recorded long GPU-Poisson trajectory mismatch remains
unresolved; short correctness and output checks do not waive it.

Initialization plus two steps transfers:

| Case | Upload bytes / copies | Download bytes / copies |
| --- | ---: | ---: |
| channel-simple | 70,128,560 / 224 | 208,894,368 / 146 |
| channel | 85,585,840 / 370 | 208,894,368 / 146 |

Compared with the immediately preceding stage-scoped candidate, uploads fall by
about two thirds and downloads by 31%. These totals include initialization;
they are not steady-state per-step traffic or a zero-transfer claim.

## Preserved regressions and next work

The first broad candidate took 29.566/62.024 s on simple109/channel100, with hidden
CPU fallback. VALUE geometry and dead-scratch retirement reduced that to
14.875/15.729 s. The latest cross-stage retention gives the table above. All
intermediate samples and failed milestones remain under the artifact root.

A separate 20-step channel profile of the **preceding stage-scoped candidate**
records 1.621 s in CUDA copy APIs, versus 0.100 s in launch APIs. Device-copy
activity is 0.745 s; it overlaps host API spans and must not be added to them.
This identifies copy submission as a substantial cost, not an exclusive-time
model for the latest candidate. The profile includes 8,323 3D and 2,880 2D copy
API calls. GPU graph reuse already occurs (17 instantiations, 1,005 launches).

The next performance work should remove timestep publication by managing the
actual adaptive loop, port or narrow the remaining wall-model/diagnostic access,
and reduce fragmented copy submission while preserving newer replicas and exact
logical coverage. Reuse scratch storage and reduce redundant boundary work once
those transfers are addressed. Keep full-process timing and all output; do not
change the workload to meet the research target.

## Artifacts

All build commands, source identities, binaries, full traces, fields, output
comparisons and timing samples are at `/tmp/fortran-resident-rk-20261109-01`.
The latest receipts are `build-06/linked.json`, `build-06/link-plan.json`,
`milestone-06/result.json`, `milestone-06/backend-qualification.json`, and
`comparison-06/{plan,result,execution}.json`. Earlier snapshots and intermediate
builds remain separate. `summary.json` includes the result distributions (one
measured sample per current case/mode) and all milestone details.

External and mixed source edits, including the dirty local solver checkout and
editor configuration, were preserved. No commit or push is claimed for this work.
