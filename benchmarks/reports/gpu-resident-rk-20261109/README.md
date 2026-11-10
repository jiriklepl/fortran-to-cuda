# GPU residency across timesteps, 2026-11-09

The current candidate keeps the checked domain across the adaptive time loop,
including RK allocation swaps, and adds GPU CFL calculation and output averages.
SGS, advection, channel filtering, FD2 diffusion loops, velocity faces/edges,
pressure-gradient work and the local `poisson-solver` pressure projection already
use GPU entries. Native wall-model computation and correction loops remain.
This is an implementation candidate, not completed production graph migration.

## Complete-process timing scout

| Case | Four-thread native | be58 | Preserved build-06 | New build-08 | Speedup over native | Lower time than be58 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simple109 | 12.133 s | 9.570 s | 11.646 s | 7.126 s | 1.70× | 25.54% |
| channel100 | 13.521 s | 11.430 s | 10.781 s | 8.373 s | 1.61× | 26.74% |

One excluded warmup and one measured sample per variant/case: 16 processes,
rotated order, CPUs 0,2,4,6, four native threads. Startup, GPU solver compilation
and plan creation, output and teardown are included. No persistent GPU provider
cache is enabled. No builds, tests or profiles ran alongside these timings.
Every process and output comparison passed; every measured external CPU load
was below the predeclared 0.5-core limit. All warmups and samples are retained.

These are fresh measurements of existing native/be58 controls, not freshly rebuilt
release controls. The previous candidate is the identical preserved build-06
binary; its timings differ from the earlier scout, so the table uses measurements
from this run. One paired win does not establish robustness. Seven rotated rounds,
the `7a612e4` control and the 2×/20× research checkpoints remain open.

## Implementation

- `elmm_pipeline/resident_time_loop.py` extends the explicit primary-channel
  lifetime through the main loop. Timestep return no longer publishes ordinary
  fields. Unsupported native output work closes the epoch before accessing data.
- Compiler-generated entries compute the original per-cell CFL arithmetic and
  weighted averages. `compiler/runtime/reductions.{h,cu,f90}` provides a generic
  dense REAL64 maximum with a separate NaN flag. Two kernels use accounted scratch
  and the existing checked external-task interface; only a 16-byte scalar record
  returns to the host. Adaptive timestep selection and scalar control remain host.
- Read-only reduction and frame publication preserve valid GPU replicas. A frame
  observes inputs only after its original time predicate is reached and the prior
  writer completes. The existing Fill/DoSave path, frame contents and completion
  guarantees are retained. No output frames are removed.
- Final publication now precedes diagnostic dumps and field deallocation, including
  the full20 build. The local solver remains the GPU pressure backend, with no CPU
  FFTW plans in the candidate; controls retain CPU PoisFFT and four verified plans.
- The earlier compiler VALUE ABI, verified GPU entry selection and dead-scratch
  retirement remain in use. See [the preserved build-06 report](README-build-06.md)
  for those changes. The candidate omits the old owner/placement wrappers, but
  repository-wide legacy removal and public build integration are still unfinished.

## Milestone verification

Both primary configurations pass all 20 fields, bounds, halos and every output at
2 steps, retaining the approved deterministic pressure reference and
`atol=1e-6`, `rtol=1e-5`. Runtime backend identities confirm actual GPU execution
of the numerical stages and CFL, with zero generated-CPU events in managed
entries. Averaging is reached and verified in the simple short case; channel
averaging starts later. Each short trace records one GPU CFL reduction after
native first-step setup, with a 16-byte scalar download.

The new CUDA reduction milestone passes queued-producer ordering, read-only
replica preservation, irregular sizes through 1,048,579 elements, negative values,
NaNs, infinities, pre-submission rejection and memory-budget exhaustion. The
preceding 149 compiler and 21 transfer-source checks remain recorded; the full
suite was not rerun for this implementation scout.

The first build-07 run failed because a scratch token was incorrectly passed as
an external workspace reservation. The runtime rejected it before reduction
submission. Build-08 uses the separately accounted scratch correctly. The failed
build/run is preserved and contributes no timing sample.

The prior natural2458 GPU-Poisson trajectory mismatch remains unresolved. Short
checks and primary output comparisons do not waive full natural109/2458 all20
qualification or the existing tolerances.

Initialization plus two-step field-transfer totals (scalar reduction separate):

| Case | Upload bytes / copies | Download bytes / copies |
| --- | ---: | ---: |
| channel-simple | 66,864,272 / 155 | 209,880,384 / 124 |
| channel | 54,715,248 / 224 | 190,002,720 / 120 |

These include initialization and final publication. They are not steady-state
per-step counts, and this is not a zero-transfer claim.

## Remaining hot path

A separate 20-step channel diagnosis of build-08 records 0.477 s in the
three CUDA copy APIs, 0.098 s in kernel/graph launch APIs and 0.184 s of device
copy activity. Host API and device durations overlap and must not be added.
It records 118.429 MB H2D and 1,388.312 MB D2H, including initialization and
teardown. The earlier build-05 profile recorded 1.621 s in copy APIs and 0.745 s
of device-copy activity; these diagnostic observations are not untraced speedups.

The next implementation target is native wall-model work and its correction
loops, which still require field observations each RK stage. Port those numerical
operations or replace full-field reads with proved accessed regions. Then address
remaining scratch churn, fragmented transfers, boundary launches and startup
costs. Preserve output and complete-process timing. Generic graph-policy migration
and legacy deletion should follow the working candidate, not delay comparisons.

## Reproduction and preserved evidence

Artifact root: `/tmp/fortran-resident-rk-20261109-01`.

- Sources: `time-entries-07/time-entries.json`, `integrated-07/time-integration.json`.
- Build: `build-08/{linked,link-plan}.json`, with immutable build-07/06 bases.
- Milestone: `milestone-08/{plan,result,backend-qualification}.json`.
- Reduction: `reduction-milestone-08.log` and the maintained CUDA test.
- Timing: `comparison-08/{plan,result,execution}.json`, every run directory and log.
- Separate diagnosis: `profile-08/channel20.nsys-rep` and CSV summaries.

The preparation/build/milestone/timing scripts are retained in the artifact root.
`summary.json` includes all timing samples, correctness rows and backend evidence;
the previous report and summary are preserved separately. External/mixed source
edits and the dirty local solver checkout were preserved. No commit or push is
claimed for this work.
