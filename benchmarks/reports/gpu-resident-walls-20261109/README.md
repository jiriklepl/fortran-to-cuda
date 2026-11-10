# GPU wall-model milestone, 2026-11-09

The primary planar wall stress calculations and FD2 wall diffusion corrections
now execute on GPU inside the existing across-timestep domain. This removes
whole-field host observations from that RK interval. SGS, advection, filtering,
velocity boundaries, pressure projection through the local `poisson-solver`, CFL
arithmetic and reached time averages retain their existing GPU implementations.
Small host wall diagnostics, scalar control, initialization and output remain.

## Complete-process timing scout

| Case | Four-thread native | be58 | Previous build-08 | New wall candidate | Speedup over native | Lower time than be58 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simple109 | 11.897 s | 9.502 s | 6.936 s | 4.614 s | 2.58× | 51.44% |
| channel100 | 13.722 s | 11.199 s | 8.280 s | 6.047 s | 2.27× | 46.00% |

This is one excluded warmup plus one measurement per variant/case: 16 processes
in rotated order on CPUs 0,2,4,6. Startup, solver compilation/plans, all output and
teardown are included. No persistent GPU provider cache is enabled. No builds,
tests or profiles ran alongside timings. All process/output checks passed; all
external CPU observations were below the predeclared 0.5-core threshold. All
samples, including warmups, are preserved without trimming or retries.

The previous binary is the preserved across-timestep candidate from
[the preceding report](../gpu-resident-rk-20261109/README.md), freshly measured in
this scout. Native/be58 controls are existing binaries with fresh measurements,
not rebuilt release controls. These single paired results exceed 2× native but
**do not qualify the formal checkpoint**: seven rotated rounds, fresh release
controls including `7a612e4`, and full correctness qualification remain required.

## Changes

`elmm_pipeline/resident_walls.py` prepares ten ordinary compiler entries: four
wall-stress calculations and six paired diffusion corrections. The original
interpolation and correction statements are retained in normalized array views.
The iterative wall law runs inside each GPU kernel; its convergence and 50-step
limit retain the original order. The pinned GNU Fortran 15 scaled `NORM2`
calculation and the original literal kinds are preserved. No custom CUDA wall
kernel or new coherence engine was introduced.

The adapter validates complete, unique rectangular z planes, flat stationary
walls, uniform spacing, positive molecular viscosity, the supported wall model,
and absence of x/y wall records and thermal/scalar extensions. Unsupported
configurations close the domain before the original host path. The scalarized source
helpers and both coordinators have checked fingerprints: changed physics is rejected for review.
Correction expressions are extracted directly from the prepared original source.

Wall state lives in ordinary domain-owned Fortran planes. Read-only diagnostics
publish the needed small plane values without invalidating device replicas.
All wall records are published before unmanaged output/finalization. GPU flux
writes are unconditional after private scalar calculation, so the existing
coherence engine proves an overwrite without requiring an old flux value.

The pre-SGS wall viscosity write is omitted only in this supported interval:
Sigma SGS immediately overwrites every affected interior cell and consumes no
previous viscosity there. Wall friction state is still updated. Adjacent wall
correction loops are fused per plane while preserving each cell's accumulation
order; validated topology establishes disjoint output cells.

This is still an explicit ELMM source adapter and reproducible integration
candidate. General compiler graph lifetime/placement migration, public build
wiring and repository-wide legacy removal are unfinished. The candidate continues
to omit the old source-owner/placement wrappers. External and mixed edits,
including the dirty solver checkout, were preserved.

## Milestone correctness

- Both configurations pass all 20 fields, bounds, halos and all outputs at two steps.
- Natural simple109 passes all 20 fields, bounds, halos and all outputs against a
  fresh four-thread native run. Maximum absolute field difference: `5.67542901e-07`.
- Channel100 passes the same full20/bounds/halo/output check against a fresh native
  run. Maximum absolute field difference: `1.44559375e-06`.
- All checks retain `atol=1e-6`, `rtol=1e-5`, the approved deterministic pressure
  reference, original physics and output settings. Acceptance uses both absolute
  and relative tolerance, not the maximum absolute difference alone.
- Runtime implementation identities show all ten wall entries on GPU in both
  short runs and zero generated-CPU events in managed entries. Each run has
  twelve launches per stress entry and six per paired-correction entry, including
  graph-node submissions. Candidate CPU FFTW plan count remains zero; native
  controls retain four verified CPU plans.

Testing was concentrated at this milestone. The full maintained suites were not
rerun. Natural2458 was not rerun, and the previously recorded GPU-Poisson
long-trajectory mismatch remains unresolved. Passing primary horizons does not
waive that requirement or relax tolerances.

The failed intermediate build and first execution remain in the artifact root.
The first execution correctly rejected a conservative upward read of conditional
flux output. The final form computes a private scalar and performs one proved
array overwrite. Failed candidates supplied no performance samples.

## Transfer and timeline evidence

Initialization plus two-step field transfers:

| Case | Upload bytes / copies | Download bytes / copies |
| --- | ---: | ---: |
| channel-simple | 60,310,672 / 127 | 115,526,688 / 49 |
| channel | 48,161,648 / 196 | 95,845,632 / 48 |

These include initialization and final publication; they are not per-step
steady-state rates. Scalar CFL records and small diagnostic publication remain.
No zero-transfer claim is made.

A separate 20-step channel Nsight diagnosis enables individual CUDA graph-node
tracing. It records **48.175 MB H2D and 99.385 MB D2H**, including initialization
and teardown; the preceding build-08 diagnosis recorded 118.429/1,388.312 MB.
The current device-copy duration is **0.024 s**, with **0.070 s** in the three
copy APIs. Host API durations overlap device work and must not be added.
These separately traced runs are diagnostic evidence, not untraced speedups.

The current trace records 18,227 GPU kernels on the runtime stream, with a union
of **0.567 s** of active kernel intervals. Selected categories:

| GPU work, 20 steps | Recorded device seconds |
| --- | ---: |
| SGS | 0.102 |
| Filter3ord | 0.084 |
| Fourth-order advection, three components | 0.122 |
| FFT kernels | 0.051 |
| Main FD2 diffusion loops | 0.032 |

CUDA synchronization API spans total 0.417 s and kernel/graph launch APIs 0.095 s;
these overlap GPU work. They identify areas to investigate, not additional
exclusive time. The full category list and interval-union calculation are retained.

At the profiled per-step kernel rate, 100 steps would contain roughly 2.8 s of
GPU kernel work. This is an extrapolation, including profile effects, not a hard
ceiling or forecast. The current 20× native budget is roughly 0.686 s for the
whole channel100 process: removing host traffic alone cannot close that gap.

Next work should reuse compiler-managed temporary storage, remove unnecessary
synchronization and repeated boundary work where dependencies prove it safe, and
optimize the measured SGS/filter/advection kernels. Startup, solver plan creation,
small diagnostic reductions and output also remain in the complete-process budget.
Continue measured comparisons while completing production graph integration and
removing obsolete policy/configuration paths.

## Artifacts

Root: `/tmp/fortran-resident-walls-20261109-01`.

- `entries-04/wall-entries.json`, `integrated-04/wall-integration.json`.
- `build-04/{linked,compile-plan,link-plan}.json`, all compiler/link logs.
- `milestone-04/{result,backend-qualification}.json` and complete traces/fields.
- `natural109/result.json`, `channel100-full20/result.json` and all field/output evidence.
- `helper-contract.json`, plus the pinned Fortran `NORM2` lowering inspection.
- `comparison-04/{plan,result,execution}.json` and every timing sample.
- `profile-04/{channel20.nsys-rep,channel20.sqlite,attribution.json}` and CSV summaries.

Preparation/build/milestone/timing/profile scripts and failed attempts are retained.
`summary.json` embeds results and evidence for this checkpoint. No commit or push
is claimed. The overall implementation goal remains active.
