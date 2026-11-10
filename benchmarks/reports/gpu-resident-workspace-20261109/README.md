# Resident workspace milestone, 2026-11-09

Filter snapshots now use domain-owned reusable storage. RK coefficient inputs
have explicit caller-owned SAVE TARGET storage. Their source-call boundaries no
longer retire these buffers and wait for the device. The across-timestep GPU
numerical path and local `poisson-solver` backend are retained.

| Case | Native, four threads | be58 | Previous GPU-wall build | Workspace candidate | Native speedup | Lower time than previous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simple109 | 11.643 s | 9.318 s | 4.669 s | 4.074 s | 2.86× | 12.7% |
| channel100 | 12.966 s | 11.087 s | 6.128 s | 4.670 s | 2.78× | 23.8% |

This complete-process scout has one excluded warmup and one measured sample per
case/variant, 16 processes in rotated order on CPUs 0,2,4,6. All output, startup,
solver plan creation and teardown are included. No persistent GPU solver cache
is enabled. No build, test or profile ran alongside timing. Every sample is
retained; no retries or trimming occurred. External CPU activity stayed below
the predeclared 0.5-core threshold, and all owned processes were reaped.

Native/be58 and the previous GPU-wall binaries were freshly measured, not rebuilt.
These are single paired observations, **not the formal seven-round research gate**.
The previous binary is the preserved [wall milestone](../gpu-resident-walls-20261109/README.md).

The generic Fortran domain now owns rank-three REAL64 workspace slots identified
by compiler site and exact bounds/shape. A source invocation borrows a slot;
release revokes the borrow without freeing storage or waiting. Reacquisition
submits earlier users and discards their dead values through the existing
coherence runtime. Subsequent writes use the same ordered stream. Slots survive
epoch boundaries; close drains users before freeing their host backing storage.
Different shapes/sites cannot alias, and overlapping borrows are rejected.
Replica validity, pending work, eviction and device accounting remain in the
existing coherence engine. No second memory-management implementation was added.

The ELMM adapter binds filter snapshots as uninitialized private work, retaining
the original complete snapshot and three directional numerical kernels. Native
execution still allocates and deallocates a temporary per invocation. RK beta/rho
are three-element coefficient arrays, not field-sized work buffers. Explicit
SAVE TARGET actuals replace potentially temporary PARAMETER-expression actuals;
their initialization expressions and numerical use are unchanged. The two
Convection-return retire calls are removed only with that owner-lifetime proof.

Read-only observations continue to preserve device replicas. The optimized path
includes SGS, advection, diffusion, filtering, velocity boundaries, wall stress
and correction, the full pressure sequence, CFL arithmetic and reached time
averages. Small host observations/control, initialization and output remain.
Public production build wiring, general graph migration and legacy removal are
unfinished; this is an integrated source-adapter candidate, not completion of
those architecture requirements. All external and mixed edits were preserved.

## Milestone checks

The focused GPU runtime test passes queued producer/consumer ordering across 40
slot reuses, nonzero bounds, distinct shapes/sites, epoch retention, overlapping
borrow rejection and teardown. It observes two device allocations, one initial
upload, no intermediate waits/downloads, and only the two expected result
downloads across publication and close. Dead private snapshots are not downloaded.

Both two-step configurations pass all 20 fields, bounds, halos and outputs.
Natural simple109 and channel100 pass the same checks against fresh native runs
with four threads, the approved deterministic pressure reference, unchanged
physics and output, and `atol=1e-6`, `rtol=1e-5`. Maximum absolute field errors are
`5.67542901e-07` and `1.44559375e-06`, respectively;
acceptance uses the combined absolute/relative criterion. Runtime implementation
identities confirm actual GPU execution in the managed stages and zero generated
CPU events. Candidates create zero CPU FFTW plans; native controls retain four.

Tests were concentrated at this integration milestone. The full maintained
suites and natural2458 were not rerun. The prior long GPU-Poisson trajectory
mismatch remains unresolved; these primary-horizon passes do not waive it.

## Separate 20-step channel profile

| CUDA activity | Previous wall build | Workspace candidate |
| --- | ---: | ---: |
| Stream synchronizations | 694 | 56 |
| Pool allocations / async frees | 678 / 678 | 42 / 42 |
| Graph launches | 804 | 625 |
| Kernel interval union | 0.567 s | 0.566 s |
| Device copy duration | 0.024 s | 0.024 s |

The new trace retains 18,227 GPU kernels: the numerical work has not decreased.
Its approximately 48.174 MB H2D and 99.385 MB D2H include initialization and
teardown and are not per-step traffic. Explicit synchronization API spans fall
from 0.417 s to 0.0009 s. Some waiting appears instead inside pageable copy/free
APIs: copy API spans rise from 0.070 s to 0.203 s, and `cudaFree` spans include
0.157 s. API spans overlap device activity and must not be added or interpreted
as exclusive saved time. The isolated scout establishes the application gain.

SGS, filtering and fourth-order advection still dominate measured numerical work.
The next performance slice should reduce filter snapshots/launches across its
directional passes and optimize measured kernels, while production build wiring
replaces the remaining temporary orchestration. Any filter buffer alternation
must preserve cells skipped by its direction-dependent mask and all halos.
Repeated boundary calls need a dependence proof: corners are filled before faces,
so apparent duplication alone does not establish that a call is redundant.
Startup, solver plan creation, host scalar observations and output remain in the
20× budget. No 20× feasibility or formal 2× checkpoint is claimed.

## Evidence

Artifact root: `/tmp/fortran-resident-workspace-20261109-01`.

- `integrated/workspace-integration.json`; `build/{compile-plan,link-plan,linked}.json`.
- `unit-02.log`, `milestone/{result,backend-qualification}.json` and full runtime traces.
- `natural109/result.json`, `channel100-full20/result.json` and all fields/outputs.
- `comparison/{plan,result,execution}.json`, every warmup/measurement and load observation.
- `profile/{channel20.nsys-rep,channel20.sqlite,attribution.json}` and CSV summaries.

Preparation, build, qualification, timing and profile scripts are retained.
`summary.json` embeds the checkpoint evidence. No commit or push is claimed;
the overall implementation goal remains active.
