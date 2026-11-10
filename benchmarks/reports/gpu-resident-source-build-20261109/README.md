# GPU hot path: reproducible source-build milestone, 2026-11-09

The resident channel implementation now builds from source through
`pipeline.py resident-build`. It regenerates 248 checked CUDA numerical entries,
uses one compiler coherence runtime, and compiles the supplied local
`poisson-solver` implementation out of tree. No historical objects or Fortran
modules enter either executable. Source preparation preserves read-only
replicas and domain-owned workspaces through the existing runtime.

Fresh native and GPU builds pass both primary application checks. The three-round
comparison measures approximately **3.02× / 2.74× native**. This establishes a
working source-build milestone, not a new optimization over the retained GPU
candidate and not the full research/release gate.

## Complete-process comparison

Seconds; medians of three rotated measurements after one excluded warmup per
case/variant. Four threads, affinity `0,2,4,6`, unchanged configurations and all
output; startup, GPU solver plan creation, computation, output and teardown are
included. Builds, tests and profiles were stopped during timing. All 24 timing
processes pass output comparison. All owned groups are reaped. The maximum
external CPU load was 0.232 cores, below the predeclared 0.5-core bound.

| Case | Fresh native | Retained workspace GPU | Fresh source-built GPU | Speedup vs native |
| --- | ---: | ---: | ---: | ---: |
| channel-simple | 12.829688 | 4.523265 | 4.246858 | 3.02× |
| channel | 14.014121 | 5.104008 | 5.120248 | 2.74× |

The candidate wins **3/3 paired comparisons against native in each case**, but
only **1/3 against the retained GPU candidate in each case**. Relative to that
candidate, medians are 6.11% lower in simple109 and 0.32% higher in channel100.
The distributions vary substantially; these samples do not establish an
implementation speed gain over the retained candidate. They preserve its broad
acceleration; the channel median differs by 0.32%.
No samples were retried, discarded or trimmed.

| Case | Variant | All measured samples (s) |
| --- | --- | --- |
| channel-simple | native | 12.107637 / 12.829688 / 14.360678 |
| channel-simple | previous | 4.051083 / 4.807502 / 4.523265 |
| channel-simple | candidate | 4.053322 / 4.246858 / 4.711281 |
| channel | native | 14.014121 / 13.756546 / 14.215735 |
| channel | previous | 5.104008 / 5.450352 / 4.692839 |
| channel | candidate | 5.120248 / 5.308487 / 4.777250 |

## Correctness and reached GPU work

Eight separate diagnostic processes cover native/candidate pairs for two steps
in both cases, natural simple109 and channel100. All 20 allocations, bounds,
halos and outputs pass at `atol=1e-6`, `rtol=1e-5`. Both variants retain the
approved deterministic TOP/BOTTOM pressure reference. Native creates the four
verified CPU FFTW plans; the GPU candidate creates none.

Runtime launch/graph-kernel records establish GPU execution of SGS, Convection,
advection, filtering where enabled, diffusion, velocity boundaries, wall models
and corrections, pressure RHS/velocity/pressure updates, CFL and enabled time
averages. No generated-CPU execution appears for those numerical entries.
The candidate's dynamic linkage resolves its freshly built local GPU Poisson
library; native does not load that library. This is execution evidence, not a
claim of zero transfers or GPU-only startup/output.

## Source build and interface correction

The new command fixes GNU Fortran/DP/OpenMP and the supported channel adapter;
old placement/window switches do not configure it. Source paths, CUDA
architecture and compilers are explicit. The local solver base is
`4a1f1bb7e6f9dab6a2934b62d9b8655cf3a9f99a`; its current edits are snapshotted and
hashed. VkFFT is pinned. The provider and ELMM use GNU Fortran 15, with G++ 14 and
CUDA 13.5 targeting SM86. Compiler-required floating-point flags are enforced.
The build emits native/GPU and separate full20 diagnostic executables, exact
source identities, CMake compile commands and retained attempt logs.

Fresh extraction also accepts nine FD4 diffusion regions formerly rejected by
the older compiler snapshot. Those extra generated entries are not a claim of
new FD4 execution in the primary FD2-diffusion configurations.

A full build exposed an interface inconsistency hidden by reused old objects:
`ScalarBoundaries` arguments had gained `TARGET`, while the callback interface in
`ScalarDiffusion` retained the old declaration. The RK source adapter now prepares
that module consistently, and the dependent modules rebuild normally. The
failed attempt is preserved; successful compilation and all application checks
follow the source fix. Resuming a build checks dependency snapshots and retains
old logs; it does not import another experiment's objects.

## Evidence and remaining work

All samples, complete outputs, build manifests, trace records and failure logs
are under `/tmp/fortran-resident-source-build-20261109-02`.
`qualification.json` binds the successful build to the milestone results;
`comparison/plan.json`, `result.json` and `execution.json` retain the timing
protocol, samples and process cleanup. `summary.json` includes identities and
per-field qualification summaries.

This is not a release qualification. Natural channel2458 remains unresolved
from the earlier GPU-Poisson trajectory comparison. The maintained full suites
and fresh four-control/seven-round protocol remain open. The 20× objective is
not achieved; neither workload nor timing definition has changed.

Next implementation work should use this source-build path and profile its
critical path before changing kernels or launch scheduling. The complete
numerical GPU path is now reachable and reproducible; prioritize the measured
SGS/advection/filter costs, startup and output rather than another transfer-only
policy experiment. Retain the rejected filter experiment as negative evidence.
Generic graph scheduling and deletion of remaining obsolete public policies
still need migration work; the new application path does not claim those tasks
complete.
