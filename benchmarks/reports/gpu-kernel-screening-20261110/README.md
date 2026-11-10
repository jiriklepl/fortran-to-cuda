# GPU kernel screening — 2026-11-10

**Neither experiment improved both primary cases. Both production changes were removed.** The selected implementation remains the bounded asynchronous-output build, with the local GPU Poisson provider and resident hot path. All 43 GPU-renderer source identities match that selected build again. No new calibration or full-release claim is made.

An exact coordinate decoder replaced wide division with shifts or checked 32-bit division where possible. All 130 numerical entries were regenerated and recompiled in 277.27 seconds. Fortran interfaces and runtime headers remained identical; application/native/provider objects came from the preceding complete public build.

Eight fresh processes pass the strict 20-field/output checks for simple2, channel2, simple109 and channel100. Candidate fields exactly match the selected GPU build. Natural channel2458 passes the approved independent 1% L2 criterion across 20 fields and 114 outputs; all 40 field/bounds files match the prior GPU trajectory exactly. Worst L2 remains **0.154896%**, while the largest local error remains **1.467061% of reference peak**. Primary tolerances remain unchanged.

The isolated comparison used one excluded warmup and seven rotated complete-process measurements. All 48 processes pass output validation and the external-load gate. Native controls were freshly executed from the preceding complete build.

| Case | Native median | Selected GPU median | Decoder candidate | Selected/native | Candidate paired wins vs selected |
|---|---:|---:|---:|---:|---:|
| simple109 | 11.460198 s | 3.030292 s | 3.020381 s | **3.78×** | 4/7 |
| channel100 | 12.938242 s | 3.492289 s | 3.513704 s | **3.70×** | 3/7 |

The decoder improves simple109 by only **0.33%** and regresses channel100 by **0.61%**. It fails the predeclared requirement of at least 1% lower median in both cases. There are no fresh be58/7a612e4 controls, so this is not the four-control release protocol.

A second, generic experiment stopped empty iterations after a convergence guard became false while preserving the Fortran terminal loop counter. It passed 192 semantic comparisons and exact primary-field checks. Only four wall entries were rebuilt for screening. The first pilot mistakenly created cache directories with mode0755; the provider rejected them and recompiled FFT kernels each run. That entire attempt and its initial decision are preserved. One corrected whole pilot used the same binaries, mode0700, and verified cached cubins on every run.

The corrected three-round pilot improves simple109 by **1.30%** and regresses channel100 by **1.42%**. It fails its predeclared 2% promotion gate. This source change was also removed without an expanded rebuild or long-run campaign.

The coordinate milestone initially had 214 passing checks and one pre-existing whole-package provenance assertion failure, reproduced with the old emitter. The test now compares executable mapping and launch behavior. After restoration, all 150 focused checks pass, including an actual CUDA full-width decoder test. The full maintained suites were not run.

Separate profiles of the **retained executable**, taken after timing, identify the next targets:

| Profile | Kernel launches | BoundUVW launches | GPU idle within kernel window | SGS device activity |
|---|---:|---:|---:|---:|
| simple109 | 58,353 | 29,448 | 0.641 s | 0.485 s |
| channel100 | 70,238 | 37,818 | 0.889 s | 0.446 s |

These are traced diagnostic durations, not additive wall-time components or a speedup forecast. The next substantial compiler experiment should reduce small boundary-kernel fragmentation while preserving face/edge/corner ordering, then measure actual graph-argument changes. The runtime already suppresses node updates for identical argument values; adding a second equality cache would duplicate existing work. SGS, CD4, filter and wall computation remain important.

Full memory accounting, residual legacy/test migration, current FD4/operator qualification and fresh four-control release timing remain open. The 20× objective is unmet: the current native medians imply total budgets of **0.573/0.647 seconds**. No workload or timing definition has changed. All samples, paired ratios, diagnostic profiles, source archives, binary identities and preserved failed attempts are linked in [summary.json](summary.json).
