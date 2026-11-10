# Full application regeneration and private-scalar fusion — 2026-11-10

The fresh candidate passes primary correctness and the user-approved natural channel2458 criterion. It does **not** pass the declared performance promotion gate. The selected cached GPU baseline remains unchanged.

## Complete-process comparison

One excluded warmup and seven rotated measurements per variant/case, four native threads on CPUs 0,2,4,6, all original output frames and process completion included. Each GPU variant/case has its own explicit solver cache, populated by its excluded warmup. Cache loading stays inside the measured process. All 48 processes, all output comparisons and the external-load gate pass. No measurements were discarded or retried. Builds, tests and profiling were stopped during timing.

| Case | Fresh native (s) | Selected cached GPU (s) | Candidate (s) | Candidate/native speedup | Median reduction vs cached GPU | Paired wins vs cached GPU |
|---|---:|---:|---:|---:|---:|---:|
| channel-simple | 11.814167 | 3.140862 | 3.137491 | 3.77× | 0.11% | 5/7 |
| channel | 13.115184 | 3.775469 | 3.738645 | 3.51× | 0.98% | 5/7 |

The predeclared gate requires at least 1% lower medians in both cases and all seven paired wins against the cached GPU control. Neither condition passes. The candidate wins all seven comparisons against native in both cases, but its incremental gain over the selected GPU implementation is small. [All distributions, paired ratios, gates and SHA-bound receipts](summary.json) are retained. This is a focused three-variant comparison; historical be58/7a results remain intact, and no fresh four-control release campaign is claimed.

## Implementation and executed coverage

The public build regenerated the entire application and local Poisson provider, including a fresh native control. Generic grouping reduced numerical interfaces from 248 to 130, preserving the original source nests. It compiled 430 translation units through 734 Ninja steps. Then three CD4 CUDA entries/interfaces and their advection coordinator were recompiled against that fresh build to enable private-scalar fusion; all other fresh objects and the local provider were retained exactly. An immutable compiler snapshot binds those entries, and its fusion AST matches the production implementation.

The existing analysis already proves that a private scalar has a definition before every per-iteration read and no escaping value. The fusion pass now uses that proof when successive source passes reuse a temporary name. The combined body still passes the ordinary dependency analysis. Live-out values, carried scalars and cross-point array conflicts remain barriers. No numerical expressions or tolerances were changed.

Runtime short-case ledgers still show 31/28 required GPU procedures and no generated CPU numerical execution. Periodic launches fall by 75% in both cases. In channel, each CD4 velocity routine falls from 18 launches to six; the three directional passes now share a kernel. All four application qualification cases pass against freshly executed native controls, with all 20 fields exactly equal to the preceding GPU control. The GPU runs create no CPU FFTW plans.

The focused compiler/trace milestone passes 123 checks. A broader compiler sweep was deliberately interrupted after 878 passes and three failures to proceed with the requested comparison. Those failures exposed a trace reader that still expected copy-model V2 while the runtime emits V3; the reader and fixtures now require V3, with a clear error for older traces. All three real-runtime cases subsequently pass. The broad 12,508-case selection remains incomplete; this is **not an all-tests-passing release**. Initial collection/import failure, the interrupted run, the first link failure from stale coordinator entry identities, and corrected test-assertion failures remain recorded.

## Approved natural channel2458

The acceptance rule and input identities were recorded before a fresh candidate execution. All 2,458 natural steps, 20 fields and 114 outputs pass the approved independent per-field/output-component 1% relative L2 rule. Bounds, layouts, coordinates, labels and output/frame inventories remain exact; time columns retain the original pointwise tolerance. Primary/operator tolerances are unchanged. The earlier four-thread native reference was reused with verified identities; no new native2458 process is claimed.

Worst relative L2 is **0.154896%** (frame-y-49.vtk:p). Largest local error is **1.467061% of reference peak** (fields-Pr.bin), retained explicitly. All 40 field/bounds files are byte-identical to the previous approved GPU trajectory. Historical strict pointwise failures remain failures.

## Remaining measured budget

Separate warm-cache Nsight profiles show 2.059/2.695 seconds of kernel activity and 0.028/0.032 seconds of device-copy activity. GPU idle intervals inside the kernel window total 0.669/0.798 seconds. These diagnostic categories are not an additive wall-time decomposition; CPU API calls overlap device work.

CD4 device time remains about 0.194 seconds per velocity component despite the lower launch count. Fused kernels use 48–54 registers with no recorded local-memory spill. SGS remains approximately 0.536/0.487 seconds; channel filtering takes 0.404 seconds and FFT kernels approximately 0.263/0.239 seconds. Wrapper reduction has not substantially changed those costs.

Fresh native medians put the 20× whole-process budgets at 0.591/0.656 seconds. Current profiled kernel durations alone exceed them by 3.49×/4.11×. Holding those kernel durations fixed and hypothetically eliminating every other cost gives ceilings of only 5.74×/4.87× native. This is a conditional bound on the current implementation, not a hardware limit or forecast.

The next performance work should target measured SGS, filtering, CD4 and wall-kernel throughput: inspect address generation and repeated loads, introduce proven invariant reuse or tiling where useful, and retain changes only after operator/application milestones and complete-process comparisons. Graph scheduling, residual idle time and remaining policy cleanup stay open. Further wrapper reductions alone are insufficient for the measured 20× budget.
