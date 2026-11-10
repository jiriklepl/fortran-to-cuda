# Boundary grouping and CUDA packaging screening — 2026-11-10

**The selected GPU application is preserved.** Neither boundary fusion nor CUDA translation-unit aggregation improved both primary cases. The retained compiler change removes empty coherence transactions around private integer/logical setup. It passes 63 milestone checks, including eight new CPU/GPU cases. All 130 ELMM CUDA entries regenerate byte for byte; all four executables and the Poisson provider match the selected build exactly. This change therefore claims no new ELMM speedup.

The selected seven-round comparison remains **3.78× native for simple109 and 3.70× for channel100**, at 3.030292/3.492289 seconds. Those measurements and the approved long-run qualification are in the [preceding report](../gpu-kernel-screening-20261110/README.md). No complete release or 20× claim is made.

Each experiment below used exact primary 20-field comparison first, one excluded warmup and three rotated complete-process measurements against the selected GPU build. Original outputs pass strict native comparison, cached FFT images are verified, and every run passes the predeclared external-load limit. These are screening pilots without native timing. Promotion required at least 2% lower median in **both** cases.

| Experiment | simple109 selected → candidate | Median improvement | channel100 selected → candidate | Median improvement |
|---|---:|---:|---:|---:|
| Boundary union | 2.987292 → 3.323049 s | -11.24% | 3.510433 → 3.925199 s | -11.82% |
| Union + local setup fix | 2.969322 → 2.981556 s | -0.41% | 3.566629 → 3.552425 s | +0.40% |
| One CUDA translation unit | 3.064827 → 3.015075 s | +1.62% | 3.494197 → 3.561998 s | -1.94% |

The generic fusion experiment joined disjoint rank-one/two iteration domains while preserving original bounds, empty-domain guards and dependency verification. Six reached boundary entries changed from three regions to one. Diagnostic link aliases accommodated changed compiler-issued symbol identities; the initial failed exact-interface assertion is preserved. These were not complete production builds. Fusion has been removed, restoring the exact pre-experiment transform.

Separate profiles confirm that boundary launches fell from **29,448 to 9,816** in simple109 and **37,818 to 12,606** in channel100. Despite that reduction, the first pilot regressed by more than 11%. The extra bound snapshots opened empty host access transactions. Omitting those transactions removed the large regression, but the combined change still failed the promotion gate. Traced activity is diagnostic and is not added to untraced wall time.

The retained fix applies only to private INTEGER/LOGICAL assignments with no managed-array access or floating arithmetic, after the array-bearing entry has validated its context. Observable scalar writes, floating effects and array access retain the existing transaction path; execution tracking remains intact. The focused milestone checks device/host execution, intermediate data preservation and zero-trip cases. Full maintained suites remain pending. The scoped emitter fingerprint changed, so no unchanged numerical-cost calibration is asserted.

The packaging experiment compiled all 130 byte-identical numerical CUDA sources into one translation unit, using the same arithmetic flags, application objects and local provider. It compiled in 305.12 seconds and reduced the executable from 21,796,872 to 20,144,472 bytes. Exact primary fields pass, but channel100 regresses. No production packaging change is retained.

The selected seven-round logs also expose a necessary constraint on the 20× objective:

| Case | Median wall time outside reported step timers | Entire-process 20× budget |
|---|---:|---:|
| simple109 | 0.735 s | 0.573 s |
| channel100 | 0.693 s | 0.647 s |

This subtraction is **not an exclusive startup measurement or a hardware ceiling**. The application prints the sum of host step timers at 0.01-second resolution. The residual includes initialization, final output/teardown, untimed loop control and possibly deferred GPU work. Holding that residual fixed would prevent 20× even if the reported step time vanished. Finer phase attribution is required before changing startup or output behavior.

Next work should attribute those phases and measured SGS/CD4/filter or graph-preparation costs, rather than infer payoff from launch count alone. Full graph/queue/driver and host-memory budgeting, residual legacy migration, current FD4/operator qualification, maintained release suites and the fresh four-control comparison remain open. The approved natural-channel rule remains independent 1% relative L2 per field/output component; primary/operator tolerances and all output inventory/layout/time checks stay strict.

Source archives, failed attempts, binary identities, complete samples, paired wins and verified evidence are recorded in [summary.json](summary.json).
