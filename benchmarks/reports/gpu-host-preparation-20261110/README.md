# Host coordinate validation and measured idle time — 2026-11-10

The candidate passes its declared performance and correctness milestones and becomes the selected comparison baseline. The 20× research objective and full release qualification remain open. [Complete evidence, samples and receipts](summary.json) include both timing cohorts and the rejected prototype.

## Complete-process comparison

Each cohort uses one excluded warmup and seven rotated measurements for native, the selected exact-division GPU control and the candidate, on both primary cases. Native retains four threads; CPU affinity is 0,2,4,6. Startup, all outputs and process completion remain included. Each GPU variant/case has a fresh explicit solver-cache directory populated by its excluded warmup. No builds, tests or profiles ran during timing.

The first 48-process cohort passes all output checks and the numerical median thresholds, but one sample's accounted external CPU load is 0.547647 cores against the declared 0.5 limit. It is invalid for promotion and retained in full. No persistent unrelated busy process was found. After three five-second quiet probes measured 0.048/0.034/0.120 cores, one fresh complete cohort was run. Samples are never pooled across cohorts; no individual measurement is discarded or repeated. The experiment was limited to one whole-cohort retry.

Fresh-cohort results:

| Case | Native (s) | Selected GPU (s) | Candidate (s) | Native speedup | Median reduction vs selected GPU | Paired wins |
|---|---:|---:|---:|---:|---:|---:|
| channel-simple | 11.612853 | 3.080063 | 3.033684 | 3.83× | 1.51% | 5/7 |
| channel | 13.133487 | 3.620017 | 3.523741 | 3.73× | 2.66% | 6/7 |

The predeclared gate is at least 1% lower median in both cases, plus correct outputs and the external-load gate. Robust improvement additionally requires all seven paired wins in both cases. The fresh cohort's validity is **True**, median gate **True**, and all-pairs condition **False**. Its maximum accounted external CPU load is 0.306 cores. This is a focused three-variant development comparison, not a new be58/7a release campaign.

## Implementation and verification

CPU sampling identified the generic `offload::index(long double,bool&)` section-coordinate validator as a significant host preparation cost. It now checks finite range before converting to an integer and checks integrality with a round trip, avoiding a separate truncation. A guard excludes the rounded-up upper endpoint on targets whose long-double precision cannot represent LLONG_MAX. The original rejection order is retained to avoid additional floating exceptions for NaNs and just-out-of-range fractions.

Both optimized and sanitized checks pass 640,168 comparisons under all four rounding modes, with identical validity, results and exception flags on the current GNU/x86 target. The first prototype failed those flag checks and remains recorded. The normal isolated helper median is about 12% lower; that alone is not application speed evidence.

All 130 numerical CUDA objects were recompiled with the updated common header. Their numerical kernel source and Fortran interfaces are byte-identical to the preceding candidate; other application/runtime/provider objects are retained exactly. This is not another full public source regeneration. The collective preparation-cost identity now includes the offload header and rejects stale evidence; automatic-placement calibration remains open.

The focused milestone passes 162 tests. Both short cases and simple109/channel100 pass all 20 fields, bounds and original outputs against freshly executed native controls at `atol=1e-6, rtol=1e-5`. All 20 fields match the preceding GPU baseline exactly. Short ledgers retain 31/28 required GPU procedures, no generated CPU numerical events and zero CPU FFTW plans. The native binary comes from the preceding full public build with matching sources; it was freshly executed for these comparisons.

A fresh natural channel2458 run passes the approved independent 1% relative L2 criterion across 20 fields and 114 outputs. Worst L2 is 0.154896% (frame-y-49.vtk:p); largest local error is 1.467061% of reference peak (fields-Pr.bin). All 40/40 field/bounds files match the preceding GPU trajectory. Zero references require max abs<=1e-6; exact structures and original time checks remain required. The earlier approved native reference was reused with verified identities. Historical pointwise failures remain failures.

## What the diagnosis ruled out

The existing kernels already use non-aliasing pointer guarantees. An affine address-sharing prototype reduces 31 SGS address expressions to four bases and 114 CD4 expressions to five bases, lowering register use. All 28 diagnostic cases are bit-exact, including odd subdomains and mixed masks, but summed full-domain medians improve only 0.295%/0.229%, failing its 5% gate. It was not integrated.

The selected baseline's separate warm-cache profiles contain 0.801/0.782 seconds of GPU idle time inside the kernel window. Of that, 0.687/0.643 seconds precede entry to the next observed launch/copy API. Only 0.0021/0.0027 seconds lie between nodes of one graph submission. These are disjoint device-idle intervals; overlapping host API durations are not added to device work. Before-API delay does not by itself prove exclusive CPU computation.

A separate CPU-sampling run passes both primary output checks. On the thread with the most CUDA API calls, the index helper accounts for 174/1166 and 124/1693 samples during GPU-idle intervals. This thread need not be the initial process thread. Allocation/coherence work and activity in the OpenMP runtime are also prominent. The samples alone do not identify the cause of that runtime activity. Sample proportions are not wall-time attribution; stack unwinding yielded only exclusive instruction pointers.

The new candidate's separate warm-cache profiles show kernel unions of 1.994/2.527 seconds and GPU idle within the kernel window of 0.672/1.027 seconds. The fresh native medians give entire 20× budgets of 0.581/0.657 seconds. Channel idle time is higher than in the earlier profile, so these traces do not establish a consistent idle-time reduction. They are not a second timing cohort or an additive wall-time decomposition.

The next milestone should complete public regeneration and current calibration, audit remaining generic policy/legacy cleanup, and run maintained release qualification. Subsequent performance work should address measured host preparation and team coordination rather than revisit the rejected address-only or block-size-only variants. The 20× target remains a research objective; the last measured kernel-only budget already exceeds the entire target time.
