# Performance history and measured cost breakdown

The selected GPU implementation runs `channel-simple`'s natural 109 steps in **3.030 s (3.78× native)** and shortened `channel`'s 100 steps in **3.492 s (3.70× native)**. These are complete-process medians, including initialization and output, from the latest retained seven-round comparison. Both cases beat their native control in all seven paired rounds. This is a substantial gain; the 20× research target remains unmet.

The code snapshot is **`620e630e21735b8e90b496ed68621c444e26b9aa`**. Its wrap-up regenerated 130 entries and reproduced all four selected ELMM binary identities. The final packaging adjustment accepts the committed Poisson provider without changing its source hashes. **No new timing campaign was run for the commit**: the numbers belong to the previously measured, retained binary, whose SHA-256 is `2d5df06329cb8447990b0d4b39327ddf803f9d3113036ca2db7d1998cb69300e`. The rejected coordinate-decoder experiment is excluded from the selected result. [Selected measurements](../../gpu-kernel-screening-20261110/README.md), [regeneration evidence](../../gpu-current-state-wrap-20261110/README.md).

## Comparable native and major-commit controls

The most recent **common four-control cohort** predates the solver-cache and final kernel/host improvements. It directly compares fresh native, `be58ab8`, `7a612e4`, and the resident implementation. Each cell is median seconds followed by speedup over that cohort's four-thread native control.

| Implementation in the November 10 four-control cohort | simple109 | channel100 |
|---|---:|---:|
| Four-thread native, CPU PoisFFT | 11.673 (1.00×) | 13.190 (1.00×) |
| `be58ab8`: historical SGS offload | 9.397 (1.24×) | 11.289 (1.17×) |
| `7a612e4`: graph redesign, refreshed calibration | 16.685 (0.70×) | 11.310 (1.17×) |
| November 10 resident GPU path and GPU Poisson | 4.032 (2.89×) | 4.673 (2.82×) |

The resident implementation wins all seven pairs against all three controls in both cases. The fresh `7a612e4` build chooses additional Convection offloading for simple109 under its refreshed cost profile; the October observation that it offloaded only SGS does not describe this fresh build. Its regression is retained. [Four-control protocol and results](../../gpu-resident-four-controls-20261110/README.md).

The latest selected state has its own native control; it was **not** rerun against fresh `be58ab8` and `7a612e4` in that final cohort.

| Latest retained cohort | simple109 | channel100 |
|---|---:|---:|
| Native median | 11.460 s | 12.938 s |
| Selected GPU median | **3.030 s** | **3.492 s** |
| GPU min–max, all seven samples | 2.960–3.077 s | 3.473–3.588 s |
| GPU inclusive Q1–Q3 | 3.004–3.049 s | 3.478–3.544 s |
| Native/GPU median speedup | **3.78×** | **3.70×** |
| Native/GPU paired speedup range | 3.73–3.93× | 3.63–3.78× |
| Paired wins over native | 7/7 | 7/7 |
| Complete-process budget for 20× | 0.573 s | 0.647 s |

All these cohorts use one excluded warmup and seven rotated complete-process measurements per variant and case. November results use four native threads and CPU affinity `0,2,4,6` on the Ryzen 5 5600G / RTX 3060 Ti setup. GPU solver-cache measurements use a warm persistent **compiled-binary disk cache** after the excluded warmup; each timed process still creates its CUDA context and solver plans. Cold-cache first-process time is not represented by those warm-cache medians. The cache milestone's excluded cold warmups were 4.174/4.903 s. [Cache protocol](../../gpu-local-solver-cache-20261110/README.md).

## Progression, including regressions

Each speedup below uses the **native measurement from that row's own cohort**. Absolute times across rows are historical observations, not paired comparisons. The October cohort used different CPU affinity (`0..11`) and an earlier CUDA runtime. Dated working-tree milestones were accumulated into the latest snapshot; they were not individual source commits.

| Commit or measured milestone | simple109: seconds (native speedup) | channel100: seconds (native speedup) | Interpretation |
|---|---:|---:|---|
| October 22 `be58ab8` control | 9.651 (1.21×) | 11.454 (1.16×) | Original historical comparison |
| October 22 `7a612e4` | 9.499 (1.23×) | 11.354 (1.17×) | SGS-only profiles; modest parity improvement |
| November 8 `2619b7f` | 11.271 (1.04×) | 16.321 (0.81×) | Empty-native-call retention did not deliver both-case improvement |
| November 8 `03c695d` | 11.186 (1.04×) | 16.236 (0.80×) | Compact-copy calibration; regression remained |
| November 10 resident path | 4.032 (2.89×) | 4.673 (2.82×) | Broad GPU coverage and GPU pressure solve |
| November 10 solver binary cache | 3.190 (3.60×) | 3.778 (3.45×) | 21.36% / 19.62% below same-cohort uncached GPU |
| November 10 exact division and grouping | 3.096 (3.71×) | 3.641 (3.60×) | 2.91% / 4.33% below same-cohort previous GPU |
| November 10 host preparation | 3.034 (3.83×) | 3.524 (3.73×) | 1.51% / 2.66%; wins 5/7 and 6/7, not a robust paired win |
| November 10 asynchronous output | 3.004 (3.83×) | 3.529 (3.73×) | 0.55% / 0.15%; wins 3/7 and 4/7, not a robust gain |
| Latest retained binary, snapshot `620e630` | **3.030 (3.78×)** | **3.492 (3.70×)** | Later remeasurement of selected implementation |

`4d944f1` fixed calibration-cache ownership between the failed compact-copy work and the later resident implementation; there is no separately qualified full timing row for that commit. The primary 109/100-step cases produce **no post-initial asynchronous frames**, so the asynchronous-output implementation cannot explain their small observed incremental difference. The first host-preparation campaign was invalidated by external CPU activity; only its complete valid replacement cohort is included.

Sources: [October baseline](../../graph-redesign-parity-20261022/README.md), [2619b7f](../../gpu-residency-speedup-20261108/README.md), [03c695d](../../gpu-residency-copy-v3-speedup-20261108/README.md), [exact division](../../gpu-exact-division-20261110/README.md), [host preparation](../../gpu-host-preparation-20261110/README.md), [asynchronous output](../../gpu-async-output-20261110/README.md). Every measured sample, native baseline and paired ratio is retained in [data.json](data.json); the flat [timing CSV](timing-samples.csv) contains 420 measurements across nine cohorts, including unselected controls/experiments with explicit mode names.

## Physical transfers and GPU computation

These are **separate traced full-run diagnostics**, not the untraced wall-time samples above. MB means decimal megabytes. Counts and bytes include initialization, scalar results, publication and final output; they are not per-step values. CUDA memory kinds 1/2 are H2D/D2H.

| Traced implementation / case | H2D MB (copies) | D2H MB (copies) | GPU copy interval union | GPU kernel interval union | Kernel launches |
|---|---:|---:|---:|---:|---:|
| `be58ab8`, simple109 | 4,398.580 (2,952) | 1,375.732 (328) | 0.642 s | 0.631 s | 328 |
| Selected GPU, simple109 | 60.323 (135) | 115.528 (157) | 0.025 s | 1.989 s | 58,353 |
| `be58ab8`, channel100 | 4,036.502 (2,709) | 1,262.486 (301) | 0.614 s | 0.579 s | 301 |
| Selected GPU, channel100 | 48.174 (204) | 115.115 (441) | 0.026 s | 2.527 s | 70,238 |

The increase in GPU kernel activity accompanies moving substantially more numerical work off the CPU. The large reduction in transfer bytes does not imply that transfer time previously dominated the whole application. For example, `be58`'s host copy-API spans are 0.888/0.849 s, and its device copy intervals are 0.642/0.614 s; those spans overlap and **must not be added**. [Historical profile](../../gpu-residency-recovered-gap-20261109/README.md), [durable historical extract](be58-profile-extract.json), [selected profile](../../gpu-kernel-screening-20261110/summary.json).

The selected GPU has 0.641/0.889 s without recorded GPU kernel/copy activity **inside its first-to-last-kernel window** (2.639/3.426 s). This excludes startup before the first kernel and final activity after the last kernel. It identifies remaining gaps, not a directly recoverable speedup or an exclusive host-runtime charge.

The following decomposition sums kernel activity by procedure family from those same selected traces. It measures device work, not end-to-end contribution. Individual kernel durations can overlap; the total activity sum (1.991/2.528 s) is slightly larger than the kernel interval union above.

| GPU procedure family | simple109 activity | channel100 activity |
|---|---:|---:|
| SGS | 0.485 s | 0.446 s |
| Convection and advection | 0.246 s | 0.583 s |
| Diffusion | 0.166 s | 0.152 s |
| Channel filter | 0.000 s | 0.326 s |
| Wall model stresses and corrections | 0.348 s | 0.318 s |
| Velocity / viscosity boundaries | 0.096 s | 0.121 s |
| Poisson FFT kernels | 0.260 s | 0.239 s |
| Poisson packing, denominator, normalization | 0.059 s | 0.054 s |
| Pressure RHS, halos, velocity / pressure correction and pressure gradient | 0.137 s | 0.126 s |
| Pressure reference | 0.088 s | 0.081 s |
| RK array updates | 0.077 s | 0.070 s |
| Time control and reductions | 0.029 s | 0.012 s |
| **Summed kernel activity** | **1.991 s** | **2.528 s** |

Exact procedure names, launch counts, unrounded times and the grouping map are in [gpu-stage-times.csv](gpu-stage-times.csv) and [gpu-stage-groups.json](gpu-stage-groups.json). For example, `BoundUVW` alone launches 29,448/37,818 kernels, while SGS launches 327/300; launch count and kernel duration expose different optimization opportunities.

## Initialization, host phases and output

The dedicated phase experiment adds a complementary view. Values below are **single untraced diagnostic observations**, not seven-run medians. Instrumented host phases are disjoint, but a host call may enqueue device work whose completion is charged to a later phase.

| Main-thread phase | simple native | simple GPU | channel native | channel GPU |
|---|---:|---:|---:|---:|
| Initial conditions, including initial pressure setup | 0.070 s | 0.639 s | 0.063 s | 0.517 s |
| RK calls | 10.971 s | 2.289 s | 12.748 s | 1.726 s |
| `OutTStep` calls | 0.227 s | 0.004 s | 0.030 s | 1.150 s |
| Resident finalization | 0.000 s | 0.032 s | 0.000 s | 0.021 s |
| Final output | 0.066 s | 0.070 s | 0.057 s | 0.052 s |
| Other recorded setup/control/teardown phases | 0.068 s | 0.122 s | 0.066 s | 0.089 s |

The channel GPU's 1.150 s `OutTStep` value is **not serialization time**: it includes waits for queued work. In the separate traced diagnostic, 0.917 s of a 1.187 s output-call span overlaps GPU activity. Narrowed untraced instrumentation puts GPU adapter creation at approximately 0.323/0.298 s, including solver creation at 0.251/0.227 s; these nested intervals must not be added. Profiling itself increases CUDA initialization cost. [Phase analysis](../../gpu-phase-attribution-20261110/README.md), [unrounded phase CSV](host-phase-times.csv).

There is no comparable exclusive CPU-per-procedure / initialization / output breakdown for **every historical commit**. An additive table claiming one would mix sampling, nested host spans, overlapping CUDA activity and different cohorts. The retained measurements establish the change from SGS-only offload to a broadly resident GPU path, its traffic reduction, and current launch/computation costs without that inference.

## Evidence and limits

The latest selected binary retains strict primary-case checks and the approved natural channel2458 criterion: independent per-field/output relative L2 limit of 1%, with exact layouts and inventories. Its worst retained natural-run L2 error is 0.155%; the stricter natural-run pointwise criterion still fails. These qualifications and focused milestone suites do not imply full release qualification. No new benchmark or application test was performed to assemble this report.

[sources.json](sources.json) records SHA-256 hashes for each source summary and the archived `be58` extract. [build_tables.py](build_tables.py) reproduces the tables' underlying data, checks each seven-sample median and verifies GPU-stage sums. The 20× goal would require a further **5.29×/5.40×** reduction from the selected complete-process medians. Startup, remaining numerical work and launch gaps all remain relevant; transfer elimination alone does not establish that result.
