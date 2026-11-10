# Exact dynamic division: measured GPU improvement — 2026-11-10

The candidate passes its predeclared retention gate and wins all seven paired comparisons in both primary cases. It becomes the selected performance baseline at `/tmp/fortran-exact-divide-20261110-01`. Primary correctness remains at the original pointwise tolerances; a fresh natural channel2458 run passes the explicitly approved componentwise L2 criterion. The 20× research objective and full release qualification remain open.

## Complete-process comparison

One excluded warmup and seven rotated measurements per variant/case; four native threads on CPUs 0,2,4,6; complete startup, every original output frame and process completion included. Each GPU variant/case has a separate fresh explicit solver-cache directory, populated by its excluded warmup. Cache lookup/loading stays inside process timing. All 48 processes and output comparisons pass, with no discarded measurements or retries. Maximum external load is 0.394 CPU cores, below the declared 0.5 limit. No builds, tests or profiles ran concurrently with timing.

| Case | Fresh native (s) | Previous selected GPU (s) | Candidate (s) | Native speedup | Median reduction vs previous GPU | Paired wins vs previous GPU |
|---|---:|---:|---:|---:|---:|---:|
| channel-simple | 11.494660 | 3.188409 | 3.095687 | 3.71× | 2.91% | 7/7 |
| channel | 13.096450 | 3.805290 | 3.640654 | 3.60× | 4.33% | 7/7 |

The gate, recorded before any candidate application measurements, required channel's median to fall at least 3%, with channel-simple no more than 1% slower. Both pass. All seven paired wins against both native and the previous GPU control additionally establish the declared robust-improvement condition. [All samples, paired ratios, gates and SHA-bound evidence](summary.json) are retained. This is a three-variant development comparison, not a fresh be58/7a four-control release campaign. Those historical controls remain intact.

The candidate includes the earlier grouping/private-scalar fusion work as well as exact division. The complete-process difference is their combined effect relative to the selected cached control; it cannot all be attributed to this arithmetic change alone.

## Generic compiler implementation

Dynamic floating division now checks whether the divisor is a normal power of two with a normal exactly representable reciprocal, and the numerator is finite. If so, multiplication by that exact reciprocal has the same real result and single rounding as division, including subnormal results and signed zero under the required build flags. Every other value retains ordinary division. The operands are evaluated once after the usual arithmetic conversions. Integer division, constant divisors and host division retain their previous behavior. No approximate reciprocal, reassociation or relaxed compiler flags are introduced.

The generic IR scan examines all 130 current resident entries and regenerates all 21 containing dynamic floating division. It recompiles their numerical CUDA objects against the preceding fresh public build. Fortran interfaces are byte-identical, so existing coordinators, remaining application objects and the local Poisson provider are retained exactly. An immutable compiler snapshot binds the build. This is not another full public regeneration. There is no ELMM procedure-name specialization in selection or lowering.

The existing numerical-calibration identity already binds both the C-family emitter and numeric runtime header and rejects stale backend evidence. This optimization changes no runtime ABI. Automatic-placement costs have not been recalibrated; these application runs use explicit GPU placement.

## Milestone verification

The source/numerical-contract set passes 170 checks. The CUDA operator check passes 180,224 bitwise comparisons over float/double exponent ranges, signs, mixed integer numerators, ordinary and exceptional operands, signed zero, subnormals and overflow. Both short cases plus simple109/channel100 pass all 20 fields, bounds and original outputs against freshly executed native controls at `atol=1e-6, rtol=1e-5`. Their fields are exactly equal to the preceding GPU result.

Short-case runtime ledgers establish 31/28 required GPU procedures, with no generated CPU numerical events and zero CPU FFTW plans. The local Poisson implementation remains the pressure backend. These focused milestones do not close the incomplete maintained suite recorded in the [previous report](../gpu-full-fusion-20261110/README.md).

The fresh natural channel run completes all 2,458 steps and passes 20 fields and 114 outputs under the approved independent per-field/output-component 1% relative L2 limit. Zero-reference components require maximum absolute error at most 1e-6. Bounds, layouts, coordinates, labels and output/frame inventories remain exact. Time checks and primary/operator checks retain their original tolerances. The immutable earlier four-thread native reference is reused with verified identities; no new native2458 execution is claimed.

Worst relative L2 is **0.154896%** (frame-y-49.vtk:p). Largest local error is **1.467061% of reference peak** (fields-Pr.bin). All 40/40 field/bounds files match the preceding approved GPU trajectory byte for byte. Historical strict pointwise failures remain recorded as failures.

## Kernel evidence and remaining budget

The isolated SGS pilot preserves every output bit and lowers kernel medians by about 9% on both saved primary-case states. A unit-divisor filter pilot lowers primary all-fluid kernel medians by 21–23%; its masked cases are 4–7% slower. The latter uses a narrower unit-only guard than production. These are diagnostic observations, not whole-application speed claims or a guarantee of improvement on arbitrary inputs.

Separate warm-cache full-primary profiles compare this candidate with the preceding full-build/private-fusion candidate:

| Case | Procedure | Previous profile (s) | Current profile (s) | Reduction |
|---|---|---:|---:|---:|
| channel-simple | subgrid::sgs_sigma_uniform | 0.536449 | 0.489676 | 8.72% |
| channel | subgrid::sgs_sigma_uniform | 0.487200 | 0.443596 | 8.95% |
| channel | filters::filter3ord | 0.403723 | 0.325450 | 19.39% |
| channel | momentumadvection::cd4divu | 0.194383 | 0.177075 | 8.90% |
| channel | momentumadvection::cd4divv | 0.194166 | 0.177229 | 8.72% |
| channel | momentumadvection::cd4divw | 0.192520 | 0.178035 | 7.52% |

Current kernel unions are 2.007/2.517 seconds; device-copy unions are 0.025/0.024 seconds. GPU idle intervals inside the kernel window total 0.801/0.782 seconds. CPU API spans overlap device work and are not added to these durations.

Fresh native medians put the 20× entire-process budgets at 0.575/0.655 seconds. Holding profiled kernels fixed and eliminating all other costs would cap speedup at 5.73×/5.20×. This conditional bound describes the current implementation, not a hardware ceiling. Kernel throughput and measured idle time still need substantial work.

Next work should target the remaining SGS/CD4/filter/wall costs through checked load/address reuse and profitable fusion or tiling, plus runtime submission gaps. Keep the new selected baseline for comparisons. Full public regeneration of the new lowering, current automatic-placement calibration, remaining generic policy cleanup and the maintained release suites remain open.

The first filter pilot's illegal-access failure (incorrect harness axis extents), its corrected separate run, and the first entry-generation failure (missing original collective context) are preserved. The corrected candidate retains the original collective context and exact Fortran interfaces. No failed evidence was overwritten or relabeled as a pass.
