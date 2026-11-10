# Joined native read admission: ELMM comparison, 2026-11-09

The broader GPU candidate remains slower than be58. Exact native entry-read admission reduces actual ELMM uploads, but its private application slice improves median wall time by only 4.65% on simple109 and 0.34% on channel100. It loses every measured pair against be58. This is not a substantial acceleration or a production promotion.

| Case | Native | Previous candidate | Experimental fix | be58 | Fix vs previous | Fix vs be58 | Native speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| simple109 | 12.951798 s | 12.433427 s | 11.854679 s | 10.726356 s | -4.65% | +10.52% | 1.093× |
| channel100 | 14.809714 s | 19.066973 s | 19.001446 s | 13.092735 s | -0.34% | +45.13% | 0.779× |

One excluded warmup and three rotated measurements per case/mode: 32 complete processes. These are fresh samples of frozen control binaries, not fresh control builds. Four threads on CPUs 0,2,4,6; RTX 3060 Ti; CPU PoisFFT; approved deterministic TOP pressure reference; unchanged physics, output and tolerances (`atol=1e-6`, `rtol=1e-5`). Startup and teardown remain included. All output comparisons and four FFTW-plan checks pass. No build, test or profile overlapped timing. Maximum observed external CPU activity was 0.463924 core, below the predeclared whole-pilot rejection limit of 0.5. The controller finished in 562.32 seconds and reaped every owned process group.

The fix wins 3/3 paired comparisons against the previous candidate in each case, but only by 0.34–0.66% per channel pair. It wins 3/3 against native in simple and 0/3 in channel; against be58 it wins 0/3 in both. This pilot cannot qualify seven-round research gates, and it contains no new 7a612e4 control. Neither case reaches 2× native. Improvements over a regressed candidate do not establish progress beyond the fastest existing implementation.

## Samples and paired results

All measured samples, in round order. Excluded warmups remain in the raw result; no outliers were discarded.

| Case | Mode | Measured seconds |
| --- | --- | --- |
| channel-simple | native | 12.934075, 13.177254, 12.951798 |
| channel-simple | original | 12.509492, 12.433427, 12.197346 |
| channel-simple | candidate | 11.979261, 11.854679, 11.780899 |
| channel-simple | be58 | 10.669414, 11.311046, 10.726356 |
| channel | native | 14.803497, 14.809714, 14.928644 |
| channel | original | 19.214784, 19.010695, 19.066973 |
| channel | candidate | 19.088433, 18.918804, 19.001446 |
| channel | be58 | 12.674867, 13.100978, 13.092735 |

| Case | Control | Fix/control paired ratios | Wins |
| --- | --- | --- | ---: |
| channel-simple | be58 | 1.122767, 1.048062, 1.098313 | 0/3 |
| channel-simple | native | 0.926178, 0.899632, 0.909596 | 3/3 |
| channel-simple | original | 0.957614, 0.953452, 0.965858 | 3/3 |
| channel | be58 | 1.506007, 1.444076, 1.451297 | 0/3 |
| channel | native | 1.289454, 1.277459, 1.272818 | 0/3 |
| channel | original | 0.993424, 0.995166, 0.996563 | 3/3 |

## What changed and what did not

The compiler now issues exact source-bound entry reads for complete original joined OpenMP groups containing unconditional array assignments with independent input/output storage. It reuses the existing source-read admission and coherence machinery. It rejects conditionals, scalar/bound mutation, uncertain operands and shared read/write roots; copied, foreign or unregistered completion receipts are rejected. No runtime ABI, calibration, kernel or numerical-source change is involved.

At RK stages 2 and 3, the Convection initializer previously retired the SGS domain when its original incoming Ustar/Vstar/Wstar values lacked compiler definition credit. Exact admission removes all four such retirements in each initialization-plus-two-step trace and keeps the unchanged U/V/W GPU replicas. The original host buffers remain valid. No definition of untouched regions is inferred.

| Case | Previous H2D bytes | Fix H2D bytes | D2H bytes, unchanged | Total payload reduction | Copy events, previous → fix | Domains, previous → fix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| channel-simple | 134,664,192 | 81,051,648 | 146,316,352 | 19.08% | 148 → 96 | 16 → 12 |
| channel | 184,158,208 | 130,516,992 | 146,316,352 | 16.23% | 730 → 766 | 40 → 36 |

These are initialization-plus-two-step totals, not steady-state per-step traffic. Allocations fall 60→48 in simple and 78→66 in channel. Numerical launch inventories remain exactly equal: 126 and 402 launches respectively. Channel has more upload events despite fewer uploaded bytes. All 146,316,352 downloaded bytes remain in both cases. The experiment does not extend GPU coverage.

The existing-trace attribution confirms no GPU launch after the first diffusion cut in simple or velocity-boundary cut in channel, before the next RK stage. Those CPU consumers still need the produced values. Preserving clean replicas across the initializer cannot eliminate these required downloads. In channel the three CD4div routines each still launch 18 numerical kernels per RK stage. Source-shaped tiling, CPU consumers and transfer/launch costs need to be evaluated as complete groups.

## Qualification and disposition

90 related CPU/source tests pass; focused lint and whitespace checks pass. The compiled before/after regression passes with CUDA enabled and hidden, including zero-trip execution with an invalid unused coefficient index, full-field/halo bitwise comparison, unchanged-input single upload and no download between the two GPU launches. Its initial sandbox CUDA failure and a corrected expected-byte assertion are retained; the final CUDA receipt records reuse of unchanged compiled binaries, not a new passing pytest run.

The actual ELMM initializer passes the original source/strict-consumer proof check. Six separate native/previous/fix short runs pass all 20 fields, bounds, halos and outputs at unchanged tolerances. Timed output comparisons do not replace full20 qualification at natural 109/2,458 steps.

The application binary is a private integration slice: three checked admissions are inserted into the existing emitted initializer, after proving their rectangles exactly match the original prepared reads. Only dynamics.o is rebuilt; all other application/runtime objects are unchanged. This is not a normally regenerated production package. The generic compiler implementation remains unqualified work in progress. Normal regeneration, the full maintained suite and current natural 2,458-step qualification remain required before a release.

Do not spend another full regeneration or seven-round campaign on this fix alone for performance. Keep be58 as the fastest measured reference. The next bounded experiment should compare a complete native Convection call with its complete GPU group, including scratch, host preparation, transfers and launches. Use that result to implement group pricing and coarsen profitable work; do not force unpriced partial coverage. To remove the remaining downloads, a later change must keep the dependent consumer chain on GPU or prove a larger coherent lifetime; deleting required publication is not an optimization. The complete pressure chain remains an open workstream.

## Evidence

- [Protocol and exact input hashes](/tmp/fortran-joined-entry-pilot-20261109-01/plan.json)
- [All timings, warmups and output comparisons](/tmp/fortran-joined-entry-pilot-20261109-01/result.json)
- [Assessment](/tmp/fortran-joined-entry-pilot-20261109-01/assessment.json)
- [Completion and cleanup](/tmp/fortran-joined-entry-pilot-20261109-01/execution.json)
- [All20 short checks and physical-copy ledgers](/tmp/fortran-joined-entry-short-20261109-01/result.json)
- [Remaining domains and source attribution](/tmp/fortran-joined-entry-short-20261109-01/domain-attribution.json)
- [Reproducible trace analysis](/tmp/fortran-joined-entry-short-20261109-01/analyze_domains.py)
- [Original ELMM source consumer](/tmp/fortran-joined-entry-elmm-consumer-20261109-01/result.json)
- [Private build receipt](/tmp/fortran-joined-entry-private-20261109-01/build-result.json)
- [90 related source tests](/tmp/fortran-joined-entry-reads-related-20261109-01.xml)
- [CUDA before/after completion](/tmp/fortran-joined-entry-reads-cuda-20261109-01/outside-sandbox-result.json)
- [Machine-readable summary](summary.json)

Candidate SHA-256: `50a3634b75fc3a176dcca17bae13b0ba8e50f3b064457e19347c1f7c81df1f04`. Runtime identity remains `d0e179c4821005b8fe9d39eab1f378e4ca4b67a5d4da0ed1c9dffd282b978d52`. External edits and historical controls are preserved.
