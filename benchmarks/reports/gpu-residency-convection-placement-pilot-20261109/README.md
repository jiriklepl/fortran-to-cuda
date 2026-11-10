# Complete Convection placement diagnostic — 2026-11-09

Keeping Convection on CPU recovers much of the channel regression: median complete-process time falls 27.47% compared with forced Convection offloading using the same current SGS path. Simple109 improves only 1.66%, with a loss in one paired round. Both remain slower than be58. This diagnostic is not a production placement rule or a performance promotion.

| Case | Native | Current SGS + GPU Convection | Current SGS + CPU Convection | be58 | CPU Convection change | Gap to be58 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simple109 | 13.012988 s | 11.905802 s | 11.707750 s | 10.945415 s | -1.66% | +6.96% |
| channel100 | 14.846632 s | 18.910757 s | 13.715633 s | 12.934806 s | -27.47% | +6.04% |

The experimental binary changes only tsteps.o relative to the preceding joined-entry fix. Before the original Convection call, it executes the parent’s existing checked final-publication sequence, waits for pending work, retires the domain, and calls the normal Convection entry with the original arguments. The SGS path, all other objects, runtime, kernels and calibration are identical. No required publication is deleted and no source work is replayed. Original native work remains compiled in the current instrumented source; this experiment does not replace it with separately optimized native objects.

One excluded warmup and three rotated measurements per case/mode complete 32 processes. Four threads, CPUs 0,2,4,6, RTX 3060 Ti, approved deterministic TOP pressure reference, CPU PoisFFT, unchanged physics/output and `atol=1e-6`, `rtol=1e-5`. All output comparisons and four FFTW-plan checks pass. No build/test/profile overlapped timing. Maximum external CPU activity was 0.420238 core, below the declared 0.5-core rejection limit. Execution finished in 536.33 seconds; every owned process group is absent.

These are fresh samples of frozen controls, not fresh control builds. Native speedups are 1.111× and 1.082×. The CPU-Convection variant wins 2/3 pairs against GPU Convection in simple and 3/3 in channel; it loses all three pairs against be58 in both cases. The full seven-round gates and fresh 7a612e4 comparison are not qualified.

| Case | Mode | All measured seconds, in round order |
| --- | --- | --- |
| channel-simple | native | 13.179207, 13.012988, 12.776353 |
| channel-simple | original | 11.905802, 11.919836, 11.632313 |
| channel-simple | candidate | 11.831584, 11.688812, 11.707750 |
| channel-simple | be58 | 11.197846, 10.739520, 10.945415 |
| channel | native | 14.935082, 14.831100, 14.846632 |
| channel | original | 18.910757, 18.731274, 18.955208 |
| channel | candidate | 13.795481, 13.695487, 13.715633 |
| channel | be58 | 12.937210, 12.667771, 12.934806 |

Here `original` means the preceding GPU-Convection candidate, and `candidate` means the whole-call CPU diagnostic. Warmups and every paired ratio are retained in the raw result and summary.

Six separate native/GPU-Convection/CPU-Convection short runs pass all 20 fields, bounds, halos and outputs. In each case, CPU Convection retains six SGS and 36 viscosity-boundary GPU launches and removes all other numerical GPU launches. Across initialization plus two steps, H2D is 80,461,824 bytes and D2H 27,181,440 bytes: 107,643,264 bytes in 60 copy events, six domains and 24 allocations.

| Case | GPU-Convection bytes | CPU-Convection bytes | Reduction | Numerical launches | Copy events |
| --- | ---: | ---: | ---: | ---: | ---: |
| channel-simple | 227,368,000 | 107,643,264 | 52.66% | 126 → 42 | 96 → 60 |
| channel | 276,833,344 | 107,643,264 | 61.12% | 402 → 42 | 766 → 60 |

These totals include initialization and two steps; they are not per-step or full-timing-run totals. Full20 short correctness does not replace natural 109/2,458-step qualification. The generated-source change is private and all historical controls remain intact.

The controlled placement difference identifies forced partial Convection offloading as a substantial source of the channel regression. It does not establish that Convection should always run on CPU: a larger resident group or coarser GPU implementation could change the result. The remaining 6–7% gap to be58 is outside the added Convection offloading and still needs attribution; do not assign it to kernel or runtime overhead without measurement.

Next, the normal compiler is generating the same SGS→Convection region under its existing public AUTO policy using the frozen current worktree, identical runtime and physical-copy calibration. The profile has no source-task protocol prices, so the policy must retain unsupported work natively rather than assume it is free. No named ELMM placement override, new policy or relaxed proof issuer is added. Generated decisions, actual GPU execution, full-field correctness and wall time must each be checked independently.

- [Pilot plan](/tmp/fortran-convection-native-pilot-20261109-01/plan.json)
- [All timings and output checks](/tmp/fortran-convection-native-pilot-20261109-01/result.json)
- [Assessment](/tmp/fortran-convection-native-pilot-20261109-01/assessment.json)
- [Terminal process receipt](/tmp/fortran-convection-native-pilot-20261109-01/execution.json)
- [Short full20 and physical-copy ledgers](/tmp/fortran-convection-native-short-20261109-01/result.json)
- [Exact private source replacement](/tmp/fortran-convection-native-private-20261109-01/replacement.f90)
- [Private build inputs and commands](/tmp/fortran-convection-native-private-20261109-01/plan.json)
- [Private build completion](/tmp/fortran-convection-native-private-20261109-01/build-result.json)
- [Public AUTO generation plan](/tmp/fortran-auto-placement-20261109-01/plan.json)
- [Machine-readable summary](summary.json)

Experimental binary SHA-256: `1239a241774f929dcaca49e44cdc1f9e82a5e8ffd7fbd7ef3d2fc78cd61ccccd`. Runtime identity remains `d0e179c4821005b8fe9d39eab1f378e4ca4b67a5d4da0ed1c9dffd282b978d52`.
