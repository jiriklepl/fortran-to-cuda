# Resident GPU path: fresh four-control comparison, 2026-11-10

The complete supported numerical GPU path is generated and built from current source through `pipeline.py resident-build`, including the supplied local Poisson implementation. This campaign compares it with fresh four-thread native, be58 and pinned 7a612e4 builds.

| Case | Native | be58 | 7a612e4 | Resident GPU | Speedup vs native | Lower time vs 7a |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simple109 | 11.672559 s | 9.397384 s | 16.684979 s | 4.032128 s | 2.89× | 75.83% |
| channel100 | 13.190278 s | 11.289168 s | 11.309814 s | 4.673073 s | 2.82× | 58.68% |

Comparison valid under the predeclared external-CPU bound: **True**. Candidate paired wins: `{"channel": {"baseline-7a612": 7, "be58": 7, "native": 7}, "channel-simple": {"baseline-7a612": 7, "be58": 7, "native": 7}}`.

One excluded warmup and seven measured rounds per mode and case, rotated in the same declared order. All 64 processes retain startup, solver planning, computation, output and teardown. All output comparisons pass. No build, test or profile runs alongside timing; no sample is discarded, retried or trimmed. Maximum measured external CPU is 0.302 cores (limit 0.5). All owned process groups are reaped.

## All measured samples

| Case | Mode | Seven complete-process times, seconds |
| --- | --- | --- |
| channel-simple | native | 11.520945 / 11.490644 / 11.672559 / 11.853239 / 11.807071 / 12.364729 / 11.538128 |
| channel-simple | be58 | 9.388982 / 9.591768 / 9.485463 / 9.397384 / 9.254509 / 9.365401 / 9.988301 |
| channel-simple | baseline-7a612 | 16.684979 / 16.422605 / 16.766044 / 16.388899 / 16.782185 / 16.584300 / 16.764413 |
| channel-simple | candidate | 4.038811 / 4.082831 / 4.032128 / 3.957383 / 4.000090 / 4.152691 / 4.001546 |
| channel | native | 13.020375 / 13.190278 / 13.333535 / 13.195987 / 13.226637 / 12.894903 / 12.981944 |
| channel | be58 | 11.296541 / 11.755548 / 11.367539 / 11.228538 / 11.207898 / 11.215342 / 11.289168 |
| channel | baseline-7a612 | 11.034455 / 11.285284 / 11.309814 / 11.212442 / 11.488938 / 11.621454 / 11.337556 |
| channel | candidate | 4.772202 / 4.718647 / 4.699788 / 4.655319 / 4.673073 / 4.607535 / 4.635772 |

## Implementation and qualification

All 248 numerical entries and their manifests were regenerated, and all 1,083 build steps completed in a fresh directory. No historical object or Fortran module enters the candidate or native builds. Generated CUDA and the shared runtime retain strict arithmetic flags. The separately declared local solver retains its existing CMake arithmetic configuration, including default CUDA FMA; it is not claimed to share the compiler’s `--fmad=false` setting. GPU executables resolve the freshly built local provider; native executables do not load it.

The runtime execution ledger establishes the reached numerical GPU stages (31 procedure identities in simple and 28 in channel), including pressure preparation, the local solver, halo handling and corrections. It reports no generated CPU numerical events in this slice. This does not claim initialization, scalar control or output are GPU-only.

All four variants pass all 20 fields, bounds, halos and complete outputs for both two-step cases, natural simple109 and channel100 at unchanged `atol=1e-6`, `rtol=1e-5`. The approved deterministic pressure reference and four native threads are retained. These are milestone checks; the complete maintained test suites and natural channel2458 are not newly qualified. The earlier long GPU-Poisson trajectory discrepancy remains unresolved.

## Historical control compatibility

The first fresh 7a build used its prior calibration, which rejected the system update from G++ 14.4.0 to 14.4.1 and ran on CPU. Its numerical results and failed backend check are retained; it contributes no timing sample. The pinned runtime was freshly calibrated, and the unchanged pinned compiler regenerated entries using the same AUTO/graph/DIRECT policy before fresh compilation.

The new GCC banner ends in a snapshot date. The pinned parser cannot parse that suffix. A documented derived profile removes only that date after verifying the actual compiler predefined macros; all raw observations, rates and remaining identities are unchanged. The original measured profile is retained. The current compiler parser now accepts this GCC format, with 159 profile tests passing. No historical compiler selection code was changed.

## Limits and next work

Median checkpoint flags: `{"ten_percent_below_7a": true, "twenty_times_native": false, "twice_native": true}`. A 20× result would require 0.584 s and 0.660 s for these cases. The research target is not achieved; these thresholds are not forecasts.

This is a comparison of the complete resident implementation against the original controls. It does not establish the isolated value of packed graph arguments. The earlier packed-argument pilot failed its all-pairs retention gate and remains recorded as failed. Its 47-case runtime milestone and the separate 32-case graph queue calibration remain supporting correctness/cost evidence, not additional application timing samples.

Further work should target measured kernel, launch, startup and output costs. Generic scheduling migration, remaining legacy-policy deletion and full natural-run qualification remain open. Do not represent the raw queue calibration as a fitted automatic-placement profile.

## Evidence

Full build, validation, all timing outputs and controllers: `/tmp/fortran-packed-regenerated-20261110-01`. Fresh pinned-control calibration, raw banner and macro proof, source generation and build receipts: `/tmp/fortran-7a612-cxx-refresh-20261110-01`. The machine-readable [summary](summary.json) binds every listed receipt by SHA-256.

## Follow-up

The [local solver cache milestone](../gpu-local-solver-cache-20261110/README.md) records a further isolated performance gain, complete primary profiles, and the newly repeated natural2458 failure. The original samples and qualification scope above remain unchanged.
