# Fresh ELMM speedup comparison — 2026-11-08

Candidate: the application-qualified frozen `2619b7f` empty-native CALL snapshot. Later compiler commits and the uncommitted descriptor work are excluded. All three controls were newly built on GNU Fortran 15/CUDA 13.5; `7a612e4` uses a fresh calibration of its own pinned runtime. The candidate reuses its qualified build from today.

| Case | Native median (s) | be58 median (s) | 7a612e4 median (s) | Candidate median (s) | Native speedup | Candidate vs 7a612e4 | Wins vs 7a612e4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| channel-simple (109 steps) | 11.768500 | 9.448505 | 16.801202 | 11.271427 | 1.044× | -32.91% wall time | 7/7 |
| channel (100 steps) | 13.186841 | 11.257578 | 11.340621 | 16.321227 | 0.808× | +43.92% wall time | 0/7 |

Every case/mode has one excluded warmup and seven measured runs. Modes rotate by round; all 56 samples are retained without outlier removal or reruns. Timing covers the complete production process, including startup, the original CPU PoisFFT solver, transfers, all output and teardown. Four threads run on CPUs 0,2,4,6; the RTX 3060 Ti is selected by UUID. All modes use the same four verified FFTW plans and approved deterministic pressure reference. No tracing is enabled in timing runs.

All eight short checks pass all 20 fields, bounds, halos and outputs at `atol=1e-6`, `rtol=1e-5`. Every output from every timing process passes comparison with the corresponding fresh native warmup. The candidate previously passed natural 109/2,458-step qualification. **These timing results are preliminary: the freshly built controls have not repeated their natural 109/2,458-step qualification.** No release or research milestone is accepted by this packet.

The predeclared first application checkpoint requires at least 10% lower median than fresh `7a612e4` in both cases; robust improvement additionally requires all seven paired wins. The 2× and 20× objectives compare with native. Numerical thresholds below describe the measurements only.

Measured gates across both cases: `{"median_ten_percent_gain": false, "native_20x": false, "native_2x": false, "robust_vs_7a612": false}`.

| Case | Complete-process 20× budget (s) | Candidate / target budget |
| --- | ---: | ---: |
| channel-simple | 0.588425 | 19.155× |
| channel | 0.659342 | 24.754× |

This is the measured remaining gap, not a forecast that residency alone can close it.

## channel-simple: all measured samples

| Mode | Seven wall times (s), round order | Min–max (s) | Inclusive IQR (s) |
| --- | --- | ---: | ---: |
| baseline-7a612 | 16.923195, 16.801202, 16.879189, 16.436702, 16.619843, 16.829473, 16.585451 | 16.436702–16.923195 | 0.251684 |
| be58 | 9.319101, 9.442800, 9.890596, 9.574221, 9.448505, 9.411357, 9.828969 | 9.319101–9.890596 | 0.274516 |
| candidate | 11.571563, 11.084778, 11.289514, 11.271427, 11.139424, 11.168513, 11.278811 | 11.084778–11.571563 | 0.130194 |
| native | 11.627561, 11.588529, 11.982926, 11.838587, 11.665852, 11.768500, 11.887016 | 11.588529–11.982926 | 0.216095 |

| Control | Candidate/control median ratio | Paired wins | Seven paired ratios |
| --- | ---: | ---: | --- |
| baseline-7a612 | 0.670870 | 7/7 | 0.683769, 0.659761, 0.668842, 0.685747, 0.670248, 0.663628, 0.680042 |
| be58 | 1.192932 | 0/7 | 1.241704, 1.173887, 1.141439, 1.177268, 1.178961, 1.186706, 1.147507 |
| native | 0.957762 | 7/7 | 0.995184, 0.956530, 0.942133, 0.952092, 0.954874, 0.949018, 0.948835 |

Descriptive thresholds: `{"candidate_vs_7a612_ratio": 0.670870269566284, "median_ten_percent_gain": true, "native_20x": false, "native_2x": false, "native_speedup": 1.0441003342587913, "robust_vs_7a612": true, "scope": "overall checkpoint requires both cases; all seven pairs retained"}`.

## channel: all measured samples

| Mode | Seven wall times (s), round order | Min–max (s) | Inclusive IQR (s) |
| --- | --- | ---: | ---: |
| baseline-7a612 | 11.466095, 11.340621, 11.193291, 11.209602, 11.312461, 11.530432, 11.788685 | 11.193291–11.788685 | 0.237231 |
| be58 | 11.215422, 11.257578, 11.267263, 11.067623, 11.451595, 11.175500, 11.379456 | 11.067623–11.451595 | 0.127899 |
| candidate | 16.059751, 16.321227, 16.140675, 16.865588, 16.520658, 17.304858, 16.082764 | 16.059751–17.304858 | 0.581403 |
| native | 13.045789, 13.184089, 13.202423, 13.431819, 13.273362, 13.165307, 13.186841 | 13.045789–13.431819 | 0.063195 |

| Control | Candidate/control median ratio | Paired wins | Seven paired ratios |
| --- | ---: | ---: | --- |
| baseline-7a612 | 1.439183 | 0/7 | 1.400630, 1.439183, 1.441995, 1.504566, 1.460395, 1.500799, 1.364254 |
| be58 | 1.449799 | 0/7 | 1.431935, 1.449799, 1.432528, 1.523867, 1.442651, 1.548464, 1.413316 |
| native | 1.237690 | 0/7 | 1.231029, 1.237949, 1.222554, 1.255644, 1.244648, 1.314429, 1.219607 |

Descriptive thresholds: `{"candidate_vs_7a612_ratio": 1.439182819582848, "median_ten_percent_gain": false, "native_20x": false, "native_2x": false, "native_speedup": 0.807956481412454, "robust_vs_7a612": false, "scope": "overall checkpoint requires both cases; all seven pairs retained"}`.

## Evidence

- [Protocol and frozen identities](/tmp/fortran-speedup-empty-native-20261108-01/protocol.json)
- [All raw runs and controller completion](/tmp/fortran-speedup-empty-native-20261108-01/completion.json)
- [Short full-field checks](/tmp/fortran-speedup-empty-native-20261108-01/short-qualification.json)
- [Candidate natural qualification](/tmp/fortran-empty-native-rk-natural-20261108-02/qualification.json)
- [Fresh control build qualification](/tmp/fortran-copy-owner-controls-20261108-01/controls/qualification.json)
- [Fresh pinned 7a runtime calibration](/tmp/fortran-7a612-runtime9517-driver-refresh-20261108-01/qualification.json)
- [Machine-readable distributions and paired results](summary.json)

Existing short candidate diagnostics report unchanged physical copy totals after the empty-CALL change: it moves one source-level cut without establishing a whole-application traffic reduction. Those traced diagnostics are separate evidence and are not timing samples. No causal attribution follows from this comparison alone.

## Separate short runtime diagnostics and next experiment

These traced checks include initialization plus **two steps**. They are separate from the 56 untraced timing samples. The table counts compact runtime records; it does not infer steady-state per-step traffic, total CUDA kernel counts, GPU duration or an exclusive-time breakdown.

| Case | Mode | Launch records | Upload + download records | Upload bytes | Download bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| channel-simple | be58 | 7 | 70 | 93,872,128 | 29,360,128 |
| channel-simple | baseline-7a612 | 97 | 358 | 202,040,320 | 123,460,288 |
| channel-simple | candidate | 126 | 148 | 134,664,192 | 146,316,352 |
| channel | be58 | 7 | 70 | 93,872,128 | 29,360,128 |
| channel | baseline-7a612 | 7 | 70 | 93,872,128 | 29,360,128 |
| channel | candidate | 402 | 730 | 184,158,208 | 146,316,352 |

Candidate compute placement remains forced `sections`; physical transfers use calibrated `auto`, and GPU source-compute pricing remains unqualified. This is not production AUTO qualification. The freshly calibrated historical `7a612e4` control makes different reached runtime choices between the two cases; its new simple109 result must not be described using the old report's SGS-only profile.

The channel candidate has substantially more runtime launch and copy records than either historical control. This is a diagnostic lead, not proof that these events account for the observed regression. The next bounded experiment should isolate channel Convection placement and retained residency, obtain a CPU/GPU timeline on this exact frozen build, and distinguish kernel time, launch/management time and transfers before choosing another optimization. Keep the fresh four-mode comparison as the acceptance check. Further source-proof work alone is not evidence of a performance gain.

[Machine-readable short diagnostics and raw log hashes](short-trace-summary.json).
