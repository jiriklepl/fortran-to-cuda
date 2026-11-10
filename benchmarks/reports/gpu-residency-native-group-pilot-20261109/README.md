# ELMM native-group retention: performance pilot — 2026-11-09

The candidate partially recovers the recent channel regression, but delivers no substantial improvement in both cases. It remains slower than be58 in every measured pair. The intended application transfer reduction is absent. This candidate is not promoted.

| Case | Native median | Previous median | Candidate median | be58 median | Change vs previous | Change vs be58 | Native speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| channel-simple | 12.950160 s | 12.578384 s | 12.577033 s | 10.709479 s | -0.01% | +17.44% | 1.030x |
| channel | 14.803092 s | 19.984053 s | 18.497837 s | 13.017861 s | -7.44% | +42.10% | 0.800x |

Each case/mode has one excluded warmup and three rotated measurements, with no discarded outliers: 32 complete processes. Primary cases are simple109 and channel100. Four threads on CPUs 0,2,4,6; RTX 3060 Ti; unchanged physics, approved deterministic TOP pressure reference, CPU PoisFFT and four verified FFTW plans. Startup, original output and teardown remain in the process time. Controls reuse frozen binaries from the preceding campaign; every sample here is fresh. This pilot does not replace the seven-round protocol or provide a new 7a612e4 comparison.

All original output comparisons pass at atol=1e-6, rtol=1e-5. No build, test or profile overlapped timing. The predeclared whole-pilot rejection limit was 0.5 external CPU core averaged over any process; the observed maximum was 0.459256. The controller reaped every owned process group.

The simple median difference from the previous candidate is only 0.001352 seconds (0.01%), effectively a tie. Channel is 7.44% faster by median and wins all three pairs against that previous candidate. Candidate wins against be58 are 0/3 in each case. The native speedups remain far below the 2x and 20x objectives.

All measured samples are retained below in round order; excluded warmups remain in the raw result.

| Case | Mode | Three measured process times (s) | Min–max (s) |
| --- | --- | --- | ---: |
| channel-simple | native | 14.859502, 12.874764, 12.950160 | 12.874764–14.859502 |
| channel-simple | previous candidate | 13.037272, 12.245791, 12.578384 | 12.245791–13.037272 |
| channel-simple | candidate | 12.577033, 13.416803, 12.147672 | 12.147672–13.416803 |
| channel-simple | be58 | 10.366428, 11.141702, 10.709479 | 10.366428–11.141702 |
| channel | native | 14.681706, 14.803092, 15.435160 | 14.681706–15.435160 |
| channel | previous candidate | 19.437180, 19.984053, 20.350141 | 19.437180–20.350141 |
| channel | candidate | 19.196405, 18.497837, 18.024682 | 18.024682–19.196405 |
| channel | be58 | 13.017861, 12.922769, 13.039882 | 12.922769–13.039882 |

| Case | Control | Candidate/control paired ratios | Candidate wins |
| --- | --- | --- | ---: |
| channel-simple | native | 0.846397, 1.042101, 0.938033 | 2/3 |
| channel-simple | original | 0.964698, 1.095626, 0.965758 | 2/3 |
| channel-simple | be58 | 1.213247, 1.204197, 1.134292 | 0/3 |
| channel | native | 1.307505, 1.249593, 1.167768 | 0/3 |
| channel | original | 0.987613, 0.925630, 0.885728 | 3/3 |
| channel | be58 | 1.474621, 1.431414, 1.382273 | 0/3 |

Six separate short native/previous/candidate runs pass all 20 fields, bounds, halos and outputs. Numerical launches and allocations match. Initialization-plus-two-step physical transfer totals are unchanged:

| Case | H2D bytes | D2H bytes | Copy events | Previous → candidate contexts |
| --- | ---: | ---: | ---: | ---: |
| channel-simple | 134,664,192 | 146,316,352 | 148 | 16 → 16 |
| channel | 184,158,208 | 146,316,352 | 730 | 70 → 40 |

These totals include initialization and two steps; they are not steady-state per-step traffic. Lower channel context counts do not establish retained payloads. The new source contract still rejects the targeted native groups because reservations for child-local scalars (`tnz` and `reg`) cannot span the lexical owner. Those groups become explicit reached retirements. These targeted groups therefore do not exercise the retained native-group path demonstrated by the small regression.

Normal generation and strict package validation pass in 2,046.40 seconds; the prior generation took 1,275.68 seconds. Public build and diagnostic relink pass in 246.35 seconds. GPU leaf inventory and runtime identity are unchanged. The shared native-host issuer/reader transport is version 2; runtime ABI is unchanged. Compute placement remains forced-sections diagnostic, with matching calibrated AUTO physical copies. Automatic compute placement is not qualified.

The prior focused validation comprises 144 source/reader checks and six CUDA/hidden-device checks. It demonstrates preserved read-only replicas, removal of a 1,728-byte redundant upload, exact 24-byte partial updates and real child-handle forwarding. The missing automatic-local-scalar integration case must be covered. Full maintained-suite testing and this candidate’s natural 2,458-step qualification remain pending.

Next work should explain the remaining measured gap to be58 using exclusive CPU/GPU attribution, including runtime planning, access preparation and launch costs. The local-scalar rejection needs an exact nested-owner consumer reproducer before another full regeneration. Do not infer an application improvement from source-proof admission or combine overlapping profiler durations.

Evidence:

- [Timing protocol](/tmp/fortran-retention-may-host-pilot-20261109-01/plan.json)
- [Every run, output comparison and sample](/tmp/fortran-retention-may-host-pilot-20261109-01/result.json)
- [Pilot completion and cleanup](/tmp/fortran-retention-may-host-pilot-20261109-01/execution.json)
- [Full20 short qualification and physical copy ledgers](/tmp/fortran-retention-may-host-short-20261109-01/result.json)
- [Remaining application admission failures](/tmp/fortran-retention-may-host-20261109-01/application-retention-admission.json)
- [Generated-route and GPU-leaf comparison](/tmp/fortran-retention-may-host-20261109-01/retention-comparison.json)
- [Frozen source provenance](/tmp/fortran-retention-may-host-20261109-01/snapshot-provenance.json)
- [Build and relink completion](/tmp/fortran-retention-may-host-build-20261109-01/execution.json)
- [Production binary receipt](/tmp/fortran-retention-may-host-build-20261109-01/work/production.json)
- [Machine-readable summary](summary.json)

Candidate SHA-256: `54054fe2e69c420f4fc5526983c374e1f51c500cbbe5f9dc837d94fa6f3949eb`.
Runtime identity: `d0e179c4821005b8fe9d39eab1f378e4ca4b67a5d4da0ed1c9dffd282b978d52`.
