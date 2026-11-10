# Production copy-policy comparison — 2026-11-08

The candidate was normally regenerated and publicly rebuilt from the application-qualified 2619b7f compiler basis with only the generic copy-policy files from 03c695d. It uses fresh matching calibration for runtime `6f32137c`. The later native-CALL changes and external descriptor draft are excluded. Native, be58 and 7a612e4 reuse the fresh controls built earlier today; every timing sample below is new. No private copy interposer is used.

| Case | Native (s) | be58 (s) | 7a612e4 (s) | Candidate (s) | Native speedup | Candidate vs 7a612e4 | Paired wins vs 7a612e4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| channel-simple (109 steps) | 11.586099 | 9.466523 | 16.634371 | 11.185685 | 1.036× | -32.76% wall time | 7/7 |
| channel (100 steps) | 13.031851 | 10.992937 | 11.105197 | 16.235619 | 0.803× | +46.20% wall time | 0/7 |

Four threads on CPUs 0,2,4,6; RTX 3060 Ti selected by UUID; the same approved deterministic pressure reference and four verified FFTW plans. Each case/mode has one excluded warmup and seven rotated measurements, without outlier removal or reruns. Complete production-process time includes startup, the original CPU PoisFFT solver, all output and teardown. Tracing and diagnosis are separate.

Eight short checks pass all 20 fields, bounds, halos and outputs at atol=1e-6 and rtol=1e-5. All 64 timing processes pass output comparison. The new candidate passes natural simple109 checks, with all 20 fields bitwise equal to the prior candidate; the channel2,458-step check is still running at publication time. Fresh controls have not repeated their natural qualification; this packet is preliminary and does not constitute a milestone release or a complete maintained-suite pass.

Measured gates across both cases: `{"median_ten_percent_gain": false, "native_20x": false, "native_2x": false, "robust_vs_7a612": false}`.

## channel-simple: all measured samples

| Mode | Seven wall times (s), in round order | Min–max (s) | Inclusive IQR (s) |
| --- | --- | ---: | ---: |
| baseline-7a612 | 16.659166, 17.207616, 16.812066, 16.456432, 16.520815, 16.634371, 16.410430 | 16.410430–17.207616 | 0.246992 |
| be58 | 9.466523, 9.164206, 9.490503, 9.259112, 9.339999, 9.927536, 9.608278 | 9.164206–9.927536 | 0.249835 |
| candidate | 11.409580, 10.976485, 11.262049, 11.178084, 11.185685, 11.004180, 11.355690 | 10.976485–11.409580 | 0.217738 |
| native | 11.665861, 11.483772, 11.715962, 11.788747, 11.375839, 11.524302, 11.586099 | 11.375839–11.788747 | 0.186875 |

| Control | Candidate/control median ratio | Paired wins | Seven paired ratios |
| --- | ---: | ---: | --- |
| baseline-7a612 | 0.672444 | 7/7 | 0.684883, 0.637885, 0.669879, 0.679253, 0.677066, 0.661533, 0.691980 |
| be58 | 1.181604 | 0/7 | 1.205256, 1.197756, 1.186665, 1.207252, 1.197611, 1.108450, 1.181865 |
| native | 0.965440 | 7/7 | 0.978032, 0.955826, 0.961257, 0.948200, 0.983284, 0.954867, 0.980113 |

The measured 20× complete-process budget is 0.579305 s; this candidate remains 19.309× above it.

## channel: all measured samples

| Mode | Seven wall times (s), in round order | Min–max (s) | Inclusive IQR (s) |
| --- | --- | ---: | ---: |
| baseline-7a612 | 11.457744, 11.078179, 10.984922, 11.668675, 11.090728, 11.789429, 11.105197 | 10.984922–11.789429 | 0.478756 |
| be58 | 11.227381, 11.177370, 10.967463, 11.271748, 10.804177, 10.872864, 10.992937 | 10.804177–11.271748 | 0.282212 |
| candidate | 15.965608, 16.118358, 16.235619, 16.605974, 16.412160, 16.097478, 16.514088 | 15.965608–16.605974 | 0.355206 |
| native | 12.942995, 13.098194, 12.884054, 13.149670, 13.071038, 13.031851, 12.940452 | 12.884054–13.149670 | 0.142892 |

| Control | Candidate/control median ratio | Paired wins | Seven paired ratios |
| --- | ---: | ---: | --- |
| baseline-7a612 | 1.461984 | 0/7 | 1.393434, 1.454965, 1.477991, 1.423124, 1.479809, 1.365416, 1.487059 |
| be58 | 1.476914 | 0/7 | 1.422024, 1.442053, 1.480344, 1.473239, 1.519057, 1.480519, 1.502245 |
| native | 1.245841 | 0/7 | 1.233533, 1.230579, 1.260133, 1.262843, 1.255613, 1.235241, 1.276160 |

The measured 20× complete-process budget is 0.651593 s; this candidate remains 24.917× above it.

## Evidence

- [Frozen protocol](/tmp/fortran-copy-v3-speedup-20261108-01/protocol.json)
- [Complete timing runs](/tmp/fortran-copy-v3-speedup-20261108-01/completion.json)
- [All 20 fields in short checks](/tmp/fortran-copy-v3-speedup-20261108-01/short-qualification.json)
- [Natural correctness run](/tmp/fortran-copy-v3-natural-20261108-01)
- [Distributions and paired results](summary.json)
- [Copy-policy motivation, private pilot and tests](../gpu-residency-copy-v3-20261108/README.md)
- [Earlier four-mode comparison](../gpu-residency-speedup-20261108/README.md)

The earlier private three-pair packing pilot and earlier candidate timings are separate experiments. Do not interpret cross-campaign median changes as seven paired wins. Forced sections compute remains diagnostic; calibrated AUTO physical copies do not qualify GPU source-compute pricing.
