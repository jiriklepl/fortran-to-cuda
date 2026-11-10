# Local Poisson kernel cache and measured critical path, 2026-11-10

The explicit local `poisson-solver` now reuses compiled VkFFT cubins across processes. This ports the existing bounded cache from the older vendored provider while preserving current stream ordering, allocation accounting and the device API. Numerical source, arithmetic and field residency are unchanged.

| Case | Native, four threads | GPU without cache | GPU with warm cache | Speedup vs native | Lower time vs uncached GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| simple109 | 11.480052 s | 4.056624 s | 3.190193 s | 3.60× | 21.36% |
| channel100 | 13.016128 s | 4.700712 s | 3.778208 s | 3.45× | 19.62% |

Predeclared cache retention gate (≥5% median improvement in both cases, all seven pairs won, external CPU <0.5 cores): **True**. Paired wins: `{"channel": {"gpu-uncached": 7, "native": 7}, "channel-simple": {"gpu-uncached": 7, "native": 7}}`. All 48 complete-process output comparisons pass; maximum external CPU is 0.355 cores. All owned process groups are reaped.

One excluded warmup and seven measurements per mode/case, with rotated order. The candidate starts each case with a fresh private cache; its excluded warmup compiles and stores kernels. Subsequent measurements reuse that cache. All process startup, cache lookup, misses, module loading, computation, output and teardown are timed. No tracing, builds or tests overlap measurements. No samples are trimmed or retried.

| Case | Excluded native warmup | Excluded uncached GPU warmup | Excluded empty-cache GPU warmup |
| --- | ---: | ---: | ---: |
| channel-simple | 11.379718 s | 4.134541 s | 4.173692 s |
| channel | 12.841574 s | 4.642650 s | 4.903283 s |

Empty-cache startup remains a separate cost. These are warm-cache results under the declared excluded-warmup protocol; they are not claims about a first-ever process. The [fresh four-control comparison](../gpu-resident-four-controls-20261110/README.md) remains intact. Here the exact fresh native and uncached-GPU executables are reused; the cached candidate reuses the exact qualified application objects and relinks a newly built local provider. A new full compiler regeneration is not claimed.

## Samples

| Case | Mode | Seven complete-process seconds |
| --- | --- | --- |
| channel-simple | native | 11.480052 / 11.376359 / 11.267840 / 11.540239 / 11.394593 / 11.586997 / 11.533927 |
| channel-simple | gpu-uncached | 4.056624 / 4.118821 / 4.069083 / 4.056004 / 4.092555 / 4.038276 / 3.991107 |
| channel-simple | candidate | 3.212997 / 3.156876 / 3.190193 / 3.155183 / 3.147024 / 3.234956 / 3.244065 |
| channel | native | 13.082388 / 12.949743 / 12.837959 / 12.885629 / 13.048837 / 13.016128 / 13.345797 |
| channel | gpu-uncached | 4.712692 / 4.700712 / 4.688308 / 4.710261 / 4.663360 / 4.646802 / 4.721783 |
| channel | candidate | 3.778208 / 3.762843 / 3.814490 / 3.822560 / 3.813992 / 3.748750 / 3.772747 |

## Cache milestone

Five host/API contract tests pass. On both primary cases, disabled and empty caches execute 12 NVRTC compilations; warm caches execute none; corrupting one stored image causes exactly one recompilation. All 20 fields, including bounds and halos, are bit-identical to the qualified uncached GPU control in all four modes. All outputs pass comparison against native at unchanged `atol=1e-6`, `rtol=1e-5`.

The first stricter byte-exact output assertion failed only for channel `Retau.unf`, with maximum absolute difference 9.09e-12. Existing repeated runs of the unchanged GPU control already vary up to 1.55e-11 in that diagnostic. This failed assertion and repeatability evidence are preserved. The corrected channel milestone retains exact comparisons for all fields and every other output; Retau uses the existing output tolerances. No source physics or tolerance was changed.

Warm solver creation takes about 0.23–0.24 s in separate instrumented milestone runs, versus about 1.1 s without caching. These diagnostics are not performance samples. Cache keys bind generated source/options, build recipe, NVRTC/builtins/driver bytes and device identity. The cache has 16 MiB per-image and 64 MiB disk bounds, private files, bounded live images, checksums, atomic writes and normal compilation on misses. Set `POISSON_SOLVER_KERNEL_CACHE_DIR` explicitly; omission disables it. Installed identity and recipe files bind the contract.

## Measured remaining work

The full primary profiles below use the uncached GPU control. Overlapping CPU API spans and GPU activity are not added together. Instrumented durations guide optimization and are not substitutes for untraced wall time.

| Profile | GPU kernel union | Device copy union | GPU idle inside first–last kernel interval | Largest measured kernels |
| --- | ---: | ---: | ---: | --- |
| channel-simple | 2.040 s | 0.023 s | 0.651 s | subgrid::sgs_sigma_uniform 0.525 s; VkFFT_main 0.259 s; momentumdiffusion::momentumdiffusion_nobranch_2ord 0.164 s; resident_wall::stress1 0.085 s |
| channel | 2.710 s | 0.024 s | 0.705 s | subgrid::sgs_sigma_uniform 0.485 s; filters::filter3ord 0.402 s; VkFFT_main 0.239 s; momentumadvection::cd4divu 0.194 s |

The next performance work should target SGS in both cases and filter/advection in channel, then measured queue/submission gaps. Physical copy time is now small. Kernel activity alone exceeds the approximately 0.6-second process budget required for 20×, so residency or startup improvements alone cannot reach that target. Preserve the rejected index-width/filter experiments and the failed isolated packed-runtime retention gate.

## Long-run release blocker

The current uncached source-built native and GPU executables both complete the natural channel run at 2,458 steps, but **15 of 20 fields and 48 of 114 outputs fail** the original pointwise tolerances. There are no nonfinite values. Five passing fields are inactive/empty scalar fields; this is not broad physical qualification. Maximum differences include U 6.30e-4, pressure 8.45e-4 and Ustar 2.41e-2. The strict long-run failure remains a release blocker, and caching does not resolve it.

Prior native-only low-bit perturbation evidence demonstrates trajectory sensitivity, but does not qualify the GPU solver or prove current differences harmless. No tolerance or workload change is made. Full natural-run qualification, maintained-suite milestone, remaining generic scheduling migration/legacy removal and the 20× research objective remain open.

## Evidence

Controller, sources, linked inputs, all outputs and samples: `/tmp/fortran-local-solver-cache-20261110-01`. Full profiles: `/tmp/fortran-resident-critical-path-20261110-01`. Natural-run failure: `/tmp/fortran-resident-long-20261110-01`. The [machine-readable summary](summary.json) binds the listed receipts by SHA-256.
