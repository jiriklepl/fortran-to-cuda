# Long-run wall/solver isolation and SGS geometry, 2026-11-10

The remaining strict channel2458 failure is narrowed to the newly moved wall-stress/pressure-solver path. Returning only PoisFFT to CPU does not fix it. Returning both wall-stress arithmetic and PoisFFT to CPU passes every field/output check while the remaining numerical entries retain GPU execution. These are diagnostic builds, not a replacement for the requested complete GPU implementation. The selected [cached GPU performance checkpoint](../gpu-local-solver-cache-20261110/README.md) stays unchanged at 3.60×/3.45× native.

| Wall stress | Pressure solver | Fields beyond tolerance | Outputs beyond tolerance | Result |
| --- | --- | ---: | ---: | --- |
| GPU | GPU | 15/20 | 48/114 | FAIL |
| GPU | CPU | 15/20 | 48/114 | FAIL |
| CPU | GPU | 15/20 | 16/114 | FAIL |
| CPU | CPU | 0/20 | 0/114 | PASS |

Every process completes naturally at 2,458 steps and exits normally. Tolerances remain `atol=1e-6`, `rtol=1e-5`, with four native threads and the approved deterministic pressure reference. All diagnostic short cases pass before their long run. No diagnostic duration is used as a speed comparison.

## Controlled changes and execution evidence

For the CPU solver variant, only the resident solve call is replaced. Exact host access transactions read the RHS interior and overwrite the Phi interior; GPU halo fill, pressure correction and remaining stages are unchanged. A preloaded guard aborts any GPU solver execution, and the four native FFTW plan identities must match the approved control. GPU plan construction still occurs for the resident provider/halo path; this diagnostic does not claim zero GPU provider setup.

For CPU wall stress, only the four generated wall-stress entry mode arguments change. The same packed wall state, arithmetic source, generic coherence and GPU wall corrections remain. Compiler implementation IDs in short traces identify all four expected generated-CPU wall procedures and 24 remaining numerical GPU procedures. No unexpected generated-CPU procedures appear. Both solver choices use this same wall object.

The combined CPU-wall/CPU-solver diagnostic matches **all 20 binary fields and all 20 shape/bounds files byte for byte** against the reviewed historical be58 long-run control. The current native reference likewise matches all 40 corresponding historical native files. Historical payloads were authenticated against SHA-256 entries in the committed 2026-10-22 report before comparison. The combined diagnostic’s maximum absolute field error versus native is 1.059737481143297e-6, and it passes the combined absolute/relative criterion.

## What this establishes and what remains uncertain

The long failure cannot be attributed solely to the GPU Poisson replacement. With CPU wall stress and GPU Poisson it falls from 48 failing outputs to 16; changing both gives the exact previously qualified GPU trajectory. This localizes the next investigation, but does not by itself prove a particular libm rounding defect, solver defect or synchronization defect. Host execution also introduces completion points, so synchronization remains a hypothesis until checked separately.

The wall helper has a piecewise logarithmic law, an early convergence test and a 50-iteration failure result of zero. It is a concrete place to compare CPU/GPU behavior on identical reached inputs. The provider should separately be compared against pinned PoisFFT using identical reached RHS data, with correct FD2/FD4 residual operators and gauge handling. Prior native-only low-bit perturbation evidence demonstrates sensitivity, but neither that evidence nor this factorial experiment waives the strict long-run requirement. No new physics, tolerance, production CPU fallback or library-specific math emulation is introduced here.

## Synchronization-only follow-up

A separate diagnostic inserts the existing `fort_scope_wait` before and after each GPU wall-stress entry and GPU pressure solve. This flushes queued commands and waits for completion, preserving arithmetic, GPU placement and access declarations. Only the temporary wall and pressure objects are rebuilt; all remaining application objects and the cached provider are unchanged. No synchronization is added to production.

Both the two-step run and natural 2,458-step run produce all **40 field/bounds files byte-identically** to their original GPU references. The long run therefore still fails 15 fields and 48 outputs at the unchanged tolerances. The short execution audit identifies all 28 expected channel numerical GPU procedures and zero CPU FFTW plans. Both processes exit normally and their process groups are absent.

Waiting at these operation boundaries does not repair the discrepancy. This rules out a missing completion point at these tested boundaries as the observed cause; it does not prove every possible internal synchronization property or identify the numerical cause. Next compare identical reached inputs through CPU/GPU wall arithmetic and the two solvers. Evidence: `/tmp/fortran-gpu-wait-isolation-20261110-01`, including exact source diffs, build/run receipts and `audit.json`. Diagnostic duration is excluded from performance comparisons.

## SGS launch-size result

Hardware-counter profiling was attempted and denied by `ERR_NVGPUCTRPERM`; no counter metrics or counter-based bottleneck claim is made. A separate CUDA-event sweep runs the exact generated SGS kernel with saved simple109/channel100 fields and representative scalar values. Five rotated measurement rounds follow one excluded warmup, with 64 kernel executions per sample. Outputs at every block size are byte-identical to the 256-thread reference. These are isolated-kernel measurements, not end-to-end samples.

| Fixture | 32 threads | 64 threads | 128 threads | 256 threads | 512 threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| simple109 | 1.599984 ms | 1.599280 ms | 1.599680 ms | 1.604464 ms | 1.597776 ms |
| channel100 | 1.603824 ms | 1.605216 ms | 1.608224 ms | 1.599088 ms | 1.605920 ms |

All medians are within roughly 0.6% of the existing geometry. No useful launch-size gain is established, so no new launch heuristic or option is added. The occupancy API suggested 576 threads; it is not substituted for measured performance evidence. The original generated kernel and production compiler remain unchanged.

## Evidence

Solver isolation: `/tmp/fortran-pressure-cpu-isolation-20261110-01`. Wall factorial: `/tmp/fortran-wall-cpu-isolation-20261110-02`. SGS sweep: `/tmp/fortran-sgs-geometry-20261110-01`. Hardware-counter attempt: `/tmp/fortran-sgs-kernel-profile-20261110-01`. The initial wall diagnostic build failed due to an inline comment consuming a Fortran continuation marker; its log is retained separately and it was never executed. The corrected build uses plain constant-mode substitution. The [machine-readable summary](summary.json) binds exact receipts.

The full GPU implementation still fails the strict long-run milestone. The research objective and remaining production migration/cleanup stay open; no completion or release claim is made.
