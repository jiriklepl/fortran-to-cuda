# Single-operation copy packing — 2026-11-08

Status: generation, build, short and natural correctness pass, but the [completed production comparison](../gpu-residency-copy-v3-speedup-20261108/README.md) shows no meaningful improvement and zero selections of the intended single-operation path. No speedup or research checkpoint is accepted.

## Evidence driving this change

The fresh four-mode comparison remains [recorded separately](../gpu-residency-speedup-20261108/README.md). Its channel100 regression prompted three separate Nsight diagnoses on the exact frozen binaries. All outputs passed. Main-thread CUDA API intervals overlap GPU work and must not be added to GPU durations.

| channel100 diagnostic | be58 | 7a612e4 | Prior candidate |
| --- | ---: | ---: | ---: |
| Kernels | 301 | 301 | 21,892 |
| Kernel activity, s | 0.579 | 0.580 | 1.504 |
| Copy operations | 3,010 | 3,010 | 32,188 |
| Copied bytes | 5,298,987,008 | 5,298,987,008 | 16,523,728,000 |
| Copy activity, s | 0.614 | 0.625 | 2.633 |
| Main-thread CUDA API union, s | 1.673 | 1.674 | 7.985 |

The candidate issued 22,996 `cudaMemcpy3DAsync` calls with 5.528 s of API intervals. A separate two-step geometry probe found costly 24-byte-wide faces with 64×64 or 58×58 rows, pitch 1,072 bytes and physical height 70. These cannot be flattened into one 2D copy without crossing gaps.

A private interposer used the existing exact gather kernel and bounded 2 MiB workspace for narrow D2H faces. Both two-step cases passed all 20 fields, bounds, halos, outputs and FFTW-plan checks. A predeclared channel100 pilot then used one excluded warmup and three alternating measured pairs. Original samples were 16.096806556, 16.810902271 and 16.237522630 s; packed samples were 14.592455874, 14.224691554 and 14.638801630 s. Medians improved 16.237522630 → 14.592455874 s (10.13%), with 3/3 paired wins. This diagnostic is not the production policy or the seven-round checkpoint; it remains slower than the fresh native and historical controls.

## Production implementation

Physical-copy ABI, observation and calibration schema version 3 admit a single operation only inside measured geometry support. Direct, cold compact and reused compact paths are measured for single dense and strided operations across byte/int32/real64 layouts. Unmeasured shapes and explicit DIRECT retain direct transport. Old v1/v2 copy profiles are rejected; regenerate calibration and artifacts. The numerical fit method itself is unchanged.

The existing runtime still owns exact regions, replica validity, workspace accounting, completion, startup credit and failure poisoning. Read-only movement retains the source replica. No new coherence engine or interposer enters production.

Fresh calibration matches runtime `6f32137c39a5a76dd5a5b57b087240efa364a407049fd5792703951f7b75e65e`, profile SHA256 `471aab90098f6184d6137904ea084c1cf09574f4d214c88b6edde5a0a0ef62e2`. Eight independent held-out shapes produced 48 records and 384 verified complete-path batches, each with an excluded warmup and seven rotated observations. All five nominal warm selections were faster in measurement. The four selected 24-byte faces had reused/direct ratios 0.108–0.141; tested small and wide shapes that lost with packing remained direct. This validates a limited geometry sample, not a global prediction bound or application gain. Pending work, workspace availability and cold startup still affect actual admission.

## Validation and remaining work

- Initial affected CPU checks: 102 passed; actual CUDA transport/startup checks: 34 passed.
- Final manifest/profile checks: 42 passed, including emitted ABI version 3.
- Related profile, planning, batch, pinned and coalesced checks: 276 passed.
- Companion source transport: 36 passed, including actual current and frozen historical compiler transports.
- These groups overlap. They are not a complete maintained-suite run.

Production regeneration uses the application-qualified 2619b7f compiler basis plus only the 16 copy-policy files; the external descriptor draft and later native-CALL changes are excluded. The plan, source guards and process ownership are in `/tmp/fortran-copy-v3-parent-20261108-01`. Public build, strict short correctness, seven-round application timing and natural correctness are complete. Both natural cases pass all 20 fields, bounds, halos and outputs; field bytes match the previous candidate exactly. The transfer admission check fails: no single-operation compact copy executes. Relative to be58, median wall time is 18.16% higher in simple109 and 47.69% higher in channel100, with no paired wins. The microbenchmark and private-interposer gains did not establish a production improvement.

Raw evidence: `/tmp/fortran-channel-regression-profile-20261108-02`, `/tmp/fortran-copy-geometry-20261108-01`, `/tmp/fortran-single-copy-pack-20261108-02`, `/tmp/fortran-single-copy-pack-pilot-20261108-01`, `/tmp/fortran-copy-v3-calibration-20261108-01`, `/tmp/fortran-copy-v3-holdout-20261108-01`. Failed first profiler/probe attempts remain recorded. Reports remain local under the ignored directory.
