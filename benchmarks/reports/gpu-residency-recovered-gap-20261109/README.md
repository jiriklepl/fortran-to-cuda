# Recovered candidate: remaining regression diagnosis

The latest completed untraced pilot still loses to be58 in both cases:
11.707750 versus 10.945415 seconds for simple109 (**6.96% slower**), and
13.715633 versus 12.934806 seconds for channel100 (**6.04% slower**).
Both lose all three measured pairs. Keeping Convection native recovered
27.47% on channel relative to forced partial GPU Convection; that recovery
is not a new performance win. See the [complete pilot](../gpu-residency-convection-placement-pilot-20261109/README.md).
These are one excluded warmup plus three rotated measurements, not the
seven-round qualification protocol. No new untraced timings were collected.

## New separate profiles

Four complete-process CPU/CUDA profiles compare the previously timed private
CPU-Convection candidate and unchanged be58 binary. Four threads, affinity
0/2/4/6, approved deterministic pressure reference, original CPU solver,
matching four FFTW plans, physics and outputs are preserved. All application
and output checks pass at atol=1e-6, rtol=1e-5. All owned profiler process groups
are absent after completion. These output checks do not replace natural
2,458-step full20 qualification.

| Case / mode | GPU kernels | Host copy API union (s) | All host CUDA API union (s) | GPU kernels (s) | GPU copies (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| channel / be58 | 301 | 0.848765 | 1.746859 | 0.579275 | 0.614075 |
| channel / candidate | 2100 | 1.008112 | 1.962886 | 0.590324 | 0.729899 |
| channel-simple / be58 | 328 | 0.887569 | 1.782289 | 0.631246 | 0.641884 |
| channel-simple / candidate | 2289 | 1.027291 | 2.005344 | 0.632982 | 0.742820 |

These spans overlap; **do not add the columns or treat profiled durations as
untraced timing samples**. The candidate runs 300/327 SGS kernels plus six
viscosity boundary kernels for each SGS invocation. be58 runs 301/328 SGS
kernels. All six candidate boundary kernels together account for only
0.003736/0.003990 seconds of GPU execution. CUDA launch API spans are
0.0708/0.0654 seconds for the candidate, versus 0.0174/0.0173 seconds for be58
(rounded diagnostic values). Eliminating those launch spans alone is not a
credible explanation for, or remedy to, the full remaining application gap.

Channel bulk-copy API counts are close: 1,200 cudaMemcpy3DAsync and 1,800
cudaMemcpy2DAsync in the candidate versus 1,204 and 1,806 in be58. Total
payload is 5,382,163,200 versus 5,298,987,008 bytes. Candidate downloads are
1,359,072,000 versus 1,262,485,504 bytes; upload payload is slightly lower.
This contrasts with the earlier forced-Convection profile's 22,996 3D calls
and 16.524 GB. Avoid optimizing that obsolete placement as though it were
the recovered candidate.

## CPU evidence and its limits

| Case / mode | Main-thread leaf samples | Allocator/free leaf samples | Inclusive coherence samples |
| --- | ---: | ---: | ---: |
| channel / be58 | 24642 | 73 (0.30%) | 21 |
| channel / candidate | 26825 | 1025 (3.82%) | 282 |
| channel-simple / be58 | 18870 | 81 (0.43%) | 15 |
| channel-simple / candidate | 20575 | 1046 (5.08%) | 323 |

Allocator counts use function-name matching as a diagnostic heuristic.
Inclusive coherence counts overlap other categories. Neither is wall time.
Many call stacks are incomplete: no allocator sample has a resolved runtime
ancestor, so the extra allocations cannot all be assigned to coherence.
The candidate does sample coherence set difference, coalescing and planning
inputs directly. Enclosing ELF symbols confirm that generated `master`
labels include ordinary OpenMP numerical loops; they are not evidence that
those computations execute serially.

The useful next question is whether repeated region/planning allocations
and publication work explain enough of the remaining gap to justify changing
them. First measure their counts and caller attribution on this recovered
placement. Do not undertake another broad runtime rewrite based on sample
shares alone. Launch batching remains useful later, but this profile does not
support prioritizing it as the main regression fix.

## Environment and interrupted build

The first profile attempt stopped before application execution because its
input guard detected external updates to g++-14 and libstdc++.so. The second
attempt records their new hashes; both unchanged application binaries use
the same current runtime. Earlier pilot timings belong to their recorded
previous environment. New profiles do not revise those timings.

The AUTO generation at `/tmp/fortran-auto-placement-20261109-01` lost its
tool session and has no completion receipt. A host-namespace process check
found no matching generation process. Its retained execution file still says
RUNNING; `interruption-audit.json` records the later observation without
rewriting that historical file. No package, build or performance success is
claimed. It has not been restarted. Historical AUTO results already declined
Convection GPU work, so merely reproducing that placement is not an acceleration
result.

## Evidence

- [Profile plan and fixed inputs](/tmp/fortran-recovered-gap-profile-20261109-02/plan.json)
- [Application/output receipts](/tmp/fortran-recovered-gap-profile-20261109-02/result.json)
- [CUDA intervals and kernel inventory](/tmp/fortran-recovered-gap-profile-20261109-02/analysis.json)
- [Reproducible CUDA analysis](/tmp/fortran-recovered-gap-profile-20261109-02/analyze.py)
- [Resolved CPU samples](/tmp/fortran-recovered-gap-profile-20261109-02/cpu-analysis.json)
- [Reproducible CPU resolution](/tmp/fortran-recovered-gap-profile-20261109-02/resolve_cpu.py)
- [Recorded environment change](/tmp/fortran-recovered-gap-profile-20261109-02/environment-delta.json)
- [Interrupted-generation audit](/tmp/fortran-auto-placement-20261109-01/interruption-audit.json)

No compiler/runtime implementation is promoted by this diagnosis. Full
maintained suites, current natural correctness, seven-round performance
qualification, 2× and 20× objectives remain open.
