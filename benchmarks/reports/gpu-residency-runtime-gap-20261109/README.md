# GPU residency: measured runtime gap and an unpromoted planner experiment

The broader candidate still loses to be58. The new exact copy-planner shortcut
removes a sampled CPU hotspot but establishes no meaningful application gain.
Its source and tests are archived under `/tmp`; the production runtime header
has been restored byte-for-byte to HEAD. External edits remain intact.

The preceding [fresh four-mode pilot](../gpu-residency-native-group-pilot-20261109/README.md)
found the normal candidate 17.44% slower than be58 on simple109 and 42.10%
slower on channel100, losing every measured pair. That campaign is separate
from the private experiment below; do not compare medians across campaigns.

## Untraced application comparison

One excluded warmup and three rotated measurements for each case/mode:
24 complete processes, all samples retained. Four physical CPUs (0,2,4,6),
four OpenMP threads, RTX 3060 Ti, CPU PoisFFT, approved deterministic TOP
pressure reference, unchanged physics and outputs. No build, test or profile
overlapped timing. Maximum observed external CPU load was
0.4491 core, below the declared 0.5-core limit.
All output comparisons pass `atol=1e-6`, `rtol=1e-5`; the runner verifies
the four FFTW plans. Timed VTK outputs do not replace the separate full20
diagnostic qualification. This pilot does not qualify the seven-round gates.

| Case | Normal candidate | Private shortcut | be58 | Shortcut vs normal | Paired wins vs normal / be58 |
| --- | ---: | ---: | ---: | ---: | ---: |
| simple109 | 12.901912 s | 12.832135 s | 12.138835 s | -0.54% | 2/3 / 0/3 |
| channel100 | 19.316871 s | 19.293448 s | 13.786600 s | -0.12% | 2/3 / 0/3 |

Negative percentages mean lower wall time. Both cases win only two of
three pairs against normal, and none against be58. Variability is substantial;
the data do not establish a useful speedup. No new native ratios are claimed
because this campaign did not run fresh native controls.

| Case | Mode | All measured seconds, in round order |
| --- | --- | --- |
| channel-simple | original | 13.736891, 12.570250, 12.901912 |
| channel-simple | candidate | 12.832135, 12.181793, 13.813211 |
| channel-simple | be58 | 12.227130, 10.487429, 12.138835 |
| channel | original | 19.316871, 18.956129, 20.996492 |
| channel | candidate | 19.070269, 19.293448, 19.697935 |
| channel | be58 | 13.103791, 13.786600, 13.931884 |

Here `original` is the normal current candidate and `candidate` is the
private runtime shortcut. All paired ratios, excluded warmups, input hashes,
[results](/tmp/fortran-periodic-copy-pilot-20261109-01/result.json), [assessment](/tmp/fortran-periodic-copy-pilot-20261109-01/assessment.json),
and [completed execution receipt](/tmp/fortran-periodic-copy-pilot-20261109-01/execution.json) are retained.

## Separate CPU/GPU diagnosis

Four full production profiles compare normal candidate and be58 in both
cases, with process-tree CPU sampling and CUDA traces. All output and FFTW
checks pass. These are diagnostic profiles, not performance samples.

| Case / mode | Host API union | Host copy API union | GPU kernels | GPU copies | Host API ∩ GPU | Host API ∪ GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| channel / be58 | 1.969897 s | 0.966303 s | 0.625814 s | 0.690799 s | 1.299320 s | 1.987191 s |
| channel / candidate | 9.314829 s | 6.696167 s | 1.404997 s | 2.746787 s | 3.811201 s | 9.655412 s |
| channel-simple / be58 | 2.010767 s | 1.043119 s | 0.588464 s | 0.745027 s | 1.321004 s | 2.023254 s |
| channel-simple / candidate | 4.581861 s | 3.351781 s | 0.890231 s | 2.183238 s | 3.028178 s | 4.627152 s |

**Do not add these columns.** Host API spans overlap GPU activity.
They are not an exclusive whole-process critical-path budget. CPU leaf-sample
shares are not elapsed-time fractions, and a generated `master` symbol does
not demonstrate serial execution: the outlined CPU work spans four threads.

In channel, `cudaMemcpy3DAsync` grows from 1,204 calls / 0.856 seconds of host
spans in be58 to 22,996 calls / 6.204 seconds in the normal candidate. Device
copy events total 16,523,728,000 bytes in the candidate versus 5,298,987,008
in be58. These totals cover the full 100-step run, including initialization.
GPU kernel counts are 22,492 versus 301; the candidate count includes copy
packing kernels as well as numerical work. Diffusion, filtering and pressure
still contribute CPU work. This evidence prioritizes transfers and mixed
placement over another isolated copy-planner micro-optimization.

[Analysis and overlap definitions](/tmp/fortran-runtime-gap-profile-20261109-02/analysis.json),
[analysis script](/tmp/fortran-runtime-gap-profile-20261109-02/analyze.py), [profile receipts](/tmp/fortran-runtime-gap-profile-20261109-02/result.json).
The first attempt failed during setup, before any application execution,
because an older frozen helper capped capture at 30 seconds; it is retained
at `/tmp/fortran-runtime-gap-profile-20261109-01`. The successful run used
the current frozen helper and a declared 90-second capture bound.

## Tested planner experiment and disposition

The private patch rejects provably disjoint periodic rows or plane envelopes
using pitch greatest-common-divisor tests, retaining the old interval merge
for inconclusive cases. It analytically preserves the canonical merge work
count, so physical-copy cost features and decisions remain unchanged.

The 64 focused CPU checks and 23 CUDA checks pass. The new test compares
12,000 deterministic random shapes with an independent byte-interval oracle
and the previous merge, including its work count; it also covers overlap,
interleaving, high offsets and large row counts. An initial manually classified
fixture was actually overlapping; both implementations correctly rejected it.
That fixture was corrected and the initial failed XML remains available.

Six native/normal/private short controls pass all 20 fields, bounds, halos
and outputs. Copy bytes/counts, numerical launches, contexts and allocations
are exactly equal in both cases. The private full channel profile also has
identical copy bytes/events and 22,492 kernels. Main-thread `make_plan` leaf
share falls from 4.443% (1,748 samples) to 0.109% (39 samples). That is a
sampled hotspot reduction, not a demonstrated application speedup. Only
1.32% of the old planner samples overlap GPU activity, so GPU overlap alone
does not explain the weak untraced result.

The controlled private rebuild deliberately reused historical calibration
to isolate the runtime algorithm while preserving decisions. A production
promotion would require fresh runtime-matched calibration and regeneration.
The application result does not justify that work now. The patch and test
are isolated, and `compiler/runtime/section_batch.hpp` matches HEAD again.

[Archived patch](/tmp/fortran-periodic-copy-private-20261109-01/runtime-experiment.patch),
[test source](/tmp/fortran-periodic-copy-private-20261109-01/test_section_batch_overlap.py),
[disposition](/tmp/fortran-periodic-copy-private-20261109-01/worktree-disposition.json),
[CPU tests](/tmp/fortran-periodic-copy-overlap-20261109-02.xml),
[CUDA tests](/tmp/fortran-periodic-copy-overlap-cuda-20261109-01.xml),
[short correctness](/tmp/fortran-periodic-copy-short-20261109-01/result.json),
[profile comparison](/tmp/fortran-periodic-copy-profile-20261109-01/planner-comparison.json).

## Corrected next target

The generated scope covers only `TMarchRK3` lines 122–130: SGS and
Convection. Placement is forced sections, and the manifest explicitly says
source-issued tile tasks and scratch preparation are unpriced. This is not
qualified automatic group placement.

At RK stages 2 and 3, Convection lines 267–295 initialize `U2/V2/W2` from
`Ustar/Vstar/Wstar`. The reached native fallback retires the domain after
SGS, releasing the read-only `U/V/W` device replicas before advection.
Both two-step traces contain four such retirements. This is an intervening
CPU operation with later GPU consumers, exactly the replica-retention case
the design needs to handle. First diagnose its native transaction and prove
that those unchanged fields survive it in the actual emitted child path.

By contrast, after the first diffusion cut in simple, or the first velocity
boundary cut in channel, neither trace records another GPU launch before
the next RK stage. The final velocity accumulation currently remains CPU.
Fixing only the later child-local scalar rejection therefore cannot be
assumed to remove the application round trip. Extending retention through
that tail needs a concrete later GPU consumer or a wider storage lifetime.

[Domain, copy and stage attribution](/tmp/fortran-runtime-gap-profile-20261109-02/domain-attribution.json) is
reproducible with [this script](/tmp/fortran-runtime-gap-profile-20261109-02/analyze_domains.py); it only reads
the existing manifest and traces. Deferred-copy provenance denotes the
submission position and must not be mistaken for its original requester.

The next experiment must remove measured redundant uploads across the
stage-2/3 initializer or demonstrate a profitable complete GPU group. Use
a bounded compiled source regression and short full20/copy-ledger gate
before another expensive full generation. Keep be58 as the performance
reference and stop expanding forced partial offloading without an end-to-end
payoff. Group pricing and the complete resident pressure chain remain larger
workstreams. The full maintained suite, current natural 2,458-step correctness
and all new seven-round acceleration gates remain open.
