# Production copy-policy comparison — 2026-11-08

The candidate was normally regenerated and publicly rebuilt from the application-qualified 2619b7f compiler basis with only the generic copy-policy files from 03c695d. It uses fresh matching calibration for runtime `6f32137c`. The later native-CALL changes and external descriptor draft are excluded. Native, be58 and 7a612e4 reuse the fresh controls built earlier today; every timing sample below is new. No private copy interposer is used.

| Case | Native (s) | be58 (s) | 7a612e4 (s) | Candidate (s) | Native speedup | Candidate vs 7a612e4 | Paired wins vs 7a612e4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| channel-simple (109 steps) | 11.586099 | 9.466523 | 16.634371 | 11.185685 | 1.036× | -32.76% wall time | 7/7 |
| channel (100 steps) | 13.031851 | 10.992937 | 11.105197 | 16.235619 | 0.803× | +46.20% wall time | 0/7 |

Four threads on CPUs 0,2,4,6; RTX 3060 Ti selected by UUID; the same approved deterministic pressure reference and four verified FFTW plans. Each case/mode has one excluded warmup and seven rotated measurements, without outlier removal or reruns. Complete production-process time includes startup, the original CPU PoisFFT solver, all output and teardown. Tracing and diagnosis are separate.

Eight short checks pass all 20 fields, bounds, halos and outputs at atol=1e-6 and rtol=1e-5. All 64 timing processes pass output comparison. The new candidate also passes natural 109/2,458-step all-field/halo/output checks against qualified native and prior-candidate references. Fresh controls have not repeated their natural qualification; this packet is preliminary and does not constitute a milestone release or a complete maintained-suite pass.

Measured gates across both cases: `{"median_ten_percent_gain": false, "native_20x": false, "native_2x": false, "robust_vs_7a612": false}`.

The candidate is **18.16% slower than be58 in simple109 and 47.69% slower in channel100**, losing all seven paired comparisons in both cases. This is a failed performance checkpoint despite passing correctness. Compared with the previous candidate campaign, both medians change by less than 1%; these separate campaigns do not establish an improvement.

The intended single-operation copy path executes **zero times** in both short diagnostics. Payload totals, numerical launch records and the 28 channel multi-operation compact copies are unchanged. The [explicit admission failure](/tmp/fortran-copy-v3-speedup-20261108-01/short-transport-assessment.json) and [original failing check](/tmp/fortran-copy-v3-speedup-20261108-01/short-transport.log) remain recorded. The private packing pilot's gain did not reach production. Future experiments must demonstrate actual path selection and an application pilot gain before another full campaign; be58 remains the fastest measured baseline.

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
- [Natural 109/2,458-step checks](/tmp/fortran-copy-v3-natural-20261108-01/result.json)
- [Distributions and paired results](summary.json)
- [Copy-policy motivation, private pilot and tests](../gpu-residency-copy-v3-20261108/README.md)
- [Earlier four-mode comparison](../gpu-residency-speedup-20261108/README.md)

The earlier private three-pair packing pilot and earlier candidate timings are separate experiments. Do not interpret cross-campaign median changes as seven paired wins. Forced sections compute remains diagnostic; calibrated AUTO physical copies do not qualify GPU source-compute pricing.

## Follow-up diagnosis (private, not production qualification)

The 98,304-byte narrow downloads are attributed to `domain_role=storage`. The domain-close observer creates a fresh execution `Context` whose `copy_costs` is empty, although AUTO transfer mode survives in `Residency`. The intended calibrated transport therefore cannot run there.

An isolated runtime build moved transport calibration to the residency owner. Both short cases pass all 20 fields/halos/outputs, but no new packing occurs: channel retirement groups mix 17 operations, including narrow faces and large wide transfers, and the whole group fails the packing threshold. A first per-region fallback also lost useful multi-operation startup observations; that failed experiment is retained.

The revised private fallback retains groups with a profitable warm startup observation and prices remaining download regions separately. In the two-step channel check it executes 60 single-operation and 29 multi-operation compact copies, versus zero and 28 in production. Both cases pass all 20 fields, bounds, halos and outputs. Physical payload totals and numerical launch records are unchanged. No source computation, mathematical tolerance or output is changed. This uses historical calibration for a modified runtime and does not establish production pricing authority.

A production-main relink was prepared for a predeclared pilot against the original candidate and be58, with one excluded warmup and three rotated rounds per case. External CPU search activity contaminated the pilot: simple partition and be58 samples inflated to 22.110 and 25.170 seconds. The owned pilot was stopped, all recorded samples were retained, and no external process was modified. **No speedup inference or production promotion follows from this interrupted pilot.**

- [Lifetime-only diagnosis](/tmp/fortran-copy-domain-cost-probe-20261108-01/result.json)
- [Failed naive partition experiment](/tmp/fortran-copy-partition-probe-20261108-01/result.json)
- [Revised short qualification](/tmp/fortran-copy-partition-probe-20261108-02/result.json)
- [Exact payload and numerical-launch comparison](/tmp/fortran-copy-partition-probe-20261108-02/ledger-comparison.json)
- [Private runtime source](/tmp/fortran-copy-partition-probe-20261108-02/scoped_runtime.cu)
- [Declared pilot protocol](/tmp/fortran-copy-partition-pilot-20261108-01/plan.json)
- [Contamination and interruption](/tmp/fortran-copy-partition-pilot-20261108-01/interruption.json)

Next acceptance gate: rerun the small pilot on idle hardware, retain the best existing baseline, and require a paired application improvement before normal generation and a full campaign. Production integration would also need domain-lifetime regression coverage, matching planner partition pricing and fresh runtime calibration; these remain pending.

## Completed private pilot and integration decision

The replacement pilot used currently idle hardware, the declared same CPUs/GPU, one excluded warmup and three rotated measurements for each variant/case. All 24 processes pass every output comparison. The observed external process family stayed below the predeclared 0.5-core guard (maximum 0.223); wall times nevertheless remain variable. This is a small diagnostic, not the seven-round research protocol.

| Case | Prior candidate median | Private partition median | be58 median | Change vs candidate | Wins vs candidate / be58 |
| --- | ---: | ---: | ---: | ---: | ---: |
| channel-simple | 13.297212 s | 13.473925 s | 11.585342 s | +1.33% | 2/3 / 0/3 |
| channel | 17.295553 s | 16.563960 s | 12.333101 s | -4.23% | 2/3 / 0/3 |

The partition experiment fails the two-case acceptance gate and is **not integrated**. Every measured private sample loses to be58. All samples, warmups, paired ratios, input hashes and output checks remain in the [pilot result](/tmp/fortran-copy-partition-pilot-20261108-02/result.json) and [declared protocol](/tmp/fortran-copy-partition-pilot-20261108-02/plan.json).

The production source changes only the confirmed ownership defect: the residency domain now retains physical-copy calibration across borrowed-context detachment and final publication. Children can replace or invalidate that model; separate domains remain isolated. Compute costs and execution plans remain context-local. The stale copy-ABI error message now reports version 3. No new partition policy, calibration override or application-specific dispatch is added.

Validation: 71 CPU checks and 45 actual CUDA checks pass, including ten new CPU and five new CUDA cases. Related planning/profile/source checks yield 100 passes plus one companion-import setup failure; rerunning that exact test with the companion path passes. The external descriptor draft remains uncommitted and unchanged. Focused lint and whitespace checks pass. These are overlapping scoped checks, not the full maintained suite.

Runtime identity is now `d0e179c4821005b8fe9d39eab1f378e4ca4b67a5d4da0ed1c9dffd282b978d52`. Existing profiles/artifacts require regeneration and matching calibration before application promotion. This ownership correction has no newly qualified end-to-end speedup.

The [transfer attribution](/tmp/fortran-copy-partition-pilot-20261108-02/transfer-attribution.json) locates all 146,316,352 simple-case download bytes and 143,696,704 of 146,316,352 channel download bytes at domain retirement over initialization plus two steps. Retirement uses the last procedure attribution, not the producer of every byte. The next performance target is retained coherent domains across the necessary native diffusion/boundary calls, with exact host effects and fewer full-field publications. Further copy packing does not reduce these payload totals.

Test evidence: `/tmp/fortran-domain-copy-cost-cpu-20261108-01.xml`, `/tmp/fortran-domain-copy-cost-cuda-20261108-01.xml`, `/tmp/fortran-domain-copy-cost-related-20261108-01.xml`, `/tmp/fortran-domain-copy-cost-related-20261108-02.xml`.


## Current-tree retention follow-up: no traffic reduction

The frozen `4d944f1` tree plus the preserved external joined-descriptor draft
normally generates 294 artifacts and passes the strict package reader. Fresh
scoped/physical calibration matches runtime `d0e179c4`; public production build
and all-field diagnostic relink pass. No private copy partition is included.

Six new short processes (native, prior candidate, current candidate in both
cases) pass all 20 fields, halos, bounds, original outputs and all four FFTW
plans. Copy bytes, copy event counts and numerical kernel inventories are
**exactly unchanged** from the prior candidate:

| Initialization plus two steps | H2D bytes | D2H bytes | Copy events |
| --- | ---: | ---: | ---: |
| channel-simple | 134,664,192 | 146,316,352 | 148 |
| channel | 184,158,208 | 146,316,352 | 730 |

This fails the declared transfer-reduction gate. No new application timing or
speedup claim follows. Generated diffusion and velocity-boundary replacements
are byte-identical to the previous candidate; only `filters::filter` changes its
top-level return-route inventory. The next diagnosis must exercise the real
native-group descriptor consumer before another application regeneration.

- [Frozen generation and strict-reader result](/tmp/fortran-retention-current-20261108-01/result.json)
- [Fresh calibration](/tmp/fortran-retention-calibration-20261108-01/qualification.json)
- [Public build and owned-process cleanup](/tmp/fortran-retention-rk-build-20261108-01/execution.json)
- [Six short controls and failed transfer gate](/tmp/fortran-retention-short-20261108-01/result.json)
- [Artifact and route comparison](/tmp/fortran-retention-current-20261108-01/retention-comparison.json)


The bounded consumer diagnosis finds a concrete earlier failure: normalization
from source nodes to graph IDs removes ordinary comments, while the joined-team
issuer correctly requires the exact complete original sequence. All four actual
ELMM groups therefore lost their descriptor proof. A nine-line correction
recovers the original group using the existing source matcher and rechecks its
unchanged graph selection before requesting the same proof. External draft edits
remain preserved; this correction and its regression are currently uncommitted.

The initial 15 focused checks pass, and the new positive regression fails against
the previous frozen source as expected. With the correction, all three actual
velocity-boundary groups have complete native segment effects. Diffusion's
allocation proof now succeeds but its indexed wall-model record payload accesses
remain unsupported in that consumer. No new full application regeneration or
speedup claim follows from this source-level result.

- [Original exact-consumer rejection](/tmp/fortran-native-descriptor-consumer-20261108-02/result.json)
- [Corrected exact-consumer diagnosis](/tmp/fortran-native-descriptor-consumer-20261108-03/result.json)
- [Correction relative to the preserved external draft](/tmp/fortran-native-descriptor-comments-20261108-01/root-fix.patch)
- [Focused checks](/tmp/fortran-native-descriptor-comments-20261108-01/focused.xml)
- [Regression against the old source, expected failure](/tmp/fortran-native-descriptor-comments-20261108-01/before-fix.xml)

Final scoped validation passes 96 unique source checks: the original 11 joined-
descriptor cases, five comment/normal-emission cases and 80 related checks.
The initial nonexistent test-file invocation ran no tests and is retained; the
corrected related selection passes once. Lint and whitespace checks pass. This
is not a full maintained-suite or application performance qualification.
[Validation record](/tmp/fortran-native-descriptor-comments-20261108-01/result.json).


## Full native-consumer check supersedes the summary-only diagnosis

The preceding summary-only query did not enable the metadata/read options used
by the actual native fragment consumer. A first attempt to exercise that
consumer also incorrectly reused the parent's inline-source mapper; its original
selection rejection is a diagnostic setup failure. Both records remain intact.

The corrected bounded query creates the original child's `InlineRegions`, uses
`fragment(..., kind="joined native OpenMP", native_metadata=True,
native_host_reads=True)`, and retains all frozen source guards. It accepts the
complete second-order diffusion group, including all 18 wall-model record arrays,
and all three velocity-boundary groups. Fourth-order five-point diffusion still
rejects its local `fl` allocation. These are source admission results, not
application transfer results.

A proposed broader native-group transaction interface was isolated under `/tmp`
and removed from both worktrees before generation. No new runtime, reader schema
or coherence path is included. One new normal generation uses the existing
matching runtime calibration and the narrow comment-selection correction.

- [Exact existing-consumer result](/tmp/fortran-comment-fix-original-fragment-20261109-02/result.json)
- [Unpromoted broader interface prototype](/tmp/fortran-native-host-group-prototype-20261109-01/status.json)
- [Frozen current generation plan](/tmp/fortran-retention-comments-20261109-01/plan.json)


## Narrow retention correction: completed pilot still regresses

Normal generation from frozen `4d944f1` plus the preserved external joined-
descriptor draft and the narrow comment-recovery correction passes the strict
reader. The public build and all-field diagnostic relink pass. The production
binary is `cd3684da710eacf294cb183a9a8f80d412ce95b87de4d255b706f61c4e05ef6a`.
It uses the matching `d0e179c4` runtime calibration, forced-sections computation,
AUTO physical copies and the original CPU PoisFFT solver. The comparison includes
changes since the prior timed copy-v3 candidate; it does not isolate the timing
impact of the comment-recovery hunk alone.

All six new short native/prior/current processes pass all 20 fields, bounds,
halos, original outputs and the four verified FFTW plans at unchanged tolerances.
Physical payloads, copy counts, allocations and numerical launches are unchanged
in both cases. Channel's host-commit notifications increase from 514 to 622 over
initialization plus two steps. No speedup follows from that event-count change.

The first small pilot stopped at its predeclared external CPU limit
(0.500118 cores averaged over one process). Its 11 completed samples remain
recorded and the entire attempt is invalid for performance acceptance. A fresh
10-second idle check measured 0.167 busy cores. One new complete pilot used the
same 0.5-core guard and reused no samples. All 24 processes passed every original
output comparison; maximum external CPU activity was 0.423 cores. Each variant
has one excluded warmup and three rotated complete-process measurements in each
primary case. No build, test or profile ran concurrently.

| Case | Prior candidate median | Current median | be58 median | Change vs prior | Change vs be58 | Paired wins vs prior / be58 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simple109 | 11.475700 s | 11.392340 s | 9.983604 s | -0.73% | +14.11% | 2/3 / 0/3 |
| channel100 | 16.843643 s | 17.526326 s | 11.823843 s | +4.05% | +48.23% | 0/3 / 0/3 |

The two-case median gate fails. The candidate loses every paired comparison with
be58. It is not promoted and does not warrant another full timing campaign.
This three-round pilot does not satisfy or replace the seven-round research
protocol. No new full maintained-suite or natural 109/2,458-step qualification
is claimed for this candidate.

| Case / mode | Three measured wall times (s), in round order |
| --- | --- |
| simple / prior | 11.511490, 11.358907, 11.475700 |
| simple / current | 11.392340, 11.621821, 11.355209 |
| simple / be58 | 9.616170, 10.012758, 9.983604 |
| channel / prior | 17.011206, 16.843643, 16.404736 |
| channel / current | 17.625132, 16.934070, 17.526326 |
| channel / be58 | 11.403492, 11.823843, 12.043530 |

The trace now supplies the missing runtime gate. Source admission removes two
second-order diffusion return cuts, but ordered-definition preflight still
returns `uninitialized_read` 16 times in simple and 70 times in channel. Those
failures and the domain closures match by domain. The first failure is at
`momentumdiffusion_nobranch_2ord` (original lines 188–286) for simple and
`bounduvw_do_BC` (432–1083) for channel. This does not establish that the original
application reads undefined data: the declared region may be too broad or its
initialization evidence may be incomplete. Identify the exact rejected region
and its defining source work before changing the contract. Keep the runtime
correctness check intact. A compiled reproducer must show retained coherence
and an actual copy-ledger reduction before another full application generation.

- [Generation and frozen source guards](/tmp/fortran-retention-comments-20261109-01/result.json)
- [Source return-route comparison](/tmp/fortran-retention-comments-20261109-01/retention-comparison.json)
- [Public build and cleanup](/tmp/fortran-retention-comments-build-20261109-01/execution.json)
- [Six short controls and physical ledger](/tmp/fortran-retention-comments-short-20261109-01/result.json)
- [Reached preflight failures and domain closures](/tmp/fortran-retention-comments-short-20261109-01/reached-fallbacks.json)
- [Invalidated first pilot, all samples retained](/tmp/fortran-retention-comments-pilot-20261109-01/result.json)
- [Fresh pilot protocol](/tmp/fortran-retention-comments-pilot-20261109-02/plan.json)
- [Fresh pilot, all samples and paired ratios](/tmp/fortran-retention-comments-pilot-20261109-02/result.json)
- [Failed performance assessment](/tmp/fortran-retention-comments-pilot-20261109-02/assessment.json)


## Reached native may-access correction under qualification

The new compiled reproducer demonstrates the missing distinction: a conservative
original host may-read is not a requirement to define its entire envelope. Before
the correction, two GPU kernels surrounding a nontrivial original joined host group
produce correct fields but close two domains and re-upload 1,728 unchanged interior
bytes. The corrected compiler uses the existing original-host transaction and one
shared CALL/group source proof. Its typed proof transport is version 2; the matching
reader checks the completion, original selection, effects and region identity.
Runtime source and ABI are unchanged.

Six final CUDA/hidden-device checks pass for whole envelopes, precise regions and
real checked child-argument forwarding. Every field, halo and bound matches native
execution. The whole-envelope and child cases retain one domain and avoid the
1,728-byte interior re-upload. The precise case re-uploads only 24 modified interior
bytes; untouched device regions and the read-only replica survive. Unknown holes
remain unknown and numerical reads retain their strict definition checks.

All 81 focused source/reader checks pass. Earlier failed fixtures and setup attempts
remain recorded. The bare ELMM child diagnostic lacks the normal caller's capture
mapping; its missing-formal-storage errors do not establish how normal ModuleOwner
admission behaves. One frozen normal application generation is running. Its build,
short controls and application performance result remain outstanding. No new speedup
or full maintained-suite claim follows from these focused results.

- [Qualification and source hashes](/tmp/fortran-native-host-group-qualification-20261109-01/result.json)
- [Expected failing CUDA regression before the correction](/tmp/fortran-native-group-before-20261109-01.xml)
- [Six final CUDA controls](/tmp/fortran-native-group-final-20261109-01.xml)
- [81 source/reader checks](/tmp/fortran-native-host-group-source-20261109-07.xml)
- [Frozen normal application generation](/tmp/fortran-retention-may-host-20261109-01/plan.json)
