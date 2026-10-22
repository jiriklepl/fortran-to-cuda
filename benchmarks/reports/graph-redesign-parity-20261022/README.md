# Graph redesign: fresh common-basis comparison, 2026-10-22

The redesigned compiler meets the predeclared median-performance criterion against the original be58 SGS/direct compiler in **both** primary configurations. These are small measured improvements with overlapping variation: each case wins four of seven paired rounds, and the separate stronger `comfortable` criterion is false. This establishes the requested median parity checkpoint, not a large or statistically established speedup.

| Configuration | Timed steps | Native median | Original compiler median | Redesign median | Redesign wall reduction vs original |
| --- | ---: | ---: | ---: | ---: | ---: |
| channel-simple | 109, natural duration | 11.654672 s | 9.651285 s | 9.498941 s | 1.5785% |
| channel | 100, timing only | 13.258415 s | 11.453850 s | 11.354034 s | 0.8715% |

Each configuration has one excluded warmup and seven fresh measured rounds per mode, with cyclic mode rotation. All 48 production rows remain in [the CSV](all-timing-rows.csv); [all samples, paired ratios and inclusive quartiles](all-samples.md) are readable separately. [The analysis JSON](performance.json) retains exact binary64 sample values and the unchanged existing target evaluator results. Acceptance required candidate median/original median <= 1 independently in both complete campaigns, with no 1% goal margin, retries, filtering or performance-based early stop. Native-relative wall reductions were 18.50% and 14.36%; the separate historical 20% native target remains unmet.

Production measurements used the unchanged public complete-process timer, including startup, computation, transfers, solver, output and teardown. Timing runs have authentic combined `run.log` streams. Traced/full-field/profiler diagnostics have separate streams and do not contribute timing samples. CPU affinity was 0..11, OpenMP used four threads, and other builds/tests/calibration/profiling were held. The hardware was RTX 3060 Ti / Ryzen 5600G; the new independent calibration attested driver API 13050 and CUDA runtime 13040. Every mode used the same four verified FFTW plans.

## Correctness and actual GPU work

All three modes were freshly generated/built from the same approved application source basis. Eight standalone diagnostic processes cover native, an independent native repeat, be58 and the redesign at natural 109/2458 steps. Every run passes all 20 float64 fields, original shapes/bounds/halos, finiteness and VTK output at unchanged `atol=1e-6`, `rtol=1e-5`. Native repeats are byte-identical in every field; be58 and redesign are also byte-identical in every field in both cases. Maximum absolute errors against native are 4.243521476432477e-7 and 1.059737481143297e-6 respectively.

The final public comparison imports only the independently validated native/native-repeat/be58 controls, then **runs fresh candidate full-natural correctness again** in each case. It imports no timing samples. Its 59 actual application processes (29 simple, 30 channel) all pass the four-plan FFTW and private input checks; all public and final private postguards pass. The eight standalone natural runs are additional diagnostics, not performance samples.

Separate actual CUPTI SQLite records agree with runtime evidence for both compiler modes: simple has 328 numerical kernels, 4,398,579,712 H2D bytes and 1,375,731,712 D2H bytes; channel has 301 numerical kernels, 4,036,501,504 H2D bytes and 1,262,485,504 D2H bytes. There are zero carrier kernels. These measurements establish GPU execution; they do not show a reduction in kernel count or transfer volume relative to be58. Earlier two-step checks reduced recorded waits from 14 to 7 while retaining the same kernels and bytes.

## Implementation and identity

The follow-up adds generic safe scalar-prefix execution before lazy context creation, uses the existing shared CPU-identity cache, and fits transfer calibration using nonnegative relative least squares. Original raw calibration samples remain recorded; no runtime compatibility/error threshold was relaxed. The companion validator now requires explicit successful native-source preflight evidence: an unavailable whole-owner cost estimate cannot prove that execution is all-native. The shared application preparation utility reconstructs clean approved sources for both public launch paths; upstream caches are never patched.

The approved ELMM derivative is local commit `be38bbbec88756ef0f32d67b44dc412a71e0e6e8` from upstream `ddb426fca5b020216083f02a218292fea70c1df2`; only the approved non-PAR TOP/BOTTOM plane-average pressure reduction changes. Primary configurations retain TOP referencing and four threads. PoisFFT remains `256a2338254c08f611881b2df5e72bdfa73bc257`. [The public patch and preparation instructions](../../../containers/README.md) explain reconstruction.

The original compiler is exact `be58ab809296b97ea39d701828a2621adfc6b3dc`. The tested candidate was frozen from main parent `1a8ddc886dc4243bc62d833ef3463c68612c6f22` plus the reviewed changes: compiler tree SHA256 `57a3073369769534c1a1d57b1ae6f2ed1bdeff5499a0ea7da6aa5c3389abb401`, code SHA256 `c8d7e2f31cea5dc6625884e800ebf88759cd23da55d513d923ef786644785840`. All 668 compiler files match the audited current source. Runtime SHA256 remains `9517e2476a8182d00f52d62574c9573549439091eaca7a3d6ffc343f7590a4aa`; corrected profile SHA256 is `000a78e8ed362b62b0f6f85026210072cbf4f7dd3d706308854e0188cc826e6e`; prepared-build manifest SHA256 is `7c36bb39a9c3711c05a9685337af6f6fd160f0f1ed39ef744c0ea180621a69e1`.

The companion checkout is committed as `55ba91723749eb28269c471819e3045ee8ae507e`, parent `40bd7843fc07c0e7dae02d55c18aaff07bf75428`. [Its patch](companion-followup.diff), [commit receipt](companion-commit.json) and archived postimages preserve reconstruction because that checkout is ignored by the main repository. Its source bytes exactly match those tested before committing. Commit metadata is not retroactively substituted into historical build receipts.

## Regression evidence

[The source/test audit](source-test-audit.json) and [targeted evidence index](targeted-test-evidence.json) preserve separate, overlapping groups: host identity 47 CPU + 2 actual CUDA nodes; prefix 76 effective nodes plus 57 merged nodes; calibration 210 nodes; containers 201 then 59 final affected nodes; companion 130 nodes plus 273 subtests, followed by the separately enabled capability node passing. These counts must not be added as unique coverage. Earlier incomplete/failed prefix attempts and the original companion capability skip remain recorded alongside their successful replacements. No additional targeted check was found materially missing.

The [previous committed snapshot](../graph-redesign-snapshot-20261022/README.md) has the historical full inventory of 8,899 passes and two structural skips. That is not a claim that the entire suite was rerun for this follow-up. Current scoped checks and fresh full-application validation pass. Existing repository lint/format debt also remains; no full-lint-clean claim is made.

## Reproduction and retained evidence

[The archive manifest](archive-manifest.json) maps every member of `evidence.tar.gz` to its exact original path, size and SHA256. Every archive member was reopened and verified. It includes both unchanged raw public reports, all timing logs/run records/FFTW logs, diagnostic streams, four primary CUPTI SQLite databases, current runtime proof records, natural control/candidate records, preparation/calibration receipts, execution recipes, original commands, scoped test evidence, and independent reviews. Read the original reviewed `final-campaign-plan.json` and executed recipes after extracting the archive; they retain the original absolute paths and are evidence of the actual execution, not a path-portable installer.

[External references](external-evidence-references.json) retain identities for omitted field/shape/VTK/Nsight payloads; archived prepared-build manifests additionally authenticate large binaries and generated proofs. The package intentionally does not vendor gigabytes of application outputs. For new runs use the public [three-way workflow](../../../containers/README.md), reconstruct the approved common application basis, calibrate the actual machine, build fresh, and use a new output directory. Historical driver-bound profiles and timings must not be retagged. The earlier snapshot and its original-pressure measurements remain unchanged.

## Next work after this checkpoint

| Order | Change | Exit criterion |
| --- | --- | --- |
| 1 | Give storage identities/versioned regions a lifetime beyond lexical contexts. A read-only copy adds a valid replica; only writes invalidate overlapping replicas. Bound mirror retention by bytes and next-use cost. | CPU→GPU→CPU→GPU with unchanged data uploads once; real writes, aliases, allocation generations and host escapes synchronize exactly where required. Close the preserved residency-cut fixture before claiming general cross-context retention. |
| 2 | Schedule explicit dependency-graph tasks, including host-only work, and group movable computation by legal topological ordering. Preserve data/anti/output dependencies, observable calls and source-required ordering. | Host-only tasks collect at necessary graph boundaries; movable host steps do not force repeated device publication. No hard work-unit-count window determines legality or memory lifetime. |
| 3 | Pool allocations and synchronize at consumers; price live-byte pressure and movement over the scheduled graph. Keep unknown costs explicitly unavailable. | Actual CUPTI/runtime copy ledgers decrease on the primary cases and mixed read-only/write fixtures, with full natural correctness unchanged. |
| 4 | Fuse compatible adjacent tasks; use graph replay/parameter updates for stable residual launch sequences. Keep trivial host preparation cheap. | Lower measured launch/coordination overhead and end-to-end medians; numerical kernels and transfer bytes remain independently audited. |
| 5 | Retain one canonical graph AUTO path and minimal forced-placement diagnostics; remove superseded flags/presets only after inventorying their callers. | One maintained policy path, clear errors for removed options, focused regression coverage, and the same three-way primary-case benchmark protocol. |

The preserved cross-context read-only mirror loss and narrow-first native-hint limitation are still open; this checkpoint does not relabel them as fixed. The existing snapshot records their failing assertions and exact traffic evidence.
