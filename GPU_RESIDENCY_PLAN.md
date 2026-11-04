# Generic GPU residency and the ELMM 20× research target

This is the implementation plan approved on 2026-10-22. It supersedes the
conceptual proposal: its projected payoff table was not measurement evidence.
The original proposal is preserved at
`/tmp/fortran-residency-implementation-24/GPU_RESIDENCY_PLAN.original.md`.
Historical reports and timing controls remain unchanged.

## Current integration checkpoint: 2026-11-04

The generic guarded-condition and immutable-container changes pass
[381 focused checks](/tmp/fortran-residency-generic-integration-20261104-01/qualification.json).
The [fresh maintained campaign](/tmp/fortran-maintained-current807-c3-native-full-14/qualification-current-checkpoint.json),
including the owned emission implementation below, passes all 12,942 selected compiler cases in 48 disjoint whole-file batches;
this is not a single pytest invocation. Both mandatory CUDA IEEE cases pass.
The [paired pipeline run](/tmp/fortran-maintained-current807-c3-native-full-14/execution/pipeline/initial-01/completion.json)
passes 4,502 tests with exactly 29 declared optional skips. The compiler's 1,309
marked CUDA exclusions include the nine new guarded-condition modes already
covered by the separate device fixture; the paired exclusions remain 18.
All source/tool/profile guards hold and all owned test groups exit.

Later `ELSEIF` predicates retain lazy, once-only evaluation and publish their
required host views only when reached. Read-only publication keeps the device
replica current. Generated packages match all 243 files (240 artifacts and three
manifests) from the qualified 12-case device fixture; this integration reuses
that evidence rather than claiming another GPU run. Exact builtin tuple and
frozenset containers no longer save redundant membership snapshots, while every
mutable descendant, source attachment and owning identity edge remains checked.
Builtin-type checks use identity, preventing subclass spoofing. The actual-parent
[census](/tmp/fortran-parent26-retained-state-diagnostic-01/qualification-observation.json)
counted 40,344,451 saved identity slots, including 37,280,084 for those immutable
containers. This motivates the change; application benefit remains unmeasured.

[Normal parent attempt 27](/tmp/fortran-parent-graph-current-27/result.json)
also ends at the unchanged 2,400-second target limit, with the same original
SGS→Convection interval, runtime `807bdd62…` and CUDA 13.5 profile `d32bc253…`.
All 1,524 source and 1,996 input guards hold, and the controller and worker group
exit. No package, application or GPU execution is accepted. The terminal stack
is in the third full `StorageAdmission._check()` within one `admits_many()` call.
Each check still traverses whole-plan state and retained attachments; removing
immutable-container snapshots has not resolved that repeated-work structure.
The next bounded change must preserve callback boundaries and validation while
removing duplicate checks, followed by normal compilation within the same cap.
No current-runtime ELMM application correctness or performance is qualified.

The [admission consolidation](/tmp/fortran-admission-single-fence-20261104-01/qualification.json)
passes 139 focused cases, including 15 new callback/table-integrity regressions
and nine existing full-artifact equivalence checks. It keeps the existing full
check once after iterable callbacks, followed by exact builtin table and anchor
validation before callback-free lookup. Entry/exit, source, lifetime, dependency
and public validators remain unchanged. One-binding and 2,048-binding controls
both require one full sweep per membership query. [Normal attempt 28](/tmp/fortran-parent-graph-current-28/result.json)
also reaches the unchanged 2,400-second target limit. All 1,525 source and 2,019
input guards hold; the controller and worker group exit without a package,
application or GPU execution. Its terminal stack is in the sole remaining full
retained-attachment scan. The admission consolidation is correct but insufficient
to remove the whole-plan-per-unit scaling problem. The fresh maintained campaign
above passes, with all source files and process groups independently audited;
current-runtime application correctness and speedup remain unqualified.

Owned primitive emission inputs now separate callback-bearing preparation from
publication of finished text. Full semantic checks remain at batch boundaries;
unit-local canonical slots avoid a whole-plan scan for each prepared unit. The
existing generation transaction keeps atomic adoption and rollback, with sealed
output checks across final validators and cache adoption. The
[component qualification](/tmp/fortran-owned-emission-image-20261104-01/qualification.json)
reconciles 199 distinct passing checks, including hierarchical children and
module companions, exact output/manifest equality and strict-package validation.
This is a reconciled focused gate, not one clean 199-test invocation; earlier
failures and their corrections remain recorded. For 1/7/64 real flat units,
full validation calls are 5/5/5, versus 7/13/70 on the old path. Hierarchical
1/7-unit controls also stay at five checks. Callback poisoning, source/slot and
transaction-marker rejection, exact ticket consumption and preparation,
publication and cache rollback pass. The fresh maintained campaign above covers
these exact source bytes; it establishes no application timing improvement.

[Normal attempt 29](/tmp/fortran-parent-graph-current-29/result.json) stops before
legacy emission because the selected owner cannot yet use the owned-input path.
Its capability report identifies 103 supported ordinary units and 13 unsupported
combined OpenMP numerical-team units across 52 members. No other owner blocker
is reported. The attempt finishes in 1,084.735 seconds, with all 1,528 source and
2,059 input guards held and both process groups exited. It produces no accepted
package or application execution. The next separate slice ports the existing
team emitter's finished patches into primitive inputs, preserving its original
participation, failure and copyback behavior. It also hardens capture of the
final-publication guard before callbacks. Those changes need their own tests and
normal-parent acceptance; the maintained result above does not qualify them.

Fresh historical controls require a matching original-runtime calibration too.
The first fresh `7a612e4` control used its historical CUDA 13.4 profile and
correctly fell back to CPU under CUDA 13.5. That failed coverage gate remains
recorded. The [fresh original-runtime calibration](/tmp/fortran-7a612-runtime9517-cuda135-calibration-01/qualification.json)
uses unchanged samplers and policy, issuing profile `e2a1cd94…`. Fresh control
builds pass. The [six-case scientific gate](/tmp/fortran-current807-cuda135-controls-02/control-correctness-04/qualification.json)
and [independent audit](/tmp/fortran-current807-cuda135-controls-02/control-correctness-04/peer-review-terminal.json)
pass native, be58 and 7a612e4 on both two-step cases: all 20 fields, bounds, halos,
6/10 outputs, direct application return codes and four FFTW plans. Both GPU
controls retain exactly seven SGS kernels. Fresh AUTO additionally executes 90
Convection-owner kernels in channel-simple; channel has no additional kernels.
The corrected observer recognizes only declared compiled Convection owners and
catalog context. Those additional launches retain coarse owner attribution;
historical native-operation tags do not identify an exact GPU implementation.
Earlier harness failures and the SGS-only-total assertion failure remain recorded.
These are short control checks, not natural-run or performance qualification.

The entries below preserve earlier checkpoints and their limitations.

Build06 passes all four two-step GPU/hidden-GPU cases and both natural runs
(109 and 2,458 steps), including all 20 fields, bounds, halos and all 6/114
output files at `atol=1e-6`, `rtol=1e-5`. The
[natural-run receipt](/tmp/fortran-startup-application-natural-06/qualification.json)
qualifies runtime `6d8484fe…` with CPU PoisFFT. It does not qualify the runtime
changes introduced in `1295d77` or the private shared-parent implementation.

Two regressions in that external increment are corrected: automatic local arrays
must publish before a reached host boundary unless their values are proved dead;
scratch growth must match the capacity used by source quotes and the reference
runtime. The local-array regression and all 32 affected source tests pass.
The existing GPU quote gate fails two modes before the scratch fix and passes
all three afterward. Arena reuse and the async allocator remain. All 109
selected reference-runtime tests pass for the retained region-limit and
continuation changes, scratch handling, direct transport and source quotes;
six marked CUDA cases are outside that selection.

The current runtime is `807bdd62…`. Its CUDA 13.4 calibration issued profile
`47031362…`; that [receipt](/tmp/fortran-current-runtime-calibration-807bdd62-01/qualification.json)
remains historical evidence. The environment now supplies CUDA 13.5 and the old
toolkit is absent. A [fresh calibration](/tmp/fortran-current-runtime-calibration-807bdd62-cuda135-01/qualification.json)
passes in 250.81 seconds on the same GPU with four native threads and issues
profile `d32bc253…`. All 1,182 input guards hold, all owned worker groups exit,
and physical-copy, structural source-submit and native-source cost models are
available. Legacy scalar-source pricing remains explicitly excluded. This is a
new measurement, with no old coefficients reused. Current-runtime application
correctness and performance remain unqualified. The earlier fresh native, be58
and 7a612e4 comparison controls passed focused correctness checks.

The maintained compiler checkpoint `705efb5` has the following verification.
The original maintained compiler invocation fails and remains recorded. Its
stale region/continuation fixtures and earlier diagnostic expectation have been
corrected; separate whole-module follow-ups also resolve the affinity, native
fixture-window and sanitizer-environment issues. All 12,515 original cases have
passing latest results; these do not constitute a clean full invocation. The
native Python 3.13.11 environment preserves the
existing dependencies while supplying Linux PIDFD support and avoiding a global
C++ preload. The clean pipeline rerun passes 4,406 tests with 29 declared optional
skips. The fresh integrated compiler selection completes with 12,523 passes and
one test-only failure: temporary analysis objects can reuse an address while the
test retains only their `id()`. The repaired test keeps the original objects
alive and preserves its identity/count assertions; all eight tests in that
module pass. [Latest coverage](/tmp/fortran-latest-current-12524-reconciliation-20261102-01/qualification.json)
is therefore 12,524 passing cases, with zero skips;
the [original full invocation](/tmp/fortran-maintained-current807-c3-native-full-03/execution/compiler/completion.json)
remains recorded as failed, not relabelled as a clean invocation. The paired
reader is committed as `9d0386b`; its import-only test repair passes all 290
module cases and Ruff after the full pipeline check.

The actual RK parent attempt [17-02](/tmp/fortran-parent-graph-current-17-02/result.json)
declines at the guarded `mean_UVW` call: its joined fixed-array reduction lacks
the lifetime evidence needed for safe reentry. The integrated fix reuses the existing
graph-owned association receipts; it does not grant payload access to the
unproved `Utype` allocation. Its 122 CPU tests and three existing CUDA retirement
controls pass. The [fresh parent attempt 18](/tmp/fortran-parent-graph-current-18/result.json)
passes that barrier, then declines at the scalar-OUT reduction in `mean_TKE`
(`linearforcing.f90:171`). All source guards hold; no package or application is
emitted. The subsequent scalar-formal retirement extension passes 658 CPU cases, two
fixture checks and three [CUDA modes](/tmp/fortran-scalar-formal-retirement-gpu-02/gpu-01/qualification.json),
including returned-scalar consumption, masked/unmasked host work, both guards,
full fields/halos and exact copy ledgers. Its [fresh maintained compiler selection](/tmp/fortran-maintained-current807-c3-native-full-04/execution/compiler/completion.json)
passes all 12,558 selected cases in one clean run, with no failures, errors,
skips or recoveries. All 1,502 source guards hold and owned process groups exit.
The 1,279 marked CUDA cases are outside this CPU selection; the new three-mode
device control passes separately. A current-runtime application build and strict
natural runs remain subsequent gates.

The pipeline observer fix is committed as `5cd8ae1`. It retains the original
catalog owner when authenticating serial inline fallbacks, whose existing
completion record intentionally omits a procedure. Exact source, worker, span
and checked-route validation remain required. Its [whole pipeline run](/tmp/fortran-inline-fallback-catalog-fix-20261102-01/whole-pipeline-01/completion.json)
passes 4,427 tests with 29 declared optional skips and no failures; saved actual
hidden/traced logs and forged-record controls also pass. Compiler/runtime
execution is unchanged by this observer fix.

The [scalar-qualified parent retry 19-02](/tmp/fortran-parent-graph-current-19-02/result.json)
passes `mean_TKE`, then declines at the guarded Rayleigh-damping call in
`sponge.f90:618`: ownership/lifetime evidence for its coefficient and profile
components is incomplete. All 1,338 source and 380 external guards hold; no
package or application is emitted. The next diagnostic work inventories related
native component-storage barriers before another full parent attempt.

The association receipt change is now applied after 288 affected CPU cases, two
native/source fixture checks and three device modes pass. It reuses existing
read-only declaration and component checks, without GPU layout or initialization
authority. The [next parent attempt 20](/tmp/fortran-parent-graph-current-20/result.json)
clears damping and declines at the original diffusion team in
`momentum_diffusion.f90:862–1214`: the local allocatable shared `Fl` flux scratch
has no admitted storage/definition facts. Its tiled loops are also rejected by
the private-scratch alternative; the whole numerical team reports a mapped-span
limitation across include origins. These are separate lifetime, definedness and
device-lowering questions. No scratch facts are fabricated, and no application
package or performance result is accepted from this attempt.

The [fresh compiler selection for the association increment](/tmp/fortran-maintained-current807-c3-native-full-05/execution/compiler/completion.json)
passes all 12,586 selected cases in one clean invocation, with no failures,
errors or skips and all 1,504 source guards intact. The 1,282 marked CUDA cases
remain outside this CPU selection; its new three-mode device fixture passes
separately. Removing trailing spaces from the imported Fortran test fixture
preserves Python/Fortran syntax trees and line counts; all 28 affected CPU
fixture checks pass again. The runtime and calibration identities are unchanged.
The new `benchmarks/evaluate_speedup.py` results are exploratory single samples.
The approved seven rotated measurements, full-process timing and strict
correctness gates still govern performance claims. The 10%, 2× and 20× targets
remain open.

The ordinary multi-DO native-team increment is now applied. It retains the exact
original OpenMP join and descriptor-team receipts for shared local scratch, and
classifies original read-only primitive fields of TARGET record arrays only
inside that whole-group retirement context. It grants no GPU payload, layout,
alias, capture or initialization authority. Authenticated native joins use the
existing 65,536-node source bound while retaining every statement; ordinary work
and numerical admission keep their existing 256-operation limits. The original
function traversal remains available when no routine execution index exists.

The [component qualification](/tmp/fortran-ordinary-multido-retirement-gpu-01/qualification-final.json)
passes 862 distinct cases: 857 affected CPU tests, two fixture controls and three
CUDA modes. The fresh actual-source audit admits all three diffusion groups,
retaining 486/204/198 statements, with no escapes or fabricated scratch facts.
The device fixture checks full fields, halos, aliases, record canaries, original
execution counts and physical-copy ledgers: a skipped native group preserves
the domain without intervening copies; a reached group publishes the required
fields before native execution and reopens coherently. Failed intermediate
function-index tests and the first sandboxed CUDA run remain recorded. Runtime
and calibration identities are unchanged. The
[parent retry 21](/tmp/fortran-parent-graph-current-21/result.json) clears diffusion,
then declines while validating the original `ResistanceForce` call at
`dynamics.f90:425`: its lifecycle asks for stable-storage/definition facts for
`filters::filter_ratios`. All 1,345 source and 1,786 input guards hold, with no
fallback search or application emitted. The
[fresh full compiler selection](/tmp/fortran-maintained-current807-c3-native-full-06/execution/compiler/completion.json)
finishes with 12,679 passes, 11 failures and two skips. The failures expose a
second indexed-lookup fallback bug in read-only allocatable input checks; a
missing lexical-parent index entry raises instead of returning an empty result.
The two native-marked tests that dynamically request CUDA skip in the sandbox.
All guards hold and owned process groups exit. This is a failed invocation;
repairs and a fresh run with CUDA access remain required. This component result
does not qualify current-runtime ELMM correctness or an application speedup.

The applied successor restores the original occurrence walk when the parent index
has no entry; all 70 affected forwarding/input/index cases pass. It also reuses
the storage epoch's managed-resource and native-reservation inventory in
same-family CALL validation, so scalar-element backing reservations require
their original permits instead of invented managed capture facts. All 309
affected compiler cases, two fixture controls and three CUDA modes pass, with
mutable coefficient-table inputs, full fields/halos and exact copy ledgers.
The paired reader now recognizes already-validated native reservation roots in
implicit child forwarding; its nine focused controls pass. The
[complete paired pipeline gate](/tmp/fortran-conditional-native-opaque-reader-fix-05/whole-pipeline-01/completion.json)
passes 4,435 cases with the exact 29 declared optional skips. All 4,464 selected
node IDs, source guards and owned-process exits are verified. Earlier stopped
snapshots, the test-only terminal-blank-line correction and the restored genuine
Git test fixture remain recorded. A final import-order-only cleanup for the new
paired test passes that repository's Ruff configuration; the import set and
non-import syntax/positions are unchanged, and all eight affected tests pass
again. The
[combined qualification](/tmp/fortran-same-family-resource-role-gpu-02/qualification-final.json)
keeps its 384 main component cases separate from the paired-suite totals.
The [fresh full compiler run 07](/tmp/fortran-maintained-current807-c3-native-full-07/execution/compiler/completion.json)
retains all previous cases and passes all 12,711 selected compiler cases in one
clean invocation, with zero failures, errors or skips. All 1,512 source guards
hold and owned process groups exit. The two native CUDA diagnostics that
previously skipped now pass; 1,288 marked CUDA cases remain outside this selection,
with the new component CUDA modes qualified separately.
[Parent attempt 22](/tmp/fortran-parent-graph-current-22/result.json) clears the
missing-facts failure, then declines at the same original native CALL while
checking a lexical caller that inherits its parent's resource-slot namespace.
All 1,348 source and 1,826 input guards hold; no package or application is emitted.
Moving this check later is insufficient unless final source-authenticated CALL
projection also supports the inherited namespace. A small reproduction confirms
the same rejection at final epoch construction, and a flat serialized control
exposes missing transitive native-reservation use in the reader. These remain
explicit integration limitations. Current-runtime ELMM correctness and speedup
remain unqualified.

The inherited-namespace increment is now applied. Argumentless internal routines
retain their actual lexical storage namespace and original CALL receipts;
they keep `parameterized=false`. Source projections use version 2 when that
relation is present, and returned-domain contracts use version 4 for lexical
child routes. Native reservations and managed fields used only by descendants
follow the same checked CALL graph. Storage admission detects changes to the
retained receipts, including in-place record edits. An original, unlabeled
`CONTINUE` now uses the existing native effect proof without retiring residency.
The runtime, ABI and calibration profile are unchanged.

The [fresh complete compiler selection 08](/tmp/fortran-maintained-current807-c3-native-full-08/execution/compiler/completion.json)
passes all 12,737 selected cases, retaining all 12,711 previous cases and adding
26. There are zero failures, errors, skips or recoveries; all 1,515 source guards
hold and all owned process groups exit. The 1,294 marked CUDA cases remain
outside this selection. The separate
[device gate](/tmp/fortran-inherited-lexical-gpu-03/qualification.json) passes six
CUDA cases and two serialized-package controls. Both GPU variants match full
native fields, halos, bounds and runtime coefficient values. The read-only
variant, including `CONTINUE`, retains one domain with zero intermediate payload
copies. The writing native CALL publishes and retires, transferring `8*n` bytes
in each direction before the consumer. Cold and zero-trip paths inspect no
unallocated storage; GPU-hidden execution has no physical GPU events.

The [paired reader qualification](/tmp/fortran-inherited-lexical-namespace-fix-01/qualification-reader-final.json)
records 4,476 passing tests and the exact 29 declared optional skips. Its single
pytest process succeeds. The original controller remains recorded as failed:
its JUnit mapper misparsed six parameter names containing `::`. Two independent
audits using the pinned pytest address mapping establish a bijection between all
4,505 selected nodes and raw results, with no missing, extra or duplicate cases.
This is an audited clean pytest invocation, not a clean controller invocation.

[Parent attempt 23](/tmp/fortran-parent-graph-current-23/result.json) clears the
previous namespace rejection but fails later with `IndexError` in edit-boundary
validation of an empty helper body. All 1,515 source and 1,847 input guards hold;
there is no fallback search, emitted package or application execution. A separate
configured-source inventory identifies `wallmodels::updatesurfacetemperatures`
as genuinely empty, with six read-only array formals. The empty-body repair
remains a subsequent private increment. Current-runtime application correctness,
performance comparisons and the 10%, 2× and 20× checkpoints remain unqualified.

The empty-callee increment is now applied. A child without executable statements
must match the captured original empty body before its nonexistent executable
interval is skipped. Empty executable selections still reject. Generated ENTRY
controls use the original `CONTAINS` or `END SUBROUTINE` as their insertion
anchor; header and anchor checks reject shared physical source lines. Original
entry, specification, CALL, return and lifetime checks remain required. An audit
of the exact configured ELMM sources passes the selected root and all 51 child
edit-boundary checks, including the genuine six-input empty helper.

Ordinary numerical descendants now propagate implicit managed uses through the
same fully checked returned-CALL relation. All source, descriptor, slot and
canonical associations are checked before use obligations are discharged.
Unused actuals, unrelated siblings and cyclic or disconnected relations grant
no use. Duplicate ordinary-use seeding in the typed projection helper is removed;
typed field-slot and varying-association transport remains. This adds no runtime,
ABI, manifest schema, capture or initialization authority.

The [fresh complete compiler selection 09](/tmp/fortran-maintained-current807-c3-native-full-09/execution/compiler/completion.json)
passes all 12,760 selected cases with zero failures, errors, skips or retries.
It retains all 12,737 previous cases and adds 23. All 1,518 source and 118 external
guards hold, and every owned process exits. The
[fresh paired selection](/tmp/fortran-empty-member-edit-boundary-fix-01/qualification-reader-final.json)
has 4,502 passes and the exact 29 declared optional skips, with complete coverage
of all 4,531 selected nodes. Both pytest and its controller succeed. The 1,300
marked compiler CUDA cases remain outside its selection; six new
[CUDA cases and two native controls](/tmp/fortran-empty-member-gpu-01/qualification.json)
pass separately on the same frozen source. Direct and nested empty helpers match
full native fields, bounds and defined scalar values, retain producer/consumer
domains with zero intermediate copies, and preserve cold, zero-trip and hidden
execution. Runtime `807bdd62…` and profile `47031362…` remain unchanged.

[Parent attempt 24](/tmp/fortran-parent-graph-current-24/result.json) reaches its
40-minute target deadline during numerical-unit emission. All 1,518 source and
1,867 input guards hold; no package, fallback search or application is accepted.
The timeout stack shows a storage-membership query taking the full validation
path through the epoch, returned CALL coverage and native source receipts.
That stack identifies a cost to investigate; it does not attribute the entire
interval to that operation. A small reproduction has `position_epoch=None`;
the existing admission fast path excludes its hierarchical owner and returned
contract. Extending that eligibility requires complete attachment and policy
fences. A separate bounded late
compiler profile preserves the failed attempt and grants no compiler verdict or
artifact acceptance. Current-runtime application correctness and performance
remain unqualified.

The [bounded late profile](/tmp/fortran-parent24-compiler-late-profile-01/qualification.json)
captures 149.91 seconds with all 3,396 inputs unchanged. Intrinsic-source
attachment checks account for 42.99 exclusive seconds in that window. The
observed epoch and returned-CALL validation spans overlap; the profile supplies
neither a compiler verdict nor an application timing result.

The hierarchical admission extension is now applied through the existing
`StorageAdmission` mechanism. Exact sealed selected-return hierarchical owners
retain the original full validators at entry and normal exit. Cached membership
requires unchanged selected-root, draft, graph, descriptor-consumer, source CALL,
handoff and native-preservation attachments. Other unsupported lifetime families
keep full checks. Retained traversal stops at source-service boundaries while
source-session, dependency, policy and live-input checks remain active.

All 117 focused cases pass. Five cases emit byte-identical complete output
dictionaries, including manifests, proof identities and telemetry, with caching
enabled or disabled; four late atomic-task mutations reject. These comparisons
use normal generation transactions on the same parsed source authority without
stripping fields. Scoped fixtures perform two full storage validations, and
retained payload grows with the selected graph. Actual ELMM compilation speedup
is not established by those fixture counts.

The [fresh maintained qualification](/tmp/fortran-hierarchical-storage-admission-final-maintained-qualification-01/qualification.json)
passes all 12,822 selected compiler cases with zero failures, errors, skips or
retries, and all 4,502 paired cases with the exact 29 declared optional skips.
All 1,520 source guards and the respective 134/188 external guards hold; all
owned processes exit. The 1,300 marked compiler CUDA cases remain outside the
selection. Existing runtime `807bdd62…` GPU field/copy controls are reused through
the emitted-code equivalence evidence; no new residency GPU fixture run is
claimed. Runtime, ABI, public schemas and profile `47031362…` are unchanged.
[Parent attempt 25](/tmp/fortran-parent-graph-current-25/plan.json) was interrupted
by the environment reset without a worker result or completion record. Its old
session handle and host worker are absent. The
[interruption receipt](/tmp/fortran-parent25-interruption-20261104-01/qualification.json)
confirms that all 1,520 source and 1,896 input guards still hold; it establishes
neither compiler success nor a timeout. The original attempt files remain intact.
A [bounded diagnostic](/tmp/fortran-parent25-compiler-profile-01/plan.json) uses
the same source and first candidate, with 300 unprofiled seconds followed by a
120-second profile window. It too was interrupted before final statistics or a
worker result. Its saved start-of-profile stack locates native dependency-proof
validation during the original nested source scan; all 3,424 inputs still match.
This partial stack supplies no timing attribution or compiler verdict.
Application correctness and the 10%, 2× and 20× performance gates remain open.

The retired strict-marker selector is removed from lowering and discovery.
Explicit `--require-markers` or Python `require_markers` requests now fail with
migration guidance; marker comments remain accepted. All 110 affected frontend,
discovery and CLI tests pass. Source-registry checks now iterate builtin values
when retained key identities and order match, preserving the original key-type
fence and authoritative ID comparison. All 229 affected source tests pass. The
[composed current-tree check](/tmp/fortran-main-residency-cleanup-composition-20261104-02/qualification.json)
passes 140 cases, including complete emitted-artifact equivalence and late
mutation rejection. The [fresh whole-suite qualification](/tmp/fortran-maintained-current807-c3-native-full-11/qualification-current-checkpoint.json)
passes all 12,837 compiler cases in 48 disjoint whole-file batches and all 4,502
paired cases in one invocation, with exactly 29 declared optional skips. Both
mandatory CUDA IEEE cases pass; the 1,300/18 marked CUDA exclusions are unchanged.
Independent audits verify the complete node unions, source/tool/profile guards
and exited process groups. The compiler campaign is not a single pytest invocation.
A separate
retained-state comparator experiment is not promoted: its final small-fixture
measurements do not justify the added comparison logic.

The [exact original `Bound_Phi` probe](/tmp/fortran-bound-phi-generic-probe-20261104-01/qualification.json)
preserves runtime dimensions, boundary flags and all serial branches, with only
Phi captured. It returns a strictly validated package with zero GPU scopes:
all six face updates stop at the generic `ELSEIF` publication restriction.
Read-only host flags need no payload copy, but their existing native alias
reservations must still precede each reached predicate. The next generic slice
preserves lazy predicate evaluation while emitting those existing access units.
Its [private composition](/tmp/fortran-elseif-current-main-composition-20261104-01/qualification.json)
passes 332 CPU checks and the 12-case device fixture (three CPU cases and nine
traced/untraced/hidden device modes), with 104 traced kernels verified. A reached
managed predicate downloads four bytes without invalidating its GPU replica or
requiring a later upload. Existing conservative alias retirements remain explicit.
The original Bound_Phi body now generates 34 segments and eight GPU leaves;
two reversed-section typed-guard cuts remain. Generated artifacts match the
qualified device build. This private work has no actual ELMM pressure execution
or natural-run qualification yet.

[Parent attempt 26](/tmp/fortran-parent-graph-current-26/result.json), using the
new calibration and unchanged selected interval/budgets, ends at the 2,400-second
target limit. All 1,520 source and 1,966 external guards hold, owned groups exit,
and no package is accepted. Its stack is inside active `StorageAdmission`
retained-state reconstruction during unit emission, rather than the earlier
full-epoch path. This identifies the next diagnostic location, not an exclusive
attribution of the entire compile interval. The first-admission census/profile above
now supplies evidence for the integrated immutable-container change.

The following entries preserve earlier checkpoints and their limitations.

The [repeated-registration label fix](benchmarks/reports/gpu-residency-implementation-20261023/repeated-registration-labels-20261102/README.md)
is applied after 76 relevant tests pass, including native Fortran compilation;
the exact regression fails against the old emitter. Cloned registration blocks
now have distinct construct names, with unchanged registration bodies and
single-occurrence output. A private copy of the complete failing boundary module
also compiles with label-only changes. Fresh corrected SGS generation passes;
normal Convection regeneration is running. The failed application build remains
recorded, and no corrected application or speedup is qualified yet.

The [invocation storage-admission change](benchmarks/reports/gpu-residency-implementation-20261023/storage-invocation-admission-20261102/README.md)
is applied after 261 source tests and eight existing CUDA/source controls pass.
Ordinary sealed invocation plans now use the existing synchronous admission
scope: 30 membership queries require two full storage-record checks instead of
30, with the issued record unchanged. Source publication, prerequisite ownership,
native resource roles and final full validation remain checked. The runtime,
calibration and manifest schema are unchanged. Fresh normal SGS and Convection
packages now pass strict validation; Convection generation completes in 1,190
seconds. Their composition passes, but the subsequent application build finds
duplicate generated Fortran block labels in cloned boundary-registration paths.
The [failed build record](/tmp/fortran-startup-application-build-05/completion.json)
is retained. Corrected generation, application correctness and speedup remain
open gates.

The [baseline V4 regression checkpoint](benchmarks/reports/gpu-residency-implementation-20261023/maintained-v4-20261102-addendum-01/README.md)
passes all 12,056 selected compiler cases without failures, errors, skips or
recoveries. The companion selection passes 4,333 cases with 29 declared optional
skips. All 1,243 baseline source guards match the tested snapshots. Marked CUDA
exclusions and focused device gates remain separate. This includes the applied
atomic retention/reuse and native-wrapper changes below; application generation
and performance qualification remain unresolved.

The [atomic retention follow-up](benchmarks/reports/gpu-residency-implementation-20261023/atomic-arena-provider-empty-retention-20261101-addendum-01/README.md)
is applied. Three empty original host teams execute while preserving the live
producer/consumer domain, readonly input and dirty output; their intervening
payload transfers are zero. Full fields, halos, bounds and genuine OpenMP team
entries are checked. Latest device coverage is 18 distinct passes after one
test-only hidden-mode observer recovery; the raw 17-pass/one-failure invocation
remains recorded. The [final compiler composition](benchmarks/reports/gpu-residency-implementation-20261023/compiler-atomic-composition-20261102-addendum-01/README.md)
passes 354 CPU/source cases and Ruff, with 13 promoted paths. It reuses immutable
atomic source facts and checked provider artifacts through the existing
authority machinery, preserving source/policy/publication checks. Runtime and
calibration are unchanged. Application correctness and performance qualification
remain pending. The full V3 selections below precede this increment.

The [bounded Convection profile](benchmarks/reports/gpu-residency-implementation-20261023/convection-profile03-20261102/README.md)
confirms atomic derivations fall from 112 to seven, with validation rederivations
falling from 104 to zero. The measured prefix reaches 15 initial provider
emissions rather than two. The source bases also differ in native CALL support,
so these observations establish removed repetition and further progress, without
a whole-compilation speedup claim. No compiler verdict or artifacts were accepted;
the next gate is normal generation on the qualified source.

The [subsequent normal run and late profile](benchmarks/reports/gpu-residency-implementation-20261023/convection-late-validation-20261102/README.md)
retain another 1,500-second Convection timeout without artifacts or a source
verdict; SGS alone passes strict validation. In the separate 900–1,200-second
profile window, six whole-storage-epoch validations account for 253.6 cumulative
seconds. Retained atomic-provider validation accounts for 233.4 seconds, with
zero atomic derivations or provider regenerations. These nested spans overlap.
The applied storage-admission change above targets this repeated validation.
No application speedup follows from the profile or isolated validation counts.

The [native-only wrapper gate](benchmarks/reports/gpu-residency-implementation-20261023/native-wrapper-eligibility-20261102-addendum-01/README.md)
is also applied. It checks the complete native graph and current child-routing
policy before constructing a companion with no managed GPU work, then retains
the existing whole-CALL retirement and fresh admission. Unknown or competing
GPU routes keep their original preference. Source gates reconcile 231 CPU cases;
six wrapper controls verify all 27 field/halo/bounds comparisons, one genuine
original wrapper entry per reached call, and fatal retirement without replay.
The unused nested-transaction experiment remains outside production. Parent16
now passes the prior ResistanceForce/Apply frontier but declines at the guarded
`mean_UVW` call in `linearforcing.f90:169`; explicit-shape association and joined
reduction storage authority remain unresolved. Its terminal record is additive
to the wrapper packet. Standalone Convection generation has since timed out as
recorded above; no shared package or application result is established by these
component tests.

The [previous compiler V3 selection](benchmarks/reports/gpu-residency-implementation-20261023/maintained-compiler-v3-20261101-addendum-01/README.md)
passes all 11,995 selected cases in one complete invocation, without failures,
errors, skips or recoveries. Its [companion selection](benchmarks/reports/gpu-residency-implementation-20261023/maintained-companions-v3-20261101-addendum-01/README.md)
passes 4,333 cases with 29 declared optional skips. Both qualify the applied
SECTION projection, encoder and native CALL composition with runtime `6d8484fe…`
and profile `429150e1…`. Marked CUDA exclusions and focused device gates remain
separate. Private proof/provider reuse, empty-path retention and line-default
changes are excluded; no new application or performance milestone follows.

The [committed compiler checkpoint selection](benchmarks/reports/gpu-residency-implementation-20261023/maintained-compiler-v2-20261101-addendum-01/README.md)
has 11,868 passing latest distinct results after five test-only whole-file
recoveries; its failed complete invocation remains recorded. The separate
[companion gate](benchmarks/reports/gpu-residency-implementation-20261023/maintained-companions-v2-20261101-addendum-01/README.md)
passes 4,202 cases with 29 declared optional skips. Production code is unchanged
across the recoveries. Marked CUDA exclusions, focused device gates and pending
application qualification are stated separately.
These maintained results describe commit `db3791a` with runtime `9db3a6e8…`;
they precede the subsequent startup-policy changes.

The [startup-credit successor](benchmarks/reports/gpu-residency-implementation-20261023/compact-startup-addendum-20261101/README.md)
is applied with fresh runtime `6d8484fe…` and profile `429150e1…`. Separate
device gates pass 30 behavioral cases and five logging checks on their stated
runtime bases. Bounded credit can seed the existing pool; each purchase retains
its full cold cost. Modeled pricing stays separate from completed transfers.
Source-task and source-submit costs remain unavailable. Fresh SGS generation
passes. The initial fresh Convection attempt declined at a single-association
restriction; the qualified per-call projection below addresses that restriction.
Fresh regeneration and application qualification are pending. No
current application speedup follows; the current maintained gate is recorded above.

The [SECTION call-projection follow-up](benchmarks/reports/gpu-residency-implementation-20261023/section-call-projection-20261101/README.md)
is applied after 473 distinct CPU cases and three CUDA modes pass. The original
periodic helper's three field associations now use the existing checked CALL DAG;
graph effects include every possible allocation without granting definitions or
alias authority. Initial uploads of separate readonly inputs remain necessary;
the repeated first call transfers no payload. The packet retains the original
overbroad trace-test failure and its stronger per-allocation recovery. Independent
SGS/Convection regeneration is underway; shared-parent integration remains separate.

The [compiler-profile follow-up](benchmarks/reports/gpu-residency-implementation-20261023/profile-identity-fastpath-20261101/README.md)
preserves the subsequent 1,500-second Convection timeout as incomplete compilation.
A bounded profile identifies repeated calibration JSON serialization as a major
cost. The applied C-encoder fast path preserves canonical bytes, both fresh
digests and mutation checks, with 115 CPU cases passing. The next normal
Convection attempt also reaches 1,500 seconds without artifacts. A second bounded
profile confirms lower hashing cost and identifies repeated atomic proof
derivation and provider generation as further costs. An independent reuse change
is private; no application result follows from either profile or helper timing.

The [line geometry experiment](benchmarks/reports/gpu-residency-implementation-20261023/line-geometry-experiment-20261101/README.md)
validates all 1,728 component rows and 2,592 launches. For three mask patterns,
128 threads with a 16 MiB batching target reduces summed twelve-call host medians
from 9.321/10.092/3.237 ms to 5.017/5.899/2.464 ms, with seven paired wins each.
Actual scratch is 4.39–4.59 MB. Kernel bodies are unchanged; event intervals include
host submission gaps. The clean default remains private pending qualification
and application measurements.

The [native CALL preservation slice](benchmarks/reports/gpu-residency-implementation-20261023/native-call-preservation-20261101/README.md)
and [combined SECTION follow-up](benchmarks/reports/gpu-residency-implementation-20261023/native-section-composition-20261101/README.md)
are now applied. A separate structural view of the existing graph proves normal
return without imports, publication or hidden lifetime changes; it preserves the
same conditional access precondition as first admission and grants no payload or
definition authority. The original ResistanceForce source closure passes. Device
gates verify full retirement/fresh rearm and failure without replay. Exact-site
CALL validation fixes canonical-versus-local slots and repeated-site ambiguity.
Active combined paths retain the new working domain across three SECTION calls;
inactive paths still create three empty metadata-only domains. Broad current
regression now passes as recorded above; an actual shared ELMM parent remains pending.

The [kernel resource diagnosis](benchmarks/reports/kernel-resources-20261101/README.md)
identifies small Filter grids and serial line processing in the previous
application build. Its Nsight Compute attempt stopped on NVIDIA counter
permission denial; ordinary CUDA remains available. No counters, new correctness
qualification or performance sample were obtained. Block size and a proved
parallel line mapping remain experiments, without a projected payoff.

The [native conditional-call follow-up](benchmarks/reports/gpu-residency-implementation-20261023/native-if-call-retirement-20261101/README.md)
proves the original `ComputeTDiff` IF and both OpenMP branches through the
existing graph/storage/completion contracts. Separate CPU gates and three real
CUDA modes pass; false metadata guards preserve residency, true guards retire
once, and effectful predicates publish before evaluation. The packet preserves
the compiler watchdog recovery and its limits. Complete parent generation,
latest application correctness and performance remain separate gates.

The [complete physical-copy V2 backend](benchmarks/reports/compact-physical-transfers-20261101/README.md)
is applied with 52 focused CPU, 263 profile/emission and 14 CUDA checks in separate
gates. A single fresh calibration supplies six complete direct/cold/reuse models
across 61 geometries. The shared process pool bounds idle storage separately from
the domain's active field/scratch/workspace budget. Historical ELMM groups select
no cold allocation, although many channel downloads qualify with a cached pair;
the startup-credit successor addresses that limitation. This is implementation and
calibration evidence, with no application gain or full-current-suite claim.
Structural source-submit pricing remains unavailable independently of these copy
models. The previous residual model is superseded rather than kept as a second
production cost path.

The [promoted compiler composition and reached-loop follow-up](benchmarks/reports/gpu-residency-implementation-20261023/reached-loop-composition-addendum-01/README.md)
now preserve a live GPU domain across empty original native counted loops.
The composed native-call/guarded-SECTION/selector revision passes 1,310 CPU/reader
cases and five CUDA cases. The subsequent loop revision separately passes
322 CPU checks and three CUDA cases: twelve empty-loop handoffs have no
intermediate payload copies, while nonempty loops retain exact retirement/rearm.
All original headers, final iterator values, fields, halos and zero-trip behavior
are checked. These overlapping gates are not a full-current-suite result and
include no new runtime ABI or physical-copy implementation. Parent12's compiler
`SIGSEGV` remains a no-verdict result; no ELMM shared-parent retention or latest
application speedup follows from this component evidence.

The latest [CPU-pressure four-mode campaign](benchmarks/reports/four-mode-cpu-pressure-20261031/README.md)
passes all 64 process/output checks but **fails every performance checkpoint**.
Candidate medians are 15.446/24.151 seconds on simple109/channel100, versus
11.077/13.752 seconds for fresh `7a612` measurements: **39.4%/75.6% slower**,
with zero of seven paired wins against every control. Native medians are
11.819/13.310 seconds, giving 20× budgets of 0.591/0.665 seconds. This candidate
passes strict natural109/2458 correctness using CPU PoisFFT; the separate
GPU-pressure long-channel failure remains open. SGS and Convection are explicitly
forced with `--gpu-policy sections`, so automatic placement profitability is
unqualified. Shared residency and measured runtime costs remain priorities.

The [new node-level CUPTI profiles](benchmarks/reports/cpu-pressure-cupti-20261031/README.md)
record 23.75/31.77 GB across 23,118/71,310 copies. Channel includes all 600 expected
graph-node kernels and makes 52,201 downloads, many only 4–9 KB. Its 51,904
`cudaMemcpy3DAsync` calls occupy 11.673 seconds of summed host API spans, overlapping
GPU work. Safe physical-copy grouping must complement precise effects and moving
consuming work to GPU. Existing upload-only coalescing cannot address this download
fragmentation. Source freshness, initialized regions and queued users must constrain
any wider copy; no unsupported causal attribution to one compiler change follows.

The [profiling-tool correction](benchmarks/reports/nsys-graph-node-metrics-20261031/README.md)
and its [zero-ID correction](benchmarks/reports/nsys-graph-node-zero-correction-20261031/README.md)
are applied with 94 passing current CPU tests. Both capture commands explicitly request
graph-node tracing. Old aggregate graph intervals remain separate from individual
kernel counts and make the kernel inventory incomplete; tracing options alone do
not establish full-application coverage. NULL and zero IDs do not count as graph
nodes. Four saved captures were read back without
running new applications or changing historical measurements.

The [exact geometry diagnostic](benchmarks/reports/copy-geometry-20261031/README.md)
now records 562 pre-copy region states across both two-step cases, preserving
all 20 field bytes and original traffic. Offline witnesses can reduce channel's
1,045 downloads to 799 with 1,655,376 extra bytes, bounded by 12.5%/64 KiB per
batch and exact current/initialized source coverage. This is an opportunity,
not an executed transport optimization. Existing pinned staging still issues
pitched device copies; compact GPU gathering would need separate ordering,
workspace, failure and cost qualification. Wider copying alone has limited
payoff, so parent residency and GPU consumers remain the main priorities.

The [original Filter3ord chain](benchmarks/reports/original-filter-line-chain-20261031/README.md)
now passes 36 byte-exact comparisons across native, CUDA and GPU-hidden paths,
including full fields, masks, halos and zero-trip shapes. Four original filter
calls share canonical handles with a GPU producer and consumer; repeating the
first axis transfers no intervening payload, and host reads preserve device
validity. The generic line-snapshot and scalar-call readability implementation
and strict readers are applied to the working tree. This is a component gate,
not ELMM performance qualification. Channel's saved trace contains 58.76 MB of
Filter downloads followed by GPU diffusion uploads on the same generations;
post-filter boundary consumers still require evaluation. Channel-simple does not
reach these filter calls in that trace.
The [channel-sized component follow-up](benchmarks/reports/original-filter-line-chain-20261031/channel-size-addendum-01/README.md)
adds nine byte-exact comparisons at 128×64×64. Its existing 4 MiB scratch budget
splits each filter into two kernels, while the repeated call still copies no
payload. The first small-shape launch-count assertion failed and is preserved;
the successor checks the actual bounded batching without changing production code.

The [multi-field original-module successor](benchmarks/reports/original-filter-line-chain-20261031/multi-field-successor-20261101/README.md)
now passes 24 byte-exact native/CUDA/GPU-hidden comparisons for three distinct
field/mask pairs, including different channel-sized layouts. All nine canonical
allocations retain their identities, and repeating the first field after the
other two requires no payload copies. The compiler and strict reader now bind
every original call separately and include every possible field in graph effects;
the [integrated composition](benchmarks/reports/source-task-call-projection-20261101/README.md)
passes 1,379 CPU cases and seven CUDA cases. This
clears the previous singleton-association blocker for Convection regeneration.
The [regenerated application](benchmarks/reports/filter-convection-20261101/README.md)
now passes both two-step cases at unchanged tolerances, with all 20 field payloads
byte-identical to the prior CPU-pressure candidate. Channel executes 108 filter
kernels and removes 91,059,072 transferred bytes over initialization plus two
steps, but adds 504 physical copy operations. Channel-simple does not reach the
filters and its traffic is unchanged. Six separate Convection domains remain;
this result does not establish parent retention or an end-to-end speedup.
The [natural109/2458 qualification](benchmarks/reports/filter-convection-20261101/natural-20261101/README.md)
also passes all fields and outputs at the original tolerances. All forty field
payloads and their shapes match the prior qualified candidate byte for byte;
119 of 120 output files match byte for byte, with only a tiny diagnostic change.
These correctness runs are separate from performance measurement.

The [complete primary profiles](benchmarks/reports/filter-convection-20261101/primary-profiles-20261101/README.md)
show channel transferring 4.553 GB less but making 25,200 more copies. Its GPU
activity occupies 7.711 seconds of a 24.171-second GPU span; 9.070 seconds of
copy-API spans intersect GPU-idle intervals. Filter kernels take 1.665 seconds.
These overlapping diagnostic durations cannot be added or treated as a speedup
forecast. The next transport work targets exact batching and host-access
consolidation alongside parent retention. The paired profile helpers are applied
after 46 CPU cases; anonymous kernel names retain all possible source origins.

The [fixed control-point extension](benchmarks/reports/atomic-line-composition-20261031/README.md)
is also applied after 621 joint CPU/reader checks. Its separate nine-case CUDA
gate preserves a GPU-written control point across a four-byte host read and a
later GPU consumer; inactive points remain untouched. Simulated fatal API errors
after completed prior work abort without replay. This extends the existing
conditional DO-only team contract and coherence machinery; it does not yet admit
ELMM's periodic SECTIONS group or qualify application residency or performance.

The [current hierarchy composition](benchmarks/reports/current-hierarchy-composition-20261031/README.md)
now applies the generic checked-child residency slice, scalar-component call
associations and fixed record-field storage checks together with those line and
atomic features. All 1,897 CPU/reader cases across 49 files and six CUDA cases
pass. The CUDA ledger verifies 24 complete field comparisons and 20 mixed
producer/host-consumer intervals, including read-only replica retention, partial
writes, zero-trip paths and fatal failure without replay. Allocation exchanges
still publish and reopen an epoch in this slice. Actual SGS→Convection retention
and application acceleration remain unqualified; later component-CALL,
multi-array call-projection and native-effect-graph work is separate.

The earlier [cached GPU-pressure campaign](benchmarks/reports/four-mode-cached-20261031/README.md)
remains preserved: 18.245/18.967-second medians, 59.1%/38.6% slower than its fresh
`7a612` controls. Its solver and compiler basis differ from the current campaign.

The separate [CUPTI captures](benchmarks/reports/gpu-residency-implementation-20261023/cached-primary-cupti-addendum-01/README.md)
record 26.50/34.30 GB of host/device traffic across 23,796/24,534 copies.
After the [graph-interval correction](benchmarks/reports/cpu-pressure-cupti-20261031/README.md),
GPU activity occupies 6.157/6.865 seconds; idle inside the first-to-last GPU
activity spans totals 11.520/11.628 seconds. Of that idle time, 7.787/8.993
seconds overlaps neither GPU work nor recorded CUDA APIs. The separate
[CPU callback budget](benchmarks/reports/cached-cpu-budget-20261031/README.md)
attributes 11.162/10.560 seconds to runtime wrappers, including waits and transfers.
The [fine diffusion measurement](benchmarks/reports/fine-diffusion-budget-20261031/README.md)
identifies 3.547/3.275 seconds in generated point-list checking and rectangle
normalization, versus 0.058/0.056 seconds in the original wall-flux bodies after
excluding measured nested barriers. Optimize this repeated inspection together
with shared residency. These separate captures are diagnostic; do not add their
times or overlapping host API and GPU durations.

The [translated-footprint optimization](benchmarks/reports/translated-indirect-footprints-20261031/README.md)
is now implemented: adjacent references reuse an exact bounded footprint inside
one host-operation preparation, retaining original bounds, integer-overflow and
work-budget checks. All 63 CPU cases and the CUDA coherence check pass. The
isolated source-matched scan medians sum to 8.777 ms before and 3.075 ms after;
one scan is 10.5% slower. This isolated result is not an ELMM timing gain. The
current application passes correctness but fails the fresh timing gates above.

The [source-bound transfer ledger](benchmarks/reports/gpu-residency-implementation-20261023/cached-transfer-attribution-addendum-01/README.md)
confirms the three main diffusion loops already run on GPU. The immediate
traffic targets are Convection's owning-return publication and channel's native
periodic corner/edge operation. The latter falls back to whole-field access
and downloads 94.10 MB in the saved initialization-plus-two-step trace. The
[implemented accounting correction](benchmarks/reports/gpu-residency-implementation-20261023/native-effect-budgets-addendum-01/README.md)
now applies the runtime's existing 32-rectangle limit separately to each effect
list. Its exact 20 reads and 20 possible writes fit without changing geometry or
granting overwrite credit; 95 CPU cases and a generated CUDA copy-ledger check
pass. The [regenerated CPU-pressure application](benchmarks/reports/cpu-pressure-transfer-20261031/README.md)
passes both short all-field/output checks, but total ordinary-component traffic
is unchanged: 91.48 MB removed from the periodic download reappears at
`BoundUVW_do_BC` and `Filter3ord`. The precise contract works locally; later
native consumers still need the values. GPU coverage and shared parent residency
must address those demands. This path retains CPU pressure while the
GPU-pressure long-channel failure remains open.

That CPU-pressure baseline now also [passes natural109 and channel2458](benchmarks/reports/cpu-pressure-natural-20261031/README.md):
all 20 fields, bounds, halos and all 6/114 outputs at the unchanged tolerances.
It provides a qualified baseline for further residency work. The runs shared
the GPU with correctness tests and are not performance samples; they do not
qualify the separate GPU-pressure backend.

The [translated-footprint optimization in ELMM](benchmarks/reports/cpu-pressure-optimized-20261031/README.md)
now also passes both natural cases and both short GPU coverage gates. Final fields
are byte-identical to that CPU-pressure baseline; transfer totals are unchanged.
The completed four-mode measurements above fail every performance checkpoint.

The [source-fact reuse change](benchmarks/reports/gpu-residency-implementation-20261023/source-facts-addendum-01/README.md)
uses the existing immutable source arena and passes the final focused CPU gates
and six composed CUDA cases. Connected-call candidate ordering now reaches the
intended SGS→Convection pair first. The next complete attempt finishes with an
empty package: its storage proof cannot classify a read-only derived-type array
in the possible Buoyancy closure. A private generic fix passes 569 CPU checks
and all 13 original source-function projections, preserving the other 12 results.
The renewed parent compile also finishes with an empty package: the next blocker
is the procedure-valued `fun` argument in `ResistanceForce::apply`. Shared
application residency remains unqualified. The [private hierarchical child-region
implementation and parent attempts](benchmarks/reports/gpu-residency-implementation-20261023/checked-child-residency-addendum-01/README.md)
reuse existing checked-return coverage and synchronous retirement
cuts, avoiding a whole optional-descendant storage closure. Its small mixed
producer/host-read/consumer sequence passes six CUDA cases; 346 strict-reader cases
also pass. The next actual parent attempt reaches into diffusion but fails on
classification of an automatic local allocatable as a module allocation. These
small gates do not qualify the actual ELMM parent domain. The
[local-root correction](benchmarks/reports/gpu-residency-implementation-20261023/checked-child-residency-addendum-01/local-root-successor-01/README.md)
passes 169 focused cases: only genuine module declarations enter the module
inventory, and automatic locals gain neither SAVE nor return lifetime. Its joint
composition now [passes 1,117 CPU/reader cases and six CUDA cases](benchmarks/reports/residency-line-composition-20261031/README.md).
The copy-ledger checks confirm read-only replica preservation and exact partial
uploads. The renewed actual-parent attempt passes the allocation-classification
failure but declines the intended pair at a returned-call association check;
actual ELMM retention remains unqualified.
The [scalar-component correction](benchmarks/reports/residency-line-composition-20261031/scalar-component-successor-01/README.md)
passes 180 CPU cases. Fresh projections of unchanged read-only scalar components
are authenticated through their original object, selector and declarations;
ordinary array bindings still require exact identity. The prior parent search was
intentionally interrupted after the target-pair decline, not reported as a complete
empty generation. The [next checked-child diagnosis](benchmarks/reports/residency-line-composition-20261031/checked-child-frontier-successor-01/README.md)
passes that association check and identifies the exact remaining native group
in `ComputeViscsWM`. Its storage classifier rejects harmless descriptor inquiries
and fixed scalar fields because another field owns an allocation. A private
field-sensitive correction passes 432 cases and both original SIZE/field-read
projections; complete native-call and joined-lifetime coverage remain open.
The diagnostic's metadata-span failure is preserved separately from those two
successful projections. No actual ELMM parent domain is qualified.
Those hierarchy, scalar-component and field-read changes are now included in
the [promoted current composition](benchmarks/reports/current-hierarchy-composition-20261031/README.md)
described above. The earlier packets retain their original private status and
test bases; they are not additional tests on the current composition.

The [joined-native boundary integration](benchmarks/reports/joined-native-boundary-20261101/README.md)
is now applied after 956 CPU cases, two emitted-transport cases and four CUDA
cases. The original `ComputeViscsWM` group has complete independent native-graph,
team and storage proofs across 22 callees and 18 scalar-field associations, with
no storage frontier or escape. Diagnostics and STOP remain observable, and
completion applies only on normal return. The actual parent retry passed that
group and then declined a scalar table-element CALL argument in `SubgridModel`.
Its first-candidate rejection is preserved without fallback; no ELMM parent
package or cross-child retention follows from that attempt.

The [scalar-element and native-transform integration](benchmarks/reports/hierarchical-scalar-elements-20261101/README.md)
is now applied. It passes 416 scalar CPU/reader cases, three CUDA cases with
eighteen complete field comparisons, and 222 separately counted native-transform
cases. All five original SGS table-element calls now qualify; the backing table
stays on the host and its writes retain the existing retirement cuts. Native
`MATMUL`/`TRANSPOSE` operand reads now pass the original stability loop's complete
graph/team/storage checks. The next parent attempt passed those barriers and
declined the optional `SGS_MixedTimeScale` call: its local SAVE allocations must
stay in the original native callee, whose nested OpenMP completion and storage
effects need a whole-call contract. No parent artifact or application retention
is qualified yet.

The earlier [maintained compiler suite](benchmarks/reports/maintained-composed-compiler-20261031/README.md)
has 10,971 distinct cases passing on its frozen basis after exact reconciliation. The raw
full invocation retains 10,893 passes and 76 failures; fixture recovery and reviewed
14-file test migrations replace those failures without changing production code.
Subsequent line, atomic-point and hierarchical work is outside that frozen suite basis.
The [fresh maintained selection](benchmarks/reports/gpu-residency-implementation-20261023/maintained-current-20261101-addendum-01/README.md)
now has 11,600 distinct passing cases across 531 files on the current frozen
implementation. Its original four-shard run retains 11,595 passes, four skips
and one setup error: two-core affinity conflicted with four-core calibration
fixtures. Both affected whole files pass all 73 cases with the correct affinity;
no production or test assertion changed. The original invocation remains failed.
The selection excludes 1,211 explicitly CUDA-marked cases; selected unmarked GPU
correctness tests ran. Later private retirement, conditional-boundary and
transport work is outside this frozen basis.

The [fresh companion selection](benchmarks/reports/gpu-residency-implementation-20261023/maintained-companions-20261101-addendum-01/README.md)
has 4,071 distinct passing cases and 29 declared optional skips across 4,100
selected cases, with 18 marked CUDA cases deselected. Pipeline has 3,790 passes,
containers 220 and benchmarks 61. The original pipeline failure is preserved;
updating its obsolete invalid profiling duration from 31 to 121 seconds passes
all 25 cases in the affected file against unchanged frozen production code.
Twenty-four repeated passes are counted once. The independently verified
packet excludes Git metadata and does not qualify later private changes.

The [companion V2 packet](benchmarks/reports/gpu-residency-implementation-20261023/maintained-companions-v2-20261101-addendum-01/README.md)
records a fresh 4,202 passes, 29 declared optional skips and zero failures/errors
across 4,231 selected cases, plus 524 passing subtests. Runtime `9db3a6e8`,
profile `ea3f80d8`, promoted IF-CALL8 and all 2,122 recorded guards are bound.
The portable verifier omits all Git and compiled bytes; 18 marked CUDA cases
remain excluded from this selected integration qualification.

The earlier [checkpoint52](benchmarks/reports/gpu-residency-implementation-20261023/four-mode-checkpoint-52/README.md)
passes all 64 fresh applications' correctness checks but **fails the primary
10% performance gate in both cases**: it is **5.8%/2.1% slower than fresh 7a612**
on simple109/channel100, with 1/7 paired wins each. It beats native in all
seven pairs per case, with 11.6%/11.8% lower medians. The 2×/20× targets and
production GPU AUTO cost qualification remain unmet. All startup, original
output, solver, transfers, FFTW work and teardown remain in the clock. All eight
warmups and 56 samples are retained, including large simple native/be58 samples.
Historical checkpoint51 remains intact; the separate campaigns do not isolate
the effect of the launch-count change.

## Objective and evidence

Target `channel-simple` and `channel`, one NVIDIA GPU, double precision and the
uniform-grid configurations. Seek 20× complete-process acceleration against
four-thread native ELMM. This is a research objective, not a forecast.

The [committed parity report](benchmarks/reports/graph-redesign-parity-20261022/README.md)
establishes the starting point at main `7a612e4`, companion `55ba917`:

- Profiles establish SGS GPU execution; Convection remains native.
- Approximately 52,813,824 bytes and 30 transfers per steady-state step. The
  previous 123 MB / 70 count includes initialization and two steps.
- Recorded copy activity is approximately 0.67/0.62 seconds. It does not
  establish that transfers consume 85% of complete-process wall time.
- Recorded FFT solve intervals are approximately 0.90/0.83 seconds. Replacing
  the solver alone cannot establish 20×.
- Native medians of 11.65467/13.25841 seconds imply 20× budgets of approximately
  0.583/0.663 seconds, including startup and output. Recorded SGS activity is
  already near the first budget.

The fresh GPU timeline and CPU budgets above replace the speculative savings.
Refresh the affected measurements after implementation changes; diagnostic
tracing and untraced performance measurements remain separate evidence.

The later current-tool combined SGS/Convection/GPU-pressure build passes
two-step checks in both cases and natural simple109. Its
[natural channel2458 comparison](benchmarks/reports/gpu-residency-implementation-20261023/combined-natural-addendum-01/README.md)
fails 15 fields and 16 of 114 outputs at the original tolerances. Retained solver
plans, correct call counts and successful short tests do not close that gate.
Continue generic residency and setup-cost work independently; require a passing
long comparison before treating GPU pressure as a qualified application backend.

The [bounded solver compilation cache](benchmarks/reports/gpu-kernel-cache-20261031/README.md)
and its authenticated campaign launcher are now implemented. A freshly built
[cached combined candidate](benchmarks/reports/gpu-residency-implementation-20261023/cached-pressure-addendum-01/README.md)
passes four cold/warm short applications at the original field/output tolerances.
Warm processes perform zero NVRTC compilations; cold processes perform twelve.
All field and VTK bytes match cold to warm. The original blanket byte gate fails
on channel's six-value `Retau` diagnostic (maximum difference `2.73e-12`) and is
preserved separately from the passing tolerance comparisons. This cache retains
compiled code, not application fields. The fresh primary timing above establishes
the current regression; it does not qualify a performance checkpoint.

## Architecture

Storage belongs to a caller-owned residency domain, independently of an
execution context. Checked children borrow it. Returning to an unmanaged caller
publishes observable values. Teardown waits for outstanding uses and releases
storage. Domain activation is lazy; zero-trip paths must not inspect unallocated
Fortran descriptors.

Reuse the scoped runtime's canonical allocations and regional coherence. Read
copies add replicas; writes invalidate only overlapping regions. Track initialized
regions, generation, physical offsets, bounds, extents, strides, pending completion
and eviction. Preserve the latest required value, queued readers and workspace
reservations. Proved descriptor transfers move storage identity; unknown
allocation/association mutations retire the affected epoch.

The checked compiler graph is authoritative for execution and storage epochs.
Preserve data, anti-, output, lifetime, control, IEEE and observable-effect edges.
Regroup independent host tasks inside their original control regions. Price
connected GPU groups with shared transfers, current replicas, launch overhead
and memory pressure. Search budgets must never truncate the execution graph or
limit residency to a fixed work-unit count. Verify every emitted schedule.

External device tasks use typed access transactions, not unchecked reserved
pointers. Their contracts include library/configuration identity, canonical
buffer generations/layouts/regions, stream/device, workspace and completion.
Rejection before submission may permit checked fallback. Failure after a
submission attempt poisons the execution and must never replay source work.

Version runtime/manifest contracts together, regenerate artifacts and recalibrate
changed runtime costs. Keep one coherence implementation and one production
policy. Remove obsolete bridge/catalog/placement paths after their migration.

## Work and exit gates

### A. Budget and solver feasibility

Measure initialization, SGS, Convection, diffusion, boundaries, pressure
preparation, FFT, pressure correction, runtime and output. Partition exclusive
main-thread wall time, retain worker CPU separately, and correlate with GPU
activity without double-counting overlaps. Record actual backend choices, copy
reasons, allocation churn and idle gaps.

Pin supplied `poisson-solver` at
`4a1f1bb7e6f9dab6a2934b62d9b8655cf3a9f99a`. Correct its actual library packaging,
matching Fortran compiler, explicit architecture, pinned dependencies,
out-of-tree generation and install artifacts. Differentially qualify against
pinned PoisFFT. Exit with a measured 20× feasibility gap and solver equivalence.

### B. Shared-residency slice

Issue storage epochs from sealed checked source/lifetime facts. First qualify
GPU child → nontrivial readonly host operation → GPU child, then an allocation
swap. Unchanged data uploads once; host reads preserve device validity; partial
writes affect only their regions. Preserve mandatory cuts until exact effect and
lifetime evidence permits replacement. Disable native-owner shortcuts in live
domains and include graph/descriptor/generation/path facts in replacement keys.

Apply the mechanism to SGS→Convection in `TMarchRK3`. Close the preserved
residency-cut regression before claiming general retention.

The original-source query29 now clears `Uf = U` in `SGS_MixedTimeScale` using
conditional descriptor contract21. It retains actual allocation effects and
requires that the private scratch have no canonical registration anywhere in
the complete graph domain. Epoch2 and child consumers preserve that obligation.
The [source-contract packet](benchmarks/reports/gpu-residency-implementation-20261023/graph-lifetime-contracts-addendum-01/README.md)
preserves the core's 1,584 passing checks, the strict reader's 63, and the combined
private-allocation/enclosing-context gate 733 overlapping checks. None grants
an application epoch. The exact unchanged query30 clears the filter's original
parallel group at line156–193 and stops at the orphan worksharing directive at
line197, after `END PARALLEL`. Preserve its original caller/team semantics when
adding descriptor support. Continue checking every original
call, OpenMP boundary and protected allocation mutation. Do not substitute
scalar-fill reasoning or assume the model branch is untaken.

[Conditional descriptor23/epoch3](benchmarks/reports/gpu-residency-implementation-20261023/original-worksharing-storage-addendum-01/README.md)
now retain the whole unchanged native callee
around original orphan worksharing. Complete body scanning and independent
native completion remain required. The integrated producer has 175 distinct
passing cases, the strict reader 267, and a further current storage/graph gate
115; both actual emitted manifests pass transport validation. Final publication
and cross-package composition enforce the whole-callee edit exclusion. Original
query31 completes setup in 76.892 seconds and reaches the unchanged 180-second
proof limit during repeated allocation-source validation. Its partial traversal
advances beyond the earlier filter calls, but grants no completed descriptor
proof or application residency. Finish immutable derivation ownership before
another original-source query.

The first compiler-cost correction is qualified: local-allocation factory
queries consume one fresh derivation, while standalone validation remains
fresh. Its 185-check gate passes. A matched compiler profile halves these
derivations from 8,184 to 4,092 and reduces the profiled test from 182.181 to
119.831 seconds; this does not measure application speedup. Implicit worksharing
end support now passes its 359-check team gate and 611-check descriptor22
integration gate, with independent review. It retains the original loop-end
barrier, complete parent join and all private-allocation exclusions; it grants
no execution or payload authority. The original query30 safely declines within
its original limits; pressure18 again times out in setup at 240 seconds before
composition. Its stack still shows repeated source-receipt validation. Reduce
that compiler cost before another bounded pressure query. Temporary kind-import
lookups are now integrated after a 323-check gate and independent review; they
retain numeric evaluation order and discard their caches before later scopes.
The [cost packet](benchmarks/reports/gpu-residency-implementation-20261023/import-kind-cost-addendum-01/README.md)
establishes exact records and fewer declaration scans on a small constructor,
not a full ELMM parse or application speedup.
The later [declaration-cache extension](/tmp/fortran-kind-import-declaration-cache-01/final-02/qualification.json)
is promoted after 326 distinct affected tests. It reuses exact owner selector
inventories and root binding lookups within one pass, while evaluating kind
values freshly. Complete analysis snapshots match byte-for-byte for 12 small
controls and the actual 73-source input. One matched constructor measurement
falls from 114.35 to 48.11 seconds; full compilation and application performance
remain separate measurements.

The first descriptor-factory reuse
experiment remains held after a slower single matched comparison. Its narrowed
successor is now integrated: one fresh derivation per cached issuance, fresh
standalone validation, exact consumed-token checks and no traversal of unrelated
module sibling bodies. It passes 969 affected checks on the prior kind basis
and 155 integration checks on the current combined basis; ten canonical records
retain their IDs and counters. The original failed live guards from the
concurrent, authorized kind-cache promotion remain recorded separately.
Source-sized global capacity was independently qualified by 122 checks.

The new complete-Convection generation now finishes in 627.323 seconds within
the original 900-second limit. Its one lexical owner contains 16 GPU leaves
and three source device tasks; the full checked plan, saved package, runtime
and all 178 artifacts pass strict transport validation. This diagnostic uses
the unchanged original selection and explicit sections placement, with the
approved migration to matching runtime profile43. Fresh SGS generation on the
same source paths, public composition, production build, full20 diagnostic
relink and artifact authentication now pass. Equal file bytes alone did not
make the earlier differing source-coordinate maps safely composable. Placement
remains unpriced. The first two-step simple diagnostic passes all 20 fields,
bounds, halos, VTK and fixed FFTW plans. Its original coverage check fails because
deferred kernel launches inherit the later host flush position. Preserved trace
reanalysis finds 132 launches under the original Convection owner and 23 accepted
CUDA implementation IDs; these are separate observations, not per-unit launch
attribution. The [completed short/profile qualification](/tmp/fortran-full-convection-current-01/diagnostic-qualification-01/qualification.json)
passes all eight processes; CUPTI independently confirms original Convection
GPU kernels. Initialization plus two steps records 139 direct kernels and
453,033,744 transferred bytes in simple, and 415 direct kernels, six separate
graph submissions and 652,750,608 bytes in channel. These increased traffic
totals reinforce the need for shared residency; coverage is not a speedup claim.
Both [natural 109/2,458-step runs](/tmp/fortran-full-convection-current-01/natural-qualification-01/qualification.json)
also pass all 20 fields, bounds, halos and outputs at unchanged tolerances.
Complete-process performance for this build remains unmeasured.
This package retains the original default 4 MiB batching target;
it does not isolate a comparison against candidate52's 16 MiB configuration.
The channel trace already keeps one Convection domain through an invocation;
it does not close after every child. Next, retain the parent SGS→Convection
interval and prove precise native halo effects for `BoundViscosity`, whose
ordinary affine face assignments should require only their affected regions.
Keep unsupported MPI/scatter paths conservative. The batching target alone
cannot explain the extra traffic: simple executes no tile kernels, and the
earlier controlled batching change preserved copy bytes.

The independently reviewed queued-provenance correction now preserves original
positions at queue acceptance. Direct launches retain those positions; graph
submissions expose separately counted logical members, not extra launch API
calls or completion events. The runtime passes 49 checks and the reader 96;
allocation-failure controls preserve execution with unknown provenance. This
observation change has a [fresh matching calibration](/tmp/fortran-source-task-calibration-44/qualification.json).
Native management costs qualify; GPU source and structural submission costs
fail the unchanged holdout checks and remain unavailable. This campaign retains
the earlier non-graph-enabled submission benchmark, so it does not qualify
production graph-enabled owner costs. Existing application qualifications
retain their original frozen runtime.

Pressure19 completes setup in 236.970 seconds but exhausts the unchanged
180-second composition budget in local-allocation source validation. All 967
frozen compiler/pipeline files, 911 qualified inputs and 95 original inputs hold;
no storage composition or execution is qualified. This confirms the need to
simplify production proof ownership instead of adding another cache
around mutable state. Use a sealed semantic-source store, immutable authority
contexts and an issuer-owned arena of immutable derivations. Derive each complete
closure once per context; changed source or policy creates a new context.
Equivalent native projections may share completed source facts. A speculative
generation transaction gets separate consumption handles, which cannot survive
rollback; already completed immutable source facts remain reusable.
Retire unsealed production issuance after migration. The earlier `SourceSession`
alone did not close normal projections, policy and parser parent links. The
[integrated first arena slice](/tmp/fortran-proof-arena-01/final-15/qualification.json)
now closes those inputs and derives local-allocation proofs once, importing only
the exact existing joined-team prerequisite. It passes 391 focused tests and
the original source-bound consumer. A single sequential diagnostic measures
23.51 seconds versus 48.12 seconds on the old compiler; this is neither a repeated
compiler benchmark nor application speedup. Independent review and guarded
promotion cover all 898 compiler files. A separate typed worksharing receipt
fix preserves exact original identities across generation transactions; both
successive orphan/private owners now publish, and both retained private-owner
plans validate. Final source verification and unknown-closure rejection remain.
[Original SGS→Convection query32](/tmp/fortran-sgs-convection-storage-probe-32/qualification.json)
now completes setup in 87.364 seconds and the proof in 81.051 seconds within
unchanged limits. It safely declines at `Viscosity = molecular_viscosity` in
`SubgridModel`, where the descriptor checker still treats a scalar fill as
possible array reallocation. The [generic scalar-fill rule](/tmp/fortran-descriptor-scalar-fill-01/final-11/qualification.json)
is now integrated after independent review and guarded promotion of eight files.
It preserves an allocated primitive array's shape without proving allocation
presence or payload initialization. Defined or unavailable assignment namespaces
remain a conservative rejection. The broad gate passes 584 checks; the final
lexical-scope repair passes 339 affected checks and preserves genuine emitted
records through the unchanged reader. Four reproduced late `BLOCK`-import
mutation failures are now rejected. Foreign source sessions are rejected before
installation, and immutable arena prerequisites no longer trigger unrelated
identity-property inspection. The unchanged [original query33](/tmp/fortran-sgs-convection-storage-probe-33/qualification.json)
now completes its proof in 97.251 seconds and safely declines in
`ComputeUVWFluxesWM`, where the native wall-model code passes derived-type
arrays with TARGET and uses a local element pointer. The scalar-fill frontier
is cleared; no parent epoch or application retention is qualified. Investigate
source-owned record-region associations and disjointness rather than treating
an inactive branch as safe. Invocation-local retirement alone cannot discharge
an escaped alias on a subsequent invocation.

The unchanged [pressure query20](/tmp/fortran-poisson-api-24/supported-pressure-path-source-20/qualification.json)
completes setup in 104.772 seconds but reaches the original 180-second
composition limit while repeatedly validating joined-BLOCK syntax receipts.
This is a timeout, not a semantic rejection. Both queries preserve the current
900 compiler files, their original/configured sources and existing facts.
The [joined-BLOCK optimization](/tmp/fortran-joined-block-namespace-reuse-01/qualification-03.json)
is integrated after review. It reuses authenticated immutable syntax membership
while retaining namespace, declaration, child-proof and publication checks.
Eight comparison fixtures retain identical public proof records and identities;
this is compiler-cost evidence, not application acceleration. The conditional
descriptor-type normalization repair and associated test API migrations pass a
combined [632-case gate](/tmp/fortran-maintained23-combined-recovery-01/qualification.json),
with no failures or skips. [Pressure query21](/tmp/fortran-poisson-api-24/supported-pressure-path-source-21/qualification.json)
now finishes setup in 93.024 seconds and composition in 120.376 seconds within
the original limits. It safely declines at the original `PoissMG(Phi,RHS)` call
because a native allocation effect is unproved. It produces no storage proof;
the separate original GNU compilation passes. All 977 compiler/pipeline files,
921 qualified inputs and 95 original sources remain unchanged. Check the
supported-path exclusion of this multigrid branch before adding new allocation
support; this diagnosis is not GPU pressure execution or application speedup.

The completed [maintained23 run](/tmp/fortran-residency-maintained-23-01/qualification-rebased-03-01/qualification.json)
has 10,094 passes and 44 failures, with no errors or skips; all 10,138 selected
cases ran and all input guards held. The 1,181 explicitly marked CUDA tests
were deselected. Its frozen inputs precede the repairs. The
[reviewed recovery index](/tmp/fortran-maintained23-reviewed-recovery-index-03.json)
maps every failure to a passing focused whole-file gate. Immutable-API migrations
retain deliberate corruption and consumer-side rejection checks. The current
905 compiler files plus 20 fixtures match the reviewed composition. The fresh
[maintained24 run](benchmarks/reports/gpu-residency-implementation-20261023/maintained-baseline24-addendum-01/README.md)
passes all 10,246 selected cases with zero failures, errors or skips, in
4,230.45 seconds. All source, fixture, tool, package and profile guards hold;
actual CUDA access passed preflight. The 1,181 explicitly marked CUDA cases
are deselected. Later source prototypes are excluded from this run.

A complete marked-CUDA gate on this composition remains outstanding. Its
source-task pricing tests require an independently qualified source profile;
profile44 explicitly leaves GPU source pricing unavailable. Device access alone
does not satisfy that prerequisite, and an older profile must not substitute for
the current runtime's qualification.

A [bounded launch-cost diagnosis](/tmp/fortran-source-pricing-gap-diagnosis-01/qualification-01/qualification.json)
runs the unchanged submission benchmark once and a separate instrumented
observer in two fixed orders. All three processes pass their protocol checks.
The unmodified benchmark's 77 source medians fit the unchanged profile44 bound
on this run, but affinity differs from the original calibration and this does
not replace its failed qualification. Slow observer rounds spend 94–98% of
their source interval in the CUDA enqueue callback, with approximately equal
thread CPU time and wall time. No completion-event operations occur in those
intervals. This supports investigating batching and a separate queued/graph
cost protocol; it does not establish a memory-management defect or qualify
graph costs for the current direct-submit provider.

The fresh [companion05 checks](benchmarks/reports/gpu-residency-implementation-20261023/maintained-companion05-addendum-01/README.md)
use the same 925-file compiler/fixture basis with the current pipeline and
containers. Pipeline testing passes 2,777 cases with 39 unchanged opt-in/fixture
skips and 18 marked-CUDA deselections; all 220 container cases pass. Complete
selected-case coverage and all 1,885 frozen-file guards hold. Successful
subtests are recorded separately from these counts.

The reviewed [record-array descriptor slice](/tmp/fortran-owned-record-region-01/integration/final-13/qualification.json)
is integrated. It admits an exact local PRIVATE pointer associated with one
element of an independently owned fixed-record allocation, while retaining
complete original team, callee and hidden-effect checks. All 93 record controls
pass; a corrected module suite adds 29 passes. The genuine 38-artifact fixture
passes transport and separate record-integrity checks, but still closes before
the native record call and reopens afterward. It neither establishes retention
through that call nor qualifies the actual wall-model closure.

The [executable record-call fixture](/tmp/fortran-owned-record-region-01/integration/runtime-prepared-03/oracle-correction-04/qualification.json)
matches native values, allocation bounds and halos at three sizes. Enabled
execution meets its original ledger: four launches, two uploads and four
downloads across two stages; the second read-only host access does not repeat
the input download. Both zero-trip/unallocated paths have empty traces. The
original controller exits with failure because its hidden-device oracle
incorrectly forbids host-only domain bookkeeping. A separately reviewed
analysis of those unchanged traces confirms zero GPU launches, copies,
allocations or submissions in fallback. That correction does not relabel the
original failed run. This remains a synthetic fixture with residency cuts;
uninterrupted retention and the actual wall-model closure remain unqualified.

A separate [complete-body candidate](/tmp/fortran-owned-record-body-02/final-08/qualification.json)
passes 291 affected tests and independent review. It reuses the existing storage
effect scan for the whole original callee, including fixed-size function results
and host-defined constant bounds. It is excluded from maintained24. On the
unchanged real [SGS→Convection query34](/tmp/fortran-sgs-convection-storage-probe-34/qualification.json),
setup takes 92.286 seconds and the proof takes 88.196 seconds. The next rejection
is the original diagnostic `error_stop` call, not a timeout. Compose the existing
original terminating-call proof before adding broader character support. No
parent residency or execution is qualified by this result.

The isolated [terminating-call integration](/tmp/fortran-record-termination-03/final-21/qualification.json)
subsequently passes all 386 affected tests and [independent review](/tmp/fortran-record-termination-03/runtime-acquisition-review-03/review.json).
It reuses the exact original diagnostic-and-STOP proof and keeps one complete
storage-effect scan; it adds no general character-call permission. Fresh
[SGS→Convection query35](/tmp/fortran-sgs-convection-storage-probe-35/qualification.json)
passes that earlier barrier but still declines on an incomplete mutation or
escape closure in the wall-model record call. Setup takes 100.937 seconds and
proof takes 122.289 seconds, within their original limits. All frozen inputs
hold. The aggregate rejection does not identify its first underlying effect;
the next diagnostic must retain those rows. This candidate remains outside
maintained24, and the real parent residency gate remains open.

The isolated SELECT CASE extension passes 423 affected tests with the one
separately covered emission test excluded. It scans the selector, case bounds
and every arm through the existing effect machinery. With that extension,
[query36](/tmp/fortran-sgs-convection-storage-probe-36/qualification.json)
finishes setup in 84.818 seconds but reaches the unchanged 180-second proof
limit during repeated terminating-call receipt validation. It produces no
admission verdict. Its diagnostic also exceeds the declared value capacity;
no partial effect inventory is treated as complete. The next compiler change
must reuse validated immutable facts while retaining exact source mutation
checks, and the diagnostic must represent shared provenance without repeating
its entire tree. Neither timeout warrants weakening the storage proof.

The terminating-call immutable-fact implementation and accumulated storage
changes now pass a [445-test combined gate](/tmp/fortran-storage-chain-live-composition-01/final-01/qualification.json)
across 13 complete files. Independent review passes and the exact 14 changed
files are promoted. [Query37](/tmp/fortran-sgs-convection-storage-probe-37/qualification.json)
then finishes setup in 88.429 seconds but still reaches the original 180-second
proof limit during repeated owned-record validation. No admission follows.
The subsequent [shared dependency receipts and owned-record reuse](
/tmp/fortran-owned-record-authority-arena-01/final-13/qualification.json)
pass 428 focused tests and independent review; six files are promoted. Original
public records remain unchanged. [Query38](
/tmp/fortran-sgs-convection-storage-probe-38/qualification.json) finishes setup
in 84.043 seconds and again reaches the unchanged 180-second proof limit,
without a semantic verdict. Its complete selected-record diagnostic succeeds:
96 effect rows and all direct dependencies for the active wall-model record
are retained. The subsequent [private receipt composition](benchmarks/reports/gpu-residency-implementation-20261023/record-receipt-cost-addendum-01/README.md)
passes 458 checks and is promoted: checkpoints inherit exact expected metadata,
capture only new closure dependencies and validate one deduplicated union.
Query39 nevertheless reaches the same 180-second proof limit after 99.961 seconds
of setup, now in repeated deep receipt comparisons. Its selected diagnostic is
invalid because no active record-call token was available; zero captured rows
do not establish an empty effect set. Neither query supplies a semantic verdict.
Both primary configurations normally reach the wall-model call, so treating it
as an untaken branch would not resolve the application cut.
The subsequent [query40 diagnostic](/tmp/fortran-sgs-convection-storage-probe-40/qualification.json)
uses the same compiler and source inputs with a separately declared 600-second
proof allowance. It reaches a semantic decline after 227.621 seconds of proof
and postmortem work: the `TIBPoint_Viscosity` call in `SubgridStresses` lacks a
supported source-function storage mutation/escape closure. This is a concrete
next compiler gap, not admission of the shared interval or a compiler speedup.
The earlier 180-second failures remain recorded. Its record-call diagnostic is
invalid because this failure belongs to a different proof family.
The [read-only record function capability](/tmp/fortran-readonly-record-functions-01/final-14/qualification.json)
is now promoted after independent review. It recognizes exact scalar
`INTENT(IN)` record arguments and nested allocatable component reads while
retaining complete callee storage effects and original scalar indices. Unknown
calls, descriptor mutation and unsafe associations still prevent retention.
The final production passes 521 affected checks; all 54 new controls pass after
a test-only selector correction. It grants no payload or execution authority.
The actual [query42](/tmp/fortran-sgs-convection-storage-probe-42/qualification.json)
still declines the same function closure after 264.050 seconds of proof and
postmortem work. A visible derived-record assignment interface causes the
storage scanner to reject unrelated intrinsic real assignments. Query41's setup crash is retained, and
query42 removes only periodic stack sampling. Neither establishes residency.
The [assignment-applicability correction](/tmp/fortran-assignment-applicability-01/final-07/qualification.json)
is now promoted after independent review and 840 passing checks across 21 test
files. It excludes concrete record assignment procedures only when their
original left argument cannot accept the original primitive target. Every
exported host/USE namespace remains checked; matching, polymorphic and unknown
procedures still block retention. Complete RHS effects remain intact. This
compiler result does not by itself close the application residency cut.
The first direct actual-source closure succeeds, but its inventory exposes an
existing registration bug: repeated generic interface blocks overwrite earlier
specifics. The [registration correction](/tmp/fortran-assignment-applicability-01/final-09/qualification.json)
is now promoted after 848 passing checks across the same 21 test files, including
both orders of matching/unrelated interfaces and ordinary generic calls. The
[corrected actual-source closure](/tmp/fortran-tibpoint-viscosity-closure-02/qualification.json)
then preserves both `ScalFlIBPtoIBP` and `VelIBPtoIBP` in every exclusion and
reports no storage mutations, escapes or unresolved effects. The enclosing
SGS→Convection interval still requires its separate checked-window qualification.
The [next complete query](/tmp/fortran-sgs-convection-storage-probe-43/qualification.json)
clears that function-effect frontier but declines at the fixed direct-call
closure budget on `TMarchRK3 → Convection → CDV_Variable_Z`. It finishes within
its deadline; no shared residency is admitted. The
[source-sized closure change](/tmp/fortran-source-sized-closure-01/final-04/qualification.json)
now replaces the default 64-procedure count with the complete original parser
inventory and separates factory request counts from individual lookup budgets.
It passes 710 distinct affected cases across the recorded split gate, including
70-callee coverage, unsafe tails, recursion and explicit-cap controls. The
default depth of eight remains; larger metadata retention has no memory-cost
qualification. The original interval still needs its own admission result.
Later consumer-validation and function-demand batching changes reduce repeated
work on matched small queries, but the unchanged source46 interval still reaches
its 600-second proof deadline. A subsequent
[focused compiler gate](benchmarks/reports/gpu-residency-implementation-20261023/compiler-residency-progress-addendum-01/README.md)
qualifies compact source-registry receipts with 685 passing cases. All original
validation checks remain; an exact owned-record issuance falls from 17.494 to
9.845 instrumented seconds with identical public bytes. This does not establish
whole-interval admission or application acceleration.
The [unchanged source47 query](/tmp/fortran-sgs-convection-storage-probe-47/qualification.json)
also reaches the 600-second proof deadline after 45.85 seconds of setup. All
original inputs hold; it reaches Convection→BuoyancyForce while rechecking prior
wall-model record dependencies. It issues no semantic verdict or epoch. The
subsequent change partitions original CALL/IF regions around control-local lifetime cuts,
rather than raising this deadline or retrying another metadata-only optimization.
Each retained region still needs its own descriptor proof. Cuts must have
complete synchronous/no-escape lifetime evidence; an unknown predicate cuts
before the whole branch, and an unproved escape keeps native execution. This
[control-local region slice](benchmarks/reports/gpu-residency-implementation-20261023/control-local-residency-addendum-01/README.md)
is now integrated with the strict version-2 reader: 70 affected CPU checks,
two CUDA/hidden-device checks and 940 reader checks pass. Its conditional-swap
control preserves both GPU children: no between-child transfer on the untaken
branch, publication and fresh registration on the taken branch. Fields, bounds
and unallocated zero trips match native. Runtime ABI and application source
remain unchanged; actual SGS→Convection admission remains open.
Separately, generic native IF unions and stable entry-prefix scalar copies now
derive `BoundViscosity`'s six read and six possible write faces. Its 130-check
gate preserves conditional values and grants no logical-definition or lifetime
authority. A subsequent generated CUDA control passes two sizes and both branch
outcomes, transferring only 16 bytes each way between its GPU children while
retaining interiors. Actual ELMM parent admission and its transfer ledger remain
open.
The separately qualified selected-record diagnostic retains all rows of each
selected active record; it explicitly does not claim the complete registry or
semantic frontier. Full-registry output still exceeds its finite capacity.

The separate [original-call origin query](/tmp/fortran-program-call-origin-source-01/reviewed-source-04/qualification.json)
now issues origins for both RK pressure-call occurrences and the initialization
pressure call. All 927 compiler/fixture, 95 original and 73 configured input
guards hold. The source-only fix permits an exact original CALL to remain a
payload boundary while its allocation origins are checked. Checked dispatch
must still carry that evidence at every hop; unknown/public entries retain the
native route. The excluded multigrid call still needs independent hidden-effect
and no-escape evidence. These origins grant no execution or lifetime authority.

The [selective-footprint assessment](/tmp/fortran-excluded-edge-ownership-01/selective-footprint-assessment-01.json)
identifies a narrower route than modeling every multigrid allocation: close the
original reachable access graph, including recursive components, and check
whether it can address the protected allocation. Incoming aliases remain a
separate obligation. The actual closure reaches `gesvx`; its interface is absent
from the configured source inventory and the existing FFT contracts do not
qualify it. Keep that external edge conservative until its precise ABI and
effects are established.

The isolated access-footprint producer passes 36 focused controls and independent
review. Its [original `PoissMG` query](/tmp/fortran-poissmg-footprint-source-01/reviewed-source-02/qualification.json)
finishes setup in 92.053 seconds and analysis in 1.953 seconds, with all
929 compiler/fixture, 95 original and 73 configured inputs unchanged. It reaches
ten source procedures but retains 107 unknown-access records: 93 arise in the
open LAPACK namespace and related resolution, four concern directive clauses,
and ten concern diagnostic output or termination. These are grouped causes,
not 107 independent compiler features. Phi/RHS are present as positive controls;
the other 14 queried declarations are absent as direct identities, but the
incomplete closure excludes none of them. Incoming aliases and checked dispatch
remain separate obligations. No pressure execution or retention is admitted.

The [complete maintained22 suite](benchmarks/reports/gpu-residency-implementation-20261023/maintained-baseline22-addendum-01/README.md)
passes 9,728 tests with no failures, errors
or skips; 1,175 explicitly marked CUDA tests are deselected. This is the frozen
889-file compiler basis, preceding the separately qualified kind-import and
descriptor-factory changes. It does not qualify later changes automatically.

### C. GPU coverage and host regrouping

Build checked graphs for reached RK work, including Convection, diffusion,
boundaries and pressure preparation/correction. Extend generic source support and
verified effect summaries. Move independent host tasks around GPU groups; use
precise native transactions for dependent work. Forced placement is diagnostic.
Exit only when profiles show intended work actually on GPU and application wall
time falls; generated CUDA alone is insufficient.

### D. Complete pressure projection

Integrate an explicit ELMM adapter via the generic external-task contract.
Packing, FFT and unpacking use the runtime stream or explicit event handoffs.
Provide a status-returning C ABI with by-value `c_ptr` device pointers and catch
C++ exceptions. Budget persistent plans, workspace and temporary peaks.

Match real lengths, boundary types and FD2/FD4, never spectral defaults. Verify
dense Fortran layout and physical origins before accepting Phi/RHS ghost widths
of two/one. Write only Phi interior, preserving RHS and untouched halos.

Qualify producer→solver→halo-fill→consumer, a full pressure invocation, then an
RK stage. Integrate velocity boundaries and mass-flux correction, RHS, solve,
`Bound_Phi`, velocity/pressure correction and approved pressure referencing.
No host payload transfers may remain within that qualified GPU chain.

The [fresh current-runtime backend build and numerical gate](benchmarks/reports/gpu-residency-implementation-20261023/current-pressure-backend-addendum-01/README.md)
pass 14 pinned-PoisFFT differential cases and ten independent FD2/FD4 cases.
Actual loader records establish the new library in all eleven test processes;
all 370 input guards hold. Maximum manufactured-solution error is
`1.33e-15`, and maximum discrete residual is `7.82e-14`. The rebuilt library
differs from the historical binary, so these are fresh tests rather than a
transferred numerical qualification. The subsequent
[actual standalone solver integration](benchmarks/reports/gpu-residency-implementation-20261023/standalone-gpu-poisfft-addendum-01/README.md)
passes eight short diagnostic processes across both configurations, including
all 20 field layouts, bounds, halos and VTK outputs. Each smoke candidate commits
seven original solver calls on GPU; each profile records 77 kernels. The seven
separate domains and 84 module loads/unloads per case establish repeated provider
lifecycle, while GPU kernels total about 8 ms. Direct plan-creation time is not
measured in that profile. The subsequent [direct call diagnostic](benchmarks/reports/gpu-residency-implementation-20261023/solver-call-timing-addendum-01/README.md)
measures seven solver creations at 2.52/2.63 seconds total for simple/channel;
medians after the first create are 0.217/0.211 seconds. All four fresh short
applications pass strict field and output checks. Tracing is enabled, so these
host intervals are not performance samples or a rigorous untraced startup
lower bound. Retaining a checked domain/provider is the next priority;
surrounding pressure work remains native, and long natural runs,
projection-wide residency and production performance remain unqualified.

Plan reuse can be delivered before cross-stage field retention. The smaller
slice uses an explicit enclosing owner for the solver domain and workspace,
forwards that ownership through selected original calls, then publishes and
unregisters every field before each return to native work. No host field pointer
or replica survives that boundary. Each later call checks the current query,
layout and configuration and registers fresh field handles. The measured
repeated setup cost justifies selecting the named main-program owner across
the actual time loop, through `TMarchRK3` and `PressureCorrection`, rather than
recreating the plan once per timestep.
Unknown dispatch uses the existing standalone path. This changes solver storage
lifetime, not the proof requirements for retaining fields across native work.

The [provider-only lifetime implementation](/tmp/fortran-provider-only-lifetime-01/live-rebase-01/closure-01/qualification.json)
is now promoted after independent source review and 379 focused passes, with
six unchanged opt-in skips. It requires a separately reviewed provider-state
contract, checks the original call path, preserves standalone callers and
retires the plan before closing its domain. Rejection before submission disables
that owner's GPU path; failure after submission is fatal. The current pinned
solver's retained-state review is bound to the exact provider artifacts.
Actual main-program generation and the public build now pass. The first short
application run agrees in every field/output but falls back to the native solver:
the native prefix creates a context outside the provider lifetime, so the later
field registration has generation zero and is rejected. The runtime boundary
behaves correctly. The [context-creation correction](/tmp/fortran-provider-lifetime-prefix-01/qualification-02.json)
is now promoted after independent review. Native and external paths borrow the
same owner and generation, normal publication preserves the private plan, and
rearming respects a prior rejection. Sixteen focused controls pass, including
real native prefixes, PROGRAM owners and stale-context rejection. The earlier
121 passing broader checks and corrected test-oracle history are retained.
Fresh actual generation, strict reading and the public build pass. The
[corrected short/profile gate](/tmp/fortran-retained-poissfft-owner-01/qualification-diagnostics-02.json)
passes eight fresh processes across both configurations, including every field,
bound, halo and output. Each candidate uses one initialization plan and a second
plan for all six main-loop solver calls. Both domains retire and all seven
borrowed contexts detach. Field traffic remains unchanged: every solver return
still publishes its fields. The subsequent [natural-length check](benchmarks/reports/gpu-residency-implementation-20261023/retained-pressure-plan-reuse-addendum-01/README.md)
passes simple109 but **fails channel2458** at the unchanged tolerances: 15
populated fields and 16 VTK outputs disagree, with maximum field error about
`3.18e-4`. Both applications complete all steps and outputs. The long candidate
uses two plans for 7,375 solves, but correct lifecycle reuse does not establish
scientific correctness. The GPU solver remains unqualified for the full channel
case pending diagnosis; field retention and end-to-end speedup are unproven.

The [same-candidate CPU fallback control](/tmp/fortran-retained-poissfft-owner-01/qualification-natural-hidden-01.json)
passes the full 2,458 steps with bit-identical values in all 20 fields and all
114 outputs passing. It executes no GPU solver calls. The divergence is therefore
specific to the exercised GPU path; its cause remains unresolved. Differential
checks at the actual channel geometry pass for both a [single RHS](/tmp/fortran-poisson-actual-geometry-differential-01/qualification.json)
and [64 changing inputs through retained plans](/tmp/fortran-poisson-plan-reuse-differential-01/qualification-02.json),
with maximum solution differences `8.53e-14` and `6.11e-16`, respectively.
RHS, halo, gauge and residual checks pass. These bounded checks do not qualify
the full trajectory or waive its unchanged tolerance gate.
The [seven actual short-run solver inputs](/tmp/fortran-poisson-original-rhs-capture-01/qualification.json)
also pass differential, residual, gauge, RHS and halo checks, with maximum
solution difference `1.11e-15`. All 20 field payloads and shapes match the earlier
GPU run byte-for-byte. The diagnostic's stricter complete-byte gate fails only
for six `Retau.unf` reduction values differing by 2–5 ULPs; all normal field and
output tolerances pass. The same diagnostic output varies in prior GPU runs.
This does not establish correctness of the long evolving trajectory.

The [seven late solver captures](/tmp/fortran-poisson-late-rhs-capture-01/qualification.json)
also pass the same local checks, including all three RK stages at steps 1,989
and 2,458. Maximum GPU/CPU solution difference is `2.60e-18`. The complete
diagnostic run reproduces the previous GPU fields and spatial outputs exactly,
including the 15 field and 16 VTK tolerance failures. Its seven added capture
synchronizations therefore do not change the observed trajectory. Unsampled
calls and earlier RHS construction remain outside these local checks; the
full-channel correctness gate remains failed.

A [CPU-only sensitivity control](/tmp/fortran-poisson-cpu-ulp-sensitivity-01/qualification.json)
flips the lowest mantissa bit of each finite pressure-solver output, after all
7,375 original CPU solves. Four native threads and the exact FFTW plans remain
unchanged. It completes 2,458 steps but also fails 15 field and 16 VTK comparisons,
with maximum velocity difference about `3.17e-4`. Tiny solver perturbations can
therefore trigger this long-run tolerance failure. This diagnostic neither
replaces the native reference nor establishes the GPU discrepancy's sole cause.
The [direct trajectory comparison](/tmp/fortran-poisson-cpu-ulp-sensitivity-01/comparison-01/qualification.json)
finds that perturbed CPU and GPU agree within the original tolerances in all
20 fields and all 114 outputs. Their first stored pressure and velocity failures
against native occur at the same frames. This strengthens the sensitivity
evidence without changing the original-reference acceptance gate.

The generic [external entry-read capability](benchmarks/reports/gpu-residency-implementation-20261023/external-entry-reads-addendum-01/README.md)
is now promoted after 112 affected host passes (six existing numerical opt-in
skips), independent review and seven GPU controls. It binds a distinct reviewed
original-library must-read contract to exact source/query/provider artifacts.
Admission occurs only after matching layouts and successful provider creation;
existing current device regions remain intact. The completed-producer/native-read/
external-consumer control has no second input upload. This is a prerequisite
for the separately qualified short solver integration above. Fresh standalone
`Poiss_PoisFFT` generation, the strict package reader, public application build
and short field comparisons pass. The public `prepare_lazy_poisfft_contract`
now accepts an explicit reviewed `required_entry_read_prerequisite` and returns
the final provider, task and source proof together. Its affected host gate has
41 passes and six unchanged native opt-in skips; one fresh real-source
preparation matches the already qualified task, provider binding, RHS region
and six guards. No numerical result alone grants entry-read or lifetime authority.

### E. RK stages and timesteps

Prove allocation exchanges so mirrors follow storage, retire unknown mutations,
and extend domains across the adaptive time loop. Port frequent boundary,
diagnostic-reduction and scalar-control work as measured. Transfer necessary
scalars without publishing fields. Reuse plans/storage; budget fields, scratch,
workspace, graph objects and snapshots together. Evict unused clean replicas
first. Exit with no repeated full-field traffic on ordinary non-output steps and
a semantic reason for every remaining physical copy.

### F. Launches, output, kernels and cleanup

Replay stable CUDA graph segments after residency/coverage are correct; external
solvers are capture boundaries until demonstrated safe. Fuse only where
numerics/dependencies and measured whole-process time permit. Optimize remaining
measured kernels, including SGS.

A [bounded Nsight Compute attempt](/tmp/fortran-sgs-hardware-profile-01/result.json)
on the qualified candidate52 binary fails with `ERR_NVGPUCTRPERM`. Inputs remain
unchanged and the process group exits cleanly. No hardware-counter measurement
or kernel bottleneck attribution follows from this attempt.

The [checked queued-source slice](benchmarks/reports/gpu-residency-implementation-20261023/checked-queued-owner-addendum-01/README.md)
is now promoted, with six actual CUDA modes and a passing current-composition
gate. Default artifacts and checked-plan identities remain byte-identical. Its
single-kernel contract is now extended by the promoted
[complete-team transaction](benchmarks/reports/gpu-residency-implementation-20261023/checked-queued-team-addendum-01/README.md):
one complete upfront team plan, irrevocable first acceptance, one final flush
and ordered scratch reuse. All six CUDA modes pass on the original three-unit
fixture, including an 18-launch geometry. Conservative effects grant no partial
definitions; a metadata resource budget does not truncate work. Existing defaults
remain byte-identical. The retained actual ELMM callback trace has no repeated
provider key within a context, so warmup-first graph replay cannot help that
route yet. Qualify a wider owner lifetime or measured first-use graph pricing,
then measure application benefit before selecting this transport by policy.

The isolated [queued-graph protocol control](/tmp/fortran-queued-graph-calibration-prototype-01/qualification-runtime-01/qualification.json)
passes 28 host tests and both GPU processes: 60 observations cover 600 logical
kernels, six graph builds and 18 replays. Owned arguments, allocation generation
changes, scratch lifetime and field/halo checks pass with all 972 guarded inputs
unchanged. This qualifies a small measurement/control harness. Current generated
source providers still use direct submission; they need a checked queue adapter
before representative graph costs can influence placement. No cost fit or
application speedup follows from these controls.

An explicit, unpriced 16 MiB source-batch target is a separately qualified
launch experiment. Candidate52 passes its short all-field correctness/profile
checks and reduces channel tile launches from 324 to 108 with unchanged copy
bytes and CTA work; tracked payload-plus-scratch peak is 55,926,800 bytes.
Default emission is unchanged. The [candidate52 packet](benchmarks/reports/gpu-residency-implementation-20261023/candidate52-correctness-launch-addendum-01/README.md)
also passes natural 109/2,458-step checks of all 20 fields, bounds, halos and
outputs at unchanged tolerances. The subsequent [complete-process campaign52](benchmarks/reports/gpu-residency-implementation-20261023/four-mode-checkpoint-52/README.md)
fails the performance gate: candidate medians are 5.80% and 2.14% slower than
fresh `7a612e4` in simple109 and channel100, with one win in seven paired
comparisons in each case. Fewer launches alone have not established an
application gain, and this experiment does not justify changing placement policy.

Output retains all frames and contents. Use at most two immutable device/pinned
snapshot slots, capture before conflicting writes, wait for DMA before writing,
apply backpressure and join the writer before process completion.

Remove unused bridge/catalog/placement prototypes, duplicate policy paths and
obsolete options, with clear errors for removed options. Unsupported application
configurations keep a CPU path. MPI/multi-GPU expansion is deferred.

## Qualification

Memory tests cover unchanged CPU→GPU→CPU→GPU data, overlap, partial writes,
overwrite without upload, address reuse, swaps, zero-trip/unallocated paths,
SAVE state, eviction, queued-reader lifetime, unmanaged returns and failure
before/after submission. Check fields and physical-copy ledgers.

Solver tests cover FD2/FD4 manufactured solutions and matching discrete
residuals, channel boundary combinations, null space/gauge, odd/noncubic grids,
RHS preservation, ghost canaries, nondefault streams, workspace exhaustion and
complete pressure/velocity projection.

Retain approved deterministic pressure reference, four native threads, unchanged
physics and `atol=1e-6`, `rtol=1e-5`. Validate all 20 fields, bounds, halos and
outputs at natural 109/2,458 steps. Run maintained suites before releases.

Performance uses fresh native, be58, `7a612e4` and candidate builds; one excluded
warmup and seven rotated measurements, complete-process timing and all samples.
Primary cases are simple109/channel100. Longer amortization studies cannot
replace them. Median acceptance and seven-of-seven paired wins are distinct.

1. ≥10% lower median than fresh `7a612e4`, in both cases.
2. ≥2× native, in both cases.
3. ≥20× native, in both cases, with full qualification.

If the measured ceiling prevents 20×, publish the remaining gap without changing
the workload or timing definition.

## Implementation ledger (in progress)

Evidence paths below are development records, not released performance claims.

| Area | Implemented | Current verification / remaining gate |
| --- | --- | --- |
| Checked storage epoch | Exact source/plan/member bindings, managed captures versus opaque native reservations, retained cuts, lazy domain activation; checked-plan schema 2 with strict companion reader | Generated GPU→host read→GPU execution passes; unchanged inputs upload once and only the required cell is downloaded. Repeated allocated producer/consumer and unallocated zero-trip controls pass. Complete IF selections and 150 same-callee CALLs retain one issued proof and an original unconditional cut. Native convergence loops, readonly function tokens, local SAVE storage and optional scalar descriptor support pass focused checks; actual RK/projection epoch admission remains open |
| Conditional private allocations | Original independent SAVE scratch may change while unrelated caller fields remain live; descriptor21 and epoch2 retain its original declaration and mandatory absence from every canonical registration | [1,584 source / 63 reader / 733 combined checks](benchmarks/reports/gpu-residency-implementation-20261023/graph-lifetime-contracts-addendum-01/README.md) pass in their overlapping scopes. Descriptor22 adds 611 overlapping checks. Exact source30 clears the joined parallel group and stops at orphan worksharing at line197. The supported-path600-second and pressure18 setup240-second runs are incomplete; repeated proof-validation cost remains. No actual cross-stage epoch is granted |
| Complete source grouping | Removed the obsolete 32-unit numerical partition veto; full original source, directive, span, clause and numerical legality checks remain | 49 checks pass, including exact33/65-unit source bodies and rejection of an unsafe tail after65 units. This qualifies source partitioning; independent non-graph worker/operation limits and application execution gates remain |
| Domain runtime | Caller-owned coherent storage, borrowed contexts, aggregate physical statistics, clean-first eviction, budgeted workspace leases, native-hint veto and versioned domain attribution | Six actual CUDA domain/failure/eviction cases pass. The runtime uses the existing coherence engine; a read adds a replica and does not invalidate its source |
| Owner-local CALL groups | One fresh domain spans an exact original adjacent CALL suffix inside a reached IF branch; checked-plan4 carries the typed original sequence exit and checked child handles | [80 CPU / six CUDA controls and a separate overlapping 1,241-check compiler/reader gate](benchmarks/reports/gpu-residency-implementation-20261023/owner-local-call-group-addendum-01/README.md) pass. Shared read-only input uploads once per group; late rejection executes remaining coherent CPU work once; untaken unallocated paths create no domain. Qualified version2 starts definitions at NONE and closes before the branch exit. Actual Convection retention requires separate invocation-entry preservation evidence; no application gain is claimed |
| Shared source-team residency | Independently supplied invocation-entry definitions survive the exact original prefix; three checked source-team children share one local domain with original CALL mappings and fresh consuming descriptor proofs | [34 CPU / three separate setup checks and six distinct CUDA controls](benchmarks/reports/gpu-residency-implementation-20261023/shared-source-team-residency-addendum-01/README.md) qualify synthetic v3 retention. Five controls pass in GPU03; GPU04 corrects only the non-injected late-fault threshold on the same binaries. Full fields/halos/native observations pass, and a three-cell host write reuploads only 24 bytes per invocation. The exact emitted plan4/epoch3 manifest passes strict transport validation. That packet is synthetic; subsequent checkpoint51 separately qualifies actual Convection groups and natural application correctness, with a failed primary performance gate. Forced sections remain unpriced |
| Allocation identity | Source proof for an ordinary three-MOVE_ALLOC swap, post-call descriptor verification, physical generation/replica forwarding, fatal failure after mutation | Two generated native/GPU execution tests pass with different bounds/extents, queued writes, untouched halos and 64 bytes in each transfer direction |
| External transaction | Typed layouts/access regions, runtime stream, explicit submission state, poison on attempted failure; exact source/provider/configuration/box qualification | Four actual array-only cases, six original-wrapper processes and six shared-ancestor processes pass. Version 2 preserves the original saved receiver, constructor, clocks and debug operations. Two checked solves share one domain/provider/workspace; an intervening host read downloads one cell and does not cause a repeated RHS upload |
| Whole-array source support | Typed extent/conformance guards; original descriptor coordinates without unused narrowed lower bounds; constant scalar actual arithmetic stays at the original call | Three actual CUDA whole-array cases pass. Original/ENTRY SAVE scalar sharing passes native, hidden-CUDA and GPU execution |
| GPU solver | Stream-aware status ABI, exception boundary, exact allocation accounting, pinned source package and VkFFT, out-of-tree install/export | 14 differential cases / 15 solves and ten independent manufactured/discrete-residual cases pass. The actual FD4 operator is the seven-point composition of staggered fourth-order gradients; the conventional five-point second derivative is not its residual operator |
| Pressure adapter | Checked dense views, workspace reservation, exact interior output and six original halo-face operations; optional authenticated backend build | Resident producer→solver→halo-fill→consumer passes for FD2/even and FD4/odd grids without intermediate host payload transfers. The original solver wrapper passes; complete ELMM pressure projection remains pending. Recreating a plan per unmanaged call is expensive, and the constructor-preserving qualification path does not establish startup acceleration |
| Original pressure-correction coverage | One checked original PostPoisson atomic team and its ordinary pressure update compile and execute through runtime275 | [Both short cases and separate profiles pass](benchmarks/reports/gpu-residency-implementation-20261023/pressure/postpoisson-coverage-addendum-05/README.md), with all20 fields, bounds, halos, VTK and matching FFTW4 controls. Each records28 kernels/264,011,776 physical copy bytes over initialization plus two steps. These standalone wrappers add traffic and retain the CPU solver; complete pressure-domain residency and application gain remain open |
| Read-only allocatable forwarding | Source-bound numeric ALLOCATABLE INTENT(IN) association keeps the original descriptor ABI and guarded allocation status; no initializedness is inferred | [62 focused / 44 activation checks and eight direct correctness controls](benchmarks/reports/gpu-residency-implementation-20261023/pressure/readonly-allocatable-input-addendum-06/README.md) pass. Exact reached-read proof permits only its152B region; an unproved conditional read stays native with zero copies. Complete fields/halos, nonunit bounds, unallocated and hidden-device controls pass; whole pressure qualification remains open |
| Native retention | Source-sized proof walks retain complete native OpenMP teams; fixed read-only module arrays use opaque reservations without device or initialization authority; typed prepare/start/complete CPU source transactions reuse the existing coherence engine | The large-team native/hidden/GPU fixture passes. Four direct CUDA controls preserve readonly replicas with zero reupload; exact five-cell CPU writes require only 40 bytes of reupload. One generated GPU→CPU→GPU control passes with measured native-only pricing and the original CPU team executing once; GPU source pricing remains unavailable after failed fresh holdouts |
| Tile-private scratch | Original-source coverage, ordinary typed lowering, bounded scratch layout and scoped same-stream submission; complete producer and consumer evaluations retained; separately proved cooperative mapping | Sixteen CUDA cases pass with bitwise REAL4/REAL8 full-array/halo comparisons. All nine CD4 units compile; the batched provider passes 48 execution and 12 failure-after-submission cases. Candidate 33's channel profile shows 324 tile-private advection kernels. Seven measured local AUTO controls pass on the historical full-close endpoint; live native retention and whole-application profitability remain open |
| Rectangular point mapping | Multidimensional cooperative blocks distribute every proved independent point through overflow-free mixed-radix coordinates; original bodies and unconditional phase barriers remain | [114 CPU / 80 CUDA checks](benchmarks/reports/gpu-residency-implementation-20261023/rectangular-kernel-addendum-01/README.md) pass; all nine original CD4 units lower and compile. Actual execution was pending in that packet; subsequent candidate50 below qualifies actual channel CD4 coverage and correctness. One-dimensional artifacts stay byte-identical. One synthetic warmed resident comparison falls from 21.307341 ms to 8.091696 ms for 64 invocations with seven paired wins; startup/output are excluded, so this is not an ELMM or complete-process gain |
| Original-index access envelopes | Unpriced source tile teams acquire checked per-access regions through the existing coherence engine; writes preserve prior values and graph hazards remain conservative | [85 CPU / 12 CUDA / 385 reader checks](benchmarks/reports/gpu-residency-implementation-20261023/source-access-envelopes-addendum-01/README.md) pass. Read-only host access preserves GPU input; five edited interior cells alone reupload, with no halo upload. Strict provider7 grants neither definitions nor overwrites. Priced paths and changed-cost qualification remain open; later checkpoints50/51 separately qualify actual ELMM execution |
| Connected scheduling | Deterministic connected-GPU prefix pricing with shared transfers, independent alternative state and clean-first eviction estimates; issued source runtime choices are represented in the same graph | 53 scheduler checks pass, including 150 tasks retaining one upload with a one-evaluation search budget. Deferred tasks carry unknown target/cost until runtime selection, with conservative ordered tails and dependency verification. Source bridge qualification adds 27 CPU/two CUDA controls; one uses measured native-only pricing, the other synthetic GPU ranking for dispatch correctness. Strict companion transport passes419 checks. These are not application profitability results |
| Exclusive budget | Original-body instrumentation, main-thread exclusive intervals, worker CPU measured separately, coarse compile-time callback selection | [Phase A evidence](benchmarks/reports/gpu-residency-implementation-20261023/phase-a/README.md) preserves 26 diagnostics. The separate candidate51 budget has [author qualification](/tmp/fortran-candidate51-coarse-budget-01/qualification.json), [portable summary](benchmarks/reports/gpu-residency-implementation-20261023/four-mode-checkpoint-51/separate-instrumented-cpu-budget.json) and [independent exact raw-partition review](/tmp/fortran-candidate51-coarse-budget-01/benchmark-runner-independent-review-01/review.json): four diagnostic runs pass, including two full20 gates. Instrumented values are not production fractions, a ceiling, a cause of the 50-to-51 change or host/GPU overlap evidence |
| Calibration | Fresh domain-lifecycle/scoped-management and source/scratch/original-team measurements; raw evidence retained and inherited generic rates checked against hardware/toolchain | Frozen application diagnostics retain profile28/runtime `ddfca6e26af3d9cee469cdc1fd7767619b54f79a683e8454a491954f7b40a314`. Historical full-close profile31 passes its30 submit holdouts and seven local AUTO controls. Separately measured native-live campaigns40/43 qualify native management on runtime `275894c94f4c7b0fb5993155bec3047a7511814badea13970f34c24c7b4e85c1`; GPU source submission qualification fails. Campaign43 retains all77 raw rows and fails one fit point/three independent holdouts. GPU source AUTO pricing remains unavailable; synthetic dispatch checks do not qualify its costs |
| Source-submit diagnostic44 | Exact43 baseline plus forward/reverse observer and separate traced process; all four process/protocol checks pass with 95 unchanged references | [Portable raw evidence and independent reviews](benchmarks/reports/gpu-residency-implementation-20261023/source-submit-diagnostic44-addendum-01/README.md) retain all 3,864 untraced rounds, warmups, tails and signed wall/thread-CPU gaps. T1 preserves 1,932 rounds and 95,312 correlated launches/kernels with host API unions and nested `ioctl` accounting. Scheduling is unavailable and host/device alignment fails, so attribution remains **PARTIAL** with no causal or overlap claim. Historical43 setup lacks the newer production graph-enable call; no refit/profile publication, and GPU source AUTO pricing remains unavailable |
| Compiler practicality | Immutable original specification, provenance and selection indexes; directive scanners avoid formatting noncomments | Focused169/124/126/182 CPU groups overlap; the original skipped configured-source CUDA case separately passes. Historical full-Convection44 reaches900s without a package. Current generation completes in627.323s and all178 artifacts pass strict validation. Fresh SGS composition/build/relink/authentication, all eight short/profile processes and natural109/2458 correctness pass. The changed compiler/profile basis prevents a matched compilation-speed claim; this full-Convection build has no timing campaign yet |
| Active graph identity | Exact authority-keyed weak interning retains live source contracts after bounded cache eviction; source children without independent edit spans remain ordered | [137/67 focused checks and independent reviews](benchmarks/reports/gpu-residency-implementation-20261023/compiler-graph-identity-addendum-01/README.md) pass. The original RK graph clears both defects, but still admits no GPU tasks without lifetime qualification. No numerical, runtime-ABI or application-performance change is claimed |
| Application integration | Candidate 41 builds all seven SGS/CD2/CD4 roots with canonical helper identities and checked generated PUBLIC export composition; all eight short correctness/profile processes pass all20 fields, bounds, halos and outputs | Both cases execute advection on GPU: simple79 kernels/452,472,512 bytes/214copies and channel367 kernels/546,682,976 bytes/160copies over initialization plus two steps. Natural simple109 and channel2458 pass unchanged tolerances with all20 fields, bounds, halos and outputs. No application speedup is claimed. Shared SGS→Convection/projection/stage/timestep residency, launch/output work and final campaigns remain |
| Four-mode checkpoint50 | Fresh native/be58/7a612/candidate50 builds; new multidimensional mapping and default unpriced source access envelopes; independent SGS/CD2/CD4 roots and original CPU pressure. All eight short correctness/profile processes and eight separate natural full20 qualifications pass; be58/7a612/candidate field bytes match in both natural cases | [Exact 64-process checkpoint](benchmarks/reports/gpu-residency-implementation-20261023/four-mode-checkpoint-50/README.md): eight excluded warmups plus 56 measured samples, fixed rotation, four native threads/FFTW plans and full startup/output/solver/teardown clock. The 10% median gate **fails BOTH**: candidate is 16.8%/13.4% slower than fresh 7a612 for simple109/channel100, with **0/7 paired wins against both 7a612 and be58 in each case**. Small native median differences overlap variation; no robust gain. Separate profiles show actual channel 324 CD4 kernels within 367 total, 509,826,752 B/232 copies; simple keeps original advection2, 79 kernels/452,472,512 B/214 copies. Historical controls retain their original runtime/profile; candidate uses 275894c9/no profile/forced sections. No AUTO qualification, future call-group domain claim or reuse of these samples |
| Four-mode checkpoint51 | Fresh common49 sources, fresh SGS and original Convection owner with two checked three-call branch domains; actual08 source/28 GNU/13 CUDA gates, full public build and natural 109/2458 checks of all 20 fields pass. Candidate BIN and shape hashes match both GPU controls | [Fresh 64-process checkpoint](benchmarks/reports/gpu-residency-implementation-20261023/four-mode-checkpoint-51/README.md) retains eight excluded warmups and 56 samples with unchanged complete-process clock, rotation and FFTW4. The primary 10% and comfortable gates **fail BOTH**: 4.8%/6.5% slower than fresh 7a612, paired wins 1/7 and 0/7. Candidate medians are 13.8%/9.1% lower than native, with 7/7 paired wins each; 2×/20× remain unmet. Separate channel profile: 367 logical kernels including 324 tile kernels, 304 copies/347,849,408 B; simple retains advection 2, 79 kernels/178 copies/298,341,056 B. Source/API/copy and instrumented CPU diagnostics remain separate from untraced samples; no causal 50-to-51 or AUTO claim |
| Reviewed caller captures | [Source-bound audit](benchmarks/reports/gpu-residency-implementation-20261023/capture-storage-audit.json) records complete cell-type-array initialization and defined grid-spacing scalars before marching | The new `containers/requests/elmm-residency-auto.json` transports these facts. Neither opaque host reservations nor a numerical tile proof grants initializedness; unknown conditional scalar snapshots remain rejected |
| Cleanup | Removed the unused companion graph/catalog/UnifiedScopeBridge prototype and its self-only tests; removed strict-marker selection with explicit migration errors | Historical evidence remains intact. Marker comments and discovery metadata remain accepted; 110 affected frontend/discovery/CLI tests pass. Other option removal requires evidence that current callers have migrated |

Main working-tree changes and the nested solver/companion changes require separate
repository review/commits. The failed checkpoint51 performance gate is measured
evidence, not a completed 10%/2×/20× milestone or production AUTO qualification.
