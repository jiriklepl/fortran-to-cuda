# Alternating filter experiment: rejected, 2026-11-09

The filter rewrite was removed from the implementation after losing all three
paired channel100 comparisons. The [workspace candidate](../gpu-resident-workspace-20261109/README.md)
remains selected, with its binaries and source unchanged. The experiment and all
samples are preserved; no slower candidate was promoted.

| Pair | Workspace control, seconds | Filter candidate, seconds | Candidate slowdown |
| --- | ---: | ---: | ---: |
| 1 | 4.760021 | 4.842925 | 1.74% |
| 2 | 4.695215 | 4.819500 | 2.65% |
| 3 | 4.670318 | 4.806499 | 2.92% |

Medians are **4.695215 s** for the control and **4.819500 s** for the filter
candidate: **2.65% slower**, with **0/3 paired wins**. The follow-up was declared
before execution: retain only if the median improves and all three pairs win.
One excluded warmup and three rotated measurements per binary gave eight complete
process runs. All outputs passed, external CPU use stayed below the predeclared
0.5-core threshold, and all owned processes were reaped. No builds, tests or
profiles ran alongside timing. Nothing was retried or trimmed.

Two earlier scouts are also retained. The first qualified filter version gave
4.057/4.687 seconds for simple109/channel100 versus 4.111/4.742 for the workspace
control: only approximately 1% lower time in single pairs. Removing per-pass
definition invalidation gave 4.128/4.767 seconds versus 4.142/4.626: simple stayed
neutral and channel regressed. The three-pair follow-up confirmed the regression
for that final version. These exploratory runs do not replace the seven-round
release protocol or its fresh controls.

## What was implemented and measured

The experimental adapter emitted three complete-result stencil kernels plus a
final copy. Successive directions alternated between the original field and one
domain-owned workspace, reducing three full snapshots to one final copy. Each
pass carried forward skipped cells and halos. Original weights, default-REAL
thresholds, seven-term accumulation order and even the degenerate dispatch's
repeated z pass were preserved and guarded by source fingerprints.

The first execution rejected an invalid conservative footprint before numerical
submission. The footprint analyzer had extended conditional neighbor reads into
halo iterations where the stencil never runs. Identity clamps inside the valid
interior guard selected the existing conservative full-input read effects.
This fixed execution without changing numerical values. It was an adapter
normalization, not a completed generic conditional-footprint redesign.

The first qualified version used INTENT(OUT) results, causing entry-wide
definition invalidation to drain queued users on each pass. The final version
used INOUT views with compiler-proved full-overwrite task effects instead.

| Separate 20-step channel profile | Workspace | First qualified filter | Final filter |
| --- | ---: | ---: | ---: |
| Filter kernels | 1,080 | 720 | 720 |
| All GPU kernels | 18,227 | 17,867 | 17,867 |
| Graph submissions | 625 | 804 | 271 |
| Stream synchronizations | 56 | 56 | 56 |
| Union of kernel activity | 0.566 s | 0.551 s | 0.561 s |
| Span from first to last kernel | 0.863 s | 0.832 s | 0.987 s |

Fewer launches did not establish lower application time. The final trace has
approximately 0.425 s outside active kernels within that span, versus 0.297 s
for the workspace control. This suggests changed submission/overlap behavior;
it is not an exclusive CPU attribution or a speed measurement. Host CUDA API
spans overlap device activity and must not be added. Future batching decisions
need measured collection/submission latency as well as device launch costs.
Storage residency must remain independent of those execution-batch decisions.

## Correctness and disposition

The focused filter milestone passed 18 odd/noncubic, masked and degenerate-grid
cases with three repeated filter applications each. Results were bit-identical
to original Fortran, including skipped cells, halos, a halo NaN payload and
signed zero. Repeated applications did not repeatedly upload field payloads.

Both short configurations and natural simple109/channel100 passed all 20 fields,
bounds, halos and outputs at `atol=1e-6`, `rtol=1e-5`, with four native threads,
the approved deterministic pressure reference and unchanged physics/output.
The final short trace shows each new filter entry launching 18 times in channel;
simple does not execute this filter. Managed entries show no generated-CPU
execution, and candidates create no CPU FFTW plans. All numerical source and
compile/link receipts are retained. Full maintained suites and natural2458 were
not rerun; the existing long-trajectory mismatch remains unresolved.

The unprofitable `elmm_pipeline/resident_filter.py` addition was archived and
deleted. No new production option, alternate coherence engine or unused filter
implementation remains. Existing external/mixed edits and the selected workspace
implementation were preserved.

Next work is production build integration of the retained complete numerical
GPU path, replacing the manual artifact chain with a reproducible source build.
Further performance work should use measured critical-path costs, including
startup/output and graph submission, instead of further iterations on this small
filter rewrite. Generic scheduling, legacy/configuration removal, fresh controls,
seven-round measurements and full correctness qualification remain open. The 20×
target and overall implementation goal are unchanged.

## Evidence

Root: `/tmp/fortran-resident-filter-20261109-01`.

- `entries-03/filter-entries.json`, `integrated-03/filter-integration.json`, `build-03/` receipts.
- `filter-milestone-03/result.json`, `milestone-03/`, `natural109-03/`, `channel100-full20-03/`.
- Both `comparison/` and `comparison-03/`, plus `paired-followup/{plan,result,execution}.json`.
- `profile/` and `profile-03/`, including Nsight reports, SQLite, CSVs and attribution.
- Failed first version, its rejection trace, and all preparation/build/run scripts.
- `implementation/resident_filter.py` and its source identity; `reproduce_module.py`
  loads that archived adapter when replaying the experiment scripts.

`summary.json` embeds the checkpoint evidence. No commit or push is claimed.
