# Qualification requires complete comparison evidence

The pipeline could report **comfortable success without all seven native runs or without excluded warmups**. Target-gate schema 3 closes that gap for native, control and candidate together. Every available measured sample set must also match its reported statistics. Plotting and sealing recompute the current gates before publishing a report.

| Synthetic regression probe | Previous gate | Schema 3 |
|---|---|---|
| Seven rounds and one warmup for every mode | Comfortable | Comfortable |
| Missing native round 7 | Comfortable | Incomplete |
| All warmups missing | Comfortable | Incomplete |

These are synthetic evidence checks, **not application performance measurements**. The [before](evidence/before.json) and [after](evidence/after.json) records retain the observations, outcomes and evaluator hashes. The stricter replay supplies synthetic passed-validation facts explicitly. The replay script preserves its original workspace paths; it is an archived diagnostic, not a new public command.

Missing observations remain an honest incomplete diagnostic; contradictory summaries, duplicate rounds, invalid warmups and failed or traced samples reject. A partial median can still meet its numeric target, but cannot establish comfortable qualification. Older target-gate versions require their frozen plotter or a new report recomputed from complete evidence. Existing campaigns without per-case targets retain their historical interpretation.

The tested pair is compiler `09b10a502f1ed3f0307d83fc7e5412e776d9b27a` with pipeline `9babb9a` (full identities in [report.json](report.json)). Focused validation passed **92 tests** across `tests/test_campaign_targets.py` and `tests/test_campaigns.py`, plus Ruff and whitespace checks. [The receipt](evidence/receipt.json) binds the tested sources and final log. The first run's single test-construction failure is retained separately: its attempted mutation left the native ratio unchanged; correcting the test produced the passing run. No full compiler/CUDA suite or application campaign ran for this reporting-only fix.

Automatic application residency and both primary performance targets remain unfinished. The previous [application checkpoint](../diffusion-application-checkpoint-20261015/README.md) remains immutable and continues to show the isolated-diffusion regression. This slice prevents overstated qualification; it does not change those application results.
