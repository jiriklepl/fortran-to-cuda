# Fresh common-basis final comparison 23

The single predeclared campaign met inclusive candidate/be58 median parity in BOTH cases. Each case won 4/7 same-round comparisons; the separate comfortable criterion is FALSE in both. The small gains overlap observed variation, so this is not a robust/comfortable or statistical-significance claim.

All 48 timing rows are preserved: 42 measured and 6 excluded warmups. There were 59 fresh application processes in the final campaign (29 simple, 30 channel), zero retries, zero imported timing samples and no filtered rows. The 8 standalone natural diagnostic processes are separate.

Simple timings retain natural 109 steps. Channel timings use the identical 100-step cap in all modes; strict correctness retains natural 2458 steps. Both fresh candidate natural 20-field checks, native repeatability, VTK, exact four FFTW plans, current checked proof, actual separate CUPTI, and public/private postguards passed.

| Case | Native median (s) | be58 median (s) | Candidate median (s) | Candidate/be58 | Wall difference | Paired wins | Comfortable |
|---|---:|---:|---:|---:|---:|---:|---|
| channel-simple | 11.654671832 | 9.651284985 | 9.498940579 | 0.984215116823 | -1.578488% |4/7|false|
| channel | 13.258414592 | 11.453849783 | 11.354034333 | 0.991285423513 | -0.871458% |4/7|false|

## channel-simple

All samples below follow measured rounds 1–7; warmup round 0 is excluded.

| Round | Native (s) | be58 (s) | Candidate (s) | Candidate/be58 |
|---:|---:|---:|---:|---:|
|1|11.903427395000|9.498560960000|9.498940579000|1.000039965949|
|2|11.653707363001|9.999518867000|9.548801376000|0.954926082245|
|3|11.654671832001|9.698655861000|9.999028130000|1.030970504914|
|4|11.703068207000|9.499471191000|9.801991505000|1.031846016259|
|5|11.804233649000|9.799927899000|9.448697582000|0.964159908050|
|6|11.553210577000|9.499782654000|9.447622850000|0.994509368698|
|7|11.604794934000|9.651284985000|9.449908651000|0.979134764509|

| Mode | Median (s) | Q1 (s) | Q3 (s) | IQR (s) | Excluded warmup (s) |
|---|---:|---:|---:|---:|---:|
|native|11.654671832001|11.629251148500|11.753650928000|0.124399779500|11.704723768000|
|committed|9.651284985000|9.499626922500|9.749291880000|0.249664957500|9.547189088000|
|reworked|9.498940579000|9.449303116500|9.675396440500|0.226093324000|9.653426100000|

Candidate median wall time is 18.496713% lower than native. This does not satisfy the separate historical 20% native-reduction target.

## channel

All samples below follow measured rounds 1–7; warmup round 0 is excluded.

| Round | Native (s) | be58 (s) | Candidate (s) | Candidate/be58 |
|---:|---:|---:|---:|---:|
|1|13.257568026001|11.654426829999|11.402671648000|0.978398321456|
|2|13.357752903000|11.153878016999|11.302984560000|1.013368134632|
|3|13.258414592000|11.202505084000|11.253441391000|1.004546867564|
|4|13.208656557000|11.555249529000|11.255182503000|0.974031973499|
|5|13.461978716000|11.453849783000|11.354034333000|0.991285423513|
|6|13.208260253999|11.303747778000|11.451939924999|1.013110001206|
|7|13.261237048000|11.554139119000|11.354282717000|0.982702614194|

| Mode | Median (s) | Q1 (s) | Q3 (s) | IQR (s) | Excluded warmup (s) |
|---|---:|---:|---:|---:|---:|
|native|13.258414592000|13.233112291500|13.309494975500|0.076382684000|13.308939345000|
|committed|11.453849783000|11.253126431000|11.554694324000|0.301567893000|11.404811400999|
|reworked|11.354034333000|11.279083531500|11.378477182500|0.099393651000|11.151707285000|

Candidate median wall time is 14.363559% lower than native. This does not satisfy the separate historical 20% native-reduction target.

Quartiles use statistics.quantiles(n=4, method="inclusive"). The existing schema 2 target evaluator receives an explicit key-renaming projection of immutable public reports. Exact binary64 sample values are retained in JSON and all-timing-rows.csv; this Markdown rounds only for display.

No source/runtime/profile/model changes or further tests/apps occurred during analysis. Original reports and all raw streams remain unchanged.
