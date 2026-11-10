# Grouped resident entries and shifted-domain fusion, 2026-11-10

The compiler now exports adjacent serial loop nests as one checked entry, removes the 32-loop extraction cap, and fuses compatible constant-width domains with different origins. Explicit scoped placement now honors requested optimization; it previously forced optimization off. These changes reduce periodic-edge GPU launches by 75% in both short application profiles. They do **not** establish a substantial additional application speedup. The selected cached GPU baseline remains unchanged.

## Performance milestone

Both focused campaigns use complete-process timing, four native threads, one excluded warmup per variant/case and seven rotated measurements. Each GPU variant starts with its own fresh private code cache, populated by its excluded warmup. Native, the selected cached GPU executable and the candidate are freshly measured. All 96 processes pass output comparison and both campaigns pass the external-load gate. All samples, including slow ones, remain recorded; no traced or build work ran concurrently. The historical four-control comparison remains separate.

| Candidate | Case | Native median | Cached GPU median | Candidate median | Change vs cached GPU | Paired GPU wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Grouping only | channel-simple | 12.091225 s | 3.179428 s | 3.206666 s | +0.857% time | 3/7 |
| Grouping only | channel | 14.137050 s | 3.861494 s | 3.856452 s | -0.131% time | 5/7 |
| Grouping + fusion | channel-simple | 11.696086 s | 3.146443 s | 3.116725 s | -0.945% time | 4/7 |
| Grouping + fusion | channel | 13.053224 s | 3.827718 s | 3.814604 s | -0.343% time | 3/7 |

The declared promotion gate was at least 1% lower median time in both cases and wins in every paired comparison. Neither candidate passes. The fused candidate is 3.75×/3.42× its same-campaign native medians, but its incremental result is not robust enough to replace the selected baseline. Reducing wrapper and launch counts alone has not resolved the measured kernel/critical-path costs. [All samples, pairs and receipts](summary.json) are retained.

## Implementation and correctness

Original sibling identity and adjacency checks prevent a grouped selection from skipping calls, assignments, branches or OpenMP boundaries. The numerical dependency verifier still orders passes and proves every fusion, including read-after-write, anti- and output dependencies. Shifted fusion reconstructs each original iterator through a bounded ordinal; numerical expressions retain their order. The ELMM adapter checks source-nest coverage rather than the number of exported entry calls.

Boundary interfaces shrink from 128 to 46 and periodic interfaces from 20 to four, with exact coverage of all 148 original source nests. Fusion then reduces the 20 static periodic kernel regions to four. Runtime short-case ledgers show periodic kernels dropping from 372 to 93 in channel-simple and 516 to 129 in channel; all other procedure counts stay identical. There remain 31/28 executed numerical GPU procedures.

The focused milestone passes 133 tests. Both short cases, simple109 and channel100 retain all 20 fields exactly against the GPU control, and all primary outputs pass original native tolerances. Only boundary/periodic entries and the coordinator were rebuilt; other qualified application/runtime objects and the same explicitly supplied local Poisson provider were reused. This is not a complete source regeneration or full release suite.

## Approved natural channel2458 milestone

The user explicitly approved the 1% relative L2 criterion on 2026-11-10. A **new** fused-candidate process completes all 2,458 natural steps and passes all 20 fields and 114 outputs against the immutable four-thread native reference. The largest componentwise relative L2 error is **0.154896%** (frame-y-49.vtk:p). The largest local error remains reported: **1.467061% of peak reference magnitude** (fields-Pr.bin). Zero-reference components use maximum absolute error 1e-6. Bounds, layouts, field/component labels, coordinates and output/frame inventories remain exact; time columns retain the original pointwise check. Primary/operator rules are unchanged.

The criterion and inputs were recorded before this new run. The earlier native reference was reused and verified against its original SHA receipts; no fresh native process is claimed. The original pointwise failures and the earlier post hoc review remain untouched. This milestone qualifies the new run under the newly approved rule, without relabeling historical failures.

## Retained unsuccessful attempts

A wall-loop early-exit pilot preserves captured outputs but gives insufficient kernel improvement; no production change was retained. Cooperative pressure-plane loading preserves exact addition order but is slower on the primary plane; that source change was reverted. The initial entry preparation retained single-loop extraction and ended on a wrong temporary manifest path. The first fusion generation still had optimization disabled. A subsequent partial link needed the reused grouped-interface module path; a shared symbol-renaming regression caused four focused failures and was corrected before the final 133-test milestone. All failed logs and intermediate artifacts remain in the referenced evidence roots.

The next performance work must address measured kernel execution or larger independent task groups. The generic grouping/fusion capability remains implementation work under evaluation; the performance-selected executable is unchanged. Full public regeneration, maintained release suites, remaining policy/legacy cleanup and the 20× research objective remain open.
