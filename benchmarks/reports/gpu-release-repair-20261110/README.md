# Shared-interface and release-check repair — 2026-11-10

**A real Fortran ABI generation bug and a pipeline metadata mismatch are fixed.** Canonical names now reserve seven characters for `_handle`, so long array names produce valid identifiers within Fortran's 63-character limit. The migrated rank-15 native compilation check reproduced the overflow before the fix and passes afterward.

The pipeline reader now recognizes the compiler's completion record for consecutive serial loops. It retains the exact source, span, catalog, owner and checked-return-contract requirements. Historical single-loop records remain readable for frozen controls; arbitrary completion records still reject. No retired execution policy is restored.

Public CLI and integration fixtures now exercise shared artifact schema 2, graph scopes and current source-policy options. Native/CUDA compilation checks consume `shared_entry.cu`, `shared_interface.f90` and the common Fortran runtime. Retired ordinary output names, per-call policy and slab-transfer expectations are removed from these fixtures.

The first pipeline audit had 4,685 passes, six failures and 87 skips. One failure used a stale SCons environment whose Python could not import SCons; reruns select the working Python explicitly. The other failures came from retired-interface fixtures and the completion-record mismatch. The first focused repair run had 188 passes, three skips and the newly exposed identifier overflow. After fixing it, 41 focused compiler checks pass; the one sandbox-skipped CUDA execution check passes separately with device access.

The complete pipeline rerun reports:

`4695 passed, 87 skipped, 55 warnings, 519 subtests passed in 467.08s (0:07:47)`

The serial compiler analysis audit hit its declared 1,200-second deadline and remains an incomplete attempt. Its log is preserved. The replacement four-process analysis cohort is **FAIL** at report creation; its plan contains every selected test file exactly once. Native/CUDA-marked tests are excluded from that cohort, so it cannot establish full native/CUDA release qualification. Read [summary.json](summary.json) for terminal evidence when available.

All 130 ELMM numerical entries regenerated in 18.17 seconds with **zero changed CUDA sources**. Shared interfaces/runtime headers and all four executable identities remain exact against the selected asynchronous-output build. This uses byte-identical objects; it is not a fresh complete application/provider build. No new application speedup is claimed. The emitter-input identity changed, so no unchanged numerical-cost calibration is asserted.

The selected seven-round performance remains **3.78× native for simple109 and 3.70× for channel100**. The independent FD4 residual and startup diagnosis are recorded in the [solver milestone](../gpu-solver-residual-20261110/README.md) and [phase report](../gpu-phase-attribution-20261110/README.md). Full memory accounting, remaining legacy/test migration, release qualification, changed-cost calibration and the fresh four-control comparison remain open. The 20× research objective is unmet.

All failed attempts, source archives, test results, build commands and binary identities are linked in [summary.json](summary.json).
