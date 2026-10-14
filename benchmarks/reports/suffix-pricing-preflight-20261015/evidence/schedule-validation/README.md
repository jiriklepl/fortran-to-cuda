# Effective OpenMP schedule validation

The final source passed **99 host checks and the selected real-CUDA schedule test**. Both final runners recorded identical source inventories before and after execution. This is correctness evidence for protocol 4, with synthetic profitable costs in the CUDA selector test. It provides no application admission, calibration measurement, application performance or target qualification.

The CUDA test retains `schedule(runtime)` in the original loops and keeps `OMP_SCHEDULE=static` unchanged. Matching effective settings launch three workers. Changing the effective kind, chunk or monotonic modifier selects native execution with complete byte-identical float64 fields, including halos, and no residency runtime trace events. The accumulating output detects replay of a completed invocation or accumulation. The source guard also precedes CUDA identity queries and ownership.

| Execution | Result | Evidence |
|---|---|---|
| Initial host file | 98 passed | [Receipt](baseline-host/final.json), [log](baseline-host/pytest.log) |
| Initial complete CUDA file | 2 passed; new test failed to compile | [Receipt](baseline-cuda/final.json), [log](baseline-cuda/pytest.log) |
| Intermediate host file | 98 passed | [Receipt](intermediate-host/final.json), [log](intermediate-host/pytest.log) |
| Intermediate selected CUDA test | Failed the monotonic identity assertion | [Receipt](intermediate-cuda/final.json), [log](intermediate-cuda/pytest.log) |
| Final host file | 99 passed | [Receipt](final-host/final.json), [log](final-host/pytest.log) |
| Final selected CUDA test | 1 passed | [Receipt](final-cuda/final.json), [log](final-cuda/pytest.log) |

The first CUDA failure came from GFortran 15 not exporting `omp_sched_monotonic` in `omp_lib`; the fixture now sets bit 31 directly. The second failure exposed GNU's legacy Fortran getter dropping that bit. The [independent ABI probe](../abi-probe/receipt.json) established that the setter preserves it and the C getter returns it. The final independent native sampler uses the C getter, with a native regression covering the real modifier. Neither assertion nor admission policy was weakened.

The complete CUDA file was **not rerun on the final source**. Its two existing tests passed on the initial source; the new test alone passed on the final source. Final observations must not be described as a full-file CUDA pass.

`source-inventory-baseline.json`, `source-inventory-intermediate.json` and `source-inventory-final.json` bind the tested files; corresponding patches bind each working tree to parent commit `3e48c9d`. The final inventory and patch are [verified against committed compiler source](committed-source-binding.json) `9a027f8`, paired with pipeline `bbe44c1`. Identical host/CUDA inventories are stored once per revision. The initial host log was assembled from returned stdout chunks, and its selected source hashes were recorded at completion; subsequent runners record full before/after inventories and direct logs. See [metadata](metadata.json) for exact selections and provenance.

The saved runners contain their exact commands and CPU affinities. Reproduction requires copying a runner to a fresh diagnostic directory and using the matching source revision; do not execute it inside this immutable report. No binaries or generated fixture source trees are included.
