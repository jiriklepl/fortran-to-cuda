# Maintained performance comparisons

`benchmarks.harness.strategies` owns the generic six-policy comparison and
`benchmarks.harness.plot_strategies` reproduces its charts from the public JSON
report. They require an explicit compiler checkout and offline calibration;
neither imports compiler IR. The architecture defaults to the selected profile,
and compiler/profiler paths can be supplied explicitly. Calibration and timing
series belong to one hardware/toolchain/thread-budget identity.

```sh
python -m benchmarks.harness.strategies --compiler-root /explicit/compiler \
  --calibration-profile /explicit/profile.json --output /new/result \
  --cases pointwise boundary stencil compute \
  --modes native current sections auto chunked hybrid --host-threads 4 \
  --grids 64x48x32 128x64x64 256x128x128 --rounds 3 --warmup-rounds 1 --skip-profile
python -m benchmarks.harness.plot_strategies /new/result/report.json \
  --output /new/result/figures --label 'Reviewed compiler milestone'
```

The application campaign runner, memory guard, full-field validation and snapshot
sealer live in the independent pipeline repository as
`elmm_pipeline.campaigns`. Supply its repository explicitly. Its campaign JSON
selects committed inputs, approved build artifacts, all application cases and
additional commands such as the six-policy suite above. It runs the frozen
pipeline snapshot, saves the compatible repository pair, verifies input hashes
before each phase, separates diagnostics from timing, and preserves failed
attempts. No maintained tool imports or patches a dated result script.

Keep generated binaries, campaign configuration/evidence and immutable plots
under `benchmarks/results`; do not commit them. Commit maintained implementation,
fixtures, tests and documentation in their owning repository. Fresh baseline and
candidate samples must accompany each compiler milestone. Historical timings may
only be reused after exact relevant-artifact equivalence; a matching source
filename, shape or allocation address is insufficient.
