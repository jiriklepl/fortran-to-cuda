# Native and local compiler comparison

`benchmarks.harness.compare` compares unchanged CDU/CDV/CDW Fortran sources with
fresh local compiler output at optimization levels 0 and 1. CPU runs use GNU
Fortran and C++; GPU modes also include original NVFortran, handwritten OpenACC
and generated CUDA. Generation, builds, commands and measurements stay in a new
or empty output directory.

```sh
# CPU correctness and owned-session validation on all cases.
python -m benchmarks.harness.compare --output /tmp/native-local-cpu --resident

# Add a bounded timing series.
python -m benchmarks.harness.compare --output /tmp/native-local-timings \
  --cases CDU --timing-grids 64x64x64 128x128x128 \
  --iterations 20 --warmup 3 --rounds 3 --resident

# Build GPU variants without executing GPU code.
python -m benchmarks.harness.compare --output /tmp/native-local-build \
  --gpu compile --resident --compute-capability 86

# Run GPU correctness and timing with the same native/local drivers.
python -m benchmarks.harness.compare --output /tmp/native-local-gpu \
  --gpu run --resident --timing-grids 64x64x64 --compute-capability 86

python -m pytest benchmarks/tests
```

Set `--fc`, `--cxx`, `--acc-fc`, `--nvcc`, `--cuda-host-cxx` and
`--compute-capability` for the installed toolchain and GPU. GPU modes require the
NVIDIA HPC SDK for the handwritten OpenACC control and a compatible CUDA toolkit.
CPU comparisons require no external source-transformation environments.

Correctness uses noncubic, singleton, cubic and partial-block shapes:
`5x4x3`, `1x1x1`, `16x16x16`, and `259x7x5`. Every interior value must be finite,
the output length must match, and values must agree with original GNU Fortran
within absolute tolerance `1e-10`. Unwritten output halos have no defined
original value and are excluded. Resident correctness repeats the call twice
before retrieving its output.

The two measured contracts remain distinct:

- **Drop-in:** each call starts from current host inputs and returns output to
  the host, including the implementation's allocation, transfer, execution,
  synchronization and cleanup.
- **Resident batch:** OpenACC uses an enclosing data region; local C++/CUDA uses
  its generated create/run/retrieve/destroy session. Inputs stay unchanged and
  only the final output is needed. The timer includes setup, final retrieval
  and teardown. CPU sessions also include their owned-buffer copies.

Both local optimization levels are retained: level 0 keeps the original four
passes, while level 1 applies checked fusion where supported. Normal `-O3`
settings and default fused multiply-add behavior apply; these are independent
kernel comparisons, not complete ELMM application timings.

GPU execution requires an in-device OpenACC probe and actual per-implementation
kernel traces. CUDA validation also checks allocation, transfer and launch
counts. Trace diagnostics are collected only during validation. Timing uses
identical initialized inputs, warmups and iteration counts, consumes a checksum,
and retains individual process samples. Failures are recorded in `report.json`;
`commands.json` retains commands, return codes and their output.

Historical comparison campaigns are preserved under `benchmarks/results`,
including the [earlier comparison](../results/tool-comparison-2026-10-02.json),
[local capability report](../results/compiler-capabilities-2026-10-02.json), and
[CPU/GPU rerun](../results/tool-comparison-rerun-20261002T155622Z/report.json).
Their compiler identities, dependencies and measurements describe those
campaigns. The removed external-tool adapters are no longer maintained or
required by the current runner.

For generic runtime-policy and complete-application campaigns, see
[campaigns.md](campaigns.md). Their independent compiler/pipeline interfaces,
calibration, thread budgets and acceptance gates remain separate.
