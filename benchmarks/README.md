# Benchmark suite

Three momentum-advection kernels (CDU, CDV and CDW), seven implementation
variants, and a [Loki/PSyclone comparison](docs/tool-comparison.md). Run all
commands below from the repository root.

## Layout

```text
benchmarks/
├── Makefile                   # one build entry point for every case/variant
├── cases/<case>/              # maintained Fortran sources and callers
│   ├── Fortran/               # serial reference
│   ├── Fortran-OMP/
│   ├── Fortran-ACC/           # handwritten OpenACC baseline
│   ├── main.f90               # timing driver
│   └── test_main.f90           # deterministic correctness driver
├── generated/<case>/          # committed compiler output: CPP, CPP-OMP, CUDA
├── common/                    # shared build flags and C++/CUDA support header
├── harness/                   # generation, builds, checking, timing and plotting
├── tools/                     # Loki/PSyclone recipes and dependency pins
├── tests/                     # harness and recipe regression tests
├── docs/                      # comparison method and interpretation
├── results/                   # baseline.csv, comparison JSON and figures/
└── bin/                       # ignored build output
```

Tool modules contain transformation logic and necessary parser compatibility
handling. Argument parsing, input validation, provenance, drivers and measurements
belong to the shared harness. Adding a tool does not require another benchmark
runner or build tree. Comparison runs write generated sources and reports to
their chosen output directory.

## Build and run

```bash
make -C benchmarks CASE=CDU VARIANT=Fortran
./benchmarks/bin/CDU_Fortran_NX64_NY64_NZ64_NITER100_NWARMUP5/benchmark

make -C benchmarks CASE=CDU VARIANT=Fortran-ACC
make -C benchmarks CASE=CDU VARIANT=CUDA NX=256 NY=256 NZ=256 CUDA_ARCH=sm_86
make -C benchmarks CASE=CDU VARIANT=CUDA-pinned CUDA_ARCH=sm_86
```

Variants are `Fortran`, `Fortran-OMP`, `Fortran-ACC`, `CPP`, `CPP-OMP`, `CUDA`
and `CUDA-pinned`. Defaults are `CASE=CDV`, `VARIANT=Fortran`, a `64x64x64`
grid, `NITER=100` and `NWARMUP=5`. Pinned builds reuse `generated/<case>/CUDA/`.
Override compilers and flags with `FC`, `CXX`, `FC_ACC`, `CUDA_HOME`, `NVCC`
and the variables in [`common/defaults.mk`](common/defaults.mk).
`make -C benchmarks clean` removes the selected build; `clean-all` removes the
entire `bin/` tree.

CPU builds need GNU Make and compatible Fortran/C++ compilers. OpenACC uses
NVIDIA HPC SDK `nvfortran`; CUDA uses `nvcc` and a compatible GPU/host compiler.
Install the repository's Python requirements for source generation and plotting.
The committed generated sources can be built without running the Python compiler.

## Generate, check and measure

```bash
# Regenerate local compiler outputs (optionally select cases: CDU CDV).
python -m benchmarks.harness.generate

# Compare deterministic 16x16x16 output with the serial Fortran reference.
python -m benchmarks.harness.check       # all cases
python -m benchmarks.harness.check CDU   # one case

# Sweep configured cases/variants/grids; CSV on stdout, progress on stderr.
python -m benchmarks.harness.run > benchmarks/results/latest.csv

# Plot the archived baseline, or pass --results benchmarks/results/latest.csv.
python -m benchmarks.harness.plot
# Output: benchmarks/results/figures/grid_<NX>x<NY>x<NZ>_niter<N>.{png,pdf}
```

Configure the sweep near the top of [`harness/run.py`](harness/run.py): cases,
variants, grids, iterations, warmups and rounds. Correctness uses absolute
tolerance `1e-10`; check the runner's output for unavailable build/runtime
variants. Correctness builds append `_test_main` to the build directory name so
they cannot replace timing executables. Figures report seconds per Gcell·iteration.

For Loki/PSyclone, install the separate environments described in the
[comparison guide](docs/tool-comparison.md), then use the shared commands:

```bash
/tmp/loki-env/bin/python -m benchmarks.harness.transform --tool loki \
  --input benchmarks/cases/CDU/Fortran/cdu.f90 --case CDU \
  --target openacc --output /tmp/cdu-loki.f90

python -m benchmarks.harness.compare --output /tmp/tools-cpu \
  --loki-python /tmp/loki-env/bin/python \
  --psyclone-python /tmp/psyclone-env/bin/python

python -m pytest benchmarks/tests
```

The comparison guide documents CPU/GPU modes, fusion, caller data residency,
isolated recipe tests and the [archived measurements](results/tool-comparison-2026-10-02.json).

Local comparisons build both the compiler's optimized default and an
`--opt-level 0` baseline. Checked fusion produces one kernel per stencil instead
of four. `--resident` also checks and times owned compiler sessions on both CPU
and CUDA: the timed batch includes creation, repeated runs, final output retrieval
and destruction. CUDA validation checks allocation, transfer and kernel counts
with `FORT_RUNTIME_TRACE=1`; tracing is disabled during timing.

```bash
# Local-only CPU comparison, including persistent sessions.
python -m benchmarks.harness.compare --output /tmp/local-cpu \
  --tools --gpu off --resident

# Include CUDA resident batches where the GPU and OpenACC baseline are available.
python -m benchmarks.harness.compare --output /tmp/local-gpu \
  --tools --gpu run --resident
```

Archived reports describe the compiler revision recorded in each report. Rerun
the comparison to measure these optimizations; correctness tests impose no
performance thresholds.

The [2026-10-02 compiler capability comparison](results/compiler-capabilities-2026-10-02.json)
records 60 passing CPU correctness checks and 30 timing configurations, including
resident batches. For the `128x128x128` grid, mean ordinary-call times were:

| Case | Original Fortran | C++ level 0 | C++ level 1 |
| --- | ---: | ---: | ---: |
| CDU | 10.982 ms | 14.754 ms | 9.897 ms |
| CDV | 11.585 ms | 14.969 ms | 9.903 ms |
| CDW | 11.585 ms | 17.235 ms | 10.312 ms |

These are single-thread measurements on this host, with 3 warmups, 20 timed
calls per round and 3 rounds. The report includes individual samples, variability,
checksums, compiler source hashes and toolchain versions. CPU resident batches
include owned-buffer copies and were slower than ordinary CPU calls. CUDA builds
passed, but the installed driver could not execute them, so this report contains
no CUDA performance measurements.

The [full CPU/GPU rerun](docs/tool-comparison.md#rerun-with-local-fusion-and-sessions-2026-10-02)
now records **288 passing correctness checks** and **144 timing configurations**
with both libraries, local optimization levels 0/1, and resident sessions.
At `256x128x64`, optimized local C++ took **10.11–10.30 ms** per call; local CUDA
took **12.82–14.55 ms** for ordinary calls and **1.31–1.39 ms** with resident
batches. Each batch has 20 calls and includes setup, final retrieval and teardown.
[Raw results](results/tool-comparison-rerun-20261002T155622Z/report.json) and
[CSV samples](results/tool-comparison-rerun-20261002T155622Z/timings.csv) retain all measurements;
the earlier capability report above remains unchanged.

## Runtime scope and hybrid comparison

The opt-in strategy comparison uses generic pointwise, opposite boundary face,
immutable-halo stencil, and compute-heavy inputs. Decisions belong to the compiler;
the harness calls its public CLI and records its JSON response.

```bash
python -m compiler.offload.calibrate --threads 4 --precision 64 \
  --output benchmarks/results/gpu-strategies-20261006/calibration.json
python -m benchmarks.harness.strategies \
  --calibration-profile benchmarks/results/gpu-strategies-20261006/calibration.json \
  --output benchmarks/results/gpu-strategies-20261006/generic
```

Modes are `native`, `current` (whole-array CUDA), `sections`, `auto` (runtime
region selection), `chunked` (asynchronous GPU-only), and `hybrid`. All use the
same host-thread budget. The reusable calibration must match the precision and
thread budget; benchmark calls never tune compiler decisions.

Defaults are one warmup and three measured process repetitions, with seven for
comparisons near a promotion threshold. `--grids`, `--cases`, and `--modes` limit
the sweep. `--compiler-root` selects an independent compiler checkout. The
harness compares every source and destination element, including halos, after
two calls with changed input, and saves separate unprofiled timing and Nsight
activity evidence. Phase timers that synchronize the device are disabled.

Reports retain complete process wall time as well as per-call time, transferred
bytes, launches, decisions, and measured CPU/GPU/copy overlap. Packing and
decision range durations come from separate trace executions. An automatic
all-native result is valid; the requested `hybrid` mode alone does not prove that
CPU and GPU computed concurrently. Promotion requires the documented application
regression and end-to-end improvement gates, including separate ELMM results.
