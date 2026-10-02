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
