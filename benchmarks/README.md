# Benchmark suite

Three momentum-advection kernels (CDU, CDV and CDW) in seven implementation
variants. Run all commands below from the repository root.

The `CPP`, `CPP-OMP` and `CUDA` sources under `generated/` are archived output of
the compiler's retired ordinary-call backend. The current compiler emits shared
graph entries instead, so these files are maintained as fixed benchmark inputs
and are not regenerated.

## Scope preflight maps

The compiler's `--scope-preflight` report separates source support, numerical
preparation, automatic cost estimates and unproved ownership. Render the
`preflight` object from its JSON response into a fresh directory:

```bash
python -m benchmarks.harness.plot_preflight preflight.json --output new-scope-figures
```

The HTML report retains guards, resources and rejection reasons; paginated
PNG/SVG maps preserve source order within each procedure. Gray means unknown,
not requested or not applicable, and does not establish GPU eligibility.
The receipt binds the input and rendered files by hash. These diagnostics
contain no measured transfers, execution proof or application speedup.

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
├── generated/<case>/          # archived compiler output: CPP, CPP-OMP, CUDA
├── common/                    # shared build flags and C++/CUDA support header
├── harness/                   # builds, checking, timing and plotting
├── tests/                     # harness regression tests
├── results/                   # ignored: baseline.csv, reports and figures/
└── bin/                       # ignored build output
```

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
Install the repository's Python requirements for plotting. The committed
generated sources are built without running the Python compiler.

## Check and measure

```bash
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

Run the harness regression tests with `python -m pytest benchmarks/tests`.

`python -m benchmarks.harness.plot_strategies REPORT.json --output NEW_DIR`
redraws the charts of a saved policy-comparison report. The runner that produced
those reports was removed with the compiler policies it compared.
