# GPU node experiments

After a fresh clone, initialize the pipeline and solver submodules before
building an image or submitting jobs. Run source preparation on a worker with
network access:

```bash
git submodule update --init -- elmm-pipeline poisson-solver
git submodule status
```

For an existing checkout, inspect its submodule state first; initialization
should not be used to replace local work. An explicit `PIPELINE_ROOT` must
point to a complete pipeline checkout.

## Resident comparison

`./run_on_node.sh` and `containers/entrypoint.sh resident` run the `resident`
comparison. It runs the current pipeline and compiler, builds the local CUDA
Poisson solver, and compares fresh ELMM executables:

- `ELMM-native` uses native Fortran/OpenMP and CPU PoisFFT/FFTW.
- `ELMM` uses the generated resident GPU path and the local CUDA Poisson solver.

Both are built from the same pinned upstream ELMM and PoisFFT sources, unchanged.
The generated build records source snapshots, build commands, binaries and the
solver source receipt. The comparison records the configuration, toolchain,
GPU, runtime settings and complete-process timing samples. Work and output
directories must be new for every run and must be retained together.

### Prepare dependencies

By default, the job creates fresh, pinned ELMM, PoisFFT and VkFFT checkouts in
its work directory on the worker; this requires network access. The solver is
the separate `poisson-solver` submodule in this repository, and its Git history
must be available for the pipeline's ancestry check. No source checkout is
modified. To reuse explicitly prepared dependencies instead, create independent,
clean checkouts on a worker with network access:

```bash
USER=$(id -un)
export DEPS_ROOT=/work/$USER/elmm-deps
mkdir -p "$DEPS_ROOT"
git clone https://bitbucket.org/LadaF/elmm.git "$DEPS_ROOT/elmm"
git -C "$DEPS_ROOT/elmm" checkout --detach ddb426fca5b020216083f02a218292fea70c1df2
git clone https://github.com/LadaF/PoisFFT.git "$DEPS_ROOT/PoisFFT"
git -C "$DEPS_ROOT/PoisFFT" checkout --detach 256a2338254c08f611881b2df5e72bdfa73bc257
git clone https://github.com/DTolm/VkFFT.git "$DEPS_ROOT/VkFFT"
git -C "$DEPS_ROOT/VkFFT" checkout --detach 066a17c17068c0f11c9298d848c2976c71fad1c1
```

The runner checks the pinned application sources out into its own work
directory. It rejects unclean or mismatched application/VkFFT checkouts; local
solver edits are snapshotted and hashed. The selected compiler and pipeline are
the current workspace sources. Keep the same revisions and source receipts to
repeat a particular comparison.

### Run natively

Linux requires a visible NVIDIA GPU and compatible driver/toolkit, GNU Fortran
15, a CUDA-compatible C++ compiler, FFTW (single, double and OpenMP), BLAS and
LAPACK. The worker needs Python 3.12+ with `venv`/`pip`; missing Python, CMake 4+
or Ninja packages are installed into a private work-directory environment. This
also requires network access. Select another compatible toolchain with
`--fortran`, `--cuda-host`, and `--nvcc`.

```bash
export SOLVER_ROOT="$PWD/poisson-solver"
export PYTHON=/path/to/venv/bin/python
OUTPUT_DIR="$PWD/results/resident-smoke" WORK_DIR="$PWD/results/resident-smoke/work" \
    ./containers/entrypoint.sh resident --steps 2 --repeats 1 --warmups 0
```

Remove `--steps` to retain the selected configuration's natural end. Defaults
are four application threads, four build jobs, one excluded warmup and seven
measured repetitions. `--steps` caps all diagnostics and timings; it is a smoke
or explicitly limited workload, not the natural-duration run.

Each invocation runs one case:

| Case | Configuration | Advection order | Configured end time |
|---|---|---:|---:|
| `channel-simple` (default) | Reviewed `containers/cases/channel-simple` input | 2 | 0.5 |
| `channel` | Pinned ELMM `examples/channel` input | 4 | 50 |

`--config DIR` selects a custom configuration directory instead of the case's
input. `--grid`, `--domain-size`, `--end-time`, `--cfl`, `--advection` and
`--re` override single settings of the selected configuration.
`--threads` sets the thread count both runs start with and `--jobs` the build's.
ELMM's initial state depends on the count, so the two runs must share it.
`--gpu-threads` (default 1) is the GPU run's team while its fields are on the
device, where its host code runs serially between kernels and a larger team
only adds the cost of waking it. The GPU run may hold what the device has free
when it starts, less a reserve for the driver; `--gpu-budget` (bytes, or a count
with `K`, `M` or `G`) states a limit instead, for a device shared with other work.
`--repeats`, `--warmups`, `--timeout`, `--build-timeout`, `--gpu` (visible CUDA
ordinal or UUID), `--fortran`, `--nvcc` and `--cuda-host` are also available;
run `entrypoint.sh resident --help` or `run_on_node.sh --help` for all options.
The unmodified natural `channel` uses the approved 1% validation rule described
below; `channel-simple`, capped runs and custom or overridden configurations
remain pointwise.

### Run with Apptainer on Slurm

Build the image on an allocated worker. It includes the compiler and pipeline,
but the actual job binds the current workspace, the dependency checkouts and
the independent solver checkout read-only. Fakeroot builds require the site
administrator's subuid/subgid setup; alternatively use an existing image.

```bash
./containers/build-apptainer.sh --fakeroot -o /work/$USER/fortran-resident.sif

./run_on_node.sh hopper01 \
    --runtime apptainer --sif /work/$USER/fortran-resident.sif \
    --time 02:00:00 --output-dir /work/$USER/resident-smoke \
    --work-dir /scratch/tmp/$USER-resident-smoke \
    --steps 2 --repeats 1 --warmups 0
```

Use `--account ACCOUNT` and `--partition PARTITION` where required; verify
current nodes, accounts and partitions with `sinfo`/`smypartition`. For a
Chimera GPU node such as `mff-a2-19`, specify `--partition gpu-ffa`. Use
`--dry-run` to inspect the generated job script without submitting it.
`--runtime native` uses the same experiment on an allocated worker with the
native dependencies installed. On Chimera, `$TMPDIR` is RAM charged to the
job's memory limit; `/scratch/tmp` is appropriate for larger builds.
Use `--deps "$DEPS_ROOT"` to reuse the explicit checkouts above and `--solver`
to select a different local solver checkout.

The report is `OUTPUT_DIR/resident/report.json`; its paths into the work tree
refer to the container's mount points when run inside Apptainer. Absolute and
relative complete-wall timing plots are written to `OUTPUT_DIR/resident/figures`
in PNG, SVG and PDF formats. Three diagnostic processes compare a native
reference, native repeat and resident run. A traced diagnostic verifies GPU
launches for SGS and convection; it does not independently establish that every
pressure solve used the CUDA provider. Production timing runs are untraced,
alternate native/GPU order and are validated against the native reference.
Initially empty, owner-only diagnostic and timing kernel caches are separate;
an excluded warmup populates the timing cache. With zero warmups the first
timing includes cache misses and compilation.

Both executables also report how their wall time divides into initialization,
the time-stepping loop and finalization (which includes writing the results
and, for the GPU run, bringing the fields back). When those lines are present
the summary of each mode carries `phases` with samples and medians, the report
adds `loop_speedup` next to the complete-process `speedup`, and the bars of the
timing plots are stacked by phase; what the phases do not cover of a process,
such as start-up and kernel cache lookups, shows as process overhead.

`--nsys` adds one more run of the GPU executable under Nsight Systems after
the timing runs, and `--nsys-steps N` caps that run's time steps. The profile
(`OUTPUT_DIR/resident/nsys/resident.nsys-rep`) comes with text and CSV
summaries of kernels, CUDA API calls and memory transfers. The kernels run as
nodes of CUDA graphs, so the profile traces graph nodes; a profile taken with
the default graph tracing lists almost no kernels and makes the GPU look idle.
`cudaMemcpyAsync` time in the API summary is mostly the host waiting for the
kernels queued before a small download, not transfer time. A profile that
fails is recorded in the report and does not fail the comparison.

For the unmodified natural `channel`, every run must reach the time limit at
the native reference's step count (2,458 steps in the 4-thread reference setup;
adaptive CFL timestepping and thread-dependent seeding can yield a different
step count on other thread counts or platforms) and retain all 20 fields and
114 outputs. The approved rule requires
at most 1% relative L2 error independently per field and output component;
zero-reference components use maximum absolute error `1e-6`. Bounds, layouts,
coordinates, labels and output inventories must match exactly. Time columns
retain the original `atol=1e-6`, `rtol=1e-5` pointwise checks. The report also
retains the original pointwise field and final/averaged VTK results, even when
those stricter checks fail. Short, `channel-simple` and custom runs retain the
original pointwise checks on full fields, bounds and final/available averaged
VTK output. Natural `channel` defaults to a two-hour Slurm allocation and a
one-hour per-process timeout; both can be overridden.

Field comparison failures retain the diagnostic stage and full per-field
statistics in the report. To inspect saved dumps, including from earlier runs,
without rebuilding or using a GPU, using the original pointwise rule:

```sh
python3 containers/resident_comparison.py --inspect-diagnostics OUTPUT_DIR/resident/diagnostics
```

Use the experiment's Python environment. For an Apptainer run, the same check
can be run with the existing image and output directory:

```sh
OUTPUT_DIR=/path/to/run WORKSPACE_ROOT="$PWD" GPU_DEVICES=none \
    ./containers/run-apptainer.sh resident --inspect-diagnostics /output/resident/diagnostics
```

These are reproducible comparison measurements, not release qualification:
they do not check external machine load or CPU FFTW plan counts. The 1% rule is
limited to natural `channel` and does not change primary/operator checks.
A capped smoke run is not evidence of full-workload performance.

### Source basis

Both executables are built from upstream ELMM
`ddb426fca5b020216083f02a218292fea70c1df2` and the pinned PoisFFT, unchanged.
ELMM sums the pressure reference of a boundary plane with an OpenMP reduction,
so its value depends in rounding on the number of threads; the resident
executable computes the same sum on the device in a fixed order. The fields of
the two executables are compared with the tolerances described above, not bit for bit.
`containers/source_basis.json` pins the repository revisions and the hashes of
the reviewed sources, includes and build files; drift in the sources or in that
file fails before the build. The `channel-simple` input records its cloud
repository revision and parser conversions in `cases/channel-simple/origin.json`.

The comparison checks the sources out under `WORK_DIR/resident/application-basis`.
The same preparation can also run on its own, using upstream checkouts as
read-only inputs:

```bash
.venv/bin/python containers/application_basis.py \
    --upstream-root "$DEPS_ROOT" --output out/applications
```

It creates clean ELMM/PoisFFT Git siblings at the pinned revisions and records
every command in `OUTPUT-preparation.json` beside them. Reusing a destination
validates its pinned revisions and all tracked inputs; dirty or unexpected
sources are rejected, never reset.


## On the MFF clusters

The [cluster guide](https://gitlab.mff.cuni.cz/mff/hpc/clusters) supplied with this
work lists these presets: `hopper01` uses `gpu-ffa-gpulab`; `bw01` uses
`ffa-gpulab`. Both have two GPUs. The guide's Blackwell GRES label conflicts with
its hardware description, so the runner requests one GPU without a model name
and queries the allocated CUDA device for its actual architecture. Presets only
select a partition; `--partition` overrides them. Check live availability with
`sinfo` and choose an account with `--account` when needed.

```bash
# Resident comparison on bw01 with an explicit partition and account.
./run_on_node.sh -p gpu-short -A kdss bw01

# Natural channel comparison with the approved 1% rule and timing plots.
./run_on_node.sh -p gpu-short -A kdss --case channel bw01

# Inspect the job script without submitting or executing numerical work.
./run_on_node.sh hopper01 --dry-run

# An explicitly capped smoke test; not evidence of full-workload performance.
./run_on_node.sh hopper01 --steps 3 --repeats 1 --warmups 0 \
    --output-dir /work/USER/resident-smoke-01

# More build resources, while retaining the four-thread timing budget.
./run_on_node.sh hopper01 --jobs 16 --cpus 16 --mem 64G \
    --output-dir /work/USER/resident-build16-01
```

Run builds and experiments on allocated worker nodes. The submission wrapper
performs no headnode package installation and never falls back to a local run
when Slurm is missing. `--no-wait` submits without waiting; otherwise scheduler
failure propagates to the caller. Extra `resident` options follow `--`; checked
path/resource options must precede it. Use `--config DIR` for a custom config.
`run_on_node.sh` at the repository root forwards to `containers/run_node.sh`.

The launcher prints the output/work directories, report, figure and build log
locations, job script and Slurm log paths before submission, then the job ID
and monitoring commands. Resource defaults are 8 CPUs, 32G, one GPU and
01:30:00 (02:00:00 for the natural `channel`), written as `#SBATCH` lines so
site/user `SBATCH_*` settings retain their precedence. `--jobs` defaults to the
CPU request and is capped to a smaller actual allocation; an explicit `--jobs`
is retained. `--interactive` uses `srun` with the same request.

On the worker, the script initializes environment modules when necessary and
loads `apptainer`, `cuda`, `gcc` and the Python module alternatives before
choosing the container runtime. Module attempts and the loaded module list are
saved to `modules.log`, whose path is printed on submission and worker exit.
Missing optional modules remain nonfatal when inherited tools are available.
Native execution selects an available Python 3.12 or newer unless `PYTHON` is
supplied explicitly, and prefers `gfortran-15` when available unless `FC` is
supplied. Plotting may use a different interpreter with NumPy and Matplotlib;
for native runs, set `PLOT_PYTHON` to select it. The plotter accepts Python
3.9+ and reads the JSON report without importing compiler or pipeline source.
The selected interpreter is printed. An explicit `PLOT_PYTHON` is checked,
never silently replaced. Container runs use their packaged plotting
interpreter; host interpreter paths are not forwarded into the container. The
launcher does not install packages into a system Python on the headnode.

`--runtime auto` uses Apptainer/Singularity when its image is available, otherwise
the installed native toolchain on the **allocated worker**. Select
`--runtime apptainer` to require the image. The command never builds an image
implicitly.

Results and generated work default to persistent output storage. On gpulab,
`$TMPDIR` is local SSD; on Chimera it is RAM and counts against job memory. Use
explicit `--work-dir` when moving large build artifacts to scratch. Keep the
work directory while its report is needed: the report refers to generated
artifacts and executable hashes there. Scratch is caller-owned and is never
deleted by the experiment. Choose a new output/work directory for each run.

## Build and inspect an image

```bash
./containers/build-docker.sh
./containers/build-apptainer.sh --fakeroot -o /work/USER/fortran.sif

OUTPUT_DIR="$PWD/out-doctor" ./containers/run-docker.sh doctor
SIF_FILE=/work/USER/fortran.sif ./containers/run-apptainer.sh doctor
```

The base is `nvidia/cuda:12.8.1-devel-ubuntu24.04`, with GNU Fortran 15,
GCC/G++ 14, FFTW/OpenBLAS/LAPACK and pinned compiler/parser/build dependencies.
CUDA 12.8 supports Hopper and both Blackwell target families; GPU naming alone
does not determine the compute capability. See NVIDIA's
[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html)
and [GPU list](https://developer.nvidia.com/cuda/gpus).
`--arch auto` detects the selected device; an explicit mismatching or unsupported
target fails before building. An actual CUDA launch, not `nvidia-smi` alone,
establishes that the generated variant executed.

Image builds exclude result archives, host environments and caches. The
Apptainer helper stages a small source-only context before `%files` runs;
`CONTAINER_BUILD_TMPDIR` controls that staging location. Build images on a worker
node. `--fakeroot` requires the cluster administrator's subuid/subgid setup.
Images are prepared from current source files, including the independent
`elmm-pipeline` checkout; keep compatible compiler and pipeline revisions.

Docker commands that are Podman compatibility launchers are detected and use
Podman CDI GPU flags. Podman GPU runs require a configured NVIDIA CDI runtime.
For Docker or Podman inside Slurm, use `GPU_DEVICES=GPU-UUID`; scheduler-local
numeric ordinals are not safe physical Docker/CDI device selectors. Apptainer/native runs
preserve the scheduler's CUDA visibility and resolve its CUDA ordinals to UUIDs.
CPU-only `doctor`, plotting and tests do not require a GPU.

Source overrides are explicit: `WORKSPACE_ROOT`, `PIPELINE_ROOT` and
`COMPILER_ROOT` mount read-only source directories in the container. The
Apptainer launcher also binds `SOLVER_ROOT` and `DEPS_ROOT` read-only. The
compiler and pipeline remain independent checkouts.

## Plot a saved report

Images contain no result archive. Mount the saved report and choose it
explicitly; plotting never substitutes old results for a fresh run.

```bash
# Native: redraw the figures of one saved comparison report.
PYTHON=.venv/bin/python ./containers/reproduce_plots.sh \
    --experiment results/RUN/resident/report.json --output-dir results/RUN/replot

# Container: mount the run under /output.
OUTPUT_DIR=/work/USER/RUN ./containers/run-docker.sh reproduce-plots \
    --experiment /output/resident/report.json --output-dir /output/replot
```

Figures are written to a new `figures` directory inside the output directory;
an existing one is never replaced. `--policy-report FILE` instead redraws the
older six-policy microbenchmark report into `ordinary-figures`, labeled as
historical measurements.

## Other commands

`doctor` reports the tools, selected GPU, CPU affinity and effective memory
headroom; it needs no GPU. `run-benchmarks` is a separate, bounded generic
microbenchmark check and timing run; it does not measure the ELMM application.
It runs under the pipeline's memory guard, which keeps a 3 GiB reserve using
the tighter of host and applicable cgroup budgets and admits one guarded job at
a time. If the guarded command fails, the launcher prints the end of `run.log`
and the paths of the full log and `guard.json`; the failure status is returned.
`entrypoint.sh test` runs the tests in `containers/tests`. Arbitrary commands
such as `bash` remain available through either container launcher.
