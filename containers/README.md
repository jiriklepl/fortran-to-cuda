# GPU node experiments

`./run_on_node.sh` remains the top-level Slurm command. It forwards to
`containers/run_node.sh`; local, Docker/Podman and Apptainer entrypoints run the
same experiment. The experiment builds **pristine upstream native ELMM and a
fresh compiler-generated SGS sections/direct variant**, checks final and available
averaged VTK fields,
verifies actual GPU launches separately, then measures complete process time.
The target is 20% less wall time than original native (`GPU/native <= 0.80`),
not another 20% over the previous SGS result. Reports also give the speedup factor.

## On the MFF clusters

The [cluster guide](https://gitlab.mff.cuni.cz/mff/hpc/clusters) supplied with this
work confirms these presets: `hopper01` uses `gpu-ffa-gpulab`; `bw01` uses
`ffa-gpulab`. Both have two GPUs. The guide's Blackwell GRES label conflicts with
its hardware description, so the runner requests one GPU without a model name
and queries the allocated CUDA device for its actual architecture. Presets only
select a partition; `--partition` overrides them. Check live availability with
`sinfo` and choose an account with `--account` when needed.

```bash
# Existing invocation: options may precede the node, as before.
./run_on_node.sh -p gpu-short -A kdss bw01

# Inspect the job script without submitting or executing numerical work.
./run_on_node.sh hopper01 --dry-run

# Fresh, full-duration channel-simple comparison: four runtime threads,
# eight build jobs, one warmup and seven rotated measured repetitions per mode.
./run_on_node.sh hopper01 --runtime apptainer --sif /work/USER/fortran.sif \
    --output-dir /work/USER/sgs-hopper-01 --work-dir /scratch/tmp/USER-sgs-hopper-01

# The upstream ELMM channel configuration, on the other target.
./run_on_node.sh bw01 --case channel --runtime apptainer \
    --sif /work/USER/fortran.sif --time 08:00:00 \
    --output-dir /work/USER/sgs-blackwell-01 -- --timeout 7200

# An explicitly capped smoke test. Its report cannot pass the performance gate.
./run_on_node.sh hopper01 --steps 3 --repeats 1 --warmups 0 \
    --output-dir /work/USER/sgs-smoke-01

# More build resources, while retaining the four-thread timing budget.
./run_on_node.sh hopper01 --jobs 16 --cpus 16 --mem 64G \
    --output-dir /work/USER/sgs-build16-01
```

Run builds and experiments on allocated worker nodes. The submission wrapper
performs no headnode package installation and never falls back to a local run
when Slurm is missing. `--no-wait` submits without waiting; otherwise scheduler
failure propagates to the caller. Extra experiment options follow `--`; checked
path/resource options must precede it. Use `--config DIR` for a custom config.

The launcher prints the output/work directories, report and figure locations,
job script and Slurm log paths before submission, then the job ID and monitoring
commands. Resource defaults remain 8 CPUs, 32G, one GPU and 01:30:00, in the
original `#SBATCH` form so site/user `SBATCH_*` settings retain their precedence.
`--jobs` defaults to the CPU request and is capped to a smaller actual allocation;
an explicit `--jobs` is retained. `--interactive` uses `srun` with the same request.

On the worker, the script initializes environment modules when necessary and
loads `apptainer`, `cuda`, `gcc` and the original Python module alternatives
before choosing the container runtime. Module attempts and the loaded module
list are saved to `modules.log`, whose path is printed on submission and worker
exit. Missing optional modules remain
nonfatal when inherited tools are available.
Native execution selects an available Python 3.12 or newer unless `PYTHON` is
supplied explicitly. Plotting may use a
different interpreter with NumPy and Matplotlib; for native runs, set
`PLOT_PYTHON` to select it. Container runs use their packaged plotting interpreter;
host interpreter paths are not forwarded into the container.
Source extraction runs inside the prepared compiler/parser environment, so the
newer system Python does not need fparser, NumPy or Matplotlib installed globally.
The paired-result plotter accepts Python 3.9+ and reads JSON receipts without
importing compiler or pipeline source. Its selected interpreter is printed and
recorded with the figures. An explicit `PLOT_PYTHON` is checked, never silently
replaced. Existing source environments and system plotting packages are reused;
the launcher does not install packages into a system Python on the headnode.
An already prepared `elmm-pipeline/.cache` is reused, as in the original workflow;
missing preparation uses writable work storage. Native execution also retains
the original preference for `gfortran-15` when available, unless `FC` is supplied.

`--runtime auto` uses Apptainer/Singularity when its image is available, otherwise
the installed native toolchain on the **allocated worker**. Select
`--runtime apptainer` to require the image. The command never builds an image
implicitly. The original standalone launchers and top-level wrapper remain
available.

Results and generated work default to persistent output storage. On gpulab,
`$TMPDIR` is local SSD; on Chimera it is RAM and counts against job memory. Use
explicit `--work-dir` when moving large build artifacts to scratch. Keep the
work directory while its report is needed: receipts refer to generated artifacts
and executable hashes there. Scratch is caller-owned and is never deleted by
the experiment. Choose a new output/work experiment directory for each run.

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

## Preparation and fresh runs

```bash
# Provision pinned ELMM/PoisFFT sources and tools into writable persistent work.
# Can be done before timing on a worker with network access.
OUTPUT_DIR="$PWD/out-prepared" ./containers/run-docker.sh prepare

# Uses that private cache; no writes into the baked or read-only source tree.
OUTPUT_DIR="$PWD/out-prepared" ./containers/run-docker.sh run-elmm --case channel-simple

# Native workflow on an already allocated, configured worker.
PYTHON=/path/to/python ./containers/entrypoint.sh run-elmm \
    --case channel --output-dir /work/USER/channel-native-tools-01
```

The default has no timestep cap. `--steps` is explicit smoke-test control.
The upstream channel case retains its supplied end time 50; channel-simple
retains 0.5. Allow enough Slurm wall time and per-run `--timeout` for channel.
`--threads` changes the actual application budget; only the default four-thread
series is eligible for this comparison target. `--jobs` controls builds only.
`--repeats`, `--warmups`, `--timeout`, `--gpu` (visible CUDA ordinal or UUID),
`--nvcc` and `--cuda-host` are forwarded to the same workflow. Run
`entrypoint.sh run-elmm --help` or `run_on_node.sh --help` for options.

Source overrides are explicit: `WORKSPACE_ROOT`, `PIPELINE_ROOT` and
`COMPILER_ROOT` mount read-only source directories in the container. Prepared
pipeline tools and application copies live in writable work storage. Advanced
native callers can supply `--app-root` (clean pinned ELMM/PoisFFT Git siblings),
`--compiler-python`, `--scons-python` and `--native-library-dir` to reuse existing
provisioning. The compiler and pipeline remain independent checkouts.

The SGS preset is an explicit application experiment, not a generic compiler
heuristic. Its portable contract pins reviewed caller capture facts and source
hashes; source/contract drift fails before compilation. The channel-simple
fixture records its cloud repository revision and parser conversions. Upstream
ELMM files are not repaired or manually changed for the native baseline.

Each run records tools, GPU UUID, CPU affinity, effective memory headroom,
source/artifact/configuration hashes, separate correctness/GPU diagnostics,
raw timings, medians and target status. Generated CUDA retains the compiler's
strict separate-arithmetic build contract. Final VTK comparison uses the existing
`atol=1e-6, rtol=1e-5`; this float32 output check does **not** establish float64
full-array or halo agreement. Two native diagnostic runs must agree at those same
tolerances before GPU diagnostics and timing. A nonrepeatable native reference
fails the campaign; it does not weaken the candidate check or shorten the case.
Native/GPU diagnostics precede timing and never
enter timing medians. Smoke runs and other thread budgets cannot claim the gate.
No policy is automatically promoted by this runner.

If guarded execution fails, the launcher prints the end of `run.log` containing
the worker's error and the full log/`guard.json` paths. The guard receipt stays
on disk; it is not dumped into the Slurm console. Early module diagnostics are
in `modules.log`. CUDA discovery accepts the GCC target `x86_64-redhat-linux`
and uses the selected toolkit's normal `targets/x86_64-linux/lib` directory.

For an explicitly requested timing series despite native roundoff variation:

```bash
./run_on_node.sh -p gpu-short -A kdss bw01 --case channel -- --allow-native-variation
```

This continues when native/native comparisons fail, retaining all original
tolerances and comparison failures. Results and plots are labeled unvalidated;
the timing summary is printed, but correctness and promotion gates remain unmet.
GPU execution failures and source-definition errors still fail the run.

Heavy commands are sequential under the maintained pipeline memory guard, which
keeps a 3 GiB reserve using the tighter of host and applicable cgroup budgets.
Failures retain logs and return failure; they are not converted into successful
reports. A new run never deletes an earlier result directory.

## Plot explicit saved measurements

Images contain no historical result archive. Mount the saved evidence and choose
it explicitly; plotting never silently substitutes old results for fresh runs.

```bash
# Native example using the maintained sealed-snapshot reader.
PYTHON=.venv/bin/python ./containers/reproduce_plots.sh \
    --snapshot benchmarks/results/EXPERIMENT/snapshot.json \
    --output-dir benchmarks/results/EXPERIMENT/plots/new-snapshot

# Container: mount the evidence under /output; all snapshot-relative files travel
# with it. Absolute-path campaign receipts require their original artifact paths.
OUTPUT_DIR=/work/USER/saved-evidence ./containers/run-docker.sh reproduce-plots \
    --snapshot /output/snapshot.json --output-dir /output/plots/new-snapshot
```

`--campaign DIR` redraws a maintained campaign report; `--policy-report FILE`
redraws the older six-policy microbenchmark report. Historical inputs are labeled
as such. Output plot directories must be new. Fresh `run-elmm` produces its own
native/SGS figures alongside its report, without importing archived timings.
Use `--experiment FILE` to redraw one of those fresh paired reports.

`run-benchmarks` is a separate, bounded generic microbenchmark check/timing run;
it does not measure the ELMM application or its 20% target. `entrypoint.sh test`
runs the container workflow tests. Arbitrary commands such as `bash` remain
available through either container launcher.
