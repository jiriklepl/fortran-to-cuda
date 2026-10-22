# GPU node experiments

`./run_on_node.sh` remains the top-level Slurm command. It forwards to
`containers/run_node.sh`; local, Docker/Podman and Apptainer entrypoints run the
same experiment. The experiment builds **native ELMM on the reviewed common source basis and a
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

# Explicitly reproduce the earlier 50-step advection-order comparisons.
./run_on_node.sh -p gpu-short -A kdss bw01 --case second-order --steps 50
./run_on_node.sh -p gpu-short -A kdss bw01 --case fourth-order --steps 50

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
After upstream setup or cache reuse, the workflow reconstructs clean comparison
checkouts under `WORK_DIR/application-basis`. Upstream caches are never patched.

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

Each invocation runs one selected case. The presets are:

| Case | Configuration | Advection order | Configured end time |
|---|---|---:|---:|
| `channel-simple` | Reviewed `containers/cases/channel-simple` input | 2 | 0.5 |
| `channel` | Pinned ELMM `examples/channel` input | 4 | 50 |
| `second-order` | Same upstream channel input, explicit advection override | 2 | 50 |
| `fourth-order` | Same upstream channel input, explicit advection override | 4 | 50 |

No preset adds a timestep cap. `--steps` is explicit smoke-test control. Earlier
second-/fourth-order campaign cases used a 50-step cap; reproduce that duration
with `--steps 50`, rather than treating those historical timings as full-duration
preset measurements. Allow enough Slurm wall time and per-run `--timeout` for
the channel-derived presets.

`--config DIR` overrides the preset configuration and its advection selection.
Reports and plots use the custom directory's name, with the requested preset
recorded separately. The original directory name survives container binding.
Only an explicit `--steps` changes a custom configuration's timestep limit.
`--threads` changes the actual application budget; only the default four-thread
series is eligible for this comparison target. `--jobs` controls builds only.
`--repeats`, `--warmups`, `--timeout`, `--gpu` (visible CUDA ordinal or UUID),
`--nvcc` and `--cuda-host` are forwarded to the same workflow. Run
`entrypoint.sh run-elmm --help` or `run_on_node.sh --help` for options.

`--scope-transfers {direct,pinned,pipelined,auto,coalesced}` transports the compiler's
existing transfer selection; `direct` remains the default. The compiler owns
eligibility and fallback. Reports and plots identify the selected control.
For node runs, supply it after `--`, for example
`./run_on_node.sh -p gpu-short -A kdss bw01 -- --scope-transfers pinned`.
The experimental `coalesced` control combines eligible upload sections into a
dense physical span with at most 12.5% extra bytes, provided that the whole
canonical source resource is defined and host current. Output copies stay exact, and ineligible uploads use
direct copies. Cost-selected plans retain exact direct execution until calibrated.
Reports and plot labels identify this explicit control; it uses no pinned staging.

Source overrides are explicit: `WORKSPACE_ROOT`, `PIPELINE_ROOT` and
`COMPILER_ROOT` mount read-only source directories in the container. Prepared
pipeline tools and application copies live in writable work storage. Advanced
native callers can supply `--app-root` (clean pinned ELMM/PoisFFT Git siblings),
`--compiler-python`, `--scons-python` and `--native-library-dir` to reuse existing
provisioning. The compiler and pipeline remain independent checkouts.

The SGS preset is an explicit application experiment, not a generic compiler
heuristic. Its portable contract pins reviewed caller capture facts and source
hashes; source/contract drift fails before compilation. The channel-simple
fixture records its cloud repository revision and parser conversions. All modes
use the same approved ELMM derivative of upstream
`ddb426fca5b020216083f02a218292fea70c1df2`: only the TOP/BOTTOM plane-average
reduction uses fixed-order `MASTER` accumulation followed by its required barrier.
These configurations retain TOP referencing.
The user-approved [patch](patches/elmm-deterministic-top.patch) produces pressure
SHA256 `65d6ee76dc4b63b4545a1bfd481dbf875cb168f55a484bac7d0a39b78f21f7d0`
and deterministic local commit `be38bbbec88756ef0f32d67b44dc412a71e0e6e8`.
PoisFFT, capture facts, includes, build options and numerical tolerances remain
unchanged. This derivative is reconstructed locally; it is not advertised as a
revision available from the upstream remote. Historical pristine-upstream
measurements and private pressure-only links retain their original provenance.

The shared preparation utility can also run explicitly, using already prepared
upstream repositories as read-only inputs:

```bash
.venv/bin/python containers/application_basis.py \
    --upstream-root elmm-pipeline/.cache/repos --output out/approved-applications
```

It creates clean ELMM/PoisFFT Git siblings and records the patch, fixed commit
metadata and every command beside them. Reusing a destination validates its
pinned revisions and all tracked inputs; dirty or unexpected sources are
rejected, never reset. An explicit `--app-root` bypasses preparation and still
must pass the same strict source-contract and clean-checkout checks.

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

The retained SGS comparison explicitly selects GPU execution with section
transfers; it does not use automatic CPU/GPU placement. Compiler placement
`auto` uses offline calibration and current dimensions to compare complete CPU
and GPU alternatives, requiring a modeled 20% advantage. Missing or incompatible
estimates select native execution. Automatic SGS cost applicability and complete
application performance still need validation, so the measured forced-SGS gain
does not promote automatic placement. `--scope-transfers auto` independently
selects a transfer mechanism; it does not enable CPU/GPU placement selection.

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

## Fresh three-way compiler comparison (`e2e`)

`reproduce_e2e.py` builds native ELMM on that common approved basis, the committed compiler's
SGS/direct control, and a selected reworked compiler graph. It never executes
historical binaries or reuses historical timings. The committed compiler comes
from `git archive` of the selected revision, irrespective of dirty/staged files;
the candidate compiler is frozen in a separate snapshot before compilation.
The default committed revision is `be58ab809296b97ea39d701828a2621adfc6b3dc`, the historical
SGS/direct control. `--committed-ref` can explicitly select another compatible
historical compiler revision.
When `--app-root` is omitted, the same shared utility reconstructs the basis from
the pipeline's upstream cache into a sibling of the new output directory named
`OUTPUT-application-basis`. Existing explicit prepared roots remain supported.

```bash
# Fresh full-duration comparison, one warmup and seven rotated samples per mode.
# Use the prepared parser environment; --python can select it independently.
.venv/bin/python containers/reproduce_e2e.py \
    --committed-ref be58ab809296b97ea39d701828a2621adfc6b3dc \
    --candidate-request path/to/graph-request.json \
    --output-dir out/comparison-01

# Bind the checkout (including Git history and the prepared pipeline cache)
# read-only to let the container archive exact committed compiler bytes.
WORKSPACE_ROOT="$PWD" OUTPUT_DIR="$PWD/out" ./containers/run-docker.sh e2e \
    --candidate-request /workspace/containers/requests/elmm-sgs-convection-graph.json \
    --committed-root /workspace --committed-ref be58ab809296b97ea39d701828a2621adfc6b3dc --output-dir /output/comparison-01

# Primary fourth-order channel: preserve natural-duration correctness, while
# profiling and timing use the same explicit 100-step duration in all modes.
.venv/bin/python containers/reproduce_e2e.py --case channel \
    --timing-steps 100 --timeout 3600 --require-cuda-profile \
    --candidate-request path/to/graph-request.json --output-dir out/channel-01
```

The candidate request uses application-relative paths and the compiler's capture
facts; the runner transports those facts without inferring source semantics:

```json
{
  "schema_version": 1,
  "source": "elmm/src/dynamics.f90",
  "entry": "dynamics::selected_procedure",
  "source_files": ["elmm/src/subgrid.f90"],
  "capture_facts": {},
  "options": {"gpu_policy": "sections", "scope_execution": "graph", "scope_transfers": "direct"}
}
```

Supply the actual checked entry and capture facts for the workload. Omitting
`--candidate-request` selects the same SGS numerical coverage as the committed
control through the candidate's graph execution; it does not claim broad graph
coverage. The committed archive retains its original bounded/direct strategy.
The primary workloads are `channel-simple` and the
original fourth-order `channel`, including its filter and wall model. Legacy
`second-order`/`fourth-order` presets remain available for component diagnostics;
they do not replace either primary workload. The original configured duration is
preserved for correctness unless `--steps` explicitly requests an all-phase smoke
test. `--timing-steps N` caps only timing, warmup and profiling runs, using a
separate native reference at that same duration. It retains natural-duration
float64 correctness and reports capped performance distinctly from full-duration
performance. `--config DIR`
selects custom inputs. Output/work directories must be new.

Every comparison writes `prepared-builds.json` after compiling. A second case
can use `--prepared-builds PATH` to retain those exact executables while
collecting every correctness, profile and timing run afresh. Compiler snapshots,
pristine source identities, selected roots, thread/build options, every executed
build receipt and generated artifact must still match. The native executable
must contain only original compilation units, and the float64 diagnostic must
match the current observer. Changed artifacts are rejected; historical runs and
timing summaries are never imported. Use the frozen candidate compiler path
from the first report with `--candidate-root` when reusing its build set.
Comparison records refer to original proof manifests by sidecar path and SHA-256;
they do not embed another copy of every proof. Reuse authenticates and loads the
unchanged sidecars before checking their generated artifacts.
Nsight profiling explicitly requests CUDA graph node activity, so actual kernel
counts remain available when the application submits CUDA graphs. Graph API
submissions, logical kernels, direct launches and device activity stay separate.

`--control-evidence PATH` additionally accepts an explicit
`fresh-session-natural-controls` manifest for separately executed native and
committed natural controls from the same session. It requires prepared builds
and revalidates complete float64 fields, original shapes/bounds, VTK bytes,
logs, numerical input files, executable/build identities and the recorded
thread/environment contract. Original strict comparison results and native
repeatability failures remain unchanged and visible in the report. The final
candidate's natural correctness runs, runtime traces, profiles and every rotated
production timing sample remain fresh. No historical timing result is imported.

A candidate request may also be a nonempty JSON array of these objects. Each
root is generated independently against pristine source bytes, with a separate
240-second compiler limit. The compiler's composition command authenticates and
merges original source edits, rejects conflicting replacements, and shares one
runtime build. The runner authenticates all component and merged artifacts
before changing the application copy; original component proof manifests remain
separate for runtime diagnostics. The supplied
`requests/elmm-sgs-convection-graph.json` forces eligible numerical work onto the
GPU while adding the wider Convection graph. This is a coverage experiment,
not a performance recommendation: native neighbors can make those extra
transfers expensive. `requests/elmm-sgs-convection-auto.json` supplies the same
roots and capture facts with generic cost-based placement. It requires an
independently measured applicable calibration profile; missing costs retain
native execution. Selecting these entry roots does not prescribe numerical
leaves, movement, or scheduling. Compare actual transfer counts and complete
wall measurements before claiming an improvement.

Pass the external profile with `--calibration-profile /absolute/path/to/hardware.json`;
it is applied only to candidate roots. Its bytes, GPU UUID, CPU identity, thread
count, float64 contract and scoped runtime identity are recorded and checked.
Conflicting per-root profiles are rejected, and prepared builds reject changed
profile bytes. The committed control retains its original policy. An all-native
AUTO result requires explicit runtime decision or pre-execution decline evidence
and retains every numerical gate. It reports unavailable CUDA activity when none
occurred; it is not an offloading performance achievement.

Each mode gets a fresh production build with executed build-command receipts.
Separate diagnostic mains dump every element of the eleven maintained float64
fields and any allocated optional/averaged fields, including halos, original
bounds and shape. Production outputs and diagnostics must agree with the fresh
native reference, and native repeatability is checked. The established field
tolerances remain `atol=1e-6`, `rtol=1e-5`; matching NaNs/infinities fail.

Timing samples use untraced production executables and complete process wall
time. Traced diagnostics report observed runtime kernel submissions, transfer bytes,
allocations and waits from runtime events, without inferring counts from call
names or source structure. Additional Nsight Systems runs collect actual CUPTI
kernel/copy counts, bytes, device activity durations, and separate CUDA API
counts and CPU durations. Use `--require-cuda-profile` when those diagnostics
must be available for GPU execution; a historical control that executes an
untransformed native branch retains its zero observed runtime activity and
unavailable device profile. The candidate must demonstrate CUDA execution or
an explicit generic AUTO decision to retain native execution.
`--skip-cuda-profile` explicitly skips profiling. Neither traced
runtime nor Nsight durations enter the timing samples.
Complete parsed runtime evidence is written once per trace to a SHA-256-bound
`gpu-activity.json` sidecar. Reports retain observed counters, source identities,
modeled decision summaries and preflight declines; readers can authenticate and
restore complete records with `load_runtime_activity`. Generic policy labels
are accepted only from literal trace calls in authenticated generated CUDA;
versioned source positions provide attribution.
`report.json` retains every sample, field comparison,
manifest and input hash; `report.md` presents native/committed/reworked ratios.

For a shorter development comparison of already built artifacts, run
`python containers/probe_prepared_e2e.py --prepared-builds PREPARED.json --output NEW_DIR`.
It authenticates the original sources, compiler snapshots, build receipts and
binaries, then collects one warmup and three rotated samples for each mode.
The simple case keeps its natural109 steps; channel uses100 matched steps.
Each execution checks completed steps and VTK output against a fresh original
native warmup. `--validate-only` authenticates and prepares inputs without
executing an application. These probes do not validate full float64 storage or
grant final performance qualification; final acceptance still uses the full
comparison above with seven samples and natural-duration correctness.

New candidate snapshots omit `compiler/_local` reports and cache directories.
A separate `code_equivalence` digest records the included source files.
Historical snapshot hashes and the files they authenticate remain unchanged.
`--allow-validation-failure` explicitly permits collecting unqualified timings
while preserving all mismatches. An all-phase smoke run or failed correctness check cannot
qualify a performance claim. Keep the work directory with the report because
build receipts reference its artifacts.
