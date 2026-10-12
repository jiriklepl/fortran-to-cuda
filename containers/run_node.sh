#!/usr/bin/env bash
# Submit the same SGS workflow used by local/container entrypoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
node="" partition="" account="" config="" steps=""
output="${OUTPUT_DIR:-}" work="${WORK_DIR:-}"
case_name=channel-simple threads=4 jobs="" repeats=7 warmups=1 arch=auto
cpus=8 memory=32G time_limit=01:30:00 gpus=1 runtime=auto
sif="${SIF_FILE:-${workspace}/fortran-abomination.sif}"
wait_job=1 dry_run=0 interactive=0 jobs_explicit=0
extra=()

usage() {
    cat <<'EOF'
Usage: run_on_node.sh [OPTIONS] NODE [-- EXTRA_RUN_ELMM_ARGS...]
Submit a fresh native/SGS comparison; no local fallback or headnode setup.

  --case channel-simple|channel    Default: channel-simple
  --config DIR                    Explicit application configuration
  --steps N                       Optional timestep cap (smoke experiment)
  --threads N --jobs N             Defaults: 4 runtime threads, build jobs = CPUs
  --repeats N --warmups N          Defaults: 7 measured, 1 warmup
  --arch ARCH                     Default: auto, detected on the worker GPU
  -o, --output-dir DIR             Logs/results (default: results/runs/NODE-TIME)
  --work-dir DIR                   Writable work (default: OUTPUT/work)
  --runtime auto|apptainer|native  Default: auto
  --sif FILE                      Or SIF_FILE; auto uses it when available
  -p, --partition NAME            Presets: hopper01=gpu-ffa-gpulab, bw01=ffa-gpulab
  -A, --account NAME              Optional SLURM account
  -c, --cpus N --mem SIZE          Defaults: 8 CPUs, 32G; CPUs cover jobs/threads
  -t, --time HH:MM:SS --gpus N     Defaults: 01:30:00, 1 GPU
  --no-wait                       Submit without sbatch --wait
  --interactive                   Run on the allocated node via srun
  --dry-run                       Create/print the quoted job script only
  -h, --help
EOF
}
fail() { echo "ERROR: $*" >&2; exit 2; }
value() { [[ $# -ge 2 && -n "$2" ]] || fail "missing value for $1"; }
positive() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --case) value "$@"; case_name="$2"; shift 2 ;;
        --config) value "$@"; config="$2"; shift 2 ;;
        --steps) value "$@"; steps="$2"; shift 2 ;;
        --threads) value "$@"; threads="$2"; shift 2 ;;
        --jobs) value "$@"; jobs="$2"; jobs_explicit=1; shift 2 ;;
        --repeats) value "$@"; repeats="$2"; shift 2 ;;
        --warmups) value "$@"; warmups="$2"; shift 2 ;;
        --arch) value "$@"; arch="$2"; shift 2 ;;
        -o|--output|--output-dir) value "$@"; output="$2"; shift 2 ;;
        --work-dir) value "$@"; work="$2"; shift 2 ;;
        --runtime) value "$@"; runtime="$2"; shift 2 ;;
        --sif) value "$@"; sif="$2"; shift 2 ;;
        -p|--partition) value "$@"; partition="$2"; shift 2 ;;
        -A|--account) value "$@"; account="$2"; shift 2 ;;
        -c|--cpus) value "$@"; cpus="$2"; shift 2 ;;
        --mem) value "$@"; memory="$2"; shift 2 ;;
        -t|--time) value "$@"; time_limit="$2"; shift 2 ;;
        --gpus) value "$@"; gpus="$2"; shift 2 ;;
        --no-wait|--background) wait_job=0; shift ;;
        --wait) wait_job=1; shift ;;
        --interactive) interactive=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --) shift; extra=("$@"); break ;;
        -*) fail "unknown option $1" ;;
        *) [[ -z "$node" ]] || fail "unexpected argument $1 (use -- for workflow arguments)"; node="$1"; shift ;;
    esac
done
for argument in "${extra[@]}"; do
    case "${argument%%=*}" in
        --case|--config|--steps|--threads|--jobs|--repeats|--warmups|--arch|--output-dir|--output|-o|--work-dir)
            fail "use ${argument%%=*} before -- so resource and container paths can be checked" ;;
    esac
done
[[ -n "$node" && "$node" =~ ^[A-Za-z0-9_.-]+$ ]] || fail "one valid SLURM node is required"
[[ "$case_name" == channel-simple || "$case_name" == channel ]] || fail "case must be channel-simple or channel"
[[ "$runtime" == auto || "$runtime" == apptainer || "$runtime" == native ]] || fail "runtime must be auto, apptainer or native"
jobs="${jobs:-$cpus}"
for name in threads jobs repeats cpus gpus; do positive "$name" "${!name}"; done
[[ "$warmups" =~ ^(0|[1-9][0-9]*)$ ]] || fail "warmups must be a nonnegative integer"
[[ -z "$steps" ]] || positive steps "$steps"
(( cpus >= jobs && cpus >= threads )) || fail "cpus must cover both jobs and runtime threads"
[[ "$arch" =~ ^(auto|native|(sm_)?[0-9]+[af]?)$ ]] || fail "invalid CUDA architecture"
[[ -n "$partition" ]] || case "$node" in
    hopper01) partition=gpu-ffa-gpulab ;;
    bw01) partition=ffa-gpulab ;;
    *) fail "--partition is required for node $node" ;;
esac
workspace="$(cd -- "$workspace" && pwd -P)"
[[ -x "$workspace/containers/entrypoint.sh" ]] || fail "workspace entrypoint is missing"
[[ -z "$config" ]] || config="$(cd -- "$config" && pwd -P)"
[[ -n "$output" ]] || output="$workspace/results/runs/$node-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p -- "$output"
output="$(cd -- "$output" && pwd -P)"
work="$(realpath -m -- "${work:-$output/work}")"
sif="$(realpath -m -- "$sif")"
job_script="$output/submit-$node.sh"
[[ ! -e "$job_script" ]] || fail "existing job script must remain immutable: $job_script"
workflow=(run-elmm --case "$case_name" --threads "$threads")
jobs_index=$((${#workflow[@]} + 1))
workflow+=(--jobs "$jobs" --repeats "$repeats" --warmups "$warmups" --arch "$arch")
config_index=-1
if [[ -n "$config" ]]; then
    config_index=$((${#workflow[@]} + 1))
    workflow+=(--config "$config")
fi
[[ -z "$steps" ]] || workflow+=(--steps "$steps")
workflow+=("${extra[@]}")

# Keep resource requests in the batch file, as in the original launcher. Slurm
# site/user SBATCH_* settings can override these defaults; CLI requests cannot.
slurm_directive() {
    local argument="$2"
    [[ "$argument" != *$'\n'* && "$argument" != *$'\r'* ]] || fail "invalid Slurm $1 value"
    argument="${argument//\\/\\\\}"
    argument="${argument//\"/\\\"}"
    printf '#SBATCH --%s="%s"\n' "$1" "$argument"
}

# %q preserves spaces and shell metacharacters without interpolating user text.
(
    set -o noclobber
    {
        printf '%s\n' '#!/usr/bin/env bash'
        slurm_directive job-name "elmm-$node"
        slurm_directive nodelist "$node"
        slurm_directive partition "$partition"
        slurm_directive gpus "$gpus"
        slurm_directive cpus-per-task "$cpus"
        slurm_directive mem "$memory"
        slurm_directive time "$time_limit"
        slurm_directive output "$output/slurm-%j.out"
        slurm_directive error "$output/slurm-%j.err"
        [[ -z "$account" ]] || slurm_directive account "$account"
        printf '%s\n' 'set -euo pipefail'
        printf 'workspace=%q\noutput=%q\nwork=%q\nsif=%q\nruntime=%q\nconfig=%q\nconfig_index=%q\n' \
            "$workspace" "$output" "$work" "$sif" "$runtime" "$config" "$config_index"
        printf 'jobs_index=%q\njobs_explicit=%q\n' "$jobs_index" "$jobs_explicit"
        printf 'workflow=('; printf ' %q' "${workflow[@]}"; printf ' )\n'
        cat <<'EOF'
printf 'Worker: %s\nJob ID: %s\nOutput directory: %s\nWork directory: %s\n' \
    "${SLURMD_NODENAME:-${HOSTNAME:-unknown}}" "${SLURM_JOB_ID:-interactive}" "$output" "$work"
trap 'printf "Job exit status: %s\nResults and logs: %s\nReport: %s/experiment/report.json\nFigures: %s/experiment/figures\n" "$?" "$output" "$output" "$output"' EXIT
if [[ "$jobs_explicit" == 0 && "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]] && \
        (( SLURM_CPUS_PER_TASK < workflow[jobs_index] )); then
    workflow[jobs_index]="$SLURM_CPUS_PER_TASK"
fi
printf 'Allocated CPUs: %s\nBuild jobs: %s\n' "${SLURM_CPUS_PER_TASK:-scheduler default}" "${workflow[jobs_index]}"

# Restore the cluster module setup before deciding whether Apptainer is present.
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh || true
fi
if type module >/dev/null 2>&1; then
    echo 'Loading cluster modules: apptainer, cuda, gcc, Python...'
    module load apptainer || true
    module load cuda || true
    module load gcc || true
    module load python3.12 2>/dev/null || module load python/3.12 2>/dev/null || module load python-3.12 2>/dev/null || true
fi
if [[ "$runtime" == auto ]]; then
    if [[ -f "$sif" ]] && { command -v apptainer >/dev/null || command -v singularity >/dev/null; }; then
        runtime=apptainer
    else
        runtime=native
    fi
fi
mkdir -p -- "$work"
if [[ "$runtime" == apptainer ]]; then
    printf 'Running with Apptainer image: %s\n' "$sif"
    case "$work$config$workspace$output" in
        *:*|*,*) echo 'ERROR: Apptainer bind paths cannot contain colon or comma.' >&2; exit 2 ;;
    esac
    case "$work" in
        "$output"/*) container_work="/output/${work#"$output"/}" ;;
        *) container_work=/work ;;
    esac
    binds="$work:$container_work"
    if [[ -n "$config" ]]; then
        binds+=",$config:/container-config:ro"
        workflow[config_index]=/container-config
    fi
    export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+$APPTAINER_BINDPATH,}$binds"
    export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH:+$SINGULARITY_BINDPATH,}$binds"
    env OUTPUT_DIR="$output" WORKSPACE_ROOT="$workspace" SIF_FILE="$sif" \
        "$workspace/containers/run-apptainer.sh" "${workflow[@]}" \
        --output-dir /output --work-dir "$container_work"
    exit 0
fi
if [[ -z "${PYTHON:-}" ]]; then
    for candidate in python3.12 /usr/bin/python3.12 /usr/local/bin/python3.12 python3.13 python3.14 python3; do
        if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 12))' >/dev/null 2>&1; then
            export PYTHON="$candidate"
            break
        fi
    done
fi
if [[ -z "${PYTHON:-}" ]] || ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
    echo 'ERROR: Native compiler/pipeline execution requires Python 3.12+; load its module or set PYTHON.' >&2
    exit 2
fi
printf 'Running natively; compiler/pipeline Python: %s\n' "$PYTHON"
if [[ -z "${FC:-}" ]] && command -v gfortran-15 >/dev/null; then
    export FC=gfortran-15
fi
env WORKSPACE_ROOT="$workspace" WORK_DIR="$work" \
    "$workspace/containers/entrypoint.sh" "${workflow[@]}" \
    --output-dir "$output" --work-dir "$work"
EOF
    } > "$job_script"
)
chmod +x "$job_script"
printf 'Target node: %s\nPartition: %s\nAccount: %s\n' "$node" "$partition" "${account:-site default}"
printf 'Batch defaults: %s CPUs, %s memory, %s GPU(s), %s wall time\n' "$cpus" "$memory" "$gpus" "$time_limit"
printf 'Output directory: %s\nWork directory: %s\nJob script: %s\n' "$output" "$work" "$job_script"
printf 'Report: %s/experiment/report.json\nFigures: %s/experiment/figures\n' "$output" "$output"
printf 'Slurm logs: %s/slurm-<job-id>.out and .err\n' "$output"
if (( interactive )); then
    submission=(srun --nodelist="$node" --partition="$partition" --gpus="$gpus"
                --cpus-per-task="$cpus" --mem="$memory" --time="$time_limit")
    [[ -z "$account" ]] || submission+=(--account="$account")
    submission+=(bash "$job_script")
else
    submission=(sbatch --parsable)
    (( ! wait_job )) || submission+=(--wait)
    submission+=("$job_script")
fi
if (( dry_run )); then
    cat "$job_script"
    printf 'Submission command:'; printf ' %q' "${submission[@]}"; printf '\n'
    exit 0
fi
command -v "${submission[0]}" >/dev/null || fail "${submission[0]} is unavailable; use --dry-run or invoke entrypoint.sh locally"
if (( interactive )); then
    exec "${submission[@]}"
fi
echo "Submitting Slurm job on $node..."
status=0
submitted="$("${submission[@]}")" || status=$?
[[ -z "$submitted" ]] || printf '%s\n' "$submitted"
job_id="${submitted%%;*}"
if [[ "$job_id" =~ ^[0-9]+$ ]]; then
    printf 'Job ID: %s\nLog files: %s/slurm-%s.out\n           %s/slurm-%s.err\n' \
        "$job_id" "$output" "$job_id" "$output" "$job_id"
    printf 'Status: squeue -j %s\nFollow log: tail -f %q\nCancel: scancel %s\n' \
        "$job_id" "$output/slurm-$job_id.out" "$job_id"
fi
printf 'Submission/job exit status: %s\nResults: %s\n' "$status" "$output"
exit "$status"
