#!/usr/bin/env bash
# ==============================================================================
# run_node.sh — SLURM runner for ELMM pipeline on MFF HPC Metacenter nodes
#
# Automatically configures node-specific hardware, SLURM partitions, CUDA arch,
# and GPU resources for bw01 and hopper01, runnable from the cluster headnode.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Supported target nodes per MFF HPC Metacenter specification:
#   hopper01: AMD EPYC 9554 (96c/192t, 1024 GB RAM), 2x NVIDIA H100 PCIe 80GB (sm_90)
#   bw01:     AMD EPYC 9454 (48c/48t, 768 GB RAM), 2x NVIDIA RTX PRO Blackwell 96GB (sm_100)
SUPPORTED_NODES=("bw01" "hopper01")

NODE=""
MODE="elmm"
OUTPUT_DIR=""
PARTITION=""
ACCOUNT=""
TIME_LIMIT="01:30:00"
CPUS_PER_TASK=8
MEM_LIMIT="32G"
GPUS_COUNT=1
ELMM_STEPS=10
ELMM_THREADS=4
WAIT_FOR_JOB=1
INTERACTIVE=0
DRY_RUN=0

print_usage() {
    cat <<EOF
================================================================================
 ELMM Pipeline SLURM Runner (MFF HPC Metacenter)
================================================================================
Usage:
  $(basename "$0") <bw01|hopper01> [OPTIONS]

Target Nodes (strictly restricted):
  hopper01   NVIDIA H100 PCIe 80GB (Hopper CC 9.0, sm_90)
             CPU: AMD EPYC 9554 (96 cores / 192 threads, 1024 GB RAM)
             Default partition: gpu-ffa-gpulab

  bw01       NVIDIA RTX PRO Blackwell Server Edition 96GB (CC 10.0, sm_100)
             CPU: AMD EPYC 9454 (48 cores / 48 threads, 768 GB RAM)
             Default partition: ffa-gpulab

Options:
  -m, --mode MODE        Execution mode:
                           elmm        - Run fresh ELMM pipeline on GPU + produce report graphs (default)
                           benchmarks  - Run fresh microbenchmarks sweep on GPU + produce report graphs
                           all         - Run both ELMM pipeline + microbenchmarks + produce all graphs
                           reproduce   - Generate all report graphs from existing data without new runs
  -o, --output-dir DIR   Output directory for logs and figures (default: results/runs/<node>-<timestamp>)
  -p, --partition PART   Slurm partition override
  -A, --account ACCT     Slurm billing account override
  -t, --time TIME        Job walltime limit (default: 01:30:00)
  -c, --cpus CPUS        CPUs allocated per task (default: 8)
      --mem MEM          Memory allocation (default: 32G)
      --gpus COUNT       GPUs allocated (default: 1)
      --steps N          Number of ELMM timesteps (default: 10)
      --threads N        Number of OpenMP host threads (default: 4)
      --no-wait          Submit job to background without waiting
      --background       Alias for --no-wait
      --interactive      Run interactively via srun instead of sbatch
      --dry-run          Show Slurm script and submission command without executing
  -h, --help             Show this help message and exit

Examples:
  # Run ELMM pipeline on hopper01 and wait for publication graphs
  $(basename "$0") hopper01

  # Run on bw01 with custom output directory
  $(basename "$0") bw01 -o ./results/bw01-experiment

  # Submit without waiting (background execution)
  $(basename "$0") hopper01 --no-wait

  # Dry-run to inspect generated Slurm job script
  $(basename "$0") hopper01 --dry-run
================================================================================
EOF
}

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -A|--account)
            ACCOUNT="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -c|--cpus)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --mem)
            MEM_LIMIT="$2"
            shift 2
            ;;
        --gpus)
            GPUS_COUNT="$2"
            shift 2
            ;;
        --steps)
            ELMM_STEPS="$2"
            shift 2
            ;;
        --threads)
            ELMM_THREADS="$2"
            shift 2
            ;;
        --no-wait|--background)
            WAIT_FOR_JOB=0
            shift
            ;;
        --wait)
            WAIT_FOR_JOB=1
            shift
            ;;
        --interactive)
            INTERACTIVE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        hopper01|bw01)
            NODE="$1"
            shift
            ;;
        *)
            if [[ -z "${NODE}" && ! "$1" =~ ^- ]]; then
                NODE="$1"
                shift
            else
                echo "ERROR: Unknown option or parameter '$1'" >&2
                print_usage >&2
                exit 1
            fi
            ;;
    esac
done

# Validate target node
if [[ -z "${NODE}" ]]; then
    echo "ERROR: Missing target node. Specify either 'bw01' or 'hopper01'." >&2
    print_usage >&2
    exit 1
fi

if [[ "${NODE}" != "hopper01" && "${NODE}" != "bw01" ]]; then
    echo "ERROR: Node '${NODE}' is not supported. Only 'bw01' and 'hopper01' are valid targets." >&2
    echo "Available nodes: ${SUPPORTED_NODES[*]}" >&2
    exit 1
fi

# Configure node-specific settings
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${WORKSPACE_ROOT}/results/runs/${NODE}-${TIMESTAMP}"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

case "${NODE}" in
    hopper01)
        DEFAULT_PARTITION="gpu-ffa-gpulab"
        CUDA_ARCH="sm_90"
        GPU_ARCH_NUM="90"
        NODE_DESC="NVIDIA H100 PCIe 80GB (Hopper sm_90) on AMD EPYC 9554 (96c/192t, 1024GB RAM)"
        ;;
    bw01)
        DEFAULT_PARTITION="ffa-gpulab"
        CUDA_ARCH="sm_100"
        GPU_ARCH_NUM="100"
        NODE_DESC="NVIDIA RTX PRO Blackwell Server Edition 96GB (Blackwell sm_100) on AMD EPYC 9454 (48c/48t, 768GB RAM)"
        ;;
esac

PARTITION="${PARTITION:-${DEFAULT_PARTITION}}"

# Map mode to reproduce_plots.sh flags
case "${MODE}" in
    elmm)
        REPRODUCE_FLAG="--run-elmm"
        ;;
    benchmarks)
        REPRODUCE_FLAG="--run-benchmarks"
        ;;
    all)
        REPRODUCE_FLAG="--all"
        ;;
    reproduce)
        REPRODUCE_FLAG="--from-existing"
        ;;
    *)
        echo "ERROR: Unknown mode '${MODE}'. Choose elmm, benchmarks, all, or reproduce." >&2
        exit 1
        ;;
esac

ACCOUNT_OPT=""
if [[ -n "${ACCOUNT}" ]]; then
    ACCOUNT_OPT="#SBATCH --account=${ACCOUNT}"
fi

JOB_SCRIPT="${OUTPUT_DIR}/submit_${NODE}.sbatch"

# Generate the SLURM submission batch script
cat <<'EOF' > "${JOB_SCRIPT}"
#!/usr/bin/env bash
# ==============================================================================
# Auto-generated ELMM Pipeline SLURM Job Script
# ==============================================================================
#SBATCH --job-name=elmm-@@NODE@@
#SBATCH --nodelist=@@NODE@@
#SBATCH --partition=@@PARTITION@@
#SBATCH --gpus=@@GPUS_COUNT@@
#SBATCH --cpus-per-task=@@CPUS_PER_TASK@@
#SBATCH --mem=@@MEM_LIMIT@@
#SBATCH --time=@@TIME_LIMIT@@
#SBATCH --output=@@OUTPUT_DIR@@/slurm-%j.out
#SBATCH --error=@@OUTPUT_DIR@@/slurm-%j.err
@@ACCOUNT_OPT@@

set -euo pipefail

WORKSPACE_ROOT="@@WORKSPACE_ROOT@@"
OUTPUT_DIR="@@OUTPUT_DIR@@"
CUDA_ARCH="@@CUDA_ARCH@@"
NODE="@@NODE@@"
REPRODUCE_FLAG="@@REPRODUCE_FLAG@@"

echo "================================================================================"
echo " Starting ELMM Pipeline Execution on Worker Node: $(hostname)"
echo " Slurm Job ID:     ${SLURM_JOB_ID:-N/A}"
echo " Slurm Nodelist:   ${SLURM_NODELIST:-$(hostname)}"
echo " Allocated Cores:  ${SLURM_CPUS_PER_TASK:-@@CPUS_PER_TASK@@}"
echo " Allocated GPUs:   ${SLURM_GPUS_ON_NODE:-@@GPUS_COUNT@@}"
echo " Target Arch:      ${CUDA_ARCH}"
echo " Output Directory: ${OUTPUT_DIR}"
echo " Date:             $(date -u)"
echo "================================================================================"

# 1. Load HPC Environment Modules (LMod) if available on the node
if command -v module &>/dev/null || type module &>/dev/null 2>&1; then
    echo "Loading cluster environment modules..."
    module load apptainer 2>/dev/null || true
    module load cuda 2>/dev/null || true
    module load gcc 2>/dev/null || true
    module load python3.12 2>/dev/null || module load python/3.12 2>/dev/null || module load python-3.12 2>/dev/null || true
elif [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load apptainer 2>/dev/null || true
    module load cuda 2>/dev/null || true
    module load gcc 2>/dev/null || true
    module load python3.12 2>/dev/null || module load python/3.12 2>/dev/null || module load python-3.12 2>/dev/null || true
fi

# 2. Check for GPU device visibility
if command -v nvidia-smi &>/dev/null; then
    echo "GPU Status on Node:"
    nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found in PATH."
fi

# 3. Execution: Apptainer container or native environment
SIF_FILE="${WORKSPACE_ROOT}/fortran-abomination.sif"
APPTAINER_BIN=""
if command -v apptainer &>/dev/null; then
    APPTAINER_BIN="apptainer"
elif command -v singularity &>/dev/null; then
    APPTAINER_BIN="singularity"
fi

if [[ -f "${SIF_FILE}" && -n "${APPTAINER_BIN}" ]]; then
    echo ""
    echo "Executing via Apptainer image (${SIF_FILE}) with GPU acceleration (--nv)..."
    "${APPTAINER_BIN}" run \
        --nv \
        --bind "${OUTPUT_DIR}:/output" \
        --bind "${WORKSPACE_ROOT}:/workspace" \
        "${SIF_FILE}" \
        "${WORKSPACE_ROOT}/containers/reproduce_plots.sh" \
        ${REPRODUCE_FLAG} \
        --arch "${CUDA_ARCH}" \
        -o /output
else
    echo ""
    echo "Executing natively on worker node ${NODE}..."
    export MPLCONFIGDIR="/tmp/matplotlib-${USER:-tmp}"
    mkdir -p "${MPLCONFIGDIR}"
    chmod 777 "${MPLCONFIGDIR}" 2>/dev/null || true

    # Explicitly discover and export Python 3.12+ (worker nodes default to 3.9)
    for py_cand in python3.12 /usr/bin/python3.12 /usr/local/bin/python3.12 python3.13 python3.11; do
        if command -v "${py_cand}" &>/dev/null; then
            export PYTHON="$(command -v "${py_cand}")"
            echo "Selected Python interpreter: ${PYTHON} ($(${PYTHON} --version 2>&1))"
            break
        elif [[ -x "${py_cand}" ]]; then
            export PYTHON="${py_cand}"
            echo "Selected Python interpreter: ${PYTHON} ($(${PYTHON} --version 2>&1))"
            break
        fi
    done

    "${WORKSPACE_ROOT}/containers/reproduce_plots.sh" \
        ${REPRODUCE_FLAG} \
        --arch "${CUDA_ARCH}" \
        -o "${OUTPUT_DIR}"
fi

echo ""
echo "================================================================================"
echo " Pipeline execution completed successfully on $(hostname)"
echo " Output directory: ${OUTPUT_DIR}"
echo " Generated report figures and artifacts:"
ls -lh "${OUTPUT_DIR}" | grep -E '\.(png|pdf|svg|json|md)$' || true
echo "================================================================================"
EOF

# Substitute parameters into job script
sed -i \
    -e "s|@@NODE@@|${NODE}|g" \
    -e "s|@@PARTITION@@|${PARTITION}|g" \
    -e "s|@@GPUS_COUNT@@|${GPUS_COUNT}|g" \
    -e "s|@@CPUS_PER_TASK@@|${CPUS_PER_TASK}|g" \
    -e "s|@@MEM_LIMIT@@|${MEM_LIMIT}|g" \
    -e "s|@@TIME_LIMIT@@|${TIME_LIMIT}|g" \
    -e "s|@@OUTPUT_DIR@@|${OUTPUT_DIR}|g" \
    -e "s|@@ACCOUNT_OPT@@|${ACCOUNT_OPT}|g" \
    -e "s|@@WORKSPACE_ROOT@@|${WORKSPACE_ROOT}|g" \
    -e "s|@@CUDA_ARCH@@|${CUDA_ARCH}|g" \
    -e "s|@@REPRODUCE_FLAG@@|${REPRODUCE_FLAG}|g" \
    "${JOB_SCRIPT}"
chmod +x "${JOB_SCRIPT}"

echo "================================================================================"
echo " ELMM Pipeline Node Dispatcher"
echo " Target Node:      ${NODE}"
echo " Hardware:         ${NODE_DESC}"
echo " Slurm Partition:  ${PARTITION}"
echo " CUDA Arch Target: ${CUDA_ARCH}"
echo " Execution Mode:   ${MODE}"
echo " Output Directory: ${OUTPUT_DIR}"
echo " Job Script:       ${JOB_SCRIPT}"
echo "================================================================================"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY-RUN] Generated batch script (${JOB_SCRIPT}):"
    echo "--------------------------------------------------------------------------------"
    cat "${JOB_SCRIPT}"
    echo "--------------------------------------------------------------------------------"
    echo "[DRY-RUN] Slurm command: sbatch ${JOB_SCRIPT}"
    exit 0
fi

# Pre-bootstrap pipeline dependencies on headnode if needed (headnodes have internet access)
if [[ "${MODE}" == "elmm" || "${MODE}" == "all" ]]; then
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        HEAD_PY=""
        for p in python3.12 /usr/bin/python3.12 /usr/local/bin/python3.12 python3.13 python3.11 python3; do
            if command -v "$p" &>/dev/null && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' &>/dev/null; then
                HEAD_PY="$(command -v "$p")"
                break
            fi
        done
        if [[ -n "${HEAD_PY}" ]]; then
            # Ensure matplotlib and numpy are installed for Python 3.12 in user site (~/.local)
            if ! "${HEAD_PY}" -c 'import matplotlib, numpy' &>/dev/null; then
                echo "Installing matplotlib and numpy for ${HEAD_PY} into user site..."
                "${HEAD_PY}" -m pip install --quiet --user matplotlib numpy 2>/dev/null || true
            fi
            # Ensure elmm-pipeline setup is initialized
            if [[ ! -f "${WORKSPACE_ROOT}/elmm-pipeline/.cache/setup.json" && -d "${WORKSPACE_ROOT}/elmm-pipeline" ]]; then
                echo ""
                echo "Pre-bootstrapping ELMM pipeline on headnode (where internet/PyPI is accessible)..."
                (cd "${WORKSPACE_ROOT}/elmm-pipeline" && "${HEAD_PY}" pipeline.py setup --tools baseline,local --local-compiler "${WORKSPACE_ROOT}") || {
                    echo "Notice: Headnode pre-bootstrap returned non-zero code. Proceeding to job submission..."
                }
            fi
        fi
    fi
fi

# Check if Slurm sbatch is available
if ! command -v sbatch &>/dev/null; then
    echo "NOTICE: 'sbatch' command was not found on this machine."
    echo "If running on the cluster headnode (hpc.troja.mff.cuni.cz / parlab / gpulab),"
    echo "ensure SLURM client tools are in your PATH."
    echo ""
    echo "Running locally in fallback simulation mode for testing..."
    bash "${JOB_SCRIPT}"
    exit 0
fi

if [[ "${INTERACTIVE}" -eq 1 ]]; then
    echo "Submitting interactive job on ${NODE} via srun..."
    srun \
        --nodelist="${NODE}" \
        --partition="${PARTITION}" \
        --gpus="${GPUS_COUNT}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEM_LIMIT}" \
        --time="${TIME_LIMIT}" \
        bash "${JOB_SCRIPT}"
    exit 0
fi

# Submit batch job via sbatch
echo "Submitting SLURM batch job to ${NODE}..."
SBATCH_ARGS=()
if [[ "${WAIT_FOR_JOB}" -eq 1 ]]; then
    SBATCH_ARGS+=("--wait")
fi

SUBMIT_OUT=$(sbatch "${SBATCH_ARGS[@]}" "${JOB_SCRIPT}")
echo "${SUBMIT_OUT}"

JOB_ID=$(echo "${SUBMIT_OUT}" | grep -o -E '[0-9]+' | tail -n1 || echo "")

if [[ -n "${JOB_ID}" ]]; then
    LOG_OUT="${OUTPUT_DIR}/slurm-${JOB_ID}.out"
    LOG_ERR="${OUTPUT_DIR}/slurm-${JOB_ID}.err"
    echo ""
    echo "Job ID:    ${JOB_ID}"
    echo "Log files: ${LOG_OUT}"
    echo "           ${LOG_ERR}"

    if [[ "${WAIT_FOR_JOB}" -eq 0 ]]; then
        echo ""
        echo "Job submitted in background."
        echo "To check job status:     squeue -j ${JOB_ID}"
        echo "To view execution log:   tail -f ${LOG_OUT}"
        echo "To cancel job:           scancel ${JOB_ID}"
        exit 0
    fi

    # Job has completed (--wait was active)
    echo ""
    echo "================================================================================"
    echo " Job ${JOB_ID} on node ${NODE} has finished!"
    if [[ -f "${LOG_OUT}" ]]; then
        echo "--- Job Output Summary (last 20 lines) ---"
        tail -n 20 "${LOG_OUT}"
        echo "------------------------------------------"
    fi
fi

# Verify generated publication figures and reports
echo ""
echo "================================================================================"
echo " Generated Publication Figures & Reports in ${OUTPUT_DIR}:"
echo "================================================================================"

REPORT_FILES=(
    "complete-wall-relative.png:Six-policy complete application wall-clock ratio vs native"
    "complete-wall-relative.pdf:Six-policy complete application wall-clock ratio (PDF vector)"
    "complete-wall-relative.svg:Six-policy complete application wall-clock ratio (SVG vector)"
    "entry-time.png:Execution time per entry call across grid sizes"
    "transfer-volume.png:Host/device data transfer volume per entry"
    "scoped-wall-relative.png:Scoped region application comparison across momentum/SGS"
    "scoped-transfers-wall-relative.png:Scoped transfer controls (pinned, direct, pipelined)"
    "grid_512x512x512_niter100.png:Microbenchmark memory transfer vs kernel execution time"
    "elmm-run-timings.png:Fresh ELMM application benchmark run step & wall timings"
    "scoped-application-report.md:Verified timing summary table and provenance receipts"
    "elmm-run-report.md:Fresh ELMM run execution and verification report"
    "manifest.json:Cryptographic SHA-256 digests and file metadata manifest"
)

FOUND_COUNT=0
for entry in "${REPORT_FILES[@]}"; do
    fname="${entry%%:*}"
    desc="${entry#*:}"
    fpath="${OUTPUT_DIR}/${fname}"
    if [[ -f "${fpath}" ]]; then
        FOUND_COUNT=$((FOUND_COUNT + 1))
        fsize=$(ls -lh "${fpath}" | awk '{print $5}')
        printf "  ✓ %-35s (%6s) - %s\n" "${fname}" "${fsize}" "${desc}"
    fi
done

echo ""
echo "Total artifacts generated: ${FOUND_COUNT}"
echo "All files saved in:        ${OUTPUT_DIR}"
echo "================================================================================"
