#!/usr/bin/env bash
# ==============================================================================
# reproduce_plots.sh — Reproduce benchmark & policy figures from the pipeline
#
# Generates all publication-grade figures found in benchmark results:
#   1. grid_512x512x512_niter100.{png,pdf} (Memory vs Kernel time)
#   2. complete-wall-relative.{png,pdf,svg} (Whole application vs Native)
#   3. entry-time.{png,pdf,svg}            (Time per entry call)
#   4. transfer-volume.{png,pdf,svg}        (Host/Device data volume)
#   5. scoped-wall-relative.{png,pdf,svg}  (Scoped region application timing)
#   6. scoped-transfers-wall-relative.{png,pdf,svg} (Scoped transfer controls)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default output directory: /output if running in container, else local out/figures
if [[ -d "/output" ]] && [[ -w "/output" ]]; then
    OUTPUT_DIR="/output"
elif [[ -d "/output" ]]; then
    OUTPUT_DIR="/output"
else
    OUTPUT_DIR="${WORKSPACE_ROOT}/benchmarks/results/figures_reproduced"
fi

MODE="from-existing"
CUDA_ARCH=""
VERBOSE=0

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -o, --output-dir DIR   Directory where generated plots will be saved
                         (default: /output if present, else benchmarks/results/figures_reproduced)
  --from-existing        Generate all plots from archived benchmark reports and data (default)
  --run-benchmarks       Run fresh microbenchmark sweep on GPU and generate plots
  --run-elmm             Run fresh ELMM application pipeline (requires visible NVIDIA GPU)
  --all                  Run fresh microbenchmarks + ELMM pipeline + generate all plots
  --arch ARCH            Target CUDA architecture (e.g. sm_80, sm_90, sm_89, sm_86, sm_70)
  -v, --verbose          Enable verbose logging
  -h, --help             Show this help message and exit

Examples:
  # Reproduce all plots immediately into /output
  $(basename "$0") --output-dir /output

  # Run fresh benchmarks on an NVIDIA A100 GPU and output plots
  $(basename "$0") --run-benchmarks --arch sm_80 -o /output

  # Run fresh benchmarks on an NVIDIA H100 GPU and output plots
  $(basename "$0") --run-benchmarks --arch sm_90 -o /output
EOF
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --from-existing)
            MODE="from-existing"
            shift
            ;;
        --run-benchmarks)
            MODE="run-benchmarks"
            shift
            ;;
        --run-elmm)
            MODE="run-elmm"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

if [[ "${VERBOSE}" -eq 1 ]]; then
    set -x
fi

# Detect Python environment
if [[ -x "/opt/venv/bin/python" ]]; then
    PYTHON="/opt/venv/bin/python"
elif [[ -x "${WORKSPACE_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${WORKSPACE_ROOT}/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    echo "ERROR: No Python interpreter found." >&2
    exit 1
fi

# Setup matplotlib cache dir
export MPLCONFIGDIR="/tmp/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
chmod 777 "${MPLCONFIGDIR}" 2>/dev/null || true

# Setup PYTHONPATH
export PYTHONPATH="${WORKSPACE_ROOT}:${WORKSPACE_ROOT}/elmm-pipeline:${PYTHONPATH:-}"

# Create destination directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/ordinary-figures"

echo "================================================================================"
echo " Pipeline Figure Reproduction Engine"
echo " Workspace:   ${WORKSPACE_ROOT}"
echo " Output Dir:  ${OUTPUT_DIR}"
echo " Python:      $(${PYTHON} --version 2>&1) (${PYTHON})"
echo " Mode:        ${MODE}"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo 'N/A')"
    DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo 'N/A')"
    echo " GPU Device:  ${GPU_NAME} (Driver: ${DRIVER_VER})"
else
    echo " GPU Device:  None detected (Offline/Container CPU Mode)"
fi
echo "================================================================================"

# Execute fresh benchmarks if requested
if [[ "${MODE}" == "run-benchmarks" || "${MODE}" == "all" ]]; then
    echo ""
    echo "[Step 1/2] Running benchmark suite on GPU..."
    
    # Auto-detect CUDA_ARCH if not explicitly provided
    if [[ -z "${CUDA_ARCH}" ]] && command -v nvidia-smi &>/dev/null; then
        CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.' || true)"
        if [[ -n "${CAP}" ]]; then
            CUDA_ARCH="sm_${CAP}"
            echo "  Auto-detected CUDA architecture: ${CUDA_ARCH}"
        fi
    fi

    MAKE_ARGS=()
    if [[ -n "${CUDA_ARCH}" ]]; then
        MAKE_ARGS+=("CUDA_ARCH=${CUDA_ARCH}")
    fi
    if command -v gfortran-15 &>/dev/null; then
        MAKE_ARGS+=("FC=gfortran-15")
    fi
    if command -v g++-14 &>/dev/null; then
        MAKE_ARGS+=("CXX=g++-14")
    fi

    echo "  Validating deterministic correctness across cases..."
    "${PYTHON}" -m benchmarks.harness.check

    FRESH_CSV="${OUTPUT_DIR}/fresh_baseline.csv"
    echo "  Running measurement sweep -> ${FRESH_CSV}..."
    "${PYTHON}" -m benchmarks.harness.run > "${FRESH_CSV}"

    echo "  Generating harness plots from fresh sweep..."
    "${PYTHON}" -m benchmarks.harness.plot --results "${FRESH_CSV}"
fi

# Execute ELMM pipeline if requested
if [[ "${MODE}" == "run-elmm" || "${MODE}" == "all" ]]; then
    echo ""
    echo "[Step 2/2] Running ELMM application pipeline..."
    if [[ -d "${WORKSPACE_ROOT}/elmm-pipeline" ]]; then
        ELMM_ARGS=(run --example channel --steps 10 --threads 4 --repeats 3 --warmups 1)
        if command -v gfortran-15 &>/dev/null; then
            ELMM_ARGS+=(--fc gfortran-15)
        fi
        if command -v nvcc &>/dev/null && command -v nvidia-smi &>/dev/null; then
            ELMM_ARGS+=(--target gpu --cuda-host g++-14)
            if [[ -n "${CUDA_ARCH}" && "${CUDA_ARCH}" != "native" ]]; then
                # Strip sm_ prefix if present for pipeline argument
                NUM_ARCH="${CUDA_ARCH#sm_}"
                ELMM_ARGS+=(--gpu-arch "${NUM_ARCH}")
            fi
        fi
        (cd "${WORKSPACE_ROOT}/elmm-pipeline" && "${PYTHON}" pipeline.py "${ELMM_ARGS[@]}")
    else
        echo "  WARNING: elmm-pipeline directory not found. Skipping ELMM run."
    fi
fi

echo ""
echo "Generating publication plots..."

# ------------------------------------------------------------------------------
# 1. Harness Microbenchmark Plots (grid_{gx}x{gy}x{gz}_niter{niter}.{png,pdf})
# ------------------------------------------------------------------------------
echo "  [1/4] Harness microbenchmark figures (Memory transfer vs Kernel time)..."
BASELINE_CSV="${WORKSPACE_ROOT}/benchmarks/results/baseline.csv"
if [[ -f "${BASELINE_CSV}" ]]; then
    "${PYTHON}" -m benchmarks.harness.plot --results "${BASELINE_CSV}"
    
    # Copy generated figures to output
    if [[ -d "${WORKSPACE_ROOT}/benchmarks/results/figures" ]]; then
        cp -v "${WORKSPACE_ROOT}/benchmarks/results/figures"/grid_*.{png,pdf} "${OUTPUT_DIR}/" 2>/dev/null || true
    fi
else
    echo "    Notice: ${BASELINE_CSV} not found, skipping."
fi

# ------------------------------------------------------------------------------
# 2. Campaign Ordinary 6-Policy Figures
#    (complete-wall-relative, entry-time, transfer-volume)
# ------------------------------------------------------------------------------
echo "  [2/4] Policy comparison figures (complete-wall-relative, entry-time, transfer-volume)..."
# Locate campaign report
CAMPAIGN_REPORT=""
POTENTIAL_REPORTS=(
    "${WORKSPACE_ROOT}/benchmarks/results/generic-elmm-capabilities-20261009/campaigns/20261009T152400Z-m2-source-regions-boundary-views/ordinary/report.json"
    "${WORKSPACE_ROOT}/benchmarks/results/generic-elmm-capabilities-20261009/campaigns/20261009T130153Z-m1-numerical-closures-verified/ordinary/report.json"
    "${WORKSPACE_ROOT}/benchmarks/results/structured-offload-20261007T145715Z/report.json"
)

for r in "${POTENTIAL_REPORTS[@]}"; do
    if [[ -f "$r" ]]; then
        CAMPAIGN_REPORT="$r"
        break
    fi
done

if [[ -n "${CAMPAIGN_REPORT}" && -f "${CAMPAIGN_REPORT}" ]]; then
    echo "    Using report: ${CAMPAIGN_REPORT}"
    "${PYTHON}" "${WORKSPACE_ROOT}/benchmarks/harness/plot_strategies.py" \
        "${CAMPAIGN_REPORT}" \
        --output "${OUTPUT_DIR}/ordinary-figures" \
        --label "Datacenter GPU Pipeline"

    # Also copy main policy plots to top-level output for direct access
    cp -v "${OUTPUT_DIR}/ordinary-figures"/complete-wall-relative.{png,pdf,svg} "${OUTPUT_DIR}/" 2>/dev/null || true
    cp -v "${OUTPUT_DIR}/ordinary-figures"/entry-time.{png,pdf,svg} "${OUTPUT_DIR}/" 2>/dev/null || true
    cp -v "${OUTPUT_DIR}/ordinary-figures"/transfer-volume.{png,pdf,svg} "${OUTPUT_DIR}/" 2>/dev/null || true
else
    echo "    Notice: No campaign ordinary report found, skipping."
fi

# ------------------------------------------------------------------------------
# 3. Scoped Application Wall Relative Plot (scoped-wall-relative.{png,pdf,svg})
# ------------------------------------------------------------------------------
echo "  [3/4] Scoped region application comparison figure (scoped-wall-relative)..."
M2_CAMPAIGN="${WORKSPACE_ROOT}/benchmarks/results/generic-elmm-capabilities-20261009/campaigns/20261009T152400Z-m2-source-regions-boundary-views"
M2_SCRIPT="${WORKSPACE_ROOT}/benchmarks/results/generic-elmm-capabilities-20261009/report_limited_m2.py"

if [[ -d "${M2_CAMPAIGN}" && -f "${M2_SCRIPT}" ]]; then
    "${PYTHON}" "${M2_SCRIPT}" --campaign "${M2_CAMPAIGN}" >/dev/null 2>&1 || true
    if [[ -d "${M2_CAMPAIGN}/figures" ]]; then
        cp -v "${M2_CAMPAIGN}/figures"/scoped-wall-relative.{png,pdf,svg} "${OUTPUT_DIR}/" 2>/dev/null || true
    fi
    if [[ -f "${M2_CAMPAIGN}/report.md" ]]; then
        cp -v "${M2_CAMPAIGN}/report.md" "${OUTPUT_DIR}/scoped-application-report.md" 2>/dev/null || true
    fi
else
    echo "    Notice: M2 campaign directory not found, skipping."
fi

# ------------------------------------------------------------------------------
# 4. Scoped Transfer Wall Relative Plot (scoped-transfers-wall-relative)
# ------------------------------------------------------------------------------
echo "  [4/4] Scoped transfer control figures (scoped-transfers-wall-relative)..."
POTENTIAL_TRANSFER_DIRS=(
    "${WORKSPACE_ROOT}/benchmarks/results/generic-residency-20261009/scoped-generic-2b-20261009T054608Z-2b-pipelined-scoped-transfers-accepted/figures"
    "${WORKSPACE_ROOT}/benchmarks/results/generic-residency-20261009/milestones/20261009T065104Z-pipelined-scoped-transfers/scoped-generic-figures"
)

for tdir in "${POTENTIAL_TRANSFER_DIRS[@]}"; do
    if [[ -d "$tdir" && -f "$tdir/scoped-transfers-wall-relative.png" ]]; then
        cp -v "$tdir"/scoped-transfers-wall-relative.{png,pdf,svg} "${OUTPUT_DIR}/" 2>/dev/null || true
        break
    fi
done

# ------------------------------------------------------------------------------
# Generate provenance manifest
# ------------------------------------------------------------------------------
MANIFEST="${OUTPUT_DIR}/manifest.json"
echo ""
echo "Writing provenance manifest -> ${MANIFEST}..."

"${PYTHON}" - <<EOF
import json, hashlib, os, sys, datetime, socket

output_dir = "${OUTPUT_DIR}"
files = []
for fname in sorted(os.listdir(output_dir)):
    fpath = os.path.join(output_dir, fname)
    if os.path.isfile(fpath) and fname != "manifest.json":
        with open(fpath, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        files.append({
            "name": fname,
            "size_bytes": os.path.getsize(fpath),
            "sha256": h
        })

manifest = {
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "python_version": sys.version,
    "plots_count": len(files),
    "plots": files
}

with open("${MANIFEST}", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest written with {len(files)} generated artifacts.")
EOF

echo ""
echo "================================================================================"
echo " Reproduction Complete! Summary of generated artifacts in ${OUTPUT_DIR}:"
ls -lh "${OUTPUT_DIR}" | grep -E "\.(png|pdf|svg|json|md)$" || true
echo "================================================================================"

