#!/usr/bin/env bash
# ==============================================================================
# run-apptainer.sh — Run Apptainer SIF image with GPU passthrough and bind mounts
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SIF_FILE="${SIF_FILE:-${WORKSPACE_ROOT}/fortran-abomination.sif}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/out}"

APPTAINER_BIN="apptainer"
if ! command -v apptainer &>/dev/null; then
    if command -v singularity &>/dev/null; then
        APPTAINER_BIN="singularity"
    else
        echo "ERROR: Neither 'apptainer' nor 'singularity' found." >&2
        exit 1
    fi
fi

if [[ ! -f "${SIF_FILE}" ]]; then
    echo "ERROR: SIF image not found at '${SIF_FILE}'." >&2
    echo "Run ${SCRIPT_DIR}/build-apptainer.sh first." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

NV_FLAGS=()
if command -v nvidia-smi &>/dev/null; then
    NV_FLAGS=("--nv")
    echo "NVIDIA GPU detected; enabling GPU acceleration (--nv)."
else
    echo "No NVIDIA GPU detected; running in CPU-only mode."
fi

echo "================================================================================"
echo " Running SIF Image: ${SIF_FILE}"
echo " Output Directory:  ${OUTPUT_DIR} -> /output"
echo " Command:           ${*:-default (reproduce-plots)}"
echo "================================================================================"

"${APPTAINER_BIN}" run \
    "${NV_FLAGS[@]}" \
    --bind "${OUTPUT_DIR}:/output" \
    "${SIF_FILE}" \
    "$@"

