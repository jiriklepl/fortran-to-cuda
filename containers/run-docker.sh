#!/usr/bin/env bash
# ==============================================================================
# run-docker.sh — Run the pipeline container with GPU support and volume mounts
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-fortran-abomination:latest}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/out}"

# Detect container engine
ENGINE="docker"
if ! command -v docker &>/dev/null; then
    if command -v podman &>/dev/null; then
        ENGINE="podman"
    else
        echo "ERROR: Neither 'docker' nor 'podman' found." >&2
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}"

# Check for GPU support
GPU_FLAGS=()
if command -v nvidia-smi &>/dev/null; then
    if [[ "${ENGINE}" == "docker" ]]; then
        GPU_FLAGS=("--gpus" "all")
    else
        GPU_FLAGS=("--device" "nvidia.com/gpu=all")
    fi
    echo "NVIDIA GPU detected; enabling GPU acceleration (${GPU_FLAGS[*]})."
else
    echo "No NVIDIA GPU detected; running in CPU-only mode."
fi

# Run container
echo "================================================================================"
echo " Running ${IMAGE_TAG}"
echo " Output Directory: ${OUTPUT_DIR} -> /output"
echo " User ID:          $(id -u):$(id -g)"
echo " Command:          ${*:-default (reproduce-plots)}"
echo "================================================================================"

"${ENGINE}" run --rm -it \
    "${GPU_FLAGS[@]}" \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -v "${OUTPUT_DIR}:/output" \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    "${IMAGE_TAG}" \
    "$@"

