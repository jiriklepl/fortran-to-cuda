#!/usr/bin/env bash
# ==============================================================================
# build-docker.sh — Build the Docker image for the pipeline
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-fortran-abomination:latest}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.6.2-devel-ubuntu24.04}"
BUILD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --base-image)
            BASE_IMAGE="$2"
            shift 2
            ;;
        --no-cache)
            BUILD_ARGS+=("--no-cache")
            shift
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [-t TAG] [--base-image IMAGE] [--no-cache]"
            exit 0
            ;;
        *)
            BUILD_ARGS+=("$1")
            shift
            ;;
    esac
done

ENGINE="docker"
if ! command -v docker &>/dev/null; then
    if command -v podman &>/dev/null; then
        ENGINE="podman"
    else
        echo "ERROR: Neither 'docker' nor 'podman' command was found." >&2
        echo "Please install Docker or Podman, or build with Apptainer on HPC." >&2
        exit 1
    fi
fi

echo "================================================================================"
echo " Building Container Image"
echo " Engine:     ${ENGINE}"
echo " Tag:        ${IMAGE_TAG}"
echo " Base Image: ${BASE_IMAGE}"
echo " Context:    ${WORKSPACE_ROOT}"
echo " Dockerfile: ${SCRIPT_DIR}/Dockerfile"
echo "================================================================================"

"${ENGINE}" build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_TAG}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    "${BUILD_ARGS[@]}" \
    "${WORKSPACE_ROOT}"

echo ""
echo "Successfully built ${IMAGE_TAG}."

