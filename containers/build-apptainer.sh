#!/usr/bin/env bash
# ==============================================================================
# build-apptainer.sh — Build Apptainer / Singularity SIF image
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SIF_OUTPUT="${SIF_OUTPUT:-${WORKSPACE_ROOT}/fortran-abomination.sif}"
DEF_FILE="${SCRIPT_DIR}/Apptainer.def"
FROM_DOCKER=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            SIF_OUTPUT="$2"
            shift 2
            ;;
        --from-docker)
            FROM_DOCKER=1
            shift
            ;;
        --fakeroot)
            EXTRA_ARGS+=("--fakeroot")
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -o, --output FILE   Output SIF file path (default: ./fortran-abomination.sif)
  --from-docker       Build directly from local Docker image (docker-daemon)
  --fakeroot          Use --fakeroot for non-root build
  -h, --help          Show this help message
EOF
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

APPTAINER_BIN="apptainer"
if ! command -v apptainer &>/dev/null; then
    if command -v singularity &>/dev/null; then
        APPTAINER_BIN="singularity"
    else
        echo "ERROR: Neither 'apptainer' nor 'singularity' found." >&2
        exit 1
    fi
fi

echo "================================================================================"
echo " Building Apptainer / Singularity SIF Image"
echo " Tool:       ${APPTAINER_BIN}"
echo " Output SIF: ${SIF_OUTPUT}"
echo " Context:    ${WORKSPACE_ROOT}"
echo "================================================================================"

if [[ "${FROM_DOCKER}" -eq 1 ]]; then
    DOCKER_TAG="${DOCKER_TAG:-fortran-abomination:latest}"
    echo "Converting local Docker image '${DOCKER_TAG}' to SIF..."
    "${APPTAINER_BIN}" build "${EXTRA_ARGS[@]}" "${SIF_OUTPUT}" "docker-daemon:${DOCKER_TAG}"
else
    echo "Building SIF from definition file '${DEF_FILE}'..."
    (cd "${WORKSPACE_ROOT}" && "${APPTAINER_BIN}" build "${EXTRA_ARGS[@]}" "${SIF_OUTPUT}" "${DEF_FILE}")
fi

echo ""
echo "Successfully generated: ${SIF_OUTPUT}"

