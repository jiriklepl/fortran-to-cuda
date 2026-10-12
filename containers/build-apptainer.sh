#!/usr/bin/env bash
# ==============================================================================
# build-apptainer.sh — Build Apptainer / Singularity SIF image
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SIF_OUTPUT="${SIF_OUTPUT:-${WORKSPACE_ROOT}/fortran-abomination.sif}"
FROM_DOCKER=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            [[ $# -ge 2 ]] || { echo "missing output path" >&2; exit 2; }
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

SIF_OUTPUT="$(realpath -m -- "$SIF_OUTPUT")"
[[ ! -e "$SIF_OUTPUT" ]] || { echo "ERROR: Refusing to replace $SIF_OUTPUT" >&2; exit 2; }
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
    # %files has no Docker-style ignore support. Stage source before building,
    # so caches and 100+ GB result archives never enter the image or sandbox.
    context="$(mktemp -d "${CONTAINER_BUILD_TMPDIR:-${TMPDIR:-/tmp}}/fortran-image.XXXXXXXX")"
    trap 'rm -rf -- "$context"' EXIT
    tar -C "$WORKSPACE_ROOT" --exclude-vcs --exclude=.cache --exclude=.venv \
        --exclude=work --exclude=results --exclude=bin --exclude=__pycache__ \
        --exclude=.pytest_cache --exclude='*.pyc' --exclude='*.o' --exclude='*.mod' \
        --exclude=CODE_MAP.md --exclude=MEMORY_MODEL_PLAN.md \
        -cf - requirements.txt compiler benchmarks elmm-pipeline containers | tar -C "$context" -xf -
    (cd -- "$context" && "$APPTAINER_BIN" build "${EXTRA_ARGS[@]}" "$SIF_OUTPUT" containers/Apptainer.def)
fi

echo ""
echo "Successfully generated: ${SIF_OUTPUT}"

