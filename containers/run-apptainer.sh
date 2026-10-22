#!/usr/bin/env bash
# Run a SIF with writable output and only explicitly supplied source overrides.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SIF_FILE="${SIF_FILE:-${SCRIPT_DIR}/../fortran-abomination.sif}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../out}"

fail() { echo "ERROR: $*" >&2; exit 1; }
directory() {
    [[ -d "$1" ]] || fail "Directory not found: $1"
    (cd -- "$1" && pwd -P)
}

if command -v apptainer >/dev/null 2>&1; then
    ENGINE=apptainer
elif command -v singularity >/dev/null 2>&1; then
    ENGINE=singularity
else
    fail "Neither apptainer nor singularity was found."
fi
[[ -f "$SIF_FILE" ]] || fail "SIF image not found: $SIF_FILE"
SIF_FILE="$(directory "$(dirname -- "$SIF_FILE")")/$(basename -- "$SIF_FILE")"
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(directory "$OUTPUT_DIR")"
[[ -w "$OUTPUT_DIR" ]] || fail "Output directory is not writable: $OUTPUT_DIR"

flags=(run --cleanenv --no-home --pwd /output --bind "$OUTPUT_DIR:/output"
       --env OUTPUT_DIR=/output --env WORK_DIR=/output/work
       --env WORKSPACE_ROOT=/workspace --env PIPELINE_ROOT=/workspace/elmm-pipeline
       --env HOME=/output --env MPLCONFIGDIR=/tmp/matplotlib)
if [[ ${WORKSPACE_ROOT+x} ]]; then
    flags+=(--bind "$(directory "$WORKSPACE_ROOT"):/workspace:ro")
fi
if [[ ${PIPELINE_ROOT+x} ]]; then
    flags+=(--bind "$(directory "$PIPELINE_ROOT"):/workspace/elmm-pipeline:ro")
fi
if [[ ${COMPILER_ROOT+x} ]]; then
    flags+=(--bind "$(directory "$COMPILER_ROOT"):/compiler-source:ro"
            --env COMPILER_ROOT=/compiler-source)
fi

selection=
requested=false
if [[ ${GPU_DEVICES+x} ]]; then
    selection="$GPU_DEVICES"
    requested=true
elif [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    selection="$CUDA_VISIBLE_DEVICES"
    requested=true
else
    case "${1:-doctor}" in
        run-elmm|elmm|run-benchmarks|benchmark|e2e|benchmark-e2e|run-e2e) selection=all; requested=true ;;
    esac
fi
if [[ "$requested" == true ]]; then
    case "$selection" in
        ''|-1|none|void) flags+=(--env CUDA_VISIBLE_DEVICES=) ;;
        all) flags+=(--nv) ;;
        *) flags+=(--nv --env "CUDA_VISIBLE_DEVICES=$selection") ;;
    esac
fi

# --cleanenv still permits these host injection variables; use the flags above.
for name in "${!APPTAINERENV_@}" "${!SINGULARITYENV_@}"; do
    [[ -z "$name" ]] || unset "$name"
done
printf 'Container runtime: %s\nSIF image: %s\nOutput directory: %s -> /output\n' \
    "$ENGINE" "$SIF_FILE" "$OUTPUT_DIR"
printf 'Command:'
if (( $# )); then
    printf ' %q' "$@"
else
    printf ' <image default>'
fi
printf '\n'
exec "$ENGINE" "${flags[@]}" "$SIF_FILE" "$@"
