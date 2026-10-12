#!/usr/bin/env bash
# Run the baked image, optionally replacing explicit source directories.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
IMAGE_TAG="${IMAGE_TAG:-fortran-abomination:latest}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../out}"

fail() { echo "ERROR: $*" >&2; exit 1; }
directory() {
    [[ -d "$1" ]] || fail "Directory not found: $1"
    (cd -- "$1" && pwd -P)
}

if command -v docker >/dev/null 2>&1; then
    ENGINE=docker
    ENGINE_KIND=docker
    # Some clusters provide Podman through the Docker compatibility command.
    version="$(docker --version 2>/dev/null || true)"
    if [[ "${version,,}" == *podman* ]]; then ENGINE_KIND=podman; fi
elif command -v podman >/dev/null 2>&1; then
    ENGINE=podman
    ENGINE_KIND=podman
else
    fail "Neither docker nor podman was found."
fi
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(directory "$OUTPUT_DIR")"
[[ -w "$OUTPUT_DIR" ]] || fail "Output directory is not writable: $OUTPUT_DIR"

host_uid="$(id -u)"
flags=(run --rm --user "$host_uid:$(id -g)" --workdir /output
       --volume "$OUTPUT_DIR:/output"
       --env OUTPUT_DIR=/output --env WORK_DIR=/output/work
       --env WORKSPACE_ROOT=/workspace --env PIPELINE_ROOT=/workspace/elmm-pipeline
       --env HOME=/output --env MPLCONFIGDIR=/tmp/matplotlib)
if [[ "$ENGINE_KIND" == podman && "$host_uid" != 0 ]]; then
    # Keep the caller's IDs mapped to themselves so host-owned output is writable.
    flags+=(--userns keep-id)
fi
if [[ -t 0 && -t 1 ]]; then flags+=(-it); fi
if [[ ${WORKSPACE_ROOT+x} ]]; then
    flags+=(--volume "$(directory "$WORKSPACE_ROOT"):/workspace:ro")
fi
if [[ ${PIPELINE_ROOT+x} ]]; then
    flags+=(--volume "$(directory "$PIPELINE_ROOT"):/workspace/elmm-pipeline:ro")
fi
if [[ ${COMPILER_ROOT+x} ]]; then
    flags+=(--volume "$(directory "$COMPILER_ROOT"):/compiler-source:ro"
            --env COMPILER_ROOT=/compiler-source)
fi

# CUDA visibility is an explicit device selection, not evidence that CUDA works.
selection=
requested=false
visible=void
if [[ ${GPU_DEVICES+x} ]]; then
    selection="$GPU_DEVICES"
    requested=true
elif [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    selection="$CUDA_VISIBLE_DEVICES"
    requested=true
else
    case "${1:-doctor}" in
        run-elmm|elmm|run-benchmarks|benchmark) selection=all; requested=true ;;
    esac
fi
if [[ -n "${SLURM_JOB_ID:-${SLURM_JOBID:-}}" &&
      "$selection" =~ (^|,)[0-9]+(,|$) ]]; then
    fail "SLURM GPU ordinals may differ from Docker/CDI device IDs; set GPU_DEVICES=GPU-UUID for the allocated GPU."
fi
if [[ "$requested" == true ]]; then
    case "$selection" in
        ''|-1|none|void) flags+=(--env CUDA_VISIBLE_DEVICES=) ;;
        *)
            visible="$selection"
            if [[ "$ENGINE_KIND" == docker ]]; then
                if [[ "$selection" == all ]]; then
                    flags+=(--gpus all)
                else
                    # Docker parses this argument as CSV; keep a device list together.
                    flags+=(--gpus "\"device=$selection\"")
                fi
            else
                # CDI supplies devices; image defaults must not trigger legacy hooks.
                flags+=(--unsetenv NVIDIA_VISIBLE_DEVICES --unsetenv CUDA_VISIBLE_DEVICES)
                IFS=, read -r -a devices <<< "$selection"
                for device in "${devices[@]}"; do
                    [[ -n "$device" ]] || fail "Empty GPU device in selection: $selection"
                    flags+=(--device "nvidia.com/gpu=$device")
                done
            fi
            # Do not apply host CUDA ordinals again after filtering NVIDIA devices.
            ;;
    esac
fi
if [[ "$ENGINE_KIND" == docker ]]; then
    # Also constrain hosts configured to use the NVIDIA runtime by default.
    flags+=(--env "NVIDIA_VISIBLE_DEVICES=$visible")
elif [[ "$visible" == void ]]; then
    flags+=(--env NVIDIA_VISIBLE_DEVICES=void)
fi

exec "$ENGINE" "${flags[@]}" "$IMAGE_TAG" "$@"
