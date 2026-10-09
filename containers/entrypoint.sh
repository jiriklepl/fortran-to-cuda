#!/usr/bin/env bash
# ==============================================================================
# entrypoint.sh — Container entrypoint for Fortran/CUDA benchmark pipeline
# ==============================================================================
set -e

# Setup environment variables
export MPLCONFIGDIR="/tmp/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
chmod 777 "${MPLCONFIGDIR}" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "/workspace" ]]; then
    export WORKSPACE_ROOT="/workspace"
else
    export WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
export PYTHONPATH="${WORKSPACE_ROOT}:${WORKSPACE_ROOT}/elmm-pipeline:${PYTHONPATH:-}"

# Create /output if missing and ensure writable
mkdir -p /output 2>/dev/null || true
chmod 777 /output 2>/dev/null || true

# If called with no arguments, default to reproduce-plots
if [[ $# -eq 0 ]]; then
    set -- "reproduce-plots"
fi

COMMAND="$1"
shift || true

case "${COMMAND}" in
    reproduce-plots|reproduce)
        exec "${WORKSPACE_ROOT}/containers/reproduce_plots.sh" --from-existing "$@"
        ;;
    run-benchmarks|benchmark)
        exec "${WORKSPACE_ROOT}/containers/reproduce_plots.sh" --run-benchmarks "$@"
        ;;
    run-elmm|elmm)
        exec "${WORKSPACE_ROOT}/containers/reproduce_plots.sh" --run-elmm "$@"
        ;;
    test)
        echo "Running pytest test suites..."
        exec pytest "${WORKSPACE_ROOT}/compiler/tests" "${WORKSPACE_ROOT}/benchmarks/tests" "$@"
        ;;
    bash|sh)
        exec /bin/bash "$@"
        ;;
    *)
        # Execute arbitrary user command
        exec "${COMMAND}" "$@"
        ;;
esac
