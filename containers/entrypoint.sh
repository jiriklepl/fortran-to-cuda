#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export PIPELINE_ROOT="${PIPELINE_ROOT:-${WORKSPACE_ROOT}/elmm-pipeline}"
export COMPILER_ROOT="${COMPILER_ROOT:-${WORKSPACE_ROOT}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fortran-plots-${UID}}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${MPLCONFIGDIR}"
PYTHON="${PYTHON:-python3}"
if [[ $# -eq 0 ]]; then set -- doctor; fi
case "$1" in
    e2e|benchmark-e2e|run-e2e)
        shift
        exec "${PYTHON}" "${SCRIPT_DIR}/reproduce_e2e.py" "$@" ;;
    doctor|prepare|run-elmm|elmm|reproduce-plots|reproduce|run-benchmarks|benchmark)
        exec "${PYTHON}" "${SCRIPT_DIR}/workflow.py" "$@" ;;
    test)
        shift
        export PYTHONPATH="${PIPELINE_ROOT}:${WORKSPACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        exec "${PYTHON}" -m pytest "${WORKSPACE_ROOT}/containers/tests" "$@" ;;
    *) exec "$@" ;;
esac
