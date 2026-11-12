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
    resident)
        shift
        cmake_ready=false
        if command -v cmake >/dev/null 2>&1 && version="$(cmake --version 2>/dev/null)" && \
                [[ "$version" =~ ^cmake\ version\ ([0-9]+)\. ]] && (( BASH_REMATCH[1] >= 4 )); then
            cmake_ready=true
        fi
        if ! "${PYTHON}" -c 'import fparser, islpy, numpy, matplotlib' >/dev/null 2>&1 || \
                [[ "$cmake_ready" != true ]] || ! command -v ninja >/dev/null 2>&1; then
            work="${WORK_DIR:-${OUTPUT_DIR:-${WORKSPACE_ROOT}/out}/work}"
            environment="${work}/resident-python"
            [[ ! -e "$environment" ]] || { echo "ERROR: resident Python environment already exists: $environment" >&2; exit 1; }
            mkdir -p -- "$work"
            echo "Preparing resident Python and build tools in $environment"
            "${PYTHON}" -m venv "$environment"
            "$environment/bin/python" -m pip install --disable-pip-version-check \
                -r "${WORKSPACE_ROOT}/requirements.txt" numpy==2.5.3 cmake==4.0.3 ninja==1.13.0
            export PATH="$environment/bin:$PATH"
            PYTHON="$environment/bin/python"
        fi
        exec "${PYTHON}" "${SCRIPT_DIR}/resident_comparison.py" "$@" ;;
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
