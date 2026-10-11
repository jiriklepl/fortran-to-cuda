#!/usr/bin/env bash
# ==============================================================================
# run_on_node.sh — Run ELMM pipeline on HPC node (bw01 or hopper01)
# Wrapper around containers/run_node.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/containers/run_node.sh" "$@"
