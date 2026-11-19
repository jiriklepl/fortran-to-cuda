#!/usr/bin/env bash
# ==============================================================================
# Stable top-level entrypoint for fresh native/resident GPU comparisons on Slurm.
# See containers/README.md; all arguments go to containers/run_node.sh.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/containers/run_node.sh" "$@"
