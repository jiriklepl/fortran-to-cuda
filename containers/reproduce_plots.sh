#!/usr/bin/env bash
# Explicit saved inputs only. Fresh application measurements use run-elmm.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/entrypoint.sh" reproduce-plots "$@"
