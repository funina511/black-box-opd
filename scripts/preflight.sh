#!/usr/bin/env bash
# Accepts the same --model / --dataset / --run / --set selectors as training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${GAD_PYTHON:-python}" \
  "$ROOT/gad/tools/preflight.py" "$@"
