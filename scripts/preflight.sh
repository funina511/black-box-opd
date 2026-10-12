#!/usr/bin/env bash
# Accepts the same --model / --dataset / --run / --set selectors as training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${GAD_PYTHON:-/mnt/jinbo/miniconda3/envs/gad/bin/python}" \
  "$ROOT/gad/tools/preflight.py" "$@"
