#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${GAD_PYTHON:-python}" \
  "$ROOT/gad/tools/benchmark_train.py" "$@"
