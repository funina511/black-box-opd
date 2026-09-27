#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/runs/gad_qwen3_1.7b/qwen3-1.7b-4x3090"
mkdir -p "$RUN/reports"
exec "${GAD_PYTHON:-/mnt/jinbo/miniconda3/envs/gad/bin/python}" "$ROOT/gad/tools/preflight.py" \
  --config "$ROOT/experiments/qwen3_1.7b/train.yaml" \
  --report "$RUN/reports/preflight_$(date -u +%Y%m%dT%H%M%SZ).json" "$@"
