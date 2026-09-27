#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/runs/gad_qwen3_1.7b/qwen3-1.7b-4x3090"
exec bash "$ROOT/gad/scripts/pipeline/run_math_eval.sh" \
  --out-dir "$RUN/eval" --python "${GAD_PYTHON:-/mnt/jinbo/miniconda3/envs/gad/bin/python}" "$@"
