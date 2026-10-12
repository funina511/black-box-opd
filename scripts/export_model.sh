#!/usr/bin/env bash
# Export a stage checkpoint from its saved run configuration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="runs/gad_qwen3_0.6b_17k/qwen3-0.6b-17k-n2"
STAGE="gad"
STEP="latest"
selection=(--run "$ROOT/$RUN")
for arg in "$@"; do
  case "$arg" in
    --run|--run=*|--config|--config=*|--experiment|--experiment=*|--model-name|--model-name=*)
      selection=(); break ;;
  esac
done
exec "${GAD_PYTHON:-/mnt/jinbo/miniconda3/envs/gad/bin/python}" \
  "$ROOT/gad/tools/export_model.py" "${selection[@]}" --stage "$STAGE" --step "$STEP" "$@"
