#!/usr/bin/env bash
# Edit these values for NEW experiments; full configs are generated under runs/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="" # Empty: use train.yaml selection.model; e.g. qwen3_0.6b
DATASET="" # Empty: use train.yaml selection.dataset; e.g. dapo_math_5k
NAME=""
STAGES="seqkd,warmup,gad"
OVERRIDES=() # e.g. ("rollout.n=2" "budget.stage_epochs.gad=3")

selection=(--name "$NAME")
[[ -z "$MODEL" ]] || selection+=(--model "$MODEL")
[[ -z "$DATASET" ]] || selection+=(--dataset "$DATASET")
for arg in "$@"; do
  case "$arg" in
    --run|--run=*|--config|--config=*|--model-name|--model-name=*|--experiment|--experiment=*|--experiment-name|--experiment-name=*)
      selection=(); break ;;
  esac
done
if [[ ${#selection[@]} -gt 0 ]]; then
  for value in "${OVERRIDES[@]}"; do selection+=(--set "$value"); done
fi
exec "${GAD_PYTHON:-/mnt/jinbo/miniconda3/envs/gad/bin/python}" "$ROOT/gad/tools/pipeline.py" \
  --stages "$STAGES" "${selection[@]}" "$@"
