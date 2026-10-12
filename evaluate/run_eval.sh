#!/usr/bin/env bash
# Select an existing RUN; model/dataset/sampling settings are configured below.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${GAD_PYTHON:-/mnt/jinbo/miniconda3/envs/gad/bin/python}"
RUN="runs/gad_qwen3_0.6b_17k/qwen3-0.6b-17k-n2"
PROFILE="oprd_dapo_nonthinking_n8_16k"

# Select stage models by name. "base" comes from train.yaml;
# SeqKD/Warmup/GAD final checkpoints are located using the training run ID.
# Completed sharded checkpoints are exported to Hugging Face format automatically.
MODELS=(base seqkd gad)
# For any external checkpoint, add "label=/absolute/model/path" here and
# put "label" in MODELS. The directory must contain config.json.
MODEL_PATHS=()
# Optional saved step for a stage that was intentionally stopped before its
# configured final target, e.g. CHECKPOINT_STEPS=("gad=542").
# The evaluator verifies the checkpoint shards before exporting it.
CHECKPOINT_STEPS=()

# Select datasets and their exact Parquet files.
DATA_ROOT="/mnt/jinbo/RLRM/OPRD/datasets/test_data_dapo"
DATASETS=(
  "AMC23=$DATA_ROOT/AMC23/test.parquet"
  "AIME24=$DATA_ROOT/AIME24/test.parquet"
  "AIME25=$DATA_ROOT/AIME25/test.parquet"
  "MATH-500=$DATA_ROOT/MATH-500/test.parquet"
)
OPRD_ROOT="/mnt/jinbo/RLRM/OPRD"

GPU_IDS="1,2,3,4"
N=8
MATH500_N=2
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=-1
REPETITION_PENALTY=1.0
MAX_TOKENS=8192
MAX_MODEL_LEN=10240
GPU_MEMORY_UTILIZATION=0.85
SEED=42
MAX_PROMPTS=0  # 0 = full dataset; e.g. 2 for a small smoke run

# Change PROFILE for a new result set when models, data, or sampling change.
# --dry-run checks the plan without GPU use; --summary reads existing scores.
join_csv() {
  local IFS=,
  printf '%s' "$*"
}

eval_args=("$@")
selection=(--run "$ROOT/$RUN")
for arg in "$@"; do
  case "$arg" in
    --run|--run=*|--config|--config=*|--train-config|--train-config=*|--experiment|--experiment=*|--model-name|--model-name=*)
      selection=(); break ;;
  esac
done

task_names=()
dataset_args=()
for entry in "${DATASETS[@]}"; do
  if [[ "$entry" != *=* ]]; then
    echo "Invalid DATASETS entry: $entry (expected NAME=PATH)" >&2
    exit 2
  fi
  task_names+=("${entry%%=*}")
  dataset_args+=(--dataset-path "$entry")
done
model_args=()
for entry in "${MODEL_PATHS[@]}"; do
  model_args+=(--model-path "$entry")
done
checkpoint_args=()
for entry in "${CHECKPOINT_STEPS[@]}"; do
  checkpoint_args+=(--checkpoint-step "$entry")
done

exec "$PYTHON" "$ROOT/evaluate/evaluate.py" \
  "${selection[@]}" --oprd-root "$OPRD_ROOT" --dataset-root "$DATA_ROOT" \
  --profile "$PROFILE" --methods "$(join_csv "${MODELS[@]}")" \
  --tasks "$(join_csv "${task_names[@]}")" \
  "${model_args[@]}" "${dataset_args[@]}" "${checkpoint_args[@]}" \
  --gpu-ids "$GPU_IDS" --n "$N" --math500-n "$MATH500_N" \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
  --repetition-penalty "$REPETITION_PENALTY" \
  --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --seed "$SEED" \
  --max-prompts "$MAX_PROMPTS" "${eval_args[@]}"
