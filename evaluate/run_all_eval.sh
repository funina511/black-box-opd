#!/usr/bin/env bash
# Eight tasks, four samples per question, using the existing sequential evaluator.
# Separate profile: do not mix these results with historical per-task sample counts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/assets/datasets/math_eval/extended_dapo_v2"
# Install fault diagnostics before native-library imports on this H200 host.
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Keep vLLM/Inductor compilation caches on the H200 temporary storage.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/bloxopd_h200_cache/vllm}"

exec bash "$ROOT/evaluate/run_eval.sh" \
  --profile all8_dapo_v2_af1_n4_16k_h200_graph \
  --tasks AMC23,AIME24,AIME25,MATH-500,Minerva,Olympiad-Bench,GPQA,MMLU-Pro-Math \
  --dataset-path "Minerva=$EXT/Minerva/test.parquet" \
  --dataset-path "Olympiad-Bench=$EXT/Olympiad-Bench/test.parquet" \
  --dataset-path "GPQA=$EXT/GPQA/test.parquet" \
  --dataset-path "MMLU-Pro-Math=$EXT/MMLU-Pro-Math/test.parquet" \
  --gpu-ids 0 --n 4 --math500-n 4 \
  --max-tokens 16384 --max-model-len 18432 \
  --gpu-memory-utilization 0.85 --no-enforce-eager \
  "$@"
