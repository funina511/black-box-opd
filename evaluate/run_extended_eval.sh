#!/usr/bin/env bash
# Extended tasks only, with the same DAPO answer-line prompt as AIME/AMC/MATH-500 and training.
# Sampling defaults match run_eval.sh except per-task sample counts; later arguments override these.
# Profile af1 = Minerva/Olympiad-Bench scored with evaluate/answer_forms.py (answer_forms_v1).
# Rescore an older ext4_dapo_v2_taskn_8k result without generation:
#   bash evaluate/run_extended_eval.sh --run RUN --methods ... --allow-new-eval-scorer \
#     --reuse-generation-from ext4_dapo_v2_taskn_8k
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/assets/datasets/math_eval/extended_dapo_v2"
exec bash "$ROOT/evaluate/run_eval.sh" \
  --profile ext4_dapo_v2_af1_taskn_16k \
  --tasks Minerva,Olympiad-Bench,GPQA,MMLU-Pro-Math \
  --dataset-path "Minerva=$EXT/Minerva/test.parquet" \
  --dataset-path "Olympiad-Bench=$EXT/Olympiad-Bench/test.parquet" \
  --dataset-path "GPQA=$EXT/GPQA/test.parquet" \
  --dataset-path "MMLU-Pro-Math=$EXT/MMLU-Pro-Math/test.parquet" \
  --task-n Minerva=4 --task-n Olympiad-Bench=2 --task-n GPQA=8 --task-n MMLU-Pro-Math=2 \
  "$@"
