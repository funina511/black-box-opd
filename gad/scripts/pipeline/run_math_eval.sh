#!/usr/bin/env bash
# External math evaluation for one model directory.
#
# Usage:
#   bash LMOps/gad/scripts/pipeline/run_math_eval.sh \
#     --model /abs/path/merged_hf --parquet /abs/path/math_eval/test.parquet \
#     --out-dir /abs/path/eval/gad --label gad [--n 16] [--math500-n 2]
#
#   # dev monitor protocol (deterministic, single sample)
#   bash LMOps/gad/scripts/pipeline/run_math_eval.sh \
#     --model ... --parquet /abs/path/dapo_all_v1/gad_dev.parquet \
#     --out-dir ... --n 1 --temperature 0 --benchmarks math_dapo
#
# The scorer is CARD-aligned; pass@1 is the per-question success rate averaged
# over questions ("solved at least once" is reported separately as pass@n).
set -euo pipefail

GAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${GAD_PYTHON:-python3}"

MODEL=""
PARQUET=""
OUT_DIR=""
LABEL="model"
BENCHMARKS=""
N=""
MATH500_N=""
TEMPERATURE=""
TOP_P=""
MAX_TOKENS=""
MAX_MODEL_LEN=""
TP="2"
BACKEND="vllm"
THINKING="false"
GPU_MEM="0.85"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --parquet) PARQUET="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --math500-n) MATH500_N="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --enable-thinking) THINKING="$2"; shift 2 ;;
    --gpu-mem) GPU_MEM="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --config) EVAL_CONFIG="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# A YAML config may supply any value that was not given on the command line.
if [[ -n "${EVAL_CONFIG:-}" ]]; then
  while IFS=$'\t' read -r key value; do
    case "${key}" in
      model) [[ -z "${MODEL}" ]] && MODEL="${value}" ;;
      parquet) [[ -z "${PARQUET}" ]] && PARQUET="${value}" ;;
      out_dir) [[ -z "${OUT_DIR}" ]] && OUT_DIR="${value}" ;;
      label) [[ "${LABEL}" == "model" ]] && LABEL="${value}" ;;
      benchmarks) [[ -z "${BENCHMARKS}" ]] && BENCHMARKS="${value}" ;;
      n) [[ -z "${N}" ]] && N="${value}" ;;
      math500_n) [[ -z "${MATH500_N}" ]] && MATH500_N="${value}" ;;
      temperature) [[ -z "${TEMPERATURE}" ]] && TEMPERATURE="${value}" ;;
      top_p) [[ -z "${TOP_P}" ]] && TOP_P="${value}" ;;
      max_tokens) [[ -z "${MAX_TOKENS}" ]] && MAX_TOKENS="${value}" ;;
      max_model_len) [[ -z "${MAX_MODEL_LEN}" ]] && MAX_MODEL_LEN="${value}" ;;
      tensor_parallel_size) [[ "${TP}" == "2" ]] && TP="${value}" ;;
      backend) [[ "${BACKEND}" == "vllm" ]] && BACKEND="${value}" ;;
      enable_thinking) [[ "${THINKING}" == "false" ]] && THINKING="${value}" ;;
      gpu_memory_utilization) [[ "${GPU_MEM}" == "0.85" ]] && GPU_MEM="${value}" ;;
    esac
  done < <("${PYTHON_BIN}" - "${EVAL_CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml
data = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
for key, value in data.items():
    if isinstance(value, (dict, list)):
        continue
    print(f"{key}\t{value}")
PY
)
fi

if [[ -z "${MODEL}" || -z "${PARQUET}" || -z "${OUT_DIR}" ]]; then
  echo "ERROR: --model, --parquet and --out-dir are required" >&2
  exit 2
fi

ARGS=(
  "${GAD_DIR}/tools/evaluate_math.py"
  --model "${MODEL}"
  --parquet "${PARQUET}"
  --out-dir "${OUT_DIR}"
  --label "${LABEL}"
  --backend "${BACKEND}"
  --tp "${TP}"
  --gpu-memory-utilization "${GPU_MEM}"
  --enable-thinking "${THINKING}"
)
[[ -n "${BENCHMARKS}" ]] && ARGS+=(--benchmarks "${BENCHMARKS}")
[[ -n "${N}" ]] && ARGS+=(--n "${N}")
[[ -n "${MATH500_N}" ]] && ARGS+=(--math500-n "${MATH500_N}")
[[ -n "${TEMPERATURE}" ]] && ARGS+=(--temperature "${TEMPERATURE}")
[[ -n "${TOP_P}" ]] && ARGS+=(--top-p "${TOP_P}")
[[ -n "${MAX_TOKENS}" ]] && ARGS+=(--max-tokens "${MAX_TOKENS}")
[[ -n "${MAX_MODEL_LEN}" ]] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")

echo "[math-eval] model=${MODEL}"
echo "[math-eval] data =${PARQUET}"
echo "[math-eval] out  =${OUT_DIR}"
exec "${PYTHON_BIN}" "${ARGS[@]}" "${EXTRA[@]}"
