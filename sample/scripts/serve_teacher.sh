#!/usr/bin/env bash
# Serve the teacher as an OpenAI-compatible endpoint (vLLM) for sample_teacher.py.
#
# The teacher only produces text; no logits are needed, so a plain chat server is
# enough. Keep this process off the training GPUs budget in mind: the GAD stages
# do not need it at all -- stop it after the index is built.
#
# Usage:
#   TEACHER_MODEL_PATH=/models/Qwen3-30B-A3B-Instruct-2507 \
#   TEACHER_SERVED_NAME=Qwen3-30B-A3B-Instruct-2507 \
#   GAD_TEACHER_PORT=18080 TEACHER_TP=2 \
#   bash LMOps/sample/scripts/serve_teacher.sh
set -euo pipefail

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:?set TEACHER_MODEL_PATH to the teacher checkpoint directory}"
TEACHER_SERVED_NAME="${TEACHER_SERVED_NAME:-$(basename "${TEACHER_MODEL_PATH}")}"
GAD_TEACHER_PORT="${GAD_TEACHER_PORT:-18080}"
TEACHER_TP="${TEACHER_TP:-2}"
TEACHER_MAX_LEN="${TEACHER_MAX_LEN:-40960}"
TEACHER_GPU_MEM="${TEACHER_GPU_MEM:-0.90}"
TEACHER_GPUS="${TEACHER_GPUS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -n "${TEACHER_GPUS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}"
  echo "[serve_teacher] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "[serve_teacher] model   : ${TEACHER_MODEL_PATH}"
echo "[serve_teacher] served  : ${TEACHER_SERVED_NAME}"
echo "[serve_teacher] port     : ${GAD_TEACHER_PORT}"
echo "[serve_teacher] tp       : ${TEACHER_TP}"
echo "[serve_teacher] max-len  : ${TEACHER_MAX_LEN}"
echo "[serve_teacher] endpoint : http://127.0.0.1:${GAD_TEACHER_PORT}/v1"

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${TEACHER_MODEL_PATH}" \
  --served-model-name "${TEACHER_SERVED_NAME}" \
  --host 127.0.0.1 \
  --port "${GAD_TEACHER_PORT}" \
  --tensor-parallel-size "${TEACHER_TP}" \
  --max-model-len "${TEACHER_MAX_LEN}" \
  --gpu-memory-utilization "${TEACHER_GPU_MEM}" \
  --trust-remote-code
