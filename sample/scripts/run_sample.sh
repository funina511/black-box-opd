#!/usr/bin/env bash
# Offline data production for GAD on DAPO-Math-17k:
#   prompts -> teacher sampling -> teacher index -> GAD train/val parquet -> validation
#
# Usage:
#   bash LMOps/sample/scripts/run_sample.sh --config /abs/path/sample.yaml [--dry-run]
#                                             [--stage prompts|sample|convert_train|convert_dev|validate|eval]
#
# `--stage convert` is accepted as a group alias for `convert_train` + `convert_dev`.
# `--stage` may be repeated or given as a comma separated list (`--stage convert,validate`);
# every requested stage runs and an unknown one is rejected (exit 2) instead of silently
# skipping everything.
#
# Requires a running OpenAI-compatible teacher endpoint for the `sample` stage
# (see scripts/serve_teacher.sh). Everything else is CPU-only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG=""
DRY_RUN=0
ONLY_STAGE=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Stage names used by run_stage below, plus the accepted group aliases.
KNOWN_STAGES=(prompts sample convert_train convert_dev validate eval)
STAGE_GROUPS=(convert)
EXECUTED_STAGES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --stage) ONLY_STAGE="${ONLY_STAGE:+${ONLY_STAGE},}$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "${ONLY_STAGE}" ]]; then
  # ONLY_STAGE may be a comma separated list ("--stage sample,validate" or
  # "--stage sample --stage validate"); every element must be known.
  IFS=',' read -r -a _REQUESTED_STAGES <<< "${ONLY_STAGE}"
  for requested in "${_REQUESTED_STAGES[@]}"; do
    stage_known=0
    for candidate in "${KNOWN_STAGES[@]}" "${STAGE_GROUPS[@]}"; do
      [[ "${requested}" == "${candidate}" ]] && stage_known=1
    done
    if [[ "${stage_known}" -eq 0 ]]; then
      echo "ERROR: unknown --stage '${requested}'; valid values: ${KNOWN_STAGES[*]} (groups: ${STAGE_GROUPS[*]})" >&2
      exit 2
    fi
  done
fi

if [[ -z "${CONFIG}" ]]; then
  echo "ERROR: --config /abs/path/sample.yaml is required" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 2
fi

CONFIG="$(realpath "${CONFIG}")"

cfg() {
  "${PYTHON_BIN}" - "$CONFIG" "$1" <<'PY'
import sys
from pathlib import Path
import yaml

config_path, dotted = Path(sys.argv[1]), sys.argv[2]
data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
node = data
for part in dotted.split("."):
    if not isinstance(node, dict) or part not in node:
        print("")
        raise SystemExit(0)
    node = node[part]
path_keys = {
    "prompts.input_parquet", "prompts.out_dir", "prompts.dev_file",
    "teacher.index_out", "teacher.model_path", "teacher.log_file",
    "gad_data.out_train", "gad_data.out_val", "student.model_path",
    "student.tokenizer_path", "eval.sources_json", "eval.out_dir",
}
if dotted in path_keys and node:
    node = str((config_path.parent / Path(str(node)).expanduser()).resolve())
print("" if node is None else node)
PY
}

require_cfg() {
  local key="$1" value
  value="$(cfg "$key")"
  if [[ -z "${value}" ]]; then
    echo "ERROR: config key '${key}' is required but empty in ${CONFIG}" >&2
    exit 2
  fi
  printf '%s' "${value}"
}

stage_selected() {
  # ONLY_STAGE is a comma separated list; `--stage a --stage b` and `--stage a,b`
  # are equivalent, so an earlier --stage is never silently dropped.
  local name="$1" want
  [[ -z "${ONLY_STAGE}" ]] && return 0
  local list=",${ONLY_STAGE// /},"
  case "${name}" in
    convert_train|convert_dev) want=",convert," ;;
    *) want=",${name}," ;;
  esac
  [[ "${list}" == *"${want}"* ]] && return 0
  [[ "${list}" == *",${name},"* ]] && return 0
  return 1
}

run_stage() {
  local name="$1"; shift
  if ! stage_selected "${name}"; then
    echo "[run_sample] skipping stage '${name}'"
    return 0
  fi
  EXECUTED_STAGES+=("${name}")
  echo "[run_sample] ===== stage: ${name} ====="
  if [[ "${DRY_RUN}" == "1" ]]; then
    local rendered=""
    local arg
    for arg in "$@"; do
      rendered+="$(printf '%q' "${arg}") "
    done
    echo "[run_sample] dry-run: ${rendered}"
    return 0
  fi
  "$@"
}

OUT_DIR="$(require_cfg prompts.out_dir)"
STUDENT_MODEL="$(cfg student.model_path)"
STUDENT_TOKENIZER="$(cfg student.tokenizer_path)"
MAX_PROMPT_LEN="$(cfg lengths.max_prompt_length)"
MAX_RESPONSE_LEN="$(cfg lengths.max_response_length)"
ENABLE_THINKING="$(cfg teacher.enable_thinking)"
IPY="${PYTHON_BIN} -u"

echo "[run_sample] config   : ${CONFIG}"
echo "[run_sample] out_dir  : ${OUT_DIR}"
echo "[run_sample] python   : ${PYTHON_BIN}"
echo "[run_sample] dry-run  : ${DRY_RUN}"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUT_DIR}/logs"
  "${PYTHON_BIN}" - "${CONFIG}" "${OUT_DIR}/logs/resolved_sample_config.yaml" <<'PYCODE'
import sys
from pathlib import Path
import yaml
source, target = Path(sys.argv[1]), Path(sys.argv[2])
data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
paths = {
    "prompts": ("input_parquet", "out_dir", "dev_file"),
    "teacher": ("index_out", "model_path", "log_file"),
    "gad_data": ("out_train", "out_val"),
    "student": ("model_path", "tokenizer_path"),
    "eval": ("sources_json", "out_dir"),
}
for section, keys in paths.items():
    values = data.get(section, {})
    for key in keys:
        if values.get(key):
            values[key] = str((source.parent / Path(str(values[key])).expanduser()).resolve())
target.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")
PYCODE
fi

# ---------------------------------------------------------------- prompts ---
prompt_args=(
  "${SAMPLE_DIR}/prepare_prompts.py"
  --out-dir "${OUT_DIR}"
  --repo-id "$(cfg prompts.repo_id)"
  --filename "$(cfg prompts.filename)"
  --revision "$(cfg prompts.revision)"
  --seed "$(cfg prompts.seed)"
  --dev-size "$(cfg prompts.dev_size)"
  --on-conflict "$(cfg prompts.on_conflict)"
)
INPUT_PARQUET="$(cfg prompts.input_parquet)"
[[ -n "${INPUT_PARQUET}" ]] && prompt_args+=(--input "${INPUT_PARQUET}")
EXPECTED_UNIQUE="$(cfg prompts.expected_unique)"
[[ -n "${EXPECTED_UNIQUE}" ]] && prompt_args+=(--expected-unique "${EXPECTED_UNIQUE}")
if [[ "$(cfg prompts.card_full_train)" == "True" || "$(cfg prompts.card_full_train)" == "true" ]]; then
  prompt_args+=(--card-full-train)
  DEV_FILE="$(cfg prompts.dev_file)"
  if [[ -z "${DEV_FILE}" ]]; then
    echo "ERROR: prompts.card_full_train=true requires prompts.dev_file (an independent dev set)" >&2
    exit 2
  fi
  prompt_args+=(--dev-file "${DEV_FILE}")
  # Training on every DAPO question means an external dev benchmark may repeat
  # DAPO questions; record the overlap instead of refusing to build the split.
  prompt_args+=(--allow-dev-leak)
fi
if [[ -n "${STUDENT_TOKENIZER}" ]]; then
  prompt_args+=(--tokenizer "${STUDENT_TOKENIZER}" --max-prompt-length "${MAX_PROMPT_LEN}")
fi

run_stage prompts "${PYTHON_BIN}" "${prompt_args[@]}"

# ------------------------------------------------------------- sampling -----
INDEX_OUT="$(require_cfg teacher.index_out)"
thinking_flag="unset"
case "${ENABLE_THINKING}" in
  True|true) thinking_flag="true" ;;
  False|false) thinking_flag="false" ;;
  *) thinking_flag="unset" ;;
esac
# engine: openai (HTTP endpoint, default) | vllm_offline (teacher loaded in-process)
TEACHER_ENGINE="$(cfg teacher.engine)"
TEACHER_ENGINE="${TEACHER_ENGINE:-openai}"
LOGS_DIR="${OUT_DIR}/logs"
TEACHER_LOG_FILE="$(cfg teacher.log_file)"
TEACHER_LOG_FILE="${TEACHER_LOG_FILE:-${LOGS_DIR}/sample.log}"
SAMPLE_LOG_EVERY="$(cfg teacher.log_every)"
SAMPLE_LOG_EVERY="${SAMPLE_LOG_EVERY:-100}"

sample_args=(
  "${SAMPLE_DIR}/sample_teacher.py"
  --prompt-parquet "${OUT_DIR}/train.parquet"
  --prompt-parquet "${OUT_DIR}/dev.parquet"
  --out "${INDEX_OUT}"
  --m "$(cfg teacher.m)"
  --temperature "$(cfg teacher.temperature)"
  --top-p "$(cfg teacher.top_p)"
  --max-tokens "$(cfg teacher.max_tokens)"
  --enable-thinking "${thinking_flag}"
  # A detached run must never block on a terminal nobody reads any more.
  --progress log --log-every "${SAMPLE_LOG_EVERY}" --log-file "${TEACHER_LOG_FILE}"
)

case "${TEACHER_ENGINE}" in
  vllm_offline|vllm-offline)
    TEACHER_ENGINE="vllm_offline"
    MODEL_PATH="$(require_cfg teacher.model_path)"
    TEACHER_GPUS="$(cfg teacher.gpus)"
    if [[ -n "${TEACHER_GPUS}" ]]; then
      export CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}"
    fi
    sample_args+=(
      --backend vllm-offline
      --model-path "${MODEL_PATH}"
      --tensor-parallel-size "$(cfg teacher.tensor_parallel_size)"
      --max-model-len "$(cfg teacher.max_model_len)"
      --gpu-memory-utilization "$(cfg teacher.gpu_memory_utilization)"
    )
    TEACHER_DTYPE="$(cfg teacher.dtype)"
    [[ -n "${TEACHER_DTYPE}" ]] && sample_args+=(--dtype "${TEACHER_DTYPE}")
    TEACHER_BATCH="$(cfg teacher.batch_size)"
    [[ -n "${TEACHER_BATCH}" ]] && sample_args+=(--batch-size "${TEACHER_BATCH}")
    case "$(cfg teacher.enforce_eager)" in True|true) sample_args+=(--enforce-eager) ;; esac
    TEACHER_MODEL_ID="$(cfg teacher.model)"
    [[ -n "${TEACHER_MODEL_ID}" ]] && sample_args+=(--teacher-model "${TEACHER_MODEL_ID}")
    echo "[run_sample] teacher engine: vllm_offline (in-process; no server, no port, no sudo needed)"
    ;;
  openai|"")
    TEACHER_ENGINE="openai"
    sample_args+=(
      --backend openai
      --base-url "$(require_cfg teacher.base_url)"
      --teacher-model "$(require_cfg teacher.model)"
      --concurrency "$(cfg teacher.concurrency)"
      --timeout "$(cfg teacher.timeout)"
      --retries "$(cfg teacher.retries)"
    )
    API_KEY_ENV="$(cfg teacher.api_key_env)"
    if [[ -n "${API_KEY_ENV}" && -n "${!API_KEY_ENV:-}" ]]; then
      sample_args+=(--api-key "${!API_KEY_ENV}")
    else
      sample_args+=(--api-key EMPTY)
    fi
    echo "[run_sample] teacher engine: openai (endpoint $(cfg teacher.base_url); start it with scripts/serve_teacher.sh)"
    ;;
  *)
    echo "ERROR: unknown teacher.engine '${TEACHER_ENGINE}' (expected openai or vllm_offline)" >&2
    exit 2
    ;;
esac
[[ -n "${STUDENT_TOKENIZER}" ]] && sample_args+=(--student-tokenizer "${STUDENT_TOKENIZER}" --tokenizer-limit 500)
echo "[run_sample] sample log: ${TEACHER_LOG_FILE}"

run_stage sample "${PYTHON_BIN}" "${sample_args[@]}"

# --------------------------------------------------------------- convert ----
GAD_TRAIN="$(require_cfg gad_data.out_train)"
GAD_VAL="$(require_cfg gad_data.out_val)"
# The expected fingerprint is taken from the index header (when it already
# exists) and pinned for the conversion step, so a stale index cannot be used.
DIGEST=""
if [[ -f "${INDEX_OUT}" ]]; then
  DIGEST="$(head -n 1 "${INDEX_OUT}" | "${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.stdin.read() or "{}").get("teacher_fingerprint_digest",""))')"
fi

convert_args_for() {
  # $1 = prompt parquet, $2 = output parquet, $3 = selection; echoes the full argv
  local prompts="$1" out="$2" select="$3"
  printf '%s\n' "${SAMPLE_DIR}/build_gad_data.py"
  printf '%s\n' --prompt-parquet "${prompts}" --index "${INDEX_OUT}" --out "${out}"
  printf '%s\n' --select "${select}"
  if [[ -n "${DIGEST}" ]]; then
    printf '%s\n' --expect-fingerprint-digest "${DIGEST}"
  fi
  if [[ -n "${STUDENT_TOKENIZER}" ]]; then
    printf '%s\n' --student-tokenizer "${STUDENT_TOKENIZER}" --max-teacher-tokens "${MAX_RESPONSE_LEN}"
  fi
}

# Teacher-pool selection: "all" keeps every prompt that has a teacher answer (the ablation
# pool, and the default); "correct" keeps only prompts the teacher answered correctly,
# which is the original's correct-nontruncated pool. The original leaves its validation set
# unfiltered, so gad_data.select_val defaults to "all" independently of select.
GAD_SELECT="$(cfg gad_data.select)"; [[ -n "${GAD_SELECT}" ]] || GAD_SELECT=all
GAD_SELECT_VAL="$(cfg gad_data.select_val)"; [[ -n "${GAD_SELECT_VAL}" ]] || GAD_SELECT_VAL=all
echo "[run_sample] teacher pool selection: train=${GAD_SELECT} dev=${GAD_SELECT_VAL}"

mapfile -t CONVERT_TRAIN_ARGS < <(convert_args_for "${OUT_DIR}/train.parquet" "${GAD_TRAIN}" "${GAD_SELECT}")
mapfile -t CONVERT_DEV_ARGS < <(convert_args_for "${OUT_DIR}/dev.parquet" "${GAD_VAL}" "${GAD_SELECT_VAL}")
run_stage convert_train "${PYTHON_BIN}" "${CONVERT_TRAIN_ARGS[@]}"
run_stage convert_dev "${PYTHON_BIN}" "${CONVERT_DEV_ARGS[@]}"

# -------------------------------------------------------------- validate ----
validate_args=(
  "${SAMPLE_DIR}/validate_data.py"
  --train "${GAD_TRAIN}"
  --val "${GAD_VAL}"
  --index "${INDEX_OUT}"
  --max-prompt-length "${MAX_PROMPT_LEN}"
  --max-response-length "${MAX_RESPONSE_LEN}"
  --report "${OUT_DIR}/validation_report.json"
)
[[ -n "${DIGEST}" ]] && validate_args+=(--expect-fingerprint-digest "${DIGEST}")
[[ -n "${STUDENT_TOKENIZER}" ]] && validate_args+=(--student-tokenizer "${STUDENT_TOKENIZER}")
if [[ "$(cfg prompts.card_full_train)" == "True" || "$(cfg prompts.card_full_train)" == "true" ]]; then
  validate_args+=(--allow-val-leak)
fi
run_stage validate "${PYTHON_BIN}" "${validate_args[@]}"

# ------------------------------------------------------------------ eval ----
EVAL_OUT="$(cfg eval.out_dir)"
if [[ -n "${EVAL_OUT}" ]]; then
  eval_args=(
    "${SAMPLE_DIR}/prepare_eval_data.py"
    --out-dir "${EVAL_OUT}"
    --sources "$(cfg eval.sources_json)"
    --benchmarks "$(cfg eval.benchmarks)"
    --train-parquet "${OUT_DIR}/train.parquet"
    --val-parquet "${OUT_DIR}/dev.parquet"
  )
  if [[ "$(cfg eval.strict)" == "True" || "$(cfg eval.strict)" == "true" ]]; then
    eval_args+=(--strict)
  fi
  run_stage eval "${PYTHON_BIN}" "${eval_args[@]}"
else
  echo "[run_sample] eval.out_dir is empty; skipping external benchmark preparation"
fi

echo "[run_sample] requested stage selection: ${ONLY_STAGE:-all}"
echo "[run_sample] executed stages: ${EXECUTED_STAGES[*]-none}"

echo "[run_sample] all requested stages completed"
echo "[run_sample] GAD train : ${GAD_TRAIN}"
echo "[run_sample] GAD val   : ${GAD_VAL}"
