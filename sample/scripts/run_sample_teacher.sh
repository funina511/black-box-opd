#!/usr/bin/env bash
# One-command teacher driver for the offline `sample` stage.
#
# The teacher is an OpenAI-compatible *server*, so it is a separate long-running
# process from the sampler that talks to it over HTTP. This wrapper removes the
# "two windows" bookkeeping:
#
#   1. probe <base_url>/models - reuse an endpoint that is already serving;
#   2. otherwise start vLLM in the background (nohup, log file) and wait until the
#      endpoint is actually ready (loading ~57 GiB on 4 PCIe 3090s takes minutes);
#   3. check that the *served model name* matches `teacher.model` in the config -
#      a mismatch would otherwise fail all 17,917 requests;
#   4. run the requested data stages through run_sample.sh;
#   5. stop the server again - but only one this wrapper started itself.
#
# It also exports a clean NO_PROXY for 127.0.0.1: a host value such as
# `localhost,127.0.0.1,::1,[::1]` makes httpx raise
# `InvalidURL: Invalid port: ':1]'` when the OpenAI client is built.
#
# Usage:
#   bash LMOps/sample/scripts/run_sample_teacher.sh \
#     --config /abs/path/sample.yaml \
#     --teacher-model /models/Qwen3-30B-A3B-Instruct-2507 \
#     --teacher-gpus 1,2,3,4 --teacher-tp 4 [--stage sample] [--keep-teacher]
#
#   # a server is already up (started by you or an earlier run): never start/stop one
#   bash LMOps/sample/scripts/run_sample_teacher.sh --config ... --no-server --stage convert
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# The sampler is an HTTP client on the same host; never route that through a proxy.
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

CONFIG=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAGES="sample"
TEACHER_MODEL=""
TEACHER_SERVED_NAME=""
TEACHER_PORT="18080"
TEACHER_TP="4"
TEACHER_GPUS=""
TEACHER_MAX_LEN="12288"
TEACHER_GPU_MEM="0.90"
READY_TIMEOUT="1800"
LOG_DIR=""
KEEP_TEACHER=0
NO_SERVER=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --stage) STAGES="$2"; shift 2 ;;
    --teacher-model) TEACHER_MODEL="$2"; shift 2 ;;
    --teacher-served-name) TEACHER_SERVED_NAME="$2"; shift 2 ;;
    --teacher-port) TEACHER_PORT="$2"; shift 2 ;;
    --teacher-tp) TEACHER_TP="$2"; shift 2 ;;
    --teacher-gpus) TEACHER_GPUS="$2"; shift 2 ;;
    --teacher-max-len) TEACHER_MAX_LEN="$2"; shift 2 ;;
    --teacher-gpu-mem) TEACHER_GPU_MEM="$2"; shift 2 ;;
    --ready-timeout) READY_TIMEOUT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --keep-teacher) KEEP_TEACHER=1; shift ;;
    --no-server) NO_SERVER=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CONFIG}" || ! -f "${CONFIG}" ]]; then
  echo "ERROR: --config /abs/path/sample.yaml is required (and must exist)" >&2
  exit 2
fi

# Read a dotted key from the YAML config (same contract as run_sample.sh).
cfg() {
  "${PYTHON_BIN}" - "$CONFIG" "$1" <<'PY'
import sys
from pathlib import Path
import yaml

config_path, dotted = sys.argv[1], sys.argv[2]
data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
node = data
for part in dotted.split("."):
    if not isinstance(node, dict) or part not in node:
        print("")
        raise SystemExit(0)
    node = node[part]
print("" if node is None else node)
PY
}

BASE_URL="$(cfg teacher.base_url)"
CONFIG_MODEL="$(cfg teacher.model)"
INDEX_OUT="$(cfg teacher.index_out)"
CONFIG_ENGINE="$(cfg teacher.engine)"
CONFIG_ENGINE="${CONFIG_ENGINE:-openai}"
case "${CONFIG_ENGINE}" in
  vllm_offline|vllm-offline)
    # The teacher runs inside the sampler: nothing to start, nothing to stop.
    echo "[run_sample_teacher] teacher.engine=${CONFIG_ENGINE}: the teacher is loaded inside the sampler;"
    echo "[run_sample_teacher] no server is started (one process, no port, no sudo needed)"
    echo "[run_sample_teacher] stages: ${STAGES}"
    offline_extra=()
    [[ "${DRY_RUN}" -eq 1 ]] && offline_extra+=(--dry-run)
    bash "${SCRIPT_DIR}/run_sample.sh" --config "${CONFIG}" --python "${PYTHON_BIN}" \
      --stage "${STAGES}" "${offline_extra[@]}"
    echo "[run_sample_teacher] done (stages: ${STAGES})"
    exit 0
    ;;
esac
if [[ -z "${BASE_URL}" || -z "${CONFIG_MODEL}" ]]; then
  echo "ERROR: teacher.base_url and teacher.model must be set in ${CONFIG} (or set teacher.engine: vllm_offline)" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"
if [[ -z "${TEACHER_SERVED_NAME}" ]]; then
  TEACHER_SERVED_NAME="$(basename "${TEACHER_MODEL:-${CONFIG_MODEL}}")"
fi
if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="$(dirname "${CONFIG}")/logs"
fi
mkdir -p "${LOG_DIR}"
TEACHER_LOG="${LOG_DIR}/teacher_server.log"
TEACHER_PID_FILE="${LOG_DIR}/teacher_server.pid"

SERVER_PID=""
STARTED_SERVER=0
stop_server() {
  if [[ "${STARTED_SERVER}" -eq 1 && "${KEEP_TEACHER}" -eq 0 && -n "${SERVER_PID}" ]]; then
    echo "[run_sample_teacher] stopping the teacher server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  elif [[ "${STARTED_SERVER}" -eq 1 ]]; then
    echo "[run_sample_teacher] leaving the teacher server running (pid ${SERVER_PID}, --keep-teacher)"
    echo "${SERVER_PID}" > "${TEACHER_PID_FILE}"
  fi
}
trap stop_server EXIT

served_models() {
  curl -s --max-time 10 --noproxy '*' "${BASE_URL}/models" 2>/dev/null
}

model_is_served() {
  local payload="$1"
  "${PYTHON_BIN}" - "$CONFIG_MODEL" "$payload" <<'PY'
import json, sys
wanted, raw = sys.argv[1], sys.argv[2]
try:
    data = json.loads(raw)
except Exception:
    raise SystemExit(1)
ids = [item.get("id") for item in (data.get("data") or [])]
raise SystemExit(0 if wanted in ids else 1)
PY
}

echo "[run_sample_teacher] config      : ${CONFIG}"
echo "[run_sample_teacher] base_url    : ${BASE_URL}"
echo "[run_sample_teacher] model        : ${CONFIG_MODEL}"
echo "[run_sample_teacher] stages       : ${STAGES}"
echo "[run_sample_teacher] index_out    : ${INDEX_OUT:-<unset>}"
echo "[run_sample_teacher] NO_PROXY     : ${NO_PROXY}"

PAYLOAD="$(served_models || true)"
if [[ "${NO_SERVER}" -eq 0 && -z "${PAYLOAD}" ]]; then
  if [[ -z "${TEACHER_MODEL}" ]]; then
    echo "ERROR: no endpoint at ${BASE_URL} and --teacher-model was not given" >&2
    echo "       pass --teacher-model PATH (and optionally --teacher-gpus/--teacher-tp)" >&2
    exit 2
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[run_sample_teacher] dry-run: would serve ${TEACHER_MODEL} (tp=${TEACHER_TP}, gpus='${TEACHER_GPUS}') on port ${TEACHER_PORT}"
  else
    echo "[run_sample_teacher] starting the teacher (tp=${TEACHER_TP}, gpus='${TEACHER_GPUS:-all}', log=${TEACHER_LOG})"
    TEACHER_MODEL_PATH="${TEACHER_MODEL}" \
    TEACHER_SERVED_NAME="${TEACHER_SERVED_NAME}" \
    TEACHER_GPUS="${TEACHER_GPUS}" \
    TEACHER_TP="${TEACHER_TP}" \
    GAD_TEACHER_PORT="${TEACHER_PORT}" \
    TEACHER_MAX_LEN="${TEACHER_MAX_LEN}" \
    TEACHER_GPU_MEM="${TEACHER_GPU_MEM}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    nohup bash "${SCRIPT_DIR}/serve_teacher.sh" >> "${TEACHER_LOG}" 2>&1 &
    SERVER_PID="$!"
    STARTED_SERVER=1
    echo "${SERVER_PID}" > "${TEACHER_PID_FILE}"
    echo "[run_sample_teacher] waiting up to ${READY_TIMEOUT}s for ${BASE_URL}/models ..."
    deadline=$(( $(date +%s) + READY_TIMEOUT ))
    while true; do
      PAYLOAD="$(served_models || true)"
      [[ -n "${PAYLOAD}" ]] && break
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: the teacher process exited during start-up; see ${TEACHER_LOG}" >&2
        tail -n 20 "${TEACHER_LOG}" >&2 || true
        exit 1
      fi
      if [[ "$(date +%s)" -ge "${deadline}" ]]; then
        echo "ERROR: the teacher did not become ready within ${READY_TIMEOUT}s; see ${TEACHER_LOG}" >&2
        exit 1
      fi
      sleep 15
    done
    echo "[run_sample_teacher] teacher is ready"
  fi
elif [[ -n "${PAYLOAD}" ]]; then
  echo "[run_sample_teacher] reusing the endpoint already running at ${BASE_URL} (not started by this script)"
fi

if [[ -n "${PAYLOAD}" ]] && ! model_is_served "${PAYLOAD}"; then
  echo "ERROR: ${BASE_URL} serves a different model name than teacher.model='${CONFIG_MODEL}'" >&2
  echo "       served: $(printf '%s' "${PAYLOAD}" | head -c 200)" >&2
  echo "       fix teacher.model (or --teacher-served-name) before sampling." >&2
  exit 1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[run_sample_teacher] dry-run: would run run_sample.sh --stage ${STAGES}"
  exit 0
fi

bash "${SCRIPT_DIR}/run_sample.sh" --config "${CONFIG}" --python "${PYTHON_BIN}" --stage "${STAGES}"

if [[ -n "${INDEX_OUT}" && -f "${INDEX_OUT}" ]]; then
  "${PYTHON_BIN}" - "${INDEX_OUT}" <<'PY' || true
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
records = 0
digest = None
failures = 0
with path.open(encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("uid") is None:
            digest = row.get("teacher_fingerprint_digest")
            continue
        records += 1
        if not row.get("teacher_answers"):
            failures += 1
print(f"[run_sample_teacher] index: {records} records, digest={digest}, empty-answer records={failures}")
PY
  [[ -f "${INDEX_OUT}.failures.jsonl" ]] && echo "[run_sample_teacher] failures: ${INDEX_OUT}.failures.jsonl"
fi

echo "[run_sample_teacher] done (stages: ${STAGES})"
