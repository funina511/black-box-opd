#!/usr/bin/env bash
# Shared helpers for the GAD pipeline entry scripts.
#
# Every script in this directory is a thin wrapper around tools/pipeline.py, so
# the single source of truth for the stage commands, budgets and gates lives in
# Python (testable on CPU) instead of in shell.
set -euo pipefail

GAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_DIR="${GAD_DIR}/tools"
PIPELINE_PY="${TOOLS_DIR}/pipeline.py"

# Resolve the interpreter: $GAD_PYTHON > config run.python_bin > python3.
# The config is read with a plain text scan (no PyYAML) because the shell's own
# python3 may not have it - the interpreter we are looking for is exactly the
# one that does.
resolve_python() {
  local config="$1"
  if [[ -n "${GAD_PYTHON:-}" ]]; then
    echo "${GAD_PYTHON}"
    return 0
  fi
  if [[ -n "${config}" && -f "${config}" ]]; then
    local found
    found="$(awk '
      /^run:[[:space:]]*$/ { in_run = 1; next }
      /^[^[:space:]#]/ { in_run = 0 }
      in_run && /^[[:space:]]+python_bin:[[:space:]]*/ {
        sub(/^[[:space:]]*python_bin:[[:space:]]*/, "");
        sub(/[[:space:]]*#.*$/, "");
        gsub(/^["'"'"']|["'"'"']$/, "");
        print; exit
      }
    ' "${config}")"
    if [[ -n "${found}" ]]; then
      echo "${found}"
      return 0
    fi
  fi
  echo "python3"
}

run_pipeline() {
  local stages="$1"; shift
  local config=""
  local extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config) config="$2"; extra+=("$1" "$2"); shift 2 ;;
      *) extra+=("$1"); shift ;;
    esac
  done
  if [[ -z "${config}" ]]; then
    echo "ERROR: --config /abs/path/run.yaml is required" >&2
    exit 2
  fi
  local python_bin
  python_bin="$(resolve_python "${config}")"
  echo "[gad] stages   : ${stages}"
  echo "[gad] config   : ${config}"
  echo "[gad] python   : ${python_bin}"
  exec "${python_bin}" "${PIPELINE_PY}" --config "${config}" --stages "${stages}" "${extra[@]}"
}
