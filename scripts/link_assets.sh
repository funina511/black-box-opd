#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLRM="$(cd "$ROOT/.." && pwd)"
link_asset() {
  local link="$1" target="$2"
  if [[ ! -e "$target" ]]; then
    echo "missing shared asset: $target" >&2
    return 1
  fi
  mkdir -p "$(dirname "$link")"
  if [[ -L "$link" ]]; then
    [[ "$(realpath "$link")" == "$(realpath "$target")" ]] || {
      echo "link points elsewhere: $link" >&2; return 1;
    }
  elif [[ -e "$link" ]]; then
    echo "asset location already occupied: $link" >&2
    return 1
  else
    ln -s "$target" "$link"
  fi
  echo "$link -> $target"
}
link_asset "$ROOT/assets/models/student" "${GAD_STUDENT_MODEL:-$RLRM/model/Qwen/Qwen3-1.7B}"
link_asset "$ROOT/assets/models/teacher" "${GAD_TEACHER_MODEL:-$RLRM/model/Qwen/Qwen3-30B-A3B-Instruct-2507}"
link_asset "$ROOT/assets/datasets/dapo_math_17k" "${GAD_DAPO_DATASET:-$RLRM/dataset/DAPO-Math-17k}"
if [[ -e "${GAD_MATH_EVAL_DATASET:-$RLRM/dataset/math_eval}" ]]; then
  link_asset "$ROOT/assets/datasets/math_eval" "${GAD_MATH_EVAL_DATASET:-$RLRM/dataset/math_eval}"
fi
