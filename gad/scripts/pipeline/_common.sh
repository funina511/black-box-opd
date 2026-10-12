#!/usr/bin/env bash
# All stage wrappers use the same global/run selection as scripts/train.sh.
set -euo pipefail
GAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

run_pipeline() {
  local stages="$1"; shift
  exec bash "${GAD_DIR}/../scripts/train.sh" --stages "${stages}" "$@"
}
