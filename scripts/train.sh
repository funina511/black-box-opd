#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/gad/scripts/pipeline/run_all.sh" --config "$ROOT/experiments/qwen3_1.7b/train.yaml" "$@"
