#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/sample/scripts/run_sample.sh" --config "$ROOT/experiments/qwen3_1.7b/sample.yaml" --python "${GAD_PYTHON:-/mnt/jinbo/miniconda3/envs/gad/bin/python}" "$@"
