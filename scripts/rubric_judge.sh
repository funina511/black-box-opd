#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${GAD_PYTHON:-/mnt/jinbo/miniconda3/envs/gad/bin/python}" "$ROOT/gad/tools/rubric_judge.py" "$@"
