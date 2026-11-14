#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${GAD_PYTHON:-python}"
exec "$PYTHON" "$ROOT/rubric/pipeline.py" "$@"
