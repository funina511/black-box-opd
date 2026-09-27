#!/usr/bin/env bash
# One-click GAD pipeline: SeqKD -> Warmup -> GAD (with the Warmup handoff in between).
#
# Usage:
#   bash LMOps/gad/scripts/pipeline/run_all.sh --config /abs/path/run.yaml --dry-run
#   bash LMOps/gad/scripts/pipeline/run_all.sh --config /abs/path/run.yaml
#   bash LMOps/gad/scripts/pipeline/run_all.sh --config /abs/path/run.yaml --from-stage warmup
#   bash LMOps/gad/scripts/pipeline/run_all.sh --config /abs/path/run.yaml --to-stage warmup
#
# Exit codes: 0 success, 1 a stage failed (the failing stage is named in the log),
# 2 configuration problem. The next stage never starts after a failure.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_pipeline "seqkd,warmup,gad" "$@"
