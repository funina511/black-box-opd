#!/usr/bin/env bash
# SeqKD baseline stage only (independent of Warmup/GAD).
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_pipeline "seqkd" "$@"
