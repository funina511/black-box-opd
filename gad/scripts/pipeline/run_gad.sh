#!/usr/bin/env bash
# GAD stage only.
#
# Requires a completed Warmup run in the same run directory (or one reachable via
# --set run.output_root/run.run_id). The Warmup handoff is executed automatically:
# the last *complete* Warmup checkpoint is copied into the GAD directory, its
# actor and critic are merged to HF, and the global step S is restored so the GAD
# run stops at S + budget.gad_steps.
#
# Add --resume-gad to restore optimizer/scheduler/RNG state as well. When a
# complete GAD checkpoint already exists in the run directory, the handoff is NOT
# repeated: the recorded handoff (handoff_manifest.json, else stage_gad.json)
# supplies the initial model path and the trainer resumes in place. If the run
# already reached S + K, the trainer is skipped and the existing artifacts are
# verified instead of training one step past the budget.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_pipeline "gad" "$@"
