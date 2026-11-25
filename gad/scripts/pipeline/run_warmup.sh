#!/usr/bin/env bash
# Warmup stage only. Starts from the initial student and trains the discriminator
# plus the actor gate; ends with exportable actor and critic checkpoints.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_pipeline "warmup" "$@"
