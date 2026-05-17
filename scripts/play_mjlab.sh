#!/usr/bin/env bash
# Launcher for mjlab play / evaluation of the jamebot balance task.
#
# Jamebot-specific flags:
#   --enable-perturbations    Keep the push_robot event active during play
#                             (default: pushes disabled in play mode).
#   --quantize-int8           Also build an INT8 (CPU) copy of the actor and
#                             add a 'Use INT8 actor' toggle to the viser GUI.
#                             Lets you switch between fp32 and INT8 live as
#                             a deployment smoke test (target: 96 KB MCU).
#
# Examples:
#   ./scripts/play_mjlab.sh
#   ./scripts/play_mjlab.sh --checkpoint-file path/to/model_500.pt
#   ./scripts/play_mjlab.sh --checkpoint-file path/to/model_500.pt --enable-perturbations
#   ./scripts/play_mjlab.sh --checkpoint-file path/to/model_500.pt --quantize-int8

set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run jamebot-play Mjlab-Jamebot-Balance-v0 "$@"
