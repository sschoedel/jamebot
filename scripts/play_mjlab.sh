#!/usr/bin/env bash
# Launcher for mjlab play / evaluation of the jamebot balance task.
#
# Jamebot-specific flag:
#   --enable-perturbations    Keep the push_robot event active during play
#                             (default: pushes disabled in play mode).
#
# Examples:
#   ./scripts/play_mjlab.sh
#   ./scripts/play_mjlab.sh --checkpoint-file path/to/model_500.pt
#   ./scripts/play_mjlab.sh --checkpoint-file path/to/model_500.pt --enable-perturbations

set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run jamebot-play Mjlab-Jamebot-Balance-v0 "$@"
