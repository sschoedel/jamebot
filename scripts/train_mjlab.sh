#!/usr/bin/env bash
# Launcher for mjlab training of the jamebot balance task.
#
# Forwards all arguments to mjlab's train CLI (tyro-based). See:
#   uv run jamebot-train --help
#
# Examples:
#   ./scripts/train_mjlab.sh
#   ./scripts/train_mjlab.sh --env.scene.num-envs 2048
#   ./scripts/train_mjlab.sh --agent.algorithm.learning-rate 5e-4

set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run jamebot-train Mjlab-Jamebot-Balance-v0 "$@"
