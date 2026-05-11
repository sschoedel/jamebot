#!/usr/bin/env bash
# Play / evaluate the latest Jamebot balance checkpoint.
#
# Pass --num_envs N --checkpoint <path> if you need overrides.
set -e

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/git/IsaacLab}"
PYTHON="${PYTHON:-$HOME/miniforge3/envs/isaaclab/bin/python}"

# play.py also resolves the logs dir relative to cwd.
cd "$ISAACLAB_DIR"

"${PYTHON}" "${ISAACLAB_DIR}/scripts/reinforcement_learning/rsl_rl/play.py" \
    --task Isaac-Jamebot-Balance-Direct-v0 \
    --num_envs 16 \
    "$@"
