#!/usr/bin/env bash
# Train the Jamebot balance policy in IsaacLab with TensorBoard logging
# and periodic video rollouts.
#
# Logs:    ~/git/IsaacLab/logs/rsl_rl/jamebot_balance/<timestamp>/
# Videos:  ~/git/IsaacLab/logs/rsl_rl/jamebot_balance/<timestamp>/videos/train/
# TB:      tensorboard --logdir ~/git/IsaacLab/logs/rsl_rl/jamebot_balance
set -e

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/git/IsaacLab}"
# Call the conda env's python directly. Avoids isaaclab.sh, which calls
# `tabs 4` under `set -e` and dies when terminfo is missing.
PYTHON="${PYTHON:-$HOME/miniforge3/envs/isaaclab/bin/python}"

# train.py writes logs to ./logs/... (cwd-relative), so cd into IsaacLab
# to keep all runs under $ISAACLAB_DIR/logs/rsl_rl/jamebot_balance/.
cd "$ISAACLAB_DIR"

# Defaults: 4096 envs, --video on. video_interval / video_length count
# *vectorized step calls*, not env transitions. With ~32 step calls per
# learning iteration: video_interval=500 ≈ one video every ~15 iterations,
# video_length=400 ≈ 8 s of rollout (one full episode at our dt/decimation).
"${PYTHON}" "${ISAACLAB_DIR}/scripts/reinforcement_learning/rsl_rl/train.py" \
    --task Isaac-Jamebot-Balance-Direct-v0 \
    --headless \
    --video \
    --video_length 400 \
    --video_interval 500 \
    --logger tensorboard \
    "$@"
