# Jamebot balance training

Reaction-wheel point-foot balance task. Direct RL env registered as
`Isaac-Jamebot-Balance-Direct-v0`. Trains with rsl-rl, logs to TensorBoard,
records mp4 rollouts to disk.

## One-time: convert MJCF to USD

Re-run after editing `~/git/jamebot/robot_model/jamebot_v1.xml`.

```bash
conda activate isaaclab
cd ~/git/jamebot
python convert_mjcf_to_usd.py
```

## Train

```bash
conda activate isaaclab
cd ~/git/jamebot
./train_balance.sh
```

`train_balance.sh` defaults: `--headless --video --video_length 1500
--video_interval 50000 --logger tensorboard`. Anything else passed forwards
to `IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`, e.g.:

```bash
./train_balance.sh --num_envs 2048 --max_iterations 3000 --seed 0
```

Override env paths if needed:

```bash
ISAACLAB_DIR=/path/to/IsaacLab PYTHON=/path/to/python ./train_balance.sh
```

## TensorBoard

```bash
tensorboard --logdir ~/git/IsaacLab/logs/rsl_rl/jamebot_balance
```

Per-run artefacts:

- `logs/rsl_rl/jamebot_balance/<timestamp>/events.out.tfevents.*`
- `logs/rsl_rl/jamebot_balance/<timestamp>/videos/train/rl-video-step-*.mp4`
- `logs/rsl_rl/jamebot_balance/<timestamp>/model_*.pt`
- `logs/rsl_rl/jamebot_balance/<timestamp>/params/{env,agent}.yaml`

## Play / evaluate

```bash
cd ~/git/jamebot
./play_balance.sh                       # latest checkpoint
./play_balance.sh --checkpoint <path>   # specific checkpoint
```

## Pull videos to your Mac

Install the sync script on your Mac (one time):

```bash
scp bedroom-desktop:~/git/jamebot/sync_videos_to_mac.sh ~/sync_jamebot_videos.sh
chmod +x ~/sync_jamebot_videos.sh
```

Then on the Mac:

```bash
~/sync_jamebot_videos.sh                  # pull every run
~/sync_jamebot_videos.sh 2026-05-10*      # only matching runs (glob is remote-side)
```

Each experiment lands in its own folder: `~/jamebot_videos/<timestamp>[_run_name]/`.

## Why bypass `isaaclab.sh`

`isaaclab.sh` runs `tabs 4` under `set -e`; when terminfo is missing the
script aborts with `terminals database is inaccessible`. The launcher
scripts call the conda env's python directly to dodge this.
