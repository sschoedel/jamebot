# jumpybot

This repo holds two independent workflows for a reaction-wheel point-foot
balance robot ("jamebot"):

1. **IsaacLab RL training** from a USD model converted out of the MJCF.
2. **MuJoCo simulation** with an upright LQR balance controller.

Both share the same source MJCF (`robot_model/jamebot_v1.xml`), which is a
hand-edited version of the output from
[`onshape-to-robot`](https://github.com/Rhoban/onshape-to-robot). See
`robot_model/POST_ONSHAPE_EDITS.md` for the list of manual edits that must be
preserved across Onshape re-exports.

---

## 1. IsaacLab RL training

Reaction-wheel point-foot balance task. Direct RL env registered as
`Isaac-Jamebot-Balance-Direct-v0`. Trains with rsl-rl, logs to TensorBoard,
records mp4 rollouts to disk.

### Relevant files

- `convert_mjcf_to_usd.py` — runs IsaacLab's MJCF converter on
  `robot_model/jamebot_v1.xml` and writes `robot_model/usd_isaaclab/jamebot.usd`.
- `robot_model/usd_isaaclab/` — generated USD asset consumed by the IsaacLab env.
- `train_balance.sh` — launcher for `IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`.
- `play_balance.sh` — launcher for the matching `play.py`.
- `sync_videos_to_mac.sh` — rsync helper to pull training rollouts from a
  remote training box to a local Mac.
- `JAMEBOT.md` — extended training notes.

### One-time: convert MJCF to USD

Re-run after editing `robot_model/jamebot_v1.xml`.

```bash
conda activate isaaclab
python convert_mjcf_to_usd.py
```

### Train

```bash
conda activate isaaclab
./train_balance.sh
```

`train_balance.sh` defaults: `--headless --video --video_length 400
--video_interval 500 --logger tensorboard`. Anything else passed forwards to
`train.py`:

```bash
./train_balance.sh --num_envs 2048 --max_iterations 3000 --seed 0
```

Override env paths if needed:

```bash
ISAACLAB_DIR=/path/to/IsaacLab PYTHON=/path/to/python ./train_balance.sh
```

### TensorBoard

```bash
tensorboard --logdir ~/git/IsaacLab/logs/rsl_rl/jamebot_balance
```

Per-run artefacts:

- `logs/rsl_rl/jamebot_balance/<timestamp>/events.out.tfevents.*`
- `logs/rsl_rl/jamebot_balance/<timestamp>/videos/train/rl-video-step-*.mp4`
- `logs/rsl_rl/jamebot_balance/<timestamp>/model_*.pt`
- `logs/rsl_rl/jamebot_balance/<timestamp>/params/{env,agent}.yaml`

### Play / evaluate

```bash
./play_balance.sh                       # latest checkpoint
./play_balance.sh --checkpoint <path>   # specific checkpoint
```

### Pull videos to your Mac

Install the sync script on your Mac (one time):

```bash
scp bedroom-desktop:~/git/jumpybot/sync_videos_to_mac.sh ~/sync_jamebot_videos.sh
chmod +x ~/sync_jamebot_videos.sh
```

Then on the Mac:

```bash
~/sync_jamebot_videos.sh                # pull every run
~/sync_jamebot_videos.sh 2026-05-10*    # only matching runs (glob is remote-side)
```

Each experiment lands in its own folder: `~/jamebot_videos/<timestamp>[_run_name]/`.

---

## 2. MuJoCo sim + LQR controller

A standalone MuJoCo viewer that designs an upright LQR controller around the
current MJCF model and runs it interactively.

### Relevant files

- `main.py` — MuJoCo viewer, LQR design, and interactive controls.
- `debug_joint_ranges.py` — kinematic joint-range inspector (no physics
  stepping).
- `robot_model/scene.xml` — viewer scene; includes `jamebot_v1.xml`.
- `robot_model/jamebot_v1.xml` — hand-edited MJCF used by the sim.
- `robot_model/jamebot.xml` — raw `onshape-to-robot` output; safe to
  overwrite on re-export.
- `robot_model/config.json` — `onshape-to-robot` config.
- `robot_model/assets/` — STL collision/visual meshes.
- `robot_model/POST_ONSHAPE_EDITS.md` — manual MJCF edits to preserve.

### Open the viewer

```sh
uv run python main.py
```

### Inspect ballscrew and spring ranges kinematically

```sh
uv run python debug_joint_ranges.py
```

`debug_joint_ranges.py` fixes the root freejoint, uses `mj_forward()` without
`mj_step()`, and lets the ballscrew and spring joints move through their
compiled MuJoCo ranges. Use Up/Down for the ballscrew, Left/Right for the
spring, Shift for larger steps, `A`/`S` for auto-sweep, and `1`-`6` to jump to
min/center/max.

### Dry-run the LQR design

Print the linearization, weights, and gain matrix without opening a window:

```sh
uv run python main.py --dry-run
```

### Viewer controls

- `R`: reset the robot to an upright pose.
- `B`: toggle the balance controller.
- `G`: show or hide actuator force/position plots.
- `V`: toggle vertical-only root mode; root x/y and orientation are frozen.
- `T`: toggle foot-pinned mode; the foot pivots on a fixed ground point.
- `J`: apply a small perturbation to the robot; an arrow briefly shows the direction.
- `Space`: pause or resume simulation.
- `3`: show or hide collision meshes.
- `Esc`: close the window.

### Useful controller flags

- `--no-balance`: run the sim without calculating or applying LQR control.
- `--control-limit 10`: symmetric actuator command clipping limit.
- `--lqr-clearance -0.001`: upright reference mesh height used for linearization.
- `--no-force-plot`: hide actuator force/position plots at startup.
- `--force-plot-window 5`: seconds of actuator telemetry history to plot.

### Controller notes

The LQR cost ignores absolute base `z` height and base vertical velocity.
Height-related feedback is expressed through ballscrew length and velocity, so
the controller is not tied to an absolute root height.

On reset and during LQR linearization, scalar joints start at the midpoint of
their compiled MuJoCo joint range; unlimited scalar joints such as the
flywheels start at `0`.

Freejoint orientation feedback removes yaw before computing roll/pitch error,
so tilt is measured in the robot heading frame and pure yaw does not affect
balance feedback.
