# jumpybot

Convert the USD model to MJCF:

```sh
uv run python usd_to_mjcf.py simbot.usd simbot.xml
```

Open the current MuJoCo model:

```sh
uv run python main.py
```

Inspect the ballscrew and spring ranges kinematically, with no physics stepping:

```sh
uv run python debug_joint_ranges.py
```

By default `main.py` designs an upright LQR balance controller before opening
the viewer. To print the linearization, weights, and gain matrix without opening
a window:

```sh
uv run python main.py --dry-run
```

Controls:

- `R`: reset the robot to an upright pose.
- `B`: toggle the balance controller.
- `G`: show or hide actuator force/position plots.
- `V`: toggle vertical-only root mode; root x/y and orientation are frozen.
- `T`: toggle foot-pinned mode; the foot pivots on a fixed ground point.
- `Space`: pause or resume simulation.
- `3`: show or hide collision meshes.
- `Esc`: close the window.

`debug_joint_ranges.py` fixes the root freejoint, uses `mj_forward()` without
`mj_step()`, and lets the ballscrew and spring joints move through their compiled
MuJoCo ranges. Use Up/Down for the ballscrew, Left/Right for the spring, Shift
for larger steps, `A`/`S` for auto-sweep, and `1`-`6` to jump to min/center/max.

Useful controller flags:

- `--no-balance`: run the sim without calculating or applying LQR control.
- `--control-limit 10`: symmetric actuator command clipping limit.
- `--lqr-clearance -0.001`: upright reference mesh height used for linearization.
- `--no-force-plot`: hide actuator force/position plots at startup.
- `--force-plot-window 5`: seconds of actuator telemetry history to plot.

The LQR cost ignores absolute base `z` height and base vertical velocity.
Height-related feedback is expressed through ballscrew length and velocity, so
the controller is not tied to an absolute root height.
On reset and during LQR linearization, scalar joints start at the midpoint of
their compiled MuJoCo joint range; unlimited scalar joints such as the flywheels
start at `0`.

Freejoint orientation feedback removes yaw before computing roll/pitch error, so
tilt is measured in the robot heading frame and pure yaw does not affect balance
feedback.

`robot_model/scene.xml` includes `jamebot_v1.xml`. `onshape-to-robot` can keep
regenerating `robot_model/jamebot.xml` without overwriting the hand-edited v1
model.

The converter expects the payload files under `linear/`, matching the current
`simbot.usd` payload path:

```text
linear/simbot_edit.usd
linear/simbot_base.usd
linear/parts/*.usd
linear/Materials/Materials.usd
```

OpenUSD may warn about `OmniPBR.mdl` from the Omniverse materials. That shader
asset is not needed for this MJCF conversion.

## Lightwheel usd2mjcf

`LightwheelAI/usd2mjcf` is not installable with `uv add` because the repository
does not include `pyproject.toml` or `setup.py`. Its runtime dependencies are in
this project environment (`scipy`, `trimesh`, and `coacd`), and it can be tried
from a clone via `PYTHONPATH`.

The default Lightwheel conversion fails on this robot's joint graph. A partial
conversion works only after removing the `spring` and `flywheel_2_01` joint
nodes.
