# Post-Onshape MJCF Edits

This directory is generated from Onshape with `onshape-to-robot`, but
`jamebot_v1.xml` is the hand-edited simulation model.

Keep `robot_model/config.json` writing to `jamebot.xml`. After each new Onshape
export, copy or merge the changes into `jamebot_v1.xml` intentionally so the
manual edits below are not lost.

## Current Files

- `jamebot.xml`: latest direct output from `onshape-to-robot`; safe to overwrite.
- `jamebot_v1.xml`: working MJCF model with manual simulation edits.
- `scene.xml`: viewer scene; includes `jamebot_v1.xml`.

## Manual Edits

### Preserve hand-edited model

`scene.xml` includes `jamebot_v1.xml` instead of `jamebot.xml`:

```xml
<include file="jamebot_v1.xml" />
```

This lets future Onshape exports overwrite `jamebot.xml` without replacing the
edited simulation model.

### Torque control

The generated position actuators were replaced with motor actuators:

```xml
<motor class="jamebot" name="flywheelY" joint="flywheelY" ctrlrange="-10 10" forcerange="-10 10"/>
<motor class="jamebot" name="flywheelX" joint="flywheelX" ctrlrange="-10 10" forcerange="-10 10"/>
<motor class="jamebot" name="ballscrew" joint="ballscrew" ctrlrange="-10 10" forcerange="-10 10"/>
```

This avoids position setpoint spring-back. Hinge motors apply torque; slide
motors apply linear force.

The current actuator limits are placeholders that match the simulation
controller's default `--control-limit 10`:

- `flywheelY`: +/-10 N*m torque command and actuator-force clamp.
- `flywheelX`: +/-10 N*m torque command and actuator-force clamp.
- `ballscrew`: +/-10 N linear force command and actuator-force clamp.

Replace these with measured motor/driver limits when available.

`config.json` also sets the default Onshape export actuator type to `motor`:

```json
"joint_properties": {
  "default": {
    "actuated": true,
    "type": "motor"
  }
}
```

### Passive spring joint

The `spring` slide joint is passive, not motor-controlled. Its actuator was
removed, and the joint has spring/damper properties:

```xml
<joint axis="0 0 1" name="spring" type="slide"
       range="-0.0155623788846242 0.0044376211153758"
       damping="1" stiffness="100" springref="0"/>
```

The current values are placeholders:

- `stiffness="100"`: spring rate in N/m.
- `damping="1"`: damping in N*s/m.
- `springref="0"`: rest position in meters.

Tune these against the real spring.

`config.json` also marks this joint as unactuated and records the spring
parameters that `onshape-to-robot` can preserve:

```json
"spring": {
  "actuated": false,
  "stiffness": 100,
  "damping": 1
}
```

`springref` is not emitted by `onshape-to-robot` 1.8.2, so keep it as a manual
edit in `jamebot_v1.xml`.

### Unlimited flywheel rotation

The generated range limits were removed from both flywheel hinge joints:

```xml
<joint axis="0 0 1" name="flywheelY" type="hinge"/>
<joint axis="0 0 1" name="flywheelX" type="hinge"/>
```

This makes both flywheels unlimited in MuJoCo while keeping the ballscrew and
spring travel limits.

## Validation

Compile the current scene without opening the viewer:

```sh
uv run python - <<'PY'
import mujoco
m = mujoco.MjModel.from_xml_path("robot_model/scene.xml")
print(f"nbody={m.nbody} ngeom={m.ngeom} njnt={m.njnt} nu={m.nu}")
print([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
PY
```

Expected actuator list:

```text
['flywheelY', 'flywheelX', 'ballscrew']
```

Open the viewer:

```sh
uv run python main.py
```
