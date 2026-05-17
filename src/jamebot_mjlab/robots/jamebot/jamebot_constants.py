"""Jamebot robot constants for mjlab.

Pogo-stick balancer: floating base + 2 reaction-wheel flywheels + 1 prismatic
linear actuator (ballscrew) + 1 point-foot. 3 actuators total.

The MJCF (``jamebot_v1_1.xml``) intentionally has no ``<actuator>`` block; all
actuators are added programmatically below so we can tune stiffness, damping,
effort limits, and delay independently of the XML.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF.
##

JAMEBOT_XML: Path = Path(__file__).parent / "xmls" / "jamebot_v1_1.xml"
assert JAMEBOT_XML.exists(), f"Robot MJCF not found at {JAMEBOT_XML}"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(JAMEBOT_XML))


##
# Actuators.
##

# Ballscrew: linear actuator, position-controlled. PD gains chosen for a
# critically damped second-order response of the ballscrew + foot (moving
# mass ~0.055 kg) at ~8 Hz natural frequency. Slightly underdamped so the
# policy can drive transients without fighting heavy back-EMF.
# omega_n = 2*pi*8 ~= 50 rad/s, m ~= 0.055 kg -> k = m*omega_n^2 ~= 138 N/m.
# zeta = 0.8 -> b = 2*zeta*sqrt(k*m) ~= 4.4 N*s/m.
BALLSCREW_STIFFNESS = 140.0
BALLSCREW_DAMPING = 4.5
BALLSCREW_EFFORT_LIMIT = 10.0  # Matches v1 MJCF force limit on the slide.

BALLSCREW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("ballscrew",),
  stiffness=BALLSCREW_STIFFNESS,
  damping=BALLSCREW_DAMPING,
  effort_limit=BALLSCREW_EFFORT_LIMIT,
)

# Flywheels: small reaction wheels, velocity-controlled to match the real
# motor controllers (which run a closed-loop velocity controller on board).
# MuJoCo's <velocity> actuator is a P-controller on velocity:
#   torque = damping * (vel_target - vel_current), clipped at effort_limit.
# Damping kv chosen so a 100 rad/s velocity error saturates the 0.1 N*m
# effort limit -> rotor inertia ~6e-6 kg*m^2 gives a ~6 ms time constant,
# well below the 10 ms control period.
FLYWHEEL_DAMPING = 0.001  # N*m / (rad/s)
FLYWHEEL_EFFORT_LIMIT = 0.1  # N*m, matches the IsaacLab tuned torque scale.

FLYWHEEL_ACTUATOR = BuiltinVelocityActuatorCfg(
  target_names_expr=("flywheelX", "flywheelY"),
  damping=FLYWHEEL_DAMPING,
  effort_limit=FLYWHEEL_EFFORT_LIMIT,
)

##
# Initial state.
##

# foot bottom = body_z + 0.060 m (see jamebot_v1.xml geometry comments). With
# the ballscrew at 0 m extension, setting body_z = 0.06 lands the foot exactly
# on the floor; we add a 2 mm clearance so physics settles cleanly on reset.
HOME_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.062),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "ballscrew": 0.0,
    "flywheelX": 0.0,
    "flywheelY": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collisions.
##

# Single point-foot. condim=3 -> regularized friction cone (no torsional);
# friction tuple of length 1 sets sliding friction only.
FOOT_COLLISION = CollisionCfg(
  geom_names_expr=("foot_collision",),
  condim=3,
  priority=1,
  friction=(1.0,),
)

##
# Articulation.
##

JAMEBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(BALLSCREW_ACTUATOR, FLYWHEEL_ACTUATOR),
  soft_joint_pos_limit_factor=0.95,
)


def get_jamebot_cfg() -> EntityCfg:
  """Return a fresh EntityCfg instance for the jamebot."""
  return EntityCfg(
    spec_fn=get_spec,
    articulation=JAMEBOT_ARTICULATION,
    init_state=HOME_INIT,
    collisions=(FOOT_COLLISION,),
  )
