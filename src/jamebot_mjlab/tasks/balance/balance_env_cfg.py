"""Jamebot balance task configuration.

Port of ``Isaac-Jamebot-Balance-Direct-v0`` to mjlab's manager-based API.

Asymmetric actor-critic:
- Actor observations are restricted to what the real robot's state estimator
  outputs (9-axis IMU minus magnetometer, joint encoders) and stack 5 history
  steps to support implicit estimation of unobserved quantities (base lin vel,
  push forces) under domain randomization.
- Critic observations add privileged signals (base lin vel, base height) that
  are not available on hardware.

Action interface:
- Ballscrew (linear actuator): position-controlled via PD actuator. Policy
  output in [-1, 1] maps to a position offset around the default joint
  position (use_default_offset=True, scale=0.05 m).
- Two flywheels: velocity-controlled (matches the real motor controllers,
  which run a closed-loop velocity loop on board). Policy output in [-1, 1]
  maps to +/- 200 rad/s commanded velocity; the actuator's P-loop produces
  torque up to its 0.1 N*m effort limit.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  base_ang_vel,
  base_lin_vel,
  bad_orientation,
  flat_orientation_l2,
  is_alive,
  is_terminated,
  joint_pos_rel,
  joint_vel_l2,
  joint_vel_rel,
  last_action,
  projected_gravity,
  push_by_setting_velocity,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from jamebot_mjlab.robots.jamebot import get_jamebot_cfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

##
# Constants.
##

ROBOT = SceneEntityCfg("robot")
BALLSCREW = SceneEntityCfg("robot", joint_names=("ballscrew",))
FLYWHEELS = SceneEntityCfg("robot", joint_names=("flywheelX", "flywheelY"))

# Actor observation history depth. 5 steps at 100 Hz = 50 ms. Lets the policy
# implicitly estimate base linear velocity / push forces from the time
# evolution of gravity_b and gyro under known actions. See research notes:
# single-step MLP fails the observability requirement for asymmetric critic to
# help; 5-step plateau is a reasonable starting point for a small balancer.
ACTOR_HISTORY_LENGTH = 5

# Control rate: 100 Hz policy on top of 400 Hz physics. Matches the IsaacLab
# baseline so reward weights and termination thresholds transfer.
PHYSICS_TIMESTEP = 1.0 / 400.0
DECIMATION = 4

# Tip-over threshold. The IsaacLab task used projected_gravity_b.z > -0.4 as
# the fall trigger; acos(0.4) ~= 66 deg. bad_orientation() in mjlab takes a
# limit angle in radians.
TIPOVER_LIMIT_RAD = math.radians(66.0)


##
# Custom observation: base height (privileged, for critic only).
##


def base_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ROBOT
) -> torch.Tensor:
  """Root link world-frame z. Shape: (num_envs, 1)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


##
# Factory.
##


def _make_env_cfg() -> ManagerBasedRlEnvCfg:
  ##
  # Observations.
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel,
      params={"asset_cfg": ROBOT},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity,
      params={"asset_cfg": ROBOT},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "ballscrew_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": BALLSCREW},
      noise=Unoise(n_min=-0.001, n_max=0.001),
    ),
    "ballscrew_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": BALLSCREW},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "flywheel_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": FLYWHEELS},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "last_action": ObservationTermCfg(func=last_action),
  }

  # Critic gets unnoised versions of every actor term plus privileged signals.
  critic_terms: dict[str, ObservationTermCfg] = {
    name: ObservationTermCfg(func=t.func, params=t.params, noise=None)
    for name, t in actor_terms.items()
  }
  critic_terms["base_lin_vel"] = ObservationTermCfg(
    func=base_lin_vel,
    params={"asset_cfg": ROBOT},
  )
  critic_terms["base_height"] = ObservationTermCfg(
    func=base_height,
    params={"asset_cfg": ROBOT},
  )

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=ACTOR_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions.
  ##

  actions: dict[str, ActionTermCfg] = {
    "ballscrew_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=("ballscrew",),
      scale=0.05,
      use_default_offset=True,
    ),
    "flywheel_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=("flywheelX", "flywheelY"),
      scale=200.0,
      use_default_offset=True,
    ),
  }

  ##
  # Events.
  ##

  events = {
    "reset_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "roll": (-0.02, 0.02),
          "pitch": (-0.02, 0.02),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {},
      },
    ),
    "reset_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=("ballscrew", "flywheelX", "flywheelY")
        ),
      },
    ),
    # Random horizontal pushes via instantaneous velocity perturbations on the
    # floating base. Matches the IsaacLab push protocol (1-3 s interval,
    # ~0.3-1.5 N impulses on a 0.27 kg body -> ~0.01-0.05 m/s velocity change).
    "push_robot": EventTermCfg(
      func=push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
        },
      },
    ),
  }

  ##
  # Rewards.
  ##

  rewards = {
    "alive": RewardTermCfg(func=is_alive, weight=1.0),
    "terminated": RewardTermCfg(func=is_terminated, weight=-2.0),
    "tilt": RewardTermCfg(
      func=flat_orientation_l2,
      weight=-8.0,
      params={"asset_cfg": ROBOT},
    ),
    # Encourage flywheels to spin down at rest. Quadratic on flywheel
    # angular velocity. Weight intentionally small so it doesn't dominate the
    # balance signal early in training.
    "flywheel_vel": RewardTermCfg(
      func=joint_vel_l2,
      weight=-1e-4,
      params={"asset_cfg": FLYWHEELS},
    ),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "tipped": TerminationTermCfg(
      func=bad_orientation,
      params={"limit_angle": TIPOVER_LIMIT_RAD, "asset_cfg": ROBOT},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_jamebot_cfg()},
      num_envs=4096,
      env_spacing=1.5,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="body",
      distance=0.8,
      elevation=-15.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      # The default njmax heuristic overflows at ~72 constraints per world for
      # this robot (freejoint + 3 actuated joints + foot contact + friction
      # losses). 128 gives comfortable headroom.
      njmax=128,
      mujoco=MujocoCfg(timestep=PHYSICS_TIMESTEP),
    ),
    decimation=DECIMATION,
    episode_length_s=5.0,
  )


def jamebot_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg()
  if play:
    # Play mode: very long episodes, clean observations.
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
    cfg.scene.num_envs = 1
    # Pushes are off by default in play mode. The play wrapper sets
    # JAMEBOT_PLAY_PERTURBATIONS=1 when --enable-perturbations is passed.
    if os.environ.get("JAMEBOT_PLAY_PERTURBATIONS", "0") != "1":
      cfg.events.pop("push_robot", None)
  return cfg
