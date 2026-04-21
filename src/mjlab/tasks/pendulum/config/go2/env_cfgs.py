"""Unitree Go2 pendulum environment configuration."""

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  GO2_ACTION_SCALE,
  get_go2_pendulum_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pendulum.pendulum_env_cfg import make_pendulum_env_cfg

_FEET = ("FR", "FL", "RR", "RL")
_FOOT_GEOM_NAMES = tuple(f"{name}_foot_collision" for name in _FEET)
_FOOT_BODY_NAMES = tuple(f"{name}_foot" for name in _FEET)
_THIGH_BODY_NAMES = ("FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh")


def unitree_go2_pendulum_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 + pendulum configuration."""
  cfg = make_pendulum_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 64

  # Wire the Go2 entity.
  cfg.scene.entities = {"robot": get_go2_pendulum_robot_cfg()}

  # Contact sensors for termination + reward signals.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=_FOOT_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  base_contact_cfg = ContactSensorCfg(
    name="base_contact",
    primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  pendulum_contact_cfg = ContactSensorCfg(
    name="pendulum_contact",
    primary=ContactMatch(mode="body", pattern="pendulum_ee", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  undesired_contact_cfg = ContactSensorCfg(
    name="undesired_ground_contact",
    primary=ContactMatch(mode="body", pattern=_THIGH_BODY_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (
    feet_ground_cfg,
    base_contact_cfg,
    pendulum_contact_cfg,
    undesired_contact_cfg,
  )

  # Wire per-robot names into events + action + viewer.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE
  joint_pos_action.actuator_names = list(GO2_ACTION_SCALE.keys())

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOM_NAMES
  cfg.events["base_mass"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["push_robot"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["feet_clearance"].params["asset_cfg"].body_names = _FOOT_BODY_NAMES

  cfg.viewer.body_name = "base_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("position_goal_violation", None)
    cfg.terminations.pop("base_contact", None)
    cfg.curriculum = {}

  return cfg
