"""Task-specific rewards for the Go2 pendulum environment.

Faithful port of the reward formulas used by Isaac Lab's
``Template-Go2-Pendulum-Direct-v0`` (go2_pendulum_env.py:1153-1338).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .gait import desired_contact_states, foot_phases, swing_phase_profile

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def position_tracking(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """``exp(-||[dx, dy]|| / std)`` where the command is body-frame position error."""
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  dist = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
  return torch.exp(-dist / std)


def yaw_alignment(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """``exp(-(dyaw^2) / std^2)`` where the command exposes yaw error as element 2."""
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  yaw_err = cmd[:, 2]
  return torch.exp(-torch.square(yaw_err) / (std * std))


def pendulum_upright(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
  """``exp(-sum(pendulum_joint_pos^2) / std)``.

  ``asset_cfg.joint_names`` must select only the pendulum joints.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pend_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  error_sq = torch.sum(torch.square(pend_pos), dim=1)
  return torch.exp(-error_sq / std)


def pendulum_velocity_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """``sum(pendulum_joint_vel^2)``. Apply negative weight to penalize."""
  asset: Entity = env.scene[asset_cfg.name]
  pend_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(pend_vel), dim=1)


def balanced_movement(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """``exp(-pend_err) * ||base_lin_vel_xy||``.

  Rewards horizontal motion only while the pendulum is upright.
  ``asset_cfg.joint_names`` selects the pendulum joints; the robot's body
  velocity is read from the same entity's root data.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pend_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  pend_err = torch.sum(torch.square(pend_pos), dim=1)
  base_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[:, :2], dim=-1)
  return torch.exp(-pend_err) * base_speed


class progress:
  """Reward for making progress toward the goal.

  Returns ``prev_dist - cur_dist``, a small positive signal each step the
  robot moves closer to the goal, zero on reset (no reference to compare
  against), and negative when moving away.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._command_name: str = cfg.params["command_name"]
    self._prev_dist = torch.zeros(env.num_envs, device=env.device)
    self._first_step = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

  def __call__(self, env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    del command_name  # Consumed at init.
    cmd = env.command_manager.get_command(self._command_name)
    assert cmd is not None
    cur_dist = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    delta = self._prev_dist - cur_dist
    # Zero out the first step after reset so progress isn't rewarded for the
    # jump from arbitrary prev_dist to the sampled initial distance.
    delta = torch.where(self._first_step, torch.zeros_like(delta), delta)
    self._prev_dist = cur_dist.clone()
    self._first_step = torch.zeros_like(self._first_step)
    return delta

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._prev_dist.zero_()
      self._first_step[:] = True
    else:
      self._prev_dist[env_ids] = 0.0
      self._first_step[env_ids] = True


def action_over_limit_l2(env: ManagerBasedRlEnv, soft_limit: float) -> torch.Tensor:
  """``sum(max(|action| - soft_limit, 0)^2)``.

  Penalizes raw policy output beyond ``soft_limit`` per component. Complements
  the hard action clip (usually at a smaller magnitude) with a soft signal.
  """
  action = env.action_manager.action
  over = torch.clamp(action.abs() - soft_limit, min=0.0)
  return torch.sum(torch.square(over), dim=1)


def _command_active(
  env: ManagerBasedRlEnv,
  command_name: str | None,
  command_threshold: float,
) -> torch.Tensor:
  """Return a binary gate for locomotion rewards based on goal error magnitude."""
  if command_name is None:
    return torch.ones(env.num_envs, device=env.device)
  command = env.command_manager.get_command(command_name)
  if command is None:
    return torch.ones(env.num_envs, device=env.device)
  pos_error = torch.linalg.vector_norm(command[:, :2], dim=1)
  yaw_error = torch.abs(command[:, 2])
  return ((pos_error + yaw_error) > command_threshold).float()


def feet_clearance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  period_s: float = 0.5,
  offsets: tuple[float, ...] = (0.0, 0.5, 0.5, 0.0),
  swing_height: float = 0.08,
  stance_height: float = 0.02,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize feet deviating from the reference swing-height profile."""
  asset: Entity = env.scene[asset_cfg.name]
  phase = foot_phases(env, period_s=period_s, offsets=offsets)
  desired_contact = desired_contact_states(phase)
  swing_phase = swing_phase_profile(phase)
  target_height = swing_height * swing_phase + stance_height
  foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  cost = torch.sum(torch.square(target_height - foot_height) * (1.0 - desired_contact), dim=1)
  active = _command_active(env, command_name, command_threshold)
  env.extras["log"]["Metrics/feet_clearance_cost_mean"] = torch.mean(cost)
  return cost * active


def tracking_contacts_shaped_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  period_s: float = 0.5,
  offsets: tuple[float, ...] = (0.0, 0.5, 0.5, 0.0),
  force_sigma: float = 100.0,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize contact force on feet that are meant to be in swing."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  phase = foot_phases(env, period_s=period_s, offsets=offsets)
  desired_contact = desired_contact_states(phase)
  foot_forces = torch.linalg.vector_norm(sensor.data.force, dim=-1)
  penalty = -torch.sum(
    (1.0 - desired_contact) * (1.0 - torch.exp(-torch.square(foot_forces) / force_sigma)),
    dim=1,
  ) / foot_forces.shape[1]
  active = _command_active(env, command_name, command_threshold)
  env.extras["log"]["Metrics/swing_force_mean"] = torch.mean(foot_forces * (1.0 - desired_contact))
  return penalty * active


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_air_time: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward appropriately long steps at touchdown."""
  sensor: ContactSensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  last_air_time = sensor.data.last_air_time
  assert last_air_time is not None
  reward = torch.sum((last_air_time - target_air_time) * first_contact.float(), dim=1)
  active = _command_active(env, command_name, command_threshold)
  num_landings = torch.sum(first_contact.float())
  env.extras["log"]["Metrics/last_air_time_mean"] = torch.sum(
    last_air_time * first_contact.float()
  ) / torch.clamp(num_landings, min=1.0)
  return reward * active


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Count non-foot contacts such as thigh-ground collisions."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  is_contact = torch.linalg.vector_norm(sensor.data.force, dim=-1) > threshold
  count = torch.sum(is_contact.float(), dim=1)
  env.extras["log"]["Metrics/undesired_contacts_mean"] = torch.mean(count)
  return count
