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
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.tasks.pendulum.mdp.gait import (
  desired_contact_states,
  foot_phase,
  swing_target_height,
)

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


def feet_clearance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  period_s: float,
  offsets: tuple[float, ...],
  peak_height: float = 0.08,
  base_height: float = 0.02,
) -> torch.Tensor:
  """Swing-phase foot clearance penalty.

  ``sum_i (target_z_i - foot_z_i)^2 * (1 - desired_contact_i)``. The target
  traces a triangular wave peaking at ``peak_height + base_height`` at
  mid-swing; stance-phase contributions are masked by the smoothed contact
  flag. Apply with negative weight.
  """
  asset: Entity = env.scene[asset_cfg.name]
  phase = foot_phase(env, period_s, offsets)
  target_z = swing_target_height(
    phase, peak_height=peak_height, base_height=base_height
  )
  foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  contact = desired_contact_states(phase)
  error_sq = torch.square(target_z - foot_z) * (1.0 - contact)
  return torch.sum(error_sq, dim=-1)


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_s: float = 0.1,
) -> torch.Tensor:
  """Per-landing air-time bonus.

  Adds ``(last_air_time_i - threshold_s)`` at each foot's landing frame
  (``compute_first_contact`` mask) and sums across feet.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(env.step_dt)
  last_air = sensor.data.last_air_time
  assert last_air is not None
  per_foot = (last_air - threshold_s) * first_contact.float()
  return torch.sum(per_foot, dim=-1)


def tracking_contacts_shaped_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  period_s: float,
  offsets: tuple[float, ...],
  sigma: float = 100.0,
) -> torch.Tensor:
  """Penalize foot forces during swing (desired_contact_i ~ 0).

  For each foot: ``-(1 - desired_contact_i) * (1 - exp(-f_i^2 / sigma^2))``,
  averaged over 4 feet. Apply with positive weight (value is already
  non-positive).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  force_mag = torch.linalg.vector_norm(force, dim=-1)  # [B, F]
  phase = foot_phase(env, period_s, offsets)
  contact = desired_contact_states(phase)
  swing_force_cost = (1.0 - contact) * (1.0 - torch.exp(-(force_mag**2) / (sigma**2)))
  return -torch.mean(swing_force_cost, dim=-1)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Count of slots whose max-history force exceeds ``threshold``.

  Reads ``force_history`` ``[B, N, H, 3]``, takes L2 per substep, then max
  over history. Apply with negative weight to penalize unwanted contacts
  (e.g., thighs scraping the ground).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force_history = sensor.data.force_history
  assert force_history is not None, (
    f"Sensor '{sensor_name}' must set history_length >= 1 "
    "and include 'force' in its fields tuple."
  )
  force_mag = torch.linalg.vector_norm(force_history, dim=-1)  # [B, N, H]
  max_over_history = force_mag.max(dim=-1).values  # [B, N]
  is_contact = max_over_history > threshold
  return torch.sum(is_contact.float(), dim=-1)
