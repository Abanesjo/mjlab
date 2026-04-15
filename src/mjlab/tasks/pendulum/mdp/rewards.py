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
