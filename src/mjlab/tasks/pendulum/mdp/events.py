"""Task-specific events for the Go2 pendulum environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def reset_pendulum_angles_magnitude(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  angle_range_deg: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  velocity_range: tuple[float, float] = (0.0, 0.0),
) -> None:
  """Reset pendulum joints with magnitude-in-range, random-sign sampling.

  Isaac Lab's pendulum reset samples ``sign * U(min_deg, max_deg)`` per joint
  per env, which is not expressible as a symmetric uniform when ``min > 0``.
  This helper mirrors that: magnitude is drawn from ``U(min, max)`` and then
  multiplied by a random ``+/- 1``. Writes directly to the pendulum joints
  named by ``asset_cfg``; other joints are untouched.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()

  min_rad = math.radians(float(angle_range_deg[0]))
  max_rad = math.radians(float(angle_range_deg[1]))
  magnitude = sample_uniform(min_rad, max_rad, joint_pos.shape, env.device)
  sign = torch.where(
    torch.rand(joint_pos.shape, device=env.device) < 0.5,
    torch.ones_like(joint_pos),
    -torch.ones_like(joint_pos),
  )
  joint_pos = joint_pos + magnitude * sign
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )
