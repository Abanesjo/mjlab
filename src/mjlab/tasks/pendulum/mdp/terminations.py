"""Task-specific terminations for the Go2 pendulum environment.

Most conditions compose with the general-purpose ``sustained`` wrapper in
``mjlab.envs.mdp.terminations`` to match Isaac Lab's grace-period and
sustained-hold semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def pendulum_fallen(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, angle_rad: float
) -> torch.Tensor:
  """True when the pendulum's joint-space tilt magnitude exceeds ``angle_rad``.

  Computed as L2 norm of the pendulum joint position vector (two hinges in
  our MJCF), matching Isaac Lab's 2-DOF swing convention.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pend_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.linalg.vector_norm(pend_pos, dim=-1) > angle_rad


def position_goal_violation(
  env: ManagerBasedRlEnv, command_name: str, max_dist: float
) -> torch.Tensor:
  """True when the body-frame position error magnitude exceeds ``max_dist``."""
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  dist = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
  return dist > max_dist
