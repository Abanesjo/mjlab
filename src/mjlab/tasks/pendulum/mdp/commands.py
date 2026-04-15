"""Position-goal command for the Go2 pendulum task.

Mirrors Isaac Lab's ``Template-Go2-Pendulum-Direct-v0`` goal sampling: each
episode, sample a target XY position + yaw relative to the robot's spawn
pose. The command exposes a body-frame state error
``[dx_b, dy_b, dyaw]`` that the policy observes and that reward / termination
terms consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class PositionGoalCommandCfg(CommandTermCfg):
  """Configuration for :class:`PositionGoalCommand`.

  Samples a target pose (distance d, bearing theta_b, yaw offset psi_d) per
  episode relative to the spawn pose, then yields the body-frame state error
  ``[dx_b, dy_b, dyaw]`` each step.
  """

  entity_name: str
  """Name of the entity in the scene whose goal to track."""

  dist_range: tuple[float, float] = (0.0, 0.0)
  """Uniform range for the sampled goal distance (meters)."""

  bearing_range: tuple[float, float] = (0.0, 2.0 * math.pi)
  """Uniform range for the goal bearing (radians, world-frame from spawn)."""

  yaw_range: tuple[float, float] = (0.0, 0.0)
  """Uniform range for the goal yaw offset (radians, relative to spawn yaw)."""

  def build(self, env: ManagerBasedRlEnv) -> PositionGoalCommand:
    return PositionGoalCommand(self, env)


class PositionGoalCommand(CommandTerm):
  """Position + yaw goal sampled per episode; exposes body-frame state error."""

  cfg: PositionGoalCommandCfg

  def __init__(self, cfg: PositionGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    # World-frame goal.
    self.target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.target_yaw_w = torch.zeros(self.num_envs, device=self.device)

    # Body-frame state error exposed as the policy observation: [dx, dy, dyaw].
    self._state_error = torch.zeros(self.num_envs, 3, device=self.device)

    # Baseline distance sampled at reset; used by the progress reward.
    self.initial_distance = torch.zeros(self.num_envs, device=self.device)

    self.metrics["error_pos_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Body-frame state error ``[dx_b, dy_b, dyaw]`` with yaw wrapped to [-pi, pi]."""
    return self._state_error

  def _update_metrics(self) -> None:
    dx = self._state_error[:, 0]
    dy = self._state_error[:, 1]
    dist = torch.sqrt(dx * dx + dy * dy)
    max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_pos_xy"] += dist / max_step
    self.metrics["error_yaw"] += self._state_error[:, 2].abs() / max_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    d = r.uniform_(*self.cfg.dist_range)
    bearing = torch.empty_like(d).uniform_(*self.cfg.bearing_range)
    yaw_offset = torch.empty_like(d).uniform_(*self.cfg.yaw_range)

    root_pos = self.robot.data.root_link_pos_w[env_ids, :2]
    root_yaw = self.robot.data.heading_w[env_ids]
    self.target_pos_w[env_ids, 0] = root_pos[:, 0] + d * torch.cos(bearing)
    self.target_pos_w[env_ids, 1] = root_pos[:, 1] + d * torch.sin(bearing)
    self.target_yaw_w[env_ids] = wrap_to_pi(root_yaw + yaw_offset)
    self.initial_distance[env_ids] = d

  def _update_command(self) -> None:
    root_pos = self.robot.data.root_link_pos_w[:, :2]
    yaw = self.robot.data.heading_w
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    delta = self.target_pos_w - root_pos
    # World -> body: inverse yaw rotation.
    self._state_error[:, 0] = cos_y * delta[:, 0] + sin_y * delta[:, 1]
    self._state_error[:, 1] = -sin_y * delta[:, 0] + cos_y * delta[:, 1]
    self._state_error[:, 2] = wrap_to_pi(self.target_yaw_w - yaw)
