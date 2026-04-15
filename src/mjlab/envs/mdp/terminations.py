"""Useful methods for MDP terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.sensor.contact_sensor import ContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum."""
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Terminate when the asset's orientation exceeds the limit angle."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle


def root_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the asset's root height is below the minimum height."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < minimum_height


def nan_detection(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate environments that have NaN/Inf values in their physics state."""
  return NanGuard.detect_nans(env.sim.data)


def body_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
) -> torch.Tensor:
  """Terminate when a contact sensor's max contact force exceeds a threshold.

  Reduces per-env over all contact slots via L2-norm then max.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None, (
    f"Sensor '{sensor_name}' must include 'force' in its fields tuple."
  )
  # force: [B, N, 3]  -> L2-norm per contact -> max over contacts -> [B]
  force_mag = torch.linalg.vector_norm(force, dim=-1)
  max_force = force_mag.max(dim=-1).values
  return max_force > threshold


class sustained:
  """Termination wrapper that fires only after a sustained hold window.

  Wraps an inner termination term. Two knobs, both optional:

  - ``duration_s`` — inner condition must hold True for this many seconds of
    consecutive timesteps before the wrapper fires. Default 0 fires the step
    the inner condition first becomes True.
  - ``grace_period_s`` — suppress the wrapper for this many seconds after
    each env reset, even if the inner condition is True. Useful to avoid
    terminating during transient resets (e.g., brief base contact just after
    spawn).

  Params (on the outer :class:`TerminationTermCfg`):

  - ``inner``: ``{"func": callable, "params": dict}`` describing the inner
    condition.
  - ``duration_s``: float (default 0.0).
  - ``grace_period_s``: float (default 0.0).
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
    # Inner func identity is captured at init since it can't change at runtime;
    # all numeric knobs (duration_s, grace_period_s, and inner.params) are
    # re-read from cfg.params on every call so curricula can mutate them.
    inner = cfg.params["inner"]
    self._inner_func = inner["func"]
    self._hold_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    inner: dict[str, Any],
    duration_s: float = 0.0,
    grace_period_s: float = 0.0,
  ) -> torch.Tensor:
    inner_params: dict[str, Any] = inner.get("params", {})
    cond = self._inner_func(env, **inner_params)
    self._hold_count = torch.where(
      cond, self._hold_count + 1, torch.zeros_like(self._hold_count)
    )
    duration_steps = int(round(float(duration_s) / env.step_dt))
    grace_steps = int(round(float(grace_period_s) / env.step_dt))
    past_grace = env.episode_length_buf >= grace_steps
    if duration_steps <= 0:
      return cond & past_grace
    return (self._hold_count >= duration_steps) & past_grace

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._hold_count.zero_()
    else:
      self._hold_count[env_ids] = 0
