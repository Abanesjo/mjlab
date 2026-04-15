"""5-stage difficulty curriculum for the Go2 pendulum task.

Mirrors the progression in
``isaaclab_projects/go2_pendulum/.../go2_pendulum_env.py:29-153``: at staged
global-step thresholds, tighten the goal sampling ranges, pendulum termination
angle and sustained-hold duration, and enable/strengthen external pushes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass
class PendulumStage:
  """One difficulty stage. All fields are absolute ranges/thresholds."""

  start_step: int
  dist_range: tuple[float, float]
  bearing_range: tuple[float, float]
  yaw_range: tuple[float, float]
  pendulum_angle_rad: float
  pendulum_duration_s: float
  position_tolerance_m: float
  position_duration_s: float
  push_force_xy_range: tuple[float, float]


# Stage boundaries mirror Isaac Lab's progress thresholds at 0/20/40/60/80%
# of the total curriculum_total_steps. We use the same 2.4M-step default:
# curriculum_total_steps = 75_000 * 32 = 2_400_000.
CURRICULUM_TOTAL_STEPS_DEFAULT = 2_400_000


def default_stages(
  total_steps: int = CURRICULUM_TOTAL_STEPS_DEFAULT,
) -> list[PendulumStage]:
  """Return the 5-stage pendulum curriculum as a concrete list."""
  return [
    # Stage 1: stand + balance in place, generous pendulum tolerance.
    PendulumStage(
      start_step=0,
      dist_range=(0.0, 0.0),
      bearing_range=(0.0, 2.0 * math.pi),
      yaw_range=(0.0, 0.0),
      pendulum_angle_rad=math.radians(60.0),
      pendulum_duration_s=10.0,
      position_tolerance_m=5.0,
      position_duration_s=15.0,
      push_force_xy_range=(0.0, 0.0),
    ),
    # Stage 2: small goals, no pushes.
    PendulumStage(
      start_step=int(0.20 * total_steps),
      dist_range=(0.0, 0.2),
      bearing_range=(0.0, 2.0 * math.pi),
      yaw_range=(-0.5, 0.5),
      pendulum_angle_rad=math.radians(60.0),
      pendulum_duration_s=5.0,
      position_tolerance_m=0.5,
      position_duration_s=15.0,
      push_force_xy_range=(0.0, 0.0),
    ),
    # Stage 3: medium goals, light pushes, tighter pendulum.
    PendulumStage(
      start_step=int(0.40 * total_steps),
      dist_range=(0.1, 0.35),
      bearing_range=(0.0, 2.0 * math.pi),
      yaw_range=(-math.pi / 2, math.pi / 2),
      pendulum_angle_rad=math.radians(45.0),
      pendulum_duration_s=3.0,
      position_tolerance_m=0.3,
      position_duration_s=15.0,
      push_force_xy_range=(-5.0, 5.0),
    ),
    # Stage 4: full goals, full pushes, 30-deg pendulum.
    PendulumStage(
      start_step=int(0.60 * total_steps),
      dist_range=(0.2, 0.5),
      bearing_range=(0.0, 2.0 * math.pi),
      yaw_range=(-math.pi / 2, math.pi / 2),
      pendulum_angle_rad=math.radians(30.0),
      pendulum_duration_s=2.0,
      position_tolerance_m=0.2,
      position_duration_s=15.0,
      push_force_xy_range=(-10.0, 10.0),
    ),
    # Stage 5: tightest — 15-deg pendulum.
    PendulumStage(
      start_step=int(0.80 * total_steps),
      dist_range=(0.3, 0.5),
      bearing_range=(-math.pi / 3, math.pi / 3),
      yaw_range=(-math.pi, math.pi),
      pendulum_angle_rad=math.radians(15.0),
      pendulum_duration_s=5.0,
      position_tolerance_m=0.2,
      position_duration_s=15.0,
      push_force_xy_range=(-10.0, 10.0),
    ),
  ]


def _active_stage(stages: list[PendulumStage], step: int) -> PendulumStage:
  """Find the highest-index stage whose ``start_step`` has been reached."""
  active = stages[0]
  for s in stages:
    if step >= s.start_step:
      active = s
    else:
      break
  return active


def pendulum_difficulty(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  pendulum_termination_name: str,
  position_termination_name: str,
  push_event_name: str | None = None,
  stages: list[PendulumStage] | None = None,
) -> dict[str, torch.Tensor]:
  """Advance the 5-stage curriculum based on ``env.common_step_counter``.

  Mutates the relevant manager-cfg fields in-place so the changes take effect
  the next step without a manager rebuild.

  Args:
    env: The environment.
    env_ids: Unused (curriculum acts globally).
    command_name: Name of the ``PositionGoalCommand`` term.
    pendulum_termination_name: Name of the ``sustained(pendulum_fallen)`` term.
    position_termination_name: Name of the
      ``sustained(position_goal_violation)`` term.
    push_event_name: Optional name of the ``apply_body_impulse`` event whose
      ``force_range`` we re-tune per stage. If ``None``, pushes aren't
      modulated.
    stages: Optional override for the stage list.
  """
  del env_ids
  if stages is None:
    stages = default_stages()
  stage = _active_stage(stages, env.common_step_counter)

  # PositionGoalCommand cfg.
  cmd_cfg = env.command_manager.get_term_cfg(command_name)
  cmd_cfg.dist_range = stage.dist_range  # type: ignore[attr-defined]
  cmd_cfg.bearing_range = stage.bearing_range  # type: ignore[attr-defined]
  cmd_cfg.yaw_range = stage.yaw_range  # type: ignore[attr-defined]

  # Pendulum fallen termination (wrapped in sustained).
  pend_term_cfg = env.termination_manager.get_term_cfg(pendulum_termination_name)
  _set_sustained_knobs(
    pend_term_cfg.params,
    inner_params={"angle_rad": stage.pendulum_angle_rad},
    duration_s=stage.pendulum_duration_s,
  )

  # Position goal violation termination (wrapped in sustained).
  pos_term_cfg = env.termination_manager.get_term_cfg(position_termination_name)
  _set_sustained_knobs(
    pos_term_cfg.params,
    inner_params={"max_dist": stage.position_tolerance_m},
    duration_s=stage.position_duration_s,
  )

  # External wrench push force range.
  if push_event_name is not None:
    push_cfg = env.event_manager.get_term_cfg(push_event_name)
    push_cfg.params["force_range"] = stage.push_force_xy_range

  step_tensor = torch.tensor(float(env.common_step_counter))
  return {
    "stage_start_step": torch.tensor(float(stage.start_step)),
    "global_step": step_tensor,
    "pendulum_angle_rad": torch.tensor(stage.pendulum_angle_rad),
    "position_tolerance_m": torch.tensor(stage.position_tolerance_m),
    "dist_max": torch.tensor(stage.dist_range[1]),
  }


def _set_sustained_knobs(
  params: dict[str, Any],
  inner_params: dict[str, Any],
  duration_s: float,
) -> None:
  """Merge ``inner_params`` into the ``sustained`` term's inner params + set duration."""
  inner = params["inner"]
  merged = dict(inner.get("params", {}))
  merged.update(inner_params)
  inner["params"] = merged
  params["duration_s"] = duration_s
