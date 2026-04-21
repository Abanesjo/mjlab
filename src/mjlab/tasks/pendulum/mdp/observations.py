"""Task-specific observations for the Go2 pendulum environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .gait import clock_inputs_from_phase, foot_phases

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def clock_inputs(
  env: ManagerBasedRlEnv,
  period_s: float = 0.5,
  offsets: tuple[float, ...] = (0.0, 0.5, 0.5, 0.0),
  duty_cycle: float = 0.5,
) -> torch.Tensor:
  """Per-foot sine-wave gait clock inputs.

  Returns warped sine-wave clocks that match the Isaac Lab pendulum gait
  schedule, where stance and swing each occupy half of the sinusoid cycle.
  ``offsets = (0, 0.5, 0.5, 0)`` yields a diagonal trot pattern.
  """
  phase = foot_phases(env, period_s=period_s, offsets=offsets)
  return clock_inputs_from_phase(phase, duty_cycle=duty_cycle)
