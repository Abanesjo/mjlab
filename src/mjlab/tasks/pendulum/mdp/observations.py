"""Task-specific observations for the Go2 pendulum environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.pendulum.mdp.gait import clock_signals

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def clock_inputs(
  env: ManagerBasedRlEnv,
  period_s: float = 0.5,
  offsets: tuple[float, ...] = (0.0, 0.5, 0.5, 0.0),
) -> torch.Tensor:
  """Per-foot sine-wave gait clock inputs.

  Returns ``sin(2 pi (t/period + offset_i))`` stacked along the last axis.
  ``offsets = (0, 0.5, 0.5, 0)`` yields a diagonal trot pattern matching
  Isaac Lab's Go2 pendulum defaults.
  """
  return clock_signals(env, period_s, offsets)
