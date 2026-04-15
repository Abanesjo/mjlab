"""Task-specific observations for the Go2 pendulum environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def clock_inputs(
  env: ManagerBasedRlEnv,
  period_s: float = 0.5,
  offsets: tuple[float, ...] = (0.0, 0.5, 0.5, 0.0),
) -> torch.Tensor:
  """Per-foot sine-wave gait clock inputs.

  Returns ``[sin(2 pi (t/period + offset_i)) for each offset]`` stacked along
  the last axis. ``offsets = (0, 0.5, 0.5, 0)`` yields a diagonal trot pattern
  matching Isaac Lab's Go2 pendulum defaults.
  """
  t = env.sim.data.time  # shape (num_envs,)
  phase = t / period_s
  offsets_t = torch.tensor(offsets, device=t.device, dtype=t.dtype)
  # Broadcast: (E, 1) + (F,) -> (E, F)
  arg = (phase.unsqueeze(-1) + offsets_t) * (2.0 * math.pi)
  return torch.sin(arg)
