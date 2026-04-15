"""Shared gait-phase computation for the Go2 pendulum task.

The gait is stateless: ``env.sim.data.time`` is per-env and resets with each
episode, so every phase-derived quantity can be recomputed from ``time`` and
a few constants without per-env buffers. Both the ``clock_inputs``
observation and the four gait-shaping rewards consume the helper below so
they stay in phase with one another.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def foot_phase(
  env: ManagerBasedRlEnv,
  period_s: float,
  offsets: tuple[float, ...],
) -> torch.Tensor:
  """Per-foot gait phase in ``[0, 1)`` with shape ``[B, F]``.

  ``phase_i(t) = ((t / period_s) + offsets[i]) mod 1``.
  """
  t = env.sim.data.time  # [B]
  offsets_t = torch.as_tensor(offsets, device=t.device, dtype=t.dtype)
  base = t.unsqueeze(-1) / period_s + offsets_t
  return torch.remainder(base, 1.0)


def desired_contact_states(
  phase: torch.Tensor,
  kappa: float = 0.07,
) -> torch.Tensor:
  """Von-Mises-style smoothed stance flag in ``[0, 1]``.

  Mirrors Isaac Lab's ``go2_pendulum_env._step_contact_targets`` smoothing:
  a product of two Gaussian CDFs giving ~1 during stance (phase < 0.5) and
  ~0 during swing, with a smooth transition controlled by ``kappa``.
  """
  normal = torch.distributions.Normal(0.0, kappa)
  cdf = normal.cdf

  p0 = torch.remainder(phase, 1.0)
  p1 = torch.remainder(phase, 1.0) - 1.0
  return cdf(p0) * (1.0 - cdf(p0 - 0.5)) + cdf(p1) * (1.0 - cdf(p1 - 0.5))


def swing_target_height(
  phase: torch.Tensor,
  stance_ratio: float = 0.5,
  peak_height: float = 0.08,
  base_height: float = 0.02,
) -> torch.Tensor:
  """Per-foot target clearance, triangular over the swing half of the cycle.

  Remap raw ``phase`` so ``[0, stance_ratio)`` covers stance (mapped to
  ``[0, 0.5)``) and ``[stance_ratio, 1)`` covers swing (mapped to
  ``[0.5, 1)``). Build a triangular wave that peaks at mid-swing and is 0
  during stance, then scale.
  """
  stance = phase < stance_ratio
  mapped = torch.where(
    stance,
    phase * (0.5 / stance_ratio),
    0.5 + (phase - stance_ratio) * (0.5 / (1.0 - stance_ratio)),
  )
  triangular = 1.0 - torch.abs(1.0 - torch.clamp(mapped * 2.0 - 1.0, 0.0, 1.0) * 2.0)
  return peak_height * triangular + base_height


def clock_signals(
  env: ManagerBasedRlEnv,
  period_s: float,
  offsets: tuple[float, ...],
) -> torch.Tensor:
  """``sin(2 pi * foot_phase)`` per foot, shape ``[B, F]``."""
  return torch.sin(2.0 * math.pi * foot_phase(env, period_s, offsets))
