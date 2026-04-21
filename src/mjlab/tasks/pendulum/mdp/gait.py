"""Shared gait-phase helpers for pendulum locomotion shaping."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def foot_phases(
  env: ManagerBasedRlEnv,
  period_s: float = 0.5,
  offsets: tuple[float, ...] = (0.0, 0.5, 0.5, 0.0),
) -> torch.Tensor:
  """Return per-foot gait phases in ``[0, 1)``."""
  t = env.sim.data.time
  base_phase = t / period_s
  offsets_t = torch.tensor(offsets, device=t.device, dtype=t.dtype)
  return torch.remainder(base_phase.unsqueeze(-1) + offsets_t, 1.0)


def remap_cycle_phase(phase: torch.Tensor, duty_cycle: float = 0.5) -> torch.Tensor:
  """Warp phase so stance and swing each span half of the cycle."""
  duty = torch.full_like(phase, duty_cycle)
  remapped = phase.clone()
  stance_mask = remapped < duty
  swing_mask = remapped > duty
  remapped[stance_mask] = remapped[stance_mask] * (0.5 / duty[stance_mask])
  remapped[swing_mask] = 0.5 + (
    (remapped[swing_mask] - duty[swing_mask]) * (0.5 / (1.0 - duty[swing_mask]))
  )
  return remapped


def clock_inputs_from_phase(phase: torch.Tensor, duty_cycle: float = 0.5) -> torch.Tensor:
  """Return warped sine-wave clock inputs matching the Isaac Lab reference gait."""
  remapped = remap_cycle_phase(phase, duty_cycle=duty_cycle)
  return torch.sin(2.0 * math.pi * remapped)


def desired_contact_states(phase: torch.Tensor, smoothing_kappa: float = 0.07) -> torch.Tensor:
  """Return smooth desired stance probabilities for each foot."""
  normal = torch.distributions.normal.Normal(
    torch.tensor(0.0, device=phase.device, dtype=phase.dtype),
    torch.tensor(smoothing_kappa, device=phase.device, dtype=phase.dtype),
  )
  phase_wrapped = torch.remainder(phase, 1.0)
  return (
    normal.cdf(phase_wrapped) * (1.0 - normal.cdf(phase_wrapped - 0.5))
    + normal.cdf(phase_wrapped - 1.0) * (1.0 - normal.cdf(phase_wrapped - 1.5))
  )


def swing_phase_profile(phase: torch.Tensor) -> torch.Tensor:
  """Return a triangular swing-height profile in ``[0, 1]`` for each foot."""
  return 1.0 - torch.abs(
    1.0 - torch.clamp((phase * 2.0) - 1.0, min=0.0, max=1.0) * 2.0
  )
