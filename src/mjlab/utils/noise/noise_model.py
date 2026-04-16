from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import override

if TYPE_CHECKING:
  from mjlab.utils.noise import noise_cfg


class NoiseModel:
  """Base class for noise models."""

  def __init__(
    self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str
  ):
    self._noise_model_cfg = noise_model_cfg
    self._num_envs = num_envs
    self._device = device
    self.noise_scale: float = 1.0

    # Validate configuration.
    if not hasattr(noise_model_cfg, "noise_cfg") or noise_model_cfg.noise_cfg is None:
      raise ValueError("NoiseModelCfg must have a valid noise_cfg")

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset noise model state. Override in subclasses if needed."""

  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise to input data, scaled by :attr:`noise_scale`."""
    if self.noise_scale == 0.0:
      return data
    assert self._noise_model_cfg.noise_cfg is not None
    noisy = self._noise_model_cfg.noise_cfg.apply(data)
    if self.noise_scale != 1.0:
      noisy = data + self.noise_scale * (noisy - data)
    return noisy


class NoiseModelWithAdditiveBias(NoiseModel):
  """Noise model with additional additive bias that is constant for the duration
  of the entire episode."""

  def __init__(
    self,
    noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg,
    num_envs: int,
    device: str,
  ):
    super().__init__(noise_model_cfg, num_envs, device)

    # Validate bias configuration.
    if (
      not hasattr(noise_model_cfg, "bias_noise_cfg")
      or noise_model_cfg.bias_noise_cfg is None
    ):
      raise ValueError("NoiseModelWithAdditiveBiasCfg must have a valid bias_noise_cfg")

    self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
    self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

    # Initialize bias tensor.
    self._bias = torch.zeros((num_envs, 1), device=self._device)
    self._num_components: int | None = None
    self._bias_initialized = False

  @override
  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset bias values for specified environments."""
    indices = slice(None) if env_ids is None else env_ids
    # Sample new bias values.
    self._bias[indices] = self._bias_noise_cfg.apply(self._bias[indices])

  def _initialize_bias_shape(self, data_shape: torch.Size) -> None:
    """Initialize bias tensor shape based on data and configuration."""
    if self._sample_bias_per_component and not self._bias_initialized:
      *_, self._num_components = data_shape
      # Expand bias to match number of components.
      self._bias = self._bias.repeat(1, self._num_components)
      self._bias_initialized = True
      # Resample bias with new shape.
      self.reset()

  @override
  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise and additive bias to input data."""
    self._initialize_bias_shape(data.shape)
    if self.noise_scale == 0.0:
      return data
    noisy_data = super().__call__(data)
    return noisy_data + self.noise_scale * self._bias


class NoiseModelWithBiasDrift(NoiseModelWithAdditiveBias):
  """Additive bias with Brownian drift.

  On reset, samples an initial bias per env (and optionally per component)
  from ``bias_noise_cfg``. On each ``__call__``, adds a Gaussian increment
  ``N(0, (drift_std_per_s * sqrt(step_dt))^2)`` to the bias before applying.

  ``step_dt`` must be provided to the model via :meth:`set_step_dt` after
  construction. The observation manager currently does not do this
  automatically, so tasks that use this model should wire ``step_dt`` at
  manager-init time.
  """

  def __init__(
    self,
    noise_model_cfg: noise_cfg.NoiseModelWithBiasDriftCfg,
    num_envs: int,
    device: str,
  ):
    super().__init__(noise_model_cfg, num_envs, device)
    self._drift_std_per_s = noise_model_cfg.drift_std_per_s
    self._step_dt: float | None = None

  def set_step_dt(self, step_dt: float) -> None:
    """Configure the per-step timestep used to scale Brownian drift."""
    assert step_dt > 0.0
    self._step_dt = step_dt

  @override
  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    self._initialize_bias_shape(data.shape)
    if self._drift_std_per_s > 0.0:
      if self._step_dt is None:
        raise RuntimeError(
          "NoiseModelWithBiasDrift: call set_step_dt(dt) before the first "
          "__call__ so the Brownian increment can be scaled by sqrt(dt)."
        )
      drift_std = self._drift_std_per_s * (self._step_dt**0.5)
      self._bias = self._bias + drift_std * torch.randn_like(self._bias)
    if self.noise_scale == 0.0:
      return data
    noisy_data = NoiseModel.__call__(self, data)
    return noisy_data + self.noise_scale * self._bias
