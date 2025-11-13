"""Adapted from https://github.com/carlosinator/tabasco."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from zatom.utils.typing_utils import typecheck


class BaseTimeFactor(nn.Module, ABC):
    """Return a scalar weight per sample based on the interpolation time.

    Args:
        max_value: Upper clamp for the time-factor value.
        min_value: Lower clamp for the time-factor value.
        zero_before: If *t* ≤ this threshold, the factor is forced to zero.
        eps: Small constant to avoid division by zero.
    """

    @typecheck
    def __init__(
        self,
        max_value: float,
        min_value: float = 0.05,
        zero_before: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_value = max_value
        self.min_value = min_value
        self.zero_before = zero_before
        self.eps = eps

    @typecheck
    @abstractmethod
    def forward(self, t: Tensor) -> Tensor:
        """Return the time-factor for each element in `t`.

        Args:
            t: A tensor of shape `(batch_size,)` with interpolation times in [0, 1].

        Returns:
            A tensor of shape `(batch_size,)` with the time-factors.
        """
        pass


class InverseTimeFactor(BaseTimeFactor):
    """
    Weight ~ 1 / (1 - t)^2 as used in the Proteina paper.

    Args:
        max_value: Upper clamp for the time-factor value.
        min_value: Lower clamp for the time-factor value.
        zero_before: If *t* ≤ this threshold, the factor is forced to zero.
        eps: Small constant to avoid division by zero.
    """

    @typecheck
    def __init__(
        self,
        max_value: float = 100.0,
        min_value: float = 0.05,
        zero_before: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(max_value, min_value, zero_before, eps)

    @typecheck
    def forward(self, t: Tensor) -> Tensor:
        """
        Return `1 / (1 - t)^2` clamped to [`min_value`, `max_value`].

        Args:
            t: A tensor of shape `(batch_size,)` with interpolation times in [0, 1].

        Returns:
            A tensor of shape `(batch_size,)` with the time-factors.
        """
        norm_scale = 1 / ((1 - t + self.eps) ** 2)
        norm_scale = torch.clamp(norm_scale, min=self.min_value, max=self.max_value)
        return norm_scale * (t > self.zero_before)


class SignalToNoiseTimeFactor(BaseTimeFactor):
    """
    Weight ~ t / (1 - t) (signal-to-noise ratio).

    Args:
        max_value: Upper clamp for the time-factor value.
        min_value: Lower clamp for the time-factor value.
        zero_before: If *t* ≤ this threshold, the factor is forced to zero.
        eps: Small constant to avoid division by zero.

    Returns:
        A tensor of shape `(batch_size,)` with the time-factors.
    """

    @typecheck
    def __init__(
        self,
        max_value: float = 1.5,
        min_value: float = 0.05,
        zero_before: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(max_value, min_value, zero_before, eps)

    @typecheck
    def forward(self, t: Tensor) -> Tensor:
        """
        Return `t / (1 - t)` clamped to [`min_value`, `max_value`].

        Args:
            t: A tensor of shape `(batch_size,)` with interpolation times in [0, 1].

        Returns:
            A tensor of shape `(batch_size,)` with the time-factors.
        """
        clamped = torch.clamp(t / (1 - t + self.eps), min=self.min_value, max=self.max_value)
        return clamped * (t > self.zero_before)


class SquaredSignalToNoiseTimeFactor(BaseTimeFactor):
    """
    Weight ~ (t / (1 - t))^2 (squared SNR).

    Args:
        max_value: Upper clamp for the time-factor value.
        min_value: Lower clamp for the time-factor value.
        zero_before: If *t* ≤ this threshold, the factor is forced to zero.
        eps: Small constant to avoid division by zero.
    """

    @typecheck
    def __init__(
        self,
        max_value: float = 1.5,
        min_value: float = 0.05,
        zero_before: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(max_value, min_value, zero_before, eps)

    @typecheck
    def forward(self, t: Tensor) -> Tensor:
        """
        Return `(t / (1 - t))^2` clamped to [`min_value`, `max_value`].

        Args:
            t: A tensor of shape `(batch_size,)` with interpolation times in [0, 1].

        Returns:
            A tensor of shape `(batch_size,)` with the time-factors.
        """
        clamped = torch.clamp(t / (1 - t + self.eps) ** 2, min=self.min_value, max=self.max_value)
        return clamped * (t > self.zero_before)


class SquareTimeFactor(BaseTimeFactor):
    """Weight ~ 1 / (t^2 + 1e-2)."""

    @typecheck
    def forward(self, t: Tensor) -> Tensor:
        """Return `1 / (t^2 + 1e-2)`.

        Args:
            t: A tensor of shape `(batch_size,)` with interpolation times in [0, 1].

        Returns:
            A tensor of shape `(batch_size,)` with the time-factors.
        """
        raw_scale = 1 / (t**2 + 1e-2)
        return torch.where(t < self.zero_before, raw_scale, torch.zeros_like(t))


class ZeroTimeFactor(BaseTimeFactor):
    """Weight = 0 for all t."""

    @typecheck
    def forward(self, t: Tensor) -> Tensor:
        """Return zeros for all t.

        Args:
            t: A tensor of shape `(batch_size,)` with interpolation times in [0, 1].

        Returns:
            A tensor of shape `(batch_size,)` with the time-factors.
        """
        return torch.zeros_like(t)
