"""Adapted from https://github.com/carlosinator/tabasco."""

from abc import ABC, abstractmethod

import torch

from zatom.utils.typing_utils import typecheck


class BaseNoiseSchedule(ABC):
    """Interface for time-dependent noise scaling.

    All schedules return zero noise scaling for timesteps above `cutoff`.

    Args:
        cutoff: Timesteps above this value return zero noise.
    """

    @typecheck
    def __init__(self, cutoff: float = 0.9):
        self.cutoff = cutoff

    @typecheck
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Return per-sample noise multipliers for timesteps `t`.

        Args:
            t: A tensor of shape (batch_size,) with values in [0, 1].

        Returns:
            A tensor of shape (batch_size,) with noise scaling factors.
        """
        pass


class SampleNoiseSchedule(BaseNoiseSchedule):
    """
    Inverse schedule: `scale = 1 / (t + eps)`.

    Adds more noise at early timesteps to improve sample quality.

    Args:
        **kwargs: Forwarded to `BaseNoiseSchedule.__init__`.
    """

    @typecheck
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @typecheck
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute inverse noise scaling with small numerical stabilizer.

        Args:
            t: A tensor of shape (batch_size,) with values in [0, 1].

        Returns:
            A tensor of shape (batch_size,) with noise scaling factors.
        """
        raw_scale = 1 / (t + 1e-2)
        return torch.where(t < self.cutoff, raw_scale, torch.zeros_like(t))


class RatioSampleNoiseSchedule(BaseNoiseSchedule):
    """
    Ratio schedule: `scale = (1 - t) / (t + eps)`.

    Adds more noise at early timesteps while decreasing roughly linearly.

    Args:
        **kwargs: Forwarded to `BaseNoiseSchedule.__init__`.
    """

    @typecheck
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @typecheck
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Decrease noise roughly linearly while keeping divergence at early steps.

        Args:
            t: A tensor of shape (batch_size,) with values in [0, 1].

        Returns:
            A tensor of shape (batch_size,) with noise scaling factors.
        """
        raw_scale = (1 - t) / (t + 1e-2)
        return torch.where(t < self.cutoff, raw_scale, torch.zeros_like(t))


class SquareSampleNoiseSchedule(BaseNoiseSchedule):
    """
    Inverse-square schedule: `scale = 1 / (t**2 + eps)`.

    Args:
        **kwargs: Forwarded to `BaseNoiseSchedule.__init__`.
    """

    @typecheck
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @typecheck
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Sharper decay than inverse schedule for very early timesteps.

        Args:
            t: A tensor of shape (batch_size,) with values in [0, 1].

        Returns:
            A tensor of shape (batch_size,) with noise scaling factors.
        """
        raw_scale = 1 / (t**2 + 1e-2)
        return torch.where(t < self.cutoff, raw_scale, torch.zeros_like(t))


class ZeroSampleNoiseSchedule(BaseNoiseSchedule):
    """Schedule that always returns zero (disables additional noise).

    Args:
        **kwargs: Forwarded to `BaseNoiseSchedule.__init__`.
    """

    @typecheck
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @typecheck
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Return a tensor of zeros with same shape as `t`.

        Args:
            t: A tensor of shape (batch_size,) with values in [0, 1].

        Returns:
            A tensor of shape (batch_size,) with noise scaling factors.
        """
        return torch.zeros_like(t)
