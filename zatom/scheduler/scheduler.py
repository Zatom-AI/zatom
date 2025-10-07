from typing import Union

import torch
from flow_matching.path.scheduler import (
    CondOTScheduler,
    PolynomialConvexScheduler,
    SchedulerOutput,
)
from torch import Tensor


class EquilibriumCondOTScheduler(CondOTScheduler):
    """Equilibrium CondOT Scheduler."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, t: Tensor) -> SchedulerOutput:
        """Scheduler call."""
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=-torch.ones_like(t),
            d_sigma_t=torch.ones_like(t),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        """Inverse of kappa."""
        return kappa


class EquilibriumPolynomialConvexScheduler(PolynomialConvexScheduler):
    """Equilibrium Polynomial Scheduler."""

    def __init__(self, n: Union[float, int]) -> None:
        super().__init__(n=n)

    def __call__(self, t: Tensor) -> SchedulerOutput:
        """Scheduler call."""
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        """Inverse of kappa."""
        return torch.pow(kappa, 1.0 / self.n)
