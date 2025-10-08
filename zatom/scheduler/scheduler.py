import torch
from flow_matching.path.scheduler import (
    CondOTScheduler,
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
