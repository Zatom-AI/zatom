import torch
from torch import Tensor

from zatom.utils.typing_utils import typecheck


@typecheck
def get_sample_schedule(schedule: str, num_steps: int) -> Tensor:
    """Return monotonically increasing schedule `T` in `[0,1]`.

    Based on approach in Proteina.

    Args:
        schedule: Type of schedule to use. Must be one of (`linear`, `power`, `log`).
        num_steps: Number of steps in the schedule.

    Returns:
        A tensor representing the monotonically increasing schedule.
    """
    eff_num_steps = num_steps + 1

    if schedule == "linear":
        T = torch.linspace(0, 1, eff_num_steps)

    elif schedule == "power":
        T = torch.linspace(0, 1, eff_num_steps)
        T = T**2

    elif schedule == "log":
        T = 1.0 - torch.logspace(-2, 0, eff_num_steps).flip(0)
        T = T - torch.amin(T)
        T = T / torch.amax(T)

    else:
        raise ValueError(
            f"Invalid sample schedule: {schedule}. Must be one of (`linear`, `power`, `log`)."
        )

    return T
