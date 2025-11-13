from typing import List

import torch
from torch import Tensor

from zatom.utils.typing_utils import typecheck


@typecheck
def split_losses_by_time(flow_time: Tensor, losses: Tensor, num_bins: int) -> List[float]:
    """Split the losses by time.

    Args:
        time: Time tensor.
        losses: Losses tensor.
        num_bins: Number of bins.

    Returns:
        List of mean losses for each bin.
    """
    flow_time = flow_time.squeeze()

    bins = torch.linspace(0, 1, num_bins + 1, device=flow_time.device)
    bin_assignments = torch.bucketize(flow_time, bins) - 1

    binned_loss_means = []
    for bin_idx in range(num_bins):
        mask = bin_assignments == bin_idx
        bin_losses = losses[mask]
        binned_loss_means.append(bin_losses.mean().item())

    return binned_loss_means
