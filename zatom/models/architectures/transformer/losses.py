"""Adapted from https://github.com/carlosinator/tabasco."""

from typing import Callable, Literal, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, nn

from zatom.flow.path import FlowPath
from zatom.utils.metric_utils import split_losses_by_time
from zatom.utils.typing_utils import typecheck


class InterDistancesLoss(nn.Module):
    """Mean-squared error between predicted and reference inter-atomic distance matrices.

    Atomic coordinates are taken from the `key` in the provided `TensorDict` objects.
    Padding masks are taken from the `key_pad_mask` in the provided `TensorDict` objects.
    Only distances between real atoms (i.e. not padded) are considered in the loss.

    Args:
        key: Key that stores coordinates inside `TensorDict` objects.
        key_pad_mask: Key that stores the boolean padding mask inside `TensorDict` objects.
        distance_threshold: If provided, only atom pairs with distance <= threshold
            contribute to the loss. Units must match the coordinate system.
        sqrd: When `True` the raw *squared* distances are used instead of their square-root.
            Set this to `True` if you have pre-squared your training targets.
        loss_weight: Scalar weight applied to the computed loss.
        time_factor: Optional callable `f(t)` that rescales the per-pair loss as a
            function of the interpolation time `t`.
    """

    @typecheck
    def __init__(
        self,
        key: str = "pos",
        key_pad_mask: str = "padding_mask",
        distance_threshold: float | None = None,
        sqrd: bool = False,
        loss_weight: float = 1.0,
        time_factor: Callable | None = None,
    ):
        super().__init__()
        self.key = key
        self.key_pad_mask = key_pad_mask
        self.distance_threshold = distance_threshold
        self.sqrd = sqrd
        self.loss_weight = loss_weight
        self.time_factor = time_factor

        self.mse_loss = nn.MSELoss(reduction="none")

    @typecheck
    def inter_distances(self, coords1: Tensor, coords2: Tensor, eps: float = 1e-6) -> Tensor:
        """Compute pairwise distances between two coordinate sets.

        Args:
            coords1: Coordinate tensor of shape `(N, 3)`.
            coords2: Coordinate tensor of shape `(M, 3)`.
            eps: Numerical stability term added before `sqrt` when `sqrd` is `False`.

        Returns:
            Tensor of shape `(N, M)` containing pairwise distances. Values are squared
            distances when the instance was created with `sqrd=True`.
        """
        if self.sqrd:
            return torch.cdist(coords1, coords2, p=2) ** 2
        else:
            squared_dist = torch.cdist(coords1, coords2, p=2) ** 2
            return torch.sqrt(squared_dist + eps)

    @typecheck
    def forward(
        self,
        path: FlowPath,
        pred: TensorDict,
        compute_stats: bool = True,
        aux_mask: Tensor | None = None,
        eps: float = 1e-6,
    ) -> Tuple[Tensor, dict]:
        """Compute the inter-distance MSE loss.

        Args:
            path: `FlowPath` containing ground-truth endpoint coordinates and the
                interpolation time `t`.
            pred: `TensorDict` with predicted coordinates under the same `key` as
                specified during initialization.
            compute_stats: If `True` additionally returns distance-loss statistics binned
                by time for logging purposes.
            aux_mask: Optional mask tensor to apply to the loss calculation. Should be of shape
                `(batch_size, n_tokens, 1)`.
            eps: Numerical stability term added to denominators to avoid division by zero.

        Returns:
            * loss: Scalar tensor with the mean loss.
            * stats_dict: Dictionary with binned loss statistics. Empty when
                `compute_stats` is `False`.
        """
        real_mask = 1 - path.x_1[self.key_pad_mask].float()
        real_mask = real_mask.unsqueeze(-1)

        assert (
            aux_mask is None or aux_mask.shape == real_mask.shape
        ), f"aux_mask shape: {aux_mask.shape} != real_mask shape: {real_mask.shape}."
        real_mask = real_mask * aux_mask if aux_mask is not None else real_mask

        pred_coords = pred[self.key]
        true_coords = path.x_1[self.key]

        pred_dists = self.inter_distances(pred_coords, pred_coords, eps=eps)
        true_dists = self.inter_distances(true_coords, true_coords, eps=eps)

        mask_2d = torch.matmul(real_mask, real_mask.transpose(-1, -2))

        # Add distance threshold mask (0 for pairs where distance > threshold)
        if self.distance_threshold is not None:
            distance_mask = (true_dists <= self.distance_threshold).float()
            combined_mask = mask_2d * distance_mask
        else:
            combined_mask = mask_2d

        dists_loss = self.mse_loss(pred_dists, true_dists)
        dists_loss = dists_loss * combined_mask

        if self.time_factor:
            dists_loss = dists_loss * self.time_factor(path.t[self.key])

        if compute_stats:
            binned_losses = split_losses_by_time(path.t[self.key], dists_loss, 5)
            stats_dict = {
                **{f"dists_loss_bin_{i}": loss for i, loss in enumerate(binned_losses)},
            }
        else:
            stats_dict = {}

        loss_mask = real_mask.any(-1)
        avg_loss = dists_loss.sum() / (loss_mask.sum() + eps)

        total_loss = avg_loss * self.loss_weight
        return total_loss, stats_dict


@typecheck
def compute_force_loss(
    aux_target_masked: Tensor,
    aux_pred_masked: Tensor,
    num_atoms: int | Tensor,
    eps: float = 1e-6,
    loss_choice: Literal["mse", "mae", "huber"] = "mse",
) -> Tensor:
    """Compute force loss using MSE, MAE, or Huber loss.

    Args:
        aux_target_masked: Target force tensor of shape `(B, N_max, 3)`.
        aux_pred_masked: Predicted force tensor of shape `(B, N_max, 3)`.
        num_atoms: Number of valid atoms in the batch as an integer or tensor of shape `(,)`.
        loss_choice: Choice of loss function - "mse", "mae", or "huber".

    Returns:
        aux_loss_value: Computed force loss as a scalar tensor.
    """
    if loss_choice == "mse":
        per_atom_mse = ((aux_pred_masked - aux_target_masked) ** 2).sum(-1)  # (B, N_max)
        aux_loss_value = per_atom_mse.sum() / (num_atoms + eps)
    elif loss_choice == "mae":
        per_atom_mae = torch.abs(aux_pred_masked - aux_target_masked).sum(-1)  # (B, N_max)
        aux_loss_value = per_atom_mae.sum() / (num_atoms + eps)
    elif loss_choice == "huber":
        per_atom_huber = F.huber_loss(aux_pred_masked, aux_target_masked, reduction="none").sum(
            -1
        )  # (B, N_max)
        aux_loss_value = per_atom_huber.sum() / (num_atoms + eps)
    else:
        raise ValueError(f"Unknown loss choice: {loss_choice}")

    return aux_loss_value
