import torch
from torch import Tensor

from zatom.utils.typing_utils import typecheck


@typecheck
def apply_mask(x: Tensor, padding_mask: Tensor) -> Tensor:
    """Zero out padded atoms.

    Args:
        x: Tensor `(*, N, D)`.
        padding_mask: Bool/int tensor `(*, N)`.

    Returns:
        Masked tensor.

    NOTE: Based on the NVIDIA-Digital-Bio/proteina function of same name.
    """
    real_mask = (1 - padding_mask.int())[..., None]  # [*, N, 1]
    return x * real_mask


@typecheck
def mean_w_mask(x: Tensor, padding_mask: Tensor) -> Tensor:
    """Compute masked mean along the atom dimension.

    Padded atoms (`padding_mask == 1`) are ignored. The result keeps a broadcastable
    shape `(*, 1, D)` and padded rows are zeroed for numerical safety.

    Args:
        x: Tensor `(*, N, D)` of coordinates.
        padding_mask: Bool/int tensor `(*, N)`.

    Returns:
        Mean coordinates with same dtype/device as `x`.

    NOTE: Based on the NVIDIA-Digital-Bio/proteina function of same name.
    """
    real_mask = (1 - padding_mask.int())[..., None]  # [*, N, 1]
    padding_mask = padding_mask[..., None]  # [*, N, 1]

    num_elements = real_mask.sum(dim=-2, keepdim=True)  # [*, 1, 1]
    num_elements = torch.where(num_elements == 0, torch.tensor(1.0), num_elements)

    x_masked = torch.masked_fill(x, padding_mask, 0.0)

    mean = torch.sum(x_masked, dim=-2, keepdim=True) / num_elements  # [*, 1, D]
    mean = torch.masked_fill(mean, padding_mask, 0.0)
    return mean


@typecheck
def mask_and_zero_com(x: Tensor, padding_mask: Tensor) -> Tensor:
    """Center coordinates and zero padded atoms.

    Args:
        x: Tensor `(*, N, D)`.
        padding_mask: Bool/int tensor `(*, N)`.

    Returns:
        Centered and masked coordinates.

    NOTE: Based on the NVIDIA-Digital-Bio/proteina function of same name.
    """
    real_mask = (1 - padding_mask.int())[..., None]  # [*, N, 1]

    x = apply_mask(x, padding_mask)  # [*, N, D]
    mean = mean_w_mask(x, padding_mask)  # [*, 1, D]

    centered_x = (x - mean) * real_mask
    return centered_x
