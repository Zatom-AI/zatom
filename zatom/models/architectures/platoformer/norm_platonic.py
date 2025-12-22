from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from zatom.models.architectures.platoformer import get_platonic_group
from zatom.utils.typing_utils import typecheck


class NormPlatonic(nn.Module):
    """Equivariant LayerNorm or RMSNorm operation. Acting on Platonic regular representation
    features after reshaping (..., G*C) to (..., G, C)

    The boolean argument normalize_per_g controls whether normalization happens over the last axis
    only (False) or the last two axes (True). Elementwise affine parameters are always shared over
    the group axis -2, which is required for equivariance.

    Args:
        mode: Switch between "LayerNorm" and "RMSNorm" variant.
        solid_name: String identifying the Platonic solid group.
        c: Number of feature channels per group element.
        normalize_per_g: If False, normalizing over the last axis of size c only.
            If True, acting on regular rep indices as well. To preserve equivariance, weights and
            bias parameters are still shared over the regular rep axis!
        eps: A value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: Enables learnable weights and, for LayerNorm, biases parameters.
            Always weight-sharing over group elements to preserve equivariance.
        bias: Can switch off bias when elementwise_affine is True. Only LayerNorm supports biases.
    """

    @typecheck
    def __init__(
        self,
        mode: Literal["LayerNorm", "RMSNorm"],
        solid_name: str,  # in PLATONIC_GROUPS_3D
        c: int,  # per group element
        normalize_per_g: bool,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        assert mode in (
            "LayerNorm",
            "RMSNorm",
        ), f"Unknown mode {mode}, should be 'LayerNorm' or 'RMSNorm'."
        if mode == "RMSNorm":
            assert bias is False, "RMSNorm does not support bias summation."  # Not sure why
        self.mode = mode

        self.G = get_platonic_group(solid_name).G
        self.normalize_per_g = normalize_per_g
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Allocate weight/bias parameters
        self.weight = None
        self.bias = None
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(c, device=device, dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.zeros(c, device=device, dtype=dtype))

        if normalize_per_g:
            self.normalized_shape = (self.G, c)  # Normalize over channels + regular rep axis.
        else:
            self.normalized_shape = (c,)  # Normalize over channels only, *not* regular rep axis.

    @typecheck
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Platonic layer norm operation.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]. The last axis is split into a group
                and channel axis before normalizing, then they are flattened back together.

        Returns:
            Output tensor of shape [batch_size, seq_len, dim], normalized.
        """

        weight = self.weight
        bias = self.bias
        # Expand over group axis
        if self.normalize_per_g and weight is not None:
            weight = weight[None, :].expand(self.G, -1)
        if self.normalize_per_g and bias is not None:
            bias = bias[None, :].expand(self.G, -1)

        x = rearrange(x, "b n (g c) -> b n g c", g=self.G)

        if self.mode == "LayerNorm":
            x = F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        elif self.mode == "RMSNorm":
            x = F.rms_norm(x, self.normalized_shape, weight, self.eps)
        else:
            raise ValueError(f"Unknown mode {self.mode}, should be 'LayerNorm' or 'RMSNorm'.")

        return x.flatten(-2)  # (B, N, G*C)
