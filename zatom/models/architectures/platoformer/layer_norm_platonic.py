from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from zatom.models.architectures.platoformer import PLATONIC_GROUPS_3D
from zatom.utils.typing_utils import typecheck


class LayerNormPlatonic(nn.Module):
    """Equivariant LayerNorm operation. Acting on Platonic regular representation features after
    reshaping (..., hidden_dim) to (..., num_G, hidden_dim//num_G)

    The boolean argument normalize_per_g controls whether normalization happens over the last axis
    only (False) or the last two axes (True). Elementwise affine parameters are always shared over
    the group axis -2, which is required for equivariance.

    Args:
        solid_name: String identifying the Platonic solid group.
        hidden_dim: Hidden dimension of the features, divisible by num_G.
        normalize_per_g: If False, normalizing over the last axis of size hidden_dim//num_G only.
            If True, acting on regular rep indices as well. To preserve equivariance, weights and
            bias parameters are still shared over the regular rep axis!
        eps: A value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: Enables learnable weight and bias parameters, shared over regular rep axis.
        bias: Can switch off bias when elementwise_affine is True.
    """

    @typecheck
    def __init__(
        self,
        solid_name: str,  # in PLATONIC_GROUPS_3D.keys()
        hidden_dim: int,
        normalize_per_g: bool,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if solid_name.lower() not in PLATONIC_GROUPS_3D:
            raise ValueError(
                f"Invalid solid_name '{solid_name}'. Must be one of: {list(PLATONIC_GROUPS_3D.keys())}."
            )
        self.group = PLATONIC_GROUPS_3D[solid_name.lower()]
        self.num_G = self.group.G

        assert (
            hidden_dim % self.num_G == 0
        ), "hidden_dim needs to be divisible by the number of group elements."
        hidden_dim_g = hidden_dim // self.num_G

        self.normalize_per_g = normalize_per_g
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = None
        self.bias = None
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_dim_g, device=device, dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.zeros(hidden_dim_g, device=device, dtype=dtype))

        if normalize_per_g:
            self.normalized_shape = (
                self.num_G,
                hidden_dim_g,
            )  # Normalize over channels + regular rep axis.
        else:
            self.normalized_shape = (
                hidden_dim_g,
            )  # Normalize over channels only, *not* regular rep axis.

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
        if self.normalize_per_g and weight is not None:
            weight = weight.repeat(self.num_G, 1)
        if self.normalize_per_g and bias is not None:
            bias = bias.repeat(self.num_G, 1)

        x = rearrange(x, "b m (g c) -> b m g c", g=self.num_G)
        x = F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        x = x.flatten(-2)
        return x
