"""Transformer blocks and related modules.

Adapted from:
    - https://github.com/apple/ml-simplefold
"""

import collections.abc
from functools import partial
from itertools import repeat
from typing import Any, Tuple, Type

import torch
from torch import Tensor, nn

from zatom.models.architectures.modules.layers import SwiGLUFeedForward, modulate
from zatom.utils.typing_utils import typecheck

#################################################################################
#                               Misc. Utilities                                 #
#################################################################################


@typecheck
def _ntuple(n):
    """Return a function that converts an input to an n-tuple."""

    def parse(x):
        """Convert an input to an n-tuple of length n."""
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


#################################################################################
#                            Common Layers                                      #
#################################################################################


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks.

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden features. If None, defaults to in_features.
        out_features: Number of output features. If None, defaults to in_features.
        act_layer: Activation layer. Default: GELU.
        norm_layer: Normalization layer. If None, no normalization is applied. Default: None.
        bias: Bias for the linear layers. Can be a single bool or a tuple of two bools,
              specifying bias for each linear layer separately. Default: True.
        drop: Dropout rate. Can be a single float or a tuple of two floats,
              specifying dropout for each linear layer separately. Default: 0.0
        use_conv: Whether to use 1x1 convolutions instead of linear layers. Default: False.
        device: Device for the layers.
        dtype: Data type for the layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0], **dd)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features, **dd) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1], **dd)
        self.drop2 = nn.Dropout(drop_probs[1])

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the MLP.

        Args:
            x: Input tensor of shape (N, *, in_features) if use_conv=False,
               or (N, in_features, H, W) if use_conv=True.

        Returns:
            Output tensor of shape (N, *, out_features) if use_conv=False,
            or (N, out_features, H, W) if use_conv=True.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#################################################################################
#                                 Blocks                                        #
#################################################################################


class DiTBlock(nn.Module):
    """A Diffusion Transformer (DiT) block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Args:
        self_attention_layer: A callable that returns a self-attention layer.
        hidden_size: The hidden size of the transformer block.
        mlp_ratio: The ratio of the MLP hidden size to the hidden size. Default: 4.0
        use_swiglu: Whether to use SwiGLU activation in the MLP. Default: True
    """

    def __init__(
        self,
        self_attention_layer: Type[nn.Module],
        hidden_size: int,
        mlp_ratio: float = 4.0,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following DiT paper."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize transformer layers:
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT encoder blocks
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    @typecheck
    def forward(
        self,
        latents: Tensor,
        c: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Forward pass of the DiT block.

        Args:
            latents: Input tensor of shape (B, N, D).
            c: Conditioning tensor of shape (B, D).
            **kwargs: Additional arguments for the self-attention layer.

        Returns:
            Output tensor of shape (B, N, D).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        _latents = self.attn(modulate(self.norm1(latents), shift_msa, scale_msa), **kwargs)
        latents = latents + gate_msa.unsqueeze(1) * _latents
        latents = latents + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(latents), shift_mlp, scale_mlp)
        )
        return latents


class DiPBlock(nn.Module):
    """A Diffusion Platonic Transformer (DiP) block with adaptive layer norm zero (adaLN-Zero)
    conditioning.

    Args:
        self_attention_layer: A callable that returns a self-attention layer.
        hidden_size: The hidden size of the transformer block.
        mlp_ratio: The ratio of the MLP hidden size to the hidden size. Default: 4.0
        use_swiglu: Whether to use SwiGLU activation in the MLP. Default: True
    """

    def __init__(
        self,
        self_attention_layer: Type[nn.Module],
        hidden_size: int,
        mlp_ratio: float = 4.0,
        use_swiglu: bool = True,
    ):
        raise NotImplementedError("DiPBlock is not yet implemented. Please use DiTBlock instead.")

        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following DiT paper."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize transformer layers:
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT encoder blocks
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    @typecheck
    def forward(
        self,
        latents: Tensor,
        c: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Forward pass of the DiP block.

        Args:
            latents: Input tensor of shape (B, N, D).
            c: Conditioning tensor of shape (B, D).
            **kwargs: Additional arguments for the self-attention layer.

        Returns:
            Output tensor of shape (B, N, D).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        _latents = self.attn(modulate(self.norm1(latents), shift_msa, scale_msa), **kwargs)
        latents = latents + gate_msa.unsqueeze(1) * _latents
        latents = latents + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(latents), shift_mlp, scale_mlp)
        )
        return latents


class HomogenTrunk(nn.Module):
    """A homogeneous trunk for the DiT model.

    Args:
        block: A callable that returns a DiTBlock or DiPBlock.
        depth: The number of blocks in the trunk.

    Returns:
        Output tensor of shape (B, N, D).
    """

    def __init__(self, block: Type[nn.Module], depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([block() for _ in range(depth)])

    @typecheck
    def forward(self, latents: Tensor, c: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass of the homogeneous trunk.

        Args:
            latents: Input tensor of shape (B, N, D).
            c: Conditioning tensor of shape (B, D).
            **kwargs: Additional arguments for the blocks.

        Returns:
            Output tensor of shape (B, N, D).
        """
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)
        return latents
