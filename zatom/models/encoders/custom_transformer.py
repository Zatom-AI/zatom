"""
Global ein notation:

b   - Batch size
m   - Token sequence length
c   - Number of latent embedding channels
"""

import collections.abc
from itertools import repeat
from typing import Callable, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)
from torch.utils.checkpoint import checkpoint

from zatom.utils.training_utils import get_widest_dtype
from zatom.utils.typing_utils import Bool, Float, Int, typecheck

# Initialization
torch._dynamo.config.cache_size_limit = 1000
flex_attention = torch.compile(flex_attention, dynamic=False)


# Constants
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0

SDPA_BACKENDS = [
    SDPBackend.ERROR,
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]


# Helper functions


@typecheck
def build_attention_mask(
    mask: Bool["b m"], seq_idx: Int["b m"], dtype: str | torch.dtype  # type: ignore
) -> Bool["b 1 m m"] | Float["b 1 m m"]:  # type: ignore
    """Build an attention mask for an input batch.

    Args:
        mask: A tensor containing the non-padding mask.
        seq_idx: A tensor containing unique token sequence IDs.
        dtype: The data type of the attention mask.

    Returns:
        A boolean mask tensor or a floating-point tensor representing
        the additive mask for pairwise attention scores.
    """
    non_padding_mask = mask
    attn_mask = non_padding_mask.unsqueeze(1) & non_padding_mask.unsqueeze(2)
    attn_mask = attn_mask & (
        # Only attend to tokens with the same sequence (i.e., example)
        seq_idx.unsqueeze(1)
        == seq_idx.unsqueeze(2)
    )
    attn_mask = attn_mask.unsqueeze(1).type(dtype)
    if dtype == torch.bool:
        return attn_mask
    attn_mask.masked_fill_(attn_mask == 0, float("-inf"))
    attn_mask.masked_fill_(attn_mask == 1, 0.0)
    return attn_mask


@typecheck
def safe_softmax(scores: torch.Tensor, mask: torch.Tensor = None, dim: int = -1) -> torch.Tensor:
    """Safe softmax that avoids NaNs when a row is fully masked.

    Args:
        scores: Attention logits (float tensor).
        mask: Floating-point mask to be added to scores.
              Usually 0 for allowed positions, -inf or large negative for masked.
              Shape must be broadcastable to scores.
        dim: Dimension over which to apply softmax.

    Returns:
        probs: Softmax probabilities with masked positions set to 0.
    """
    if mask is not None:
        scores = scores + mask

        # Detect fully masked rows (all entries == -inf)
        fully_masked = torch.isinf(mask) & (mask < 0)
        fully_masked = fully_masked.all(dim=dim, keepdim=True)

        # For fully masked rows, replace scores with zeros before softmax
        # This ensures exp(0) = 1 and denominator != 0
        scores = torch.where(fully_masked, torch.zeros_like(scores), scores)

    probs = torch.softmax(scores, dim=dim)

    if mask is not None:
        # Zero out masked positions in probabilities
        probs = torch.where(torch.isinf(mask) & (mask < 0), torch.zeros_like(probs), probs)

    return probs


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


# Modules
@typecheck
class MPFourier(nn.Module):
    """Magnitude-preserving Fourier features.

    Adapted from https://github.com/NVlabs/edm2.

    Args:
        num_channels: The number of channels for the Fourier
            features.
        bandwidth: The bandwidth for the frequency distribution.
            Defaults to 1. This controls the spread of the frequencies.
    """

    def __init__(self, num_channels: int, bandwidth: int = 1):
        super().__init__()
        self.num_channels = num_channels

        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    @typecheck
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for computing magnitude-preserving Fourier features.

        Args:
            x: Input scalar tensor of shape [..., 1].

        Returns:
            Output tensor of shape [..., num_channels] containing
            the Fourier features.
        """
        shape = x.shape
        y = x.to(torch.float32).reshape(-1)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype).reshape(*shape, self.num_channels)


@typecheck
class MPFourierEmbedding(nn.Module):
    """Magnitude-preserving Fourier feature embedding for variable-dimensional scalar inputs. Each
    scalar dimension is independently mapped to `width` Fourier channels using magnitude-preserving
    Fourier features.

    Args:
        num_channels: Number of Fourier channels per input dimension.
        bandwidth: Bandwidth `b` controlling frequency spread.
        input_dim: Number of scalar input dimensions (e.g., 1 for timestep, 3 for 3D coords).
    """

    def __init__(self, num_channels: int, bandwidth: int = 1, input_dim: int = 1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim * num_channels

        self.embedders = nn.ModuleList(
            [MPFourier(num_channels, bandwidth=bandwidth) for _ in range(input_dim)]
        )

    @typecheck
    def forward(
        self, x: Float["b m {self.input_dim}"] | Float["m {self.input_dim}"]  # type: ignore
    ) -> Float["b m {self.output_dim}"] | Float["m {self.output_dim}"]:  # type: ignore
        """Forward pass for the magnitude-preserving Fourier embedding.

        Args:
            x: Input tensor of shape [..., m, input_dim], where `m` is the sequence length
                or number of scalar inputs, and `input_dim` is the number of scalar dimensions.

        Returns:
            Output tensor of shape [..., m, output_dim], where `output_dim` is
                `input_dim * width`, containing the Fourier features for each input dimension.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input with last dimension {self.input_dim}, got {x.shape[-1]}"
            )

        components = x.unbind(dim=-1)  # List of `input_dim` tensors of shape [...,]
        features = [embed(c) for embed, c in zip(self.embedders, components)]  # Each [..., width]
        return torch.cat(features, dim=-1)  # [..., output_dim]


@typecheck
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply `bias=False`.

    Adapted from https://github.com/karpathy/nanoGPT.

    Args:
        ndim: Number of dimensions to normalize over.
        bias: Whether to use a bias term in the normalization.
    """

    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    @typecheck
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for LayerNorm.

        Args:
            input: Input tensor to normalize.

        Returns:
            Normalized tensor.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)


@typecheck
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer, and related networks.

    Adapted from
    https://github.com/huggingface/pytorch-image-models.

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden features. If None, uses
            in_features.
        out_features: Number of output features. If None, uses
            in_features.
        act_layer: Activation layer to use.
        norm_layer: Normalization layer to use. If None, no
            normalization is applied.
        bias: Whether to use bias in the linear layers.
        drop: Dropout probability for the first and second linear
        layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Type[nn.Module] | Callable = nn.GELU,
        norm_layer: Type[nn.Module] | Callable | None = None,
        bias: bool = False,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    @typecheck
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim].

        Returns:
            Output tensor of the same shape as input.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


@typecheck
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verification)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position up to
    ``max_seq_len`` by computing this during init.

    Adapted from https://github.com/pytorch/torchtune.

    Args:
        dim: Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``.
        max_seq_len: Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed.
        base: The base for the geometric progression used to compute
            the rotation angles.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        self.rope_init()

    def rope_init(self):
        """Initialize the RoPE parameters."""
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    @typecheck
    def build_rope_cache(self, max_seq_len: int = 2048):
        """Build the RoPE cache for the given maximum sequence length.

        Args:
            max_seq_len: Maximum sequence length for which to build
                the cache. If the sequence length exceeds this, the cache
                will be recomputed.
        """
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

        # Perform outer product of theta and position index;
        # Output tensor has a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # Cache both the cos and sin components,
        # so the output shape is [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    @typecheck
    def forward(
        self, x: Float["b m h c"], *, input_pos: Int["b m"] | None = None  # type: ignore
    ) -> Float["b m h c"]:  # type: ignore
        """Forward pass for Rotary Positional Embeddings.

        Notation used for tensor shapes:
            - b: batch size
            - m: sequence length
            - h: num heads
            - c: head dim

        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads,
                head_dim].
            input_pos: Optional tensor of shape [batch_size, seq_len]
                containing the positions of the input tokens. If not provided,
                the cache will be used for the entire sequence length.

        Returns:
            Output tensor of the same shape as input, with RoPE
            applied.
        """
        # NOTE: Input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # Extract the values based on whether input_pos is set or not
        rope_cache = self.cache[input_pos] if input_pos is not None else self.cache[:seq_len]

        # Reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # Tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # Reshape the cache for broadcasting
        # Tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # Otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # NOTE: Tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # NOTE: Tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


@typecheck
class Attention(nn.Module):
    """Standard multi-head self-attention module with QKV projection.

    This module implements the standard multi-head attention mechanism
    used in transformers. It supports both the fused attention
    implementation (scaled_dot_product_attention) for efficiency when
    available, and a manual implementation otherwise. The module
    includes options for QK normalization, attention dropout, and
    projection dropout.

    Adapted from
    https://github.com/huggingface/pytorch-image-models.

    Args:
        dim: Input dimension of the token embeddings.
        num_heads: Number of attention heads.
        context_length: Maximum sequence length for the rotary
            positional embeddings.
        rope_base: Base for the rotary positional embeddings.
        qkv_bias: Whether to use bias in the query, key, value
            projections.
        qk_norm: Whether to apply normalization to query and key
            vectors.
        scale_norm: Whether to apply scaling to the output of the
            attention mechanism.
        proj_bias: Whether to use bias in the output projection.
        flex_attn: Whether to use PyTorch's FlexAttention.
        fused_attn: Whether to use PyTorch's `scaled_dot_product_attention`.
        jvp_attn: Whether to use a Triton kernel for Jacobian-vector product (JVP) Flash Attention.
        attn_drop: Dropout rate applied to the attention weights.
        proj_drop: Dropout rate applied after the output projection.
        norm_layer: Normalization layer constructor for QK
            normalization if enabled.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        context_length: int | None = 2048,
        rope_base: int | None = 10_000,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        scale_norm: bool = False,
        proj_bias: bool = False,
        flex_attn: bool = False,
        fused_attn: bool = True,
        jvp_attn: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
        norm_layer: Type[nn.Module] | None = None,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert (
            sum([flex_attn, fused_attn, jvp_attn]) <= 1
        ), "Only one of flex_attn, fused_attn, or jvp_attn can be True."

        if flex_attn or jvp_attn:
            assert (
                attn_drop == 0.0
            ), "The `attn_drop` option cannot be used with FlexAttention or JVP Attention."

        dim_head = dim // num_heads
        self.rotary_emb = (
            RotaryPositionalEmbeddings(
                # Rotary embeddings on half the dims per head
                dim=dim_head,
                max_seq_len=context_length,
                base=rope_base,
            )
            if context_length is not None and rope_base is not None
            else None
        )
        if qk_norm or scale_norm:
            assert (
                norm_layer is not None
            ), "norm_layer must be provided if qk_norm or scale_norm is True"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.flex_attn = flex_attn
        self.fused_attn = fused_attn
        self.jvp_attn = jvp_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    @property
    def device(self) -> torch.device:
        """Return the device of the attention module."""
        return next(self.parameters()).device

    @typecheck
    def forward(
        self,
        x: Float["b m c"],  # type: ignore
        pos_ids: Int["b m"] | None = None,  # type: ignore
        attn_mask: BlockMask | Bool["b h m m"] | Float["b 1 m m"] | None = None,  # type: ignore
    ) -> Float["b m c"]:  # type: ignore
        """Forward pass for the attention layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim].
            pos_ids: Optional position IDs tensor of shape
                [batch_size, seq_len].
            attn_mask: Optional BlockMask or attention mask of shape
                [batch_size, 1 or num_heads (h), seq_len, seq_len].

        Returns:
            Output tensor of shape [batch_size, seq_len, dim].
        """
        B, M, C = x.shape

        # Prepare queries, keys, and values
        qkv = self.qkv(x).reshape(B, M, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Maybe apply RoPE before performing scaled dot product attention
        if self.rotary_emb is not None:
            # NOTE: RoPE expects its inputs to be of shape (b, m, h, c)
            q = self.rotary_emb(q.permute(0, 2, 1, 3), input_pos=pos_ids).permute(0, 2, 1, 3)
            k = self.rotary_emb(k.permute(0, 2, 1, 3), input_pos=pos_ids).permute(0, 2, 1, 3)

        # Apply attention variant
        if self.flex_attn:
            amp_device_type = self.device.type
            amp_dtype = get_widest_dtype(q, k, v)

            @torch.amp.custom_fwd(device_type=amp_device_type, cast_inputs=amp_dtype)
            def mixed_precision_flex_attention(
                q: Float["b m c"],  # type: ignore
                k: Float["b m c"],  # type: ignore
                v: Float["b m c"],  # type: ignore
                block_mask: BlockMask | None = None,  # type: ignore
            ) -> Float["b m c"]:  # type: ignore
                """Ensure that attention is computed consistently in the widest mixed precision
                using FlexAttention.

                Args:
                    q: Query tensor of shape [batch_size, seq_len,
                        dim].
                    k: Key tensor of shape [batch_size, seq_len,
                        dim].
                    v: Value tensor of shape [batch_size, seq_len,
                        dim].
                    block_mask: Optional BlockMask.

                Returns:
                    Output tensor of shape [batch_size, seq_len,
                        dim].
                """
                return flex_attention(q, k, v, block_mask=block_mask)

            x = mixed_precision_flex_attention(
                q,
                k,
                v,
                block_mask=attn_mask,
            )

        elif self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        elif self.jvp_attn:
            from jvp_flash_attention.jvp_attention import attention as jvp_attention

            x = jvp_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                # dropout_p=self.attn_drop.p if self.training else 0.0,  # NOTE: JVP attention dropout is currently unsupported
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = safe_softmax(attn, mask=attn_mask, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, M, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@typecheck
class Block(nn.Module):
    """Transformer block with pre-normalization.

    Adapted from
    https://github.com/huggingface/pytorch-image-models.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        context_length: Maximum sequence length for the rotary
            positional embeddings.
        rope_base: Base for the rotary positional embeddings.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        proj_drop: Projection dropout rate.
        attn_drop: Attention dropout rate.
        qkv_bias: If True, add a learnable bias to query, key, value.
        qk_norm: If True, apply normalization to query and key.
        scale_attn_norm: If True, apply scaling to attention
            normalization.
        scale_mlp_norm: If True, apply scaling to MLP normalization.
        proj_bias: If True, add bias to output projection.
        flex_attn: Whether to use PyTorch's FlexAttention.
        fused_attn: Whether to use PyTorch's `scaled_dot_product_attention`.
        jvp_attn: Whether to use a Triton kernel for Jacobian-vector product (JVP) Flash Attention.
        checkpoint_activations: If True, use activation checkpointing
            for memory efficiency during the forward pass.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        mlp_layer: MLP layer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_length: int | None = 2048,
        rope_base: int | None = 10_000,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.1,
        attn_drop: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = False,
        flex_attn: bool = False,
        fused_attn: bool = True,
        jvp_attn: bool = False,
        checkpoint_activations: bool = False,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()

        assert (
            sum([flex_attn, fused_attn, jvp_attn]) <= 1
        ), "Only one of flex_attn, fused_attn, or jvp_attn can be True."

        self.checkpoint_activations = checkpoint_activations

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            context_length=context_length,
            rope_base=rope_base,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            flex_attn=flex_attn,
            fused_attn=fused_attn,
            jvp_attn=jvp_attn,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )

    @typecheck
    def forward(
        self,
        x: Float["b m c"],  # type: ignore
        pos_ids: Int["b m"] | None = None,  # type: ignore
        attn_mask: BlockMask | Bool["b h m m"] | Float["b 1 m m"] | None = None,  # type: ignore
    ) -> Float["b m c"]:  # type: ignore
        """Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim].
            pos_ids: Optional position IDs tensor of shape
                [batch_size, seq_len]. This is used for rotary positional
                embeddings.
            attn_mask: Optional BlockMask or attention mask tensor of
            shape [batch_size, 1 or num_heads (h), seq_len, seq_len].

        Returns:
            Output tensor of the same shape as input.
        """

        def attn_fn(x: Float["b m c"]) -> Float["b m c"]:  # type: ignore
            """Attention function for the block."""
            return self.attn(self.norm1(x), pos_ids=pos_ids, attn_mask=attn_mask)

        def mlp_fn(x: Float["b m c"]) -> Float["b m c"]:  # type: ignore
            """MLP function for the block."""
            return self.mlp(self.norm2(x))

        if self.checkpoint_activations and not torch.jit.is_scripting():
            x = x + checkpoint(attn_fn, x, use_reentrant=False)
            x = x + checkpoint(mlp_fn, x, use_reentrant=False)
        else:
            x = x + self.attn(self.norm1(x), pos_ids=pos_ids, attn_mask=attn_mask)
            x = x + self.mlp(self.norm2(x))

        return x


if __name__ == "__main__":
    _ = Block()
