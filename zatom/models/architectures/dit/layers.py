"""Attention layers.

Adapted from:
    - https://github.com/apple/ml-simplefold
"""

import math
import warnings

import torch
import torch.nn.functional as F
from einops import rearrange
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.linear import PlatonicLinear
from torch import Tensor, nn
from torch.nn.attention import sdpa_kernel

from zatom.utils.training_utils import SDPA_BACKENDS
from zatom.utils.typing_utils import typecheck

# Suppress warnings from JVP Flash Attention for padding rows that are fully masked
warnings.filterwarnings(
    "ignore", message=".*divide by zero encountered in matmul.*", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", message=".*overflow encountered in matmul.*", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", message=".*invalid value encountered in matmul.*", category=RuntimeWarning
)

#################################################################################
#                               Misc. Utilities                                 #
#################################################################################


@typecheck
def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply feature-wise linear modulation.

    Args:
        x: Input tensor of shape (B, N, C).
        shift: Shift tensor of shape (B, C).
        scale: Scale tensor of shape (B, C).

    Returns:
        Modulated tensor of shape (B, N, C).
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        d: Model size.
        p: Partial RMSNorm, valid value [0, 1], default -1.0 (disabled).
        eps: Epsilon value, default 1e-8.
        bias: Whether to use bias term for RMSNorm, disabled by
              default because RMSNorm doesn't enforce re-centering invariance.
    """

    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RMSNorm.

        Args:
            x: Input tensor of shape (..., d).

        Returns:
            Normalized tensor of the same shape as input.
        """
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


#################################################################################
#                            Attention Layers                                   #
#################################################################################


class SelfAttentionLayer(nn.Module):
    """Multi-head Self-Attention Layer.

    Args:
        hidden_size: The dimension of the input and output embeddings.
        num_heads: Number of attention heads.
        qkv_bias: If True, add a learnable bias to query, key, value projections.
        qk_scale: Override default scaling of QK^T.
        attn_drop: Dropout probability for attention weights.
        proj_drop: Dropout probability for output projection.
        use_bias: If True, add a learnable bias to the output projection.
        qk_norm: If True, apply RMSNorm to queries and keys.
        jvp_attn: Whether to use JVP Flash Attention instead of PyTorch's Scaled Dot Product Attention.
        pos_embedder: Optional positional embedding module.
        linear_target: The linear layer class to use (default: nn.Linear).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias: bool = True,
        qk_norm: bool = True,
        jvp_attn: bool = False,
        pos_embedder: nn.Module | None = None,
        linear_target: nn.Module = nn.Linear,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.jvp_attn = jvp_attn

        if jvp_attn and not isinstance(self, EfficientSelfAttentionLayer):
            raise ValueError(
                "The `jvp_attn` option can only be used with `EfficientSelfAttentionLayer`."
            )

        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = linear_target(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_target(hidden_size, hidden_size, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.pos_embedder = pos_embedder

    @typecheck
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward pass of the Self-Attention Layer.

        Args:
            x: Input tensor of shape (B, N, C).
            kwargs: Additional arguments, e.g., positional embeddings.

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        pos = kwargs.get("pos")

        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # Make torchscript happy (as one cannot use tensor as tuple)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientSelfAttentionLayer(SelfAttentionLayer):
    """Efficient Multi-head Self-Attention Layer using FlashAttention.

    Started from
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @typecheck
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward pass of the Efficient Self-Attention Layer.

        Args:
            x: Input tensor of shape (B, N, C).
            kwargs: Additional arguments, e.g., positional embeddings.

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        attn_mask = kwargs.get("attention_mask")
        pos = kwargs.get("pos")
        sdpa_backends = kwargs.get("sdpa_backends", SDPA_BACKENDS)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = qkv.unbind(0)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)

        q, k = self.q_norm(q), self.k_norm(k)

        # JVP Flash Attention, with support for second-order derivatives
        if self.jvp_attn:
            from jvp_flash_attention.jvp_attention import JVPAttn

            x = JVPAttn.fwd_dual(q, k, v, attn_mask=attn_mask)

        # PyTorch's Scaled Dot Product Attention
        else:
            with sdpa_kernel(sdpa_backends):
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

            if not self.training:
                # Ensure no NaNs appear during eval if a (padding) row is (fully) masked
                # NOTE: This appears to be a bug in PyTorch's fused attention: https://github.com/pytorch/pytorch/issues/163997
                fully_masked = (~attn_mask).all(dim=-1, keepdim=True)
                x = torch.where(fully_masked, torch.zeros_like(x), x)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PlatonicEfficientSelfAttentionLayer(nn.Module):
    """Platonic Efficient Multi-head Self-Attention Layer using FlashAttention.

    Started from
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py.

    Args:
        hidden_size: The dimension of the input and output embeddings.
        solid: The name of the Platonic solid (e.g., `tetrahedron`, `octahedron`, `icosahedron`) to define the symmetry group.
        num_heads: Number of attention heads.
        qkv_bias: If True, add a learnable bias to query, key, value projections.
        qk_scale: Override default scaling of QK^T.
        attn_drop: Dropout probability for attention weights.
        proj_drop: Dropout probability for output projection.
        use_bias: If True, add a learnable bias to the output projection.
        qk_norm: If True, apply RMSNorm to queries and keys.
        jvp_attn: Whether to use JVP Flash Attention instead of PyTorch's Scaled Dot Product Attention.
        pos_embedder: Optional positional embedding module.
    """

    def __init__(
        self,
        hidden_size: int,
        solid: str,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias: bool = True,
        qk_norm: bool = True,
        jvp_attn: bool = False,
        pos_embedder: nn.Module | None = None,
    ):
        super().__init__()

        if solid.lower() not in PLATONIC_GROUPS:
            raise ValueError(
                f"Solid '{solid}' not recognized. Available groups are: {list(PLATONIC_GROUPS.keys())}"
            )

        group = PLATONIC_GROUPS[solid.lower()]
        self.num_G = group.G

        self.num_heads = num_heads
        self.jvp_attn = jvp_attn

        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = PlatonicLinear(hidden_size, hidden_size * 3, solid=solid, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = PlatonicLinear(hidden_size, hidden_size, solid=solid, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = RMSNorm(hidden_size // self.num_G) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(hidden_size // self.num_G) if qk_norm else nn.Identity()

        head_dim_G = hidden_size // self.num_G

        self.num_heads_G = num_heads // self.num_G
        self.pos_embedder_head_dim = head_dim_G // self.num_heads_G

        self.pos_embedder = pos_embedder(
            num_heads=self.num_heads_G,
            head_dim=self.pos_embedder_head_dim,
        )

    @typecheck
    def group_normalize(self, x: Tensor, norm_layer: RMSNorm) -> Tensor:
        """Helper to apply RMSNorm on the per-group-element dimension.

        Args:
            x: Input tensor of shape [..., G*C]
            norm_layer: RMSNorm module to apply on the per-group-element dimension.

        Returns:
            Normalized tensor of the same shape as input [..., G*C].
        """
        leading_dims = x.shape[:-1]

        # Reshape to expose group axis: [..., G*C] -> [..., G, C]
        x_reshaped = x.view(*leading_dims, self.num_G, -1)

        # Apply normalization
        normed_reshaped = norm_layer(x_reshaped)

        # Reshape back to original convention
        return normed_reshaped.view(*leading_dims, -1)

    @typecheck
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward pass of the Efficient Self-Attention Layer.

        Args:
            x: Input tensor of shape (B, N, C).
            kwargs: Additional arguments, e.g., positional embeddings.

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        G, H, D_h = self.num_G, self.num_heads_G, self.pos_embedder_head_dim

        attn_mask = kwargs.get("attention_mask")
        pos = kwargs.get("pos")
        sdpa_backends = kwargs.get("sdpa_backends", SDPA_BACKENDS)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = qkv.unbind(0)

        if self.pos_embedder and pos is not None:
            q = self.pos_embedder(q.reshape(B, N, G, H, D_h), pos).reshape(B, G * H, N, D_h)
            k = self.pos_embedder(k.reshape(B, N, G, H, D_h), pos).reshape(B, G * H, N, D_h)

        q = self.group_normalize(q.reshape(B, N, G * H * D_h), self.q_norm).reshape(
            B, G * H, N, D_h
        )
        k = self.group_normalize(k.reshape(B, N, G * H * D_h), self.k_norm).reshape(
            B, G * H, N, D_h
        )

        # JVP Flash Attention, with support for second-order derivatives
        if self.jvp_attn:
            from jvp_flash_attention.jvp_attention import JVPAttn

            x = JVPAttn.fwd_dual(q, k, v, attn_mask=attn_mask)

        # PyTorch's Scaled Dot Product Attention
        else:
            with sdpa_kernel(sdpa_backends):
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

            if not self.training:
                # Ensure no NaNs appear during eval if a (padding) row is (fully) masked
                # NOTE: This appears to be a bug in PyTorch's fused attention: https://github.com/pytorch/pytorch/issues/163997
                fully_masked = (~attn_mask).all(dim=-1, keepdim=True)
                x = torch.where(fully_masked, torch.zeros_like(x), x)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


#################################################################################
#                              FeedForward Layer                                #
#################################################################################


class SwiGLUFeedForward(nn.Module):
    """SwiGLU Feed Forward Layer.

    Args:
        dim: Input and output dimension.
        hidden_dim: Hidden dimension.
        multiple_of: Ensure the hidden dimension is a multiple of this value.
    """

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights."""
        torch.nn.init.xavier_uniform_(self.w1.weight)
        torch.nn.init.xavier_uniform_(self.w2.weight)
        torch.nn.init.xavier_uniform_(self.w3.weight)
        if self.w1.bias is not None:
            torch.nn.init.constant_(self.w1.bias, 0)
        if self.w2.bias is not None:
            torch.nn.init.constant_(self.w2.bias, 0)
        if self.w3.bias is not None:
            torch.nn.init.constant_(self.w3.bias, 0)

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the SwiGLU Feed Forward Layer.

        Args:
            x: Input tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, N, C).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class PlatonicSwiGLUFeedForward(nn.Module):
    """Platonic SwiGLU Feed Forward Layer.

    Args:
        dim: Input and output dimension.
        hidden_dim: Hidden dimension.
        solid: The name of the Platonic solid (e.g., `tetrahedron`, `octahedron`, `icosahedron`) to define the symmetry group.
        multiple_of: Ensure the hidden dimension is a multiple of this value.
    """

    def __init__(self, dim: int, hidden_dim: int, solid: str, multiple_of: int = 192):
        super().__init__()

        if solid.lower() not in PLATONIC_GROUPS:
            raise ValueError(
                f"Solid '{solid}' not recognized. Available groups are: {list(PLATONIC_GROUPS.keys())}"
            )

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = PlatonicLinear(dim, hidden_dim, solid=solid, bias=False)
        self.w2 = PlatonicLinear(hidden_dim, dim, solid=solid, bias=True)
        self.w3 = PlatonicLinear(dim, hidden_dim, solid=solid, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights."""
        torch.nn.init.xavier_uniform_(self.w1.kernel)
        torch.nn.init.xavier_uniform_(self.w2.kernel)
        torch.nn.init.xavier_uniform_(self.w3.kernel)
        if self.w1.bias is not None:
            torch.nn.init.constant_(self.w1.bias, 0)
        if self.w2.bias is not None:
            torch.nn.init.constant_(self.w2.bias, 0)
        if self.w3.bias is not None:
            torch.nn.init.constant_(self.w3.bias, 0)

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the SwiGLU Feed Forward Layer.

        Args:
            x: Input tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, N, C).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


#################################################################################
#                               Utility Layers                                  #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations.

    Args:
        hidden_size: The dimension of the output embeddings.
        frequency_embedding_size: The size of the sinusoidal frequency embeddings.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    @typecheck
    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """Create sinusoidal timestep embeddings.

        From https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py.

        Args:
            t: A 1-D tensor of N indices, one per batch element.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.

        Returns:
            A (N, D) tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    @typecheck
    def forward(self, t: Tensor) -> Tensor:
        """Forward pass of the Timestep Embedder.

        Args:
            t: A 1-D tensor of shape (B,) containing the timesteps.

        Returns:
            A tensor of shape (B, hidden_size) containing the timestep embeddings.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embed class labels into vector representations.

    NOTE: Also handles label dropout for context conditioning.

    Args:
        num_classes: The number of classes.
        hidden_size: The dimensionality of the hidden representations.
        dropout_prob: The dropout probability for context conditioning.
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)

    @typecheck
    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Drop labels to enable context conditioning.

        Args:
            labels: The input labels tensor.
            force_drop_ids: Optional tensor indicating which labels to drop.

        Returns:
            The modified labels tensor.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, 0, labels)
        # NOTE: 0 is the label for the null class
        return labels

    @typecheck
    def forward(
        self, labels: torch.Tensor, train: bool, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for label embedding.

        Args:
            labels: The input labels tensor.
            train: Whether the model is in training mode.
            force_drop_ids: Optional tensor indicating which labels to drop.

        Returns:
            The output embeddings tensor.
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class FinalLayer(nn.Module):
    """The final layer of DiT.

    Args:
        hidden_size: The input dimension.
        out_channels: The output dimension.
        c_dim: The dimension of the conditioning vector.
        condition_on_input: Whether to condition on the input as well as the provided context.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        c_dim: int | None = None,
        condition_on_input: bool = False,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                hidden_size + c_dim if condition_on_input else c_dim, 2 * hidden_size, bias=True
            ),
        )
        self.condition_on_input = condition_on_input
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights."""

        def _basic_init(module):
            """Initialize Linear layers with Xavier uniform."""
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize transformer layers
        self.apply(_basic_init)

        # Zero-out output layers
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    @typecheck
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Forward pass of the Final Layer.

        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, c_dim).

        Returns:
            Output tensor of shape (B, N, out_channels).
        """
        adaLN_input = (
            torch.cat((x, c.unsqueeze(-2).repeat(1, x.shape[1], 1)), dim=-1)
            if self.condition_on_input
            else c
        )
        shift, scale = self.adaLN_modulation(adaLN_input).chunk(2, dim=-1)
        x = (
            self.norm_final(x) * (1 + scale) + shift
            if self.condition_on_input
            else modulate(self.norm_final(x), shift, scale)
        )
        x = self.linear(x)
        return x


class PlatonicFinalLayer(nn.Module):
    """The final layer of DiP.

    Args:
        hidden_size: The input dimension.
        out_channels: The output dimension.
        solid: The name of the Platonic solid (e.g., `tetrahedron`, `octahedron`, `icosahedron`) to define the symmetry group.
        c_dim: The dimension of the conditioning vector.
        condition_on_input: Whether to condition on the input as well as the provided context.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        solid: str,
        c_dim: int | None = None,
        condition_on_input: bool = False,
    ):
        super().__init__()

        if solid.lower() not in PLATONIC_GROUPS:
            raise ValueError(
                f"Solid '{solid}' not recognized. Available groups are: {list(PLATONIC_GROUPS.keys())}"
            )

        group = PLATONIC_GROUPS[solid.lower()]
        self.num_G = group.G

        self.norm_final = nn.LayerNorm(
            hidden_size // self.num_G, elementwise_affine=False, eps=1e-6
        )
        self.linear = PlatonicLinear(hidden_size, out_channels, solid=solid, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            PlatonicLinear(
                hidden_size + c_dim if condition_on_input else c_dim,
                2 * hidden_size,
                solid=solid,
                bias=True,
            ),
        )
        self.condition_on_input = condition_on_input
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights."""

        def _basic_init(module):
            """Initialize Linear layers with Xavier uniform."""
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize transformer layers
        self.apply(_basic_init)

        # Zero-out output layers
        nn.init.constant_(self.adaLN_modulation[-1].kernel, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.kernel, 0)
        nn.init.constant_(self.linear.bias, 0)

    @typecheck
    def group_normalize(self, x: Tensor, norm_layer: nn.LayerNorm) -> Tensor:
        """Helper to apply LayerNorm on the per-group-element dimension.

        Args:
            x: Input tensor of shape [..., G*C]
            norm_layer: LayerNorm module to apply on the per-group-element dimension.

        Returns:
            Normalized tensor of the same shape as input [..., G*C].
        """
        leading_dims = x.shape[:-1]

        # Reshape to expose group axis: [..., G*C] -> [..., G, C]
        x_reshaped = x.view(*leading_dims, self.num_G, -1)

        # Apply normalization
        normed_reshaped = norm_layer(x_reshaped)

        # Reshape back to original convention
        return normed_reshaped.view(*leading_dims, -1)

    @typecheck
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Forward pass of the Final Layer.

        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, c_dim).

        Returns:
            Output tensor of shape (B, N, out_channels).
        """
        adaLN_input = (
            torch.cat((x, c.unsqueeze(-2).repeat(1, x.shape[1], 1)), dim=-1)
            if self.condition_on_input
            else c
        )
        shift, scale = self.adaLN_modulation(adaLN_input).chunk(2, dim=-1)
        x = (
            self.group_normalize(x, self.norm_final) * (1 + scale) + shift
            if self.condition_on_input
            else modulate(self.group_normalize(x, self.norm_final), shift, scale)
        )
        x = self.linear(x)
        return x
