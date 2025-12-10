"""Adapted from https://github.com/carlosinator/tabasco."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from zatom.models.architectures.transformer.attention import (
    AttentionBlock,
    ModernSelfAttention,
)
from zatom.models.architectures.transformer.common import (
    SwiGLUFeedForward,
    precompute_rope_theta,
)
from zatom.models.architectures.transformer.transition import Transition
from zatom.utils.typing_utils import typecheck


class TransformerBlock(nn.Module):
    """A transformer block with layer normalization and residual connections.

    This implements a standard transformer block with self-attention followed by
    a feed-forward network, with layer normalization and residual connections.

    Args:
        dim: Input and output dimension
        num_heads: Number of attention heads
        mlp_dim: Hidden dimension for the feed-forward network (defaults to 4x input dim)
        dropout: Dropout probability
        activation_type: Type of activation to use in the feed-forward network
        norm_eps: Epsilon value for layer normalization
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int = None,
        dropout: float = 0.0,
        activation_type: str = "swiglu",
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Self-attention block
        self.attn_block = AttentionBlock(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            norm_eps=norm_eps,
        )

        # Feed-forward network
        self.ff_block = Transition(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
            activation_type=activation_type,
            layer_norm=True,
        )

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            padding_mask: Boolean mask for padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Mask to prevent attention to certain positions
                Shape: [seq_len, seq_len] or [batch_size, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Self-attention with residual connection
        attn_output = self.attn_block(x, key_padding_mask=padding_mask, attn_mask=attn_mask)
        x = x + attn_output

        # Feed-forward network with residual connection
        ff_output = self.ff_block(x)
        x = x + ff_output

        return x


class ModernTransformerBlock(nn.Module):
    """A transformer block using ModernSelfAttention and SwiGLUFeedForward.

    Args:
        dim: Input and output dimension
        n_heads: Number of attention heads
        use_sdpa: Whether to use PyTorch's scaled dot-product attention
    """

    @typecheck
    def __init__(self, dim: int, n_heads: int, use_sdpa: bool = True):
        super().__init__()
        self.attention = ModernSelfAttention(dim, n_heads, use_sdpa=use_sdpa)
        self.feed_forward = SwiGLUFeedForward(dim=dim, hidden_dim=4 * dim)

        self.attention_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)

    @typecheck
    def forward(
        self, x: torch.Tensor, freqs_complex: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the modern transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            freqs_complex: Precomputed rotary frequency tensor of shape [seq_len, head_dim]
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        h = x + self.attention(self.attention_norm(x), freqs_complex, attn_mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """A standard Transformer model with multiple layers.

    This implements a sequence of transformer blocks, each containing
    self-attention and feed-forward components with residual connections.

    Args:
        dim: Model dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        repr_layer: Layer at which to additionally extract intermediate representations. If None, no intermediate representation is extracted.
        mlp_dim: Hidden dimension for feed-forward networks (defaults to 4x dim)
        dropout: Dropout probability
        activation_type: Type of activation to use in feed-forward networks
        norm_eps: Epsilon value for layer normalization
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        repr_layer: Optional[int] = None,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation_type: str = "gelu",
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.repr_layer = repr_layer

        if mlp_dim is None:
            mlp_dim = 4 * dim

        # Create a sequence of transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    activation_type=activation_type,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

        if repr_layer is not None:
            self.repr_norm = nn.LayerNorm(dim, eps=norm_eps)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the transformer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            padding_mask: Boolean mask for padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Mask to prevent attention to certain positions
                Shape: [seq_len, seq_len] or [batch_size, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim] or a tuple of
            (output tensor, intermediate representation) if `repr_layer` is set.
        """
        # Pass through each transformer block
        repr = None
        for i, layer in enumerate(self.layers):
            x = layer(x, padding_mask=padding_mask, attn_mask=attn_mask)
            if self.repr_layer is not None and i == self.repr_layer:
                repr = x.clone()

        # Apply final normalization
        x = self.norm(x)

        if self.repr_layer is not None:
            assert (
                repr is not None
            ), f"The specified `repr_layer` ({self.repr_layer}) was not reached during the forward pass."
            repr = self.repr_norm(repr)

            return x, repr

        return x


class ModernTransformer(nn.Module):
    """A modern Transformer model with RoPE embeddings, optional query-key normalization, and
    scaled dot-product attention.

    Args:
        dim: Model dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        device: Device to place the model on
        use_sdpa: Whether to use PyTorch's scaled dot-product attention
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        max_seq_len: int,
        device: str,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.device = device

        self.layers = nn.ModuleList(
            [ModernTransformerBlock(dim, num_heads, use_sdpa=use_sdpa) for _ in range(depth)]
        )
        self.norm = nn.RMSNorm(dim)

        self.freqs_complex = precompute_rope_theta(
            self.dim // self.num_heads, self.max_seq_len, device
        )

    @typecheck
    def forward(self, h: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the modern transformer.

        Args:
            h: Input tensor of shape [batch_size, seq_len, dim]
            attn_mask: Mask to prevent attention to certain positions.
                Values should be 0 for positions to attend to, and -inf for masked positions.
                To make the model causal (decoder-style), pass a causal mask.
                Shape: [batch_size, n_heads, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        _, seq_len, _ = h.shape
        assert (
            seq_len <= self.max_seq_len
        ), "Sequence length exceeds model's maximum sequence length"

        freqs_complex = self.freqs_complex[:seq_len]

        for layer in self.layers:
            h = layer(h, freqs_complex, attn_mask)

        h = self.norm(h)
        return h
