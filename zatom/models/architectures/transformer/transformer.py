"""Adapted from https://github.com/carlosinator/tabasco."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from zatom.models.architectures.transformer.attention import AttentionBlock
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
        qk_layernorm: Whether to apply layer normalization to query and key in attention
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
        qk_layernorm: bool = False,
        activation_type: str = "swiglu",
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Self-attention block
        self.attn_block = AttentionBlock(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            qk_layernorm=qk_layernorm,
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
        qk_layernorm: Whether to apply layer normalization to query and key in attention
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
        qk_layernorm: bool = False,
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
                    qk_layernorm=qk_layernorm,
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
