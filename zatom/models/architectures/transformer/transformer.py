"""Adapted from https://github.com/carlosinator/tabasco."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from zatom.models.architectures.transformer.attention import (
    AttentionBlock,
    ModernAttention,
)
from zatom.models.architectures.transformer.common import SwiGLUFeedForward
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
    """A self-attention transformer block using ModernAttention and SwiGLUFeedForward.

    Args:
        dim: Input and output dimension
        n_heads: Number of attention heads
        context_length: Maximum context length for rotary embeddings
        rope_base: Base frequency for rotary embeddings
        qk_layernorm: Whether to apply RMS normalization to queries and keys
        use_sdpa: Whether to use PyTorch's scaled dot-product attention
        jvp_attn: Whether to use JVP-compatible attention
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        n_heads: int,
        context_length: Optional[int] = 2048,
        rope_base: Optional[int] = 10_000,
        qk_layernorm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
    ):
        super().__init__()
        self.attention = ModernAttention(
            dim,
            n_heads,
            context_length=context_length,
            rope_base=rope_base,
            use_qk_norm=qk_layernorm,
            use_sdpa=use_sdpa,
            jvp_attn=jvp_attn,
        )
        self.feed_forward = SwiGLUFeedForward(dim=dim, hidden_dim=4 * dim)

        self.attention_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the modern transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            pos_ids: Position ids for rotary embeddings
                Shape: [batch_size, seq_len]
            padding_mask: Boolean mask for padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len],
                where False indicates positions to mask or the float -inf denotes masked positions

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        h = x + self.attention(
            self.attention_norm(x), pos_ids=pos_ids, padding_mask=padding_mask, attn_mask=attn_mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class ModernTransformerDecoderBlock(nn.Module):
    """A transformer decoder block using ModernAttention and SwiGLUFeedForward.

    Can be used as a drop-in replacement for `nn.TransformerDecoderLayer`.

    Args:
        dim: Input and output dimension
        n_heads: Number of attention heads
        context_length: Maximum context length for rotary embeddings
        rope_base: Base frequency for rotary embeddings
        qk_layernorm: Whether to apply RMS normalization to queries and keys
        use_sdpa: Whether to use PyTorch's scaled dot-product attention
        jvp_attn: Whether to use JVP-compatible attention
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        n_heads: int,
        context_length: Optional[int] = 2048,
        rope_base: Optional[int] = 10_000,
        qk_layernorm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
    ):
        super().__init__()
        self.self_attention = ModernAttention(
            dim,
            n_heads,
            context_length=context_length,
            rope_base=rope_base,
            use_qk_norm=qk_layernorm,
            use_sdpa=use_sdpa,
            jvp_attn=jvp_attn,
        )
        self.cross_attention = ModernAttention(
            dim,
            n_heads,
            context_length=context_length,
            rope_base=rope_base,
            use_qk_norm=qk_layernorm,
            use_sdpa=use_sdpa,
            jvp_attn=jvp_attn,
        )
        self.feed_forward = SwiGLUFeedForward(dim=dim, hidden_dim=4 * dim)

        self.self_attention_norm = nn.RMSNorm(dim)
        self.cross_attention_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the modern transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            memory: Optional memory tensor for cross-attention
                Shape: [batch_size, seq_len, dim]
            pos_ids: Position ids for rotary embeddings
                Shape: [batch_size, seq_len]
            tgt_key_padding_mask: Boolean mask for target padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            memory_key_padding_mask: Boolean mask for memory padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len],
                where False indicates positions to mask or the float -inf denotes masked positions

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        h = x + self.self_attention(
            self.self_attention_norm(x),
            pos_ids=pos_ids,
            padding_mask=tgt_key_padding_mask,
            attn_mask=attn_mask,
        )
        h = h + self.cross_attention(
            self.cross_attention_norm(h),
            memory=memory,
            pos_ids=pos_ids,
            padding_mask=memory_key_padding_mask,
            attn_mask=attn_mask,
        )
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
    """A modern Transformer model with optional RoPE embeddings, query-key normalization, and
    scaled dot-product attention.

    Args:
        dim: Model dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        context_length: Maximum context length for rotary embeddings
        rope_base: Base frequency for rotary embeddings
        repr_layer: Layer at which to additionally extract intermediate representations. If None, no intermediate representation is extracted.
        qk_layernorm: Whether to apply RMS normalization to queries and keys
        use_sdpa: Whether to use PyTorch's scaled dot-product attention
        jvp_attn: Whether to use JVP-compatible attention
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        context_length: Optional[int] = 2048,
        rope_base: Optional[int] = 10_000,
        repr_layer: Optional[int] = None,
        qk_layernorm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = context_length
        self.repr_layer = repr_layer

        self.layers = nn.ModuleList(
            [
                ModernTransformerBlock(
                    dim,
                    num_heads,
                    context_length=context_length,
                    rope_base=rope_base,
                    qk_layernorm=qk_layernorm,
                    use_sdpa=use_sdpa,
                    jvp_attn=jvp_attn,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.RMSNorm(dim)

        if repr_layer is not None:
            self.repr_norm = nn.RMSNorm(dim)

    @typecheck
    def forward(
        self,
        h: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the modern transformer.

        Args:
            h: Input tensor of shape [batch_size, seq_len, dim].
            pos_ids: Position ids for rotary embeddings
                Shape: [batch_size, seq_len]
            padding_mask: Boolean mask for padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Mask to prevent attention to certain positions.
                Values should be 0 for positions to attend to, and -inf for masked positions.
                To make the model causal (decoder-style), pass a causal mask.
                Shape: [batch_size, n_heads, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim] or a tuple of
            (output tensor, intermediate representation) if `repr_layer` is set.
        """
        _, seq_len, _ = h.shape
        assert (
            self.max_seq_len is None or seq_len <= self.max_seq_len
        ), "Sequence length exceeds model's maximum sequence length"

        repr = None
        for i, layer in enumerate(self.layers):
            h = layer(h, pos_ids=pos_ids, padding_mask=padding_mask, attn_mask=attn_mask)
            if self.repr_layer is not None and i == self.repr_layer:
                repr = h.clone()

        h = self.norm(h)

        if self.repr_layer is not None:
            assert (
                repr is not None
            ), f"The specified `repr_layer` ({self.repr_layer}) was not reached during the forward pass."
            repr = self.repr_norm(repr)

            return h, repr

        return h
