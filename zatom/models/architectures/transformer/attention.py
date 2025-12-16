"""Adapted from https://github.com/carlosinator/tabasco."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jvp_flash_attention.jvp_attention import JVPAttn

from zatom.models.architectures.transformer.positional_encoder import (
    RotaryPositionalEmbeddings,
)
from zatom.utils.typing_utils import typecheck


class Attention(nn.Module):
    """A wrapper around PyTorch's MultiheadAttention module with a simplified interface.

    This class provides a more convenient interface for using multi-head attention
    in transformer architectures, handling the reshaping and masking operations.

    Args:
        dim: Input and output dimension
        num_heads: Number of attention heads
        dropout: Dropout probability for attention weights
        bias: Whether to include bias terms in the projection layers
        batch_first: Whether input tensors are in batch-first format (batch, seq, features)
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

    @typecheck
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the multi-head attention layer.

        Args:
            query: Query tensor of shape [batch_size, seq_len_q, dim]
            key: Key tensor of shape [batch_size, seq_len_k, dim] (defaults to query if None)
            value: Value tensor of shape [batch_size, seq_len_v, dim] (defaults to key if None)
            key_padding_mask: Boolean mask for keys to ignore (True means ignore)
                Shape: [batch_size, seq_len_k]
            attn_mask: Mask to prevent attention to certain positions
                Shape: [seq_len_q, seq_len_k] or [batch_size, seq_len_q, seq_len_k]
            need_weights: Whether to return attention weights

        Returns:
            Output tensor of shape [batch_size, seq_len_q, dim]
            (and optionally attention weights if need_weights=True)
        """
        key = query if key is None else key
        value = key if value is None else value

        attn_output, attn_weights = self.mha(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )

        if need_weights:
            return attn_output, attn_weights

        return attn_output


class ModernAttention(nn.Module):
    """Modern attention module with rotary position embeddings and optional SDPA.

    Args:
        dim: Input dimension
        n_heads: Number of attention heads
        context_length: Maximum context length for rotary embeddings
        rope_base: Base frequency for rotary embeddings
        use_qk_norm: Whether to apply RMS normalization to queries and keys
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
        use_qk_norm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert not (jvp_attn and use_sdpa), "Either jvp_attn or use_sdpa can be True, not both."

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_sdpa = use_sdpa
        self.jvp_attn = jvp_attn

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        self.rotary_emb = (
            RotaryPositionalEmbeddings(
                # Rotary embeddings on half the dims per head
                dim=self.head_dim,
                max_seq_len=context_length,
                base=rope_base,
            )
            if context_length is not None and rope_base is not None
            else None
        )

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the modern attention layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            memory: Optional memory tensor for cross-attention
                Shape: [batch_size, seq_len, dim]
            pos_ids: Position ids tensor of shape [batch_size, seq_len]
            padding_mask: Boolean mask for padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len],
                where False indicates positions to mask or the float -inf denotes masked positions

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(memory if memory is not None else x)
        v = self.v_proj(memory if memory is not None else x)

        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE before performing scaled dot product attention
        if self.rotary_emb is not None:
            assert pos_ids is not None, "pos_ids must be provided when using RoPE."
            # NOTE: RoPE expects its inputs to be of shape (B, Seq_Len, H, Head_Dim)
            q = self.rotary_emb(q, input_pos=pos_ids)
            k = self.rotary_emb(k, input_pos=pos_ids)

        # Transpose for attention calculation: (B, H, Seq_Len, Head_Dim)
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        final_mask = attn_mask

        if padding_mask is not None:
            mask_broadcast = padding_mask[:, None, None, :]  # [B, 1, 1, Seq_Len]

            if final_mask is not None and final_mask.is_floating_point():
                # Create a float mask: 0.0 where we keep, -inf where we mask
                pad_mask = torch.zeros((batch_size, 1, 1, seq_len), device=x.device, dtype=q.dtype)
                pad_mask.masked_fill_(mask_broadcast, float("-inf"))
            else:
                # Create a (more memory efficient) boolean mask: False where we mask
                pad_mask = (~mask_broadcast).repeat(
                    1, self.n_heads, seq_len, 1
                )  # [B, H, Seq_Len, Seq_Len]

            # Combine with existing mask
            if final_mask is None:
                final_mask = pad_mask
            elif final_mask is not None and final_mask.is_floating_point():
                final_mask = final_mask + pad_mask
            else:
                final_mask = final_mask & pad_mask

        if self.use_sdpa:
            # Use PyTorch's optimized scaled dot-product attention.
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=final_mask)
        elif self.jvp_attn:
            output = JVPAttn.fwd_dual(q, k, v, attn_mask=final_mask)
        else:
            # Manual implementation for comparison or environments where SDPA is not available.
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

            if final_mask is not None and final_mask.is_floating_point():
                scores = scores + final_mask
            elif final_mask is not None:
                scores = scores.masked_fill(~final_mask, float("-inf"))

            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            output = torch.matmul(scores, v)

        # (B, H, Seq_Len, Head_Dim) -> (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, Dim)
        output = output.transpose(-2, -3).reshape(batch_size, seq_len, self.dim)
        return self.o_proj(output)


class AttentionBlock(nn.Module):
    """A block of attention layers with layer normalization.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear projections
        batch_first: Whether input is batch-first (batch, seq, features)
        norm_eps: Epsilon value for layer normalization
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

        self.attention = Attention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the attention block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            key_padding_mask: Boolean mask for keys to ignore (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Mask to prevent attention to certain positions
                Shape: [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            need_weights: Whether to return attention weights

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
            (and optionally attention weights if need_weights=True)
        """
        x = self.norm(x)
        x = self.attention(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )
        return x


class AdaLNAttention(nn.Module):
    """Attention module with Adaptive Layer Normalization (AdaLN).

    This implements an attention mechanism with adaptive layer normalization,
    which allows for conditioning the layer normalization parameters based on
    additional inputs.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear projections
        batch_first: Whether input is batch-first (batch, seq, features)
        norm_eps: Epsilon value for layer normalization
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm_eps = norm_eps

        # Pre-normalization layer
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

        # Adaptive LN parameters
        self.adaln_gamma = nn.Linear(dim, dim, bias=False)
        self.adaln_beta = nn.Linear(dim, dim, bias=False)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the AdaLN attention layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            context: Context tensor for conditioning the layer norm parameters
                     Shape: [batch_size, context_dim] or [batch_size, seq_len, dim]
            key_padding_mask: Boolean mask for keys to ignore (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Mask to prevent attention to certain positions
                Shape: [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            need_weights: Whether to return attention weights

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
            (and optionally attention weights if need_weights=True)
        """
        # If context is [batch_size, context_dim], expand to match sequence length
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)

        # Compute adaptive LN parameters
        gamma = self.adaln_gamma(context)
        beta = self.adaln_beta(context)

        # Apply layer normalization with adaptive parameters
        x_norm = self.norm(x)
        x_norm = x_norm * (1 + gamma) + beta

        # Apply self-attention
        attn_output, attn_weights = self.mha(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )

        if need_weights:
            return attn_output, attn_weights
        return attn_output


class CrossAdaLNAttention(nn.Module):
    """Cross-attention module with Adaptive Layer Normalization (AdaLN).

    This implements a cross-attention mechanism with adaptive layer normalization,
    allowing for conditioning the layer normalization parameters based on
    additional inputs.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear projections
        batch_first: Whether input is batch-first (batch, seq, features)
        norm_eps: Epsilon value for layer normalization
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm_eps = norm_eps

        # Pre-normalization layer
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

        # Adaptive LN parameters
        self.adaln_gamma = nn.Linear(dim, dim, bias=False)
        self.adaln_beta = nn.Linear(dim, dim, bias=False)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the CrossAdaLNAttention layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len_q, dim]
            context: Context tensor for conditioning the layer norm parameters
                     Shape: [batch_size, context_dim] or [batch_size, seq_len_q, dim]
            encoder_hidden_states: Key/value tensor from encoder
                     Shape: [batch_size, seq_len_kv, dim]
            encoder_padding_mask: Boolean mask for encoder outputs to ignore (True means ignore)
                Shape: [batch_size, seq_len_kv]
            attn_mask: Mask to prevent attention to certain positions
                Shape: [seq_len_q, seq_len_kv] or [batch_size, seq_len_q, seq_len_kv]
            need_weights: Whether to return attention weights

        Returns:
            Output tensor of shape [batch_size, seq_len_q, dim]
            (and optionally attention weights if need_weights=True)
        """
        # If context is [batch_size, context_dim], expand to match sequence length
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)

        # Compute adaptive LN parameters
        gamma = self.adaln_gamma(context)
        beta = self.adaln_beta(context)

        # Apply layer normalization with adaptive parameters
        x_norm = self.norm(x)
        x_norm = x_norm * (1 + gamma) + beta

        # Apply cross-attention
        attn_output, attn_weights = self.mha(
            query=x_norm,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )

        if need_weights:
            return attn_output, attn_weights

        return attn_output
