"""Adapted from https://github.com/carlosinator/tabasco."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from zatom.models.architectures.transformer.common import apply_rotary_embeddings
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


class ModernSelfAttention(nn.Module):
    """Modern self-attention module with rotary embeddings and optional SDPA.

    Args:
        dim: Input dimension
        n_heads: Number of attention heads
        use_qk_norm: Whether to apply RMS normalization to queries and keys
        use_sdpa: Whether to use PyTorch's scaled dot-product attention
    """

    @typecheck
    def __init__(self, dim: int, n_heads: int, use_qk_norm: bool = True, use_sdpa: bool = True):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_sdpa = use_sdpa

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

    @typecheck
    def forward(
        self, x: torch.Tensor, freqs_complex: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the modern self-attention layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            freqs_complex: Precomputed rotary frequency tensor of shape [seq_len, head_dim]
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary_embeddings(q, freqs_complex, device=x.device)
        k = apply_rotary_embeddings(k, freqs_complex, device=x.device)

        # Transpose for attention calculation: (B, H, Seq_Len, Head_Dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_sdpa:
            # Use PyTorch's optimized scaled dot-product attention.
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # Manual implementation for comparison or environments where SDPA is not available.
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    # For boolean masks, False positions are masked.
                    # We need to invert the mask if True means "keep".
                    scores.masked_fill_(~attn_mask, float("-inf"))
                else:
                    # For float masks, we directly add it.
                    scores = scores + attn_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            output = torch.matmul(scores, v)

        # (B, H, Seq_Len, Head_Dim) -> (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, Dim)
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.dim)
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
