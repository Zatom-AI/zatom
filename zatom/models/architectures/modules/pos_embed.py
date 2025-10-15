"""Positional embedding layers.

Adapted from:
    - https://github.com/apple/ml-simplefold
"""

import math
from typing import Tuple

import torch
from torch import Tensor, nn

from zatom.utils.typing_utils import typecheck

#################################################################################
#                               Misc. Utilities                                 #
#################################################################################


@typecheck
def compute_axial_cis(
    ts: torch.Tensor,
    in_dim: int,
    dim: int,
    theta: float = 100.0,
) -> torch.Tensor:
    """Compute the rotary positional embeddings for axial inputs.

    Args:
        ts: The input position tensor of shape (B, N, in_dim).
        in_dim: The dimension of the input position.
        dim: The dimension of the output position embedding. Must be divisible by 2 * in_dim.
        theta: The base frequency for the rotary embeddings.

    Returns:
        The output tensor of shape (B, N, dim).
    """
    B, N, _ = ts.shape
    freqs_all = []
    interval = 2 * in_dim
    for i in range(in_dim):
        freq = 1.0 / (
            theta
            ** (
                torch.arange(0, dim, interval, device=ts.device)[: (dim // interval)].float() / dim
            )
        )
        t = ts[..., i].flatten()
        freq_i = torch.outer(t, freq)
        freq_cis_i = torch.polar(torch.ones_like(freq_i), freq_i)
        freq_cis_i = freq_cis_i.view(B, N, -1)
        freqs_all.append(freq_cis_i)
    freqs_cis = torch.cat(freqs_all, dim=-1)
    return freqs_cis


@typecheck
def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors.

    Args:
        xq: The query tensor of shape (B, H, N, D).
        xk: The key tensor of shape (B, H, N, D).
        freqs_cis: The rotary positional embeddings of shape (B, 1, N, D).

    Returns:
        A tuple containing the modified query and key tensors, each of shape (B, H, N, D).
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


#################################################################################
#                               Embedding Layers                                #
#################################################################################


class AbsolutePositionEncoding(nn.Module):
    """Absolute position encoding layer.

    Args:
        in_dim: The dimension of the input position.
        embed_dim: The dimension of the output position embedding. Must be divisible by in_dim.
        include_input: Whether to include the original input position in the output embedding.
    """

    def __init__(self, in_dim: int, embed_dim: int, include_input: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = embed_dim
        self.include_input = include_input

        assert (
            embed_dim % in_dim == 0
        ), f"The `embed_dim` ({embed_dim}) argument must be divisible by `in_dim` ({in_dim})."
        self.embed_dim = embed_dim + in_dim if include_input else embed_dim

    @typecheck
    def forward(self, pos: Tensor) -> Tensor:
        """Get the positional encoding for each coordinate.

        Args:
            pos: The input position tensor of shape (*, in_dim).

        Returns:
            The output position embedding tensor of shape (*, embed_dim).
        """
        pos_embs = []
        for i in range(self.in_dim):
            pe = self.get_1d_pos_embed(pos[..., i])
            pos_embs.append(pe)
        if self.include_input:
            pos_embs.append(pos)
        pos_embs = torch.cat(pos_embs, dim=-1)
        return pos_embs

    @typecheck
    def get_1d_pos_embed(self, pos: Tensor) -> Tensor:
        """Get 1D sine-cosine positional embeddings.

        From https://github.com/facebookresearch/DiT/blob/main/models.py#L303.

        Args:
            pos: (M,) or (N, M), where N is batch size, M is number of positions.

        Returns:
            emb: (M, D) or (N, M, D), where D is embedding dimension per position.
        """
        embed_dim = self.hidden_dim // (self.in_dim * 2)
        omega = 2 ** torch.linspace(0, math.log(224, 2) - 1, embed_dim).to(pos.device)
        omega *= torch.pi

        if len(pos.shape) == 1:
            out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        elif len(pos.shape) == 2:
            out = torch.einsum("nm,d->nmd", pos, omega)

        emb_sin = torch.sin(out)  # (*, M, D/2)
        emb_cos = torch.cos(out)  # (*, M, D/2)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (*, M, D)
        return emb


class FourierPositionEncoding(torch.nn.Module):
    """Fourier feature positional encoding layer.

    Args:
        in_dim: The dimension of the input position.
        include_input: Whether to include the original input position in the output embedding.
        min_freq_log2: The minimum frequency in log2 scale.
        max_freq_log2: The maximum frequency in log2 scale.
        num_freqs: The number of frequency bands.
        log_sampling: Whether to use logarithmic sampling of frequency bands.
    """

    def __init__(
        self,
        in_dim: int,
        include_input: bool = False,
        min_freq_log2: float = 0,
        max_freq_log2: float = 12,
        num_freqs: int = 32,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.include_input = include_input
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """Create the frequency bands and compute the output embedding dimension."""
        d = self.in_dim
        dim_out = 0
        if self.include_input:
            dim_out += d

        min_freq = self.min_freq_log2
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(min_freq, max_freq, steps=N_freqs)  # (nf,)
        else:
            freq_bands = torch.linspace(2.0**min_freq, 2.0**max_freq, steps=N_freqs)  # (nf,)

        assert (
            freq_bands.isfinite().all()
        ), f"freq_bands NaNs: {freq_bands.isnan().any()}, Infs: {freq_bands.isinf().any()}"

        self.register_buffer("freq_bands", freq_bands)  # (nf,)
        self.embed_dim = dim_out + d * self.freq_bands.numel() * 2

    @typecheck
    def forward(
        self,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Get the positional encoding for each coordinate.

        Args:
            pos: The input position tensor of shape (*, in_dim).

        Returns:
            The output position embedding tensor of shape (*, embed_dim).
        """
        out = []
        if self.include_input:
            out = [pos]  # (*, in_dim)

        pos = pos.unsqueeze(-1) * self.freq_bands  # (*b, d, nf)

        out += [
            torch.sin(pos).flatten(start_dim=-2),  # (*b, d*nf)
            torch.cos(pos).flatten(start_dim=-2),  # (*b, d*nf)
        ]

        out = torch.cat(out, dim=-1)  # (*b, 2 * in_dim * nf (+ in_dim))
        return out


class AxialRotaryPositionEncoding(nn.Module):
    """Axial rotary positional encoding layer.

    Args:
        in_dim: The dimension of the input position.
        embed_dim: The dimension of the output position embedding. Must be divisible by num_heads.
        num_heads: The number of attention heads.
        base: The base frequency for the rotary embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_heads: int,
        base: float = 100.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim // num_heads
        self.base = base

    @typecheck
    def forward(
        self, xq: torch.Tensor, xk: torch.Tensor, pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply axial rotary positional embeddings to query and key tensors.

        Args:
            xq: The query tensor of shape (B, H, N, D).
            xk: The key tensor of shape (B, H, N, D).
            pos: The input position tensor of shape (B, N, in_dim).

        Returns:
            A tuple containing the modified query and key tensors, each of shape (B, H, N, D).
        """
        if pos.ndim == 2:
            pos = pos.unsqueeze(-1)
        freqs_cis = compute_axial_cis(pos, self.in_dim, self.embed_dim, self.base)
        freqs_cis = freqs_cis.unsqueeze(1)
        return apply_rotary_emb(xq, xk, freqs_cis.to(xq.device))
