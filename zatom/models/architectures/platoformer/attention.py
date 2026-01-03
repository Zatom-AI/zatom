"""Adapted from https://github.com/carlosinator/tabasco.
Adapted from https://github.com/niazoys/PlatonicTransformers.
"""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jvp_flash_attention.jvp_attention import JVPAttn
from platonic_transformers.models.platoformer.rope import PlatonicRoPE

from zatom.models.architectures.platoformer import get_platonic_group
from zatom.models.architectures.platoformer.linear import PlatonicLinear
from zatom.models.architectures.platoformer.norm import NormPlatonic
from zatom.models.architectures.transformer.positional_encoder import (
    RotaryPositionalEmbeddings as SequenceRoPE,
)
from zatom.utils.typing_utils import typecheck


def IS_POWER_OF_2(N: int) -> bool:
    """Helper to check whether int N is a positive power of 2."""
    return N > 0 and (N & (N - 1)) == 0


class ModernAttentionPlatonic(nn.Module):
    """More flexible reimplementation of the (dense) Platonic attention layer "PlatonicConv".
    Allows additionally for cross-attention and adds functionality from ModernAttention, e.g.
    sequence RoPE besides Platonic Euclidean RoPE and JVPAttn backend. Implements
    PlatonicTransformers' _forward_dense only, not _forward_graph.

    Args:
        c_in:                   Number of input     channels per group element.
        c_out:                  Number of output    channels per group element.
        c_qk:                   Number of query/key channels per group element and head.
        c_val:                  Number of value     channels per group element and head.
        n_heads:                Number of attention heads per group element (each group element.
                                has its own n_heads heads, i.e. there are G*n_heads total heads).
        solid_name:             String identifying the Platonic solid group.
        freq_sigma_platonic:    Standard deviation for sampling initial Platonic RoPE wave vectors.
                                Optional. None disables Platonic RoPE.
        freq_init_platonic:     Platonic RoPE wave vector init heuristic ("random" or "spiral")
        learned_freqs_platonic: Whether Platonic RoPE wave vectors are static or learned.
        bias:                   Whether the Platonic linear projection maps apply biases or not.
        mean_aggregation:       Normalization of the dynamic convolution kernel in linear_attention.
        linear_attention:       Switches from softmax to linear attention / dynamic conv mode.
                                See the original Platonic transformers paper for more details.
        use_key:                Whether a feature dependent key is used, as usual in attention, or
                                the key is replaced with static torch.ones. The latter is used for
                                dynamic convolution kernels; see the original Platonic transformers
                                paper for more details.
        context_length:         Maximum context length for sequence axis RoPE.
        sequence_rope_base:     Base frequency for sequence axis RoPE.
        use_qk_norm:            Whether to apply RMS normalization to queries and keys.
        qk_norm_per_g:          If False, normalizing over the last axis of size c only.
                                If True, acting on regular rep indices as well. Weight/bias params
                                are still shared over the group axis to preserve equivariance.
        attn_backend:           Softmax attention backend. One of "SDPA", "JVP_ATTN" or "MANUAL".
    """

    @typecheck
    def __init__(
        self,
        c_in: int,  # per group element
        c_out: int,  # per group element
        c_qk: int,  # per group element and head
        c_val: int,  # per group element and head
        n_heads: int,
        ### Platonic attention specific args
        solid_name: str,  # in PLATONIC_GROUPS_3D
        freq_sigma_platonic: Optional[float],
        freq_init_platonic: str = "random",
        learned_freqs_platonic: bool = True,
        bias: bool = False,
        mean_aggregation: bool = False,
        linear_attention: bool = False,
        use_key: bool = False,
        ### Modern attention specific args
        context_length: Optional[int] = 2048,
        sequence_rope_base: Optional[int] = 10_000,
        use_qk_norm: bool = True,
        qk_norm_per_g: bool = True,
        attn_backend: Literal["SDPA", "JVP_ATTN", "MANUAL"] = "SDPA",
    ):
        super().__init__()

        self.group = get_platonic_group(solid_name)
        self.G = G = self.group.G  # number of group elements
        self.H = H = n_heads
        self.c_in = c_in
        self.c_qk = c_qk
        self.c_val = c_val
        self.c_out = c_out

        self.mean_aggregation = mean_aggregation
        self.linear_attention = linear_attention
        self.use_key = use_key
        self.use_qk_norm = use_qk_norm
        self.attn_backend = attn_backend
        assert attn_backend in (
            "SDPA",
            "JVP_ATTN",
            "MANUAL",
        ), f"Unknown attn_backend '{attn_backend}', should be one of 'SDPA', 'JVP_ATTN', 'MANUAL'"

        # Platonic linear projectors
        self.q_proj = PlatonicLinear(c_in, H * c_qk, solid_name, bias=bias)
        self.v_proj = PlatonicLinear(c_in, H * c_val, solid_name, bias=bias)
        self.o_proj = PlatonicLinear(H * c_val, c_out, solid_name, bias=bias)
        # key-projection is optional in Platonic transformers (it might be replaced with torch.ones)
        if freq_sigma_platonic is None or use_key:
            self.k_proj = PlatonicLinear(c_in, H * c_qk, solid_name, bias=bias)
        else:
            self.k_proj = None

        # Query/key normalization
        if self.use_qk_norm:
            self.q_norm = NormPlatonic("RMSNorm", solid_name, H * c_qk, qk_norm_per_g, bias=False)
            self.k_norm = NormPlatonic("RMSNorm", solid_name, H * c_qk, qk_norm_per_g, bias=False)

        # Sequence axis RoPE
        if context_length is not None and sequence_rope_base is not None:
            # Rotary embeddings on half the dims per head
            self.sequence_rope = SequenceRoPE(
                dim=c_qk,
                max_seq_len=context_length,
                base=sequence_rope_base,
            )
        else:
            # NOTE: If either is None, disable sequence RoPE
            self.sequence_rope = None

        # Platonic RoPE for Euclidean coordinates
        if freq_sigma_platonic is not None:
            self.platonic_rope = PlatonicRoPE(
                embed_dim=G * H * c_qk,
                num_heads=H,
                head_dim=c_qk,
                solid_name=solid_name,
                spatial_dims=3,
                freq_sigma=freq_sigma_platonic,
                learned_freqs=learned_freqs_platonic,
                freq_init=freq_init_platonic,
            )
        else:
            self.platonic_rope = None

    @typecheck
    def forward(
        self,
        feat_Q: torch.Tensor,
        coords_Q: torch.Tensor,
        feat_KV: Optional[torch.Tensor] = None,
        coords_KV: Optional[torch.Tensor] = None,
        sequence_idxs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        avg_num_nodes: Optional[float] = 1.0,
    ) -> torch.Tensor:
        """Forward pass through the Platonic group equivariant modern attention layer.

        Args:
            feat_Q:        Query feature tensor,                                    [B, NQ, G*c_in]
            coords_Q:      Query Euclidean coordinates for Platonic RoPE,           [B, NQ, 3]
            feat_KV:       Key/value feature tensor (optional),                     [B, NKV, G*c_in]
                           For cross-attention. If None, set to feat_Q.
            coords_KV:     Key/value Euclidean coords for Platonic RoPE (optional), [B, NKV, 3]
                           For cross-attention. If None, set to coords_Q.
            sequence_idxs: Sequence index tensor for sequence RoPE,                 [B, NQ]
                           Valid only if SequenceRope!=None. For cross-attention
                           this only makes sense if queries/keys/values are
                           are features coming from the same sequence.
            padding_mask:  Boolean mask for padding tokens (True means ignore !!!)  [B, NKV]
            attn_mask:     Attention mask (False or -inf indicates masked entries)  [B, H, NQ, NKV]
            avg_num_nodes: Used to normalize the dynamic convolution kernel if
                           linear_attention is True and no padding mask is passed.

        Returns:
            Projected attention output tensor                                       [B, NQ, G*c_out]
        """
        if self.sequence_rope is not None:
            assert sequence_idxs is not None, "SequenceRope requires sequence_idxs to be passed."

        # Self-attention:  set KV-tensors = Q-tensors
        if feat_KV is None:
            feat_KV = feat_Q
        if coords_KV is None:
            coords_KV = coords_Q

        B, NQ, _ = feat_Q.shape
        NKV = feat_KV.size(1)
        G, H = self.G, self.H
        device = feat_Q.device
        dtype = feat_Q.dtype

        # PlatonicLinear projection to multi-head features
        q = self.q_proj(feat_Q)  # (B, NQ, G*H*c_qk)
        v = self.v_proj(feat_KV)  # (B, NKV, G*H*c_val)
        if self.k_proj is not None:
            k = self.k_proj(feat_KV)  # (B, NKV, G*H*c_qk)
        else:
            k = torch.ones((B, NKV, G * H * self.c_qk), dtype=dtype, device=device)

        # RMS-normalize queries/keys
        if self.use_qk_norm:
            q = self.q_norm(q)  # (B, NQ,  G*H*c_qk)
            k = self.k_norm(k)  # (B, NKV, G*H*c_qk)

        q = q.reshape(B, NQ, G * H, self.c_qk)
        k = k.reshape(B, NKV, G * H, self.c_qk)
        v = v.reshape(B, NKV, G * H, self.c_val)

        # Apply sequence RoPE
        if self.sequence_rope is not None:
            assert NQ == NKV and torch.allclose(
                coords_Q, coords_KV
            ), "sequence_rope is only valid for query/key sequences with corresponding indices."
            q = self.sequence_rope(q, input_pos=sequence_idxs)
            k = self.sequence_rope(k, input_pos=sequence_idxs)

        # Apply Platonic RoPE
        # Note: the order of sequence/Platonic RoPE commutes since SO(2) is Abelian / phases add up
        if self.platonic_rope is not None:
            q = q.view(B, NQ, G, H, self.c_qk)
            k = k.view(B, NKV, G, H, self.c_qk)
            q = self.platonic_rope(q, coords_Q)
            k = self.platonic_rope(k, coords_KV)
            q = q.view(B, NQ, G * H, self.c_qk)
            k = k.view(B, NKV, G * H, self.c_qk)

        if padding_mask is not None:
            # ~ for zatom's inverse padding mask convention
            padding_mask = ~padding_mask[:, None, None, :]  # (B, 1, 1, NKV)

            if attn_mask is None:
                final_mask = padding_mask

            elif attn_mask.is_floating_point():
                # Create a float mask: 0.0 where we keep, -inf where we mask
                padding_mask_float = torch.zeros((B, 1, 1, NKV), device=device, dtype=dtype)
                padding_mask_float.masked_fill_(~padding_mask, float("-inf"))
                final_mask = attn_mask + padding_mask_float

            else:  # boolean attention mask
                final_mask = attn_mask & padding_mask

            # .expand leads to shape (B,H,NQ,NKV) but keeps the underlying data tensor with
            # stride (NKV,0,0,1). This is required since attention kernels don't broadcast
            # automatically. It is more memory efficient than using e.g. .repeat, which
            # materializes the tensor explicitly, leading to infeasible O(N^2) memory IO in
            # the attention kernel.
            # Note: expansion needs to be done after logical negation ~ to avoid materialization!
            final_mask = final_mask.expand(B, H, NQ, NKV)

        elif attn_mask is not None:
            final_mask = attn_mask  # (B, H, NQ, NKV)

        else:
            final_mask = None

        ### Softmax attention
        if not self.linear_attention:
            q = q.transpose(1, 2)  # (B, G*H, NQ,  c_qk)
            k = k.transpose(1, 2)  # (B, G*H, NKV, c_qk)
            v = v.transpose(1, 2)  # (B, G*H, NKV, c_val)

            # Expand final_mask for broadcasting over group elements.
            # If the stride of the H axis is zero (expanded padding_mask), reshape is a view and the
            # G*H axis has stride zero as well, which makes the implementation memory IO efficient.
            if final_mask is not None:
                final_mask = final_mask.unsqueeze(1).expand(B, G, H, NQ, NKV)
                final_mask = final_mask.reshape(B, G * H, NQ, NKV)
                # Sanity check:  efficient strides for pure padding_mask
                if attn_mask is None:
                    # Ensure final_mask has efficient strides: (NKV, 0, 0, 1)
                    if final_mask.stride() != (NKV, 0, 0, 1):
                        # Use .as_strided to create a view with the desired strides
                        final_mask = final_mask.as_strided(
                            size=(B, G * H, NQ, NKV),
                            stride=(NKV, 0, 0, 1),
                        )
                    assert final_mask.stride() == (NKV, 0, 0, 1), (
                        "final_mask has inefficient strides, "
                        "potentially leading to O(N^2) memory IO in attention kernel."
                    )

            if self.attn_backend == "SDPA":
                # Use PyTorch's optimized scaled dot-product attention (SDPA)
                output = F.scaled_dot_product_attention(q, k, v, attn_mask=final_mask)

            elif self.attn_backend == "JVP_ATTN":
                assert IS_POWER_OF_2(NQ) and IS_POWER_OF_2(
                    NKV
                ), "'JVP_ATTN' backend currently requires NQ and NKV to be powers of 2"
                assert (
                    self.c_qk == self.c_val
                ), "'JVP_ATTN' backend currently requires c_qk == c_val"
                output = JVPAttn.fwd_dual(q, k, v, attn_mask=final_mask)

            elif self.attn_backend == "MANUAL":
                # Use manual implementation for comparison or environments where SDPA is not available
                scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.c_qk)
                if final_mask is not None and final_mask.is_floating_point():
                    scores = scores + final_mask
                elif final_mask is not None:
                    scores = scores.masked_fill(~final_mask, float("-inf"))
                scores = F.softmax(scores.float(), dim=-1).type_as(q)
                output = torch.matmul(scores, v)

            else:
                raise ValueError(f"Unknown attn_backend '{self.attn_backend}'")

            output = output.transpose(1, 2).flatten(-2, -1)  # (B, NQ, G*H*c_val)

        ### Linear attention  (dynamic convolution)
        else:
            q = q.view(B, NQ, G, H, self.c_qk)
            k = k.view(B, NKV, G, H, self.c_qk)
            v = v.view(B, NKV, G, H, self.c_val)

            if final_mask is not None:
                assert (
                    not final_mask.is_floating_point()
                ), "final_mask needs to be boolean if linear_attention is True"
                assert (final_mask.size(2) == 1) or (
                    final_mask.stride(2) == 0
                ), "Linear attention currently doesn't support query-dependent masking"

                final_mask = final_mask[:, :, 0, :].transpose(1, 2)  # (B, NKV, H)
                k = k * final_mask[:, :, None, :, None]  # (B, NKV, G, H, c_qk)
                v = v * final_mask[:, :, None, :, None]  # (B, NKV, G, H, c_val)

            kv_conv_kernel = torch.einsum("bnghc,bnghd->bghcd", k, v)

            if self.mean_aggregation and padding_mask is not None:
                # Note: inverted ~padding_mask above
                num_nodes = padding_mask.sum(dim=(1, 2, 3)).float().view(B, 1, 1, 1, 1)
                assert torch.all(num_nodes > 0)
            else:
                num_nodes = avg_num_nodes
            kv_conv_kernel = kv_conv_kernel / num_nodes

            output = torch.einsum("bnghc,bghcd->bnghd", q, kv_conv_kernel)  # (B, NQ, G, H, c_val)
            output = output.flatten(-3, -1)  # (B, NQ, G*H*c_val)

        return self.o_proj(output)  # (B, NQ, G*c_out)
