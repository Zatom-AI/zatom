"""Adapted from https://github.com/carlosinator/tabasco."""

from types import SimpleNamespace
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from zatom.models.architectures.platoformer.attention_platonic import (
    ModernAttentionPlatonic,
)
from zatom.models.architectures.platoformer.feedfwd_platonic import (
    SwiGLUFeedForwardPlatonic,
)
from zatom.models.architectures.platoformer.norm_platonic import NormPlatonic
from zatom.utils.typing_utils import typecheck


class ModernTransformerBlockPlatonic(nn.Module):
    """A self-attention transformer block using ModernAttentionPlatonic and SwiGLUFeedForward.

    Args:
        c_model:                Number of in/output channels per group element.
        c_qk:                   Number of query/key channels per group element and head.
        c_val:                  Number of value     channels per group element and head.
        n_heads:                Number of attention heads per group element (each group element.
                                has its own n_heads heads, i.e. there are G*n_heads total heads).
        solid_name:             String identifying the Platonic solid group.
        freq_sigma_platonic:    Standard deviation for sampling initial Platonic RoPE wave vectors.
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
        qk_layernorm:           Whether to apply RMS normalization to queries and keys.
        qk_norm_per_g:          If False, normalizing over the last axis of size c only.
                                If True, acting on regular rep indices as well. Weight/bias params
                                are still shared over the group axis to preserve equivariance.
        attn_backend:           Softmax attention backend. One of "SDPA", "JVP_ATTN" or "MANUAL".
        normalize_per_g:        If False, RMS normalizing over the last axis of size c_model only.
                                If True, acting on the group axis as well.
        norm_elementwise_affine: Enables learnable weights for Platonic RMSNorm.
    """

    @typecheck
    def __init__(
        self,
        c_model: int,  # per group element
        c_qk: int,  # per group element and head
        c_val: int,  # per group element and head
        n_heads: int,
        ### Platonic attention specific args
        solid_name: str,
        freq_sigma_platonic: float,
        freq_init_platonic: str = "random",
        learned_freqs_platonic: bool = True,
        bias: bool = False,
        mean_aggregation: bool = False,
        linear_attention: bool = False,
        use_key: bool = False,
        ### Modern attention specific args
        context_length: Optional[int] = 2048,
        sequence_rope_base: Optional[int] = 10_000,
        qk_layernorm: bool = True,
        qk_norm_per_g: bool = True,  # TODO: test what works best here
        attn_backend: Literal["SDPA", "JVP_ATTN", "MANUAL"] = "SDPA",
        ### Normalization args
        normalize_per_g: bool = True,  # TODO: test what works best here
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()

        self.attention = ModernAttentionPlatonic(
            c_in=c_model,
            c_out=c_model,
            c_qk=c_qk,
            c_val=c_val,
            n_heads=n_heads,
            solid_name=solid_name,
            freq_sigma_platonic=freq_sigma_platonic,
            freq_init_platonic=freq_init_platonic,
            learned_freqs_platonic=learned_freqs_platonic,
            bias=bias,
            mean_aggregation=mean_aggregation,
            linear_attention=linear_attention,
            use_key=use_key,
            context_length=context_length,
            sequence_rope_base=sequence_rope_base,
            use_qk_norm=qk_layernorm,
            qk_norm_per_g=qk_norm_per_g,
            attn_backend=attn_backend,
        )

        self.feed_forward = SwiGLUFeedForwardPlatonic(
            c_in=c_model,
            c_hid=4 * c_model,
            c_out=c_model,
            solid_name=solid_name,
        )

        norm_kwargs = vars(
            SimpleNamespace(
                mode="RMSNorm",
                solid_name=solid_name,
                c=c_model,
                normalize_per_g=normalize_per_g,
                elementwise_affine=norm_elementwise_affine,
                bias=False,
            )
        )
        self.norm_attn = NormPlatonic(**norm_kwargs)
        self.norm_ffwd = NormPlatonic(**norm_kwargs)

    @typecheck
    def forward(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
        sequence_idxs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        avg_num_nodes: Optional[float] = 1.0,
    ) -> torch.Tensor:
        """Forward pass through the Platonic modern transformer block.

        Args:
            feat:          Input feature tensor                                     [B, N, c_model]
            coords:        Euclidean coordinates tensor                             [B, N, 3]
            sequence_idxs: Sequence indices for sequence index RoPe                 [B, N]
            padding_mask:  Boolean mask for padding tokens (True means masked)      [B, N]
            attn_mask:     Attention mask (False / -inf means masked)               [B, H, N, N]
            avg_num_nodes: Used to normalize the dynamic convolution kernel if
                           linear_attention is True and no padding mask is passed.

        Returns:
            Output feature tensor                                                   [B, N, c_model]
        """
        h = self.norm_attn(feat)
        h = self.attention(
            feat_Q=h,  # used at feat_KV as well
            coords_Q=coords,  # used as coords_KV as well
            sequence_idxs=sequence_idxs,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
            avg_num_nodes=avg_num_nodes,
        )
        feat = feat + h

        h = self.norm_ffwd(feat)
        h = self.feed_forward(h)
        feat = feat + h
        return feat


class ModernTransformerDecoderBlockPlatonic(nn.Module):
    """A transformer decoder block using ModernAttentionPlatonic and SwiGLUFeedForward.

    Can be used as a Platonic group equivariant drop-in replacement for
    `nn.TransformerDecoderLayer`.

    Args:
        c_model:                Number of in/output channels per group element.
        c_qk:                   Number of query/key channels per group element and head.
        c_val:                  Number of value     channels per group element and head.
        n_heads:                Number of attention heads per group element (each group element.
                                has its own n_heads heads, i.e. there are G*n_heads total heads).
        solid_name:             String identifying the Platonic solid group.
        freq_sigma_platonic:    Standard deviation for sampling initial Platonic RoPE wave vectors.
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
        qk_layernorm:           Whether to apply RMS normalization to queries and keys.
        qk_norm_per_g:          If False, normalizing over the last axis of size c only.
                                If True, acting on regular rep indices as well. Weight/bias params
                                are still shared over the group axis to preserve equivariance.
        attn_backend:           Softmax attention backend. One of "SDPA", "JVP_ATTN" or "MANUAL".
        normalize_per_g:        If False, RMS normalizing over the last axis of size c_model only.
                                If True, acting on the group axis as well.
        norm_elementwise_affine: Enables learnable weights for Platonic RMSNorm.
    """

    @typecheck
    def __init__(
        self,
        c_model: int,  # per group element
        c_qk: int,  # per group element and head
        c_val: int,  # per group element and head
        n_heads: int,
        ### Platonic attention specific args
        solid_name: str,
        freq_sigma_platonic: float,
        freq_init_platonic: str = "random",
        learned_freqs_platonic: bool = True,
        bias: bool = False,
        mean_aggregation: bool = False,
        linear_attention: bool = False,
        use_key: bool = False,
        ### Modern attention specific args
        context_length: Optional[int] = 2048,
        sequence_rope_base: Optional[int] = 10_000,
        use_sequence_rope_cross: bool = False,
        qk_layernorm: bool = True,
        qk_norm_per_g: bool = True,  # TODO: test what works best here
        attn_backend: Literal["SDPA", "JVP_ATTN", "MANUAL"] = "SDPA",
        ### Normalization args
        normalize_per_g: bool = True,  # TODO: test what works best here
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()

        self.use_sequence_rope_cross = use_sequence_rope_cross
        attn_kwargs = vars(
            SimpleNamespace(
                c_in=c_model,
                c_out=c_model,
                c_qk=c_qk,
                c_val=c_val,
                n_heads=n_heads,
                solid_name=solid_name,
                freq_sigma_platonic=freq_sigma_platonic,
                freq_init_platonic=freq_init_platonic,
                learned_freqs_platonic=learned_freqs_platonic,
                bias=bias,
                mean_aggregation=mean_aggregation,
                linear_attention=linear_attention,
                use_key=use_key,
                use_qk_norm=qk_layernorm,
                qk_norm_per_g=qk_norm_per_g,
                attn_backend=attn_backend,
            )
        )
        self.attn_self = ModernAttentionPlatonic(
            **attn_kwargs,
            context_length=context_length,
            sequence_rope_base=sequence_rope_base,
        )
        self.attn_cross = ModernAttentionPlatonic(
            **attn_kwargs,
            context_length=context_length if use_sequence_rope_cross else None,
            sequence_rope_base=sequence_rope_base if use_sequence_rope_cross else None,
        )

        self.feed_forward = SwiGLUFeedForwardPlatonic(
            c_in=c_model,
            c_hid=4 * c_model,
            c_out=c_model,
            solid_name=solid_name,
        )

        norm_kwargs = vars(
            SimpleNamespace(
                mode="RMSNorm",
                solid_name=solid_name,
                c=c_model,
                normalize_per_g=normalize_per_g,
                elementwise_affine=norm_elementwise_affine,
                bias=False,
            )
        )
        self.norm_self = NormPlatonic(**norm_kwargs)
        self.norm_cross = NormPlatonic(**norm_kwargs)
        self.norm_ffwd = NormPlatonic(**norm_kwargs)

    @typecheck
    def forward(
        self,
        feat: torch.Tensor,
        memory: torch.Tensor,
        coords_feat: torch.Tensor,
        coords_mem: torch.Tensor,
        sequence_idxs: Optional[torch.Tensor] = None,
        padding_mask_feat: Optional[torch.Tensor] = None,
        padding_mask_mem: Optional[torch.Tensor] = None,
        attn_mask_self: Optional[torch.Tensor] = None,
        attn_mask_cross: Optional[torch.Tensor] = None,
        avg_num_nodes_self: Optional[float] = 1.0,
        avg_num_nodes_cross: Optional[float] = 1.0,
    ) -> torch.Tensor:
        """Forward pass through the Platonic modern transformer decoder block.

        Args:
            feat:                Input feature tensor for self-attention             [B, N, c_model]
            memory:              Memory tensor in cross-attention layer              [B, M, c_model]
            coords_feat:         Euclidean coordinates for feat                      [B, N, 3]
            coords_mem:          Euclidean coordinates for memory                    [B, M, 3]
            sequence_idxs:       Sequence indices for sequence index RoPe            [B, N]
            padding_mask_feat:   Boolean mask for padding tokens (True => masked)    [B, N]
            padding_mask_mem:    Boolean mask for padding tokens (True => masked)    [B, M]
            attn_mask_self:      Attention mask (False/-inf => masked)               [B, H, N, N]
            attn_mask_cross:     Attention mask (False/-inf => masked)               [B, H, N, M]
            avg_num_nodes_self:  Used to normalize the dynamic convolution kernel in self-attention
                                 if linear_attention is True and no padding mask is passed.
            avg_num_nodes_cross: Used to normalize the dynamic convolution kernel in cross-attention
                                 if linear_attention is True and no padding mask is passed.

        Returns:
            Output feature tensor                                                   [B, N, c_model]
        """
        # Self-attention
        h = self.norm_self(feat)
        h = self.attn_self(
            feat_Q=h,  # used at feat_KV as well
            coords_Q=coords_feat,  # used as coords_KV as well
            sequence_idxs=sequence_idxs,
            padding_mask=padding_mask_feat,
            attn_mask=attn_mask_self,
            avg_num_nodes=avg_num_nodes_self,
        )
        feat = feat + h

        # Cross-attention
        h = self.norm_cross(feat)
        h = self.attn_cross(
            feat_Q=h,
            feat_KV=memory,
            coords_Q=coords_feat,
            coords_KV=coords_mem,
            sequence_idxs=sequence_idxs if self.use_sequence_rope_cross else None,
            padding_mask=padding_mask_mem,
            attn_mask=attn_mask_cross,
            avg_num_nodes=avg_num_nodes_cross,
        )
        feat = feat + h

        # Feed-forward network
        h = self.norm_ffwd(feat)
        h = self.feed_forward(h)
        feat = feat + h
        return feat


class ModernTransformerPlatonic(nn.Module):
    """A modern Platonic Transformer model with optional sequence index RoPE and Platonic RoPE
    embeddings, query-key normalization, and scaled dot-product attention.

    Args:
        c_model:                Number of in/output channels per group element.
        c_qk:                   Number of query/key channels per group element and head.
        c_val:                  Number of value     channels per group element and head.
        n_heads:                Number of attention heads per group element (each group element.
                                has its own n_heads heads, i.e. there are G*n_heads total heads).
        depth:                  Number of Platonic transformer blocks.
        solid_name:             String identifying the Platonic solid group.
        freq_sigma_platonic:    Standard deviation for sampling initial Platonic RoPE wave vectors.
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
        qk_layernorm:           Whether to apply RMS normalization to queries and keys.
        qk_norm_per_g:          If False, normalizing over the last axis of size c only.
                                If True, acting on regular rep indices as well. Weight/bias params
                                are still shared over the group axis to preserve equivariance.
        attn_backend:           Softmax attention backend. One of "SDPA", "JVP_ATTN" or "MANUAL".
        normalize_per_g:        If False, RMS normalizing over the last axis of size c_model only.
                                If True, acting on the group axis as well.
        elementwise_affine:     Enables learnable weights for Platonic RMSNorm.
        repr_layer:             Layer at which to additionally extract intermediate representations.
                                If None, no intermediate representation is extracted.
        is_decoder:             Whether the transformer is used as a (cross-attention) decoder.
    """

    @typecheck
    def __init__(
        self,
        c_model: int,  # per group element
        c_qk: int,  # per group element and head
        c_val: int,  # per group element and head
        n_heads: int,
        depth: int,
        ### Platonic attention specific args
        solid_name: str,
        freq_sigma_platonic: float,  # left explicit to force user to think about this choice
        freq_init_platonic: str = "random",
        learned_freqs_platonic: bool = True,
        bias: bool = False,
        mean_aggregation: bool = False,
        linear_attention: bool = False,
        use_key: bool = False,
        ### Modern attention specific args
        context_length: Optional[int] = 2048,
        sequence_rope_base: Optional[int] = 10_000,
        qk_layernorm: bool = True,
        qk_norm_per_g: bool = True,  # TODO: test what works best here
        attn_backend: Literal["SDPA", "JVP_ATTN", "MANUAL"] = "SDPA",
        ### Normalization args
        normalize_per_g: bool = True,  # TODO: test what works best here
        norm_elementwise_affine: bool = True,
        ###
        repr_layer: Optional[int] = None,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.c_model = c_model
        self.depth = depth
        self.n_heads = n_heads
        self.max_seq_len = context_length
        self.repr_layer = repr_layer
        self.is_decoder = is_decoder

        layer_class = (
            ModernTransformerDecoderBlockPlatonic if is_decoder else ModernTransformerBlockPlatonic
        )
        self.layers = nn.ModuleList(
            [
                layer_class(
                    c_model=c_model,
                    c_qk=c_qk,
                    c_val=c_val,
                    n_heads=n_heads,
                    solid_name=solid_name,
                    freq_sigma_platonic=freq_sigma_platonic,
                    freq_init_platonic=freq_init_platonic,
                    learned_freqs_platonic=learned_freqs_platonic,
                    bias=bias,
                    mean_aggregation=mean_aggregation,
                    linear_attention=linear_attention,
                    use_key=use_key,
                    context_length=context_length,
                    sequence_rope_base=sequence_rope_base,
                    qk_layernorm=qk_layernorm,
                    qk_norm_per_g=qk_norm_per_g,
                    attn_backend=attn_backend,
                    normalize_per_g=normalize_per_g,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(depth)
            ]
        )

        norm_kwargs = vars(
            SimpleNamespace(
                mode="RMSNorm",
                solid_name=solid_name,
                c=c_model,
                normalize_per_g=normalize_per_g,
                elementwise_affine=norm_elementwise_affine,
                bias=False,
            )
        )
        if repr_layer is not None:
            self.norm_repr = NormPlatonic(**norm_kwargs)
        self.norm_final = NormPlatonic(**norm_kwargs)

    @typecheck
    def forward(
        self,
        feat: torch.Tensor,
        coords_feat: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        coords_mem: Optional[torch.Tensor] = None,
        sequence_idxs: Optional[torch.Tensor] = None,
        padding_mask_feat: Optional[torch.Tensor] = None,
        padding_mask_mem: Optional[torch.Tensor] = None,
        attn_mask_self: Optional[torch.Tensor] = None,
        attn_mask_cross: Optional[torch.Tensor] = None,
        avg_num_nodes_self: Optional[float] = 1.0,
        avg_num_nodes_cross: Optional[float] = 1.0,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Platonic modern transformer.

        Args:
            feat:               Input feature tensor for self-attention              [B, N, c_model]
            coords_feat:        Euclidean coordinates for feat                       [B, N, 3]
            memory:             Memory tensor for cross-attention if is_decoder      [B, M, c_model]
            coords_mem:         Coordinates for memory if is_decoder                 [B, M, 3]
            sequence_idxs:      Sequence indices for sequence index RoPe             [B, N]
            padding_mask_feat:  Mask for padding features (True => masked)           [B, N]
            padding_mask_mem:   Mask for padding memory tokens (True => masked)      [B, M]
            attn_mask_self:     Mask for self-attention layers                       [B, H, N, N]
            attn_mask_cross:    Mask for cross-attention layers                      [B, H, N, M]
            avg_num_nodes_self: Used to normalize the dynamic convolution kernel if
                                linear_attention is True and no padding mask is passed.
            avg_num_nodes_cross: Like avg_num_nodes_self, but for cross-attn if is_decoder is True.

        Returns:
            Output feature tensor                                                    [B, N, c_model]
            or Tuple[output tensor, intermediate repr] if `repr_layer` is set.
        """
        _, N, _ = feat.shape
        assert (
            self.max_seq_len is None or N <= self.max_seq_len
        ), "Sequence length exceeds model's maximum sequence length"

        if not self.is_decoder:
            assert (
                (memory is None)
                and (coords_mem is None)
                and (padding_mask_mem is None)
                and (attn_mask_cross is None)
                and (avg_num_nodes_cross == 1.0)
            ), "The inputs are invalid if is_decoder is False."

        repr = None
        for i, layer in enumerate(self.layers):

            if self.is_decoder:
                feat = layer(
                    feat=feat,
                    memory=memory,
                    coords_feat=coords_feat,
                    coords_mem=coords_mem,
                    sequence_idxs=sequence_idxs,
                    padding_mask_feat=padding_mask_feat,
                    padding_mask_mem=padding_mask_mem,
                    attn_mask_self=attn_mask_self,
                    attn_mask_cross=attn_mask_cross,
                    avg_num_nodes_self=avg_num_nodes_self,
                    avg_num_nodes_cross=avg_num_nodes_cross,
                )
            else:
                feat = layer(
                    feat=feat,
                    coords=coords_feat,
                    sequence_idxs=sequence_idxs,
                    padding_mask=padding_mask_feat,
                    attn_mask=attn_mask_self,
                    avg_num_nodes=avg_num_nodes_self,
                )

            if self.repr_layer is not None and i == self.repr_layer:
                repr = feat.clone()

        feat = self.norm_final(feat)

        if self.repr_layer is None:
            return feat

        else:
            assert (
                repr is not None
            ), f"The specified `repr_layer` ({self.repr_layer}) was not reached during the forward pass."
            repr = self.norm_repr(repr)

            return feat, repr
