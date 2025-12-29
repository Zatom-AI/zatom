import itertools
import unittest

import torch
import torch.nn as nn
from tqdm import tqdm

from zatom.models.architectures.platoformer import (
    PLATONIC_GROUPS_3D,
    get_platonic_group,
)
from zatom.models.architectures.platoformer.attention import (
    ModernAttentionPlatonic,
)


class TestModernAttentionPlatonic(unittest.TestCase):
    """Unit tests for ModernAttentionPlatonic."""

    def test_equivariance_modern_attention_platonic_sdpa_backend(self):
        """Equivariance unit tests for ModernAttentionPlatonic - torch sdpa backend."""
        self._test_equivariance_modern_attention_platonic(attn_backend="SDPA")

    def test_equivariance_modern_attention_platonic_jvp_backend(self):
        """Equivariance unit tests for ModernAttentionPlatonic - JVPAttn backend."""
        self._test_equivariance_modern_attention_platonic(attn_backend="JVP_ATTN")

    def test_equivariance_modern_attention_platonic_manual_backend(self):
        """Equivariance unit tests for ModernAttentionPlatonic - manual backend."""
        self._test_equivariance_modern_attention_platonic(attn_backend="MANUAL")

    def _test_equivariance_modern_attention_platonic(self, attn_backend):
        """Equivariance unit tests for ModernAttentionPlatonic."""

        c_in, c_out = 12, 24
        c_qk = c_val = 32  # JVP_ATTN backend requires equal c_qk = c_val, power of 2 and >=32
        NQ = NKV = 32  # JVP_ATTN backend has high error if NQ,NKV are not powers of 2
        B, H = 2, 4
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        for solid_name in tqdm(PLATONIC_GROUPS_3D, desc="Groups  ", position=0):

            group = get_platonic_group(solid_name)
            G = group.G

            # Random input tensors
            feat_Q = torch.randn((B, NQ, G, c_in), device=device, dtype=dtype)
            feat_KV = torch.randn((B, NKV, G, c_in), device=device, dtype=dtype)
            coords_Q = torch.randn((B, NQ, 3), device=device, dtype=dtype)
            coords_KV = torch.randn((B, NKV, 3), device=device, dtype=dtype)

            sequence_lens_Q = torch.randint(NQ // 3, NQ, (B,), device=device)
            sequence_lens_KV = torch.randint(NKV // 3, NKV, (B,), device=device)
            sequence_lens_Q[B // 2] = NQ
            sequence_lens_KV[B // 2] = NKV
            padding_mask_Q = ~(torch.arange(NQ, device=device)[None, :] < sequence_lens_Q[:, None])
            padding_mask_KV = ~(
                torch.arange(NKV, device=device)[None, :] < sequence_lens_KV[:, None]
            )
            avg_num_nodes = (~padding_mask_Q).float().sum(1).mean(0)

            attn_mask_self = torch.randn((B, H, NQ, NQ), device=device, dtype=dtype) < 1.25
            attn_mask_cross = torch.randn((B, H, NQ, NKV), device=device, dtype=dtype) < 1.25
            sequence_idxs_Q = torch.arange(NQ, device=device).repeat(
                B, 1
            )  # (B,NQ), only for self-attention

            # Loop over grid of all setting combinations that change the logic
            settings_loop = list(
                itertools.product(
                    (1.234, None),  # freq_sigma_platonic
                    (False, True),  # linear_attention
                    (False, True),  # use_key
                    (False, True),  # use_qk_norm
                    (False, True),  # use_sequence_rope
                )
            )
            for (
                freq_sigma_platonic,
                linear_attention,
                use_key,
                use_qk_norm,
                use_sequence_rope,
            ) in tqdm(settings_loop, desc="Settings", position=1, leave=False):

                if use_sequence_rope:
                    ctx_len, seq_rope_base = 2048, 10_000
                else:
                    ctx_len, seq_rope_base = None, None

                attn_module = ModernAttentionPlatonic(
                    c_in,
                    c_out,
                    c_qk,
                    c_val,
                    H,
                    solid_name,
                    freq_sigma_platonic=freq_sigma_platonic,
                    mean_aggregation=True,
                    linear_attention=linear_attention,
                    use_key=use_key,
                    context_length=ctx_len,
                    sequence_rope_base=seq_rope_base,
                    use_qk_norm=use_qk_norm,
                    attn_backend=attn_backend,
                ).to(device, dtype)

                # Loop over grid of all valid input tensor combinations
                for use_attn_mask in (False, True):
                    for use_padding_mask in (False, True):
                        for cross_attn in (False, True):

                            # Self attention
                            if not cross_attn:
                                feat_KV_ = None
                                coords_KV_ = None
                                padding_mask_KV_ = padding_mask_Q if use_padding_mask else None
                                attn_mask = attn_mask_self if use_attn_mask else None
                            # Cross-attention
                            else:
                                # No sequence RoPE, can use arbitrary KV features
                                if not use_sequence_rope:
                                    feat_KV_ = feat_KV
                                    coords_KV_ = coords_KV
                                    padding_mask_KV_ = (
                                        padding_mask_KV if use_padding_mask else None
                                    )
                                    attn_mask = attn_mask_cross if use_attn_mask else None
                                # For sequence RoPE, KV features need to come from the same sequence
                                else:
                                    feat_KV_ = torch.randn_like(feat_Q)
                                    coords_KV_ = coords_Q
                                    padding_mask_KV_ = padding_mask_Q if use_padding_mask else None
                                    attn_mask = attn_mask_self if use_attn_mask else None

                            # Linear Platonic attn implementation requires query-independent mask
                            if linear_attention and use_attn_mask:
                                attn_mask = attn_mask[:, :, 0:1, :]

                            setting_str = (
                                f"  G:                 {solid_name}\n"
                                + f"  attn_backend:      {attn_backend}\n"
                                + f"  freq_sigma:        {freq_sigma_platonic}\n"
                                + f"  linear_attention:  {linear_attention}\n"
                                + f"  use_key:           {use_key}\n"
                                + f"  use_qk_norm:       {use_qk_norm}\n"
                                + f"  use_sequence_rope: {use_sequence_rope}\n"
                                + f"  use_padding_mask:  {use_padding_mask}\n"
                                + f"  use_attn_mask:     {use_attn_mask}\n"
                                + f"  cross_attn:        {cross_attn}"
                            )

                            self._test_equivariance_inner(
                                attn_module,
                                group,
                                c_out,
                                setting_str,
                                feat_Q,
                                coords_Q,
                                feat_KV_,
                                coords_KV_,
                                sequence_idxs_Q if use_sequence_rope else None,
                                padding_mask_KV_,
                                attn_mask,
                                avg_num_nodes,
                            )

    def _test_equivariance_inner(
        self,
        attn_module,
        group,
        c_out,
        setting_str,
        # input args
        feat_Q,
        coords_Q,
        feat_KV,
        coords_KV,
        sequence_idxs_Q,
        padding_mask_KV,
        attn_mask,
        avg_num_nodes,
    ) -> None:

        B, NQ, G, _ = feat_Q.shape
        G = group.G
        device, dtype = feat_Q.device, feat_Q.dtype

        # Test equivariance for each group element
        out = attn_module(
            feat_Q.flatten(-2),
            coords_Q,
            feat_KV.flatten(-2) if feat_KV is not None else None,
            coords_KV if coords_KV is not None else None,
            sequence_idxs_Q,
            padding_mask_KV,
            attn_mask,
            avg_num_nodes,
        ).view(B, NQ, G, c_out)

        for g in range(G):

            g_indices = group.cayley_table[g, :]
            g_element = group.elements[g].to(device, dtype)

            # G-transform original output
            g_out = out[:, :, g_indices, :]

            # G-transform input features
            g_feat_Q = feat_Q[:, :, g_indices, :]
            g_feat_KV_ = feat_KV[:, :, g_indices, :] if feat_KV is not None else None

            # G-transform coordinates
            g_coords_Q = torch.einsum("ji,...j ->...i", g_element, coords_Q)
            g_coords_KV_ = (
                torch.einsum("ji,...j ->...i", g_element, coords_KV)
                if coords_KV is not None
                else None
            )

            # Compute output for transformed inputs
            out_g = attn_module(
                g_feat_Q.flatten(-2),
                g_coords_Q,
                g_feat_KV_.flatten(-2) if g_feat_KV_ is not None else None,
                g_coords_KV_ if g_coords_KV_ is not None else None,
                sequence_idxs_Q,
                padding_mask_KV,
                attn_mask,
                avg_num_nodes,
            ).view(B, NQ, G, c_out)

            self.assertTrue(
                torch.allclose(out_g, g_out, atol=4e-5),
                f"\nEquivariance test failed (g={g}):\n"
                + setting_str
                + "\n"
                + f"  max difference:    {torch.max(torch.abs(out_g - g_out))}",
            )


if __name__ == "__main__":
    unittest.main()
