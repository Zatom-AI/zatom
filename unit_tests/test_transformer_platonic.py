import unittest

import torch
import torch.nn as nn

from zatom.models.architectures.platoformer import (
    PLATONIC_GROUPS_3D,
    get_platonic_group,
)
from zatom.models.architectures.platoformer.transformer_platonic import (
    ModernTransformerBlockPlatonic,
    ModernTransformerDecoderBlockPlatonic,
    ModernTransformerPlatonic,
)


class TestModernTransformerBlockPlatonic(unittest.TestCase):
    """Unit tests for ModernTransformerBlockPlatonic."""

    def test_equivariance_transformer_block_platonic(self):
        """Equivariance unit tests for ModernTransformerBlockPlatonic."""
        c_model = 64  # per group element
        c_qk = 16  # per group element
        c_val = 8  # per group element
        B, H, N = 16, 6, 33
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        for solid_name in PLATONIC_GROUPS_3D:
            group = get_platonic_group(solid_name)
            G = group.G

            # Random input tensors
            feat = torch.randn((B, N, G * c_model), device=device, dtype=dtype)
            coords = torch.randn((B, N, 3), device=device, dtype=dtype)
            sequence_lens = torch.randint(N // 3, N, (B,), device=device)
            sequence_lens[B // 2] = N
            padding_mask = ~(torch.arange(N, device=device)[None, :] < sequence_lens[:, None])
            attn_mask = torch.randn((B, H, N, N), device=device, dtype=dtype) < 1.25
            sequence_idxs = torch.arange(N, device=device).repeat(B, 1)  # (B,NQ)

            block = ModernTransformerBlockPlatonic(
                c_model=c_model,
                c_qk=c_qk,
                c_val=c_val,
                n_heads=H,
                solid_name=solid_name,
                freq_sigma_platonic=1.234,
            ).to(device, dtype)

            # Original output, for non-transformed inputs
            block_feat = block(
                feat=feat,
                coords=coords,
                sequence_idxs=sequence_idxs,
                padding_mask=padding_mask,
                attn_mask=attn_mask,
            )

            for g in range(G):
                g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep
                g_element = group.elements[g].to(device, dtype)

                g_feat = feat.view(B, N, G, c_model)[:, :, g_indices, :].flatten(-2)
                g_coords = torch.einsum("ji,...j ->...i", g_element, coords)

                # Output for transformed inputs
                block_g_feat = block(
                    feat=g_feat,
                    coords=g_coords,
                    sequence_idxs=sequence_idxs,
                    padding_mask=padding_mask,
                    attn_mask=attn_mask,
                )

                # Transformed output from non-transformed inputs
                g_block_feat = block_feat.view(B, N, G, c_model)[:, :, g_indices, :].flatten(-2)

                self.assertTrue(
                    torch.allclose(block_g_feat, g_block_feat, atol=5e-5),
                    "\nEquivariance test failed for:\n"
                    + f"  G:              {solid_name}\n"
                    + f"  max difference: {torch.max(torch.abs(block_g_feat - g_block_feat))}",
                )


class TestModernTransformerDecoderBlockPlatonic(unittest.TestCase):
    """Unit tests for ModernTransformerDecoderBlockPlatonic."""

    def test_equivariance_transformer_decoder_block_platonic(self):
        """Equivariance unit tests for ModernTransformerDecoderBlockPlatonic."""
        c_model = 64  # per group element
        c_qk = 16  # per group element
        c_val = 8  # per group element
        B, H, N, M = 16, 6, 33, 23
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        for solid_name in PLATONIC_GROUPS_3D:
            group = get_platonic_group(solid_name)
            G = group.G

            # Random input tensors
            feat = torch.randn((B, N, G * c_model), device=device, dtype=dtype)
            memory = torch.randn((B, M, G * c_model), device=device, dtype=dtype)
            coords_feat = torch.randn((B, N, 3), device=device, dtype=dtype)
            coords_mem = torch.randn((B, M, 3), device=device, dtype=dtype)
            sequence_lens_feat = torch.randint(N // 3, N, (B,), device=device)
            sequence_lens_mem = torch.randint(M // 3, M, (B,), device=device)
            sequence_lens_feat[B // 2] = N
            sequence_lens_mem[B // 2] = M
            padding_mask_feat = ~(
                torch.arange(N, device=device)[None, :] < sequence_lens_feat[:, None]
            )
            padding_mask_mem = ~(
                torch.arange(M, device=device)[None, :] < sequence_lens_mem[:, None]
            )
            attn_mask_self = torch.randn((B, H, N, N), device=device, dtype=dtype) < 1.25
            attn_mask_cross = torch.randn((B, H, N, M), device=device, dtype=dtype) < 1.25
            sequence_idxs = torch.arange(N, device=device).repeat(B, 1)  # (B, N)

            block = ModernTransformerDecoderBlockPlatonic(
                c_model=c_model,
                c_qk=c_qk,
                c_val=c_val,
                n_heads=H,
                solid_name=solid_name,
                freq_sigma_platonic=1.234,
            ).to(device, dtype)

            # Original output, for non-transformed inputs
            block_feat = block(
                feat=feat,
                memory=memory,
                coords_feat=coords_feat,
                coords_mem=coords_mem,
                sequence_idxs=sequence_idxs,
                padding_mask_feat=padding_mask_feat,
                padding_mask_mem=padding_mask_mem,
                attn_mask_self=attn_mask_self,
                attn_mask_cross=attn_mask_cross,
            )

            for g in range(G):
                g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep
                g_element = group.elements[g].to(device, dtype)

                g_feat = feat.view(B, N, G, c_model)[:, :, g_indices, :].flatten(-2)
                g_memory = memory.view(B, M, G, c_model)[:, :, g_indices, :].flatten(-2)
                g_coords_feat = torch.einsum("ji,...j ->...i", g_element, coords_feat)
                g_coords_mem = torch.einsum("ji,...j ->...i", g_element, coords_mem)

                # Output for transformed inputs
                block_g_feat = block(
                    feat=g_feat,
                    memory=g_memory,
                    coords_feat=g_coords_feat,
                    coords_mem=g_coords_mem,
                    sequence_idxs=sequence_idxs,
                    padding_mask_feat=padding_mask_feat,
                    padding_mask_mem=padding_mask_mem,
                    attn_mask_self=attn_mask_self,
                    attn_mask_cross=attn_mask_cross,
                )

                # Transformed output from non-transformed inputs
                g_block_feat = block_feat.view(B, N, G, c_model)[:, :, g_indices, :].flatten(-2)

                self.assertTrue(
                    torch.allclose(block_g_feat, g_block_feat, atol=5e-5),
                    "\nEquivariance test failed for:\n"
                    + f"  G:              {solid_name}\n"
                    + f"  max difference: {torch.max(torch.abs(block_g_feat - g_block_feat))}",
                )


class TestModernTransformerPlatonic(unittest.TestCase):
    """Unit tests for ModernTransformerPlatonic."""

    def test_equivariance_transformer_platonic(self):
        """Equivariance unit tests for ModernTransformerPlatonic."""
        c_model = 64  # per group element
        c_qk = 16  # per group element
        c_val = 8  # per group element
        B, H, N = 16, 6, 33
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        for solid_name in PLATONIC_GROUPS_3D:
            group = get_platonic_group(solid_name)
            G = group.G

            # Random input tensors
            feat = torch.randn((B, N, G * c_model), device=device, dtype=dtype)
            coords = torch.randn((B, N, 3), device=device, dtype=dtype)
            sequence_lens = torch.randint(N // 3, N, (B,), device=device)
            sequence_lens[B // 2] = N
            padding_mask = ~(torch.arange(N, device=device)[None, :] < sequence_lens[:, None])
            attn_mask = torch.randn((B, H, N, N), device=device, dtype=dtype) < 1.25
            sequence_idxs = torch.arange(N, device=device).repeat(B, 1)  # (B,NQ)

            net = ModernTransformerPlatonic(
                c_model=c_model,
                c_qk=c_qk,
                c_val=c_val,
                n_heads=H,
                solid_name=solid_name,
                freq_sigma_platonic=1.234,
                depth=6,
                repr_layer=4,
            ).to(device, dtype)

            # Original output, for non-transformed inputs
            net_feat, net_repr = net(
                feat=feat,
                coords=coords,
                sequence_idxs=sequence_idxs,
                padding_mask=padding_mask,
                attn_mask=attn_mask,
            )

            for g in range(G):
                g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep
                g_element = group.elements[g].to(device, dtype)

                g_feat = feat.view(B, N, G, c_model)[:, :, g_indices, :].flatten(-2)
                g_coords = torch.einsum("ji,...j ->...i", g_element, coords)

                # Output for transformed inputs
                net_g_feat, net_g_repr = net(
                    feat=g_feat,
                    coords=g_coords,
                    sequence_idxs=sequence_idxs,
                    padding_mask=padding_mask,
                    attn_mask=attn_mask,
                )

                # Transformed output from non-transformed inputs
                g_net_feat = net_feat.view(B, N, G, c_model)[:, :, g_indices, :].flatten(-2)
                g_net_repr = net_repr.view(B, N, G, c_model)[:, :, g_indices, :].flatten(-2)

                self.assertTrue(
                    torch.allclose(net_g_feat, g_net_feat, atol=5e-5),
                    "\nEquivariance test failed for (feat):\n"
                    + f"  G:              {solid_name}\n"
                    + f"  max difference: {torch.max(torch.abs(net_g_feat - g_net_feat))}",
                )
                self.assertTrue(
                    torch.allclose(net_g_repr, g_net_repr, atol=5e-5),
                    "\nEquivariance test failed for (repr):\n"
                    + f"  G:              {solid_name}\n"
                    + f"  max difference: {torch.max(torch.abs(net_g_repr - g_net_repr))}",
                )


if __name__ == "__main__":
    unittest.main()
