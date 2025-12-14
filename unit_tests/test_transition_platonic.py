import unittest

import torch
import torch.nn as nn

from zatom.models.architectures.platoformer import PLATONIC_GROUPS_3D
from zatom.models.architectures.platoformer.layer_norm_platonic import LayerNormPlatonic
from zatom.models.architectures.platoformer.transition_platonic import (
    FeedForwardPlatonic,
    TransitionPlatonic,
)


class TestFeedForwardPlatonic(unittest.TestCase):
    """Unit tests for FeedForwardPlatonic."""

    def test_equivariance_feed_forward_platonic(self):
        """Equivariance unit tests for FeedForwardPlatonic."""

        for solid_name in PLATONIC_GROUPS_3D:

            group = PLATONIC_GROUPS_3D[solid_name]
            num_G = group.G
            C = 8 * num_G  # divisible by num_G
            B, M = 16, 33
            x = torch.randn(B, M, C)

            ffwd = FeedForwardPlatonic(
                dim=C,
                hidden_dim=4 * C,
                solid_name=solid_name,
            )

            ffwd_x = ffwd(x)

            for g in range(num_G):
                g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep

                ffwd_g_x = ffwd(x.view(B, M, num_G, C // num_G)[:, :, g_indices, :].flatten(-2))
                g_ffwd_x = ffwd_x.view(B, M, num_G, C // num_G)[:, :, g_indices, :].flatten(-2)

                self.assertTrue(
                    torch.allclose(ffwd_g_x, g_ffwd_x, atol=1e-5),
                    "\nEquivariance test failed for:\n"
                    + f"  G:              {solid_name}\n"
                    + f"  max difference: {torch.max(torch.abs(ffwd_g_x - g_ffwd_x))}",
                )


class TestTransitionPlatonic(unittest.TestCase):
    """Unit tests for TransitionPlatonic."""

    def test_equivariance_transition_platonic(self):
        """Equivariance unit tests for TransitionPlatonic."""

        for solid_name in PLATONIC_GROUPS_3D:

            group = PLATONIC_GROUPS_3D[solid_name]
            num_G = group.G
            C = 8 * num_G  # divisible by num_G
            B, M = 16, 33
            x = torch.randn(B, M, C)

            layer_norm_platonic = LayerNormPlatonic(
                solid_name, hidden_dim=C, normalize_per_g=False
            )

            for activation_type in ("swiglu", "geglu", "gelu", "relu", "silu"):

                ffwd = TransitionPlatonic(
                    dim=C,
                    hidden_dim=4 * C,
                    solid_name=solid_name,
                    activation_type=activation_type,
                    layer_norm=layer_norm_platonic,
                )

                ffwd_x = ffwd(x)

                for g in range(num_G):
                    g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep

                    ffwd_g_x = ffwd(
                        x.view(B, M, num_G, C // num_G)[:, :, g_indices, :].flatten(-2)
                    )
                    g_ffwd_x = ffwd_x.view(B, M, num_G, C // num_G)[:, :, g_indices, :].flatten(-2)

                    self.assertTrue(
                        torch.allclose(ffwd_g_x, g_ffwd_x, atol=1e-5),
                        "\nEquivariance test failed for:\n"
                        + f"  G:              {solid_name}\n"
                        + f"  max difference: {torch.max(torch.abs(ffwd_g_x - g_ffwd_x))}",
                    )


if __name__ == "__main__":
    unittest.main()
