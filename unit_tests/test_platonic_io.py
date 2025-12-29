import unittest

import torch
import torch.nn as nn

from zatom.models.architectures.platoformer import (
    PLATONIC_GROUPS_3D,
    get_platonic_group,
)
from zatom.models.architectures.platoformer.io import (
    ProjRegularToScalar,
    ProjRegularToVector,
)


class TestProjRegularToScalar(unittest.TestCase):
    """Unit tests for ProjRegularToScalar."""

    def test_equivariance_proj_regular_to_scalar(self):
        """Invariance unit tests for regular to scalar projection module."""
        for solid_name in PLATONIC_GROUPS_3D:
            group = get_platonic_group(solid_name)
            G = group.G
            C = 8  # per group element
            B, M = 16, 33
            x = torch.randn(B, M, G * C)

            proj = ProjRegularToScalar(solid_name=solid_name)
            proj_x = proj(x)

            for g in range(G):
                g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep

                proj_g_x = proj(x.view(B, M, G, C)[:, :, g_indices, :].flatten(-2))

                self.assertTrue(
                    torch.allclose(proj_g_x, proj_x, atol=1e-5),
                    "\nEquivariance test failed for:\n"
                    + f"  G:              {solid_name}\n"
                    + f"  max difference: {torch.max(torch.abs(proj_g_x - proj_x))}",
                )


class TestProjRegularToVector(unittest.TestCase):
    """Unit tests for ProjRegularToVector."""

    def test_equivariance_proj_regular_to_vector(self):
        """Invariance unit tests for regular to vector projection module."""
        for solid_name in PLATONIC_GROUPS_3D:
            group = get_platonic_group(solid_name)
            G = group.G
            C = 8  # per group element
            B, M = 16, 33
            x = torch.randn(B, M, G * C * group.dim)

            proj = ProjRegularToVector(solid_name=solid_name, flatten=False)
            proj_x = proj(x)

            for g in range(G):
                g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep
                g_element = group.elements[g].to(x.dtype)

                proj_g_x = proj(x.view(B, M, G, C * group.dim)[:, :, g_indices, :].flatten(-2))
                g_proj_x = torch.einsum("ji,...j ->...i", g_element, proj_x.clone())

                self.assertTrue(
                    torch.allclose(proj_g_x, g_proj_x, atol=1e-5),
                    "\nEquivariance test failed for:\n"
                    + f"  G:              {solid_name}\n"
                    + f"  max difference: {torch.max(torch.abs(proj_g_x - proj_x))}",
                )


if __name__ == "__main__":
    unittest.main()
