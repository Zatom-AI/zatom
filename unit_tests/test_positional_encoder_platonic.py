import unittest
from functools import partial
from typing import Literal

import torch

from zatom.models.architectures.platoformer import (
    PLATONIC_GROUPS_3D,
    get_platonic_group,
)
from zatom.models.architectures.platoformer.positional_encoder_platonic import (
    PlatonicLinearAPE,
    PlatonicSinusoidAPE,
)


class TestPlatonicAPE(unittest.TestCase):
    """Equivariance unit tests for Platonic group equivariant absolute position embeddings.

    These modules are only equivariant wrt rotations and reflections in their Platonic group <
    O(N). They break translation equivariance.
    """

    def test_equivariance_platonic_sinusoidal_ape(self):
        """Running PlatonicSinusoidAPE equivariance tests for all Platonic groups."""
        for solid_name in PLATONIC_GROUPS_3D:
            self._test_equivariance_platonic_ape(solid_name, PlatonicAPEClass=PlatonicSinusoidAPE)

    def test_equivariance_platonic_linear_ape(self):
        """Running PlatonicLinearAPE equivariance tests for all Platonic groups."""
        for solid_name in PLATONIC_GROUPS_3D:
            self._test_equivariance_platonic_ape(solid_name, PlatonicAPEClass=PlatonicLinearAPE)

    def _test_equivariance_platonic_ape(
        self,
        solid_name: str,
        PlatonicAPEClass: Literal[PlatonicSinusoidAPE, PlatonicLinearAPE],
    ) -> None:
        """Shared equivariance test for the given Platonic APE embedding class and symmetry group.

        Looping over all group elements.
        """

        group = get_platonic_group(solid_name)
        G = group.G
        c_embed = 16
        B, N = 2, 4
        class_name = PlatonicAPEClass.__name__
        # print(f"testing {class_name}, G={solid_name}               ", end="\r")

        torch.manual_seed(42)
        pos = torch.randn(B, N, group.dim)

        if PlatonicAPEClass is PlatonicSinusoidAPE:
            PlatonicAPEClass = partial(PlatonicAPEClass, freq_sigma=4.2)

        pos_embedder = PlatonicAPEClass(
            c_embed=c_embed,
            solid_name=solid_name,
            spatial_dims=group.dim,
        )

        all_tests_passed = True
        embedding_orig = pos_embedder(pos)

        for g in range(G):
            g_indices = group.cayley_table[g, :]
            g_element = group.elements[g].to(pos.dtype)

            # Compute APE(g.pos)
            pos_transformed = torch.einsum(
                "ji,...j ->...i", g_element, pos
            )  # NOTE: acting as transposed element ???
            embedding_ape_g = pos_embedder(pos_transformed)

            # Compute g.APE(pos)
            embedding_g_ape = embedding_orig.view(B, N, G, c_embed)
            embedding_g_ape = embedding_g_ape[:, :, g_indices, :]
            embedding_g_ape = embedding_g_ape.reshape(B, N, G * c_embed)

            self.assertTrue(
                torch.allclose(embedding_ape_g, embedding_g_ape, atol=1e-5),
                "\nEquivariance test failed for:\n"
                + f"  {class_name}\n"
                + f"  G = {solid_name}\n"
                + f"  max difference: {torch.max(torch.abs(embedding_g_ape - embedding_ape_g))}",
            )


if __name__ == "__main__":
    unittest.main()
