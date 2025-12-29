import unittest
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from zatom.models.architectures.platoformer import (
    PLATONIC_GROUPS_3D,
    get_platonic_group,
)
from zatom.models.architectures.platoformer.feedfwd import (
    FeedForwardPlatonic,
    SwiGLUFeedForwardPlatonic,
    TransitionPlatonic,
)
from zatom.models.architectures.platoformer.norm import NormPlatonic


def test_regular_rep_equivariance(
    op_init: Callable[[str], nn.Module],
    test_case: unittest.TestCase,
    Cin: int,
    Cout: int,
    B: int = 4,
    N: int = 23,
) -> None:
    """Abstract equivariance unit test for ops whose inputs/outputs are regular representations.
    Reduces redundancy in unit test code.

    This shared equivariance test is meant for all equivariance operations taking inputs of shape
    (B, N, G*Cin) and produce outputs of shape (B, N, G*Cout), where a Platonic group G acts via the
    regular G-reprsentation after reshaping to (B, N, G, Cxx).

    Args:
        op_init:   functools.partial __init__ of the operation to be instantiated, with the only
                   remaining argument being a solid_name string from PLATONIC_GROUPS_3D.
                   Used to instantiate the op for each possible group.
        test_case: The TestCase from which this is called, required for test_case.assertTrue(...).
        Cin:       Number of input channels per group element.
        Cout:      Number of output channels per group element.
        B:         Batch size.
        N:         Number of tokens.
    """
    for solid_name in PLATONIC_GROUPS_3D:
        group = get_platonic_group(solid_name)
        G = group.G
        x = torch.randn(B, N, G * Cin)
        op = op_init(solid_name=solid_name)

        op_x = op(x)

        for g in range(G):
            g_indices = group.cayley_table[g, :]  # row from (G,G) regular rep

            op_g_x = op(x.view(B, N, G, Cin)[:, :, g_indices, :].flatten(-2))
            g_op_x = op_x.view(B, N, G, Cout)[:, :, g_indices, :].flatten(-2)

            test_case.assertTrue(
                torch.allclose(op_g_x, g_op_x, atol=1e-5),
                "\nEquivariance test failed for:\n"
                + f"  G:              {solid_name}\n"
                + f"  g:              {g}\n"
                + f"  max difference: {torch.max(torch.abs(op_g_x - g_op_x))}",
            )


class TestFeedForwardPlatonic(unittest.TestCase):
    """Unit tests for FeedForwardPlatonic."""

    def test_equivariance_feed_forward_platonic(self):
        """Equivariance unit tests for FeedForwardPlatonic."""
        Cin, Chid = 8, 16
        op_init = partial(
            FeedForwardPlatonic,
            c_io=Cin,
            c_hid=Chid,
        )
        test_regular_rep_equivariance(op_init, test_case=self, Cin=Cin, Cout=Cin)


class TestTransitionPlatonic(unittest.TestCase):
    """Unit tests for TransitionPlatonic."""

    def test_equivariance_transition_platonic(self):
        """Equivariance unit tests for TransitionPlatonic."""

        for activation_type in ("swiglu", "geglu", "gelu", "relu", "silu"):
            C = 16

            def op_init(solid_name: str) -> nn.Module:
                layer_norm_platonic = NormPlatonic(
                    "LayerNorm", solid_name, c=C, normalize_per_g=False
                )
                ffwd = TransitionPlatonic(
                    c_io=C,
                    c_hid=None,
                    solid_name=solid_name,
                    dropout=0.0,
                    activation_type=activation_type,
                    norm=layer_norm_platonic,
                )
                return ffwd

            test_regular_rep_equivariance(op_init, test_case=self, Cin=C, Cout=C)


class TestSwiGLUFeedForwardPlatonic(unittest.TestCase):
    """Unit tests for SwiGLUFeedForwardPlatonic."""

    def test_equivariance_norm_platonic(self):
        Cin, Chid, Cout = 8, 12, 16
        op_init = partial(SwiGLUFeedForwardPlatonic, Cin, Chid, Cout)
        test_regular_rep_equivariance(op_init, test_case=self, Cin=Cin, Cout=Cout)


if __name__ == "__main__":
    unittest.main()
