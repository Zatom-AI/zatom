import unittest
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

# from zatom.tests.test_platonic_linear import test_regular_rep_equivariance
from test_platonic_linear import test_regular_rep_equivariance

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
