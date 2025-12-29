"""Adapted from https://github.com/carlosinator/tabasco."""

from typing import Optional

import torch.nn.functional as F
from platonic_transformers.models.platoformer.linear import PlatonicLinear
from torch import Tensor, nn

from zatom.models.architectures.platoformer import get_platonic_group
from zatom.models.architectures.platoformer.norm import NormPlatonic
from zatom.utils.typing_utils import typecheck


class FeedForwardPlatonic(nn.Module):
    """Platonic group equivariant feed forward network with optional activation and dropout.

    Similar to the standard feed forward network used in transformer architectures, but relying on
    PlatonicLinear layers. The activation function and dropout rate are configurable.

    Args:
        c_io:       Input and output channels per group element
        c_hid:      Hidden dimension per group element
        solid_name: String identifying the Platonic solid group.
        bias:       Boolean indicating whether or not to use a bias in the linear layers.
        dropout:    Dropout probability
        activation: Activation function to use
    """

    @typecheck
    def __init__(
        self,
        c_io: int,  # per group element
        c_hid: int,  # per group element
        solid_name: str,  # in PLATONIC_GROUPS_3D
        bias: bool = True,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        G = get_platonic_group(solid_name).G
        self.net = nn.Sequential(
            PlatonicLinear(G * c_io, G * c_hid, solid=solid_name, bias=bias),
            activation(),
            nn.Dropout(dropout),
            PlatonicLinear(G * c_hid, G * c_io, solid=solid_name, bias=bias),
            nn.Dropout(dropout),
        )

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Platonic feed forward network.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        return self.net(x)


class TransitionPlatonic(nn.Module):
    """Modern Transition MLP block with SwiGLU or other activation variants, Platonic variant.

    This implements a more modern version of the feed forward network used in
    transformers, with options for different activation functions including SwiGLU
    which is used in models like PaLM and LLaMA.

    Args:
        c_io:            Input and output channels per group element
        c_hid:           Hidden dimension per group element (defaults to 4x c_io)
        solid_name:      String identifying the Platonic solid group.
        dropout:         Dropout probability
        activation_type: Type of activation to use ('swiglu', 'geglu', 'gelu', 'relu', 'silu')
        norm:            Platonic (layer)norm operation
    """

    @typecheck
    def __init__(
        self,
        c_io: int,  # per group element
        c_hid: Optional[int] = None,  # per group element
        solid_name: str = "octahedron",  # in PLATONIC_GROUPS_3D.keys()
        dropout: float = 0.0,
        activation_type: str = "swiglu",
        norm: Optional[NormPlatonic] = None,
    ):
        super().__init__()
        G = get_platonic_group(solid_name).G

        if c_hid is None:
            c_hid = 4 * c_io

        if activation_type in ("swiglu", "geglu"):
            self.w1 = PlatonicLinear(G * c_io, G * c_hid, solid=solid_name, bias=False)
            self.w2 = PlatonicLinear(G * c_io, G * c_hid, solid=solid_name, bias=False)
            self.w3 = PlatonicLinear(G * c_hid, G * c_io, solid=solid_name, bias=False)
        elif activation_type in ("gelu", "relu", "silu"):
            self.w1 = PlatonicLinear(G * c_io, G * c_hid, solid=solid_name, bias=False)
            self.w3 = PlatonicLinear(G * c_hid, G * c_io, solid=solid_name, bias=False)
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

        self.activation_type = activation_type
        self.dropout = nn.Dropout(dropout)
        self.norm = norm

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Platonic MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_len, c_io]

        Returns:
            Output tensor of shape [batch_size, seq_len, c_io]
        """
        if self.norm is not None:
            x = self.norm(x)

        if self.activation_type in ("swiglu", "geglu"):
            x1 = self.w1(x)
            x2 = self.w2(x)
            match self.activation_type:
                case "swiglu":
                    hidden = F.silu(x1) * x2
                case "geglu":
                    hidden = F.gelu(x1) * x2
        else:
            x1 = self.w1(x)
            match self.activation_type:
                case "gelu":
                    hidden = F.gelu(x1)
                case "relu":
                    hidden = F.relu(x1)
                case "silu":
                    hidden = F.silu(x1)

        output = self.w3(hidden)
        return self.dropout(output)


class SwiGLUFeedForwardPlatonic(nn.Module):
    """Platonic group equivariant feed-forward network with SwiGLU activation.

    Args:
        c_io:        Input and output channel dimension (per group element).
        c_hid:       Hidden layer channel dimension (per group element).
        multiple_of: Ensure c_hidden is a multiple of this value.
    """

    @typecheck
    def __init__(
        self,
        c_in: int,  # per group element!
        c_hid: int,  # per group element!
        c_out: int,  # per group element!
        solid_name: str,
        # multiple_of: int = 256 # might clash with channels being a multiple of G
    ) -> None:
        super().__init__()
        G = get_platonic_group(solid_name).G
        self.w_in1 = PlatonicLinear(G * c_in, G * c_hid, solid=solid_name, bias=False)
        self.w_in2 = PlatonicLinear(G * c_in, G * c_hid, solid=solid_name, bias=False)
        self.w_out = PlatonicLinear(G * c_hid, G * c_out, solid=solid_name, bias=False)

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Platonic SwiGLU feed-forward network.

        Args:
            x (Tensor): Input tensor of shape (..., c_in).

        Returns:
            Tensor: Output tensor of shape (..., c_out).
        """
        swish = F.silu(self.w_in1(x))
        x_V = self.w_in2(x)
        x = swish * x_V
        return self.w_out(x)
