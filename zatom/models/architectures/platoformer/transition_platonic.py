"""Adapted from https://github.com/carlosinator/tabasco."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from platonic_transformers.models.platoformer.linear import PlatonicLinear

from zatom.models.architectures.platoformer import PLATONIC_GROUPS_3D
from zatom.models.architectures.platoformer.layer_norm_platonic import LayerNormPlatonic
from zatom.utils.typing_utils import typecheck


class FeedForwardPlatonic(nn.Module):
    """Platonic group equivariant feed forward network with optional activation and dropout.

    Similar to the standard feed forward network used in transformer architectures, but relying on
    PlatonicLinear layers. The activation function and dropout rate are configurable.

    Args:
        dim:        Input and output dimension
        hidden_dim: Hidden dimension (typically 4x the input dimension)
        solid_name: String identifying the Platonic solid group.
        bias:       Boolean indicating whether or not to use a bias in the linear layers.
        dropout:    Dropout probability
        activation: Activation function to use
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        solid_name: str,  # in PLATONIC_GROUPS_3D.keys()
        bias: bool = True,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            PlatonicLinear(dim, hidden_dim, solid=solid_name, bias=bias),
            activation(),
            nn.Dropout(dropout),
            PlatonicLinear(hidden_dim, dim, solid=solid_name, bias=bias),
            nn.Dropout(dropout),
        )

    @typecheck
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        dim:             Input and output dimension
        hidden_dim:      Hidden dimension (defaults to 4x input dim)
        solid_name:      String identifying the Platonic solid group.
        dropout:         Dropout probability
        activation_type: Type of activation to use ('swiglu', 'geglu', 'gelu', 'relu', 'silu')
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        solid_name: str = "octahedron",  # in PLATONIC_GROUPS_3D.keys()
        dropout: float = 0.0,
        activation_type: str = "swiglu",
        layer_norm: Optional[LayerNormPlatonic] = None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        if activation_type in ("swiglu", "geglu"):
            self.w1 = PlatonicLinear(dim, hidden_dim, solid=solid_name, bias=False)
            self.w2 = PlatonicLinear(dim, hidden_dim, solid=solid_name, bias=False)
            self.w3 = PlatonicLinear(hidden_dim, dim, solid=solid_name, bias=False)
        elif activation_type in ("gelu", "relu", "silu"):
            self.w1 = PlatonicLinear(dim, hidden_dim, solid=solid_name, bias=False)
            self.w3 = PlatonicLinear(hidden_dim, dim, solid=solid_name, bias=False)
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

        self.activation_type = activation_type
        self.dropout = nn.Dropout(dropout)
        self.norm = layer_norm

    @typecheck
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Platonic MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
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
