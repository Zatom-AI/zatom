import torch
import torch.nn as nn
from platonic_transformers.models.platoformer.ape import PlatonicAPE
from platonic_transformers.models.platoformer.io import lift_vectors
from platonic_transformers.models.platoformer.linear import PlatonicLinear
from torch import Tensor

from zatom.models.architectures.platoformer import get_platonic_group
from zatom.models.architectures.transformer.positional_encoder import PositionalEncoding
from zatom.utils.typing_utils import typecheck


class PlatonicSinusoidAPE(PositionalEncoding):
    """Group-Equivariant Absolute Position Encoding wrapping PlatonicAPE to augment it with the
    PositionalEncoding interface and reparametrize embed_dim via c_embed (per group element).

    Maps Euclidean vector batches of shape (.., spatial_dims) to regular rep embedding batches
    of shape (.., G*c_embed).

    Args:
        c_embed (int):      Number of embedding channels per group element.
        solid_name (str):   Name of the Platonic solid to define the symmetry group.
        spatial_dims (int): Spatial dimensions of the Euclidean coordinates to be embedded.
    """

    @typecheck
    def __init__(
        self,
        c_embed: int,  # per group element
        solid_name: str,
        freq_sigma: float,
        spatial_dims: int = 3,
        learned_freqs: bool = False,
    ):
        super().__init__()

        self.group = get_platonic_group(solid_name)
        self.c_embed = c_embed
        self.spatial_dims = spatial_dims

        self.platonic_sinusoid_ape = PlatonicAPE(
            embed_dim=c_embed * self.group.G,
            solid_name=solid_name,
            freq_sigma=freq_sigma,
            spatial_dims=spatial_dims,
            learned_freqs=learned_freqs,
        )

    @typecheck
    def forward(self, coords: Tensor) -> Tensor:
        """Compute Platonic group-equivariant absolute sinusoidal position embeddings.

        Args:
            coords (Tensor): Euclidean coordinates tensor of shape (..., spatial_dims).

        Returns:
            Tensor: The calculated position embedding of shape (..., G*c_embed).
        """
        assert coords.size(-1) == self.spatial_dims
        return self.platonic_sinusoid_ape(coords)

    def out_dim(self) -> int:
        """Embedding dimension produced by this encoder.

        Returns:
            Integer representing the output dimension.
        """
        return self.group.G * self.c_embed


class PlatonicLinearAPE(PositionalEncoding):
    """Group-Equivariant Absolute Position Encoding via lift_vectors followed by PlatonicLinear.

    Maps Euclidean vector batches of shape (.., spatial_dims) to regular rep embedding batches
    of shape (.., G*c_embed).

    This module was added as a simple Platonic group equivariant counterpart of the *linear*
    pos_embed submodule of TransformerModule.

    Args:
        c_embed (int):      Number of embedding channels per group element.
        solid_name (str):   Name of the Platonic solid to define the symmetry group.
        spatial_dims (int): Spatial dimensions of the Euclidean coordinates to be embedded.
    """

    @typecheck
    def __init__(
        self,
        c_embed: int,
        solid_name: str,
        spatial_dims: int,
    ):
        super().__init__()

        self.group = get_platonic_group(solid_name)
        self.c_embed = c_embed
        self.spatial_dims = spatial_dims

        self.platonic_linear = PlatonicLinear(
            in_features=spatial_dims * self.group.G,  # after lifting coords to 3 regular reps
            out_features=c_embed * self.group.G,
            solid=solid_name,
        )

    @typecheck
    def forward(self, coords: Tensor) -> Tensor:
        """Compute Platonic group-equivariant absolute linear position embeddings.

        Args:
            coords (Tensor): Euclidean coordinates tensor of shape (..., spatial_dims).

        Returns:
            Tensor: The calculated position embedding of shape (..., G*c_embed).
        """
        assert coords.size(-1) == self.spatial_dims

        # Lift position vectors to regular representations.
        # lift_vectors assumes input shape (..., C, dim) and produces output shape (..., G, C*dim).
        # Since pos has shape (..., 3), we first need to unsqueeze a channel axis of size 1.
        coords = coords.unsqueeze(-2)  # (..., 1, dim)
        coords_lift = lift_vectors(coords, self.group)  # (..., G, dim)
        coords_lift = coords_lift.flatten(-2, -1)  # (..., G*dim)

        return self.platonic_linear(coords_lift)  # (..., G*c_embed)

    def out_dim(self) -> int:
        """Embedding dimension produced by this encoder.

        Returns:
            Integer representing the output dimension.
        """
        return self.group.G * self.c_embed
