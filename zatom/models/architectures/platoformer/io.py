import torch
import torch.nn as nn

from zatom.models.architectures.platoformer import get_platonic_group


class ProjRegularToScalar(nn.Module):
    """Equivariant module extracting scalars from regular rep features.

    Args:
        solid_name: String identifying the Platonic solid group.
    """

    def __init__(self, solid_name: str):
        super().__init__()
        self.group = get_platonic_group(solid_name)
        self.G = self.group.G

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ProjRegularToScalar.

        Args:
            x: regular rep feature tensor of shape (..., G*C)
        Returns:
            Scalar feature tensor of shape (..., C).
        """
        x = x.unflatten(-1, (self.G, -1))  # (..., G, C)
        x = x.mean(dim=-2)  # (..., C)
        return x


class ProjRegularToVector(nn.Module):
    """Equivariant module extracting vectors from regular rep features.

    Args:
        solid_name: String identifying the Platonic solid group.
        flatten: If True, flattens the channel and spatial axis together, if False, keeps them
            separate. See the docstring of .forward for details.
    """

    def __init__(self, solid_name: str, flatten: bool = True):
        super().__init__()
        self.group = get_platonic_group(solid_name)
        self.G = self.group.G
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ProjRegularToScalar.

        Args:
            x: regular rep feature tensor of shape (..., G*C*spatial_dim)
        Returns:
            Scalar feature tensor of shape (..., C, spatial_dim) if flatten is False
            or shape (..., C*spatial_dim) if flatten is True.
        """
        x = x.unflatten(-1, (self.G, -1, self.group.dim))  # (..., G, C, spatial_dim)
        frames = self.group.elements.type_as(x)  # (G, spatial_dim, spatial_dim)
        x = torch.einsum("gij,...gcj->...ci", frames, x)  # (..., C, spatial_dim)
        if self.flatten:
            x = x.flatten(-2)  # (..., C*spatial_dim)
        return x / self.group.G
