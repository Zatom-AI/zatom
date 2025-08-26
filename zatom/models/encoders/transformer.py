"""Transformer encoder.

Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer.
"""

import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from zatom.utils.typing_utils import typecheck


@typecheck
def get_index_embedding(indices: torch.Tensor, emb_dim: int, max_len: int = 2048) -> torch.Tensor:
    """Create sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: Offsets of size [..., num_tokens] of type integer.
        emb_dim: Dimension of the embeddings to create.
        max_len: Maximum length.

    Returns:
        Positional embedding of shape [..., num_tokens, emb_dim].
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class TransformerEncoder(nn.Module):
    """Standard Transformer encoder.

    Args:
        d_model: Dimension of the model.
        nhead: Number of attention heads.
        dim_feedforward: Dimension of the feedforward network.
        num_layers: Number of layers.
        activation: Activation function to use.
        dropout: Dropout rate.
        bias: Whether to use bias.
        norm_first: Whether to use pre-normalization in Transformer blocks.
        use_pytorch_implementation: Whether to use PyTorch's Transformer implementation.
    """

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 6,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        norm_first: bool = True,
        use_pytorch_implementation: bool = True,
    ):
        super().__init__()

        self.d_model = d_model

        self.atom_type_embedder = nn.Linear(d_model * 2, d_model, bias=True)
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.frac_coords_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.lengths_scaled_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.angles_radians_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        activation = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

    @typecheck
    def forward(
        self,
        atom_types: torch.Tensor,
        pos: torch.Tensor,
        frac_coords: torch.Tensor,
        lengths_scaled: torch.Tensor,
        angles_radians: torch.Tensor,
        token_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the Transformer encoder.

        Args:
            atom_types: Combined input and predicted atom embeddings for the batch.
            pos: Cartesian coordinates of atoms in the batch.
            frac_coords: Fractional coordinates of atoms in the batch.
            lengths_scaled: Lattice lengths tensor (with a singular global value for each batch element).
            angles_radians: Lattice angles tensor (with a singular global value for each batch element).
            token_idx: Indices of tokens in the batch.
            mask: Attention mask for the batch.

        Returns:
            Encoded token features.
        """
        x = self.atom_type_embedder(atom_types)  # [B, N, D * 2] -> [B, N, D]
        x += self.pos_embedder(pos)
        x += self.frac_coords_embedder(frac_coords)
        x += self.lengths_scaled_embedder(lengths_scaled)
        x += self.angles_radians_embedder(angles_radians)

        # Positional embedding
        x += get_index_embedding(token_idx, self.d_model)

        # Transformer forward pass
        with sdpa_kernel(SDPBackend.MATH):
            # NOTE: May need to use this context, as regular SDPA from PyTorch
            # may not support higher order gradients (e.g., for CUDA devices).
            # NOTE: May want to turn this off for inference eventually.
            x = self.transformer.forward(x, src_key_padding_mask=(~mask))

        return x
