"""Energy-based transformer (EBT).

Adapted from:
    - https://github.com/alexiglad/EBT
    - https://github.com/facebookresearch/all-atom-diffusion-transformer
"""

import math
from typing import Any

import torch
import torch.nn as nn

from zatom.utils.typing_utils import typecheck

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations."""

    def __init__(self, hidden_dim: int, frequency_embedding_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @typecheck
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Compute timestep embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    @typecheck
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass for timestep embedding."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embed class labels into vector representations.

    NOTE: Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes: int, hidden_dim: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    @typecheck
    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Drop labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, 0, labels)
        # NOTE: 0 is the label for the null class
        return labels

    @typecheck
    def forward(
        self, labels: torch.Tensor, train: bool, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for label embedding."""
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


@typecheck
def get_pos_embedding(indices: torch.Tensor, emb_dim: int, max_len: int = 2048) -> torch.Tensor:
    """Create sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: Offsets of size [..., num_tokens] of type integer.
        emb_dim: Embedding dimension.
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


#################################################################################
#                               Transformer blocks                              #
#################################################################################


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] | None = None,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    @typecheck
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#################################################################################
#                                 Core EBT Model                                #
#################################################################################


@typecheck
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate the input tensor x with the given shift and scale."""
    # NOTE: This is global modulation.
    # TODO: Explore per-token modulation.
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class EBTBlock(nn.Module):
    """An EBT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs: Any
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    @typecheck
    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the EBT block."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(_x, _x, _x, key_padding_mask=mask, need_weights=False)[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of EBT."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    @typecheck
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass for the final EBT layer."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class EBT(nn.Module):
    """Energy-based model with a Transformer decoder (i.e., an energy decoder or E-coder).

    Args:
        d_x: Input dimension.
        d_model: Model dimension.
        num_layers: Number of Transformer layers.
        nhead: Number of attention heads.
        mlp_ratio: Ratio of hidden to input dimension in MLP.
        class_dropout_prob: Probability of dropping class labels for classifier-free guidance.
        num_datasets: Number of datasets for classifier-free guidance.
        num_spacegroups: Number of spacegroups for classifier-free guidance.
    """

    def __init__(
        self,
        d_x: int = 8,
        d_model: int = 384,
        num_layers: int = 12,
        nhead: int = 6,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_datasets: int = 2,  # Classifier-free guidance input
        num_spacegroups: int = 230,  # Classifier-free guidance input
    ):
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.nhead = nhead

        self.x_embedder = nn.Linear(2 * d_x, d_model, bias=True)
        self.t_embedder = TimestepEmbedder(d_model)
        self.dataset_embedder = LabelEmbedder(num_datasets, d_model, class_dropout_prob)
        self.spacegroup_embedder = LabelEmbedder(num_spacegroups, d_model, class_dropout_prob)

        self.blocks = nn.ModuleList(
            [EBTBlock(d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(d_model, d_x)
        self.initialize_weights()

    @typecheck
    def initialize_weights(self):
        """Initialize transformer layers."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table
        nn.init.normal_(self.dataset_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.spacegroup_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in EBT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dataset_idx: torch.Tensor,
        spacegroup: torch.Tensor,
        mask: torch.Tensor,
        x_sc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of EBT.

        Args:
            x: Input data tensor (B, N, d_in).
            t: Time step for each sample (B,).
            dataset_idx: Dataset index for each sample (B,).
            spacegroup: Spacegroup index for each sample (B,).
            mask: True if valid token, False if padding (B, N).
            x_sc: Self-conditioning x (B, N, d_in).

        Returns:
            torch.Tensor: Output tensor (B, N, d_out)
        """
        # Positional embedding
        token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        pos_emb = get_pos_embedding(token_index, self.d_model)

        # Self-conditioning and input embeddings: (B, N, d)
        if x_sc is None:
            x_sc = torch.zeros_like(x)
        x = self.x_embedder(torch.cat([x, x_sc], dim=-1)) + pos_emb

        # Conditioning embeddings
        t = self.t_embedder(t.squeeze(1))  # (B, d)
        d = self.dataset_embedder(dataset_idx, self.training)  # (B, d)
        s = self.spacegroup_embedder(spacegroup, self.training)  # (B, d)
        c = t + d + s  # (B, 1, d)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, ~mask)  # (B, N, d)

        # Prediction layer
        x = self.final_layer(x, c)  # (B, N, d_out)
        x = x * mask[..., None]
        return x

    @typecheck
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dataset_idx: torch.Tensor,
        spacegroup: torch.Tensor,
        mask: torch.Tensor,
        cfg_scale: float,
        x_sc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of EBT, while also batching the unconditional forward pass for classifier-
        free guidance.

        NOTE: Assumes batch x's and class labels are ordered such that the first half are the conditional
        samples and the second half are the unconditional samples.

        Args:
            x: Input data tensor (B, N, d_in).
            t: Time step for each sample (B,).
            dataset_idx: Dataset index for each sample (B,).
            spacegroup: Spacegroup index for each sample (B,).
            mask: True if valid token, False if padding (B, N).
            cfg_scale: Classifier-free guidance scale.
            x_sc: Self-conditioning x (B, N, d_in).

        Returns:
            torch.Tensor: Output tensor (B, N, d_out)
        """
        half_x = x[: len(x) // 2]
        combined_x = torch.cat([half_x, half_x], dim=0)
        model_out = self.forward(combined_x, t, dataset_idx, spacegroup, mask, x_sc)

        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        return eps
