"""Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer."""

import copy
from typing import List, Literal

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from zatom.utils.typing_utils import typecheck


class FlowMatchingInterpolant:
    """Interpolant for simple multimodal flow matching.

    - Constructs noisy samples from clean samples during training.

    Adapted from: https://github.com/jasonkyuyim/multiflow

    Args:
        disc_feats: List of discrete feature names to corrupt.
        cont_feats: List of continuous feature names to corrupt.
        cont_feats_distributions: List of distributions for continuous features.
        mask_token_index: Index of the mask token.
        min_t: Minimum time step to sample during training.
        max_t: Maximum time step to sample during training.
        corrupt: Whether to corrupt samples during training.
        device: Device to run on.
        disc_interpolant_type: Type of discrete interpolant to use.
    """

    def __init__(
        self,
        disc_feats: List[str],
        cont_feats: List[str],
        cont_feats_distributions: List[Literal["centered_gaussian", "gaussian", "uniform"]],
        mask_token_index: int,
        max_num_nodes: int | None = None,
        min_t: float = 1e-2,
        max_t: float = 1.0,
        corrupt: bool = True,
        device: str | torch.device = "cpu",
        disc_interpolant_type: Literal["uniform", "masking"] = "masking",
    ):
        self.disc_feats = set(disc_feats)
        self.cont_feats = set(cont_feats)
        self.mask_token_index = mask_token_index
        self.max_num_nodes = max_num_nodes
        self.min_t = min_t
        self.max_t = max_t
        self.corrupt = corrupt
        self.device = device
        self.disc_interpolant_type = disc_interpolant_type

        self.feats = disc_feats + cont_feats
        self.num_tokens = mask_token_index + int(corrupt)  # +1 for the mask token if corrupting

        # Validate continuous feature distributions
        assert len(cont_feats) == len(
            cont_feats_distributions
        ), "Length of `cont_feats` must match length of `cont_feats_distributions`."
        assert all(
            dist in ("centered_gaussian", "gaussian", "uniform")
            for dist in cont_feats_distributions
        ), f"Invalid distribution ({cont_feats_distributions}) in `cont_feats_distributions`."
        self.cont_feat_to_distribution = dict(zip(cont_feats, cont_feats_distributions))

        # NOTE: To corrupt fractional coordinates (`frac_coords`), atom positions (`pos`) must be corrupted first
        assert len(self.feats) > 0, "No features to corrupt."
        self.feats.sort(reverse=True)

    @typecheck
    def _sample_t(self, batch_size: int) -> torch.Tensor:
        """Sample a time `t` uniformly from [min_t, max_t]."""
        t = torch.rand(batch_size, device=self.device)
        return t * (self.max_t - self.min_t) + self.min_t

    @typecheck
    def _gaussian(
        self, batch_size: int, num_tokens: int, emb_dim: int = 3, center: bool = True
    ) -> torch.Tensor:
        """Sample from a Gaussian distribution."""
        noise = torch.randn(batch_size, num_tokens, emb_dim, device=self.device)
        return noise - torch.mean(noise, dim=-2, keepdims=True) if center else noise

    @typecheck
    def _corrupt_disc_x(
        self,
        x_1: torch.Tensor,
        t: torch.Tensor,
        token_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Corrupt the discrete input tensor `x_1` using the noise schedule defined by `t`."""
        batch_size, num_tokens = token_mask.shape

        assert x_1.shape == (
            batch_size,
            num_tokens,
        ), f"Expected x_1 to be of shape {(batch_size, num_tokens)}, but got: {x_1.shape}"
        assert t.shape == (
            batch_size,
            1,
        ), f"Expected t to be of shape {(batch_size, 1)}, but got: {t.shape}"
        assert token_mask.shape == (
            batch_size,
            num_tokens,
        ), f"Expected token_mask to be of shape {(batch_size, num_tokens)}, but got: {token_mask.shape}"
        assert diffuse_mask.shape == (
            batch_size,
            num_tokens,
        ), f"Expected diffuse_mask to be of shape {(batch_size, num_tokens)}, but got: {diffuse_mask.shape}"

        u = torch.rand(batch_size, num_tokens, device=self.device)
        x_t = x_1.clone()

        corruption_mask = u < (1 - t)  # [B, N]

        if self.disc_interpolant_type == "masking":
            x_t[corruption_mask] = self.mask_token_index

            x_t = x_t * token_mask + self.mask_token_index * (1 - token_mask)

        elif self.disc_interpolant_type == "uniform":
            uniform_sample = torch.randint_like(x_t, low=0, high=self.num_tokens)
            x_t[corruption_mask] = uniform_sample[corruption_mask]

            x_t = x_t * token_mask + self.mask_token_index * (1 - token_mask)

        else:
            raise ValueError(f"Unknown discrete interpolant type {self.disc_interpolant_type}")

        return x_t * diffuse_mask + x_1 * (1 - diffuse_mask)

    @typecheck
    def _corrupt_cont_x(
        self,
        x_1: torch.Tensor,
        t: torch.Tensor,
        token_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
        feat: str,
    ) -> torch.Tensor:
        """Corrupt the continuous input tensor `x_1` using the noise schedule defined by `t`."""
        distribution = self.cont_feat_to_distribution[feat]
        x_0 = (
            torch.rand_like(x_1)
            if distribution == "uniform"
            else self._gaussian(*x_1.shape, center=(distribution == "centered_gaussian"))
        )
        x_t = (1 - t[..., None]) * x_0 + t[..., None] * x_1
        x_t = x_t * diffuse_mask[..., None] + x_1 * (~diffuse_mask[..., None])
        return x_t * token_mask[..., None]

    @typecheck
    def corrupt_batch(self, batch: Batch) -> Batch:
        """Corrupt a batch of data by sampling a time `t` and interpolating to noisy samples.

        Args:
            batch: Batch of clean data with the following keys:
                - atom_types (torch.Tensor): Clean (discrete) atom types tensor.
                - pos (torch.Tensor): Clean (continuous) atom positions tensor.
                - frac_coords (torch.Tensor): Clean (continuous) fractional coordinates tensor.
                - lengths_scaled (torch.Tensor): Lattice lengths tensor, after scaling by `num_atoms**(1/3)`.
                - angles_radians (torch.Tensor): Lattice angles tensor, in radians.
                - batch (torch.Tensor): Batch node index tensor.

        Returns:
            Noisy batch of data with updated (or new) values for the following keys:
                - atom_types (torch.Tensor): Noisy (discrete) atom types tensor.
                - pos (torch.Tensor): Noisy (continuous) atom positions tensor.
                - frac_coords (torch.Tensor): Noisy (continuous) fractional coordinates tensor.
                - lengths_scaled (torch.Tensor): Noisy (continuous) lattice lengths tensor.
                - angles_radians (torch.Tensor): Noisy (continuous) lattice angles tensor.
                - token_mask (torch.Tensor): Token mask tensor.
        """
        assert all(feat in batch for feat in self.feats), (
            f"Batch must contain at least the following features: {self.feats}, "
            f"but got: {list(batch.keys())}"
        )

        noisy_batch = copy.deepcopy(batch)

        # Corrupt features according to their data modality
        # (i.e., discrete or `disc` and continuous or `cont`)
        for feat in self.feats:
            # [B, N, d]
            x_1 = batch[feat]

            # Convert from PyG batch to dense batch (potentially with fixed-length max padding to stabilize GPU memory usage)
            is_global_feat = x_1.shape[0] == batch.batch_size
            x_1, mask = (
                (
                    # NOTE: Global features do not need to be densely padded
                    x_1.unsqueeze(-2),
                    torch.ones((batch.batch_size, 1), device=self.device, dtype=torch.bool),
                )
                if is_global_feat
                else to_dense_batch(x_1, batch["batch"], max_num_nodes=self.max_num_nodes)
            )

            # [B, N]
            token_mask = diffuse_mask = mask
            batch_size, _ = token_mask.shape

            # [B, 1]
            t = self._sample_t(batch_size)[:, None]
            noisy_batch[f"{feat}_t"] = t

            # Apply discrete data corruptions
            if self.corrupt and feat in self.disc_feats:
                assert x_1.dtype in (torch.int16, torch.int32, torch.int64), (
                    f"Expected {feat} to be of dtype int16, int32 or int64, "
                    f"but got: {x_1.dtype}"
                )
                x_t = self._corrupt_disc_x(x_1, t, token_mask.long(), diffuse_mask.long())

            # Apply continuous data corruptions
            elif self.corrupt and feat in self.cont_feats:
                assert x_1.dtype in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                ), (
                    f"Expected {feat} to be of dtype float16, bfloat16, float32 or float64, "
                    f"but got: {x_1.dtype}"
                )
                x_t = self._corrupt_cont_x(x_1, t, token_mask, diffuse_mask, feat)

            # Skip corruptions
            else:
                x_t = x_1

            if torch.any(torch.isnan(x_t)):
                raise ValueError(f"NaN found in `x_t` during corruption of `{feat}`.")

            noisy_batch[feat] = x_t

            if (
                not is_global_feat
            ):  # NOTE: Global feature masks are placeholders and can be ignored from this point forward
                noisy_batch["token_mask"] = (
                    mask | noisy_batch["token_mask"] if "token_mask" in noisy_batch else mask
                )

        # Return batch of corrupted features
        return noisy_batch

    def __repr__(self) -> str:
        """Return a string representation of the interpolant."""
        return f"{self.__class__.__name__}(min_t={self.min_t}, max_t={self.max_t}, corrupt={self.corrupt})"
