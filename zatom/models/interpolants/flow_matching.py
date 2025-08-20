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

        # NOTE: To corrupt fractional coordinates (`frac_coords`), atom positions (`pos`) must be corrupted first
        assert len(self.feats) > 0, "No features to corrupt."
        self.feats.sort(reverse=True)

    @typecheck
    def _sample_t(self, batch_size: int) -> torch.Tensor:
        """Sample a time `t` uniformly from [min_t, max_t]."""
        t = torch.rand(batch_size, device=self.device)
        return t * (self.max_t - self.min_t) + self.min_t

    @typecheck
    def _masked_categorical(self, batch_size: int, num_tokens: int) -> torch.Tensor:
        """Create a masked categorical distribution."""
        return torch.ones(batch_size, num_tokens, device=self.device) * self.mask_token_index

    @typecheck
    def _centered_gaussian(
        self, batch_size: int, num_tokens: int, emb_dim: int = 3
    ) -> torch.Tensor:
        """Sample from a centered Gaussian distribution."""
        noise = torch.randn(batch_size, num_tokens, emb_dim, device=self.device)
        return noise - torch.mean(noise, dim=-2, keepdims=True)

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
    ) -> torch.Tensor:
        """Corrupt the continuous input tensor `x_1` using the noise schedule defined by `t`."""
        x_0 = self._centered_gaussian(*x_1.shape)
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
                - cell (torch.Tensor): Clean (continuous) cell tensor.
                - node_is_periodic (torch.Tensor): Node periodicity mask tensor.
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
        """
        assert all(feat in batch for feat in self.feats), (
            f"Batch must contain at least the following features: {self.feats}, "
            f"but got: {list(batch.keys())}"
        )

        noisy_batch = copy.deepcopy(batch)

        # Corrupt features according to their data modality
        # (i.e., discrete or `disc` and continuous or `cont`)
        for feat in self.feats:
            # Handle special features
            if feat == "frac_coords" and self.corrupt:
                assert all(ft in batch for ft in ["pos", "cell"]), (
                    f"Batch must contain at least the following features to corrupt `frac_coords`: [`pos`, `cell`], "
                    f"but got: {list(batch.keys())}"
                )

                # Compute new fractional coordinates for samples which are periodic
                cell_per_node_inv = torch.linalg.inv(
                    batch["cell"][batch["batch"]][batch["node_is_periodic"]]
                )
                cell_all_node_inv = torch.zeros_like(batch["cell"][batch["batch"]])
                cell_all_node_inv[batch["node_is_periodic"]] = cell_per_node_inv
                dense_cell_all_node_inv, _ = to_dense_batch(
                    cell_all_node_inv,
                    batch["batch"],
                    max_num_nodes=self.max_num_nodes,
                )

                frac_coords_aug = torch.einsum(
                    "bni,bnij->bnj",
                    noisy_batch["pos"],
                    dense_cell_all_node_inv,
                )
                frac_coords_aug = frac_coords_aug % 1.0

                # Densify fractional coordinates
                noisy_batch["node_is_periodic"], _ = to_dense_batch(
                    noisy_batch["node_is_periodic"],
                    batch["batch"],
                    max_num_nodes=self.max_num_nodes,
                )
                noisy_batch["frac_coords"], _ = to_dense_batch(
                    noisy_batch["frac_coords"], batch["batch"], max_num_nodes=self.max_num_nodes
                )
                noisy_batch["frac_coords"] = torch.where(
                    (noisy_batch["token_mask"] & noisy_batch["node_is_periodic"]).unsqueeze(-1),
                    frac_coords_aug,
                    noisy_batch["frac_coords"],
                )

            # Handle all other features
            else:
                # [B, N, d]
                x_1 = batch[feat]

                # Convert from PyG batch to dense batch (potentially with fixed-length max padding to stabilize GPU memory usage)
                is_global_feat = x_1.shape[0] == batch.batch_size
                x_1, mask = (
                    (
                        # NOTE: Global features do not need to be densely padded
                        x_1.unsqueeze(-2),
                        torch.ones((batch.batch_size, 1), device=x_1.device, dtype=torch.bool),
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
                    x_t = self._corrupt_cont_x(x_1, t, token_mask, diffuse_mask)

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
