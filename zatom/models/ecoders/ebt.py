"""Energy-based transformer (EBT).

Adapted from:
    - https://github.com/alexiglad/EBT
    - https://github.com/facebookresearch/all-atom-diffusion-transformer
"""

import math
from typing import Any, Dict, List, Literal, Tuple

import torch
import torch.nn as nn
from torch._C import _SDPBackend as SDPBackend

from zatom.utils.typing_utils import typecheck

#################################################################################
#                             Embedding Layers                                  #
#################################################################################


class LabelEmbedder(nn.Module):
    """Embed class labels into vector representations.

    NOTE: Also handles label dropout for context conditioning.

    Args:
        num_classes: The number of classes.
        hidden_dim: The dimensionality of the hidden representations.
        dropout_prob: The dropout probability for context conditioning.
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
        """Drop labels to enable context conditioning.

        Args:
            labels: The input labels tensor.
            force_drop_ids: Optional tensor indicating which labels to drop.

        Returns:
            The modified labels tensor.
        """
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
        """Forward pass for label embedding.

        Args:
            labels: The input labels tensor.
            train: Whether the model is in training mode.
            force_drop_ids: Optional tensor indicating which labels to drop.

        Returns:
            The output embeddings tensor.
        """
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
    )
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    )
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


#################################################################################
#                               Transformer blocks                              #
#################################################################################


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks.

    Args:
        in_features: The number of input features.
        hidden_features: The number of hidden features.
        out_features: The number of output features.
        act_layer: The activation layer to use.
        norm_layer: The normalization layer to use.
        bias: Whether to use bias in the linear layers.
        drop: The dropout rate.
    """

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
        """Forward pass for the MLP.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
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
    """Modulate the input tensor x with the given shift and scale.

    Args:
        x: The input tensor.
        shift: The shift tensor.
        scale: The scale tensor.

    Returns:
        The modulated tensor.
    """
    # NOTE: This is global modulation.
    # TODO: Explore per-token modulation.
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class EBTBlock(nn.Module):
    """An EBT block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Args:
        hidden_dim: The dimensionality of the hidden representations.
        num_heads: The number of attention heads.
        mlp_ratio: The ratio of the MLP hidden dimension to the input dimension.
        block_kwargs: Additional keyword arguments for the block.
    """

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
        """Forward pass for the EBT block.

        Args:
            x: The input tensor.
            c: The context tensor.
            mask: The attention mask tensor.

        Returns:
            The output tensor.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        with torch.nn.attention.sdpa_kernel(
            backends=[SDPBackend.MATH]
        ):  # NOTE: May want to turn this off for inference eventually
            attn_results = self.attn(_x, _x, _x, key_padding_mask=mask, need_weights=False)[
                0
            ]  # NOTE: Need to set this, as regular SDPA from PyTorch doesn't support higher order gradients here
        x = x + gate_msa.unsqueeze(1) * attn_results
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of EBT.

    Args:
        hidden_dim: The dimensionality of the hidden representations.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_dim, 1, bias=False
        )  # NOTE: Changed this to output single scalar energy. Sum of energies of each embed will be energy function per sample. The `bias` argument must be `False`, since this is an EBM and a relative energy value doesn't affect reconstruction.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    @typecheck
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass for the final EBT layer.

        Args:
            x: The input tensor.
            c: The context tensor.

        Returns:
            The output tensor.
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class EBT(nn.Module):
    """Energy-based model with a Transformer encoder/decoder (i.e., an E-coder or `ecoder`).

    NOTE: The `_forward` pass of this model is conceptually similar to that of All-atom Diffusion Transformers (ADiTs)
    except that there is no self/time conditioning and the model outputs a single energy scalar for each example.

    Args:
        encoder: The encoder module.
        d_x: Input dimension.
        d_model: Model dimension.
        num_layers: Number of Transformer layers.
        nhead: Number of attention heads.
        mcmc_num_steps: Number of MCMC steps.
        mcmc_step_size: Markov chain Monte Carlo (MCMC) step size.
        randomize_mcmc_num_steps: Number of steps to randomize MCMC.
        randomize_mcmc_num_steps_min: Minimum number of steps to randomize MCMC.
        randomize_mcmc_step_size_scale: Scale factor for randomizing MCMC step size.
        num_datasets: Number of datasets for context conditioning.
        num_spacegroups: Number of spacegroups for context conditioning.
        mlp_ratio: Ratio of hidden to input dimension in MLP.
        class_dropout_prob: Probability of dropping class labels for context conditioning.
        langevin_dynamics_noise: Standard deviation of Langevin dynamics noise.
        clamp_futures_grad_max_change: Maximum change for clamping future gradients.
        truncate_mcmc: Whether to truncate MCMC samples.
        clamp_futures_grad: Whether to clamp future gradients.
        no_mcmc_detach: Whether to detach MCMC samples from the graph.
        no_langevin_during_eval: Whether to disable Langevin dynamics during evaluation.
        mcmc_step_size_learnable: Whether to make the MCMC step size learnable.
        randomize_mcmc_num_steps_final_landscape: Whether to randomize MCMC steps for the final landscape.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_x: int = 512,
        d_model: int = 768,
        num_layers: int = 12,
        nhead: int = 12,
        mcmc_num_steps: int = 1,
        mcmc_step_size: int = 3000,
        randomize_mcmc_num_steps: int = 2,
        randomize_mcmc_num_steps_min: int = 2,
        randomize_mcmc_step_size_scale: int = 2,
        num_datasets: int = 2,  # Context conditioning input
        num_spacegroups: int = 230,  # Context conditioning input
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        langevin_dynamics_noise: float = 0.0,
        clamp_futures_grad_max_change: float = 9.0,
        truncate_mcmc: bool = True,
        clamp_futures_grad: bool = False,
        no_mcmc_detach: bool = True,
        no_langevin_during_eval: bool = False,
        mcmc_step_size_learnable: bool = False,
        randomize_mcmc_num_steps_final_landscape: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.mcmc_num_steps = mcmc_num_steps
        self.randomize_mcmc_num_steps = randomize_mcmc_num_steps
        self.randomize_mcmc_num_steps_min = randomize_mcmc_num_steps_min
        self.randomize_mcmc_step_size_scale = randomize_mcmc_step_size_scale
        self.randomize_mcmc_num_steps_final_landscape = randomize_mcmc_num_steps_final_landscape
        self.langevin_dynamics_noise = langevin_dynamics_noise
        self.clamp_futures_grad_max_change = clamp_futures_grad_max_change
        self.truncate_mcmc = truncate_mcmc
        self.clamp_futures_grad = clamp_futures_grad
        self.no_mcmc_detach = no_mcmc_detach
        self.no_langevin_during_eval = no_langevin_during_eval

        self.alpha = nn.Parameter(
            torch.tensor(float(mcmc_step_size)), requires_grad=mcmc_step_size_learnable
        )
        self.langevin_dynamics_noise_std = nn.Parameter(
            torch.tensor(float(langevin_dynamics_noise)), requires_grad=False
        )

        self.x_embedder = nn.Linear(d_x, d_model, bias=True)
        self.dataset_embedder = LabelEmbedder(num_datasets, d_model, class_dropout_prob)
        self.spacegroup_embedder = LabelEmbedder(num_spacegroups, d_model, class_dropout_prob)
        self.learnable_embedder = nn.Embedding(1, d_model)

        self.blocks = nn.ModuleList(
            [EBTBlock(d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(d_model)
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

        # Zero-out adaLN modulation layers in EBT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)  # NOTE: Turned off bias for final layer of EBT

    @typecheck
    def _forward(
        self,
        atom_types: torch.Tensor,
        pos: torch.Tensor,
        frac_coords: torch.Tensor,
        dataset_idx: torch.Tensor,
        spacegroup: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of EBT.

        Args:
            atom_types: Atom types tensor (B, N).
            pos: Atom positions tensor (B, N, 3).
            frac_coords: Fractional coordinates tensor (B, N, 3).
            dataset_idx: Dataset index for each sample (B,).
            spacegroup: Spacegroup index for each sample (B,).
            mask: True if valid token, False if padding (B, N).

        Returns:
            Output energy tensor (B, N, 1).
        """
        # Positional embedding
        token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        pos_emb = get_pos_embedding(token_index, self.d_model)

        # Input embeddings: (B, N, d)
        x_encoding = self.encoder(atom_types, pos, frac_coords, token_index, mask)
        x = self.x_embedder(x_encoding) + pos_emb

        # Conditioning embeddings
        d = self.dataset_embedder(dataset_idx, self.training)  # (B, d)
        s = self.spacegroup_embedder(spacegroup, self.training)  # (B, d)
        lt = self.learnable_embedder(torch.zeros_like(token_index))
        c = d + s + lt  # (B, 1, d)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, ~mask)  # (B, N, d)

        # Prediction layer
        x = self.final_layer(x, c)  # (B, N, d_out)
        x = x * mask[..., None]
        return x

    @typecheck
    def forward(
        self,
        atom_types: torch.Tensor,
        pos: torch.Tensor,
        frac_coords: torch.Tensor,
        dataset_idx: torch.Tensor,
        spacegroup: torch.Tensor,
        mask: torch.Tensor,
        cell_per_node_inv: torch.Tensor,
        token_is_periodic: torch.Tensor,
        training: bool,
        no_randomness: bool = True,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
        """MCMC-driven forward pass of EBT.

        Args:
            atom_types: Atom types tensor (B, N).
            pos: Atom positions tensor (B, N, 3).
            frac_coords: Fractional coordinates tensor (B, N, 3).
            dataset_idx: Dataset index for each sample (B,).
            spacegroup: Spacegroup index for each sample (B,).
            mask: True if valid token, False if padding (B, N).
            cell_per_node_inv: Inverse cell tensor for periodic boundary conditions (flat(B * N, periodic_nodes_only), 3, 3).
            token_is_periodic: Boolean mask indicating periodic tokens (flat(B * N)).
            training: If True, enables computation graph tracking in final MCMC step.
            no_randomness: If True, disables randomness in MCMC steps.

        Returns:
            A tuple of a list of predicted modalities as a dictionary and a list of their predicted (scalar) energy values.
        """
        batch_size = atom_types.shape[0]
        pred_modals_list, pred_energies_list = [], []

        # Initialize predicted modalities
        pred_atom_types = atom_types.clone().detach()
        pred_pos = pos.clone().detach()
        pred_frac_coords = frac_coords.clone().detach()

        # Initialize `alpha` argument
        alpha = torch.clamp(self.alpha, min=0.0001)
        if not no_randomness and self.randomize_mcmc_step_size_scale != 1:
            expanded_alpha = alpha.expand(batch_size, 1)

            scale = self.randomize_mcmc_step_size_scale
            low = alpha / scale
            high = alpha * scale
            alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        mcmc_steps = (
            []
        )  # NOTE: In the general case where `randomize_mcmc_num_steps is False`, this matches length of `self.randomize_mcmc_num_steps`
        for step in range(self.mcmc_num_steps):
            if not no_randomness and self.randomize_mcmc_num_steps > 0:
                if (
                    self.randomize_mcmc_num_steps_final_landscape
                ):  # Only apply random steps to final landscape
                    if step == (self.mcmc_num_steps - 1):
                        min_steps = (
                            1
                            if self.randomize_mcmc_num_steps_min == 0
                            else self.randomize_mcmc_num_steps_min
                        )
                        repeats = torch.randint(
                            min_steps, self.randomize_mcmc_num_steps + 2, (1,)
                        ).item()
                        mcmc_steps.extend([step] * repeats)
                    else:
                        mcmc_steps.append(step)
                else:
                    min_steps = (
                        1
                        if self.randomize_mcmc_num_steps_min == 0
                        else self.randomize_mcmc_num_steps_min
                    )
                    repeats = torch.randint(
                        min_steps, self.randomize_mcmc_num_steps + 2, (1,)
                    ).item()
                    mcmc_steps.extend([step] * repeats)

            elif (
                no_randomness
                and self.randomize_mcmc_num_steps
                and self.randomize_mcmc_num_steps > 0
            ):  # Use max steps
                if step == (self.mcmc_num_steps - 1):
                    # NOTE: Empirically, this may be a better (i.e., more stable) pretraining metric by
                    # doing several steps only on final energy landscape instead of over all energy landscapes
                    mcmc_steps.extend([step] * (self.randomize_mcmc_num_steps + 1))
                else:
                    mcmc_steps.append(step)

            else:
                mcmc_steps.append(step)

        with torch.set_grad_enabled(mode=True):  # Enable gradient tracking
            for i, _ in enumerate(mcmc_steps):
                if self.no_mcmc_detach:
                    pred_atom_types = pred_atom_types.requires_grad_()
                    pred_pos = pred_pos.requires_grad_()
                    pred_frac_coords = pred_frac_coords.requires_grad_()
                else:
                    pred_atom_types = pred_atom_types.detach().requires_grad_()
                    pred_pos = pred_pos.detach().requires_grad_()
                    pred_frac_coords = pred_frac_coords.detach().requires_grad_()

                # Langevin dynamics
                if self.langevin_dynamics_noise != 0 and not (
                    no_randomness and self.no_langevin_during_eval
                ):
                    ld_noise = (
                        torch.randn_like(pred_pos.detach(), device=pred_pos.device)
                        * langevin_dynamics_noise_std
                    )
                    pred_pos = pred_pos + ld_noise

                    # Update fractional coordinates according to (de)noised atom positions
                    frac_coords_aug = torch.einsum(
                        "bi,bij->bj",
                        pred_pos[mask][token_is_periodic],
                        cell_per_node_inv,
                    )
                    frac_coords_aug = frac_coords_aug % 1.0
                    pred_frac_coords[mask][token_is_periodic] = frac_coords_aug

                pred_energy = (
                    self._forward(
                        pred_atom_types,
                        pred_pos,
                        pred_frac_coords,
                        dataset_idx,
                        spacegroup,
                        mask,
                    )
                    .squeeze(-1)
                    .mean(-1)
                )

                # Retain computation graph conditionally
                is_last_mcmc_step = i == (len(mcmc_steps) - 1)
                if self.truncate_mcmc:
                    pred_atom_types_grad, pred_pos_grad = torch.autograd.grad(
                        [pred_energy.sum()],
                        [pred_atom_types, pred_pos],
                        create_graph=training and is_last_mcmc_step,
                    )
                else:
                    pred_atom_types_grad, pred_pos_grad = torch.autograd.grad(
                        [pred_energy.sum()], [pred_atom_types, pred_pos], create_graph=training
                    )

                # Maybe clamp gradients
                if self.clamp_futures_grad:
                    min_and_max = self.clamp_futures_grad_max_change / (self.alpha)
                    pred_atom_types_grad = torch.clamp(
                        pred_atom_types_grad, min=-min_and_max, max=min_and_max
                    )
                    pred_pos_grad = torch.clamp(pred_pos_grad, min=-min_and_max, max=min_and_max)

                if (
                    torch.isnan(pred_atom_types_grad).any()
                    or torch.isinf(pred_atom_types_grad).any()
                ):
                    raise ValueError("NaN or Inf gradients detected for atom types during MCMC.")
                if torch.isnan(pred_pos_grad).any() or torch.isinf(pred_pos_grad).any():
                    raise ValueError(
                        "NaN or Inf gradients detected for atom positions during MCMC."
                    )

                # pred_atom_types = pred_atom_types - alpha * pred_atom_types_grad # TODO: Update atom types in embedding space
                pred_pos = pred_pos - alpha * pred_pos_grad

                # Update fractional coordinates according to (de)noised atom positions
                frac_coords_aug = torch.einsum(
                    "bi,bij->bj",
                    pred_pos[mask][token_is_periodic],
                    cell_per_node_inv,
                )
                frac_coords_aug = frac_coords_aug % 1.0
                pred_frac_coords[mask][token_is_periodic] = frac_coords_aug

                pred_energies_list.append(pred_energy)
                pred_modals_list.append(
                    {
                        "atom_types": pred_atom_types,
                        "pos": pred_pos,
                        "frac_coords": pred_frac_coords,
                    }
                )

        return pred_modals_list, pred_energies_list

    @typecheck
    def forward_with_loss_wrapper(
        self,
        atom_types: torch.Tensor,
        pos: torch.Tensor,
        frac_coords: torch.Tensor,
        dataset_idx: torch.Tensor,
        spacegroup: torch.Tensor,
        mask: torch.Tensor,
        cell_per_node_inv: torch.Tensor,
        token_is_periodic: torch.Tensor,
        phase: Literal["train", "sanity_check", "validate", "test", "predict"] = "train",
    ) -> Dict[str, torch.Tensor]:
        """MCMC-driven forward pass of EBT with loss calculation.

        Args:
            atom_types: Atom types tensor (B, N).
            pos: Atom positions tensor (B, N, 3).
            frac_coords: Fractional coordinates tensor (B, N, 3).
            dataset_idx: Dataset index for each sample (B,).
            spacegroup: Spacegroup index for each sample (B,).
            mask: True if valid token, False if padding (B, N).
            cell_per_node_inv: Inverse cell tensor for periodic boundary conditions (flat(B * N, periodic_nodes_only), 3, 3).
            token_is_periodic: Boolean mask indicating periodic tokens (flat(B * N)).
            phase: Current phase of the model (train, sanity_check, validate, test, predict).

        Returns:
            Dictionary of loss values.
        """
        no_randomness = False if phase == "train" else True
        training = phase == "train"

        # Denoise (generate) modalities via energy minimization
        denoised_modals_list, pred_energies_list = self.forward(
            atom_types,
            pos,
            frac_coords,
            dataset_idx,
            spacegroup,
            mask,
            cell_per_node_inv,
            token_is_periodic,
            training=training,
            no_randomness=no_randomness,
        )

        # TODO: Calculate each loss value
        total_loss = initial_loss = final_reconstruction_loss = initial_final_pred_energies_gap = (
            l1_loss
        ) = 0.0

        loss_dict = {
            "loss": total_loss,
            "initial_loss": initial_loss,
            "final_step_loss": final_reconstruction_loss,
            "initial_final_pred_energies_gap": initial_final_pred_energies_gap,
            "l1_loss": l1_loss,
        }

        return loss_dict
