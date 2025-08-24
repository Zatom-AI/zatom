"""Energy-based transformer (EBT).

Adapted from:
    - https://github.com/alexiglad/EBT
    - https://github.com/facebookresearch/all-atom-diffusion-transformer
"""

import math
from typing import Any, Dict, List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from zatom.utils.training_utils import initialize_module_weights, weighted_rigid_align
from zatom.utils.typing_utils import typecheck

# Optimize common operations
weighted_rigid_align = torch.compile(weighted_rigid_align)

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
        with sdpa_kernel(SDPBackend.MATH):
            # NOTE: May need to use this context, as regular SDPA from PyTorch
            # may not support higher order gradients (e.g., for CUDA devices).
            # NOTE: May want to turn this off for inference eventually.
            attn_results = self.attn(_x, _x, _x, key_padding_mask=mask, need_weights=False)[0]
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
        max_num_elements: Maximum number of elements in the dataset.
        mlp_ratio: Ratio of hidden to input dimension in MLP.
        class_dropout_prob: Probability of dropping class labels for context conditioning.
        langevin_dynamics_noise: Standard deviation of Langevin dynamics noise.
        weight_initialization_gain: Gain for discrete embedding weight initialization.
        clamp_futures_grad_max_change: Maximum change for clamping future gradients.
        discrete_gaussian_random_noise_scaling: Scale factor for discrete Gaussian random noise.
        discrete_absolute_clamp: Maximum absolute value for discrete predictions.
        sharpen_predicted_discrete_distribution: Sharpening factor for predicted discrete distributions.
        atom_types_reconstruction_loss_weight: Weighting factor for the atom types reconstruction loss.
        pos_reconstruction_loss_weight: Weighting factor for the atom positions reconstruction loss.
        frac_coords_reconstruction_loss_weight: Weighting factor for the atom fractional coordinates reconstruction loss.
        lengths_scaled_reconstruction_loss_weight: Weighting factor for the atom lengths (scaled) reconstruction loss.
        angles_radians_reconstruction_loss_weight: Weighting factor for the atom angles (radians) reconstruction loss.
        truncate_mcmc: Whether to truncate MCMC sample gradient trajectories.
        clamp_futures_grad: Whether to clamp future gradients.
        no_mcmc_detach: Whether to detach MCMC samples from the graph.
        no_langevin_during_eval: Whether to disable Langevin dynamics during evaluation.
        no_randomize_mcmc_step_size_scale_during_eval: Whether to disable randomizing MCMC step size scale during evaluation.
        mcmc_step_size_learnable: Whether to make the MCMC step size learnable.
        langevin_dynamics_noise_learnable: Whether to make the Langevin dynamics noise learnable.
        randomize_mcmc_num_steps_final_landscape: Whether to randomize MCMC steps for the final landscape.
        normalize_discrete_initial_condition: Whether to normalize discrete initial embeddings using softmax.
        weighted_rigid_align_pos: Whether to apply weighted rigid alignment between target and predicted atom positions for loss calculation.
        weighted_rigid_align_frac_coords: Whether to apply weighted rigid alignment between target and predicted atom fractional coordinates for loss calculation.
        add_mask_atom_type: Whether to add a mask token for atom types.
        weight_initialization_method: Initialization method for discrete embedding weights.
        discrete_denoising_initial_condition: Whether to use random or zero-based discrete denoising for initial conditions.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_x: int = 512,
        d_model: int = 768,
        num_layers: int = 12,
        nhead: int = 12,
        mcmc_num_steps: int = 2,
        mcmc_step_size: int = 5,
        randomize_mcmc_num_steps: int = 1,
        randomize_mcmc_num_steps_min: int = 1,
        randomize_mcmc_step_size_scale: int = 2,
        num_datasets: int = 2,  # Context conditioning input
        num_spacegroups: int = 230,  # Context conditioning input
        max_num_elements: int = 100,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        langevin_dynamics_noise: float = 0.1,
        weight_initialization_gain: float = 1.0,
        clamp_futures_grad_max_change: float = 9.0,
        discrete_gaussian_random_noise_scaling: float = 1.0,
        discrete_absolute_clamp: float = 0.0,
        sharpen_predicted_discrete_distribution: float = 0.0,
        atom_types_reconstruction_loss_weight: float = 1.0,
        pos_reconstruction_loss_weight: float = 10.0,
        frac_coords_reconstruction_loss_weight: float = 10.0,
        lengths_scaled_reconstruction_loss_weight: float = 1.0,
        angles_radians_reconstruction_loss_weight: float = 10.0,
        truncate_mcmc: bool = False,
        clamp_futures_grad: bool = False,
        no_mcmc_detach: bool = True,
        no_langevin_during_eval: bool = False,
        no_randomize_mcmc_step_size_scale_during_eval: bool = False,
        mcmc_step_size_learnable: bool = False,
        langevin_dynamics_noise_learnable: bool = False,
        randomize_mcmc_num_steps_final_landscape: bool = False,
        normalize_discrete_initial_condition: bool = True,
        weighted_rigid_align_pos: bool = True,
        weighted_rigid_align_frac_coords: bool = False,
        add_mask_atom_type: bool = True,
        weight_initialization_method: Literal["he", "xavier"] = "xavier",
        discrete_denoising_initial_condition: Literal["random", "zeros"] = "random",
    ):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.mcmc_num_steps = mcmc_num_steps
        self.randomize_mcmc_num_steps = randomize_mcmc_num_steps
        self.randomize_mcmc_num_steps_min = randomize_mcmc_num_steps_min
        self.randomize_mcmc_step_size_scale = randomize_mcmc_step_size_scale
        self.randomize_mcmc_num_steps_final_landscape = randomize_mcmc_num_steps_final_landscape
        self.class_dropout_prob = class_dropout_prob
        self.langevin_dynamics_noise = langevin_dynamics_noise
        self.clamp_futures_grad_max_change = clamp_futures_grad_max_change
        self.discrete_gaussian_random_noise_scaling = discrete_gaussian_random_noise_scaling
        self.discrete_absolute_clamp = discrete_absolute_clamp
        self.sharpen_predicted_discrete_distribution = sharpen_predicted_discrete_distribution
        self.atom_types_reconstruction_loss_weight = atom_types_reconstruction_loss_weight
        self.pos_reconstruction_loss_weight = pos_reconstruction_loss_weight
        self.frac_coords_reconstruction_loss_weight = frac_coords_reconstruction_loss_weight
        self.lengths_scaled_reconstruction_loss_weight = lengths_scaled_reconstruction_loss_weight
        self.angles_radians_reconstruction_loss_weight = angles_radians_reconstruction_loss_weight
        self.truncate_mcmc = truncate_mcmc
        self.clamp_futures_grad = clamp_futures_grad
        self.no_mcmc_detach = no_mcmc_detach
        self.no_langevin_during_eval = no_langevin_during_eval
        self.no_randomize_mcmc_step_size_scale_during_eval = (
            no_randomize_mcmc_step_size_scale_during_eval
        )
        self.normalize_discrete_initial_condition = normalize_discrete_initial_condition
        self.weighted_rigid_align_pos = weighted_rigid_align_pos
        self.weighted_rigid_align_frac_coords = weighted_rigid_align_frac_coords
        self.discrete_denoising_initial_condition = discrete_denoising_initial_condition

        self.alpha = nn.Parameter(
            torch.tensor(float(mcmc_step_size)), requires_grad=mcmc_step_size_learnable
        )
        self.langevin_dynamics_noise_std = nn.Parameter(
            torch.tensor(float(langevin_dynamics_noise)),
            requires_grad=langevin_dynamics_noise_learnable,
        )

        self.vocab_size = max_num_elements + int(add_mask_atom_type)
        self.atom_type_embedder = nn.Embedding(self.vocab_size, d_model)
        self.atom_type_vocab_to_embedding = nn.Linear(
            self.vocab_size, d_model, bias=False
        )  # NOTE: This is special to EBTs, since we want to input a probability distribution and predict this distribution yet EBTs need an embedding as input

        self.x_embedder = nn.Linear(d_x, d_model, bias=True)
        self.dataset_embedder = LabelEmbedder(num_datasets, d_model, class_dropout_prob)
        self.spacegroup_embedder = LabelEmbedder(num_spacegroups, d_model, class_dropout_prob)
        self.learnable_embedder = nn.Embedding(1, d_model)

        self.blocks = nn.ModuleList(
            [EBTBlock(d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(d_model)
        self.initialize_weights()

        # Initialize discrete embedding weights distinctly
        initialize_module_weights(
            self.atom_type_embedder,
            weight_initialization_method,
            weight_initialization_gain=weight_initialization_gain,
        )
        initialize_module_weights(
            self.atom_type_vocab_to_embedding,
            weight_initialization_method,
            weight_initialization_gain=weight_initialization_gain,
        )

        # Instantiate losses
        self.nll_loss = nn.NLLLoss(ignore_index=-100, reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

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
        lengths_scaled: torch.Tensor,
        angles_radians: torch.Tensor,
        dataset_idx: torch.Tensor,
        spacegroup: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of EBT.

        Args:
            atom_types: Combined input and predicted atom type embeddings tensor (B, N, D * 2).
            pos: Atom positions tensor (B, N, 3).
            frac_coords: Fractional coordinates tensor (B, N, 3).
            lengths_scaled: Scaled lengths tensor (B, 1, 3).
            angles_radians: Radian angles tensor (B, 1, 3).
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
        x_encoding = self.encoder(
            atom_types, pos, frac_coords, lengths_scaled, angles_radians, token_index, mask
        )
        x = self.x_embedder(x_encoding) + pos_emb

        # Conditioning embeddings
        d = self.dataset_embedder(dataset_idx, self.training)  # (B, d)
        s = self.spacegroup_embedder(spacegroup, self.training)  # (B, d)
        lt = self.learnable_embedder(torch.zeros_like(token_index[..., 0]))
        c = d + s + lt  # (B, d)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, ~mask)  # (B, N, d)

        # Prediction layer
        x = self.final_layer(x, c)  # (B, N, d_out)
        x = x * mask.unsqueeze(-1)  # Mask out padding tokens
        return x

    @typecheck
    def _corrupt_discrete_types(self, token_types: torch.Tensor) -> torch.Tensor:
        """Corrupt discrete token types by creating an initial noisy or zero-based tokens tensor.

        Args:
            token_types: The input token types tensor (B, S).

        Returns:
            The corrupted token types embedding tensor (B, S, V) with no gradients attached.
        """
        if self.discrete_denoising_initial_condition == "random":
            predicted_tokens = (
                torch.randn(
                    size=(token_types.shape[0], token_types.shape[1], self.vocab_size),
                    device=token_types.device,
                )
                * self.discrete_gaussian_random_noise_scaling
            )
        elif self.discrete_denoising_initial_condition == "zeros":
            predicted_tokens = torch.zeros(
                size=(token_types.shape[0], token_types.shape[1], self.vocab_size),
                device=token_types.device,
            )
        else:
            raise NotImplementedError(
                f"{self.discrete_denoising_initial_condition} option for `discrete_denoising_initial_condition` is not yet supported."
            )

        return predicted_tokens

    @typecheck
    def forward(
        self,
        atom_types: torch.Tensor,
        pos: torch.Tensor,
        frac_coords: torch.Tensor,
        lengths_scaled: torch.Tensor,
        angles_radians: torch.Tensor,
        dataset_idx: torch.Tensor,
        spacegroup: torch.Tensor,
        mask: torch.Tensor,
        training: bool,
        no_randomness: bool = True,
        return_raw_discrete_logits: bool = False,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
        """MCMC-driven forward pass of EBT.

        Args:
            atom_types: Atom types tensor (B, N).
            pos: Atom positions tensor (B, N, 3).
            frac_coords: Fractional coordinates tensor (B, N, 3).
            lengths_scaled: Lengths scaled tensor (B, 1, 3).
            angles_radians: Angles in radians tensor (B, 1, 3).
            dataset_idx: Dataset index for each sample (B,).
            spacegroup: Spacegroup index for each sample (B,).
            mask: True if valid token, False if padding (B, N).
            training: If True, enables computation graph tracking in final MCMC step.
            no_randomness: If True, disables randomness in MCMC steps.
            return_raw_discrete_logits: If True, returns raw logits instead of log-probabilities.

        Returns:
            A tuple of a list of predicted modalities as a dictionary and a list of their predicted (scalar) energy values.
        """
        batch_size = atom_types.shape[0]
        pred_modals_list, pred_energies_list = [], []

        # Initialize predicted modalities
        atom_types_input_embedding = self.atom_type_embedder(atom_types)
        pred_pos = pos.clone().detach()
        pred_frac_coords = frac_coords.clone().detach()
        pred_lengths_scaled = lengths_scaled.clone().detach()
        pred_angles_radians = angles_radians.clone().detach()

        pred_atom_types = self._corrupt_discrete_types(atom_types) * mask.unsqueeze(
            -1
        )  # [B, S, V]

        # Initialize `alpha` argument
        alpha = torch.clamp(self.alpha, min=0.0001)
        if self.randomize_mcmc_step_size_scale != 1 and not (
            no_randomness and self.no_randomize_mcmc_step_size_scale_during_eval
        ):
            expanded_alpha = alpha.expand(batch_size, 1, 1)

            scale = self.randomize_mcmc_step_size_scale
            low = alpha / scale
            high = alpha * scale
            alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        mcmc_steps = (
            []
        )  # NOTE: In the general case where `randomize_mcmc_num_steps is False`, this matches length of `self.mcmc_num_steps`
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

            elif no_randomness and self.randomize_mcmc_num_steps > 0:  # Use max steps
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
                    pred_lengths_scaled = pred_lengths_scaled.requires_grad_()
                    pred_angles_radians = pred_angles_radians.requires_grad_()
                else:
                    pred_atom_types = pred_atom_types.detach().requires_grad_()
                    pred_pos = pred_pos.detach().requires_grad_()
                    pred_frac_coords = pred_frac_coords.detach().requires_grad_()
                    pred_lengths_scaled = pred_lengths_scaled.detach().requires_grad_()
                    pred_angles_radians = pred_angles_radians.detach().requires_grad_()

                # Langevin dynamics
                if self.langevin_dynamics_noise != 0 and not (
                    no_randomness and self.no_langevin_during_eval
                ):
                    ld_atom_types_noise = (
                        torch.randn_like(pred_atom_types.detach(), device=pred_atom_types.device)
                        * langevin_dynamics_noise_std
                    )
                    ld_pos_noise = (
                        torch.randn_like(pred_pos.detach(), device=pred_pos.device)
                        * langevin_dynamics_noise_std
                    )
                    ld_frac_coords_noise = (
                        torch.randn_like(pred_frac_coords.detach(), device=pred_frac_coords.device)
                        * langevin_dynamics_noise_std
                    )
                    ld_lengths_scaled_noise = (
                        torch.randn_like(
                            pred_lengths_scaled.detach(), device=pred_lengths_scaled.device
                        )
                        * langevin_dynamics_noise_std
                    )
                    ld_angles_radians_noise = (
                        torch.randn_like(
                            pred_angles_radians.detach(), device=pred_angles_radians.device
                        )
                        * langevin_dynamics_noise_std
                    )

                    pred_atom_types = pred_atom_types + ld_atom_types_noise
                    pred_pos = pred_pos + ld_pos_noise
                    pred_frac_coords = pred_frac_coords + ld_frac_coords_noise
                    pred_lengths_scaled = pred_lengths_scaled + ld_lengths_scaled_noise
                    pred_angles_radians = pred_angles_radians + ld_angles_radians_noise

                # Combine input and current discrete modality embeddings
                if self.normalize_discrete_initial_condition:
                    pred_atom_types = F.softmax(pred_atom_types, dim=-1)

                pred_atom_type_embedding = self.atom_type_vocab_to_embedding(pred_atom_types)
                pred_atom_types_embeddings = torch.cat(
                    (atom_types_input_embedding, pred_atom_type_embedding), dim=-1
                )  # [B, S, D * 2]

                pred_energy = self._forward(
                    pred_atom_types_embeddings,
                    pred_pos,
                    pred_frac_coords,
                    pred_lengths_scaled,
                    pred_angles_radians,
                    dataset_idx,
                    spacegroup,
                    mask,
                )

                # Retain computation graph conditionally
                is_last_mcmc_step = i == (len(mcmc_steps) - 1)
                if self.truncate_mcmc:
                    (
                        pred_atom_types_grad,
                        pred_pos_grad,
                        pred_frac_coords_grad,
                        pred_lengths_scaled_grad,
                        pred_angles_radians_grad,
                    ) = torch.autograd.grad(
                        [pred_energy.sum()],
                        [
                            pred_atom_types,
                            pred_pos,
                            pred_frac_coords,
                            pred_lengths_scaled,
                            pred_angles_radians,
                        ],
                        create_graph=training and is_last_mcmc_step,
                    )
                else:
                    (
                        pred_atom_types_grad,
                        pred_pos_grad,
                        pred_frac_coords_grad,
                        pred_lengths_scaled_grad,
                        pred_angles_radians_grad,
                    ) = torch.autograd.grad(
                        [pred_energy.sum()],
                        [
                            pred_atom_types,
                            pred_pos,
                            pred_frac_coords,
                            pred_lengths_scaled,
                            pred_angles_radians,
                        ],
                        create_graph=training,
                    )

                # Maybe clamp gradients
                if self.clamp_futures_grad:
                    min_and_max = self.clamp_futures_grad_max_change / (self.alpha)
                    pred_atom_types_grad = torch.clamp(
                        pred_atom_types_grad, min=-min_and_max, max=min_and_max
                    )
                    pred_pos_grad = torch.clamp(pred_pos_grad, min=-min_and_max, max=min_and_max)
                    pred_frac_coords_grad = torch.clamp(
                        pred_frac_coords_grad, min=-min_and_max, max=min_and_max
                    )
                    pred_lengths_scaled_grad = torch.clamp(
                        pred_lengths_scaled_grad, min=-min_and_max, max=min_and_max
                    )
                    pred_angles_radians_grad = torch.clamp(
                        pred_angles_radians_grad, min=-min_and_max, max=min_and_max
                    )

                if (
                    torch.isnan(pred_atom_types_grad).any()
                    or torch.isinf(pred_atom_types_grad).any()
                ):
                    raise ValueError("NaN or Inf gradients detected for atom types during MCMC.")
                if torch.isnan(pred_pos_grad).any() or torch.isinf(pred_pos_grad).any():
                    raise ValueError(
                        "NaN or Inf gradients detected for atom positions during MCMC."
                    )
                if (
                    torch.isnan(pred_frac_coords_grad).any()
                    or torch.isinf(pred_frac_coords_grad).any()
                ):
                    raise ValueError(
                        "NaN or Inf gradients detected for fractional coordinates during MCMC."
                    )
                if (
                    torch.isnan(pred_lengths_scaled_grad).any()
                    or torch.isinf(pred_lengths_scaled_grad).any()
                ):
                    raise ValueError(
                        "NaN or Inf gradients detected for scaled lengths during MCMC."
                    )
                if (
                    torch.isnan(pred_angles_radians_grad).any()
                    or torch.isinf(pred_angles_radians_grad).any()
                ):
                    raise ValueError(
                        "NaN or Inf gradients detected for radian angles during MCMC."
                    )

                pred_atom_types = (
                    pred_atom_types - alpha * pred_atom_types_grad
                )  # NOTE: Doing this to tokens will yield an unnormalized probability distribution, which later on we will convert to a probability distribution
                pred_pos = pred_pos - alpha * pred_pos_grad
                pred_frac_coords = pred_frac_coords - alpha * pred_frac_coords_grad
                pred_lengths_scaled = pred_lengths_scaled - alpha * pred_lengths_scaled_grad
                pred_angles_radians = pred_angles_radians - alpha * pred_angles_radians_grad

                # Prepare discrete distributions
                if self.discrete_absolute_clamp != 0.0:
                    pred_atom_types = torch.clamp(
                        pred_atom_types,
                        min=-self.discrete_absolute_clamp,
                        max=self.discrete_absolute_clamp,
                    )
                if self.sharpen_predicted_discrete_distribution != 0.0:
                    pred_atom_types = (
                        pred_atom_types / self.sharpen_predicted_discrete_distribution
                    )
                if return_raw_discrete_logits:
                    pred_atom_types_for_loss = pred_atom_types.reshape(-1, self.vocab_size)
                else:
                    pred_atom_types_for_loss = F.log_softmax(pred_atom_types, dim=-1).reshape(
                        -1, self.vocab_size
                    )

                # Collect predictions
                pred_energies_list.append(pred_energy)
                pred_modals_list.append(
                    {
                        "atom_types": pred_atom_types_for_loss,  # [B * S, V]
                        "pos": pred_pos,  # [B, S, 3]
                        "frac_coords": pred_frac_coords,  # [B, S, 3]
                        "lengths_scaled": pred_lengths_scaled,  # [B, 1, 3]
                        "angles_radians": pred_angles_radians,  # [B, 1, 3]
                    }
                )

        return pred_modals_list, pred_energies_list

    @typecheck
    def forward_with_loss_wrapper(
        self,
        atom_types: torch.Tensor,
        pos: torch.Tensor,
        frac_coords: torch.Tensor,
        lengths_scaled: torch.Tensor,
        angles_radians: torch.Tensor,
        dataset_idx: torch.Tensor,
        spacegroup: torch.Tensor,
        mask: torch.Tensor,
        token_is_periodic: torch.Tensor,
        target_tensors: Dict[str, torch.Tensor],
        stage: Literal["train", "sanity_check", "validate", "test", "predict"] = "train",
    ) -> Dict[str, torch.Tensor]:
        """MCMC-driven forward pass of EBT with loss calculation.

        Args:
            atom_types: Atom types tensor (B, N).
            pos: Atom positions tensor (B, N, 3).
            frac_coords: Fractional coordinates tensor (B, N, 3).
            lengths_scaled: Lattice lengths tensor (B, 1, 3).
            angles_radians: Lattice angles tensor (B, 1, 3).
            dataset_idx: Dataset index for each sample (B,).
            spacegroup: Spacegroup index for each sample (B,).
            mask: True if valid token, False if padding (B, N).
            token_is_periodic: Boolean mask indicating periodic tokens (B, N).
            target_tensors: Dictionary containing the following target tensors for loss calculation:
                atom_types: Target atom types tensor (B, N).
                pos: Target positions tensor (B, N, 3).
                frac_coords: Target fractional coordinates tensor (B, N, 3).
                lengths_scaled: Target lattice lengths tensor (B, 1, 3).
                angles_radians: Target lattice angles tensor (B, 1, 3).
            stage: Current stage of the model (train, sanity_check, validate, test, predict).

        Returns:
            Dictionary of loss values.
        """
        no_randomness = False if stage == "train" else True
        training = stage == "train"

        # Denoise (generate) modalities via energy minimization
        denoised_modals_list, pred_energies_list = self.forward(
            atom_types,
            pos,
            frac_coords,
            lengths_scaled,
            angles_radians,
            dataset_idx,
            spacegroup,
            mask,
            training=training,
            no_randomness=no_randomness,
            return_raw_discrete_logits=True,
        )

        # Calculate each loss for each modality
        loss_dict = {}

        reconstruction_loss_dict = {modal: 0 for modal in target_tensors}
        modal_loss_fn_dict = {
            "atom_types": self.nll_loss,
            "pos": self.mse_loss,
            "frac_coords": self.mse_loss,
            "lengths_scaled": self.mse_loss,
            "angles_radians": self.mse_loss,
        }

        reconstruction_loss_weight_dict = {
            "atom_types": self.atom_types_reconstruction_loss_weight,
            "pos": self.pos_reconstruction_loss_weight,
            "frac_coords": self.frac_coords_reconstruction_loss_weight,
            "lengths_scaled": self.lengths_scaled_reconstruction_loss_weight,
            "angles_radians": self.angles_radians_reconstruction_loss_weight,
        }
        total_mcmc_steps = len(pred_energies_list)

        should_rigid_align = {
            "pos": self.weighted_rigid_align_pos,
            "frac_coords": self.weighted_rigid_align_frac_coords,
        }

        for modal in target_tensors:
            for mcmc_step, (denoised_modals, pred_energies) in enumerate(
                zip(denoised_modals_list, pred_energies_list)
            ):
                pred_modal = denoised_modals[modal]
                target_modal = target_tensors[modal]
                modal_loss_fn = modal_loss_fn_dict[modal]
                reconstruction_loss_weight = reconstruction_loss_weight_dict[modal]

                loss_mask = mask.float()
                loss_token_is_periodic = token_is_periodic.float()

                target_shape = target_modal.shape

                # Calculate modality-specific losses
                if modal == "atom_types":
                    pred_modal = F.log_softmax(pred_modal, dim=-1).reshape(-1, self.vocab_size)
                    target_modal = target_modal.reshape(-1)
                elif modal in ("pos", "frac_coords"):
                    target_modal = (
                        weighted_rigid_align(pred_modal, target_modal, mask=mask)
                        if should_rigid_align[modal]
                        else target_modal
                    )
                    loss_mask = mask.unsqueeze(-1).float()
                    loss_token_is_periodic = token_is_periodic.unsqueeze(-1).float()
                elif modal in ("lengths_scaled", "angles_radians"):
                    loss_mask = torch.ones(
                        target_shape, dtype=torch.float, device=target_modal.device
                    )
                    loss_token_is_periodic = (
                        token_is_periodic.any(-1, keepdim=True).unsqueeze(-1).float()
                    )  # NOTE: A periodic sample is one with any periodic atoms

                modal_loss_value = (
                    modal_loss_fn(pred_modal, target_modal).reshape(target_shape) * loss_mask
                )

                if modal in (
                    "frac_coords",
                    "lengths_scaled",
                    "angles_radians",
                ):  # Periodic (crystal) losses
                    modal_loss_value = modal_loss_value * loss_token_is_periodic
                elif modal == "pos":  # Non-periodic (molecule) losses
                    modal_loss_value = modal_loss_value * (1 - loss_token_is_periodic)

                if self.truncate_mcmc:
                    if mcmc_step == (total_mcmc_steps - 1):
                        reconstruction_loss_dict[modal] = modal_loss_value
                        final_reconstruction_loss = reconstruction_loss_dict[modal].detach()
                        if modal == "atom_types":
                            ppl_loss = torch.exp(modal_loss_value).detach()
                else:
                    reconstruction_loss_dict[modal] += modal_loss_value
                    if mcmc_step == (total_mcmc_steps - 1):
                        final_reconstruction_loss = modal_loss_value.detach()
                        reconstruction_loss_dict[modal] = (
                            reconstruction_loss_dict[modal] / total_mcmc_steps
                        )  # Normalize so this is indifferent to the number of MCMC steps
                        if modal == "atom_types":
                            ppl_loss = torch.exp(modal_loss_value).detach()

                # Track relevant loss values
                if mcmc_step == 0:
                    initial_loss = modal_loss_value.detach()
                    initial_pred_energies = pred_energies.squeeze().mean().detach()
                if mcmc_step == (total_mcmc_steps - 1):
                    final_pred_energies = pred_energies.squeeze().mean().detach()
                    modal_loss = modal_loss_value.detach()

            # Collect losses
            initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
            total_loss = reconstruction_loss_weight * reconstruction_loss_dict[modal]

            loss_dict.update(
                {
                    f"{modal}_loss": total_loss.mean(),
                    f"{modal}_initial_loss": initial_loss.mean(),
                    f"{modal}_final_step_loss": final_reconstruction_loss.mean(),
                    f"{modal}_initial_final_pred_energies_gap": initial_final_pred_energies_gap,
                }
            )

            if modal_loss_fn is self.nll_loss:
                loss_dict[f"{modal}_ce_loss"] = modal_loss.mean()
                loss_dict[f"{modal}_ppl_loss"] = ppl_loss.mean()
            if modal_loss_fn is self.mse_loss:
                loss_dict[f"{modal}_mse_loss"] = modal_loss.mean()

        # Aggregate losses
        loss_dict["loss"] = sum(loss_dict[f"{modal}_loss"] for modal in denoised_modals)
        loss_dict["initial_loss"] = sum(
            loss_dict[f"{modal}_initial_loss"] for modal in denoised_modals
        )
        loss_dict["final_step_loss"] = sum(
            loss_dict[f"{modal}_final_step_loss"] for modal in denoised_modals
        )
        loss_dict["initial_final_pred_energies_gap"] = sum(
            loss_dict[f"{modal}_initial_final_pred_energies_gap"] for modal in denoised_modals
        )
        loss_dict["ce_loss"] = sum(
            loss_dict[f"{modal}_ce_loss"]
            for modal in denoised_modals
            if f"{modal}_ce_loss" in loss_dict
        )
        loss_dict["ppl_loss"] = sum(
            loss_dict[f"{modal}_ppl_loss"]
            for modal in denoised_modals
            if f"{modal}_ppl_loss" in loss_dict
        )
        loss_dict["mse_loss"] = sum(
            loss_dict[f"{modal}_mse_loss"]
            for modal in denoised_modals
            if f"{modal}_mse_loss" in loss_dict
        )

        return loss_dict
