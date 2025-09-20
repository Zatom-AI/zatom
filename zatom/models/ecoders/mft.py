"""Mean flow transformer (MFT).

Adapted from:
    - https://github.com/alexiglad/EBT
    - https://github.com/Gsunshine/meanflow
    - https://github.com/facebookresearch/flow_matching
    - https://github.com/facebookresearch/all-atom-diffusion-transformer
"""

import math
from typing import Dict, List, Literal, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.loss.generalized_loss import MixturePathGeneralizedKL
from flow_matching.path import AffineProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import (
    CondOTScheduler,
    PolynomialConvexScheduler,
)
from torch.nn.attention.flex_attention import create_block_mask

from zatom.models.ecoders.ebt import EBTBlock, LabelEmbedder, modulate
from zatom.models.encoders.custom_transformer import (
    LayerNorm,
    build_attention_mask,
)
from zatom.models.encoders.transformer import get_index_embedding
from zatom.utils import pylogger
from zatom.utils.training_utils import (
    BEST_DEVICE,
    initialize_module_weights,
    weighted_rigid_align,
)
from zatom.utils.typing_utils import Bool, Float, Int, typecheck

log = pylogger.RankedLogger(__name__)

#################################################################################
#                             Embedding Layers                                  #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations.

    Args:
        hidden_dim: The dimensionality of the hidden representations.
        frequency_embedding_dim: The dimensionality of the frequency embeddings.
    """

    def __init__(self, hidden_dim: int, frequency_embedding_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: The input tensor.
            dim: The dimensionality of the output embeddings.
            max_period: The maximum period for the sinusoidal functions.

        Returns:
            The sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass for the timestep embedder.

        Args:
            t: The input tensor.

        Returns:
            The output tensor.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core MFT Model                                #
#################################################################################


class FinalLayer(nn.Module):
    """The final layer of MFT.

    Args:
        hidden_dim: The dimensionality of the hidden representations.
        out_dim: The dimensionality of the output representations.
    """

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    @typecheck
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass for the final MFT layer.

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


class MFT(nn.Module):
    """Mean flow model with a Transformer encoder/decoder (i.e., an E-coder or `ecoder`).

    NOTE: The `_forward` pass of this model is conceptually similar to that of All-atom Diffusion Transformers (ADiTs)
    except that there is no self conditioning and the model learns flows for two specified times instead of one.

    Args:
        encoder: The encoder module.
        d_x: Input dimension.
        d_model: Model dimension.
        num_layers: Number of Transformer layers.
        nhead: Number of attention heads.
        mcmc_num_steps: Number of MCMC steps.
        mcmc_step_size: Markov chain Monte Carlo (MCMC) step size.
        mcmc_step_size_lr_multiplier: Learning rate multiplier for MCMC step size.
        randomize_mcmc_num_steps: Number of steps to randomize MCMC.
        randomize_mcmc_num_steps_min: Minimum number of steps to randomize MCMC.
        num_datasets: Number of datasets for context conditioning.
        num_spacegroups: Number of spacegroups for context conditioning.
        max_num_elements: Maximum number of elements in the dataset.
        context_length: Context length for the attention mechanism.
        rope_base: Base frequency for rotary positional encoding.
        mlp_ratio: Ratio of hidden to input dimension in MLP.
        proj_drop: Dropout probability for the projection layer.
        attn_drop: Dropout probability for the attention layer.
        class_dropout_prob: Probability of dropping class labels for context conditioning.
        langevin_dynamics_noise: Standard deviation of Langevin dynamics noise.
        weight_initialization_gain: Gain for discrete embedding weight initialization.
        randomize_mcmc_step_size_scale: Scale factor for randomizing MCMC step size.
        clamp_futures_grad_max_change: Maximum change for clamping future gradients.
        discrete_gaussian_random_noise_scaling: Scale factor for discrete Gaussian random noise.
        discrete_absolute_clamp: Maximum absolute value for discrete predictions.
        sharpen_predicted_discrete_distribution: Sharpening factor for predicted discrete distributions.
        atom_types_reconstruction_loss_weight: Weighting factor for the atom types reconstruction loss.
        pos_reconstruction_loss_weight: Weighting factor for the atom positions reconstruction loss.
        frac_coords_reconstruction_loss_weight: Weighting factor for the atom fractional coordinates reconstruction loss.
        lengths_scaled_reconstruction_loss_weight: Weighting factor for the atom lengths (scaled) reconstruction loss.
        angles_radians_reconstruction_loss_weight: Weighting factor for the atom angles (radians) reconstruction loss.
        qkv_bias: If True, add a learnable bias to query, key, value.
        qk_norm: If True, apply normalization to query and key.
        scale_attn_norm: If True, apply scaling to attention
            normalization.
        proj_bias: If True, add bias to output projection.
        flex_attn: Whether to use PyTorch's FlexAttention.
        fused_attn: Whether to use PyTorch's `scaled_dot_product_attention`.
        jvp_attn: Whether to use a Triton kernel for Jacobian-vector product (JVP) Flash Attention.
        truncate_mcmc: Whether to truncate MCMC sample gradient trajectories.
        clamp_futures_grad: Whether to clamp future gradients.
        no_mcmc_detach: Whether to (not) detach MCMC samples from the graph.
        no_langevin_during_eval: Whether to disable Langevin dynamics during evaluation.
        no_randomize_mcmc_step_size_scale_during_eval: Whether to disable randomizing MCMC step size scale during evaluation.
        mcmc_step_size_learnable: Whether to make the MCMC step size learnable.
        mcmc_step_index_learnable: Whether to embed the MCMC step index.
        modality_specific_mcmc_step_sizes_learnable: Whether to make separate MCMC step sizes learnable for each modality (if `mcmc_step_size_learnable` is also `True`).
        langevin_dynamics_noise_learnable: Whether to make the Langevin dynamics noise learnable.
        randomize_mcmc_num_steps_final_landscape: Whether to randomize MCMC steps for the final landscape.
        normalize_discrete_initial_condition: Whether to normalize discrete initial embeddings using softmax.
        weighted_rigid_align_pos: Whether to apply weighted rigid alignment between target and predicted atom positions for loss calculation.
        weighted_rigid_align_frac_coords: Whether to apply weighted rigid alignment between target and predicted atom fractional coordinates for loss calculation.
        use_pytorch_implementation: Whether to use PyTorch's Transformer implementation.
        add_mask_atom_type: Whether to add a mask token for atom types.
        discrete_weight_initialization_method: Initialization method for discrete embedding weights.
        discrete_denoising_initial_condition: Whether to use random or zero-based discrete denoising for initial conditions.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_x: int = 512,
        d_model: int = 768,
        num_layers: int = 12,
        nhead: int = 12,
        mcmc_num_steps: int = 2,
        mcmc_step_size: int = 500,
        mcmc_step_size_lr_multiplier: int = 1500,  # 3x `mcmc_step_size` as a rule of thumb
        randomize_mcmc_num_steps: int = 0,
        randomize_mcmc_num_steps_min: int = 0,
        num_datasets: int = 2,  # Context conditioning input
        num_spacegroups: int = 230,  # Context conditioning input
        max_num_elements: int = 100,
        context_length: int | None = 2048,
        rope_base: int | None = 10_000,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.1,
        attn_drop: float = 0.0,
        class_dropout_prob: float = 0.1,
        langevin_dynamics_noise: float = 0.0,
        weight_initialization_gain: float = 1.0,
        randomize_mcmc_step_size_scale: float = 1.0,
        clamp_futures_grad_max_change: float = 9.0,
        discrete_gaussian_random_noise_scaling: float = 1.0,
        discrete_absolute_clamp: float = 0.0,
        sharpen_predicted_discrete_distribution: float = 0.0,
        atom_types_reconstruction_loss_weight: float = 1.0,
        pos_reconstruction_loss_weight: float = 10.0,
        frac_coords_reconstruction_loss_weight: float = 10.0,
        lengths_scaled_reconstruction_loss_weight: float = 1.0,
        angles_radians_reconstruction_loss_weight: float = 10.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        scale_attn_norm: bool = False,
        proj_bias: bool = False,
        flex_attn: bool = False,
        fused_attn: bool = True,
        jvp_attn: bool = False,
        truncate_mcmc: bool = False,
        clamp_futures_grad: bool = False,
        no_mcmc_detach: bool = False,
        no_langevin_during_eval: bool = False,
        no_randomize_mcmc_step_size_scale_during_eval: bool = False,
        mcmc_step_size_learnable: bool = True,
        mcmc_step_index_learnable: bool = False,
        modality_specific_mcmc_step_sizes_learnable: bool = True,
        langevin_dynamics_noise_learnable: bool = False,
        randomize_mcmc_num_steps_final_landscape: bool = False,
        normalize_discrete_initial_condition: bool = True,
        weighted_rigid_align_pos: bool = True,
        weighted_rigid_align_frac_coords: bool = False,
        use_pytorch_implementation: bool = False,
        add_mask_atom_type: bool = True,
        discrete_weight_initialization_method: Literal["he", "xavier"] = "xavier",
        discrete_denoising_initial_condition: Literal["random", "zeros"] = "random",
        norm_layer: Type[nn.Module] = LayerNorm,
    ):
        super().__init__()

        assert (
            sum([flex_attn, fused_attn, jvp_attn]) <= 1
        ), "Only one of flex_attn, fused_attn, or jvp_attn can be True."

        self.encoder = encoder
        self.d_model = d_model
        self.nhead = nhead
        self.context_length = context_length
        self.class_dropout_prob = class_dropout_prob
        self.atom_types_reconstruction_loss_weight = atom_types_reconstruction_loss_weight
        self.pos_reconstruction_loss_weight = pos_reconstruction_loss_weight
        self.frac_coords_reconstruction_loss_weight = frac_coords_reconstruction_loss_weight
        self.lengths_scaled_reconstruction_loss_weight = lengths_scaled_reconstruction_loss_weight
        self.angles_radians_reconstruction_loss_weight = angles_radians_reconstruction_loss_weight
        self.flex_attn = flex_attn
        self.jvp_attn = jvp_attn
        self.weighted_rigid_align_pos = weighted_rigid_align_pos
        self.weighted_rigid_align_frac_coords = weighted_rigid_align_frac_coords
        self.use_pytorch_implementation = use_pytorch_implementation
        self.discrete_denoising_initial_condition = discrete_denoising_initial_condition

        self.modals = ["atom_types", "pos", "frac_coords", "lengths_scaled", "angles_radians"]

        self.vocab_size = max_num_elements + int(add_mask_atom_type)
        self.atom_type_embedder = nn.Embedding(self.vocab_size, d_model)

        self.x_embedder = nn.Linear(d_x, d_model, bias=True)
        self.t_embedder = TimestepEmbedder(d_model)
        self.dataset_embedder = LabelEmbedder(num_datasets, d_model, class_dropout_prob)
        self.spacegroup_embedder = LabelEmbedder(num_spacegroups, d_model, class_dropout_prob)

        self.blocks = nn.ModuleList(
            [
                EBTBlock(
                    d_model,
                    nhead,
                    context_length=context_length,
                    rope_base=rope_base,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    scale_attn_norm=scale_attn_norm,
                    proj_bias=proj_bias,
                    flex_attn=flex_attn,
                    fused_attn=fused_attn,
                    jvp_attn=jvp_attn,
                    use_pytorch_implementation=use_pytorch_implementation,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = FinalLayer(d_model, d_x)

        self.atom_types_head = nn.Linear(d_model, self.vocab_size, bias=True)
        self.pos_head = nn.Linear(d_model, 3, bias=False)
        self.frac_coords_head = nn.Linear(d_model, 3, bias=False)
        self.lengths_scaled_head = nn.Linear(d_model, 3, bias=True)
        self.angles_radians_head = nn.Linear(d_model, 3, bias=True)

        self.initialize_weights()

        # Initialize discrete embedding weights distinctly
        initialize_module_weights(
            self.atom_type_embedder,
            discrete_weight_initialization_method,
            weight_initialization_gain=weight_initialization_gain,
        )

        # Instantiate paths and losses
        self.modal_type_path_dict = {
            "discrete": MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=2.0)),
            "continuous": AffineProbPath(scheduler=CondOTScheduler()),
        }

        self.nll_loss = nn.NLLLoss(ignore_index=-100, reduction="none")
        self.kl_loss = MixturePathGeneralizedKL(
            path=self.modal_type_path_dict["discrete"], reduction="none"
        )
        self.mse_loss = nn.MSELoss(reduction="none")

        self.modal_loss_fn_dict = {
            "atom_types": self.kl_loss,
            "pos": self.mse_loss,
            "frac_coords": self.mse_loss,
            "lengths_scaled": self.mse_loss,
            "angles_radians": self.mse_loss,
        }
        self.reconstruction_loss_weight_dict = {
            "atom_types": self.atom_types_reconstruction_loss_weight,
            "pos": self.pos_reconstruction_loss_weight,
            "frac_coords": self.frac_coords_reconstruction_loss_weight,
            "lengths_scaled": self.lengths_scaled_reconstruction_loss_weight,
            "angles_radians": self.angles_radians_reconstruction_loss_weight,
        }
        self.should_rigid_align = {
            "pos": self.weighted_rigid_align_pos,
            "frac_coords": self.weighted_rigid_align_frac_coords,
        }

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

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in MFT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero-out heads
        nn.init.constant_(self.atom_types_head.weight, 0)
        nn.init.constant_(self.atom_types_head.bias, 0)
        nn.init.constant_(self.pos_head.weight, 0)
        # nn.init.constant_(self.pos_head.bias, 0) # No bias for pos head
        nn.init.constant_(self.frac_coords_head.weight, 0)
        # nn.init.constant_(self.frac_coords_head.bias, 0) # No bias for frac_coords head
        nn.init.constant_(self.lengths_scaled_head.weight, 0)
        nn.init.constant_(self.lengths_scaled_head.bias, 0)
        nn.init.constant_(self.angles_radians_head.weight, 0)
        nn.init.constant_(self.angles_radians_head.bias, 0)

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    @typecheck
    def _forward(
        self,
        atom_types: Float["b m v"],  # type: ignore
        pos: Float["b m 3"],  # type: ignore
        frac_coords: Float["b m 3"],  # type: ignore
        lengths_scaled: Float["b 1 3"],  # type: ignore
        angles_radians: Float["b 1 3"],  # type: ignore
        atom_types_r: Float[" b"],  # type: ignore
        atom_types_t: Float[" b"],  # type: ignore
        pos_r: Float[" b"],  # type: ignore
        pos_t: Float[" b"],  # type: ignore
        frac_coords_r: Float[" b"],  # type: ignore
        frac_coords_t: Float[" b"],  # type: ignore
        lengths_scaled_r: Float[" b"],  # type: ignore
        lengths_scaled_t: Float[" b"],  # type: ignore
        angles_radians_r: Float[" b"],  # type: ignore
        angles_radians_t: Float[" b"],  # type: ignore
        dataset_idx: Float[" b"],  # type: ignore
        spacegroup: Float[" b"],  # type: ignore
        mask: Float["b m"],  # type: ignore
        seq_idx: Float["b m"] | None = None,  # type: ignore
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of MFT.

        Args:
            atom_types: Atom type embeddings tensor (B, N, vocab_size).
            pos: Atom positions tensor (B, N, 3).
            frac_coords: Fractional coordinates tensor (B, N, 3).
            lengths_scaled: Scaled lengths tensor (B, 1, 3).
            angles_radians: Angles in radians tensor (B, 1, 3).
            atom_types_r: Time r for atom types (B,).
            atom_types_t: Time t for atom types (B,).
            pos_r: Time r for positions (B,).
            pos_t: Time t for positions (B,).
            frac_coords_r: Time r for fractional coordinates (B,).
            frac_coords_t: Time t for fractional coordinates (B,).
            lengths_scaled_r: Time r for lengths (B,).
            lengths_scaled_t: Time t for lengths (B,).
            angles_radians_r: Time r for angles (B,).
            angles_radians_t: Time t for angles (B,).
            dataset_idx: Dataset index for each sample (B,).
            spacegroup: Spacegroup index for each sample (B,).
            mask: True if valid token, False if padding (B, N).
            seq_idx: Indices of unique token sequences in the batch (optional unless using sequence packing).

        Returns:
            Output velocity fields for each modality as a dictionary.
        """
        batch_size, num_tokens, _ = atom_types.shape

        # Organize inputs
        modals_r = torch.cat(
            [
                r.unsqueeze(-1)
                for r in [atom_types_r, pos_r, frac_coords_r, lengths_scaled_r, angles_radians_r]
            ],
            dim=-1,
        )
        modals_t = torch.cat(
            [
                t.unsqueeze(-1)
                for t in [atom_types_t, pos_t, frac_coords_t, lengths_scaled_t, angles_radians_t]
            ],
            dim=-1,
        )

        # Type-cast to work around JVP requirements
        atom_types = self.atom_type_embedder(atom_types.argmax(dim=-1)).repeat_interleave(
            2, dim=-1
        )
        dataset_idx = dataset_idx.long()
        spacegroup = spacegroup.long()
        mask = mask.bool()

        if seq_idx is not None:
            seq_idx = seq_idx.long()

        # Positional embedding
        token_idx = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1

        if self.use_pytorch_implementation:
            pos_emb = get_index_embedding(token_idx, self.d_model, max_len=self.context_length)

        # Create the attention mask
        attn_mask = None
        if not self.use_pytorch_implementation:
            if seq_idx is None:
                seq_idx = torch.ones_like(token_idx)

            def padded_document_mask_mod(
                b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
            ) -> torch.Tensor:
                """Create a padded document mask for the attention mechanism.

                Args:
                    b: Batch index.
                    h: Head index (not used in this implementation).
                    q_idx: Index of the query token.
                    kv_idx: Index of the key-value tokens.

                Returns:
                    A boolean tensor value.
                """
                seq_ids = seq_idx
                non_padding_mask = (seq_ids[b, q_idx] != 0) & (seq_ids[b, kv_idx] != 0)
                document_mask = seq_ids[b, q_idx] == seq_ids[b, kv_idx]
                return non_padding_mask & document_mask

            attn_mask = (
                create_block_mask(
                    mask_mod=padded_document_mask_mod,
                    B=batch_size,
                    H=None,
                    Q_LEN=num_tokens,
                    KV_LEN=num_tokens,
                    device=self.device,
                )
                if self.flex_attn
                else build_attention_mask(
                    mask, seq_idx, dtype=torch.bool if self.jvp_attn else pos.dtype
                )
            )
            if self.jvp_attn:
                attn_mask = attn_mask.expand(-1, self.nhead, -1, -1)  # [B, H, N, N]

        # Input embeddings: [B, N, C]
        x_encoding = self.encoder(
            atom_types,
            pos,
            frac_coords,
            lengths_scaled,
            angles_radians,
            token_idx,
            mask,
            attn_mask=attn_mask,
        )
        x = self.x_embedder(x_encoding)

        # Conditioning embeddings
        h = self.t_embedder(modals_t - modals_r).mean(-2)  # [B, C]
        t = self.t_embedder(modals_t).mean(-2)  # [B, C]
        d = self.dataset_embedder(dataset_idx, self.training)  # [B, C]
        s = self.spacegroup_embedder(spacegroup, self.training)  # [B, C]
        c = h + t + d + s  # [B, C]

        # Transformer blocks
        for block in self.blocks:
            if self.use_pytorch_implementation:  # PyTorch-native Transformer
                x += pos_emb  # Absolute positional embedding
                x = block(x, c, ~mask)  # [B, N, C]
            else:  # Custom Transformer
                x = block(x, c, attn_mask, pos_ids=token_idx)  # [B, N, C]

        # Prediction layers
        x = self.final_layer(x, c)  # [B, N, 1]
        x = x * mask.unsqueeze(-1)  # Mask out padding tokens

        # Collect predictions
        global_mask = mask.any(-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
        pred_modals_dict = {
            "atom_types": self.atom_types_head(x) * mask.unsqueeze(-1),  # [B * S, V]
            "pos": self.pos_head(x) * mask.unsqueeze(-1),  # [B, S, 3]
            "frac_coords": self.frac_coords_head(x) * mask.unsqueeze(-1),  # [B, S, 3]
            "lengths_scaled": self.lengths_scaled_head(x.mean(-2, keepdim=True))
            * global_mask,  # [B, 1, 3]
            "angles_radians": self.angles_radians_head(x.mean(-2, keepdim=True))
            * global_mask,  # [B, 1, 3]
        }

        return pred_modals_dict

    @typecheck
    def forward(
        self,
        dataset_idx: Int[" b"],  # type: ignore
        spacegroup: Int[" b"],  # type: ignore
        mask: Bool["b m"],  # type: ignore
        modal_input_dict: (
            Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] | None
        ) = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """ODE-driven forward pass of MFT.

        Args:
            dataset_idx: Dataset index for each sample.
            spacegroup: Spacegroup index for each sample.
            mask: True if valid token, False if padding.
            modal_input_dict: If not None, a dictionary specifying input modalities to use and their input metadata.
                The keys should be a subset of `["atom_types", "pos", "frac_coords", "lengths_scaled", "angles_radians"]`,
                and the values should be tuples of (time r, time t, x_t, dx_t or None).

        Returns:
            A list of predicted modalities as a dictionary.
        """
        if modal_input_dict is None:
            raise ValueError("modal_input_dict currently must be provided for MFT forward pass.")

        # Predict velocity fields for each modality
        pred_modals_dict = self._forward(
            atom_types=modal_input_dict["atom_types"][-3],
            pos=modal_input_dict["pos"][-3],
            frac_coords=modal_input_dict["frac_coords"][-3],
            lengths_scaled=modal_input_dict["lengths_scaled"][-3],
            angles_radians=modal_input_dict["angles_radians"][-3],
            atom_types_r=modal_input_dict["atom_types"][-2],
            atom_types_t=modal_input_dict["atom_types"][-1],
            pos_r=modal_input_dict["pos"][-2],
            pos_t=modal_input_dict["pos"][-1],
            frac_coords_r=modal_input_dict["frac_coords"][-2],
            frac_coords_t=modal_input_dict["frac_coords"][-1],
            lengths_scaled_r=modal_input_dict["lengths_scaled"][-2],
            lengths_scaled_t=modal_input_dict["lengths_scaled"][-1],
            angles_radians_r=modal_input_dict["angles_radians"][-2],
            angles_radians_t=modal_input_dict["angles_radians"][-1],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=mask,
        )
        return pred_modals_dict

    @typecheck
    def forward_with_loss_wrapper(
        self,
        atom_types: Int["b m"],  # type: ignore
        pos: Float["b m 3"],  # type: ignore
        frac_coords: Float["b m 3"],  # type: ignore
        lengths_scaled: Float["b 1 3"],  # type: ignore
        angles_radians: Float["b 1 3"],  # type: ignore
        dataset_idx: Int[" b"],  # type: ignore
        spacegroup: Int[" b"],  # type: ignore
        mask: Bool["b m"],  # type: ignore
        token_is_periodic: Bool["b m"],  # type: ignore
        target_tensors: Dict[str, torch.Tensor],
        stage: Literal["train", "sanity_check", "validate", "test", "predict"] = "train",
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of MFT with loss calculation.

        Args:
            atom_types: Atom types tensor.
            pos: Atom positions tensor.
            frac_coords: Fractional coordinates tensor.
            lengths_scaled: Lattice lengths tensor.
            angles_radians: Lattice angles tensor.
            dataset_idx: Dataset index for each sample.
            spacegroup: Spacegroup index for each sample.
            mask: True if valid token, False if padding.
            token_is_periodic: Boolean mask indicating periodic tokens.
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
        device = atom_types.device
        batch_size, num_tokens = atom_types.shape

        # Assign a suitable probability path to each modality
        modal_type_dict = {
            modal: "continuous" if torch.is_floating_point(target_tensors[modal]) else "discrete"
            for modal in self.modals
        }

        # Sample time points and corresponding noised inputs for each modality
        modal_input_dict = {}
        for modal in self.modals:
            modal_type = modal_type_dict[modal]
            path = self.modal_type_path_dict[modal_type]

            x_0 = target_tensors[modal]  # Clean data
            x_1 = locals()[modal]  # Noised data

            # Sample two time points from a logit-normal distribution
            r = torch.sigmoid(torch.randn(batch_size, device=device) - 0.4)
            t = torch.sigmoid(torch.randn(batch_size, device=device) - 0.4)  # Mean -0.4, var 1

            # Set r = t for 75% of the batch
            perm_mask = torch.randperm(batch_size)[int(batch_size * 0.75)]
            r[perm_mask] = t[perm_mask]

            # Ensure r <= t
            perm_mask = r > t
            r[perm_mask], t[perm_mask] = t[perm_mask], r[perm_mask]

            # Sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            dx_t = getattr(path_sample, "dx_t", None)

            # Apply mask
            if x_t.shape == (batch_size, num_tokens):  # [B, N]
                x_t *= mask
            elif x_t.shape == (batch_size, num_tokens, 3):  # [B, N, 3]
                x_t *= mask.unsqueeze(-1)
            elif x_t.shape == (batch_size, 1, 3):  # [B, 1, 3]
                x_t *= mask.any(-1, keepdim=True).unsqueeze(-1)
            else:
                raise ValueError(f"Unexpected shape for x_t: {x_t.shape}")

            # Embed discrete inputs to continuous space to ensure JVP differentiability
            if modal == "atom_types":
                x_t = F.one_hot(x_t, self.vocab_size).float()
                dx_t = F.one_hot(
                    path_sample.x_0, self.vocab_size
                ).float()  # TODO: DERIVE THIS PROPERLY

            # Collect inputs
            modal_input_dict[modal] = (dx_t, x_t, r, t)

        with torch.amp.autocast(device_type=BEST_DEVICE.type, enabled=False):
            # Predict average velocity field for each modality
            pred_modals_dict, d_pred_modals_dict = torch.func.jvp(
                self._forward,
                (
                    modal_input_dict["atom_types"][-3],  # atom_types
                    modal_input_dict["pos"][-3],  # pos
                    modal_input_dict["frac_coords"][-3],  # frac_coords
                    modal_input_dict["lengths_scaled"][-3],  # lengths_scaled
                    modal_input_dict["angles_radians"][-3],  # angles_radians
                    modal_input_dict["atom_types"][-2],  # atom_types_r
                    modal_input_dict["atom_types"][-1],  # atom_types_t
                    modal_input_dict["pos"][-2],  # pos_r
                    modal_input_dict["pos"][-1],  # pos_t
                    modal_input_dict["frac_coords"][-2],  # frac_coords_r
                    modal_input_dict["frac_coords"][-1],  # frac_coords_t
                    modal_input_dict["lengths_scaled"][-2],  # lengths_scaled_r
                    modal_input_dict["lengths_scaled"][-1],  # lengths_scaled_t
                    modal_input_dict["angles_radians"][-2],  # angles_radians_r
                    modal_input_dict["angles_radians"][-1],  # angles_radians_t
                    dataset_idx.float(),
                    spacegroup.float(),
                    mask.float(),
                ),
                (
                    modal_input_dict["atom_types"][0],  # atom_types dx_t
                    modal_input_dict["pos"][0],  # pos dx_t
                    modal_input_dict["frac_coords"][0],  # frac_coords dx_t
                    modal_input_dict["lengths_scaled"][0],  # lengths_scaled dx_t
                    modal_input_dict["angles_radians"][0],  # angles_radians dx_t
                    torch.zeros_like(modal_input_dict["atom_types"][-2]),  # atom_types_r
                    torch.ones_like(modal_input_dict["atom_types"][-1]),  # atom_types_t
                    torch.zeros_like(modal_input_dict["pos"][-2]),  # pos_r
                    torch.ones_like(modal_input_dict["pos"][-1]),  # pos_t
                    torch.zeros_like(modal_input_dict["frac_coords"][-2]),  # frac_coords_r
                    torch.ones_like(modal_input_dict["frac_coords"][-1]),  # frac_coords_t
                    torch.zeros_like(modal_input_dict["lengths_scaled"][-2]),  # lengths_scaled_r
                    torch.ones_like(modal_input_dict["lengths_scaled"][-1]),  # lengths_scaled_t
                    torch.zeros_like(modal_input_dict["angles_radians"][-2]),  # angles_radians_r
                    torch.ones_like(modal_input_dict["angles_radians"][-1]),  # angles_radians_t
                    torch.zeros_like(dataset_idx, dtype=torch.float32),
                    torch.zeros_like(spacegroup, dtype=torch.float32),
                    torch.zeros_like(mask, dtype=torch.float32),
                ),
            )

            # Calculate each loss for each modality
            loss_dict = {}
            reconstruction_loss_dict = {modal: 0 for modal in self.modals}

            for modal in self.modals:
                pred_modal = pred_modals_dict[modal]
                d_pred_modal = d_pred_modals_dict[modal]
                modal_loss_fn = self.modal_loss_fn_dict[modal]
                reconstruction_loss_weight = self.reconstruction_loss_weight_dict[modal]

                target_modal, input_modal, r, t = modal_input_dict[modal]

                loss_mask = mask.float()
                loss_token_is_periodic = token_is_periodic.float()

                target_shape = target_modal.shape

                # Calculate modality-specific losses
                if modal == "atom_types":
                    target_modal = (
                        (target_modal - (t - r)[..., None, None] * d_pred_modal)
                        .detach()
                        .argmax(-1)
                    )
                    modal_loss_value = (
                        modal_loss_fn(pred_modal, target_modal, input_modal.argmax(-1), t - r)
                        * loss_mask
                    )
                elif modal in ("pos", "frac_coords"):
                    target_modal = (
                        weighted_rigid_align(pred_modal, target_modal, mask=mask)
                        if self.should_rigid_align[modal]
                        else target_modal
                    )
                    target_modal = (
                        target_modal - (t - r)[..., None, None] * d_pred_modal
                    ).detach()
                    loss_mask = mask.unsqueeze(-1).float()
                    loss_token_is_periodic = token_is_periodic.unsqueeze(-1).float()
                elif modal in ("lengths_scaled", "angles_radians"):
                    loss_mask = torch.ones(
                        target_shape, dtype=torch.float32, device=target_modal.device
                    )
                    loss_token_is_periodic = (
                        token_is_periodic.any(-1, keepdim=True).unsqueeze(-1).float()
                    )  # NOTE: A periodic sample is one with any periodic atoms

                if modal != "atom_types":
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

                reconstruction_loss_dict[modal] += modal_loss_value

                final_reconstruction_loss = modal_loss_value.detach()
                if modal == "atom_types":
                    nll_loss = (
                        self.nll_loss(
                            F.log_softmax(pred_modal, dim=-1).reshape(-1, self.vocab_size),
                            target_modal.reshape(-1),
                        ).reshape(target_shape[:-1])
                        * loss_mask
                    )
                    ppl_loss = torch.exp(nll_loss).detach()

                # Track relevant loss values
                initial_loss = modal_loss_value.detach()
                modal_loss = modal_loss_value.detach()

                # Collect losses
                total_loss = reconstruction_loss_weight * reconstruction_loss_dict[modal]

                loss_dict.update(
                    {
                        f"{modal}_loss": total_loss.mean(),
                        f"{modal}_initial_loss": initial_loss.mean(),
                        f"{modal}_final_step_loss": final_reconstruction_loss.mean(),
                        f"{modal}_initial_final_pred_energies_gap": torch.tensor(
                            torch.nan, device=device
                        ),
                    }
                )

                if modal_loss_fn is self.kl_loss:
                    loss_dict[f"{modal}_ce_loss"] = modal_loss.mean()
                    loss_dict[f"{modal}_ppl_loss"] = ppl_loss.mean()
                if modal_loss_fn is self.mse_loss:
                    loss_dict[f"{modal}_mse_loss"] = modal_loss.mean()

            # Aggregate losses
            loss_dict["loss"] = sum(loss_dict[f"{modal}_loss"] for modal in self.modals)
            loss_dict["initial_loss"] = sum(
                loss_dict[f"{modal}_initial_loss"] for modal in self.modals
            )
            loss_dict["final_step_loss"] = sum(
                loss_dict[f"{modal}_final_step_loss"] for modal in self.modals
            )
            loss_dict["initial_final_pred_energies_gap"] = sum(
                loss_dict[f"{modal}_initial_final_pred_energies_gap"] for modal in self.modals
            )
            loss_dict["ce_loss"] = sum(
                loss_dict[f"{modal}_ce_loss"]
                for modal in self.modals
                if f"{modal}_ce_loss" in loss_dict
            )
            loss_dict["ppl_loss"] = sum(
                loss_dict[f"{modal}_ppl_loss"]
                for modal in self.modals
                if f"{modal}_ppl_loss" in loss_dict
            )
            loss_dict["mse_loss"] = sum(
                loss_dict[f"{modal}_mse_loss"]
                for modal in self.modals
                if f"{modal}_mse_loss" in loss_dict
            )

        return loss_dict
