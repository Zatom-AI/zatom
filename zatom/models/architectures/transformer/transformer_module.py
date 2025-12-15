"""Adapted from https://github.com/carlosinator/tabasco."""

from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from zatom.models.architectures.transformer.common import ChargeSpinEmbedding, SwiGLU
from zatom.models.architectures.transformer.positional_encoder import (
    SinusoidEncoding,
    TimeFourierEncoding,
)
from zatom.models.architectures.transformer.transformer import (
    ModernTransformer,
    ModernTransformerDecoderBlock,
    Transformer,
)
from zatom.utils.pylogger import RankedLogger
from zatom.utils.typing_utils import Bool, Float, Int, typecheck

log = RankedLogger(__name__, rank_zero_only=True)


class TransformerModule(nn.Module):
    """Basic Transformer model for molecule and material generation.

    Args:
        spatial_dim: Dimension of spatial coordinates (e.g., 3 for 3D).
        atom_dim: Number of atom types.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        num_aux_layers: Number of auxiliary transformer layers.
        aux_hidden_dim: Dimension auxiliary transformer layers.
        aux_layer: Layer at which to extract representations for auxiliary tasks.
        hidden_dim: Dimension of the hidden layers.
        num_properties: Number of global properties to predict.
        dataset_embedder: The dataset embedder module.
        spacegroup_embedder: The spacegroup embedder module.
        activation: Activation function to use ("SiLU", "ReLU", "SwiGLU").
        implementation: Implementation type ("reimplemented", "reimplemented_modern").
        context_length: Maximum context length for positional encoding.
        rope_base: Base for rotary positional encoding.
        qk_layernorm: Whether to apply layer normalization to query and key in attention.
        use_sdpa: Whether to use scaled dot-product attention.
        jvp_attn: Whether to use JVP attention.
        cross_attention: Whether to use cross-attention layers.
        add_sinusoid_posenc: Whether to add sinusoidal positional encoding.
        concat_combine_input: Whether to concatenate and combine inputs.
        custom_weight_init: Custom weight initialization method (None, "xavier", "kaiming", "orthogonal", "uniform", "eye", "normal").
            NOTE: "uniform" does not work well.
    """

    @typecheck
    def __init__(
        self,
        spatial_dim: int,
        atom_dim: int,
        num_heads: int,
        num_layers: int,
        num_aux_layers: int,
        aux_hidden_dim: int,
        aux_layer: int,
        hidden_dim: int,
        num_properties: int,
        dataset_embedder: nn.Module,
        spacegroup_embedder: nn.Module,
        activation: Literal["SiLU", "ReLU", "SwiGLU"] = "SiLU",
        implementation: Literal["reimplemented", "reimplemented_modern"] = "reimplemented",
        context_length: Optional[int] = 2048,
        rope_base: Optional[int] = None,
        qk_layernorm: bool = False,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
        cross_attention: bool = False,
        add_sinusoid_posenc: bool = True,
        concat_combine_input: bool = False,
        custom_weight_init: Optional[
            Literal["none", "xavier", "kaiming", "orthogonal", "uniform", "eye", "normal"]
        ] = None,
    ):
        super().__init__()

        # Normalize custom_weight_init if it's the string "None"
        if isinstance(custom_weight_init, str) and custom_weight_init.lower() == "none":
            custom_weight_init = None

        self.input_dim = spatial_dim + atom_dim
        self.time_dim = 1
        self.cond_dim = 7
        self.comb_input_dim = self.input_dim + self.time_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_aux_layers = num_aux_layers
        self.implementation = implementation
        self.cross_attention = cross_attention
        self.add_sinusoid_posenc = add_sinusoid_posenc
        self.concat_combine_input = concat_combine_input
        self.custom_weight_init = custom_weight_init

        self.dataset_embedder = dataset_embedder
        self.spacegroup_embedder = spacegroup_embedder

        self.atom_type_embed = nn.Embedding(atom_dim, hidden_dim)
        self.pos_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.frac_coords_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.lengths_scaled_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.angles_radians_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)

        if self.add_sinusoid_posenc:
            self.positional_encoding = SinusoidEncoding(
                posenc_dim=hidden_dim, max_len=context_length
            )

        if self.concat_combine_input:
            self.combine_input = nn.Linear(self.cond_dim * hidden_dim, hidden_dim)

        self.time_encoding = TimeFourierEncoding(posenc_dim=hidden_dim, max_len=200)

        if activation == "SiLU":
            activation = nn.SiLU(inplace=False)
        elif activation == "ReLU":
            activation = nn.ReLU(inplace=False)
        elif activation == "SwiGLU":
            activation = SwiGLU()
        else:
            raise ValueError(f"Invalid activation: {activation}")

        transformer_class = (
            partial(
                ModernTransformer,
                context_length=context_length,
                rope_base=rope_base,
                qk_layernorm=qk_layernorm,
                use_sdpa=use_sdpa,
                jvp_attn=jvp_attn,
            )
            if implementation == "reimplemented_modern"
            else Transformer
        )
        if self.implementation in ("reimplemented", "reimplemented_modern"):
            self.transformer_norm = (
                nn.RMSNorm(hidden_dim)
                if self.implementation == "reimplemented_modern"
                else nn.Identity()
            )
            self.transformer = transformer_class(
                dim=hidden_dim,
                depth=num_layers,
                num_heads=num_heads,
                repr_layer=aux_layer,
            )
        else:
            raise ValueError(f"Invalid implementation: {self.implementation}")

        self.out_atom_types = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, atom_dim),
        )
        self.out_pos = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )
        self.out_frac_coords = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )
        self.out_lengths_scaled = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )
        self.out_angles_radians = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )

        # Add cross attention layers
        if self.cross_attention:
            decoder_layer = partial(
                nn.TransformerDecoderLayer,
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True,
            )
            if self.implementation == "reimplemented_modern":
                decoder_layer = partial(
                    ModernTransformerDecoderBlock,
                    dim=hidden_dim,
                    n_heads=num_heads,
                    context_length=context_length,
                    rope_base=rope_base,
                    qk_layernorm=qk_layernorm,
                    use_sdpa=use_sdpa,
                    jvp_attn=jvp_attn,
                )

            self.atom_types_cross_attention = decoder_layer()
            self.pos_cross_attention = decoder_layer()
            self.frac_coords_cross_attention = decoder_layer()
            self.lengths_scaled_cross_attention = decoder_layer()
            self.angles_radians_cross_attention = decoder_layer()

        # Add auxiliary task heads
        self.global_property_proj = (
            nn.Linear(hidden_dim, aux_hidden_dim, bias=False)
            if hidden_dim != aux_hidden_dim
            else nn.Identity()
        )
        self.global_energy_proj = (
            nn.Linear(hidden_dim, aux_hidden_dim, bias=False)
            if hidden_dim != aux_hidden_dim
            else nn.Identity()
        )
        self.atomic_forces_proj = (
            nn.Linear(hidden_dim, aux_hidden_dim, bias=False)
            if hidden_dim != aux_hidden_dim
            else nn.Identity()
        )

        if self.num_aux_layers > 0:
            if self.cross_attention:
                self.global_property_cross_attention = decoder_layer()
                self.global_energy_cross_attention = decoder_layer()
                self.atomic_forces_cross_attention = decoder_layer()

            self.charge_proj = ChargeSpinEmbedding(
                embedding_type="pos_emb",
                embedding_target="charge",
                embedding_size=hidden_dim,
                grad=True,
                scale=1.0,
            )
            self.spin_proj = ChargeSpinEmbedding(
                embedding_type="pos_emb",
                embedding_target="spin",
                embedding_size=hidden_dim,
                grad=True,
                scale=1.0,
            )

            self.global_property_transformer = transformer_class(
                dim=aux_hidden_dim,
                num_heads=num_heads,
                depth=num_aux_layers,
            )
            self.global_energy_transformer = transformer_class(
                dim=aux_hidden_dim,
                num_heads=num_heads,
                depth=num_aux_layers,
            )
            self.atomic_forces_transformer = transformer_class(
                dim=aux_hidden_dim,
                num_heads=num_heads,
                depth=num_aux_layers,
            )

        self.global_property_head = nn.Linear(aux_hidden_dim, num_properties, bias=True)
        self.global_energy_head = nn.Linear(aux_hidden_dim, 1, bias=True)
        self.atomic_forces_head = nn.Linear(aux_hidden_dim, 3, bias=False)

        self.auxiliary_tasks = ["global_property", "global_energy", "atomic_forces"]

        # Initialize weights
        if self.custom_weight_init is not None:
            log.info(f"Initializing weights via {self.custom_weight_init} method.")
            self.apply(self._custom_weight_init)

    @typecheck
    def _custom_weight_init(self, module: nn.Module):
        """Initialize the weights of the module with a custom method.

        Args:
            module: The module to initialize.
        """
        for name, param in module.named_parameters():
            if "weight" in name and param.data.dim() >= 2:
                if self.custom_weight_init == "xavier":
                    nn.init.xavier_uniform_(param)
                elif self.custom_weight_init == "kaiming":
                    nn.init.kaiming_uniform_(param)
                elif self.custom_weight_init == "orthogonal":
                    nn.init.orthogonal_(param)
                elif self.custom_weight_init == "uniform":
                    nn.init.uniform_(param)
                elif self.custom_weight_init == "eye":
                    nn.init.eye_(param)
                elif self.custom_weight_init == "normal":
                    nn.init.normal_(param)
                else:
                    raise ValueError(f"Invalid custom weight init: {self.custom_weight_init}")

    @typecheck
    def forward(
        self,
        x: (
            Tuple[
                Int["b m v"],  # type: ignore - atom_types
                Float["b m 3"],  # type: ignore - pos
                Float["b m 3"],  # type: ignore - frac_coords
                Float["b 1 3"],  # type: ignore - lengths_scaled
                Float["b 1 3"],  # type: ignore - angles_radians
            ]
            | List[torch.Tensor]
        ),
        t: (
            Tuple[
                Float[" b"],  # type: ignore - atom_types_t
                Float[" b"],  # type: ignore - pos_t
                Float[" b"],  # type: ignore - frac_coords_t
                Float[" b"],  # type: ignore - lengths_scaled_t
                Float[" b"],  # type: ignore - angles_radians_t
            ]
            | List[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]
        ),
        feats: Dict[str, Tensor],
        padding_mask: Bool["b m"],  # type: ignore
        **kwargs: Any,
    ) -> Tuple[
        Tuple[
            Float["b m v"],  # type: ignore - atom_types
            Float["b m 3"],  # type: ignore - pos
            Float["b m 3"],  # type: ignore - frac_coords
            Float["b 1 3"],  # type: ignore - lengths_scaled
            Float["b 1 3"],  # type: ignore - angles_radians
        ],
        Tuple[
            Float["b 1 p"],  # type: ignore - global_property
            Float["b 1 1"],  # type: ignore - global_energy
            Float["b m 3"],  # type: ignore - atomic_forces
        ],
    ]:
        """Forward pass of the module.

        Args:
            x: Tuple or list of input tensors for each modality:
                atom_types: Atom types tensor (B, M, V), where V is the number of atom types.
                pos: Atom positions tensor (B, M, 3).
                frac_coords: Fractional coordinates tensor (B, M, 3).
                lengths_scaled: Scaled lengths tensor (B, 1, 3).
                angles_radians: Angles in radians tensor (B, 1, 3).
            t: Tuple or list of time tensors for each modality:
                atom_types_t: Time t for atom types (B,).
                pos_t: Time t for positions (B,).
                frac_coords_t: Time t for fractional coordinates (B,).
                lengths_scaled_t: Time t for lengths (B,).
                angles_radians_t: Time t for angles (B,).
            feats: Features for conditioning including:
                dataset_idx: Dataset index for each sample.
                spacegroup: Spacegroup index for each sample.
                token_is_periodic: Whether each token corresponds to a periodic sample (B, M).
            padding_mask: True if padding token, False otherwise (B, M).
            kwargs: Any additional keyword arguments.

        Returns:
            A tuple containing output velocity fields for each modality as an inner tuple
            and auxiliary task outputs as another inner tuple.
        """
        atom_types, pos, frac_coords, lengths_scaled, angles_radians = x
        atom_types_t, pos_t, frac_coords_t, lengths_scaled_t, angles_radians_t = t

        device = padding_mask.device
        batch_size, seq_len = padding_mask.shape

        dataset_idx = feats["dataset_idx"]
        spacegroup = feats["spacegroup"]

        token_is_periodic = feats["token_is_periodic"].unsqueeze(-1)
        sample_is_periodic = token_is_periodic.any(-2, keepdim=True)

        # Ensure atom positions are masked out for periodic samples and the
        # remaining continuous modalities are masked out for non-periodic samples
        pos = pos * ~token_is_periodic
        frac_coords = frac_coords * token_is_periodic
        lengths_scaled = lengths_scaled * sample_is_periodic
        angles_radians = angles_radians * sample_is_periodic

        real_mask = 1 - padding_mask.int()
        global_mask = real_mask.any(-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)

        embed_atom_types = self.atom_type_embed(atom_types.argmax(dim=-1))
        embed_pos = self.pos_embed(pos)
        embed_frac_coords = self.frac_coords_embed(frac_coords)
        embed_lengths_scaled = self.lengths_scaled_embed(lengths_scaled)
        embed_angles_radians = self.angles_radians_embed(angles_radians)

        if self.add_sinusoid_posenc:
            embed_posenc = self.positional_encoding(batch_size=batch_size, seq_len=seq_len)
        else:
            embed_posenc = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)

        modals_t = torch.cat(
            [
                t.unsqueeze(-1)
                for t in [atom_types_t, pos_t, frac_coords_t, lengths_scaled_t, angles_radians_t]
            ],
            dim=-1,
        )
        embed_time = (
            self.time_encoding(modals_t.reshape(-1))
            .reshape(batch_size, modals_t.shape[1], -1)
            .mean(-2)
        )  # (B, C), average over modalities
        embed_dataset = self.dataset_embedder(dataset_idx, self.training)  # (B, C)
        embed_spacegroup = self.spacegroup_embedder(spacegroup, self.training)  # (B, C)
        embed_conditions = (embed_time + embed_dataset + embed_spacegroup).unsqueeze(
            -2
        )  # (B, 1, C)

        assert all(
            embed.shape == (batch_size, seq_len, self.hidden_dim)
            for embed in [
                embed_atom_types,
                embed_pos,
                embed_frac_coords,
                embed_posenc,
            ]
        ), f"Embedding shapes are inconsistent. Shapes: {[embed.shape for embed in [embed_atom_types, embed_pos, embed_frac_coords, embed_posenc]]}"

        if self.concat_combine_input:
            embed_lengths_scaled = embed_lengths_scaled.repeat(1, seq_len, 1)
            embed_angles_radians = embed_angles_radians.repeat(1, seq_len, 1)
            embed_conditions = embed_conditions.repeat(1, seq_len, 1)
            h_in = torch.cat(
                [
                    embed_atom_types,
                    embed_pos,
                    embed_frac_coords,
                    embed_lengths_scaled,
                    embed_angles_radians,
                    embed_posenc,
                    embed_conditions,
                ],
                dim=-1,
            )
            assert h_in.shape == (
                batch_size,
                seq_len,
                self.cond_dim * self.hidden_dim,
            ), f"h_in.shape: {h_in.shape}"
            h_in = self.combine_input(h_in)
            assert h_in.shape == (
                batch_size,
                seq_len,
                self.hidden_dim,
            ), f"h_in.shape: {h_in.shape}"
        else:
            h_in = (
                embed_atom_types
                + embed_pos
                + embed_frac_coords
                + embed_lengths_scaled
                + embed_angles_radians
                + embed_posenc
                + embed_conditions
            )
        h_in = h_in * real_mask.unsqueeze(-1)

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        attention_kwargs = {"pos_ids": pos_ids}

        if self.implementation == "reimplemented":
            attention_kwargs.pop("pos_ids")

        if self.implementation in ("reimplemented", "reimplemented_modern"):
            h_in = self.transformer_norm(h_in)
            h_out, h_aux = self.transformer(h_in, padding_mask=padding_mask, **attention_kwargs)
        else:
            raise ValueError(f"Invalid implementation: {self.implementation}")

        h_out = h_out * real_mask.unsqueeze(-1)

        if self.cross_attention:
            h_atom = self.atom_types_cross_attention(
                h_out,
                h_in,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_pos = self.pos_cross_attention(
                h_out,
                h_in,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_frac_coords = self.frac_coords_cross_attention(
                h_out,
                h_in,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_lengths_scaled = self.lengths_scaled_cross_attention(
                h_out,
                h_in,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_angles_radians = self.angles_radians_cross_attention(
                h_out,
                h_in,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )

            out_atom_types = self.out_atom_types(h_atom)
            out_pos = self.out_pos(h_pos)
            frac_coords = self.out_frac_coords(h_frac_coords)
            lengths_scaled = self.out_lengths_scaled(h_lengths_scaled.mean(-2, keepdim=True))
            angles_radians = self.out_angles_radians(h_angles_radians.mean(-2, keepdim=True))

        else:
            out_atom_types = self.out_atom_types(h_out)
            out_pos = self.out_pos(h_out)
            frac_coords = self.out_frac_coords(h_out)
            lengths_scaled = self.out_lengths_scaled(h_out.mean(-2, keepdim=True))
            angles_radians = self.out_angles_radians(h_out.mean(-2, keepdim=True))

        h_global_property = h_aux
        h_global_energy = h_aux
        h_atomic_forces = h_aux

        if self.num_aux_layers > 0:
            if self.cross_attention:
                h_global_property = self.global_property_cross_attention(
                    h_aux,
                    h_in,
                    tgt_key_padding_mask=padding_mask,
                    memory_key_padding_mask=padding_mask,
                    **attention_kwargs,
                )
                h_global_energy = self.global_energy_cross_attention(
                    h_aux,
                    h_in,
                    tgt_key_padding_mask=padding_mask,
                    memory_key_padding_mask=padding_mask,
                    **attention_kwargs,
                )
                h_atomic_forces = self.atomic_forces_cross_attention(
                    h_aux,
                    h_in,
                    tgt_key_padding_mask=padding_mask,
                    memory_key_padding_mask=padding_mask,
                    **attention_kwargs,
                )

            ce = self.charge_proj(feats["charge"])
            se = self.spin_proj(feats["spin"])

            cse = ce.unsqueeze(-2) + se.unsqueeze(-2)  # (B, 1, C)

            h_global_energy = h_global_energy + cse
            h_atomic_forces = h_atomic_forces + cse

            h_global_property = self.global_property_transformer(
                self.global_property_proj(h_global_property),
                padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_global_energy = self.global_energy_transformer(
                self.global_energy_proj(h_global_energy),
                padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_atomic_forces = self.atomic_forces_transformer(
                self.atomic_forces_proj(h_atomic_forces),
                padding_mask=padding_mask,
                **attention_kwargs,
            )
        else:
            h_global_property = self.global_property_proj(h_global_property)
            h_global_energy = self.global_energy_proj(h_global_energy)
            h_atomic_forces = self.atomic_forces_proj(h_atomic_forces)

        global_property = (
            self.global_property_head(h_global_property.mean(-2, keepdim=True))
        ) * global_mask
        global_energy = (
            self.global_energy_head(h_global_energy.mean(-2, keepdim=True)) * global_mask
        )
        atomic_forces = self.atomic_forces_head(h_atomic_forces) * real_mask.unsqueeze(-1)

        pred_modals = (
            out_atom_types * real_mask.unsqueeze(-1),  # (B, M, V=self.vocab_size)
            out_pos * real_mask.unsqueeze(-1) * ~token_is_periodic,  # (B, M, 3)
            frac_coords * real_mask.unsqueeze(-1) * token_is_periodic,  # (B, M, 3)
            lengths_scaled * global_mask * sample_is_periodic,  # (B, 1, 3)
            angles_radians * global_mask * sample_is_periodic,  # (B, 1, 3)
        )
        pred_aux_outputs = (
            global_property,  # (B, 1, P)
            global_energy,  # (B, 1, 1)
            atomic_forces,  # (B, M, 3)
        )

        return pred_modals, pred_aux_outputs

    @typecheck
    def forward_with_cfg(
        self,
        x: (
            Tuple[
                Int["b m v"],  # type: ignore - atom_types
                Float["b m 3"],  # type: ignore - pos
                Float["b m 3"],  # type: ignore - frac_coords
                Float["b 1 3"],  # type: ignore - lengths_scaled
                Float["b 1 3"],  # type: ignore - angles_radians
            ]
            | List[torch.Tensor]
        ),
        t: (
            Tuple[
                Float[" b"],  # type: ignore - atom_types_t
                Float[" b"],  # type: ignore - pos_t
                Float[" b"],  # type: ignore - frac_coords_t
                Float[" b"],  # type: ignore - lengths_scaled_t
                Float[" b"],  # type: ignore - angles_radians_t
            ]
            | List[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]
        ),
        feats: Dict[str, Tensor],
        padding_mask: Bool["b m"],  # type: ignore
        cfg_scale: float,
        **kwargs: Any,
    ) -> Tuple[
        Tuple[
            Float["b m v"],  # type: ignore - atom_types
            Float["b m 3"],  # type: ignore - pos
            Float["b m 3"],  # type: ignore - frac_coords
            Float["b 1 3"],  # type: ignore - lengths_scaled
            Float["b 1 3"],  # type: ignore - angles_radians
        ],
        Tuple[
            Float["b 1 p"],  # type: ignore - global_property
            Float["b 1 1"],  # type: ignore - global_energy
            Float["b m 3"],  # type: ignore - atomic_forces
        ],
    ]:
        """Forward pass of TransformerModule, but also batches the unconditional forward pass for
        classifier-free guidance.

        NOTE: Assumes batch x's and class labels are ordered such that the first half are the conditional
        samples and the second half are the unconditional samples.

        Args:
            x: Tuple or list of input tensors for each modality:
                atom_types: Atom types tensor (B, M, V), where V is the number of atom types.
                pos: Atom positions tensor (B, M, 3).
                frac_coords: Fractional coordinates tensor (B, M, 3).
                lengths_scaled: Scaled lengths tensor (B, 1, 3).
                angles_radians: Angles in radians tensor (B, 1, 3).
            t: Tuple or list of time tensors for each modality:
                atom_types_t: Time t for atom types (B,).
                pos_t: Time t for positions (B,).
                frac_coords_t: Time t for fractional coordinates (B,).
                lengths_scaled_t: Time t for lengths (B,).
                angles_radians_t: Time t for angles (B,).
            feats: Features for conditioning including:
                dataset_idx: Dataset index for each sample.
                spacegroup: Spacegroup index for each sample.
                token_is_periodic: Whether each token corresponds to a periodic sample (B, M).
            padding_mask: True if padding token, False otherwise (B, M).
            cfg_scale: Classifier-free guidance scale.
            kwargs: Any additional keyword arguments.

        Returns:
            A tuple containing output velocity fields for each modality as an inner tuple
            and auxiliary task outputs as another inner tuple.
        """
        half_x = tuple(x_[: len(x_) // 2] for x_ in x)
        combined_x = tuple(torch.cat([half_x_, half_x_], dim=0) for half_x_ in half_x)
        model_out, model_aux_out = self.forward(
            combined_x,
            t,
            feats,
            padding_mask,
            **kwargs,
        )

        eps = []
        for modal in model_out:
            cond_eps, uncond_eps = torch.split(modal, len(modal) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps.append(torch.cat([half_eps, half_eps], dim=0))

        eps_aux = []
        for aux in model_aux_out:
            cond_aux, uncond_aux = torch.split(aux, len(aux) // 2, dim=0)
            half_aux = uncond_aux + cfg_scale * (cond_aux - uncond_aux)
            eps_aux.append(torch.cat([half_aux, half_aux], dim=0))

        return tuple(eps), tuple(eps_aux)
