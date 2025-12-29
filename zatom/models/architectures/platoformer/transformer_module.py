"""Adapted from https://github.com/carlosinator/tabasco."""

from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from platonic_transformers.models.platoformer.io import lift
from platonic_transformers.models.platoformer.linear import PlatonicLinear
from torch import Tensor

from zatom.models.architectures.platoformer import get_platonic_group
from zatom.models.architectures.platoformer.io import (
    ProjRegularToScalar,
    ProjRegularToVector,
)
from zatom.models.architectures.platoformer.norm import NormPlatonic
from zatom.models.architectures.platoformer.positional_encoder import (
    PlatonicLinearAPE,
    PlatonicSinusoidAPE,
)
from zatom.models.architectures.platoformer.transformer import (
    ModernTransformerDecoderBlockPlatonic,
    ModernTransformerPlatonic,
)
from zatom.models.architectures.transformer.common import ChargeSpinEmbedding
from zatom.models.architectures.transformer.positional_encoder import (
    SinusoidEncoding,
    TimeFourierEncoding,
)
from zatom.utils.pylogger import RankedLogger
from zatom.utils.typing_utils import Bool, Float, Int, typecheck

log = RankedLogger(__name__, rank_zero_only=True)


class TransformerModulePlatonic(nn.Module):
    """Platonic Transformer model for molecule and material generation.

    Equivariant under actions of a Platonic solid group G, approximating full O(3)-equivariance.
    Internal features are regular G-representations: there are channels associated to each group
    element, permuting according to the group action. These features are of shape (B, N, G*C), but
    should be thought of as shape (B, N, G, C).
    The model input are Euclidean coordinates (vectors) and a bunch of invariant tensors (scalars).
    Equivariant vector outputs are coordinates and auxiliary force predictions and, again, a lot of
    scalar quantities.

    The model sees Euclidean coordinates in two ways:
     1) An initial Platonic absolute position embedding (APE) prior to the transformer trunk. These
        position embeddings break translational symmetry, but are designed to respect Platonic group
        equivariance. One can choose different Platonic APE implementations or entirely disable APE.
     2) Platonic transformer style rotary position embeddings (RoPE) in self- and cross-attention
        layers. Platonic RoPE respects both translations and Platonic symmetries.

    Args:
        solid_name:           String identifying the Platonic solid group.
        spatial_dim:          Dimension of spatial coordinates (e.g., 3 for 3D).
        c_model:              Number of channels per group element.
        c_aux:                Number of channels per group element in the auxiliary task heads.
        num_atom_types:       Number of atom types.
        num_layers:           Number of Platonic Transformer blocks in the trunk transformer
        num_aux_layers:       Number of Platonic Transformer blocks in the auxiliary transformer.
        aux_layer:            Layer at which to extract representations for auxiliary tasks.
        num_properties:       Number of global properties to predict.
        dataset_embedder:     The dataset embedder module.
        spacegroup_embedder:  The spacegroup embedder module.
        transformer_factory:  Partial __init__ for ModernTransformerPlatonic with (c_model, depth,
                              repr_layer) filled in at runtime. Used to instantiate both the trunk
                              and auxiliary transformers.
        cross_attn_factory:   Partial __init__ for ModernTransformerDecoderBlockPlatonic with all
                              necessary arguments filled in. Used to instantiate the five prediction
                              and three auxiliary Tabasco style cross attention operations.
        coords_embed:         Absolute Euclidean coordinates embedding via PlatonicSinusoidAPE,
                              PlatonicLinearAPE or None:
                              - PlatonicSinusoidAPE/PlatonicLinearAPE break translation equivariance
                                but preserve Platonic group equivariance.
                              - None implies coordinate invariance. Due to Platonic RoPE attention,
                                the full model will be translation and Platonic group equivariant.
                                This might not work well together with cross-attention since there
                                are no absolute position embeddings to attend to!
        context_length:       Maximum context length for positional encoding.
        use_sequence_sin_ape: Whether to add sinusoidal positional encoding along *sequence*
                              dimension (in SMILES ordering).
        concat_combine_input: Whether to concatenate and combine inputs.
        normalize_per_g:      If False, Platonic normalization layers operate over the last
                              (channel) axis only. If True, acting on the group axis as well.
        custom_weight_init:   Custom weight initialization method.
                              NOTE: not yet implemented.
    """

    @typecheck
    def __init__(
        self,
        solid_name: str,  # in PLATONIC_GROUPS_3D
        spatial_dim: int,
        c_model: int,
        c_aux: int,
        num_atom_types: int,
        num_layers: int,
        num_aux_layers: int,
        aux_layer: int,
        num_properties: int,
        dataset_embedder: nn.Module,
        spacegroup_embedder: nn.Module,
        transformer_factory: Callable[..., ModernTransformerPlatonic],
        cross_attn_factory: Optional[Callable[[], ModernTransformerDecoderBlockPlatonic]],
        coords_embed: Optional[Union[PlatonicSinusoidAPE, PlatonicLinearAPE]] = None,  # type: ignore
        context_length: int = 2048,
        use_sequence_sin_ape: bool = True,
        concat_combine_input: bool = False,
        normalize_per_g: bool = True,
        custom_weight_init: Optional[
            Literal["none", "xavier", "kaiming", "orthogonal", "uniform", "eye", "normal"]
        ] = None,
        **kwargs,
    ):
        super().__init__()

        # Normalize custom_weight_init if it's the string "None"
        if isinstance(custom_weight_init, str) and custom_weight_init.lower() == "none":
            custom_weight_init = None

        self.solid_name = solid_name
        self.group = get_platonic_group(solid_name)
        self.G = G = self.group.G
        self.G_lift_scalars = partial(lift, vectors=None, group=self.group)

        self.spatial_dim = spatial_dim
        self.c_model = c_model
        self.c_aux = c_aux
        self.num_atom_types = num_atom_types
        self.num_layers = num_layers
        self.num_aux_layers = num_aux_layers
        self.aux_layer = aux_layer
        self.num_properties = num_properties
        self.context_length = context_length

        self.jvp_attn = kwargs.get("attn_backend", "MANUAL") == "JVP_ATTN"
        self.use_cross_attn = cross_attn_factory is not None
        self.use_sequence_sin_ape = use_sequence_sin_ape
        self.concat_combine_input = concat_combine_input
        self.cond_dim = 6 if coords_embed is None else 7

        # __________________________________________________________________________________________
        # __EMBEDDINGS______________________________________________________________________________

        # Global E(3)-invariant embeddings
        self.dataset_embedder = dataset_embedder
        self.spacegroup_embedder = spacegroup_embedder
        self.time_encoding = TimeFourierEncoding(posenc_dim=c_model, max_len=200)

        # Node-wise E(3)-invariant embeddings
        self.atom_type_embed = nn.Embedding(num_atom_types, c_model)
        self.frac_coords_embed = nn.Linear(
            spatial_dim, c_model, bias=False
        )  # fractional coordinates are *scalar* coefficients relative to equivariant lattice basis
        self.lengths_scaled_embed = nn.Linear(spatial_dim, c_model, bias=False)
        self.angles_radians_embed = nn.Linear(spatial_dim, c_model, bias=False)
        if use_sequence_sin_ape:
            self.sequence_sin_ape = SinusoidEncoding(
                posenc_dim=c_model, max_len=context_length
            )  # NOTE: along sequence axis in SMILES ordering, not Euclidean positions

        # Euclidean coordinates embedding. Two options:
        # - Platonic group equivariant APE  (breaking translation equivariance)
        # - None                            (fully E(3)-invariant, coords invisible to the network)
        self.coords_embed = coords_embed
        assert (
            isinstance(coords_embed, (PlatonicSinusoidAPE, PlatonicLinearAPE))
            or coords_embed is None
        ), "coords_embed needs to be an instance of PlatonicSinusoidAPE or PlatonicLinearAPE or None."

        # Linear layers for combining embeddings when using concatenation
        if concat_combine_input:
            if coords_embed is None:
                # Conventional linear layer, applied to scalars *before* lifting to G
                self.combine_input = nn.Linear(
                    in_features=c_model * self.cond_dim,
                    out_features=c_model,
                )
            else:
                # Platonic linear layer, applied *after* lifting to G
                self.combine_input = PlatonicLinear(
                    in_features=G * c_model * self.cond_dim,
                    out_features=G * c_model,
                    solid=solid_name,
                )

        # __________________________________________________________________________________________
        # __TRANSFORMER_____________________________________________________________________________

        self.transformer_norm = NormPlatonic(
            "RMSNorm", solid_name, c_model, normalize_per_g, bias=False
        )

        # Transformer trunk  (ModernTransformerPlatonic)
        self.transformer = transformer_factory(
            c_model=c_model,
            depth=num_layers,
            repr_layer=aux_layer,
        )

        # Optional cross attention layers  (ModernTransformerDecoderBlockPlatonic)
        if self.use_cross_attn:
            self.atom_types_cross_attn = cross_attn_factory()
            self.coords_cross_attn = cross_attn_factory()
            self.frac_coords_cross_attn = cross_attn_factory()
            self.lengths_scaled_cross_attn = cross_attn_factory()
            self.angles_radians_cross_attn = cross_attn_factory()

        # __________________________________________________________________________________________
        # __OUTPUT_PROJECTIONS______________________________________________________________________

        # Scalar/invariant outputs
        # - Atom types are obviously scalars.
        # - Fractional coordinates are invariant coeffs rel to an equivariant crystal lattice basis.
        # - The basis is here represented in terms of invariant lengths and angles.
        # Projections using no nonlinearities replace
        #       scalar_unlift ∘ PlatonicLinear  =  Linear ∘ scalar_unlift
        # This is valid since unlifting via G-averaging is equivalent to averaging the G-conv kernel
        # in PlatonicLinear, turning it into a conventional linear map.
        self.out_atom_types = nn.Sequential(
            PlatonicLinear(
                G * c_model, G * c_model, solid=solid_name, bias=True
            ),  # (B, N, G*c_model)
            nn.SiLU(inplace=False),
            ProjRegularToScalar(solid_name),  # (B, N, c_model)
            nn.Linear(c_model, num_atom_types, bias=True),  # (B, N, num_atom_types)
        )
        self.out_frac_coords = nn.Sequential(
            ProjRegularToScalar(solid_name),  # (B, N, c_model)
            nn.LayerNorm(c_model),
            nn.Linear(c_model, spatial_dim, bias=False),  # (B, N, spatial_dim)
        )
        self.out_lengths_scaled = nn.Sequential(
            ProjRegularToScalar(solid_name),  # (B, N, c_model)
            nn.LayerNorm(c_model),
            nn.Linear(c_model, spatial_dim, bias=False),  # (B, N, spatial_dim)
        )
        self.out_angles_radians = nn.Sequential(
            ProjRegularToScalar(solid_name),  # (B, N, c_model)
            nn.LayerNorm(c_model),
            nn.Linear(c_model, spatial_dim, bias=False),  # (B, N, spatial_dim)
        )

        # Equivariant vector outputs (coordinates)
        self.out_coords = nn.Sequential(
            NormPlatonic("LayerNorm", solid_name, c_model, normalize_per_g),
            PlatonicLinear(
                G * c_model, G * spatial_dim, solid=solid_name, bias=False
            ),  # (B, N, G*dim)
            ProjRegularToVector(solid_name, flatten=True),  # (B, N, spatial_dim)
        )

        # __________________________________________________________________________________________
        # __AUXILIARY_TASK_HEADS____________________________________________________________________

        self.auxiliary_tasks = ["global_property", "global_energy", "atomic_forces"]

        # Optional projection layers from c_model to c_aux  (omitted if they agree)
        self.global_property_proj = (
            PlatonicLinear(G * c_model, G * c_aux, solid=solid_name, bias=False)
            if c_model != c_aux
            else nn.Identity()
        )
        self.global_energy_proj = (
            PlatonicLinear(G * c_model, G * c_aux, solid=solid_name, bias=False)
            if c_model != c_aux
            else nn.Identity()
        )
        self.atomic_forces_proj = (
            PlatonicLinear(G * c_model, G * c_aux, solid=solid_name, bias=False)
            if c_model != c_aux
            else nn.Identity()
        )

        # Optional transformer heads
        if self.num_aux_layers > 0:
            if self.use_cross_attn:
                self.global_property_cross_attention = cross_attn_factory()
                self.global_energy_cross_attention = cross_attn_factory()
                self.atomic_forces_cross_attention = cross_attn_factory()

            self.global_energy_charge_proj = ChargeSpinEmbedding(
                embedding_type="pos_emb",
                embedding_target="charge",
                embedding_size=c_model,
                grad=True,
                scale=1.0,
            )
            self.global_energy_spin_proj = ChargeSpinEmbedding(
                embedding_type="pos_emb",
                embedding_target="spin",
                embedding_size=c_model,
                grad=True,
                scale=1.0,
            )

            aux_transformer_kwargs = vars(
                SimpleNamespace(
                    c_model=c_aux,
                    depth=num_aux_layers,
                    repr_layer=None,
                )
            )
            self.global_property_transformer = transformer_factory(**aux_transformer_kwargs)
            self.global_energy_transformer = transformer_factory(**aux_transformer_kwargs)
            self.atomic_forces_transformer = transformer_factory(**aux_transformer_kwargs)

        # Final projection + unlifting
        self.global_property_head = nn.Sequential(
            ProjRegularToScalar(solid_name),  # (B, N, c_aux)
            nn.Linear(c_aux, num_properties, bias=True),  # (B, N, num_properties)
        )
        self.global_energy_head = nn.Sequential(
            ProjRegularToScalar(solid_name),  # (B, N, c_aux)
            nn.Linear(c_aux, 1, bias=True),  # (B, N, 1)
        )
        self.atomic_forces_head = nn.Sequential(
            PlatonicLinear(
                G * c_aux, G * spatial_dim, solid_name, bias=False
            ),  # (B, N, G*spatial_dim)
            ProjRegularToVector(solid_name, flatten=True),  # (B, N, spatial_dim)
        )

        # __________________________________________________________________________________________
        # __WEIGHT_INIT_____________________________________________________________________________
        self.custom_weight_init = custom_weight_init
        if custom_weight_init is not None:
            log.info(f"Initializing weights via {self.custom_weight_init} method.")
            self.apply(self._custom_weight_init)

    @typecheck
    def _custom_weight_init(self, module: nn.Module):
        """Initialize the weights of the module with a custom method.

        Args:
            module: The module to initialize.
        """
        raise NotImplementedError(
            "Not yet implemented, I'll have to think about how this interferes with Platonic ops."
        )

    @typecheck
    def _get_embedding(self, x, t, feats, padding_mask, token_is_periodic, sample_is_periodic):
        """Platonic group equivariant embeddings, split off from forward for modularization and
        independent unit testing.

        Embeddings will be combined in one of the following four ways, depending on whether
        self.coords_embed is None or not  and  self.concat_combine_input is True/False:
        1) coords_embed is None. Lift after combining all scalar embeddings:
          1A) concatenate embeddings:
              - concatenate invariant embeddings along channels                  (B, N, 6*c_model)
              - combine them via nn.Linear(6*c_model, c_model)                   (B, N, c_model)
              - lift scalar embeddings to the Platonic group / regular reps      (B, N, G*c_model)
          1B) sum embeddings:
              - sum invariant embeddings                                         (B, N, c_model)
              - lift scalar embeddings to the Platonic group / regular reps      (B, N, G*c_model)
        2) coords_embed is PlatonicSinusoidAPE/PlatonicLinearAPE. Lift scalar
           embeddings before combining them with regular rep coord embeddings.
           Coordinate embeddings already come lifted to shape (B, N, G*c_model).
          2A) concatenate embeddings:
              - lift scalar embeddings to the Platonic group / regular reps   6x (B, N, G*c_model)
              - concatenate all 7=6+1 lifted embeddings along channels           (B, N, 7*G*c_model)
              - combine them via PlatonicLinear(7*G*c_model, G*c_model)          (B, N, G*c_model)
          2B) sum embeddings:
              - lift scalar embeddings to the Platonic group / regular reps   6x (B, N, G*c_model)
              - sum all 7=6+1 lifted embeddings                                  (B, N, G*c_model)

        Args: see .forward()

        Returns: Embedding tensor h_in of shape (B, N, G*c_model), transforming as regular rep.
        """

        atom_types, coords, frac_coords, lengths_scaled, angles_radians = x
        atom_types_t, coords_t, frac_coords_t, lengths_scaled_t, angles_radians_t = t

        device = padding_mask.device
        B, N = padding_mask.shape

        dataset_idx = feats["dataset_idx"]
        spacegroup = feats["spacegroup"]

        # Ensure atom coordinates are masked out for periodic samples and the
        # remaining continuous modalities are masked out for non-periodic samples
        coords = coords * ~token_is_periodic
        frac_coords = frac_coords * token_is_periodic
        lengths_scaled = lengths_scaled * sample_is_periodic
        angles_radians = angles_radians * sample_is_periodic

        # __________________________________________________________________________________________
        # __INVARIANT_EMBEDDINGS____________________________________________________________________
        # For atom-types, fractional coords, lattice vector lengths and angles.
        # Same as in the non-equivariant TransformerModule.

        # Node-wise invariant embeddings, shape (B, N, c_model)
        embed_atom_types = self.atom_type_embed(atom_types.argmax(dim=-1))
        embed_frac_coords = self.frac_coords_embed(frac_coords)  # frac_coords are scalars!
        embed_lengths_scaled = self.lengths_scaled_embed(lengths_scaled)
        embed_angles_radians = self.angles_radians_embed(angles_radians)

        if self.use_sequence_sin_ape:  # Along *sequence* axis in SMILES ordering, E(3)-invariant
            embed_sequence = self.sequence_sin_ape(batch_size=B, seq_len=N)
        else:
            embed_sequence = torch.zeros(B, N, self.c_model, device=device)

        # Global invariant time/dataset/spacegroup embeddings, shape (B, 1, c_model), summed
        modals_t = torch.cat(
            [
                t.unsqueeze(-1)
                for t in [
                    atom_types_t,
                    coords_t,
                    frac_coords_t,
                    lengths_scaled_t,
                    angles_radians_t,
                ]
            ],
            dim=-1,
        )
        embed_time = (
            self.time_encoding(modals_t.reshape(-1)).reshape(B, modals_t.shape[1], -1).mean(-2)
        )  # (B, c_model), average over modalities
        embed_dataset = self.dataset_embedder(dataset_idx, self.training)  # (B, c_model)
        embed_spacegroup = self.spacegroup_embedder(spacegroup, self.training)  # (B, c_model)
        # Sum global embeddings, shape (B, 1, c_model)
        embed_conditions = (embed_time + embed_dataset + embed_spacegroup).unsqueeze(-2)

        assert all(
            embed.shape == (B, N, self.c_model)
            for embed in [
                embed_atom_types,
                embed_frac_coords,
                embed_sequence,
            ]
        ) and all(
            embed.shape == (B, 1, self.c_model)
            for embed in [
                embed_lengths_scaled,
                embed_angles_radians,
                embed_conditions,
            ]
        ), f"Embedding shapes are inconsistent. Shapes: {[embed.shape for embed in [embed_atom_types, embed_frac_coords, embed_lengths_scaled, embed_angles_radians, embed_conditions, embed_sequence]]}"

        # __________________________________________________________________________________________
        # __NO_COORDINATE_APE_______________________________________________________________________
        # Full E(3)-invariance, allowing for translation + Platonic group equivariance via G-RoPE.
        # First combine invariant embeddings via concatenation or sum, then lift to G.
        if self.coords_embed is None:

            if self.concat_combine_input:
                # Concatenate invariant embeddings along channels, shape (B, N, 6*c_model)
                h_in = torch.cat(
                    [
                        embed_atom_types,
                        embed_frac_coords,
                        embed_sequence,
                        embed_lengths_scaled.expand(-1, N, -1),
                        embed_angles_radians.expand(-1, N, -1),
                        embed_conditions.expand(-1, N, -1),
                    ],
                    dim=-1,
                )
                assert h_in.shape == (
                    B,
                    N,
                    self.c_model * self.cond_dim,
                ), f"h_in.shape: {h_in.shape}"
                # Combine them via nn.Linear(6*c_model, c_model)
                h_in = self.combine_input(h_in)  # (B, N, c_model)
                assert h_in.shape == (B, N, self.c_model), f"h_in.shape: {h_in.shape}"

            else:
                # Sum invariant embeddings, shape (B, N, c_model)
                h_in = (
                    embed_atom_types
                    + embed_frac_coords
                    + embed_lengths_scaled
                    + embed_angles_radians
                    + embed_sequence
                    + embed_conditions
                )

            # Lift scalar embeddings to regular representation features on Platonic group
            h_in = self.G_lift_scalars(h_in)  # (B, N, G*c_model)

        # __________________________________________________________________________________________
        # __PLATONIC_COORDINATE_APE_________________________________________________________________
        # Platonic group equivariant absolute coordinate embeddings.
        # Breaking translation equivariance but preserving Platonic symmetries G < O(3).
        # First lift invariant embeddings to G, then combine via concatenation/sum.
        else:  # self.coords_embed is not None

            embed_coords = self.coords_embed(coords)  # (B, N, G*c_model)

            # Lift scalar embeddings to regular representation features on G, 6x shape (B, N, G*c_model)
            embed_atom_types = self.G_lift_scalars(embed_atom_types)
            embed_frac_coords = self.G_lift_scalars(embed_frac_coords)
            embed_sequence = self.G_lift_scalars(embed_sequence)
            embed_lengths_scaled = self.G_lift_scalars(embed_lengths_scaled).expand(-1, N, -1)
            embed_angles_radians = self.G_lift_scalars(embed_angles_radians).expand(-1, N, -1)
            embed_conditions = self.G_lift_scalars(embed_conditions).expand(-1, N, -1)

            if self.concat_combine_input:
                # Concatenate all 6+1 lifted embeddings along channels, shape (B, N, 7*G*c_model)
                h_in = torch.cat(
                    [
                        embed_atom_types.view(B, N, self.G, self.c_model),
                        embed_coords.view(B, N, self.G, self.c_model),
                        embed_frac_coords.view(B, N, self.G, self.c_model),
                        embed_lengths_scaled.view(B, N, self.G, self.c_model),
                        embed_angles_radians.view(B, N, self.G, self.c_model),
                        embed_sequence.view(B, N, self.G, self.c_model),
                        embed_conditions.view(B, N, self.G, self.c_model),
                    ],
                    dim=-1,
                ).view(B, N, self.G * self.c_model * self.cond_dim)

                # Combine them via PlatonicLinear(7*G*c_model, G*c_model)
                h_in = self.combine_input(h_in)  # (B, N, G*c_model)
                assert h_in.shape == (B, N, self.G * self.c_model), f"h_in.shape: {h_in.shape}"

            else:
                # Sum all 6+1 lifted embeddings, shape (B, N, G*c_model)
                h_in = (
                    embed_atom_types
                    + embed_coords
                    + embed_frac_coords
                    + embed_lengths_scaled
                    + embed_angles_radians
                    + embed_sequence
                    + embed_conditions
                )

        return h_in

    @typecheck
    def forward(
        self,
        x: (
            Tuple[
                Int["b m v"],  # type: ignore - atom_types
                Float["b m 3"],  # type: ignore - coords
                Float["b m 3"],  # type: ignore - frac_coords
                Float["b 1 3"],  # type: ignore - lengths_scaled
                Float["b 1 3"],  # type: ignore - angles_radians
            ]
            | List[torch.Tensor]
        ),
        t: (
            Tuple[
                Float[" b"],  # type: ignore - atom_types_t
                Float[" b"],  # type: ignore - coords_t
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
            Float["b m 3"],  # type: ignore - coords
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
                atom_types:         Atom types tensor                   (B, N, V), where V is the number of atom types.
                coords:             Atom Euclidean coordinates tensor   (B, N, 3).
                frac_coords:        Fractional coordinates tensor       (B, N, 3).
                lengths_scaled:     Scaled lengths tensor               (B, 1, 3).
                angles_radians:     Angles in radians tensor            (B, 1, 3).
            t: Tuple or list of time tensors for each modality:
                atom_types_t:       Time t for atom types               (B,).
                coords_t:           Time t for coordinates              (B,).
                frac_coords_t:      Time t for fractional coordinates   (B,).
                lengths_scaled_t:   Time t for lengths                  (B,).
                angles_radians_t:   Time t for angles                   (B,).
            feats: Features for conditioning including:
                dataset_idx:        Dataset index for each sample.
                spacegroup:         Spacegroup index for each sample.
                token_is_periodic:  Whether each token corresponds to a periodic sample (B, N).
            padding_mask: True if padding token, False otherwise (B, N).
            kwargs: Any additional keyword arguments.

        Returns:
            A tuple containing output velocity fields for each modality as an inner tuple
            and auxiliary task outputs as another inner tuple.
        """
        device = padding_mask.device
        B, N = padding_mask.shape

        coords = x[1]  # (B, N, 3)
        token_is_periodic = feats["token_is_periodic"].unsqueeze(-1)
        sample_is_periodic = token_is_periodic.any(-2, keepdim=True)
        real_mask = 1 - padding_mask.int()
        global_mask = real_mask.any(-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
        sequence_idxs = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)

        # __________________________________________________________________________________________
        # __EMBEDDINGS______________________________________________________________________________

        h_in = self._get_embedding(
            x, t, feats, padding_mask, token_is_periodic, sample_is_periodic
        )
        h_in = h_in * real_mask.unsqueeze(-1)

        # __________________________________________________________________________________________
        # __TRANSFORMER_____________________________________________________________________________

        h_in = self.transformer_norm(h_in)

        # Self-attention transformer-encoder
        h_out, h_aux = self.transformer(
            feat=h_in,
            coords_feat=coords,
            sequence_idxs=sequence_idxs,
            padding_mask_feat=padding_mask,
            attn_mask_self=None,
            # avg_num_nodes=avg_num_nodes,
        )
        h_out = h_out * real_mask.unsqueeze(-1)

        # __________________________________________________________________________________________
        # __OUTPUT_PROJECTIONS_/_CROSS_ATTN_________________________________________________________

        if self.use_cross_attn:
            cross_attn_kwargs = vars(
                SimpleNamespace(
                    feat=h_out,
                    memory=h_in,
                    coords_feat=coords,
                    coords_mem=coords,
                    sequence_idxs=sequence_idxs,
                    padding_mask_feat=padding_mask,
                    padding_mask_mem=padding_mask,
                    attn_mask_self=None,
                    attn_mask_cross=None,
                    # avg_num_nodes_self=avg_num_nodes,
                    # avg_num_nodes_cross=avg_num_nodes,
                )
            )
            h_atom = self.atom_types_cross_attn(**cross_attn_kwargs)
            h_coords = self.coords_cross_attn(**cross_attn_kwargs)
            h_frac_coords = self.frac_coords_cross_attn(**cross_attn_kwargs)
            h_lengths_scaled = self.lengths_scaled_cross_attn(**cross_attn_kwargs)
            h_angles_radians = self.angles_radians_cross_attn(**cross_attn_kwargs)

            # Final projection + unlifting
            out_atom_types = self.out_atom_types(h_atom)
            out_coords = self.out_coords(h_coords)
            frac_coords = self.out_frac_coords(h_frac_coords)
            lengths_scaled = self.out_lengths_scaled(h_lengths_scaled.mean(-2, keepdim=True))
            angles_radians = self.out_angles_radians(h_angles_radians.mean(-2, keepdim=True))

        else:
            # Direct projection + unlifting, no cross-attention
            out_atom_types = self.out_atom_types(h_out)
            out_coords = self.out_coords(h_out)
            frac_coords = self.out_frac_coords(h_out)
            lengths_scaled = self.out_lengths_scaled(h_out.mean(-2, keepdim=True))
            angles_radians = self.out_angles_radians(h_out.mean(-2, keepdim=True))

        # __________________________________________________________________________________________
        # __AUXILIARY_TASK_HEADS____________________________________________________________________

        h_global_property = h_aux
        h_global_energy = h_aux
        h_atomic_forces = h_aux

        if self.num_aux_layers > 0:

            if self.use_cross_attn:
                aux_cross_attn_kwargs = vars(
                    SimpleNamespace(
                        feat=h_aux,
                        memory=h_in,
                        coords_feat=coords,
                        coords_mem=coords,
                        sequence_idxs=sequence_idxs,
                        padding_mask_feat=padding_mask,
                        padding_mask_mem=padding_mask,
                        attn_mask_self=None,
                        attn_mask_cross=None,
                        # avg_num_nodes_self=avg_num_nodes,
                        # avg_num_nodes_cross=avg_num_nodes,
                    )
                )
                h_global_property = self.global_property_cross_attention(**aux_cross_attn_kwargs)
                h_global_energy = self.global_energy_cross_attention(**aux_cross_attn_kwargs)
                h_atomic_forces = self.atomic_forces_cross_attention(**aux_cross_attn_kwargs)

            # Compute scalar charge/spin embeddings,  lift them to the group.
            ce = self.global_energy_charge_proj(feats["charge"])
            se = self.global_energy_spin_proj(feats["spin"])
            cse = (ce + se).unsqueeze(-2)  # (B, 1, c_model)
            cse = self.G_lift_scalars(cse)  # (B, 1, G*c_model)

            h_global_energy = h_global_energy + cse
            h_atomic_forces = h_atomic_forces + cse

            aux_self_attn_kwargs = vars(
                SimpleNamespace(
                    coords_feat=coords,
                    sequence_idxs=sequence_idxs,
                    padding_mask_feat=padding_mask,
                    attn_mask_self=None,
                    # avg_num_nodes=avg_num_nodes,
                )
            )
            h_global_property = self.global_property_transformer(
                feat=self.global_property_proj(h_global_property),
                **aux_self_attn_kwargs,
            )  # (B, N, G*c_aux)
            h_global_energy = self.global_energy_transformer(
                feat=self.global_energy_proj(h_global_energy),
                **aux_self_attn_kwargs,
            )  # (B, N, G*c_aux)
            h_atomic_forces = self.atomic_forces_transformer(
                feat=self.atomic_forces_proj(h_atomic_forces),
                **aux_self_attn_kwargs,
            )  # (B, N, G*c_aux)

        else:
            h_global_property = self.global_property_proj(h_global_property)  # (B, N, G*c_aux)
            h_global_energy = self.global_energy_proj(h_global_energy)  # (B, N, G*c_aux)
            h_atomic_forces = self.atomic_forces_proj(h_atomic_forces)  # (B, N, G*c_aux)

        h_global_property_mean = h_global_property.mean(-2, keepdim=True)  # (B, 1, G*c_aux)
        h_global_energy_mean = h_global_energy.mean(-2, keepdim=True)  # (B, 1, G*c_aux)
        # Final projection + unlifting
        global_property = self.global_property_head(h_global_property_mean) * global_mask
        global_energy = self.global_energy_head(h_global_energy_mean) * global_mask
        atomic_forces = self.atomic_forces_head(h_atomic_forces) * real_mask.unsqueeze(-1)

        # __________________________________________________________________________________________
        # __RETURN_PREDICTIONS______________________________________________________________________

        pred_modals = (
            out_atom_types * real_mask.unsqueeze(-1),  # (B, N, V=self.vocab_size)
            out_coords * real_mask.unsqueeze(-1) * ~token_is_periodic,  # (B, N, 3)
            frac_coords * real_mask.unsqueeze(-1) * token_is_periodic,  # (B, N, 3)
            lengths_scaled * global_mask * sample_is_periodic,  # (B, 1, 3)
            angles_radians * global_mask * sample_is_periodic,  # (B, 1, 3)
        )

        pred_aux_outputs = (
            global_property,  # (B, 1, P)
            global_energy,  # (B, 1, 1)
            atomic_forces,  # (B, N, 3)
        )

        return pred_modals, pred_aux_outputs

    @typecheck
    def forward_with_cfg(
        self,
        x: (
            Tuple[
                Int["b m v"],  # type: ignore - atom_types
                Float["b m 3"],  # type: ignore - coords
                Float["b m 3"],  # type: ignore - frac_coords
                Float["b 1 3"],  # type: ignore - lengths_scaled
                Float["b 1 3"],  # type: ignore - angles_radians
            ]
            | List[torch.Tensor]
        ),
        t: (
            Tuple[
                Float[" b"],  # type: ignore - atom_types_t
                Float[" b"],  # type: ignore - coords_t
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
            Float["b m 3"],  # type: ignore - coords
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
                atom_types: Atom types tensor (B, N, V), where V is the number of atom types.
                coords: Atom Euclidean coordinates tensor (B, N, 3).
                frac_coords: Fractional coordinates tensor (B, N, 3).
                lengths_scaled: Scaled lengths tensor (B, 1, 3).
                angles_radians: Angles in radians tensor (B, 1, 3).
            t: Tuple or list of time tensors for each modality:
                atom_types_t: Time t for atom types (B,).
                coords_t: Time t for Euclidean coordinates (B,).
                frac_coords_t: Time t for fractional coordinates (B,).
                lengths_scaled_t: Time t for lengths (B,).
                angles_radians_t: Time t for angles (B,).
            feats: Features for conditioning including:
                dataset_idx: Dataset index for each sample.
                spacegroup: Spacegroup index for each sample.
                token_is_periodic: Whether each token corresponds to a periodic sample (B, N).
            padding_mask: True if padding token, False otherwise (B, N).
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
