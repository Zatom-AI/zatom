"""Diffusion Platonic Transformer.

Adapted from:
    - https://github.com/apple/ml-simplefold
    - https://github.com/niazoys/PlatonicTransformers
"""

import math
from typing import Dict, List, Literal, Tuple, Type

import torch
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from torch import Tensor, nn
from torch.nn.attention import SDPBackend

from zatom.models.architectures.modules.layers import FinalLayer
from zatom.utils.training_utils import SDPA_BACKENDS
from zatom.utils.typing_utils import Bool, Float, Int, typecheck

PLATONIC_GROUP_NAMES = Literal[
    "trivial",
    "tetrahedron",
    "octahedron",
    "icosahedron",
    "octahedron_reflections",
]

#################################################################################
#                        Diffusion Platonic Transformer (DiP)                   #
#################################################################################


class MultimodalDiP(nn.Module):
    """DiP model for multimodal atomic data.

    Args:
        time_embedder: The time embedder module.
        dataset_embedder: The dataset embedder module.
        spacegroup_embedder: The spacegroup embedder module.
        token_pos_embedder: The positional embedder for token positions.
        atom_pos_embedder: The positional embedder for atom positions.
        trunk: The trunk transformer module for tokens.
        atom_encoder_transformer: The transformer module for atom encoding.
        atom_decoder_transformer: The transformer module for atom decoding.
        hidden_size: The hidden size for the model.
        token_num_heads: The number of (token) attention heads for the trunk transformer.
        atom_num_heads: The number of (atom) attention heads for the atom transformers.
        atom_hidden_size_enc: The hidden size for the atom encoder transformer.
        atom_hidden_size_dec: The hidden size for the atom decoder transformer.
        atom_n_queries_enc: The number of query positions for local attention in the atom encoder.
        atom_n_keys_enc: The number of key positions for local attention in the atom encoder.
        atom_n_queries_dec: The number of query positions for local attention in the atom decoder.
        atom_n_keys_dec: The number of key positions for local attention in the atom decoder.
        max_num_elements: The maximum number of unique atom types (elements).
        use_length_condition: Whether to condition on sequence length.
        add_mask_atom_type: Whether to add a special mask atom type.
        treat_discrete_modalities_as_continuous: Whether to treat discrete modalities as continuous (one-hot) vectors for flow matching.
        remove_t_conditioning: Whether to remove time conditioning.
        jvp_attn: Whether to use JVP Flash Attention instead of PyTorch's Scaled Dot Product Attention.
        solid_name: The name of the Platonic solid (e.g., `tetrahedron`, `octahedron`, `icosahedron`) to define the symmetry group.
    """

    def __init__(
        self,
        time_embedder: Type[nn.Module],
        dataset_embedder: Type[nn.Module],
        spacegroup_embedder: Type[nn.Module],
        token_pos_embedder: Type[nn.Module],
        atom_pos_embedder: Type[nn.Module],
        trunk: Type[nn.Module],
        atom_encoder_transformer: Type[nn.Module],
        atom_decoder_transformer: Type[nn.Module],
        hidden_size: int = 768,
        token_num_heads: int = 12,
        atom_num_heads: int = 4,
        atom_hidden_size_enc: int = 256,
        atom_hidden_size_dec: int = 256,
        atom_n_queries_enc: int = 32,
        atom_n_keys_enc: int = 128,
        atom_n_queries_dec: int = 32,
        atom_n_keys_dec: int = 128,
        max_num_elements: int = 100,
        use_length_condition: bool = True,
        add_mask_atom_type: bool = True,
        treat_discrete_modalities_as_continuous: bool = False,
        remove_t_conditioning: bool = False,
        jvp_attn: bool = False,
        solid_name: PLATONIC_GROUP_NAMES = "octahedron",
    ):
        raise NotImplementedError(
            "MultimodalDiP is not yet implemented. Please use MultimodalDiT instead."
        )

        super().__init__()
        self.time_embedder = time_embedder
        self.dataset_embedder = dataset_embedder
        self.spacegroup_embedder = spacegroup_embedder

        self.atom_pos_embedder = atom_pos_embedder
        atom_pos_embed_channels = atom_pos_embedder.embed_dim
        self.token_pos_embedder = token_pos_embedder
        token_pos_embed_channels = token_pos_embedder.embed_dim

        self.trunk = trunk

        self.atom_encoder_transformer = atom_encoder_transformer
        self.atom_decoder_transformer = atom_decoder_transformer

        self.hidden_size = hidden_size
        self.token_num_heads = token_num_heads
        self.atom_num_heads = atom_num_heads
        self.use_length_condition = use_length_condition
        self.remove_t_conditioning = remove_t_conditioning
        self.jvp_attn = jvp_attn

        self.atom_hidden_size_enc = atom_hidden_size_enc
        self.atom_hidden_size_dec = atom_hidden_size_dec
        self.atom_n_queries_enc = atom_n_queries_enc
        self.atom_n_keys_enc = atom_n_keys_enc
        self.atom_n_queries_dec = atom_n_queries_dec
        self.atom_n_keys_dec = atom_n_keys_dec

        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G

        vocab_size = max_num_elements + int(add_mask_atom_type)

        self.atom_type_embedder = (
            nn.Sequential(
                nn.Linear(vocab_size, hidden_size, bias=False),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
            )
            if treat_discrete_modalities_as_continuous
            else nn.Embedding(vocab_size, hidden_size)
        )
        self.lengths_scaled_embedder = nn.Sequential(
            nn.Linear(3, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.angles_radians_embedder = nn.Sequential(
            nn.Linear(3, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        atom_feat_dim = atom_pos_embed_channels + token_pos_embed_channels + hidden_size * 3 + 1
        self.atom_feat_proj = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        self.atom_pos_proj = nn.Linear(atom_pos_embed_channels, hidden_size, bias=False)
        self.frac_coords_proj = nn.Linear(atom_pos_embed_channels, hidden_size, bias=False)

        if self.use_length_condition:
            self.length_embedder = nn.Sequential(
                nn.Linear(1, hidden_size, bias=False),
                nn.LayerNorm(hidden_size),
            )

        self.atom_in_proj = nn.Linear(hidden_size * 3, hidden_size, bias=False)

        self.context2atom_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )
        self.atom2latent_proj = nn.Sequential(
            nn.Linear(self.atom_hidden_size_enc, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.atom_enc_cond_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )
        self.atom_dec_cond_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_dec),
            nn.LayerNorm(self.atom_hidden_size_dec),
        )

        self.latent2atom_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.atom_hidden_size_dec),
        )

        self.final_layer = FinalLayer(self.atom_hidden_size_dec, hidden_size, c_dim=hidden_size)

        self.atom_types_head = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pos_head = nn.Linear(hidden_size, 3, bias=False)
        self.frac_coords_head = nn.Linear(hidden_size, 3, bias=False)
        self.lengths_scaled_head = nn.Linear(hidden_size, 3, bias=True)
        self.angles_radians_head = nn.Linear(hidden_size, 3, bias=True)

    @typecheck
    def create_local_attn_mask(
        self,
        n: int,
        n_queries: int,
        n_keys: int,
        device: torch.device | None = None,
    ) -> Tensor:
        """Create local attention bias based on query window n_queries and kv window n_keys.

        Args:
            n: the length of quiries
            n_queries: window size of quiries
            n_keys: window size of keys/values
            device: The device of the attention bias. Defaults to None.

        Returns:
            The diagonal-like global attention bias.
        """
        n_trunks = int(math.ceil(n / n_queries))
        padded_n = n_trunks * n_queries
        attn_mask = torch.zeros(padded_n, padded_n, device=device, dtype=torch.bool)
        for block_index in range(0, n_trunks):
            i = block_index * n_queries
            j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
            j2 = n_queries * block_index + (n_queries + n_keys) // 2
            attn_mask[i : i + n_queries, j1:j2] = True
        return attn_mask[:n, :n]

    @typecheck
    def create_atom_attn_mask(
        self,
        natoms: int,
        atom_n_queries: int | None = None,
        atom_n_keys: int | None = None,
        device: torch.device | None = None,
    ) -> Tensor | None:
        """Create attention mask for atoms.

        NOTE: Assumes each batch consists of a single unique example.

        Args:
            natoms: The number of atoms.
            atom_n_queries: The number of query positions for local attention.
            atom_n_keys: The number of key positions for local attention.
            device: The device of the attention mask. Defaults to None.

        Returns:
            The attention mask for atoms.
        """
        if atom_n_queries is not None and atom_n_keys is not None:
            atom_attn_mask = self.create_local_attn_mask(
                n=natoms,
                n_queries=atom_n_queries,
                n_keys=atom_n_keys,
                device=device,
            )
        else:
            atom_attn_mask = None

        return atom_attn_mask

    @typecheck
    def forward(
        self,
        x: (
            Tuple[
                Int["b m"] | Float["b m v"],  # type: ignore - atom_types
                Float["b m 3"],  # type: ignore - pos
                Float["b m 3"],  # type: ignore - frac_coords
                Float["b 1 3"],  # type: ignore - lengths_scaled
                Float["b 1 3"],  # type: ignore - angles_radians
            ]
            | List[torch.Tensor]
        ),
        t: (
            Tuple[
                Float[" b"] | Tuple[Float[" b"], Float[" b"]],  # type: ignore - atom_types_t, maybe atom_types_r as well
                Float[" b"] | Tuple[Float[" b"], Float[" b"]],  # type: ignore - pos_t, maybe pos_r as well
                Float[" b"] | Tuple[Float[" b"], Float[" b"]],  # type: ignore - frac_coords_t, maybe frac_coords_r as well
                Float[" b"] | Tuple[Float[" b"], Float[" b"]],  # type: ignore - lengths_scaled_t, maybe lengths_scaled_r as well
                Float[" b"] | Tuple[Float[" b"], Float[" b"]],  # type: ignore - angles_radians_t, maybe angles_radians_r as well
            ]
            | List[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]
        ),
        feats: Dict[str, Tensor],
        mask: Bool["b m"],  # type: ignore
        sdpa_backends: List[SDPBackend] = SDPA_BACKENDS,  # type: ignore
    ) -> Tuple[
        Float["b m v"],  # type: ignore - atom_types
        Float["b m 3"],  # type: ignore - pos
        Float["b m 3"],  # type: ignore - frac_coords
        Float["b 1 3"],  # type: ignore - lengths_scaled
        Float["b 1 3"],  # type: ignore - angles_radians
    ]:
        """Forward pass of the DiP model.

        Args:
            x: Tuple or list of input tensors for each modality:
                atom_types: Atom types tensor (B, M) or (B, M, V), where V is the number of atom types.
                pos: Atom positions tensor (B, M, 3).
                frac_coords: Fractional coordinates tensor (B, M, 3).
                lengths_scaled: Scaled lengths tensor (B, 1, 3).
                angles_radians: Angles in radians tensor (B, 1, 3).
            t: Tuple or list of time tensors for each modality:
                atom_types_t/r: Time t (and maybe also time r) for atom types (B,).
                pos_t/r: Time t (and maybe also time r) for positions (B,).
                frac_coords_t/r: Time t (and maybe also time r) for fractional coordinates (B,).
                lengths_scaled_t/r: Time t (and maybe also time r) for lengths (B,).
                angles_radians_t/r: Time t (and maybe also time r) for angles (B,).
            feats: Features for conditioning including:
                dataset_idx: Dataset index for each sample.
                spacegroup: Spacegroup index for each sample.
                ref_pos: Reference atom positions tensor.
                ref_space_uids: Reference space unique IDs tensor.
                atom_to_token: One-hot mapping from atom indices to token indices.
                atom_to_token_idx: Mapping from atom indices to token indices.
                max_num_tokens: Maximum number of unmasked tokens for each batch element.
                token_index: Indices of the tokens in the batch.
            mask: True if valid token, False if padding (B, M).
            sdpa_backends: List of SDPBackend backends to try when using fused attention. Defaults to all.

        Returns:
            Output velocity fields for each modality as a tuple.
        """
        # Organize inputs
        atom_types, pos, frac_coords, lengths_scaled, angles_radians = x
        atom_types_t, pos_t, frac_coords_t, lengths_scaled_t, angles_radians_t = t

        device = mask.device
        batch_size, num_atoms = _, num_tokens = mask.shape

        dataset_idx = feats["dataset_idx"]
        spacegroup = feats["spacegroup"]

        atom_to_token = feats["atom_to_token"]
        atom_to_token_idx = feats["atom_to_token_idx"]
        ref_space_uid = feats["ref_space_uid"]

        modals_t = torch.cat(
            [
                # Average velocity time steps
                (
                    torch.stack(t).unsqueeze(-1) * 0
                    if self.remove_t_conditioning
                    else (
                        torch.stack(t).unsqueeze(-1)
                        if isinstance(t, tuple)
                        # Instantaneous velocity time steps
                        else t.unsqueeze(-1) * 0 if self.remove_t_conditioning else t.unsqueeze(-1)
                    )
                )
                for t in [atom_types_t, pos_t, frac_coords_t, lengths_scaled_t, angles_radians_t]
            ],
            dim=-1,
        )
        if modals_t.ndim == 3:
            # (2, B, 5) -> (B, 10) when both t and r are provided
            modals_t = modals_t.reshape(batch_size, -1)

        # Create attention masks
        pairwise_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, M, M)
        attention_mask = pairwise_mask.unsqueeze(1)

        atom_attn_mask_enc = self.create_atom_attn_mask(
            natoms=num_atoms,
            atom_n_queries=self.atom_n_queries_enc,
            atom_n_keys=self.atom_n_keys_enc,
            device=device,
        )
        atom_attn_mask_dec = self.create_atom_attn_mask(
            natoms=num_atoms,
            atom_n_queries=self.atom_n_queries_dec,
            atom_n_keys=self.atom_n_keys_dec,
            device=device,
        )

        atom_attn_mask_enc = (
            attention_mask
            if atom_attn_mask_enc is None
            else atom_attn_mask_enc[None, None, ...].expand(batch_size, 1, -1, -1)
        )
        atom_attn_mask_dec = (
            attention_mask
            if atom_attn_mask_dec is None
            else atom_attn_mask_dec[None, None, ...].expand(batch_size, 1, -1, -1)
        )

        if self.jvp_attn:
            # NOTE: JVP Flash Attention expects the attention mask to be of shape (B, H, M, M)
            attention_mask = attention_mask.expand(-1, self.token_num_heads, -1, -1).contiguous()
            atom_attn_mask_enc = atom_attn_mask_enc.expand(
                -1, self.atom_num_heads, -1, -1
            ).contiguous()
            atom_attn_mask_dec = atom_attn_mask_dec.expand(
                -1, self.atom_num_heads, -1, -1
            ).contiguous()

        # Create condition embeddings for AdaLN
        t_emb = self.time_embedder(modals_t).mean(-2)  # (B, D), via averaging over all modalities
        d_emb = self.dataset_embedder(dataset_idx, self.training)  # (B, C)
        s_emb = self.spacegroup_embedder(spacegroup, self.training)  # (B, C)
        c_emb = t_emb + d_emb + s_emb
        if self.use_length_condition:
            length = feats["max_num_tokens"].float().unsqueeze(-1)
            c_emb = c_emb + self.length_embedder(torch.log(length))

        # Create atom features
        ref_pos_emb = self.atom_pos_embedder(pos=feats["ref_pos"])
        atom_token_pos = self.token_pos_embedder(pos=atom_to_token_idx.unsqueeze(-1).float())
        atom_types_emb = self.atom_type_embedder(atom_types)
        lengths_scaled_emb = self.lengths_scaled_embedder(lengths_scaled).expand(
            -1, num_atoms, -1
        )  # (B, M, D)
        angles_radians_emb = self.angles_radians_embedder(angles_radians).expand(
            -1, num_atoms, -1
        )  # (B, M, D)
        atom_feat = torch.cat(
            [
                ref_pos_emb,  # (B, M, PE1)
                atom_token_pos,  # (B, M, PE2)
                atom_types_emb,  # (B, M, C)
                lengths_scaled_emb,  # (B, M, C)
                angles_radians_emb,  # (B, M, C)
                mask.float().unsqueeze(-1),  # (B, M, 1)
            ],
            dim=-1,
        )  # (B, M, C * 5 + 1)
        atom_feat = self.atom_feat_proj(atom_feat)  # (B, M, D)

        atom_coord = self.atom_pos_embedder(pos=pos)  # (B, M, C)
        atom_coord = self.atom_pos_proj(atom_coord)  # (B, M, D)

        atom_frac_coord = self.atom_pos_embedder(pos=frac_coords)  # (B, M, C)
        atom_frac_coord = self.frac_coords_proj(atom_frac_coord)  # (B, M, D)

        atom_in = torch.cat(
            [atom_feat, atom_coord, atom_frac_coord],
            dim=-1,
        )
        atom_in = self.atom_in_proj(atom_in)  # (B, M, D)

        # Curate position embeddings for Axial RoPE
        atom_pe_pos = torch.cat(
            [
                ref_space_uid.unsqueeze(-1).float(),  # (B, M, 1)
                feats["ref_pos"],  # (B, M, 3)
            ],
            dim=-1,
        )  # (B, M, 4)
        token_pe_pos = feats["token_index"].unsqueeze(-1).float()  # (B, N, 1)

        # Run atom encoder
        atom_c_emb_enc = self.atom_enc_cond_proj(c_emb)
        atom_latent = self.context2atom_proj(atom_in)
        atom_latent = self.atom_encoder_transformer(
            latents=atom_latent,
            c=atom_c_emb_enc,
            attention_mask=atom_attn_mask_enc,
            pos=atom_pe_pos,
            sdpa_backends=sdpa_backends,
        )
        atom_latent = self.atom2latent_proj(atom_latent)

        # Grouping: aggregate atoms to tokens
        atom_to_token_mean = atom_to_token / (atom_to_token.sum(dim=1, keepdim=True) + 1e-6)
        latent = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_latent)
        assert (
            latent.shape[1] == num_tokens
        ), f"Latent must have {num_tokens} tokens, but got {latent.shape[1]}."

        # Run token trunk
        latent = self.trunk(
            latents=latent,
            c=c_emb,
            attention_mask=attention_mask,
            pos=token_pe_pos,
            sdpa_backends=sdpa_backends,
        )

        # Ungrouping: broadcast tokens to atoms
        output = torch.bmm(atom_to_token, latent)
        assert (
            output.shape[1] == num_atoms
        ), f"Output must have {num_atoms} atoms, but got {output.shape[1]}."

        # Add skip connection
        output = output + atom_latent
        output = self.latent2atom_proj(output)

        # Run atom decoder
        atom_c_emb_dec = self.atom_dec_cond_proj(c_emb)
        output = self.atom_decoder_transformer(
            latents=output,
            c=atom_c_emb_dec,
            attention_mask=atom_attn_mask_dec,
            pos=atom_pe_pos,
            sdpa_backends=sdpa_backends,
        )
        output = self.final_layer(output, c=c_emb)
        output = output * mask.unsqueeze(-1)  # Mask out padding atoms

        # Collect predictions
        global_mask = mask.any(-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
        pred_modals = (
            self.atom_types_head(output) * mask.unsqueeze(-1),  # (B, M, V=self.vocab_size)
            self.pos_head(output) * mask.unsqueeze(-1),  # (B, M, 3)
            self.frac_coords_head(output) * mask.unsqueeze(-1),  # (B, M, 3)
            self.lengths_scaled_head(output.mean(-2, keepdim=True)) * global_mask,  # (B, 1, 3)
            self.angles_radians_head(output.mean(-2, keepdim=True)) * global_mask,  # (B, 1, 3)
        )

        return pred_modals

    @typecheck
    def forward_with_cfg(
        self,
        x: (
            Tuple[
                Int["b m"] | Float["b m v"],  # type: ignore - atom_types
                Float["b m 3"],  # type: ignore - pos
                Float["b m 3"],  # type: ignore - frac_coords
                Float["b 1 3"],  # type: ignore - lengths_scaled
                Float["b 1 3"],  # type: ignore - angles_radians
            ]
            | List[torch.Tensor]
        ),
        t: (
            Tuple[
                Float[" b"] | Tuple[Float[" b"], Float[" b"]],  # type: ignore - atom_types_t, maybe atom_types_r as well
                Float[" b"] | Tuple[Float[" b"], Float[" b"]],  # type: ignore - pos_t, maybe pos_r as well
                Float[" b"] | Tuple[Float[" b"], Float[" b"]],  # type: ignore - frac_coords_t, maybe frac_coords_r as well
                Float[" b"] | Tuple[Float[" b"], Float[" b"]],  # type: ignore - lengths_scaled_t, maybe lengths_scaled_r as well
                Float[" b"] | Tuple[Float[" b"], Float[" b"]],  # type: ignore - angles_radians_t, maybe angles_radians_r as well
            ]
            | List[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]
        ),
        feats: Dict[str, Tensor],
        mask: Bool["b m"],  # type: ignore
        cfg_scale: float,
        sdpa_backends: List[SDPBackend] = SDPA_BACKENDS,  # type: ignore
    ) -> Tuple[
        Float["b m v"],  # type: ignore - atom_types
        Float["b m 3"],  # type: ignore - pos
        Float["b m 3"],  # type: ignore - frac_coords
        Float["b 1 3"],  # type: ignore - lengths_scaled
        Float["b 1 3"],  # type: ignore - angles_radians
    ]:
        """Forward pass of MultimodalDiP, but also batches the unconditional forward pass for
        classifier-free guidance.

        NOTE: Assumes batch x's and class labels are ordered such that the first half are the conditional
        samples and the second half are the unconditional samples.

        Args:
            x: Tuple or list of input tensors for each modality:
                atom_types: Atom types tensor (B, N) or (B, N, V), where V is the number of atom types.
                pos: Atom positions tensor (B, N, 3).
                frac_coords: Fractional coordinates tensor (B, N, 3).
                lengths_scaled: Scaled lengths tensor (B, 1, 3).
                angles_radians: Angles in radians tensor (B, 1, 3).
            t: Tuple or list of time tensors for each modality:
                atom_types_t/r: Time t (and maybe also time r) for atom types (B,).
                pos_t/r: Time t (and maybe also time r) for positions (B,).
                frac_coords_t/r: Time t (and maybe also time r) for fractional coordinates (B,).
                lengths_scaled_t/r: Time t (and maybe also time r) for lengths (B,).
                angles_radians_t/r: Time t (and maybe also time r) for angles (B,).
            feats: Features for conditioning including:
                dataset_idx: Dataset index for each sample.
                spacegroup: Spacegroup index for each sample.
                ref_pos: Reference atom positions tensor.
                ref_space_uids: Reference space unique IDs tensor.
                atom_to_token: One-hot mapping from atom indices to token indices.
                atom_to_token_idx: Mapping from atom indices to token indices.
                max_num_tokens: Maximum number of unmasked tokens for each batch element.
                token_index: Indices of the tokens in the batch.
            mask: True if valid token, False if padding (B, N).
            cfg_scale: Classifier-free guidance scale.
            sdpa_backends: List of SDPBackend backends to try when using fused attention. Defaults to all.

        Returns:
            Output velocity fields for each modality as a tuple.
        """
        half_x = tuple(x_[: len(x_) // 2] for x_ in x)
        combined_x = tuple(torch.cat([half_x_, half_x_], dim=0) for half_x_ in half_x)
        model_out = self.forward(
            combined_x,
            t,
            feats,
            mask,
            sdpa_backends=sdpa_backends,
        )

        eps = []
        for modal in model_out:
            cond_eps, uncond_eps = torch.split(modal, len(modal) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps.append(torch.cat([half_eps, half_eps], dim=0))

        return tuple(eps)
