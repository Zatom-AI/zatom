"""Multimodal flow transformer (MFT), version 2.

Adapted from:
    - https://github.com/apple/ml-simplefold
    - https://github.com/carlosinator/tabasco
    - https://github.com/facebookresearch/all-atom-diffusion-transformer
"""

import random
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from torch.nn.attention import SDPBackend

from zatom.flow.interpolants import Interpolant
from zatom.flow.path import FlowPath
from zatom.models.components.losses import InterDistancesLoss
from zatom.utils import pylogger
from zatom.utils.training_utils import (
    BEST_DEVICE,
    SDPA_BACKENDS,
    HistogramTimeDistribution,
)
from zatom.utils.typing_utils import Bool, typecheck

log = pylogger.RankedLogger(__name__)

LOSS_TIME_FUNCTIONS = Literal["none", "beta"]
GRAD_DECAY_METHODS = Literal["none", "linear_decay", "truncated_decay", "piecewise_decay"]


#################################################################################
#                           Multimodal Flow Transformer                         #
#################################################################################


class MFT2(nn.Module):
    """Multimodal flow model with an encoder-decoder architecture (v2).

    Typical usage:
    - `forward`:    Called during training to compute loss and optional stats.
    - `sample`:     Called during inference to run Euler sampler to generate new samples.

    Args:
        multimodal_model: The multimodal model to instantiate (e.g., MultimodalDiT).
        time_embedder: Time embedder module.
        dataset_embedder: Dataset embedder module.
        spacegroup_embedder: Spacegroup embedder module.
        token_pos_embedder: Token positional embedder module.
        atom_pos_embedder: Atom positional embedder module.
        atom_encoder_transformer: Multimodal atom encoder Transformer module.
        trunk: The Transformer trunk module.
        atom_decoder_transformer: Multimodal atom decoder Transformer module.
        atom_types_interpolant: Interpolant for atom types modality.
        pos_interpolant: Interpolant for atom positions modality.
        frac_coords_interpolant: Interpolant for fractional coordinates modality.
        lengths_scaled_interpolant: Interpolant for scaled lengths modality.
        angles_radians_interpolant: Interpolant for angles in radians modality.
        hidden_size: Hidden size of the model.
        token_num_heads: Number of (token) attention heads in the token trunk.
        atom_num_heads: Number of (atom) attention heads in the atom encoder/decoder.
        atom_hidden_size_enc: Hidden size of the atom encoder.
        atom_hidden_size_dec: Hidden size of the atom decoder.
        atom_n_queries_enc: Number of queries in the atom encoder.
        atom_n_keys_enc: Number of keys in the atom encoder.
        atom_n_queries_dec: Number of queries in the atom decoder.
        atom_n_keys_dec: Number of keys in the atom decoder.
        max_num_elements: Maximum number of elements in the dataset.
        batch_size_scale_factor: Factor by which to scale the global batch size when using a specific (e.g., 180M) model variant.
        use_length_condition: Whether to use the length condition.
        jvp_attn: Whether to use JVP Flash Attention instead of PyTorch's Scaled Dot Product Attention.
        interdist_loss: Type of interatomic distance loss to use. If None, no interatomic distance loss is used.
        time_distribution: Distribution to sample time points from. Must be one of (`uniform`, `beta`, `histogram`).
        time_alpha_factor: Alpha factor for beta time distribution.
    """

    def __init__(
        self,
        multimodal_model: partial[Callable[..., nn.Module]],
        time_embedder: nn.Module,
        dataset_embedder: nn.Module,
        spacegroup_embedder: nn.Module,
        token_pos_embedder: nn.Module,
        atom_pos_embedder: nn.Module,
        atom_encoder_transformer: nn.Module,
        trunk: nn.Module,
        atom_decoder_transformer: nn.Module,
        atom_types_interpolant: Interpolant,
        pos_interpolant: Interpolant,
        frac_coords_interpolant: Interpolant,
        lengths_scaled_interpolant: Interpolant,
        angles_radians_interpolant: Interpolant,
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
        batch_size_scale_factor: int = 1,
        use_length_condition: bool = True,
        jvp_attn: bool = False,
        interdist_loss: InterDistancesLoss | None = None,
        time_distribution: Literal["uniform", "beta", "histogram"] = "beta",
        time_alpha_factor: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__()

        self.atom_types_interpolant = atom_types_interpolant
        self.pos_interpolant = pos_interpolant
        self.frac_coords_interpolant = frac_coords_interpolant
        self.lengths_scaled_interpolant = lengths_scaled_interpolant
        self.angles_radians_interpolant = angles_radians_interpolant

        self.batch_size_scale_factor = batch_size_scale_factor
        self.class_dropout_prob = dataset_embedder.dropout_prob
        self.jvp_attn = jvp_attn
        self.interdist_loss = interdist_loss
        self.time_alpha_factor = time_alpha_factor

        self.vocab_size = max_num_elements
        self.continuous_x_1_prediction = True

        # Define time distribution
        if time_distribution == "uniform":
            self.time_distribution = torch.distributions.Uniform(0, 1)
        elif time_distribution == "beta":
            self.time_distribution = torch.distributions.Beta(time_alpha_factor, 1)
        elif time_distribution == "histogram":
            # TODO: chore: make pretty
            print("Using histogram time distribution")
            self.time_distribution = HistogramTimeDistribution(
                torch.tensor([0.05, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4])
            )
        else:
            raise ValueError(
                f"Invalid time distribution: {time_distribution}. Must be one of (`uniform`, `beta`, `histogram`)."
            )

        # Build multimodal model
        kwargs.pop("add_mask_atom_type", None)  # Remove if present
        self.model = multimodal_model(
            time_embedder=time_embedder,
            dataset_embedder=dataset_embedder,
            spacegroup_embedder=spacegroup_embedder,
            token_pos_embedder=token_pos_embedder,
            atom_pos_embedder=atom_pos_embedder,
            trunk=trunk,
            atom_encoder_transformer=atom_encoder_transformer,
            atom_decoder_transformer=atom_decoder_transformer,
            hidden_size=hidden_size,
            token_num_heads=token_num_heads,
            atom_num_heads=atom_num_heads,
            atom_hidden_size_enc=atom_hidden_size_enc,
            atom_hidden_size_dec=atom_hidden_size_dec,
            atom_n_queries_enc=atom_n_queries_enc,
            atom_n_keys_enc=atom_n_keys_enc,
            atom_n_queries_dec=atom_n_queries_dec,
            atom_n_keys_dec=atom_n_keys_dec,
            max_num_elements=max_num_elements,
            use_length_condition=use_length_condition,
            add_mask_atom_type=False,
            treat_discrete_modalities_as_continuous=False,
            remove_t_conditioning=False,
            jvp_attn=jvp_attn,
            **kwargs,
        )

        # Define modalities and auxiliary tasks
        self.modals = ["atom_types", "pos", "frac_coords", "lengths_scaled", "angles_radians"]
        self.auxiliary_tasks = self.model.auxiliary_tasks

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device

    @typecheck
    def _sample_noise_like_batch(
        self, batch: Optional[TensorDict] = None, batch_size: Optional[int] = None
    ):
        """Draw coordinate and atom-type noise compatible with `batch`.

        Args:
            batch: Optional TensorDict containing batch data.
            batch_size: Optional batch size if batch is None.

        Returns:
            A TensorDict containing noise samples.
        """
        if batch is None:
            assert hasattr(self, "data_stats"), "self.data_stats not set"
            assert batch_size is not None, "Batch size must be provided when batch is None"

            max_num_atoms = self.data_stats["max_num_atoms"]
            sampled_num_atoms = random.choices(  # nosec
                list(self.data_stats["num_atoms_histogram"].keys()),
                weights=list(self.data_stats["num_atoms_histogram"].values()),
                k=batch_size,
            )
            sampled_num_atoms = torch.tensor(sampled_num_atoms)

            pad_mask = (
                torch.arange(max_num_atoms, device=self.device)[None, :]
                >= sampled_num_atoms[:, None]
            )
            atom_types_shape = (batch_size, max_num_atoms, self.data_stats["atom_dim"])
            pos_shape = (batch_size, max_num_atoms, self.data_stats["spatial_dim"])
            frac_coords_shape = (batch_size, max_num_atoms, self.data_stats["spatial_dim"])
            lengths_scaled_shape = (batch_size, 1, 3)
            angles_radians_shape = (batch_size, 1, 3)
        else:
            pad_mask = batch["padding_mask"]
            atom_types_shape = batch["atom_types"].shape
            pos_shape = batch["pos"].shape
            frac_coords_shape = batch["frac_coords"].shape
            lengths_scaled_shape = batch["lengths_scaled"].shape
            angles_radians_shape = batch["angles_radians"].shape

        atom_types_noise = self.atom_types_interpolant.sample_noise(atom_types_shape, pad_mask)
        pos_noise = self.pos_interpolant.sample_noise(pos_shape, pad_mask)
        frac_coords_noise = self.frac_coords_interpolant.sample_noise(frac_coords_shape, pad_mask)
        lengths_scaled_noise = self.lengths_scaled_interpolant.sample_noise(
            lengths_scaled_shape, pad_mask
        )
        angles_radians_noise = self.angles_radians_interpolant.sample_noise(
            angles_radians_shape, pad_mask
        )

        noise_batch = TensorDict(
            {
                "atom_types": atom_types_noise,
                "pos": pos_noise,
                "frac_coords": frac_coords_noise,
                "lengths_scaled": lengths_scaled_noise,
                "angles_radians": angles_radians_noise,
                "padding_mask": pad_mask,
            },
            batch_size=pad_mask.shape[0],
            device=self.device,
        )

        return noise_batch

    @typecheck
    def _create_path(
        self,
        x_1: TensorDict,
        t: Optional[torch.Tensor] = None,
        noise_batch: Optional[TensorDict] = None,
    ) -> FlowPath:
        """Generate `(x_0, x_t, dx_t)` tensors for a random or given time `t`.

        Args:
            x_1: A TensorDict containing the clean data at time t=1.
            t: Optional tensor of time points. If None, sampled from the time distribution.
            noise_batch: Optional TensorDict of noise samples. If None, sampled internally.

        Returns:
            A FlowPath object containing the generated paths.
        """
        batch_size = x_1["padding_mask"].shape[0]
        pad_mask = x_1["padding_mask"]

        if t is None:
            t = self.time_distribution.sample((batch_size,)).to(x_1.device)

        if noise_batch is None:
            noise_batch = self._sample_noise_like_batch(x_1)

        x_0_atom_types, x_t_atom_types, dx_t_atom_types = self.atom_types_interpolant.create_path(
            x_1=x_1, t=t, x_0=noise_batch
        )
        x_0_pos, x_t_pos, dx_t_pos = self.pos_interpolant.create_path(
            x_1=x_1, t=t, x_0=noise_batch
        )
        x_0_frac_coords, x_t_frac_coords, dx_t_frac_coords = (
            self.frac_coords_interpolant.create_path(x_1=x_1, t=t, x_0=noise_batch)
        )
        x_0_lengths_scaled, x_t_lengths_scaled, dx_t_lengths_scaled = (
            self.lengths_scaled_interpolant.create_path(x_1=x_1, t=t, x_0=noise_batch)
        )
        x_0_angles_radians, x_t_angles_radians, dx_t_angles_radians = (
            self.angles_radians_interpolant.create_path(x_1=x_1, t=t, x_0=noise_batch)
        )

        x_0 = TensorDict(
            {
                "atom_types": x_0_atom_types,
                "pos": x_0_pos,
                "frac_coords": x_0_frac_coords,
                "lengths_scaled": x_0_lengths_scaled,
                "angles_radians": x_0_angles_radians,
                "padding_mask": pad_mask,
            },
            batch_size=batch_size,
            device=x_1.device,
        )

        x_t = TensorDict(
            {
                "atom_types": x_t_atom_types,
                "pos": x_t_pos,
                "frac_coords": x_t_frac_coords,
                "lengths_scaled": x_t_lengths_scaled,
                "angles_radians": x_t_angles_radians,
                "padding_mask": pad_mask,
            },
            batch_size=batch_size,
            device=x_1.device,
        )

        dx_t = TensorDict(
            {
                "atom_types": dx_t_atom_types,
                "pos": dx_t_pos,
                "frac_coords": dx_t_frac_coords,
                "lengths_scaled": dx_t_lengths_scaled,
                "angles_radians": dx_t_angles_radians,
                "padding_mask": pad_mask,
            },
            batch_size=batch_size,
            device=x_1.device,
        )

        return FlowPath(x_0=x_0, x_t=x_t, dx_t=dx_t, x_1=x_1, t=t)

    @typecheck
    def _call_model(self, batch: TensorDict, t: TensorDict) -> TensorDict:
        """Wrapper around `self.model` for `torch.compile` compatibility."""
        (atom_types, pos, frac_coords, lengths_scaled, angles_radians), aux_task_preds = (
            self.model(
                atom_types=batch["atom_types"],
                pos=batch["pos"],
                frac_coords=batch["frac_coords"],
                lengths_scaled=batch["lengths_scaled"],
                angles_radians=batch["angles_radians"],
                padding_mask=batch["padding_mask"],
                atom_types_t=t["atom_types"],
                pos_t=t["pos"],
                frac_coords_t=t["frac_coords"],
                lengths_scaled_t=t["lengths_scaled"],
                angles_radians_t=t["angles_radians"],
            )
        )
        assert len(aux_task_preds) == len(
            self.auxiliary_tasks
        ), f"Expected {len(self.auxiliary_tasks)} auxiliary task predictions, but got {len(aux_task_preds)}."

        return TensorDict(
            {
                "atom_types": atom_types,
                "pos": pos,
                "frac_coords": frac_coords,
                "lengths_scaled": lengths_scaled,
                "angles_radians": angles_radians,
                "padding_mask": batch["padding_mask"],
                **{aux_task: aux_task_preds[i] for i, aux_task in enumerate(self.auxiliary_tasks)},
            },
            batch_size=batch["padding_mask"].shape[0],
        )

    @typecheck
    def _compute_loss(
        self, path: FlowPath, pred: TensorDict, compute_stats: bool = True
    ) -> Tensor:
        """Compute and sum each modality's loss as well as an optional inter-distance loss for
        (non-periodic) 3D atom positions.

        Args:
            path: The FlowPath object containing the true data and noise.
            pred: The predicted TensorDict from the model.
            compute_stats: Whether to compute and return statistics.

        Returns:
            A tuple of (loss_dict, stats_dict) where loss_dict is a dictionary
            of scalar tensors and stats_dict contains detailed loss components
            and statistics.
        """
        atom_types_loss, atom_types_stats = self.atom_types_interpolant.compute_loss(
            path, pred, compute_stats
        )
        pos_loss, pos_stats = self.pos_interpolant.compute_loss(path, pred, compute_stats)
        frac_coords_loss, frac_coords_stats = self.frac_coords_interpolant.compute_loss(
            path, pred, compute_stats
        )
        lengths_scaled_loss, lengths_scaled_stats = self.lengths_scaled_interpolant.compute_loss(
            path, pred, compute_stats
        )
        angles_radians_loss, angles_radians_stats = self.angles_radians_interpolant.compute_loss(
            path, pred, compute_stats
        )
        if self.interdist_loss:
            dists_loss, dists_stats = self.interdist_loss(path, pred, compute_stats)
        else:
            dists_loss, dists_stats = 0, {}

        if compute_stats:
            stats_dict = {
                "atom_types_loss": atom_types_loss,
                "pos_loss": pos_loss,
                "frac_coords_loss": frac_coords_loss,
                "lengths_scaled_loss": lengths_scaled_loss,
                "angles_radians_loss": angles_radians_loss,
                **atom_types_stats,
                **pos_stats,
                **frac_coords_stats,
                **lengths_scaled_stats,
                **angles_radians_stats,
                **dists_stats,
            }

            atom_types_logit_norm = pred["atom_types"].norm(dim=-1)
            atom_types_logit_max, _ = pred["atom_types"].max(dim=-1)
            atom_types_logit_min, _ = pred["atom_types"].min(dim=-1)
            stats_dict["atom_types_logit_norm"] = atom_types_logit_norm.mean().item()
            stats_dict["atom_types_logit_max"] = atom_types_logit_max.mean().item()
            stats_dict["atom_types_logit_min"] = atom_types_logit_min.mean().item()

            pos_logit_norm = pred["pos"].norm(dim=-1).mean().item()
            frac_coords_logit_norm = pred["frac_coords"].norm(dim=-1).mean().item()
            lengths_scaled_logit_norm = pred["lengths_scaled"].norm(dim=-1).mean().item()
            angles_radians_logit_norm = pred["angles_radians"].norm(dim=-1).mean().item()
            stats_dict["pos_logit_norm"] = pos_logit_norm
            stats_dict["frac_coords_logit_norm"] = frac_coords_logit_norm
            stats_dict["lengths_scaled_logit_norm"] = lengths_scaled_logit_norm
            stats_dict["angles_radians_logit_norm"] = angles_radians_logit_norm
        else:
            stats_dict = {}

        total_loss = (
            atom_types_loss
            + pos_loss
            + frac_coords_loss
            + lengths_scaled_loss
            + angles_radians_loss
            + dists_loss
        )
        loss_dict = {
            "loss": total_loss,
            "atom_types_loss": atom_types_loss,
            "pos_loss": pos_loss,
            "frac_coords_loss": frac_coords_loss,
            "lengths_scaled_loss": lengths_scaled_loss,
            "angles_radians_loss": angles_radians_loss,
        }

        if self.interdist_loss:
            loss_dict["interdist_loss"] = dists_loss

        return loss_dict, stats_dict

    @typecheck
    def _step(self, x_t: TensorDict, t: TensorDict, step_size: float) -> TensorDict:
        """Single Euler step at time `t` using model-predicted velocity."""
        with torch.no_grad():
            out_batch = self._call_model(x_t, t)

        x_t["atom_types"] = self.atom_types_interpolant.step(x_t, out_batch, t, step_size)
        x_t["pos"] = self.pos_interpolant.step(x_t, out_batch, t, step_size)
        x_t["frac_coords"] = self.frac_coords_interpolant.step(x_t, out_batch, t, step_size)
        x_t["lengths_scaled"] = self.lengths_scaled_interpolant.step(x_t, out_batch, t, step_size)
        x_t["angles_radians"] = self.angles_radians_interpolant.step(x_t, out_batch, t, step_size)

        for aux_task in self.auxiliary_tasks:
            x_t[aux_task] = out_batch[aux_task]

        return x_t

    @typecheck
    def forward(
        self, batch: TensorDict, compute_stats: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, float]]]:
        """Compute training loss dictionary and optional statistics.

        Args:
            batch: A TensorDict containing the input batch data.
            compute_stats: Whether to compute and return statistics along with the loss.

        Returns:
            A tuple containing the loss dictionary and an optional dictionary of statistics.
        """
        path = self._create_path(batch)
        pred = self._call_model(path.x_t, path.t)

        loss_dict, stats_dict = self._compute_loss(path, pred, compute_stats)
        return loss_dict, stats_dict

    @typecheck
    def sample(
        self,
        feats: Dict[str, torch.Tensor],
        mask: Bool["b m"],  # type: ignore
        steps: int = 100,
        cfg_scale: float = 2.0,
        enable_zero_centering: bool = True,
        sdpa_backends: List[SDPBackend] = SDPA_BACKENDS,  # type: ignore
        modal_input_dict: (
            Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] | None
        ) = None,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], None]:
        """ODE-driven sampling with MFT using classifier-free guidance.

        Args:
            feats: Features for conditioning including:
                dataset_idx: Dataset index for each sample.
                spacegroup: Spacegroup index for each sample.
                ref_pos: Reference atom positions tensor.
                ref_space_uids: Reference space unique IDs tensor.
                atom_to_token: One-hot mapping from atom indices to token indices.
                atom_to_token_idx: Mapping from atom indices to token indices.
                max_num_tokens: Maximum number of unmasked tokens for each batch element.
                token_index: Indices of the tokens in the batch.
            mask: True if valid token, False if padding.
            steps: Number of integration steps for the multimodal ODE solver.
            cfg_scale: Classifier-free guidance scale.
            enable_zero_centering (bool): Whether to allow centering of continuous modalities
                at the origin after each denoising step. Defaults to ``True``.
            sdpa_backends: List of SDPBackend backends to try when using fused attention. Defaults to all.
            modal_input_dict: If not None, a dictionary specifying input modalities to use and their input metadata.
                The keys should be a subset of `["atom_types", "pos", "frac_coords", "lengths_scaled", "angles_radians"]`,
                and the values should be tuples of (x_t, t).
            kwargs: Additional keyword arguments (not used).

        Returns:
            A list of predicted modalities as a dictionary and a null variable (for sake of API compatibility).
        """
        device = mask.device
        batch_size, num_tokens = mask.shape

        if modal_input_dict is None:
            # Define time points and corresponding noised inputs for each modality
            modal_input_dict = {}
            for modal in self.modals:
                assert modal in kwargs, f"Missing required modality input: {modal}"
                t = torch.zeros(batch_size, device=BEST_DEVICE)

                if modal == "atom_types" and self.treat_discrete_modalities_as_continuous:
                    # Maybe convert atom types to random continuous (one-hot) vectors
                    kwargs[modal] = torch.randn(
                        batch_size, num_tokens, self.vocab_size, device=device
                    )

                # Set up modality inputs for classifier-free guidance
                kwargs[modal] = torch.cat([kwargs[modal], kwargs[modal]], dim=0)  # [2B, N, C]
                t = torch.cat([t, t], dim=0)  # [2B]

                modal_input_dict[modal] = (kwargs[modal], t)

        # Set up conditioning inputs for classifier-free guidance
        mask = torch.cat([mask, mask], dim=0)  # [2B, M]
        for feat in feats:
            if feat in ("dataset_idx", "spacegroup"):
                feats[feat] = torch.cat(
                    [feats[feat], torch.zeros_like(feats[feat])], dim=0
                )  # [2B]
            else:
                feats[feat] = torch.cat([feats[feat], feats[feat]], dim=0)  # [2B, M, ...]

        # Predict each modality in one step
        pred_modals = self.flow.sample(
            x_init=[
                modal_input_dict["atom_types"][0],
                modal_input_dict["pos"][0],
                modal_input_dict["frac_coords"][0],
                modal_input_dict["lengths_scaled"][0],
                modal_input_dict["angles_radians"][0],
            ],
            time_grid=None,  # For now, use same time point for all modalities
            device=mask.device,
            steps=steps,
            feats=feats,
            mask=mask,
            cfg_scale=cfg_scale,
            enable_zero_centering=enable_zero_centering,
            sdpa_backends=sdpa_backends,
        )

        # Prepare denoised modalities and remove null class samples
        denoised_modals_list = [
            {
                modal: (
                    # Take the first half of the batch
                    (
                        pred_modals[modal_idx].argmax(dim=-1)
                        if self.treat_discrete_modalities_as_continuous
                        else pred_modals[modal_idx]
                    )
                    .detach()
                    .chunk(2, dim=0)[0]
                    .reshape(batch_size * num_tokens)
                    if modal == "atom_types"
                    else pred_modals[modal_idx].detach().chunk(2, dim=0)[0]
                )
                for modal_idx, modal in enumerate(self.modals)
            }
        ]

        return denoised_modals_list, None
