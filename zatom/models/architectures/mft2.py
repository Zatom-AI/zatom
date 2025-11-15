"""Multimodal flow transformer (MFT).

Adapted from:
    - https://github.com/apple/ml-simplefold
    - https://github.com/carlosinator/tabasco
    - https://github.com/facebookresearch/all-atom-diffusion-transformer
"""

from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn.attention import SDPBackend

from zatom.flow.interpolants import Interpolant
from zatom.flow.path import FlowPath
from zatom.models.architectures.tabasco.losses import InterDistancesLoss
from zatom.utils import pylogger
from zatom.utils.sampling_utils import get_sample_schedule
from zatom.utils.training_utils import (
    SDPA_BACKENDS,
    HistogramTimeDistribution,
)
from zatom.utils.typing_utils import typecheck

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

#################################################################################
#                           Multimodal Flow Transformer                         #
#################################################################################


class MFT(nn.Module):
    """Multimodal flow model with an encoder-decoder architecture.

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
        condition_on_input: Whether to condition the model on the input as well as context.
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
        condition_on_input: bool = False,
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
        self.treat_discrete_modalities_as_continuous = False

        # Define time distribution
        if time_distribution == "uniform":
            self.time_distribution = torch.distributions.Uniform(0, 1)
        elif time_distribution == "beta":
            self.time_distribution = torch.distributions.Beta(time_alpha_factor, 1)
        elif time_distribution == "histogram":
            log.info("Using histogram time distribution.")
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
            treat_discrete_modalities_as_continuous=self.treat_discrete_modalities_as_continuous,
            remove_t_conditioning=False,
            condition_on_input=condition_on_input,
            jvp_attn=jvp_attn,
            **kwargs,
        )

        # Define modalities and auxiliary tasks
        self.modals = [
            "atom_types",
            "pos",
            "frac_coords",
            "lengths_scaled",
            "angles_radians",
        ]
        self.auxiliary_tasks = self.model.auxiliary_tasks

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device

    @typecheck
    def _sample_noise_like_batch(self, batch: TensorDict):
        """Draw coordinate and atom-type noise compatible with `batch`.

        Args:
            batch: TensorDict containing batch data.

        Returns:
            A TensorDict containing noise samples.
        """
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
                "lengths_scaled": lengths_scaled_noise[:, 0:1, :],
                "angles_radians": angles_radians_noise[:, 0:1, :],
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
        t: torch.Tensor | None = None,
        noise_batch: TensorDict | None = None,
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
                "lengths_scaled": x_0_lengths_scaled[:, 0:1, :],
                "angles_radians": x_0_angles_radians[:, 0:1, :],
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
                "lengths_scaled": x_t_lengths_scaled[:, 0:1, :],
                "angles_radians": x_t_angles_radians[:, 0:1, :],
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
                "lengths_scaled": dx_t_lengths_scaled[:, 0:1, :],
                "angles_radians": dx_t_angles_radians[:, 0:1, :],
                "padding_mask": pad_mask,
            },
            batch_size=batch_size,
            device=x_1.device,
        )

        t = TensorDict(
            {
                "atom_types": t,
                "pos": t,
                "frac_coords": t,
                "lengths_scaled": t,
                "angles_radians": t,
            },
            batch_size=batch_size,
            device=x_1.device,
        )

        return FlowPath(x_0=x_0, x_t=x_t, dx_t=dx_t, x_1=x_1, t=t)

    @typecheck
    def _call_model(
        self,
        x_t: TensorDict,
        x_1: TensorDict,
        t: TensorDict,
        use_cfg: bool = False,
        cfg_scale: float = 2.0,
        sdpa_backends: List[SDPBackend] = SDPA_BACKENDS,  # type: ignore
    ) -> TensorDict:
        """Wrapper around `self.model` for `torch.compile` compatibility.

        Args:
            x_t: A TensorDict containing the noised input data at time t.
            x_1: A TensorDict containing the model's input features.
            t: A TensorDict containing the time points.
            use_cfg: Whether to use classifier-free guidance.
            cfg_scale: Classifier-free guidance scale.
            sdpa_backends: List of SDPBackend backends to try
                when using fused attention. Defaults to all.

        Returns:
            A TensorDict containing the model predictions.
        """
        model_fn = (
            partial(self.model.forward_with_cfg, cfg_scale=cfg_scale)
            if use_cfg
            else self.model.forward
        )
        (
            atom_types,
            pos,
            frac_coords,
            lengths_scaled,
            angles_radians,
        ), aux_task_preds = model_fn(
            x=(
                x_t["atom_types"].argmax(dim=-1),
                x_t["pos"],
                x_t["frac_coords"],
                x_t["lengths_scaled"],
                x_t["angles_radians"],
            ),
            t=(
                t["atom_types"],
                t["pos"],
                t["frac_coords"],
                t["lengths_scaled"],
                t["angles_radians"],
            ),
            feats={
                "dataset_idx": x_1["dataset_idx"],
                "spacegroup": x_1["spacegroup"],
                "ref_pos": x_1["ref_pos"],
                "ref_space_uid": x_1["ref_space_uid"],
                "atom_to_token": x_1["atom_to_token"],
                "atom_to_token_idx": x_1["atom_to_token_idx"],
                "max_num_tokens": x_1["max_num_tokens"],
                "token_index": x_1["token_index"],
                "token_is_periodic": x_1["token_is_periodic"],
            },
            mask=~x_t["padding_mask"],
            sdpa_backends=sdpa_backends,
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
                "padding_mask": x_t["padding_mask"],
                **{aux_task: aux_task_preds[i] for i, aux_task in enumerate(self.auxiliary_tasks)},
            },
            batch_size=x_t["padding_mask"].shape[0],
            device=x_t.device,
        )

    @typecheck
    def _compute_loss(
        self,
        path: FlowPath,
        pred: TensorDict,
        compute_stats: bool = True,
        eps: float = 1e-6,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Compute and sum each modality's loss as well as an optional inter-distance loss for
        (non-periodic) 3D atom positions.

        Atom types are supervised for all examples, while 3D atom positions are only supervised
        for non-periodic examples (i.e., molecules). Conversely, fractional coordinates, scaled
        lengths, and angles in radians are only supervised for periodic examples (i.e., materials).

        If distance loss is enabled, it is only applied to non-periodic examples.

        Args:
            path: The FlowPath object containing the true data and noise.
            pred: The predicted TensorDict from the model.
            compute_stats: Whether to compute and return statistics.
            eps: A small value to avoid division by zero.

        Returns:
            A tuple of (loss_dict, stats_dict) where loss_dict is a dictionary
            of scalar tensors and stats_dict contains detailed loss components
            and statistics.
        """
        cont_aux_mask = path.x_1["token_is_periodic"]

        atom_types_loss, atom_types_stats = self.atom_types_interpolant.compute_loss(
            path, pred, compute_stats
        )
        pos_loss, pos_stats = self.pos_interpolant.compute_loss(
            path, pred, compute_stats, aux_mask=~cont_aux_mask
        )
        frac_coords_loss, frac_coords_stats = self.frac_coords_interpolant.compute_loss(
            path, pred, compute_stats, aux_mask=cont_aux_mask
        )
        lengths_scaled_loss, lengths_scaled_stats = self.lengths_scaled_interpolant.compute_loss(
            path,
            pred,
            compute_stats,
            aux_mask=cont_aux_mask.any(-1, keepdim=True),
            pool_mask=True,
        )
        angles_radians_loss, angles_radians_stats = self.angles_radians_interpolant.compute_loss(
            path,
            pred,
            compute_stats,
            aux_mask=cont_aux_mask.any(-1, keepdim=True),
            pool_mask=True,
        )
        if self.interdist_loss:
            dists_loss, dists_stats = self.interdist_loss(
                path, pred, compute_stats, aux_mask=~cont_aux_mask.unsqueeze(-1)
            )
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
            loss_dict["dists_loss"] = dists_loss

        # Add auxiliary losses
        for aux_task in self.auxiliary_tasks:
            aux_pred = pred[aux_task].squeeze(-1)
            # Requested auxiliary task → compute loss
            if aux_task in path.x_1:
                real_mask = 1 - path.x_1["padding_mask"].int()
                aux_target = path.x_1[aux_task]
                aux_mask = ~aux_target.isnan()
                if aux_target.squeeze().dim() == 1:
                    err = (aux_pred - aux_target) * aux_mask
                    aux_loss_value = torch.sum(err.abs()) / (aux_mask.sum() + eps)
                else:
                    aux_mask = aux_mask * real_mask.unsqueeze(-1)
                    n_tokens = aux_mask.all(-1).sum(dim=-1)
                    err = (aux_pred - aux_target) * aux_mask
                    aux_loss = torch.sum(err.abs(), dim=(-1, -2)) / (
                        n_tokens * err.shape[-1] + eps
                    )
                    aux_loss_value = aux_loss.sum() / (real_mask.any(-1).sum() + eps)
                loss_dict[f"aux_{aux_task}_loss"] = aux_loss_value
            # Unused auxiliary task → add zero loss to computational graph
            else:
                loss_dict[f"aux_{aux_task}_loss"] = (aux_pred * 0.0).mean()

            # Log per-atom auxiliary loss
            if aux_task == "global_energy":
                if aux_task in path.x_1:
                    real_mask = 1 - path.x_1["padding_mask"].int()
                    aux_target = path.x_1[aux_task]
                    aux_mask = ~aux_target.isnan()
                    n_tokens = real_mask.sum(dim=-1).clamp(min=1.0)
                    err = (
                        (aux_pred.squeeze(-1) / n_tokens) - (aux_target.squeeze(-1) / n_tokens)
                    ) * aux_mask.squeeze(-1)
                    per_atom_aux_loss_value = torch.sum(err.abs()) / (aux_mask.sum() + eps)
                else:
                    per_atom_aux_loss_value = (aux_pred * 0.0).mean()

                loss_dict[f"aux_{aux_task}_per_atom_loss"] = per_atom_aux_loss_value

            loss_dict["loss"] += loss_dict[f"aux_{aux_task}_loss"]

        return loss_dict, stats_dict

    @typecheck
    def _step(
        self,
        x_t: TensorDict,
        x_1: TensorDict,
        t: TensorDict,
        step_size: TensorDict,
        cfg_scale: float = 2.0,
        sdpa_backends: List[SDPBackend] = SDPA_BACKENDS,  # type: ignore
    ) -> TensorDict:
        """Single Euler step at time `t` using model-predicted velocity.

        Args:
            x_t: A TensorDict containing the noised input data at time t.
            x_1: A TensorDict containing the model's input features.
            t: A TensorDict containing the time points.
            step_size: A TensorDict containing the step sizes for each modality.
            cfg_scale: Classifier-free guidance scale.
            sdpa_backends: List of SDPBackend backends to try when using fused attention. Defaults to all.

        Returns:
            A TensorDict containing the updated data after the Euler step.
        """
        with torch.no_grad():
            out_batch = self._call_model(
                x_t,
                x_1,
                t,
                use_cfg=True,
                cfg_scale=cfg_scale,
                sdpa_backends=sdpa_backends,
            )

        x_t["atom_types"] = self.atom_types_interpolant.step(x_t, out_batch, t, step_size)
        x_t["pos"] = self.pos_interpolant.step(x_t, out_batch, t, step_size)
        x_t["frac_coords"] = self.frac_coords_interpolant.step(x_t, out_batch, t, step_size)
        x_t["lengths_scaled"] = self.lengths_scaled_interpolant.step(x_t, out_batch, t, step_size)[
            :, 0:1, :
        ]
        x_t["angles_radians"] = self.angles_radians_interpolant.step(x_t, out_batch, t, step_size)[
            :, 0:1, :
        ]

        for aux_task in self.auxiliary_tasks:
            x_t[aux_task] = out_batch[aux_task]

        return x_t

    @typecheck
    def forward(
        self, batch: TensorDict, compute_stats: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Compute training loss dictionary and optional statistics.

        Args:
            batch: A TensorDict containing the input batch data, including:
                - atom_types: Atom types tensor at *t=1*.
                - pos: Atom positions tensor at *t=1*.
                - frac_coords: Fractional coordinates tensor at *t=1*.
                - lengths_scaled: Scaled lengths tensor at *t=1*.
                - angles_radians: Angles in radians tensor at *t=1*.
                - dataset_idx: Dataset index tensor.
                - spacegroup: Spacegroup index tensor.
                - ref_pos: Reference atom positions tensor.
                - ref_space_uid: Reference space unique IDs tensor.
                - atom_to_token: One-hot mapping from atom indices to token indices.
                - atom_to_token_idx: Mapping from atom indices to token indices.
                - token_index: Indices of the tokens in the batch.
                - max_num_tokens: Maximum number of unmasked tokens for each batch element.
                - padding_mask: Padding mask tensor.
                - token_is_periodic: Periodicity mask tensor.
            compute_stats: Whether to compute and return statistics along with the loss.

        Returns:
            A tuple containing the loss dictionary and a dictionary of statistics.
        """
        path = self._create_path(batch)
        pred = self._call_model(path.x_t, path.x_1, path.t)

        loss_dict, stats_dict = self._compute_loss(path, pred, compute_stats)
        return loss_dict, stats_dict

    @typecheck
    def sample(
        self,
        batch: TensorDict,
        steps: int = 100,
        cfg_scale: float = 2.0,
        return_trajectories: bool = False,
        sdpa_backends: List[SDPBackend] = SDPA_BACKENDS,  # type: ignore
    ) -> Tuple[TensorDict, List[TensorDict]]:
        """ODE-driven sampling with MFT using classifier-free guidance.

        Args:
            batch: A TensorDict containing the input batch data, including:
                - atom_types: Placeholder atom types tensor.
                - pos: Placeholder atom positions tensor.
                - frac_coords: Placeholder fractional coordinates tensor.
                - lengths_scaled: Placeholder scaled lengths tensor.
                - angles_radians: Placeholder angles in radians tensor.
                - dataset_idx: Dataset index tensor.
                - spacegroup: Spacegroup index tensor.
                - ref_pos: Reference atom positions tensor.
                - ref_space_uid: Reference space unique IDs tensor.
                - atom_to_token: One-hot mapping from atom indices to token indices.
                - atom_to_token_idx: Mapping from atom indices to token indices.
                - token_index: Indices of the tokens in the batch.
                - max_num_tokens: Maximum number of unmasked tokens for each batch element.
                - padding_mask: Padding mask tensor.
                - token_is_periodic: Periodicity mask tensor.
            steps: Number of integration steps for the multimodal ODE solver.
            cfg_scale: Classifier-free guidance scale.
            return_trajectories: Whether to return full sampling trajectories.
            sdpa_backends: List of SDPBackend backends to try when using fused attention. Defaults to all.

        Returns:
            A tuple containing the final sampled TensorDict and a list of intermediate trajectories (if requested).
        """
        trajectories = []
        batch = batch.repeat_interleave(2, dim=0)  # For CFG: duplicate batch

        x_t = self._sample_noise_like_batch(batch)
        x_1 = TensorDict(
            {
                "dataset_idx": batch["dataset_idx"],
                "spacegroup": batch["spacegroup"],
                "ref_pos": batch["ref_pos"],
                "ref_space_uid": batch["ref_space_uid"],
                "atom_to_token": batch["atom_to_token"],
                "atom_to_token_idx": batch["atom_to_token_idx"],
                "max_num_tokens": batch["max_num_tokens"],
                "token_index": batch["token_index"],
                "token_is_periodic": batch["token_is_periodic"],
            },
            batch_size=batch.batch_size,
            device=batch.device,
        )

        T = TensorDict(
            {
                modal: get_sample_schedule(
                    schedule=getattr(self, f"{modal}_interpolant").sample_schedule,
                    num_steps=steps,
                )
                .unsqueeze(0)
                .repeat(batch.batch_size[0], 1)
                for modal in self.modals
            },
            batch_size=batch.batch_size,
            device=batch.device,
        )

        for i in range(1, steps + 1):
            t = TensorDict(
                {modal: T[modal][:, i - 1] for modal in self.modals},
                batch_size=batch.batch_size,
                device=batch.device,
            )
            dt = TensorDict(
                {modal: T[modal][:, i] - T[modal][:, i - 1] for modal in self.modals},
                batch_size=batch.batch_size,
                device=batch.device,
            )

            x_t = self._step(x_t, x_1, t, dt, cfg_scale=cfg_scale, sdpa_backends=sdpa_backends)
            if return_trajectories:
                trajectories.append(deepcopy(x_t.chunk(2, dim=0)[0].detach().cpu()))

        x_t = x_t.chunk(2, dim=0)[0]  # Return only the guided samples
        return x_t, trajectories
