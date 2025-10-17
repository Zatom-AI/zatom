"""Multimodal flow transformer (MFT).

Adapted from:
    - https://github.com/apple/ml-simplefold
    - https://github.com/facebookresearch/flow_matching
    - https://github.com/facebookresearch/all-atom-diffusion-transformer
"""

from typing import Any, Dict, List, Literal, Tuple

import torch
import torch.nn as nn
from flow_matching.path import AffineProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import (
    CondOTScheduler,
    PolynomialConvexScheduler,
)
from flow_matching.utils import expand_tensor_like
from torch.nn.attention import SDPBackend

from zatom.models.architectures.modules.dit import MultimodalDiT
from zatom.scheduler.scheduler import EquilibriumCondOTScheduler
from zatom.utils import pylogger
from zatom.utils.multimodal import Flow
from zatom.utils.training_utils import (
    BEST_DEVICE,
    SDPA_BACKENDS,
    sample_logit_normal,
    weighted_rigid_align,
)
from zatom.utils.typing_utils import Bool, Float, Int, typecheck

log = pylogger.RankedLogger(__name__)

GRAD_DECAY_METHODS = Literal["none", "linear_decay", "truncated_decay", "piecewise_decay"]


#################################################################################
#                           Multimodal Flow Transformer                         #
#################################################################################


class MFT(nn.Module):
    """Multimodal flow model with an encoder-decoder architecture.

    NOTE: The `_forward` pass of this model is conceptually similar to that of All-atom Diffusion Transformers (ADiTs)
    except that there is no self conditioning and the model learns flows for each modality.

    Args:
        time_embedder: Time embedder module.
        dataset_embedder: Dataset embedder module.
        spacegroup_embedder: Spacegroup embedder module.
        token_pos_embedder: Token positional embedder module.
        atom_pos_embedder: Atom positional embedder module.
        atom_encoder_transformer: Multimodal atom encoder Transformer module.
        trunk: The Transformer trunk module.
        atom_decoder_transformer: Multimodal atom decoder Transformer module.
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
        atom_types_loss_weight: Weighting factor for the atom types loss.
        pos_loss_weight: Weighting factor for the atom positions loss.
        frac_coords_loss_weight: Weighting factor for the atom fractional coordinates loss.
        lengths_scaled_loss_weight: Weighting factor for the atom lengths (scaled) loss.
        angles_radians_loss_weight: Weighting factor for the atom angles (radians) loss.
        grad_decay_a: Parameter `a` for gradient decay functions.
        grad_decay_b: Parameter `b` for gradient decay functions.
        grad_mul: Global multiplier for the target gradient magnitude.
        early_stopping_grad_norm (Optional[float], optional): If specified,
            sampling will stop early if the model output velocity (or gradient)
            norm with respect to each modality falls below this value. This
            effectively enables adaptive compute for sampling. Defaults to
            ``None``.
        use_length_condition: Whether to use the length condition.
        jvp_attn: Whether to use JVP Flash Attention instead of PyTorch's Scaled Dot Product Attention.
        weighted_rigid_align_pos: Whether to apply weighted rigid alignment between target and predicted atom positions for loss calculation.
        weighted_rigid_align_frac_coords: Whether to apply weighted rigid alignment between target and predicted atom fractional coordinates for loss calculation.
        continuous_x_1_prediction: Whether the model predicts clean data at t=1 for continuous modalities.
        unified_modal_time: Whether to use a single (i.e., the same) time input for all modalities.
        remove_t_conditioning: Whether to remove timestep conditioning for each modality.
        enable_eqm_mode: Whether to enable Equilibrium Matching (EqM) mode.
        add_mask_atom_type: Whether to add a mask token for atom types.
        grad_decay_method: Method for decaying the target gradient magnitude as a function of time.
            Must be one of (`none`, `linear_decay`, `truncated_decay`, `piecewise_decay`).
    """

    def __init__(
        self,
        time_embedder: nn.Module,
        dataset_embedder: nn.Module,
        spacegroup_embedder: nn.Module,
        token_pos_embedder: nn.Module,
        atom_pos_embedder: nn.Module,
        atom_encoder_transformer: nn.Module,
        trunk: nn.Module,
        atom_decoder_transformer: nn.Module,
        hidden_size: int = 1152,
        token_num_heads: int = 16,
        atom_num_heads: int = 4,
        atom_hidden_size_enc: int = 256,
        atom_hidden_size_dec: int = 256,
        atom_n_queries_enc: int = 32,
        atom_n_keys_enc: int = 128,
        atom_n_queries_dec: int = 32,
        atom_n_keys_dec: int = 128,
        max_num_elements: int = 100,
        batch_size_scale_factor: int = 1,
        atom_types_loss_weight: float = 1.0,
        pos_loss_weight: float = 10.0,
        frac_coords_loss_weight: float = 10.0,
        lengths_scaled_loss_weight: float = 1.0,
        angles_radians_loss_weight: float = 10.0,
        grad_decay_a: float = 0.8,
        grad_decay_b: float = 0.8,
        grad_mul: float = 1.0,
        early_stopping_grad_norm: float | None = None,
        use_length_condition: bool = True,
        jvp_attn: bool = False,
        weighted_rigid_align_pos: bool = True,
        weighted_rigid_align_frac_coords: bool = False,
        continuous_x_1_prediction: bool = False,
        unified_modal_time: bool = True,
        remove_t_conditioning: bool = False,
        enable_eqm_mode: bool = False,
        add_mask_atom_type: bool = True,
        grad_decay_method: GRAD_DECAY_METHODS = "none",
    ):
        super().__init__()

        if enable_eqm_mode:
            assert (
                not continuous_x_1_prediction
            ), "EqM mode requires continuous velocity prediction (i.e., continuous_x_1_prediction=False)."
            assert (
                unified_modal_time
            ), "EqM mode requires unified_modal_time to be True (i.e., a single time input for all modalities)."
            assert (
                remove_t_conditioning
            ), "EqM mode requires timestep conditioning to be disabled (i.e., remove_t_conditioning=True)."
            assert (
                grad_decay_method != "none"
            ), "EqM mode requires a gradient decay method (i.e., grad_decay_method != 'none')."

        self.batch_size_scale_factor = batch_size_scale_factor
        self.class_dropout_prob = dataset_embedder.dropout_prob
        self.atom_types_loss_weight = atom_types_loss_weight
        self.pos_loss_weight = pos_loss_weight
        self.frac_coords_loss_weight = frac_coords_loss_weight
        self.lengths_scaled_loss_weight = lengths_scaled_loss_weight
        self.angles_radians_loss_weight = angles_radians_loss_weight
        self.grad_mul = grad_mul
        self.jvp_attn = jvp_attn
        self.unified_modal_time = unified_modal_time
        self.grad_decay_method = grad_decay_method

        self.vocab_size = max_num_elements + int(add_mask_atom_type)

        # Build multimodal model
        model = MultimodalDiT(
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
            add_mask_atom_type=add_mask_atom_type,
            remove_t_conditioning=remove_t_conditioning,
            jvp_attn=jvp_attn,
        )

        # Instantiate paths and losses for Flow
        cond_ot_scheduler = EquilibriumCondOTScheduler if enable_eqm_mode else CondOTScheduler
        modalities = {
            "atom_types": {
                "path": MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0)),
                # loss omitted → Flow will use MixturePathGeneralizedKL automatically
                "weight": self.atom_types_loss_weight,
            },
            "pos": {
                "path": AffineProbPath(scheduler=cond_ot_scheduler()),
                # loss omitted → Flow will use squared error automatically
                "weight": self.pos_loss_weight,
                "x_1_prediction": continuous_x_1_prediction,
                "should_rigid_align": weighted_rigid_align_pos,
            },
            "frac_coords": {
                "path": AffineProbPath(scheduler=cond_ot_scheduler()),
                # loss omitted → Flow will use squared error automatically
                "weight": self.frac_coords_loss_weight,
                "x_1_prediction": continuous_x_1_prediction,
                "should_rigid_align": weighted_rigid_align_frac_coords,
            },
            "lengths_scaled": {
                "path": AffineProbPath(scheduler=cond_ot_scheduler()),
                # loss omitted → Flow will use squared error automatically
                "weight": self.lengths_scaled_loss_weight,
                "x_1_prediction": continuous_x_1_prediction,
            },
            "angles_radians": {
                "path": AffineProbPath(scheduler=cond_ot_scheduler()),
                # loss omitted → Flow will use squared error automatically
                "weight": self.angles_radians_loss_weight,
                "x_1_prediction": continuous_x_1_prediction,
            },
        }
        self.flow = Flow(
            model=model,
            modalities=modalities,
            model_sampling_fn="forward_with_cfg",
            early_stopping_grad_norm=early_stopping_grad_norm,
        )

        self.modals = list(modalities.keys())

        self.a, self.b = grad_decay_a, grad_decay_b
        self.c = {
            "none": lambda t: torch.ones_like(t),
            "linear_decay": lambda t: 1 - t,
            "truncated_decay": lambda t: torch.where(
                t <= self.a, torch.ones_like(t), (1 - t) / (1 - self.a)
            ),
            "piecewise_decay": lambda t: torch.where(
                t <= self.a, self.b - ((self.b - 1) / self.a) * t, (1 - t) / (1 - self.a)
            ),
        }

    @typecheck
    def sample(
        self,
        feats: Dict[str, torch.Tensor],
        mask: Bool["b m"],  # type: ignore
        steps: int = 100,
        cfg_scale: float = 2.0,
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
            sdpa_backends: List of SDPBackend backends to try when using fused attention. Defaults to all.
            modal_input_dict: If not None, a dictionary specifying input modalities to use and their input metadata.
                The keys should be a subset of `["atom_types", "pos", "frac_coords", "lengths_scaled", "angles_radians"]`,
                and the values should be tuples of (x_t, t).
            kwargs: Additional keyword arguments (not used).

        Returns:
            A list of predicted modalities as a dictionary and a null variable (for sake of API compatibility).
        """
        batch_size, num_tokens = mask.shape

        if modal_input_dict is None:
            # Define time points and corresponding noised inputs for each modality
            modal_input_dict = {}
            for modal in self.modals:
                assert modal in kwargs, f"Missing required modality input: {modal}"
                t = torch.ones(batch_size, device=BEST_DEVICE)

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
            sdpa_backends=sdpa_backends,
        )

        # Prepare denoised modalities and remove null class samples
        denoised_modals_list = [
            {
                modal: (
                    # Take the first half of the batch
                    pred_modals[modal_idx]
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

    @typecheck
    def forward_with_loss_wrapper(
        self,
        atom_types: Int["b m"],  # type: ignore
        pos: Float["b m 3"],  # type: ignore - referenced via `locals()`
        frac_coords: Float["b m 3"],  # type: ignore - referenced via `locals()`
        lengths_scaled: Float["b 1 3"],  # type: ignore - referenced via `locals()`
        angles_radians: Float["b 1 3"],  # type: ignore - referenced via `locals()`
        feats: Dict[str, torch.Tensor],
        mask: Bool["b m"],  # type: ignore
        token_is_periodic: Bool["b m"],  # type: ignore
        target_tensors: Dict[str, torch.Tensor],
        sdpa_backends: List[SDPBackend] = SDPA_BACKENDS,  # type: ignore
        epsilon: float = 1e-3,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of MFT with loss calculation.

        Args:
            atom_types: Atom types tensor.
            pos: Atom positions tensor.
            frac_coords: Fractional coordinates tensor.
            lengths_scaled: Lattice lengths tensor.
            angles_radians: Lattice angles tensor.
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
            token_is_periodic: Boolean mask indicating periodic tokens.
            target_tensors: Dictionary containing the following target tensors for loss calculation:
                atom_types: Target atom types tensor (B, N).
                pos: Target positions tensor (B, N, 3).
                frac_coords: Target fractional coordinates tensor (B, N, 3).
                lengths_scaled: Target lattice lengths tensor (B, 1, 3).
                angles_radians: Target lattice angles tensor (B, 1, 3).
            sdpa_backends: List of SDPBackend backends to try when using fused attention. Defaults to all.
            epsilon: Small constant to avoid numerical issues in loss calculation.
            kwargs: Additional keyword arguments (not used).

        Returns:
            Dictionary of loss values.
        """
        device = atom_types.device
        batch_size, num_tokens = atom_types.shape

        # Sample time points and corresponding noised inputs for each modality
        modal_input_dict = {}

        modal_t = None
        if self.unified_modal_time:
            # Sample a single time point from a logit-normal distribution for all modalities.
            # See https://arxiv.org/abs/2509.18480 for more details.
            modal_t = 0.98 * sample_logit_normal(
                n=batch_size, m=0.8, s=1.7, device=device
            ) + 0.02 * torch.rand(batch_size, device=device)
            modal_t = modal_t * (1 - 2 * epsilon) + epsilon

        for modal in self.modals:
            path = self.flow.paths[modal]

            x_0 = locals()[modal]  # Noised data
            x_1 = target_tensors[modal]  # Clean data

            # Maybe sample a time point from a logit-normal distribution for each modality
            t = modal_t
            if not self.unified_modal_time:
                t = 0.98 * sample_logit_normal(
                    n=batch_size, m=0.8, s=1.7, device=device
                ) + 0.02 * torch.rand(batch_size, device=device)
                t = t * (1 - 2 * epsilon) + epsilon

            # Sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            dx_t = getattr(path_sample, "dx_t", None)

            # Apply mask
            if x_t.shape == (batch_size, num_tokens):  # [B, M]
                x_t *= mask
            elif x_t.shape == (batch_size, num_tokens, 3):  # [B, M, 3]
                x_t *= mask.unsqueeze(-1)
            elif x_t.shape == (batch_size, 1, 3):  # [B, 1, 3]
                x_t *= mask.any(-1, keepdim=True).unsqueeze(-1)
            else:
                raise ValueError(f"Unexpected shape for x_t: {x_t.shape}")

            # Collect inputs
            modal_input_dict[modal] = [
                x_t,
                t,
                (
                    dx_t
                    * expand_tensor_like(
                        input_tensor=self.c[self.grad_decay_method](t) * self.grad_mul,
                        expand_to=dx_t,
                    )
                    if dx_t is not None
                    else None
                ),
            ]

        # Predict velocity field for each modality
        model_output = self.flow.model(
            x=(
                modal_input_dict["atom_types"][0],  # atom_types
                modal_input_dict["pos"][0],  # pos
                modal_input_dict["frac_coords"][0],  # frac_coords
                modal_input_dict["lengths_scaled"][0],  # lengths_scaled
                modal_input_dict["angles_radians"][0],  # angles_radians
            ),
            t=(
                modal_input_dict["atom_types"][1],  # atom_types_t
                modal_input_dict["pos"][1],  # pos_t
                modal_input_dict["frac_coords"][1],  # frac_coords_t
                modal_input_dict["lengths_scaled"][1],  # lengths_scaled_t
                modal_input_dict["angles_radians"][1],  # angles_radians_t
            ),
            feats=feats,
            mask=mask,
            sdpa_backends=sdpa_backends,
        )

        # Preprocess target (velocity) tensors if requested
        for idx, modal in enumerate(self.modals):
            config = self.flow.modality_configs[idx]
            path = self.flow.paths[modal]

            if not config.get("should_rigid_align", False):
                continue

            if config.get("x_1_prediction", False):
                # When parametrizing x_1, directly align target modality to predicted modality
                pred_modal = model_output[idx]
                target_tensors[modal] = weighted_rigid_align(
                    pred_modal, target_tensors[modal], mask=mask
                )
                continue  # No need to re-sample if predicting x_1 directly

            # Otherwise, re-sample target modality velocity via one-step Euler iteration
            pred_modal_vel = model_output[idx]
            target_modal = target_tensors[modal]

            x_t = modal_input_dict[modal][0]
            t = modal_input_dict[modal][1]

            # Perform a one-step Euler iteration to get predicted clean data
            pred_modal = path.velocity_to_target(
                velocity=pred_modal_vel,
                x_t=x_t,
                t=expand_tensor_like(t, x_t),
            )

            # Align target modality to predicted modality
            x_0 = locals()[modal]  # Noised data
            x_1 = target_tensors[modal] = weighted_rigid_align(
                pred_modal, target_modal, mask=mask
            )  # Clean (aligned) data

            # Re-sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            modal_input_dict[modal][2] = path_sample.dx_t

        # Calculate the loss for each modality
        training_loss, training_loss_dict = self.flow.training_loss(
            x_1=[
                target_tensors["atom_types"] * mask,  # Mask out -100 padding
                target_tensors["pos"],
                target_tensors["frac_coords"],
                target_tensors["lengths_scaled"],
                target_tensors["angles_radians"],
            ],
            x_t=[
                modal_input_dict["atom_types"][0],
                modal_input_dict["pos"][0],
                modal_input_dict["frac_coords"][0],
                modal_input_dict["lengths_scaled"][0],
                modal_input_dict["angles_radians"][0],
            ],
            dx_t=[
                modal_input_dict["atom_types"][2],
                modal_input_dict["pos"][2],
                modal_input_dict["frac_coords"][2],
                modal_input_dict["lengths_scaled"][2],
                modal_input_dict["angles_radians"][2],
            ],
            t=[
                modal_input_dict["atom_types"][1],
                modal_input_dict["pos"][1],
                modal_input_dict["frac_coords"][1],
                modal_input_dict["lengths_scaled"][1],
                modal_input_dict["angles_radians"][1],
            ],
            model_output=model_output,
            detach_loss_dict=False,
        )
        training_loss.detach_()  # NOTE: Will manually re-aggregate losses below

        # Mask and aggregate losses
        loss_dict = {}
        for idx, modal in enumerate(self.modals):
            modal_loss_value = training_loss_dict[modal]

            pred_modal = model_output[idx]
            target_modal = target_tensors[modal]

            loss_mask = mask.float()
            loss_token_is_periodic = token_is_periodic.float()

            target_shape = target_modal.shape

            # Prepare loss masks
            if modal in ("pos", "frac_coords"):
                loss_mask = mask.unsqueeze(-1).float()
                loss_token_is_periodic = token_is_periodic.unsqueeze(-1).float()
            elif modal in ("lengths_scaled", "angles_radians"):
                loss_mask = torch.ones(
                    target_shape, dtype=torch.float32, device=target_modal.device
                )
                loss_token_is_periodic = (
                    token_is_periodic.any(-1, keepdim=True).unsqueeze(-1).float()
                )  # NOTE: A periodic sample is one with any periodic atoms

            # Mask loss values
            modal_loss_value = modal_loss_value * loss_mask

            if modal in (
                "frac_coords",
                "lengths_scaled",
                "angles_radians",
            ):  # Periodic (crystal) losses
                modal_loss_value = modal_loss_value * loss_token_is_periodic
            elif modal == "pos":  # Non-periodic (molecule) losses
                modal_loss_value = modal_loss_value * (1 - loss_token_is_periodic)

            # Collect losses
            loss_dict[f"{modal}_loss"] = modal_loss_value.mean()

        # Aggregate losses
        loss_dict["loss"] = sum(loss_dict[f"{modal}_loss"] for modal in self.modals)

        return loss_dict
