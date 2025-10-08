"""Multimodal equilibrium transformer (MET).

Adapted from:
    - https://github.com/raywang4/EqM
    - https://github.com/alexiglad/EBT
    - https://github.com/facebookresearch/flow_matching
    - https://github.com/facebookresearch/all-atom-diffusion-transformer
"""

from typing import Any, Dict, List, Literal, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path import AffineProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import expand_tensor_like

from zatom.models.ecoders.mft import MultimodalModel
from zatom.models.encoders.custom_transformer import LayerNorm
from zatom.scheduler.scheduler import EquilibriumCondOTScheduler
from zatom.utils import pylogger
from zatom.utils.multimodal import Flow
from zatom.utils.training_utils import (
    BEST_DEVICE,
    weighted_rigid_align,
)
from zatom.utils.typing_utils import Bool, Float, Int, typecheck

log = pylogger.RankedLogger(__name__)

GRAD_DECAY_METHODS = Literal["linear_decay", "truncated_decay", "piecewise_decay"]


class MET(nn.Module):
    """Multimodal equilibrium matching model with a Transformer encoder/decoder (i.e., an E-coder
    or `ecoder`).

    NOTE: The `_forward` pass of this model is conceptually similar to that of All-atom Diffusion Transformers (ADiTs)
    except that there is no self (and maybe time) conditioning and the model learns equilibrium transport for each modality.

    Args:
        encoder: The encoder module.
        d_x: Input dimension.
        d_model: Model dimension.
        num_layers: Number of Transformer layers.
        nhead: Number of attention heads.
        num_datasets: Number of datasets for context conditioning.
        num_spacegroups: Number of spacegroups for context conditioning.
        max_num_elements: Maximum number of elements in the dataset.
        context_length: Context length for the attention mechanism.
        rope_base: Base frequency for rotary positional encoding.
        mlp_ratio: Ratio of hidden to input dimension in MLP.
        proj_drop: Dropout probability for the projection layer.
        attn_drop: Dropout probability for the attention layer.
        class_dropout_prob: Probability of dropping class labels for context conditioning.
        atom_types_reconstruction_loss_weight: Weighting factor for the atom types reconstruction loss.
        pos_reconstruction_loss_weight: Weighting factor for the atom positions reconstruction loss.
        frac_coords_reconstruction_loss_weight: Weighting factor for the atom fractional coordinates reconstruction loss.
        lengths_scaled_reconstruction_loss_weight: Weighting factor for the atom lengths (scaled) reconstruction loss.
        angles_radians_reconstruction_loss_weight: Weighting factor for the atom angles (radians) reconstruction loss.
        grad_decay_a: Parameter `a` for gradient decay functions.
        grad_decay_b: Parameter `b` for gradient decay functions.
        grad_mul: Global multiplier for the target gradient magnitude.
        early_stopping_grad_norm (Optional[float], optional): If specified,
            sampling will stop early if the model output velocity (or gradient)
            norm with respect to each modality falls below this value. This
            effectively enables adaptive compute for sampling. Defaults to
            ``None``.
        qkv_bias: If True, add a learnable bias to query, key, value.
        qk_norm: If True, apply normalization to query and key.
        scale_attn_norm: If True, apply scaling to attention
            normalization.
        proj_bias: If True, add bias to output projection.
        flex_attn: Whether to use PyTorch's FlexAttention.
        fused_attn: Whether to use PyTorch's `scaled_dot_product_attention`.
        jvp_attn: Whether to use a Triton kernel for Jacobian-vector product (JVP) Flash Attention.
        weighted_rigid_align_pos: Whether to apply weighted rigid alignment between target and predicted atom positions for loss calculation.
        weighted_rigid_align_frac_coords: Whether to apply weighted rigid alignment between target and predicted atom fractional coordinates for loss calculation.
        use_pytorch_implementation: Whether to use PyTorch's Transformer implementation.
        remove_t_conditioning: Whether to remove timestep conditioning for each modality.
        add_mask_atom_type: Whether to add a mask token for atom types.
        norm_layer: Normalization layer.
        grad_decay_method: Method for decaying the target gradient magnitude as a function of time.
            Must be one of (`linear_decay`, `truncated_decay`, `piecewise_decay`).
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_x: int = 512,
        d_model: int = 768,
        num_layers: int = 12,
        nhead: int = 12,
        num_datasets: int = 2,  # Context conditioning input
        num_spacegroups: int = 230,  # Context conditioning input
        max_num_elements: int = 100,
        context_length: int | None = 2048,
        rope_base: int | None = 10_000,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.1,
        attn_drop: float = 0.0,
        class_dropout_prob: float = 0.1,
        atom_types_reconstruction_loss_weight: float = 1.0,
        pos_reconstruction_loss_weight: float = 10.0,
        frac_coords_reconstruction_loss_weight: float = 10.0,
        lengths_scaled_reconstruction_loss_weight: float = 1.0,
        angles_radians_reconstruction_loss_weight: float = 10.0,
        grad_decay_a: float = 0.8,
        grad_decay_b: float = 0.8,
        grad_mul: float = 4.0,
        early_stopping_grad_norm: float | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        scale_attn_norm: bool = False,
        proj_bias: bool = False,
        flex_attn: bool = False,
        fused_attn: bool = True,
        jvp_attn: bool = False,
        weighted_rigid_align_pos: bool = True,
        weighted_rigid_align_frac_coords: bool = True,
        use_pytorch_implementation: bool = False,
        remove_t_conditioning: bool = True,
        add_mask_atom_type: bool = True,
        norm_layer: Type[nn.Module] = LayerNorm,
        grad_decay_method: GRAD_DECAY_METHODS = "truncated_decay",
    ):
        super().__init__()

        assert (
            sum([flex_attn, fused_attn, jvp_attn]) <= 1
        ), "Only one of flex_attn, fused_attn, or jvp_attn can be True."

        self.class_dropout_prob = class_dropout_prob
        self.atom_types_reconstruction_loss_weight = atom_types_reconstruction_loss_weight
        self.pos_reconstruction_loss_weight = pos_reconstruction_loss_weight
        self.frac_coords_reconstruction_loss_weight = frac_coords_reconstruction_loss_weight
        self.lengths_scaled_reconstruction_loss_weight = lengths_scaled_reconstruction_loss_weight
        self.angles_radians_reconstruction_loss_weight = angles_radians_reconstruction_loss_weight
        self.grad_mul = grad_mul
        self.jvp_attn = jvp_attn
        self.grad_decay_method = grad_decay_method

        self.vocab_size = max_num_elements + int(add_mask_atom_type)
        self.modals = ["atom_types", "pos", "frac_coords", "lengths_scaled", "angles_radians"]

        # Build multimodal model
        model = MultimodalModel(
            encoder=encoder,
            d_x=d_x,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            num_datasets=num_datasets,
            num_spacegroups=num_spacegroups,
            max_num_elements=max_num_elements,
            context_length=context_length,
            rope_base=rope_base,
            mlp_ratio=mlp_ratio,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            class_dropout_prob=class_dropout_prob,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_attn_norm=scale_attn_norm,
            proj_bias=proj_bias,
            flex_attn=flex_attn,
            fused_attn=fused_attn,
            jvp_attn=jvp_attn,
            use_pytorch_implementation=use_pytorch_implementation,
            remove_t_conditioning=remove_t_conditioning,
            add_mask_atom_type=add_mask_atom_type,
            norm_layer=norm_layer,
        )

        # Instantiate paths and losses for Flow
        modalities = {
            "atom_types": {
                "path": MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0)),
                # loss omitted → Flow will use MixturePathGeneralizedKL automatically
                "weight": self.atom_types_reconstruction_loss_weight,
            },
            "pos": {
                "path": AffineProbPath(scheduler=EquilibriumCondOTScheduler()),
                # loss omitted → Flow will use squared error automatically
                "weight": self.pos_reconstruction_loss_weight,
            },
            "frac_coords": {
                "path": AffineProbPath(scheduler=EquilibriumCondOTScheduler()),
                # loss omitted → Flow will use squared error automatically
                "weight": self.frac_coords_reconstruction_loss_weight,
            },
            "lengths_scaled": {
                "path": AffineProbPath(scheduler=EquilibriumCondOTScheduler()),
                # loss omitted → Flow will use squared error automatically
                "weight": self.lengths_scaled_reconstruction_loss_weight,
            },
            "angles_radians": {
                "path": AffineProbPath(scheduler=EquilibriumCondOTScheduler()),
                # loss omitted → Flow will use squared error automatically
                "weight": self.angles_radians_reconstruction_loss_weight,
            },
        }
        self.flow = Flow(
            model=model,
            modalities=modalities,
            model_sampling_fn="forward_with_cfg",
            early_stopping_grad_norm=early_stopping_grad_norm,
        )

        self.should_rigid_align = {
            "pos": weighted_rigid_align_pos,
            "frac_coords": weighted_rigid_align_frac_coords,
        }

        self.a, self.b = grad_decay_a, grad_decay_b
        self.c = {
            "linear_decay": lambda t: 1 - t,
            "truncated_decay": lambda t: torch.where(
                t <= self.a, torch.ones_like(t), (1 - t) / (1 - self.a)
            ),
            "piecewise_decay": lambda t: torch.where(
                t <= self.a, self.b - ((self.b - 1) / self.a) * t, (1 - t) / (1 - self.a)
            ),
        }

    @typecheck
    def forward(
        self,
        dataset_idx: Int[" b"],  # type: ignore
        spacegroup: Int[" b"],  # type: ignore
        mask: Bool["b m"],  # type: ignore
        steps: int = 100,
        cfg_scale: float = 2.0,
        modal_input_dict: (
            Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] | None
        ) = None,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], None]:
        """Forward pass of MET with classifier-free guidance.

        Args:
            dataset_idx: Dataset index for each sample.
            spacegroup: Spacegroup index for each sample.
            mask: True if valid token, False if padding.
            steps: Number of integration steps for the multimodal ODE solver.
            cfg_scale: Classifier-free guidance scale.
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
        dataset_idx_null = torch.zeros_like(dataset_idx)
        dataset_idx = torch.cat([dataset_idx, dataset_idx_null], dim=0)  # [2B]
        spacegroup_null = torch.zeros_like(spacegroup)
        spacegroup = torch.cat([spacegroup, spacegroup_null], dim=0)  # [2B]
        mask = torch.cat([mask, mask], dim=0)  # [2B, N]

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
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=mask,
            cfg_scale=cfg_scale,
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
        dataset_idx: Int[" b"],  # type: ignore
        spacegroup: Int[" b"],  # type: ignore
        mask: Bool["b m"],  # type: ignore
        token_is_periodic: Bool["b m"],  # type: ignore
        target_tensors: Dict[str, torch.Tensor],
        epsilon: float = 1e-3,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of MET with loss calculation.

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
            epsilon: Small constant to avoid numerical issues in loss calculation.
            kwargs: Additional keyword arguments (not used).

        Returns:
            Dictionary of loss values.
        """
        device = atom_types.device
        batch_size, num_tokens = atom_types.shape

        # Sample time points and corresponding noised inputs for each modality
        modal_input_dict = {}
        for modal in self.modals:
            path = self.flow.paths[modal]

            x_0 = locals()[modal]  # Noised data
            x_1 = target_tensors[modal]  # Clean data

            # Sample a time point from a uniform distribution
            t = torch.rand(batch_size, device=device) * (1 - epsilon)

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

        # Predict average velocity field for each modality
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
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=mask,
        )

        # Preprocess target tensors if requested
        for idx, modal in enumerate(self.modals):
            if modal not in ("pos", "frac_coords"):
                continue

            pred_modal = model_output[idx]
            target_modal = modal_input_dict[modal][2]

            # Align target modality to predicted modality if specified
            modal_input_dict[modal][2] = (
                weighted_rigid_align(pred_modal, target_modal, mask=mask)
                if self.should_rigid_align[modal]
                else target_modal
            )

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
        training_loss.detach_()  # Will manually re-aggregate losses below

        unused_loss = torch.tensor(torch.nan, device=device)

        # Mask and aggregate losses
        loss_dict = {}
        reconstruction_loss_dict = {modal: 0 for modal in self.modals}

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

            reconstruction_loss_dict[modal] += modal_loss_value

            # Handle atom types specially
            if modal == "atom_types":
                nll_loss = (
                    F.nll_loss(
                        input=F.log_softmax(pred_modal, dim=-1).reshape(-1, self.vocab_size),
                        target=target_modal.reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).reshape(target_shape)
                    * loss_mask
                )
                ppl_loss = torch.exp(nll_loss).detach()

            # Collect losses
            total_loss = reconstruction_loss_dict[modal]

            loss_dict.update(
                {
                    f"{modal}_loss": total_loss.mean(),
                    f"{modal}_initial_loss": unused_loss,
                    f"{modal}_final_step_loss": unused_loss,
                    f"{modal}_initial_final_pred_energies_gap": unused_loss,
                }
            )

            if modal == "atom_types":
                loss_dict[f"{modal}_ce_loss"] = unused_loss
                loss_dict[f"{modal}_ppl_loss"] = ppl_loss.mean()
            else:
                loss_dict[f"{modal}_mse_loss"] = unused_loss

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
