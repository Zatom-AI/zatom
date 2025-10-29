from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch

# flow_matching
from flow_matching.loss.generalized_loss import MixturePathGeneralizedKL
from flow_matching.path.mixture import MixtureDiscreteProbPath
from flow_matching.utils import expand_tensor_like
from torch import Tensor, nn

from zatom.solver.multimodal_solver import MultimodalSolver
from zatom.utils import pylogger

MULTIMODAL_METHOD = Literal["euler"]

log = pylogger.RankedLogger(__name__)


def _default_continuous_loss(pred: Tensor, target: Tensor, reduction: str = "none") -> Tensor:
    """Squared error loss for continuous modalities.

    Args:
        pred (Tensor): predicted velocity field.
        target (Tensor): target velocity field.
        reduction (str): reduction method, one of 'mean', 'sum', or 'none'.

    Raises:
        ValueError: if reduction is not one of 'none', 'mean', or 'sum'.

    Returns:
        Tensor: computed loss.
    """
    loss = (pred - target) ** 2

    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")


class Flow(nn.Module):
    """Generic multimodal flow matching model.

    This class aggregates multiple modalities, each with its own model, path,
    scheduler, and loss. It provides utilities for training (computing the total
    loss) and inference (sampling) across all modalities.

    Args:
        model (nn.Module):
            A model that receives a sequence of state tensors
            (one per modality) as ``x`` and a scalar time tensor ``t``,
            and returns a sequence of output tensors. For continuous modalities,
            the output is a velocity. For discrete modalities, it is the
            posterior probability `p_1t`.
        modalities (Dict[str, Dict[str, Any]]):
            Mapping from modality name to a dict with keys:
                - "path": A probability path object (e.g., MixtureDiscreteProbPath for discrete data,
                or any continuous path implementation).
                - "loss" (optional): A callable loss function. If omitted, a default loss is chosen
                based on the path type.
                - "weight" (optional): A float weight for the modality's training loss. Defaults to 1.0.
                - "x_1_prediction" (continuous paths only, optional): If True, the model is expected to predict
                the clean data `x_1` for that modality, and such predictions will be reparameterized
                as velocities during the sampling process. If False, the model is expected to predict
                the velocities directly. Defaults to False.
        model_sampling_fn (str, optional): If ``model`` is a class instance
            with multiple methods, this specifies the method to use for
            forward passes during sampling. Defaults to ``"forward"``.
        early_stopping_grad_norm (Optional[float], optional): If specified,
            sampling will stop early if the model output velocity (or gradient)
            norm with respect to each modality falls below this value. This
            effectively enables adaptive compute for sampling. Defaults to
            ``None``.
        enable_mean_flows (bool, optional): If ``True``, enables mean flows for continuous modalities.
            Defaults to ``False``. If enabled, the model is expected to predict the average velocity
            field for each continuous modality, and discrete modalities must be treated as continuous
            (one-hot) vectors via preprocessing preceding this module.
    """

    def __init__(
        self,
        model: nn.Module,
        modalities: Dict[str, Dict[str, Any]],
        model_sampling_fn: str = "forward",
        early_stopping_grad_norm: Optional[float] = None,
        enable_mean_flows: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.enable_mean_flows = enable_mean_flows

        self.paths: Dict[str, Any] = {}
        self.loss_fns: Dict[str, Callable] = {}
        self.loss_weights: Dict[str, float] = {}

        if enable_mean_flows:
            assert all(
                not isinstance(spec["path"], MixtureDiscreteProbPath)
                for spec in modalities.values()
            ), "Mean flows cannot be constructed for discrete modalities."

        for name, spec in modalities.items():
            path = spec["path"]
            self.paths[name] = path

            # Choose loss function
            loss_fn = spec.get("loss")
            if loss_fn is None:
                if isinstance(path, MixtureDiscreteProbPath):
                    loss_fn = MixturePathGeneralizedKL(path, reduction="none")
                else:
                    loss_fn = _default_continuous_loss
            self.loss_fns[name] = loss_fn
            self.loss_weights[name] = spec.get("weight", 1.0)

        # Set up Euler solver for each modality.
        self.modality_configs = [
            {
                "name": name,
                "type": (
                    "discrete" if isinstance(path, MixtureDiscreteProbPath) else "continuous"
                ),
                "path": path,
                "x_1_prediction": modalities[name].get("x_1_prediction", False),
                "should_rigid_align": modalities[name].get("should_rigid_align", None),
                "should_center_during_sampling": modalities[name].get(
                    "should_center_during_sampling", False
                ),
            }
            for name, path in self.paths.items()
        ]
        self.solver = MultimodalSolver(
            model=self.model,
            modality_configs=self.modality_configs,
            model_sampling_fn=model_sampling_fn,
            early_stopping_grad_norm=early_stopping_grad_norm,
            enable_mean_flows=enable_mean_flows,
        )

    def training_loss(
        self,
        x_1: Sequence[Tensor],
        x_t: Sequence[Tensor],
        dx_t: Sequence[Tensor],
        t: Sequence[Tensor | Tuple[Tensor, Tensor]],
        model_output: Optional[Sequence[Tensor]] = None,
        model_output_dt: Optional[Sequence[Tensor]] = None,
        detach_loss_dict: bool = True,
        mf_norm_eps: float = 1e-3,
        mf_norm_p: float = 0.75,
        **model_extras: dict,
    ) -> Tuple[Sequence[Tensor], Dict[str, Tensor]]:
        """Compute the total training loss across all modalities.

        Args:
            x_1 (Sequence[Tensor]): Sequence of tensors, one per modality,
                containing the data at time 1.
            x_t (Sequence[Tensor]): Sequence of tensors, one per modality,
                containing the data at time t.
            dx_t (Sequence[Tensor]): Sequence of tensors, one per modality,
                containing the velocity field at time t.
            t (Sequence[Tensor | Tuple[Tensor, Tensor]]): Sequence of tensors, one per modality,
                containing the time values or (start_time, end_time) tuples.
            model_output (Optional[Sequence[Tensor]]): Optional precomputed model outputs.
                If provided, these are used instead of calling the model.
            model_output_dt (Optional[Sequence[Tensor]]): Optional precomputed model output time derivatives.
                These must be provided if using mean flows.
            detach_loss_dict (bool): If ``True``, detaches individual modality losses
                from the computation graph when storing them in the loss dictionary.
                Defaults to ``True``.
            mf_norm_eps (float): Small epsilon added to the adaptive weighting
                term when using mean flows. Defaults to ``1e-3``.
            mf_norm_p (float): Exponent for the adaptive weighting term when
                using mean flows. Defaults to ``0.75``.
            **model_extras (dict): Additional keyword arguments to pass to the model.

        Returns:
            Tuple[Sequence[Tensor], Dict[str, Tensor]]:
                Scalar loss (sum of modality losses) and a dictionary
                of individual modality losses.
        """
        assert (
            len(x_1) == len(x_t) == len(dx_t) == len(t) == len(self.paths)
        ), "Input sequences must match the number of modalities."

        if model_output is not None:
            assert len(model_output) == len(
                self.paths
            ), "If provided, model outputs must match the number of modalities."

        if self.enable_mean_flows:
            # When using mean flows, t must be a tuple of (start_times, end_times).
            assert all(
                isinstance(time, tuple) and len(time) == 2 for time in t
            ), "When using mean flows, t must be a tuple of (start_times, end_times) for each modality."
            assert model_output_dt is not None and len(model_output_dt) == len(
                self.paths
            ), "When using mean flows, model_output_dt must be provided and match the number of modalities."

        loss_dict = {}
        total_loss = 0.0

        model_output = model_output or self.model(x_t, t, **model_extras)

        for i, name in enumerate(self.paths):
            path = self.paths[name]
            loss_fn = self.loss_fns[name]
            modality_config = self.modality_configs[i]

            if isinstance(path, MixtureDiscreteProbPath):
                # Discrete case: model should output logits.
                assert x_t[i].dtype == torch.long, (
                    f"Expected integer tensor for discrete modality '{name}', "
                    f"got {x_t[i].dtype}",
                )
                loss = loss_fn(
                    model_output[i],
                    x_1[i],
                    x_t[i],
                    t[i],
                )
            elif self.enable_mean_flows:
                # Continuous case with mean flows: model outputs average velocity field.
                assert x_t[i].is_floating_point(), (
                    f"Expected float tensor for continuous modality '{name}', "
                    f"got {x_t[i].dtype}",
                )
                modal_t, modal_r = t[i]
                loss = loss_fn(
                    model_output[i],
                    (
                        dx_t[i]
                        - expand_tensor_like(modal_t - modal_r, dx_t[i]) * model_output_dt[i]
                    ).detach(),
                )
                # Adaptive weighting
                adp_wt = (loss.detach() + mf_norm_eps) ** mf_norm_p
                loss = loss / adp_wt
            else:
                # Continuous case without mean flows: model returns instantaneous velocity field.
                assert x_t[i].is_floating_point(), (
                    f"Expected float tensor for continuous modality '{name}', "
                    f"got {x_t[i].dtype}",
                )
                loss = loss_fn(
                    model_output[i],
                    x_1[i] if modality_config["x_1_prediction"] else dx_t[i],
                )

            weight = self.loss_weights[name]
            loss_dict[name] = (loss.detach() if detach_loss_dict else loss) * weight
            total_loss = total_loss + loss.mean() * weight

        return total_loss, loss_dict

    def sample(
        self,
        x_init: Sequence[Tensor],
        time_grid: Optional[Tensor] = None,
        device: torch.device = torch.device("cpu"),
        steps: int = 1000,
        step_size: Optional[float] = None,
        div_free: Union[float, Callable[[float], float]] = 0.0,
        method: MULTIMODAL_METHOD = "euler",
        enable_zero_centering: bool = True,
        return_intermediates: bool = False,
        enable_grad: bool = False,
        verbose: bool = False,
        **model_extras: dict,
    ) -> Union[Sequence[Tensor], Sequence[List[Tensor]]]:
        """Generate samples for each modality using the inference scheduler.

        Args:
            x_init (Sequence[Tensor]):
                Sequence of tensors, one per modality, containing the initial states at time 0.
                For continuous modalities, this is typically Gaussian noise.
                For discrete modalities, this is typically samples from a uniform categorical distribution.
            time_grid (Optional[Tensor]): Optional tensor of time points defining the interval.
                If provided, it overrides the uniform discretization defined by `steps`.
            device (torch.device, optional): Device on which to run the sampling.
            steps (int, optional): Number of integration steps for the ODE solver.
            step_size (Optional[float]): Fixed step size for uniform discretization.
                If ``None``, the step size is computed from ``steps``.
            div_free (Union[float, Callable[[float], float]]): The coefficient
                of the divergence-free term in the probability velocity
                (for discrete modalities). Can be either a float or a time
                dependent function. Defaults to 0.0.
            method (MULTIMODAL_METHOD): Numerical integration method. Currently only ``"euler"`` is
                supported, representing a single forward step.
            enable_zero_centering (bool): Whether to allow centering of continuous modalities
                at the origin after each denoising step. Defaults to ``True``.
            return_intermediates (bool): If ``True``, returns a list of tensors for
                each modality containing the state at each intermediate time step.
            enable_grad (bool): Whether to enable gradient tracking during integration.
            verbose (bool): If ``True``, prints progress during sampling.
            **model_extras (dict): Additional keyword arguments to pass to the model.

        Returns:
            Union[Sequence[Tensor], Sequence[List[Tensor]]]: A list where each element corresponds to a modality.
            Each element is either a tensor of shape ``(batch_size, ...)`` containing the samples,
            or a list of tensors (if `return_intermediates` is True in `MultimodalSolver.sample`).
        """
        # Validate samples for each modality.
        x_init = x_init if isinstance(x_init, list) else list(x_init)
        for i, name in enumerate(self.paths):
            path = self.paths[name]

            if isinstance(path, MixtureDiscreteProbPath):
                assert x_init[i].dtype == torch.long, (
                    f"Expected integer tensor for discrete modality '{name}', "
                    f"got {x_init[i].dtype}",
                )
            else:
                assert x_init[i].is_floating_point(), (
                    f"Expected float tensor for continuous modality '{name}', "
                    f"got {x_init[i].dtype}",
                )

            x_init[i] = x_init[i].to(device)

        # Solve to obtain multimodal samples at time 1.
        step_size = step_size or (1.0 / steps)
        time_grid = (
            time_grid if time_grid is not None else torch.linspace(0.0, 1.0, steps, device=device)
        )

        if self.enable_mean_flows:
            log.info("Sampling with mean flows enabled. Applying one-step sampling.")
            step_size = 1.0  # NOTE: Mean flows only support one-step sampling for now
            time_grid = torch.tensor([0.0, 1.0], device=device)

        samples = self.solver.sample(
            x_init=x_init,
            step_size=step_size,
            div_free=div_free,
            method=method,
            time_grid=time_grid,
            enable_zero_centering=enable_zero_centering,
            return_intermediates=return_intermediates,
            enable_grad=enable_grad,
            verbose=verbose,
            **model_extras,
        )

        return samples
