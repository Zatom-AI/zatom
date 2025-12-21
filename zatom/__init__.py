import importlib
import math
import secrets
import string
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from zatom.utils.training_utils import get_lr_scheduler
from zatom.utils.typing_utils import typecheck


@typecheck
def set_omegaconf_flag_recursive(cfg: Any, flag: str, value: bool) -> None:
    """Recursively set an OmegaConf flag on all nodes in the config tree.

    Args:
        cfg: The config node (DictConfig, ListConfig, or primitive).
        flag: The OmegaConf flag name ('readonly', 'allow_objects', etc.).
        value: The boolean value to set for the flag.
    """
    if isinstance(cfg, (DictConfig, ListConfig)):
        cfg._set_flag(flag, value)
        for v in cfg.values() if isinstance(cfg, DictConfig) else cfg:
            set_omegaconf_flag_recursive(v, flag, value)


@typecheck
def generate_index(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits.

    Args:
        length: The length of the string to generate.

    Returns:
        The generated string.
    """
    alphabet = string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@typecheck
def resolve_omegaconf_variable(variable_path: str) -> Any:
    """Resolve an OmegaConf variable path to its value.

    Args:
        variable_path: The dot-separated path to the variable (e.g., "module.submodule.variable").

    Returns:
        The value of the resolved variable.
    """
    # Split the string into parts using the dot separator
    parts = variable_path.rsplit(".", 1)

    # Get the module name from the first part of the path
    module_name = parts[0]

    # Dynamically import the module using the module name
    try:
        module = importlib.import_module(module_name)
        # Use the imported module to get the requested attribute value
        attribute = getattr(module, parts[1])
    except Exception:
        module = importlib.import_module(".".join(module_name.split(".")[:-1]))
        inner_module = ".".join(module_name.split(".")[-1:])
        # Use the imported module to get the requested attribute value
        attribute = getattr(getattr(module, inner_module), parts[1])

    return attribute


@typecheck
def resolve_lr(
    lr: float,
    base_world_size: int,
    world_size: int,
    scale_factor: int | float,
    scale_sqrt: bool = False,
) -> float:
    """Resolve learning rate based on base learning rate, (base) world size, and scale factor.

    If requested, applies square root scaling rule based on the ratio of
    the world size to the base world size to preserve the variance of gradients.
    Reference: https://github.com/Lightning-AI/pytorch-lightning/discussions/3706#discussioncomment-3960433.

    Args:
        lr: Base learning rate.
        base_world_size: Base world size used for scaling.
        world_size: Current world size.
        scale_factor: Additional scale factor to apply.
        scale_sqrt: Whether to apply square root scaling based on world size.

    Returns:
        The resolved learning rate.
    """
    return (
        lr * scale_factor * math.sqrt(world_size / base_world_size)
        if scale_sqrt
        else lr * scale_factor
    )


@typecheck
def resolve_batch_size(base_size: int, scale_factor: int | float) -> int:
    """Resolve batch size based on base size and scale factor.

    Args:
        base_size: The base batch size.
        scale_factor: The scale factor to apply.

    Returns:
        The resolved batch size.
    """
    return max(1, round(base_size * scale_factor))


@typecheck
def resolve_grad_accum_steps(base_steps: int, scale_factor: int | float) -> int:
    """Resolve gradient accumulation steps based on base steps and scale factor.

    Args:
        base_steps: The base number of gradient accumulation steps.
        scale_factor: The scale factor to apply.

    Returns:
        The resolved number of gradient accumulation steps.
    """
    return max(1, round(base_steps / scale_factor))


def register_custom_omegaconf_resolvers():
    """Register custom OmegaConf resolvers."""
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
    OmegaConf.register_new_resolver("generate_index", lambda length: generate_index(length))
    OmegaConf.register_new_resolver(
        "resolve_variable",
        lambda variable_path: resolve_omegaconf_variable(variable_path),
    )
    OmegaConf.register_new_resolver(
        "resolve_lr",
        lambda lr, base_world_size, world_size, scale_factor, scale_sqrt: resolve_lr(
            lr, base_world_size, world_size, scale_factor, scale_sqrt
        ),
    )
    OmegaConf.register_new_resolver(
        "resolve_lr_scheduler",
        lambda scheduler, warmup_steps=None, total_steps=None, num_cycles=0.5, min_lr_factor=1e-5: get_lr_scheduler(
            scheduler,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            num_cycles=num_cycles,
            min_lr_factor=min_lr_factor,
        ),
    )

    OmegaConf.register_new_resolver(
        "resolve_batch_size",
        lambda base_size, scale_factor: resolve_batch_size(base_size, scale_factor),
    )
    OmegaConf.register_new_resolver(
        "resolve_grad_accum_steps",
        lambda base_steps, scale_factor: resolve_grad_accum_steps(base_steps, scale_factor),
    )
