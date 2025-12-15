import importlib
import secrets
import string
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from zatom.utils.training_utils import get_lr_scheduler


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


def generate_index(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits.

    Args:
        length: The length of the string to generate.

    Returns:
        The generated string.
    """
    alphabet = string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def resolve_omegaconf_variable(variable_path: str) -> Any:
    """Resolve an OmegaConf variable path to its value."""
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


def resolve_lr(lr: float, base_world_size: int, world_size: int, scale_factor: float) -> float:
    """Resolve learning rate based on base learning rate, (base) world size, and scale factor."""
    return lr * scale_factor * (world_size / base_world_size)  # Apply linear scaling rule


def resolve_batch_size(base_size: int, scale_factor: float) -> int:
    """Resolve batch size based on base size and scale factor."""
    return max(1, round(base_size * scale_factor))


def resolve_grad_accum_steps(base_steps: int, scale_factor: float) -> int:
    """Resolve gradient accumulation steps based on base steps and scale factor."""
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
        lambda lr, base_world_size, world_size, scale_factor: resolve_lr(
            lr, base_world_size, world_size, scale_factor
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
