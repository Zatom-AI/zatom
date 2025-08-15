import importlib
import secrets
import string
from typing import Any

from omegaconf import OmegaConf

from zatom.utils.training_utils import get_lr_scheduler


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
    # split the string into parts using the dot separator
    parts = variable_path.rsplit(".", 1)

    # get the module name from the first part of the path
    module_name = parts[0]

    # dynamically import the module using the module name
    try:
        module = importlib.import_module(module_name)
        # use the imported module to get the requested attribute value
        attribute = getattr(module, parts[1])
    except Exception:
        module = importlib.import_module(".".join(module_name.split(".")[:-1]))
        inner_module = ".".join(module_name.split(".")[-1:])
        # use the imported module to get the requested attribute value
        attribute = getattr(getattr(module, inner_module), parts[1])

    return attribute


def register_custom_omegaconf_resolvers():
    """Register custom OmegaConf resolvers."""
    OmegaConf.register_new_resolver("generate_index", lambda length: generate_index(length))
    OmegaConf.register_new_resolver(
        "resolve_variable",
        lambda variable_path: resolve_omegaconf_variable(variable_path),
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
