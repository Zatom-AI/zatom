from __future__ import annotations

from beartype import beartype
from environs import Env
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped
from torch import Tensor

from zatom.utils.utils import identity, package_available

# Environment

env = Env()
env.read_env()


# NOTE: `jaxtyping` is a misnomer, since it works for PyTorch as well


class TorchTyping:
    """Torch typing."""

    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        """Get item."""
        return self.abstract_dtype[Tensor, shapes]


Shaped = TorchTyping(Shaped)
Float = TorchTyping(Float)
Int = TorchTyping(Int)
Bool = TorchTyping(Bool)

# NOTE: Use env variable `TYPECHECK` to control whether to use `beartype` + `jaxtyping`
# NOTE: Use env variable `DEBUG` to control whether to enable `lovely_tensors`

IS_TYPECHECKING = env.bool("TYPECHECK", False)
IS_DEBUGGING = env.bool("DEBUG", False)

typecheck = jaxtyped(typechecker=beartype) if IS_TYPECHECKING else identity

if IS_TYPECHECKING:
    print("Type checking is enabled.")
else:
    print("Type checking is disabled.")

if IS_DEBUGGING:
    print("Debugging is enabled.")

    if package_available("lovely_tensors"):
        import lovely_tensors as lt

        print("With debugging enabled, monkey patching PyTorch with `lovely_tensors`.")
        lt.monkey_patch()
    else:
        print("`lovely_tensors` is not available, debugging will not be as informative.")

else:
    print("Debugging is disabled.")

__all__ = [
    "IS_TYPECHECKING",
    "IS_DEBUGGING",
    "Bool",
    "Float",
    "Int",
    "Shaped",
    "typecheck",
]
