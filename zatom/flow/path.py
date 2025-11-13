"""Adapted from https://github.com/carlosinator/tabasco."""

from dataclasses import dataclass

from tensordict import TensorDict


@dataclass
class FlowPath:
    """Container for flow path data."""

    x_1: TensorDict
    x_t: TensorDict
    dx_t: TensorDict
    x_0: TensorDict
    t: TensorDict
