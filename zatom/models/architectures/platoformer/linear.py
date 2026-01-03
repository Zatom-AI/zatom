"""Adapted from https://github.com/niazoys/PlatonicTransformers."""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from torch import Tensor

from zatom.utils.typing_utils import typecheck


class PlatonicLinear(nn.Module):
    """A Linear layer constrained to be a group convolution over a Platonic Solid group. Includes a
    corrected initialization scheme to preserve variance.

    This is a tweaked version of PlatonicLinear from the original repo. It adds two options, fused
    and cache_gidx, which improve its computational footprint a little.
    """

    @typecheck
    def __init__(
        self,
        c_in: int,  # per group element
        c_out: int,  # per group element
        solid_name: str,
        bias: bool = True,
        fused: bool = True,
        cache_gidx: bool = True,
    ) -> None:
        super().__init__()

        if solid_name.lower() not in PLATONIC_GROUPS:
            raise ValueError(
                f"Solid '{solid_name}' not recognized. Available groups are: {list(PLATONIC_GROUPS.keys())}"
            )

        group = PLATONIC_GROUPS[solid_name.lower()]
        self.G = group.G
        self.c_in = c_in
        self.c_out = c_out
        self.fused = fused
        self.cache_gidx = cache_gidx

        self.kernel = nn.Parameter(torch.empty(self.G, self.c_out, self.c_in))

        if bias:
            self.register_parameter("bias", nn.Parameter(torch.empty(self.c_out)))
        else:
            self.bias = None

        self.register_buffer("cayley_table", group.cayley_table)
        self.register_buffer("inverse_indices", group.inverse_indices)
        if cache_gidx:
            # Pre-compute kernel_group_idx
            # The original implementation recomputes this each forward pass, which is unnecessary.
            kernel_group_idx = self._expand_gidx()
            self.register_buffer("kernel_group_idx", kernel_group_idx)

        self.reset_parameters()

    @typecheck
    def reset_parameters(self) -> None:
        """Initialize the kernel and bias with variance-preserving scaling.

        Standard initializers (like Kaiming) fail to correctly infer the effective fan-in of the
        full weight matrix. We must calculate it manually as (group_size * c_in).
        """
        # Calculate the effective fan-in for the full weight matrix.
        fan_in = self.G * self.c_in

        # Initialize the kernel from a normal distribution. The std is calculated
        # to ensure the output variance is approximately equal to the input variance.
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.kernel, mean=0.0, std=std)

        if self.bias is not None:
            # Initialize bias using the same correct fan-in.
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    @typecheck
    def _expand_gidx(self) -> Tensor:
        h_indices = torch.arange(self.G, device=self.kernel.device).view(self.G, 1)
        g_indices = torch.arange(self.G, device=self.kernel.device).view(1, self.G)
        inv_g_indices = self.inverse_indices[g_indices]
        kernel_group_idx = self.cayley_table[inv_g_indices, h_indices]
        return kernel_group_idx

    @typecheck
    def get_weight(self) -> Tensor:
        """Constructs the full [G*Cout, G*Cin] weight matrix from the fundamental kernel."""
        if self.cache_gidx:
            kernel_group_idx = self.kernel_group_idx
        else:
            kernel_group_idx = self._expand_gidx()

        expanded_kernel = self.kernel[kernel_group_idx]  # (Gout, Gin, Cout, Cin)
        weight = expanded_kernel.permute(0, 2, 1, 3).reshape(
            self.G * self.c_out, self.G * self.c_in
        )
        return weight

    @typecheck
    def _forward_orig(self, x: Tensor) -> Tensor:
        """Original PlatonicLinear implementation."""
        weight = self.get_weight()
        output = F.linear(x, weight, None)

        if self.bias is not None:
            output_shape = output.shape
            output = output.view(*output_shape[:-1], self.G, self.c_out)
            output = output + self.bias
            output = output.view(output_shape)

        return output

    @typecheck
    def _forward_fused(self, x: Tensor) -> Tensor:
        """Optimized implementation, fuses matmul with bias summation via F.linear."""
        bias = None
        if self.bias is not None:
            if self.G == 1:
                bias = self.bias
            else:
                bias = self.bias.view(1, self.c_out).expand(self.G, -1).flatten()  # (G*C,)
        weight = self.get_weight()
        return F.linear(x, weight, bias)

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward of PlatonicLinear."""
        if self.fused:
            return self._forward_fused(x)
        else:
            return self._forward_orig(x)

    @typecheck
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(G={self.G}, c_in={self.c_in}, c_out={self.c_out}, bias={self.bias is not None})"


if __name__ == "__main__":

    import time

    from zatom.models.architectures.platoformer import PLATONIC_GROUPS_3D

    def benchmark(
        solid_name: str,
        cache_gidx: bool,
        fused: bool,
        N_runs: int = 128,
        N_warmup: int = 16,
        bias: bool = True,
        baseline: float = None,
    ):
        group = PLATONIC_GROUPS[solid_name]
        G = group.G

        B = 256
        N = 128
        C = 1024 // G
        device = "cuda" if torch.cuda.is_available() else "cpu"

        x = torch.randn((B, N, G * C), device=device)

        linear = PlatonicLinear(
            c_in=C, c_out=C, solid_name=solid_name, bias=bias, cache_gidx=cache_gidx, fused=fused
        ).to(device)

        for _ in range(N_warmup):
            y = linear(x)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(N_runs):
            y = linear(x)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        dt = (t1 - t0) / N_runs * 1000  # ms per run

        perc = ""
        if baseline is not None:
            perc = 100 * dt / baseline
            perc = f"  ({perc:5.2f}%)"

        print(
            f"  N_runs={N_runs},  ({B}, {N}, {G:2d}, {C}),  cache_gidx={str(cache_gidx):<5},  fused={str(fused):<5}:  {dt:7.3f}ms"
            + perc
        )

        return dt

    for solid_name in PLATONIC_GROUPS_3D:
        print(solid_name)
        baseline = None
        for fused in (False, True):
            for cache_gidx in (False, True):
                dt = benchmark(solid_name, cache_gidx=cache_gidx, fused=fused, baseline=baseline)
                if (not cache_gidx) and (not fused):
                    baseline = dt
