from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple

import rootutils
import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor, enable_grad
from torch.nn import MSELoss
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.flop_counter import FlopCounterMode

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from zatom.models.kernels.flash_attention import JVPAttn


def mpi_to_flops(ms_per_iter: float, flop_count: int) -> float:
    """Convert milliseconds per iteration to FLOPS.

    Args:
        ms_per_iter: Milliseconds per iteration.
        flop_count: Number of floating point operations.

    Returns:
        The number of FLOPS.
    """
    iters_per_second = 1e3 / ms_per_iter
    return iters_per_second * flop_count


def fmt_flops(flops: int) -> str:
    """Return a string representation of FLOPS in TFLOP/s."""
    return f"{flops / 1e12:5.1f} TFLOP/s"


# Python *please* bring back support for generic NamedTuples
def get_flop_count(f: Callable[[], Any], display_ops=True) -> int:
    """Get the number of floating point operations (FLOPs) for a given function.

    Args:
        f: The function to measure FLOPs for.
        display_ops: Whether to display the FLOPs during measurement.

    Returns:
        The number of FLOPs.
    """
    flop_counter = FlopCounterMode(display=display_ops)
    with flop_counter:
        f()
    return flop_counter.get_total_flops()


class QKV(NamedTuple):
    """Query, Key, Value tensors."""

    q: Tensor
    k: Tensor
    v: Tensor


class UnpackedDualQKV(NamedTuple):
    """Unpacked dual Query, Key, Value tensors."""

    primal: QKV
    tangent: QKV


@dataclass
class Args:
    """Training arguments."""

    bsz: int
    model_dim: int
    head_dim: int
    seq_len: int

    @staticmethod
    def get_parser() -> ArgumentParser:
        """Get the argument parser for training."""
        parser = ArgumentParser()
        parser.add_argument("--bsz", default=1, type=int)
        parser.add_argument("--model-dim", default=320, type=int)
        parser.add_argument("--head-dim", default=64, type=int)
        parser.add_argument("--seq-len", default=128, type=int)
        return parser

    @staticmethod
    def from_namespace(namespace: Namespace) -> Args:
        """Create Args from a namespace."""
        args = Args(**vars(namespace))
        return args


def main(args: Args) -> None:
    """Main training loop."""
    device = torch.device("cuda")
    dtype = torch.float16
    seed = 42
    gen = torch.Generator(device=device)

    heads = args.model_dim // args.head_dim
    q_p, q_t, k_p, k_t, v_p, v_t, target = (
        torch.randn(
            args.bsz,
            heads,
            args.seq_len,
            args.head_dim,
            device=device,
            dtype=dtype,
            generator=gen.manual_seed(seed + ix),
        )
        for ix in range(7)
    )
    # for t in (q_p, k_p, v_p):
    #     t.requires_grad = True
    #     t.retain_grad()

    # NOTE: MSELoss only works for torch.func.jvp(), if we use MSELoss with fwAD invocation, we get the following error:
    # `ZeroTensors are immutable. Please use the materialized zero tensor obtained using .clone() if you want a mutable tensor.`
    def loss_fn(out: Tensor, target: Tensor) -> Tensor:
        """Compute the mean squared error loss.

        Args:
            out: The output tensor.
            target: The target tensor.

        Returns:
            The mean squared error loss.
        """
        return (out - target).square().mean()

    def gimme_grads(t: Tensor) -> Tensor:
        """Get a tensor with gradients enabled.

        Args:
            t: The input tensor.

        Returns:
            A tensor with gradients enabled.
        """
        t.requires_grad = True
        t.retain_grad()
        return t

    def make_qkv(
        q_p: Tensor, k_p: Tensor, v_p: Tensor, q_t: Tensor, k_t: Tensor, v_t: Tensor
    ) -> QKV:
        """Make a QKV tuple from the given tensors.

        Args:
            q_p: The query projection tensor.
            k_p: The key projection tensor.
            v_p: The value projection tensor.
            q_t: The query tangent tensor.
            k_t: The key tangent tensor.
            v_t: The value tangent tensor.

        Returns:
            A QKV tuple containing the query, key, and value tensors.
        """
        return QKV(
            q=gimme_grads(fwAD.make_dual(q_p, q_t)),
            k=gimme_grads(fwAD.make_dual(k_p, k_t)),
            v=gimme_grads(fwAD.make_dual(v_p, v_t)),
        )

    def make_qkv_unpacked(
        q_p: Tensor, k_p: Tensor, v_p: Tensor, q_t: Tensor, k_t: Tensor, v_t: Tensor
    ) -> UnpackedDualQKV:
        """Make an unpacked dual QKV from the given tensors.

        Args:
            q_p: The query projection tensor.
            k_p: The key projection tensor.
            v_p: The value projection tensor.
            q_t: The query tangent tensor.
            k_t: The key tangent tensor.
            v_t: The value tangent tensor.

        Returns:
            An unpacked dual QKV containing the primal and tangent QKV tensors.
        """
        return UnpackedDualQKV(
            primal=QKV(
                q=gimme_grads(q_p),
                k=gimme_grads(k_p),
                v=gimme_grads(v_p),
            ),
            tangent=QKV(
                q=q_t,
                k=k_t,
                v=v_t,
            ),
        )

    for is_causal in (False, True):
        with sdpa_kernel(SDPBackend.MATH), fwAD.dual_level(), enable_grad():
            q0, k0, v0 = make_qkv(
                q_p.clone(),
                k_p.clone(),
                v_p.clone(),
                q_t.clone(),
                k_t.clone(),
                v_t.clone(),
            )

            sdpa_out = scaled_dot_product_attention(q0, k0, v0, is_causal=is_causal)
            sdpa_out.retain_grad()
            sdpa_op, sdpa_ot = fwAD.unpack_dual(sdpa_out)

            loss0: Tensor = loss_fn(sdpa_out, target)
            loss0.backward()

            q1, k1, v1 = make_qkv(
                q_p.clone(),
                k_p.clone(),
                v_p.clone(),
                q_t.clone(),
                k_t.clone(),
                v_t.clone(),
            )

            dual_out = JVPAttn.fwd_dual(q1, k1, v1, causal=is_causal)
            dual_out.retain_grad()
            dual_op, dual_ot = fwAD.unpack_dual(dual_out)

            torch.testing.assert_close(
                dual_op, sdpa_op, atol=5e-3 if is_causal else 5e-4, rtol=1e-5
            )
            # TODO: Improve this accuracy
            torch.testing.assert_close(
                dual_ot, sdpa_ot, atol=5e-3 if is_causal else 1e-3, rtol=1e-5
            )

            loss1: Tensor = loss_fn(dual_out, target)
            torch.testing.assert_close(loss1, loss0, atol=5e-4, rtol=1e-5)
            loss1.backward()
            torch.testing.assert_close(q1.grad, q0.grad, atol=5e-4, rtol=1e-5)
            torch.testing.assert_close(k1.grad, k0.grad, atol=5e-4, rtol=1e-5)
            torch.testing.assert_close(v1.grad, v0.grad, atol=5e-4, rtol=1e-5)

        mse_fn = MSELoss()
        with enable_grad():
            qkv_p, qkv_t = make_qkv_unpacked(
                q_p.clone(),
                k_p.clone(),
                v_p.clone(),
                q_t.clone(),
                k_t.clone(),
                v_t.clone(),
            )
            j_p: Tensor
            j_t: Tensor
            j_p, j_t = torch.func.jvp(partial(JVPAttn.fwd_dual, causal=is_causal), qkv_p, qkv_t)
            j_p.retain_grad()
            loss2: Tensor = mse_fn(j_p, target)
            torch.testing.assert_close(loss2, loss0, atol=5e-4, rtol=1e-5)
            loss2.backward()
            torch.testing.assert_close(j_p, sdpa_op, atol=1e-3 if is_causal else 5e-4, rtol=1e-5)
            # TODO: Improve this accuracy
            torch.testing.assert_close(j_t, sdpa_ot, atol=5e-3 if is_causal else 1e-3, rtol=1e-5)
            q2, k2, v2 = qkv_p
            torch.testing.assert_close(q2.grad, q0.grad, atol=5e-4, rtol=1e-5)
            torch.testing.assert_close(k2.grad, k0.grad, atol=5e-4, rtol=1e-5)
            torch.testing.assert_close(v2.grad, v0.grad, atol=5e-4, rtol=1e-5)

        print(f"Passed all assertions with `is_causal={is_causal}`.")


if __name__ == "__main__":
    parser = Args.get_parser()
    args_untyped: Namespace = parser.parse_args()
    args: Args = Args.from_namespace(args_untyped)
    main(args)
