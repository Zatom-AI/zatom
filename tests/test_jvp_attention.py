from __future__ import annotations

import gc
import os
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import rootutils
import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor, enable_grad
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.flop_counter import FlopCounterMode

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from zatom.models.kernels.jvp_attention import JVPAttn


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


def get_flop_count(f: Callable[[], Any], display_ops: bool = False) -> int:
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


def measure_memory_usage(f: Callable[[], Any]) -> tuple[float, float]:
    """Measure GPU memory usage of a function.

    Args:
        f: The function to measure.

    Returns:
        Tuple of (allocated_mb, reserved_mb) memory in megabytes.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()

    f()

    torch.cuda.synchronize()

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    allocated_mb = (peak_allocated - initial_allocated) / (1024 * 1024)
    reserved_mb = (peak_reserved - initial_reserved) / (1024 * 1024)

    return allocated_mb, reserved_mb


def benchmark_function(
    f: Callable[[], Any], warmup_iters: int = 10, benchmark_iters: int = 100
) -> float:
    """Benchmark a function's execution time.

    Args:
        f: The function to benchmark.
        warmup_iters: Number of warmup iterations.
        benchmark_iters: Number of benchmark iterations.

    Returns:
        Average time per iteration in milliseconds.
    """
    # Warmup
    for _ in range(warmup_iters):
        f()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(benchmark_iters):
        f()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) * 1000 / benchmark_iters
    return avg_time_ms


class QKV(NamedTuple):
    """Query, Key, Value tensors."""

    q: Tensor
    k: Tensor
    v: Tensor


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for numerical validation."""

    primal_error: float
    tangent_error: float
    loss_error: float
    q_grad_error: float
    k_grad_error: float
    v_grad_error: float

    @property
    def max_error(self) -> float:
        """Return the maximum error across all metrics."""
        return max(
            self.primal_error,
            self.tangent_error,
            self.loss_error,
            self.q_grad_error,
            self.k_grad_error,
            self.v_grad_error,
        )

    def is_accurate(self, tolerance: float = 5e-3) -> bool:
        """Check if all errors are within tolerance."""
        return self.max_error < tolerance


class BenchmarkResult(NamedTuple):
    """Results from a single benchmark run."""

    seq_len: int
    is_causal: bool
    method: str  # 'sdpa' or 'jvp_attn'
    time_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    flops: int | None = None
    accuracy: AccuracyMetrics | None = None


@dataclass
class Args:
    """Training arguments."""

    bsz: int
    model_dim: int
    head_dim: int
    seq_lengths: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    warmup_iters: int = 10
    benchmark_iters: int = 100
    dtype: str = "float16"
    seed: int = 42
    validate_gradients: bool = True

    @staticmethod
    def get_parser() -> ArgumentParser:
        """Get the argument parser for training."""
        parser = ArgumentParser()
        parser.add_argument("--bsz", default=1, type=int)
        parser.add_argument("--model-dim", default=320, type=int)
        parser.add_argument("--head-dim", default=64, type=int)
        parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
        parser.add_argument("--warmup-iters", default=10, type=int)
        parser.add_argument("--benchmark-iters", default=100, type=int)
        parser.add_argument(
            "--dtype", default="float16", choices=["float16", "float32", "bfloat16"]
        )
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument(
            "--no-validate-gradients",
            action="store_true",
            help="Skip gradient validation",
        )
        return parser

    @staticmethod
    def from_namespace(namespace: Namespace) -> Args:
        """Create Args from a namespace."""
        kwargs = vars(namespace)
        validate_gradients = not kwargs.pop("no_validate_gradients", False)
        kwargs["validate_gradients"] = validate_gradients
        return Args(**kwargs)


def create_test_tensors(
    args: Args, seq_len: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, ...]:
    """Create test tensors for benchmarking.

    Returns:
        Tuple of (q_p, q_t, k_p, k_t, v_p, v_t, target) tensors.
    """
    gen = torch.Generator(device=device).manual_seed(args.seed)
    heads = args.model_dim // args.head_dim

    tensors = tuple(
        torch.randn(
            args.bsz,
            heads,
            seq_len,
            args.head_dim,
            device=device,
            dtype=dtype,
            generator=gen,
        )
        for _ in range(7)
    )

    return tensors


def loss_fn(out: Tensor, target: Tensor) -> Tensor:
    """Compute the mean squared error loss.

    Args:
        out: The output tensor.
        target: The target tensor.

    Returns:
        The mean squared error loss.
    """
    return (out - target).square().mean()


def make_qkv_with_grad(
    q_p: Tensor, k_p: Tensor, v_p: Tensor, q_t: Tensor, k_t: Tensor, v_t: Tensor
) -> QKV:
    """Make a QKV tuple with gradients enabled."""
    # Create dual tensors
    q = fwAD.make_dual(q_p, q_t)
    k = fwAD.make_dual(k_p, k_t)
    v = fwAD.make_dual(v_p, v_t)

    for t in (q, k, v):
        t.requires_grad = True
        t.retain_grad()

    return QKV(q, k, v)


def make_qkv(q_p: Tensor, k_p: Tensor, v_p: Tensor, q_t: Tensor, k_t: Tensor, v_t: Tensor) -> QKV:
    """Make a QKV tuple from the given tensors with dual numbers."""
    q = fwAD.make_dual(q_p, q_t)
    k = fwAD.make_dual(k_p, k_t)
    v = fwAD.make_dual(v_p, v_t)
    return QKV(q, k, v)


def compute_absolute_error(tensor1: Tensor, tensor2: Tensor) -> float:
    """Compute maximum absolute error between two tensors."""
    absolute_error = (tensor1 - tensor2).abs()
    return absolute_error.max().item()


def validate_accuracy_and_gradients(
    q_p: Tensor,
    k_p: Tensor,
    v_p: Tensor,
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    target: Tensor,
    is_causal: bool,
    tolerance: float = 5e-3,
) -> AccuracyMetrics:
    """Validate numerical accuracy and gradient matching between SDPA and JVP attention.

    Returns:
        AccuracyMetrics containing all error measurements.
    """
    with sdpa_kernel(SDPBackend.MATH), fwAD.dual_level(), enable_grad():
        # Run SDPA
        q0, k0, v0 = make_qkv_with_grad(
            q_p.clone(), k_p.clone(), v_p.clone(), q_t.clone(), k_t.clone(), v_t.clone()
        )

        sdpa_out = scaled_dot_product_attention(q0, k0, v0, is_causal=is_causal)
        sdpa_out.retain_grad()
        sdpa_op, sdpa_ot = fwAD.unpack_dual(sdpa_out)

        loss0 = loss_fn(sdpa_out, target)
        loss0.backward()

        # Run JVP Attention
        q1, k1, v1 = make_qkv_with_grad(
            q_p.clone(), k_p.clone(), v_p.clone(), q_t.clone(), k_t.clone(), v_t.clone()
        )

        jvp_out = JVPAttn.fwd_dual(q1, k1, v1, causal=is_causal)
        jvp_out.retain_grad()
        jvp_op, jvp_ot = fwAD.unpack_dual(jvp_out)

        loss1 = loss_fn(jvp_out, target)
        loss1.backward()

        # Compute errors
        primal_error = compute_absolute_error(jvp_op, sdpa_op)
        tangent_error = compute_absolute_error(jvp_ot, sdpa_ot)
        loss_error = compute_absolute_error(loss1, loss0)

        # Compute gradient errors
        q_grad_error = compute_absolute_error(q1.grad, q0.grad)
        k_grad_error = compute_absolute_error(k1.grad, k0.grad)
        v_grad_error = compute_absolute_error(v1.grad, v0.grad)

        metrics = AccuracyMetrics(
            primal_error=primal_error,
            tangent_error=tangent_error,
            loss_error=loss_error,
            q_grad_error=q_grad_error,
            k_grad_error=k_grad_error,
            v_grad_error=v_grad_error,
        )

        # Validate using torch.testing.assert_close
        try:
            torch.testing.assert_close(
                jvp_op, sdpa_op, atol=tolerance if is_causal else 5e-4, rtol=1e-5
            )
            torch.testing.assert_close(
                jvp_ot, sdpa_ot, atol=tolerance if is_causal else 1e-3, rtol=1e-5
            )
            torch.testing.assert_close(loss1, loss0, atol=5e-4, rtol=1e-5)

            torch.testing.assert_close(q1.grad, q0.grad, atol=5e-4, rtol=1e-5)
            torch.testing.assert_close(k1.grad, k0.grad, atol=5e-4, rtol=1e-5)
            torch.testing.assert_close(v1.grad, v0.grad, atol=5e-4, rtol=1e-5)

        except AssertionError as e:
            print(f"  ⚠️  Accuracy validation failed: {e}")

        return metrics


def run_benchmark_suite(args: Args) -> list[BenchmarkResult]:
    """Run comprehensive benchmarks across different configurations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    results = []

    for seq_len in args.seq_lengths:
        print(f"\n{'='*60}")
        print(f"Benchmarking sequence length: {seq_len}")
        print(f"{'='*60}")

        # Create test tensors
        q_p, q_t, k_p, k_t, v_p, v_t, target = create_test_tensors(args, seq_len, device, dtype)

        for is_causal in [False, True]:
            print(f"\nCausal: {is_causal}")
            print("-" * 40)

            # Validate accuracy and gradients first
            if args.validate_gradients:
                print("Validating accuracy and gradients...")
                accuracy_metrics = validate_accuracy_and_gradients(
                    q_p, k_p, v_p, q_t, k_t, v_t, target, is_causal
                )

                print(f"  Primal error: {accuracy_metrics.primal_error:.2e}")
                print(f"  Tangent error: {accuracy_metrics.tangent_error:.2e}")
                print(f"  Loss error: {accuracy_metrics.loss_error:.2e}")
                print(f"  Q gradient error: {accuracy_metrics.q_grad_error:.2e}")
                print(f"  K gradient error: {accuracy_metrics.k_grad_error:.2e}")
                print(f"  V gradient error: {accuracy_metrics.v_grad_error:.2e}")

                if accuracy_metrics.is_accurate():
                    print("  ✓ All accuracy checks passed!")
                else:
                    print(f"  ⚠️  Max error {accuracy_metrics.max_error:.2e} exceeds tolerance")
            else:
                accuracy_metrics = None

            # Benchmark performance
            with sdpa_kernel(SDPBackend.MATH), fwAD.dual_level(), enable_grad():
                # Create functions for benchmarking
                def run_sdpa():
                    q, k, v = make_qkv(
                        q_p.clone(),
                        k_p.clone(),
                        v_p.clone(),
                        q_t.clone(),
                        k_t.clone(),
                        v_t.clone(),
                    )
                    out = scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                    return out

                def run_jvp_attn():
                    q, k, v = make_qkv(
                        q_p.clone(),
                        k_p.clone(),
                        v_p.clone(),
                        q_t.clone(),
                        k_t.clone(),
                        v_t.clone(),
                    )
                    out = JVPAttn.fwd_dual(q, k, v, causal=is_causal)
                    return out

                # Measure SDPA performance
                print("\nBenchmarking performance...")
                sdpa_time = benchmark_function(run_sdpa, args.warmup_iters, args.benchmark_iters)
                sdpa_mem_alloc, sdpa_mem_reserved = measure_memory_usage(run_sdpa)
                sdpa_flops = get_flop_count(run_sdpa, display_ops=False)

                # Measure JVP Attention performance
                jvp_time = benchmark_function(
                    run_jvp_attn, args.warmup_iters, args.benchmark_iters
                )
                jvp_mem_alloc, jvp_mem_reserved = measure_memory_usage(run_jvp_attn)
                jvp_flops = get_flop_count(run_jvp_attn, display_ops=False)

                # Store results
                results.append(
                    BenchmarkResult(
                        seq_len=seq_len,
                        is_causal=is_causal,
                        method="sdpa",
                        time_ms=sdpa_time,
                        memory_allocated_mb=sdpa_mem_alloc,
                        memory_reserved_mb=sdpa_mem_reserved,
                        flops=sdpa_flops,
                        accuracy=None,
                    )
                )

                results.append(
                    BenchmarkResult(
                        seq_len=seq_len,
                        is_causal=is_causal,
                        method="jvp_attn",
                        time_ms=jvp_time,
                        memory_allocated_mb=jvp_mem_alloc,
                        memory_reserved_mb=jvp_mem_reserved,
                        flops=jvp_flops,
                        accuracy=accuracy_metrics,
                    )
                )

                # Print results
                print("PyTorch SDPA:")
                print(f"  Time: {sdpa_time:.3f} ms")
                print(
                    f"  Memory (alloc/reserved): {sdpa_mem_alloc:.2f}/{sdpa_mem_reserved:.2f} MB"
                )
                print(f"  FLOPS: {fmt_flops(mpi_to_flops(sdpa_time, sdpa_flops))}")

                print("\nJVP Attention:")
                print(f"  Time: {jvp_time:.3f} ms")
                print(f"  Memory (alloc/reserved): {jvp_mem_alloc:.2f}/{jvp_mem_reserved:.2f} MB")
                print(f"  FLOPS: {fmt_flops(mpi_to_flops(jvp_time, jvp_flops))}")

                print(f"\nSpeedup: {sdpa_time/jvp_time:.2f}x")
                print(f"Memory ratio: {jvp_mem_alloc/sdpa_mem_alloc:.2f}x")

    return results


def print_summary_table(results: list[BenchmarkResult]) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)

    # Group results by seq_len and causal
    from collections import defaultdict

    grouped = defaultdict(dict)

    for r in results:
        key = (r.seq_len, r.is_causal)
        grouped[key][r.method] = r

    # Print header
    print(
        f"{'Seq Len':<10} {'Causal':<8} {'Method':<10} "
        f"{'Time (ms)':<12} {'Mem (MB)':<12} {'TFLOP/s':<12} "
        f"{'Max Error':<12} {'Grad Check':<10}"
    )
    print("-" * 90)

    for (seq_len, is_causal), methods in sorted(grouped.items()):
        for method in ["sdpa", "jvp_attn"]:
            if method in methods:
                r = methods[method]
                flops_str = fmt_flops(mpi_to_flops(r.time_ms, r.flops)) if r.flops else "N/A"

                if r.accuracy:
                    error_str = f"{r.accuracy.max_error:.2e}"
                    grad_check = "✓" if r.accuracy.is_accurate() else "✗"
                else:
                    error_str = "baseline"
                    grad_check = "N/A"

                print(
                    f"{seq_len:<10} {str(is_causal):<8} {method:<10} "
                    f"{r.time_ms:<12.3f} {r.memory_allocated_mb:<12.2f} "
                    f"{flops_str:<12} {error_str:<12} {grad_check:<10}"
                )
        print()


def main(args: Args) -> None:
    """Main benchmarking loop."""
    print("Flash Attention JVP Kernel Benchmark")
    print(
        f"Configuration: bsz={args.bsz}, model_dim={args.model_dim}, "
        f"head_dim={args.head_dim}, dtype={args.dtype}"
    )
    print(f"Gradient validation: {'Enabled' if args.validate_gradients else 'Disabled'}")

    results = run_benchmark_suite(args)
    print_summary_table(results)

    # Optional: Save results to file
    import json

    # Convert results to JSON-serializable format
    results_data = []
    for r in results:
        result_dict = r._asdict()
        if r.accuracy:
            result_dict["accuracy"] = {
                "primal_error": r.accuracy.primal_error,
                "tangent_error": r.accuracy.tangent_error,
                "loss_error": r.accuracy.loss_error,
                "q_grad_error": r.accuracy.q_grad_error,
                "k_grad_error": r.accuracy.k_grad_error,
                "v_grad_error": r.accuracy.v_grad_error,
                "max_error": r.accuracy.max_error,
                "is_accurate": r.accuracy.is_accurate(),
            }
        results_data.append(result_dict)

    output_filepath = os.path.join("tests", "benchmark_results.json")
    with open(output_filepath, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_filepath}")


if __name__ == "__main__":
    parser = Args.get_parser()
    namespace = parser.parse_args()
    args = Args.from_namespace(namespace)
    main(args)
