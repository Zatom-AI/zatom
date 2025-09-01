import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from zatom.models.kernels import triton_attention

torch.backends.cuda.matmul.allow_tf32 = True


def time_fn(fn, iters=10):
    """Times a function over a number of iterations.

    Args:
        fn: The function to time.
        iters: The number of iterations to run.
    """
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def benchmark_scaling(
    B=4, H=8, D=64, dtype=torch.float16, causal=True, Ns=[128, 256, 512, 1024, 2048]
):
    """Benchmark scaling of attention mechanisms.

    Args:
        B: Batch size.
        H: Number of heads.
        D: Dimensionality of the model.
        dtype: Data type of the tensors.
        causal: Whether to use causal attention.
        Ns: List of sequence lengths to test.
    """
    fwd_times_torch = []
    fwd_times_triton = []
    fwbw_times_torch = []
    fwbw_times_triton = []

    device = "cuda"

    for N in Ns:
        print(f"\n--- N={N} ---")
        Q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

        # Warmup
        for _ in range(3):
            F.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=causal
            )
            triton_attention(Q, K, V, mask=None, causal=causal, dropout_p=0.0)

        # Forward timing
        t_torch_fwd = time_fn(
            lambda: F.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=causal
            )
        )
        t_triton_fwd = time_fn(
            lambda: triton_attention(Q, K, V, mask=None, causal=causal, dropout_p=0.0)
        )

        # Forward+Backward timing
        def torch_fwbw():
            """Benchmark PyTorch SDPA (fwd+bwd)"""
            out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=causal
            )
            loss = out.sum()
            loss.backward(retain_graph=True)

        def triton_fwbw():
            """Benchmark Triton FA (fwd+bwd)"""
            out = triton_attention(Q, K, V, mask=None, causal=causal, dropout_p=0.0)
            loss = out.sum()
            loss.backward(retain_graph=True)

        t_torch_fwbw = time_fn(torch_fwbw)
        t_triton_fwbw = time_fn(triton_fwbw)

        fwd_times_torch.append(t_torch_fwd * 1e3)
        fwd_times_triton.append(t_triton_fwd * 1e3)
        fwbw_times_torch.append(t_torch_fwbw * 1e3)
        fwbw_times_triton.append(t_triton_fwbw * 1e3)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(Ns, fwd_times_torch, label="PyTorch SDPA (fwd)", marker="o")
    plt.plot(Ns, fwd_times_triton, label="Triton FA (fwd, auto-tuned)", marker="o")
    plt.xlabel("Sequence length N")
    plt.ylabel("Forward time (ms)")
    plt.title(f"Forward Scaling B={B}, H={H}, D={D}, dtype={dtype}, causal={causal}")
    plt.legend()
    plt.grid(True)
    plt.savefig("forward_scaling.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(Ns, fwbw_times_torch, label="PyTorch SDPA (fwd+bwd)", marker="o")
    plt.plot(Ns, fwbw_times_triton, label="Triton FA (fwd+bwd, auto-tuned)", marker="o")
    plt.xlabel("Sequence length N")
    plt.ylabel("Forward+Backward time (ms)")
    plt.title(f"Fwd+Bwd Scaling B={B}, H={H}, D={D}, dtype={dtype}, causal={causal}")
    plt.legend()
    plt.grid(True)
    plt.savefig("fwbw_scaling.png", dpi=150)
    plt.show()


def benchmark_second_order(B=2, H=4, N=256, D=64, dtype=torch.float16, causal=True, iters=5):
    """Benchmark second-order gradients for attention mechanisms.

    Args:
        B: Batch size.
        H: Number of heads.
        N: Sequence length.
        D: Dimensionality of the model.
        dtype: Data type of the tensors.
        causal: Whether to use causal attention.
        iters: Number of iterations to run.
    """
    device = "cuda"
    torch.manual_seed(0)

    Q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

    # Torch SDPA second-order
    def torch_2nd():
        """Benchmark PyTorch SDPA (2nd order)"""
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=causal
            )
        loss = out.sum()
        grad1 = torch.autograd.grad(loss, (Q, K, V), create_graph=True)
        grad2 = torch.autograd.grad(sum([g.sum() for g in grad1]), (Q, K, V), retain_graph=True)

    # Triton FA second-order
    def triton_2nd():
        """Benchmark Triton FA (2nd order)"""
        out = triton_attention(Q, K, V, mask=None, causal=causal, dropout_p=0.0)
        loss = out.sum()
        grad1 = torch.autograd.grad(loss, (Q, K, V), create_graph=True)
        grad2 = torch.autograd.grad(sum([g.sum() for g in grad1]), (Q, K, V), retain_graph=True)

    # Warmup
    for _ in range(2):
        torch_2nd()
        triton_2nd()

    # Timing
    torch_time = time_fn(torch_2nd, iters=iters)
    triton_time = time_fn(triton_2nd, iters=iters)

    print(f"Second-order grad time (PyTorch SDPA): {torch_time*1e3:.3f} ms")
    print(f"Second-order grad time (Triton FA, auto-tuned):    {triton_time*1e3:.3f} ms")


if __name__ == "__main__":
    # Scaling profile
    benchmark_scaling(B=2, H=4, D=64, dtype=torch.float16, causal=True, Ns=[128, 256, 512, 1024])

    # Second-order grad benchmark
    benchmark_second_order(B=2, H=4, N=256, D=64, dtype=torch.float16, causal=True)

    # Example outputs
    # Second-order grad time (PyTorch SDPA): 42.315 ms
    # Second-order grad time (Triton FA, auto-tuned):    17.842 ms
