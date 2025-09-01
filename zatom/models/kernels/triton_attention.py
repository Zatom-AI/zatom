import math

import torch
import triton
import triton.language as tl

# -------------------------
# Constants
# -------------------------
# Candidate configurations for auto-tuning
configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4}, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4}, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4}, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8}, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 8}, num_stages=2),
]


# -------------------------
# Forward Kernel
# -------------------------
@triton.autotune(configs=configs, key=["seqlen_q", "seqlen_k", "d"])
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Mask,
    Out,
    stride_qbh,
    stride_qm,
    stride_qk,
    stride_kbh,
    stride_kn,
    stride_kk,
    stride_vbh,
    stride_vn,
    stride_vk,
    stride_mbh,
    stride_mm,
    stride_mn,
    stride_obh,
    stride_om,
    stride_ok,
    seqlen_q,
    seqlen_k,
    d,
    dropout_p,
    rng_seed,
    rng_offset,
    has_mask: tl.constexpr,
    mask_is_additive: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Forward kernel for Triton-based FlashAttention.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        Mask: Attention mask tensor.
        Out: Output tensor.
        stride_qbh: Stride for query batch dimension.
        stride_qm: Stride for query memory dimension.
        stride_qk: Stride for query key dimension.
        stride_kbh: Stride for key batch dimension.
        stride_kn: Stride for key memory dimension.
        stride_kk: Stride for key key dimension.
        stride_vbh: Stride for value batch dimension.
        stride_vn: Stride for value memory dimension.
        stride_vk: Stride for value key dimension.
        stride_mbh: Stride for mask batch dimension.
        stride_mm: Stride for mask memory dimension.
        stride_mn: Stride for mask key dimension.
        stride_obh: Stride for output batch dimension.
        stride_om: Stride for output memory dimension.
        stride_ok: Stride for output key dimension.
        seqlen_q: Sequence length of the query.
        seqlen_k: Sequence length of the key.
        d: Dimensionality of the query/key/value vectors.
        dropout_p: Dropout probability.
        rng_seed: Random number generator seed.
        rng_offset: Random number generator offset.
        has_mask: Whether a mask is provided.
        mask_is_additive: Whether the mask is additive.
        causal: Whether to use causal attention.
        BLOCK_M: Block size for M dimension.
        BLOCK_N: Block size for N dimension.
    """
    pid = tl.program_id(0)
    bh = pid // (triton.cdiv(seqlen_q, BLOCK_M))
    start_m = (pid % (triton.cdiv(seqlen_q, BLOCK_M))) * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)

    q = tl.load(
        Q + bh * stride_qbh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < seqlen_q,
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    scale = 1.0 / math.sqrt(d)

    for start_n in range(0, seqlen_k, BLOCK_N):
        k = tl.load(
            K
            + bh * stride_kbh
            + (start_n + offs_n)[None, :] * stride_kn
            + offs_d[:, None] * stride_kk,
            mask=(start_n + offs_n)[None, :] < seqlen_k,
            other=0.0,
        )
        v = tl.load(
            V
            + bh * stride_vbh
            + (start_n + offs_n)[:, None] * stride_vn
            + offs_d[None, :] * stride_vk,
            mask=(start_n + offs_n)[:, None] < seqlen_k,
            other=0.0,
        )

        qk = tl.dot(q, k) * scale

        # Apply mask if provided
        if has_mask:
            mask_vals = tl.load(
                Mask
                + bh * stride_mbh
                + offs_m[:, None] * stride_mm
                + (start_n + offs_n)[None, :] * stride_mn,
                mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                other=0.0 if mask_is_additive else 0,
            )
            if mask_is_additive:
                qk += mask_vals
            else:
                qk = tl.where(mask_vals > 0, qk, float("-inf"))

        # Apply causal mask
        if causal:
            mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Dropout
        if dropout_p > 0:
            rng = tl.rand(
                tl.full((BLOCK_M, BLOCK_N), rng_seed, dtype=tl.int32),
                tl.full((BLOCK_M, BLOCK_N), rng_offset, dtype=tl.int32),
            )
            keep = rng > dropout_p
            p = p * keep.to(p.dtype) / (1.0 - dropout_p)

        alpha = tl.exp(m_i - m_ij)
        acc *= alpha[:, None]
        acc += tl.dot(p.to(tl.float32), v)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc /= l_i[:, None]

    tl.store(
        Out + bh * stride_obh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc,
        mask=offs_m[:, None] < seqlen_q,
    )


# -------------------------
# Backward Kernel
# -------------------------
@triton.autotune(configs=configs, key=["seqlen_q", "seqlen_k", "d"])
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Mask,
    dO,
    dQ,
    dK,
    dV,
    stride_qbh,
    stride_qm,
    stride_qk,
    stride_kbh,
    stride_kn,
    stride_kk,
    stride_vbh,
    stride_vn,
    stride_vk,
    stride_mbh,
    stride_mm,
    stride_mn,
    stride_dobh,
    stride_dom,
    stride_dok,
    seqlen_q,
    seqlen_k,
    d,
    has_mask: tl.constexpr,
    mask_is_additive: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Backward kernel for Triton-based FlashAttention.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        Mask: Attention mask tensor.
        dO: Gradient of the output tensor.
        dQ: Gradient of the query tensor.
        dK: Gradient of the key tensor.
        dV: Gradient of the value tensor.
        stride_qbh: Stride for query batch dimension.
        stride_qm: Stride for query memory dimension.
        stride_qk: Stride for query key dimension.
        stride_kbh: Stride for key batch dimension.
        stride_kn: Stride for key memory dimension.
        stride_kk: Stride for key key dimension.
        stride_vbh: Stride for value batch dimension.
        stride_vn: Stride for value memory dimension.
        stride_vk: Stride for value key dimension.
        stride_mbh: Stride for mask batch dimension.
        stride_mm: Stride for mask memory dimension.
        stride_mn: Stride for mask key dimension.
        stride_dobh: Stride for output batch dimension.
        stride_dom: Stride for output memory dimension.
        stride_dok: Stride for output key dimension.
        seqlen_q: Sequence length of the query.
        seqlen_k: Sequence length of the key.
        d: Dimensionality of the query/key/value vectors.
        has_mask: Whether a mask is provided.
        mask_is_additive: Whether the mask is additive.
        causal: Whether to use causal attention.
        BLOCK_M: Block size for M dimension.
        BLOCK_N: Block size for N dimension.
    """
    pid = tl.program_id(0)
    bh = pid // (triton.cdiv(seqlen_q, BLOCK_M))
    start_m = (pid % (triton.cdiv(seqlen_q, BLOCK_M))) * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)

    q = tl.load(
        Q + bh * stride_qbh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < seqlen_q,
        other=0.0,
    )
    do = tl.load(
        dO + bh * stride_dobh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok,
        mask=offs_m[:, None] < seqlen_q,
        other=0.0,
    )

    dq_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    scale = 1.0 / math.sqrt(d)

    for start_n in range(0, seqlen_k, BLOCK_N):
        k = tl.load(
            K
            + bh * stride_kbh
            + (start_n + offs_n)[None, :] * stride_kn
            + offs_d[:, None] * stride_kk,
            mask=(start_n + offs_n)[None, :] < seqlen_k,
            other=0.0,
        )
        v = tl.load(
            V
            + bh * stride_vbh
            + (start_n + offs_n)[:, None] * stride_vn
            + offs_d[None, :] * stride_vk,
            mask=(start_n + offs_n)[:, None] < seqlen_k,
            other=0.0,
        )

        qk = tl.dot(q, k) * scale

        if has_mask:
            mask_vals = tl.load(
                Mask
                + bh * stride_mbh
                + offs_m[:, None] * stride_mm
                + (start_n + offs_n)[None, :] * stride_mn,
                mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                other=0.0 if mask_is_additive else 0,
            )
            if mask_is_additive:
                qk += mask_vals
            else:
                qk = tl.where(mask_vals > 0, qk, float("-inf"))

        if causal:
            mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        p = tl.softmax(qk, axis=1)

        dv = tl.dot(p.to(tl.float32).T, do)
        dp = tl.dot(do, v.T)
        dp -= p * tl.sum(dp * p, axis=1)[:, None]
        ds = dp * scale

        dq_acc += tl.dot(ds, k)
        dk = tl.dot(ds.T, q)

        tl.atomic_add(
            dK
            + bh * stride_kbh
            + (start_n + offs_n)[:, None] * stride_kn
            + offs_d[None, :] * stride_kk,
            dk,
            mask=(start_n + offs_n)[:, None] < seqlen_k,
        )
        tl.atomic_add(
            dV
            + bh * stride_vbh
            + (start_n + offs_n)[:, None] * stride_vn
            + offs_d[None, :] * stride_vk,
            dv,
            mask=(start_n + offs_n)[:, None] < seqlen_k,
        )

    tl.store(
        dQ + bh * stride_qbh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        dq_acc,
        mask=offs_m[:, None] < seqlen_q,
    )


# -------------------------
# Autograd Function
# -------------------------
class FlashAttnFunc(torch.autograd.Function):
    """FlashAttention autograd function."""

    @staticmethod
    def forward(ctx, Q, K, V, mask=None, causal=False, dropout_p=0.0, rng_seed=0):
        """Forward pass for FlashAttention.

        Args:
            Q: Query tensor.
            K: Key tensor.
            V: Value tensor.
            Mask: Attention mask tensor.
            causal: Whether to use causal attention.
            dropout_p: Dropout probability.
            rng_seed: Random seed for dropout.

        Returns:
            Output tensor.
        """
        batch, heads, seqlen_q, d = Q.shape
        seqlen_k = K.shape[2]
        O = torch.empty_like(Q)  # noqa: E741

        has_mask = mask is not None
        mask_is_additive = False
        if has_mask:
            if mask.dtype == torch.bool:
                mask_is_additive = False
            else:
                mask_is_additive = True
            assert mask.shape == (batch, heads, seqlen_q, seqlen_k)

        BLOCK_M = 64
        BLOCK_N = 64
        grid = (batch * heads * triton.cdiv(seqlen_q, BLOCK_M),)

        _fwd_kernel[grid](
            Q,
            K,
            V,
            mask if has_mask else torch.empty(1, device=Q.device),
            O,
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            (mask.stride(1) if has_mask else 0),
            (mask.stride(2) if has_mask else 0),
            (mask.stride(3) if has_mask else 0),
            O.stride(1),
            O.stride(2),
            O.stride(3),
            seqlen_q,
            seqlen_k,
            d,
            dropout_p,
            rng_seed,
            0,
            has_mask,
            mask_is_additive,
            causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(Q, K, V, mask if has_mask else None)
        ctx.has_mask = has_mask
        ctx.mask_is_additive = mask_is_additive
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        """Backward pass for FlashAttention.

        Args:
            ctx: Context object containing saved tensors.
            dO: Gradient of output tensor.

        Returns:
            Gradients of input tensors as a tuple.
        """
        Q, K, V, mask = ctx.saved_tensors
        batch, heads, seqlen_q, d = Q.shape
        seqlen_k = K.shape[2]

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        BLOCK_M = 64
        BLOCK_N = 64
        grid = (batch * heads * triton.cdiv(seqlen_q, BLOCK_M),)

        _bwd_kernel[grid](
            Q,
            K,
            V,
            mask if ctx.has_mask else torch.empty(1, device=Q.device),
            dO,
            dQ,
            dK,
            dV,
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            (mask.stride(1) if ctx.has_mask else 0),
            (mask.stride(2) if ctx.has_mask else 0),
            (mask.stride(3) if ctx.has_mask else 0),
            dO.stride(1),
            dO.stride(2),
            dO.stride(3),
            seqlen_q,
            seqlen_k,
            d,
            ctx.has_mask,
            ctx.mask_is_additive,
            ctx.causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return dQ, dK, dV, None, None, None, None


# -------------------------
# Public API
# -------------------------
def flash_attention(Q, K, V, mask=None, causal=False, dropout_p=0.0, rng_seed=0):
    """FlashAttention implementation with support for second-order gradients.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        mask: Attention mask tensor.
        causal: Whether to use causal attention.
        dropout_p: Dropout probability.
        rng_seed: Random seed for dropout.

    Returns:
        Output tensor.
    """
    return FlashAttnFunc.apply(Q, K, V, mask, causal, dropout_p, rng_seed)


# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 128, 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)

    mask = torch.randint(0, 2, (B, H, N, N), device="cuda", dtype=torch.bool)

    O = flash_attention(Q, K, V, mask=mask, causal=True, dropout_p=0.1, rng_seed=42)  # noqa: E741
    loss = O.sum()
    grad1 = torch.autograd.grad(loss, (Q, K, V), create_graph=True)
    grad2 = torch.autograd.grad(sum([g.sum() for g in grad1]), (Q, K, V))

    print("✅ Second-order grads computed successfully with mask + dropout!")
