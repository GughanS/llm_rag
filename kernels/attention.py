"""
Triton fused block-tiled attention kernel (FlashAttention).

This implements exact attention with a causal mask using standard 1D pointer arithmetic
for maximum compatibility across Triton versions (avoiding make_block_ptr bugs).
"""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Calculate base pointers
    qvk_offset = off_hz * stride_qh
    q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + qvk_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptrs = V + qvk_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # Initialize m_i, l_i, acc
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    qk_scale = sm_scale * 1.44269504  # log2(e)

    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K and V
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[None, :] < N_CTX, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        
        # Q * K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale
        
        # Causal mask
        if STAGE == 1:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))
            
        # m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        
        # scale p and acc
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        
        # update acc
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        
        l_i = l_i_new
        m_i = m_i_new

    # Epilogue
    m_i += tl.math.log2(l_i)
    acc = acc.to(tl.float16)
    
    # Store out
    out_ptrs = Out + qvk_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)
    
    # Store m
    off_hz_m = off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)


@triton.jit
def _bwd_preprocess(
    Out, DO, Delta,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_doz, stride_doh, stride_dom, stride_don,
    N_CTX, BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    
    offs_d = tl.arange(0, D_HEAD)
    
    o_ptrs = Out + off_hz * stride_oh + off_m[:, None] * stride_om + offs_d[None, :] * stride_on
    do_ptrs = DO + off_hz * stride_doh + off_m[:, None] * stride_dom + offs_d[None, :] * stride_don
    
    o = tl.load(o_ptrs, mask=off_m[:, None] < N_CTX, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=off_m[:, None] < N_CTX, other=0.0).to(tl.float32)
    
    delta = tl.sum(o * do, axis=1)
    
    delta_ptrs = Delta + off_hz * N_CTX + off_m
    tl.store(delta_ptrs, delta, mask=off_m < N_CTX)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO, DQ, DK, DV, M, Delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    qvk_offset = off_hz * stride_qh
    
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    k_ptrs = K + qvk_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + qvk_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
    
    dk_accum = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv_accum = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    for start_m in range(start_n * BLOCK_N // BLOCK_M, tl.cdiv(N_CTX, BLOCK_M)):
        offs_m_curr = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        
        q_ptrs = Q + qvk_offset + offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX, other=0.0)
        
        do_ptrs = DO + qvk_offset + offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_curr = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX, other=0.0)
        
        # qk
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        
        # mask
        mask = offs_m_curr[:, None] >= offs_n[None, :]
        qk = tl.where(mask, qk, float("-inf"))
        
        m_ptrs = M + off_hz * N_CTX + offs_m_curr
        m = tl.load(m_ptrs, mask=offs_m_curr < N_CTX, other=0.0)
        
        delta_ptrs = Delta + off_hz * N_CTX + offs_m_curr
        delta = tl.load(delta_ptrs, mask=offs_m_curr < N_CTX, other=0.0)
        
        p = tl.math.exp2(qk - m[:, None] * 1.44269504)
        
        # dv
        p_tensor = p.to(tl.float16)
        dv_accum += tl.dot(tl.trans(p_tensor), do_curr)
        
        # dp
        dp = tl.dot(do_curr, tl.trans(v))
        
        # ds
        ds = p * (dp - delta[:, None]) * sm_scale
        ds_tensor = ds.to(tl.float16)
        
        # dk
        dk_accum += tl.dot(tl.trans(ds_tensor), q)
        
        # dq
        dq_ptrs = DQ + qvk_offset + offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qk
        
        # We need atomic add here if multiple blocks update same DQ
        # But this loop calculates DQ sequentially for this block! No wait.
        # DQ is updated across different N blocks. This is a race condition if not atomic.
        # But Triton block parallelizes over N. Multiple N-blocks will update the same M-block of DQ.
        # So we MUST use tl.atomic_add.
        # Since this is a simple educational implementation, let's just use tl.atomic_add for dq.
        dq_update = tl.dot(ds_tensor, k)
        tl.atomic_add(dq_ptrs, dq_update, mask=offs_m_curr[:, None] < N_CTX)
        
    dk_ptrs = DK + qvk_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    dv_ptrs = DV + qvk_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    tl.store(dk_ptrs, dk_accum.to(tl.float16), mask=offs_n[:, None] < N_CTX)
    tl.store(dv_ptrs, dv_accum.to(tl.float16), mask=offs_n[:, None] < N_CTX)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        BLOCK_DMODEL = q.shape[-1]
        assert BLOCK_DMODEL in {16, 32, 64, 128}
        
        batch, nheads, seqlen, d_head = q.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        
        out = torch.empty_like(q)
        M = torch.empty((batch, nheads, seqlen), device=q.device, dtype=torch.float32)
        
        BLOCK_M = 64
        BLOCK_N = 64
        
        grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
        
        _attn_fwd[grid](
            q, k, v, sm_scale, M, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch, nheads, seqlen,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
            STAGE=1 if causal else 0
        )
        
        ctx.save_for_backward(q, k, v, out, M)
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = BLOCK_DMODEL
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, M = ctx.saved_tensors
        do = do.contiguous()
        
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        batch, nheads, seqlen, d_head = q.shape
        Delta = torch.empty_like(M)
        
        BLOCK_M = 32
        grid_pre = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
        
        _bwd_preprocess[grid_pre](
            out, do, Delta,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            seqlen, BLOCK_M=BLOCK_M, D_HEAD=ctx.BLOCK_DMODEL,
            num_stages=1,
        )
        
        BLOCK_N = 32
        grid = (triton.cdiv(seqlen, BLOCK_N), batch * nheads)
        
        _bwd_kernel[grid](
            q, k, v, ctx.sm_scale, out, do, dq, dk, dv, M, Delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            batch, nheads, seqlen,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=ctx.BLOCK_DMODEL, BLOCK_N=BLOCK_N,
            num_stages=1,
        )
        
        return dq, dk, dv, None, None


def triton_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert q.shape == k.shape == v.shape
    assert q.dtype in (torch.float16, torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(q.size(-1))
    return _Attention.apply(q, k, v, causal, sm_scale)
