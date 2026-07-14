"""
Triton fused block-tiled attention kernel (FlashAttention).

This implements exact attention with a causal mask, optimizing memory bandwidth
by keeping the attention matrix $QK^T$ in SRAM and incrementally updating the
softmax denominator.

Reference:
Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
Triton official tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,  #
    K_ptrs, V_ptrs,  #
    start_m, qk_scale,  #
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
):
    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_ptrs + (start_n * BLOCK_DMODEL))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale
        
        # causal mask
        if STAGE == 1:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))
            
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(V_ptrs + (start_n * BLOCK_DMODEL))
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, M, Out,  #
    stride_qz, stride_qh, stride_qm, stride_qk,  #
    stride_kz, stride_kh, stride_kn, stride_kk,  #
    stride_vz, stride_vh, stride_vn, stride_vk,  #
    stride_oz, stride_oh, stride_om, stride_on,  #
    Z, H, N_CTX,  #
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr  #
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # load q
    q = tl.load(Q_block_ptr)
    
    # log2(e) for base-2 exponentiation in inner loop
    qk_scale = sm_scale * 1.44269504
    
    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale,  
        BLOCK_M, BLOCK_DMODEL, BLOCK_N, STAGE, offs_m, offs_n
    )
    
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc.to(tl.float16)
    tl.store(O_block_ptr, acc)
    
    # write off m
    off_hz_m = off_hz * N_CTX + start_m * BLOCK_M
    m_ptrs = M + off_hz_m + tl.arange(0, BLOCK_M)
    tl.store(m_ptrs, m_i)


@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_doz, stride_doh, stride_dom, stride_don,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    
    O_block_ptr = tl.make_block_ptr(
        base=Out + off_hz * stride_oh,
        shape=(1048576, D_HEAD), # arbitrary max seq len
        strides=(stride_om, stride_on),
        offsets=(off_m[0], 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + off_hz * stride_doh,
        shape=(1048576, D_HEAD),
        strides=(stride_dom, stride_don),
        offsets=(off_m[0], 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    
    # load
    o = tl.load(O_block_ptr).to(tl.float32)
    do = tl.load(DO_block_ptr).to(tl.float32)
    
    # compute
    delta = tl.sum(o * do, axis=1)
    
    # write
    delta_ptrs = Delta + off_hz * 1048576 + off_m # Max seq len safe
    tl.store(delta_ptrs, delta)


@triton.jit
def _bwd_kernel_one_col_block(
    start_n, Q, K, V, do, dq, dk, dv, sm_scale,
    q_stride_z, q_stride_h, q_stride_m, q_stride_k,
    k_stride_z, k_stride_h, k_stride_n, k_stride_k,
    v_stride_z, v_stride_h, v_stride_n, v_stride_k,
    do_stride_z, do_stride_h, do_stride_m, do_stride_k,
    dq_stride_z, dq_stride_h, dq_stride_m, dq_stride_k,
    dk_stride_z, dk_stride_h, dk_stride_n, dk_stride_k,
    dv_stride_z, dv_stride_h, dv_stride_n, dv_stride_k,
    M, Delta, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # This is a simplified backwards pass for causal attention.
    # In a full production implementation, backward is much more complex
    # and splits the work across sequence length for high occupancy.
    # Here we do a straightforward implementation.
    
    off_hz = tl.program_id(1)
    
    # Offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    
    # Pointers
    qvk_offset = off_hz * q_stride_h
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(k_stride_k, k_stride_n),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(v_stride_k, v_stride_n),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    
    dk_accum = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv_accum = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over M (queries)
    for start_m in range(start_n * BLOCK_N // BLOCK_M, tl.cdiv(N_CTX, BLOCK_M)):
        offs_m_curr = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(q_stride_m, q_stride_k),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        q = tl.load(Q_block_ptr)
        
        DO_block_ptr = tl.make_block_ptr(
            base=do + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(do_stride_m, do_stride_k),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        do_curr = tl.load(DO_block_ptr)
        
        # compute qk
        qk = tl.dot(q, k) * sm_scale
        
        # causal mask
        mask = offs_m_curr[:, None] >= offs_n[None, :]
        qk = tl.where(mask, qk, float("-inf"))
        
        # Load m and delta
        m = tl.load(M + off_hz * N_CTX + offs_m_curr)
        delta = tl.load(Delta + off_hz * N_CTX + offs_m_curr)
        
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
        DQ_block_ptr = tl.make_block_ptr(
            base=dq + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(dq_stride_m, dq_stride_k),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        dq_curr = tl.load(DQ_block_ptr)
        dq_curr += tl.dot(ds_tensor, tl.trans(k))
        tl.store(DQ_block_ptr, dq_curr)

    # store dk, dv
    DK_block_ptr = tl.make_block_ptr(
        base=dk + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(dk_stride_n, dk_stride_k),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=dv + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(dv_stride_n, dv_stride_k),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(DK_block_ptr, dk_accum.to(tl.float16))
    tl.store(DV_block_ptr, dv_accum.to(tl.float16))


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO,
    DQ, DK, DV,
    M, Delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Backward pass over keys/values (parallelizing over sequence length N)
    start_n = tl.program_id(0)
    _bwd_kernel_one_col_block(
        start_n, Q, K, V, DO, DQ, DK, DV, sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        M, Delta, N_CTX,
        BLOCK_M, BLOCK_DMODEL, BLOCK_N,
    )


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        BLOCK_DMODEL = q.shape[-1]
        assert BLOCK_DMODEL in {16, 32, 64, 128}
        
        batch, nheads, seqlen, d_head = q.shape
        
        # Ensure tensors are contiguous and on correct device
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        out = torch.empty_like(q)
        M = torch.empty((batch, nheads, seqlen), device=q.device, dtype=torch.float32)
        
        # Block tuning for T4 (Compute Capability 7.5) vs A100 (8.0)
        # T4 has 64KB shared memory per SM, so we keep blocks small
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
        
        # Precompute Delta
        Delta = torch.empty_like(M)
        BLOCK_M = 64
        grid_pre = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
        
        _bwd_preprocess[grid_pre](
            out, do,
            Delta,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            BLOCK_M=BLOCK_M, D_HEAD=ctx.BLOCK_DMODEL,
        )
        
        BLOCK_N = 64
        grid = (triton.cdiv(seqlen, BLOCK_N), batch * nheads)
        
        _bwd_kernel[grid](
            q, k, v, ctx.sm_scale, out, do,
            dq, dk, dv,
            M, Delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            batch, nheads, seqlen,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=ctx.BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        )
        
        return dq, dk, dv, None, None


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Computes exact attention using a custom Triton kernel.
    
    Args:
        q: query tensor of shape (batch, n_heads, seq_len, d_head)
        k: key tensor of shape (batch, n_heads, seq_len, d_head)
        v: value tensor of shape (batch, n_heads, seq_len, d_head)
        causal: whether to apply a causal (autoregressive) mask
        
    Returns:
        output tensor of shape (batch, n_heads, seq_len, d_head)
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert q.shape == k.shape == v.shape
    assert q.dtype in (torch.float16, torch.bfloat16)
    
    sm_scale = 1.0 / math.sqrt(q.size(-1))
    return _Attention.apply(q, k, v, causal, sm_scale)
