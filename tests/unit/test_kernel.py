"""
Tests for Triton fused attention kernel vs PyTorch SDPA.

Checks numerical parity (forward and backward) and validates ADR constraints.
"""
import pytest
import torch
import torch.nn.functional as F


def test_triton_fused_attention_forward():
    """
    Requirement: Max absolute difference < 1e-2 vs F.scaled_dot_product_attention 
    in fp16, across various sequence lengths and random Q/K/V tensors.
    """
    try:
        import triton
    except ImportError:
        pytest.skip("Triton not installed, skipping kernel tests.")

    from kernels.attention import triton_flash_attention

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping Triton kernel tests.")

    torch.manual_seed(42)
    
    batch, nheads, seqlen, d_head = 2, 4, 1024, 64
    
    # Initialize tensors
    q = torch.randn(batch, nheads, seqlen, d_head, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, nheads, seqlen, d_head, dtype=torch.float16, device="cuda")
    v = torch.randn(batch, nheads, seqlen, d_head, dtype=torch.float16, device="cuda")
    
    # PyTorch SDPA reference
    ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Custom Triton kernel
    tri_out = triton_flash_attention(q, k, v, causal=True)
    
    # Compare
    max_diff = (ref_out - tri_out).abs().max().item()
    mean_diff = (ref_out - tri_out).abs().mean().item()
    
    print(f"\nForward max_diff: {max_diff:.5f}, mean_diff: {mean_diff:.5f}")
    
    # ADR Constraint: < 1e-2 for fp16
    assert max_diff < 1e-2, f"Forward pass max difference {max_diff} exceeds 1e-2 tolerance"


def test_triton_fused_attention_backward():
    """
    Requirement: Gradients must be mathematically correct to ensure training converges.
    """
    try:
        import triton
    except ImportError:
        pytest.skip("Triton not installed, skipping kernel tests.")

    from kernels.attention import triton_flash_attention

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping Triton kernel tests.")

    torch.manual_seed(42)
    
    batch, nheads, seqlen, d_head = 2, 4, 512, 64
    
    q_ref = torch.randn(batch, nheads, seqlen, d_head, dtype=torch.float16, device="cuda").requires_grad_(True)
    k_ref = torch.randn(batch, nheads, seqlen, d_head, dtype=torch.float16, device="cuda").requires_grad_(True)
    v_ref = torch.randn(batch, nheads, seqlen, d_head, dtype=torch.float16, device="cuda").requires_grad_(True)
    
    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)
    
    do = torch.randn(batch, nheads, seqlen, d_head, dtype=torch.float16, device="cuda")
    
    # PyTorch SDPA reference backward
    ref_out = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    ref_out.backward(do)
    
    # Custom Triton kernel backward
    tri_out = triton_flash_attention(q_tri, k_tri, v_tri, causal=True)
    tri_out.backward(do)
    
    dq_diff = (q_ref.grad - q_tri.grad).abs().max().item()
    dk_diff = (k_ref.grad - k_tri.grad).abs().max().item()
    dv_diff = (v_ref.grad - v_tri.grad).abs().max().item()
    
    print(f"\nBackward max_diffs -> dq: {dq_diff:.5f}, dk: {dk_diff:.5f}, dv: {dv_diff:.5f}")
    
    # Tolerance for gradients is slightly higher due to different accumulation orders in bfloat16/float16
    assert dq_diff < 5e-2, f"dq max difference {dq_diff} too high"
    assert dk_diff < 5e-2, f"dk max difference {dk_diff} too high"
    assert dv_diff < 5e-2, f"dv max difference {dv_diff} too high"
