"""
Causal self-attention with Rotary Positional Embeddings (RoPE).

RoPE encodes relative position directly into the Q/K dot-product by rotating
pairs of dimensions at frequencies determined by their index. This avoids
learned positional embedding matrices and extrapolates better to unseen
sequence lengths.

The attention backend defaults to PyTorch's F.scaled_dot_product_attention
(SDPA), which automatically selects FlashAttention-2 or Memory-Efficient
Attention under the hood. Phase 2 will add a custom Triton kernel as an
alternative backend via the config.attn_backend flag.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import TransformerConfig


def precompute_rope_frequencies(
    d_head: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for RoPE.

    Returns:
        cos: (max_seq_len, d_head)  — cosine component
        sin: (max_seq_len, d_head)  — sine component
    """
    # Frequency for each pair of dimensions: θ_i = theta^{-2i/d}
    dim_pairs = d_head // 2
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    # Position indices
    positions = torch.arange(max_seq_len, device=device).float()
    # Outer product: (seq_len, dim_pairs)
    angles = torch.outer(positions, freqs)
    # Duplicate to cover full d_head: (seq_len, d_head)
    cos = torch.cos(angles).repeat(1, 2)
    sin = torch.sin(angles).repeat(1, 2)
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings to a tensor.

    Args:
        x: (batch, n_heads, seq_len, d_head)
        cos: (seq_len, d_head) or broadcastable
        sin: (seq_len, d_head) or broadcastable

    Returns:
        Rotated tensor with same shape as x.
    """
    d_head = x.shape[-1]
    # Split into pairs and rotate
    x_rotated = torch.cat([-x[..., d_head // 2:], x[..., :d_head // 2]], dim=-1)
    # Reshape cos/sin for broadcasting: (1, 1, seq_len, d_head)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + x_rotated * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Projection layout (no bias, following LLaMA convention):
        Q, K, V ← separate linear projections (not fused, for clarity)
        out     ← linear projection after concatenating heads

    The causal mask is handled by SDPA's `is_causal=True` flag, which is
    both faster and more memory-efficient than materialising a mask tensor.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Q, K, V projections — no bias (LLaMA convention)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute RoPE tables — registered as buffers (saved with state_dict,
        # moved to device with .to(), but not trained)
        cos, sin = precompute_rope_frequencies(
            config.d_head, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            start_pos: position offset for incremental decoding (generation).
                       During training this is always 0.

        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape

        # Project to Q, K, V and reshape to (batch, n_heads, seq_len, d_head)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K (not V — rotary embeddings are position-dependent
        # similarity modifiers, not value transformations)
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Attention — SDPA handles the causal mask and fused softmax internally
        # dropout_p is only applied during training
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.config.dropout if self.training else 0.0,
        )

        # Merge heads and project out
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(attn_out))
