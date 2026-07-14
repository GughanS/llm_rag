"""
Core layers — RMSNorm and SwiGLU FeedForward.

RMSNorm:  Simpler and faster than LayerNorm (no mean centering).
          Used by LLaMA, Gemma, and most modern LLMs.

SwiGLU:   Gated feed-forward with SiLU activation.
          Better empirical performance than GELU (PaLM, LLaMA).
          Uses 3 linear projections → parameter count is 3 × d_model × d_ff
          (vs 2 × d_model × d_ff for standard FFN), but gains are worth it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import TransformerConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    Normalises by RMS of activations, no mean centering, no bias.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    out = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down

    Three weight matrices: W_gate, W_up ∈ R^{d_model × d_ff},
                           W_down ∈ R^{d_ff × d_model}.
    No bias terms — following LLaMA convention.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        gate = F.silu(self.w_gate(x))       # (batch, seq_len, d_ff)
        up = self.w_up(x)                    # (batch, seq_len, d_ff)
        return self.dropout(self.w_down(gate * up))
