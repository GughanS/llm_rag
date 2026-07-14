"""
Full decoder-only transformer — stacks TransformerBlocks into a complete
language model with token embedding, RMSNorm, and an LM head.

Architecture (pre-norm, following GPT-2/LLaMA):
    token_embedding → [TransformerBlock × N] → RMSNorm → lm_head

Each TransformerBlock:
    x = x + Attention(RMSNorm(x))
    x = x + FeedForward(RMSNorm(x))

Weight tying: embedding.weight is shared with lm_head.weight to reduce
parameter count by ~25M (vocab_size × d_model).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import TransformerConfig
from model.attention import CausalSelfAttention
from model.layers import RMSNorm, SwiGLUFeedForward


class TransformerBlock(nn.Module):
    """Single transformer decoder block (pre-norm)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = SwiGLUFeedForward(config)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.attn_norm(x), start_pos=start_pos)
        x = x + self.ff(self.ff_norm(x))
        return x


class Transformer(nn.Module):
    """Decoder-only transformer language model.

    Produces logits over the vocabulary for next-token prediction.
    Includes a generate() method for autoregressive text generation.
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()
        self.config = config

        # Token embedding (no learned positional embedding — RoPE handles position)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final norm before LM head
        self.norm = RMSNorm(config.d_model)

        # Language modelling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding ↔ LM-head weights
        if config.weight_tying:
            self.lm_head.weight = self.token_emb.weight

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise weights following GPT-2 conventions.

        - Linear layers: N(0, 0.02)
        - Embeddings: N(0, 0.02)
        - Residual projections scaled by 1/√(2·n_layers) to stabilise
          deep residual streams.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale residual projections (out_proj in attention, w_down in FFN)
            if hasattr(module, "_is_residual"):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len) — token indices
            targets:   (batch, seq_len) — target token indices for loss
            start_pos: position offset for incremental decoding

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   scalar cross-entropy loss if targets provided, else None
        """
        B, T = input_ids.shape

        # Token embedding (no positional embedding — RoPE is in attention)
        x = self.drop(self.token_emb(input_ids))

        # Transformer blocks
        for block in self.blocks:
            x = block(x, start_pos=start_pos)

        # Final norm → logits
        x = self.norm(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # padding token
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with temperature, top-k, and top-p sampling.

        Args:
            input_ids: (batch, seq_len) — prompt token IDs
            max_new_tokens: number of tokens to generate
            temperature: softmax temperature (1.0 = neutral, <1 = sharper, >1 = flatter)
            top_k: keep only top-k logits before sampling
            top_p: nucleus sampling threshold (if provided, overrides top_k)

        Returns:
            (batch, seq_len + max_new_tokens) — prompt + generated tokens
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if sequence has grown too long
            idx_cond = input_ids[:, -self.config.max_seq_len :]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (batch, vocab_size) — last position only

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right so that the first token above threshold is kept
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
