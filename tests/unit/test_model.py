"""
Unit tests for the from-scratch transformer — Phase 1.

Tests cover:
  - Output shapes for all components
  - Forward pass smoke test (no NaN, no Inf)
  - Causal mask correctness
  - Weight tying
  - Config serialisation round-trip
  - Gradient flow through all parameters
  - Deterministic output with fixed seed
  - Parameter count verification
  - Generation output shape
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from model.config import TransformerConfig
from model.attention import CausalSelfAttention, precompute_rope_frequencies, apply_rope
from model.layers import RMSNorm, SwiGLUFeedForward
from model.transformer import Transformer, TransformerBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config() -> TransformerConfig:
    """A tiny config for fast unit tests (not the full 25M model)."""
    return TransformerConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
        dropout=0.0,  # no dropout for deterministic tests
    )


@pytest.fixture
def batch(small_config: TransformerConfig) -> torch.Tensor:
    """Random input_ids batch."""
    return torch.randint(
        0, small_config.vocab_size, (2, small_config.max_seq_len)
    )


# ---------------------------------------------------------------------------
# TransformerConfig tests
# ---------------------------------------------------------------------------

class TestTransformerConfig:

    def test_d_head(self):
        cfg = TransformerConfig(d_model=512, n_heads=8)
        assert cfg.d_head == 64

    def test_d_head_not_divisible(self):
        with pytest.raises(AssertionError):
            _ = TransformerConfig(d_model=100, n_heads=7).d_head

    def test_from_name(self):
        cfg = TransformerConfig.from_name("tinystories-25m")
        assert cfg.n_layers == 12
        assert cfg.d_model == 256

    def test_from_name_invalid(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            TransformerConfig.from_name("nonexistent")

    def test_json_roundtrip(self, small_config: TransformerConfig):
        json_str = small_config.to_json()
        restored = TransformerConfig.from_json(json_str)
        assert restored.n_layers == small_config.n_layers
        assert restored.d_model == small_config.d_model
        assert restored.n_heads == small_config.n_heads

    def test_json_file_roundtrip(self, small_config: TransformerConfig, tmp_path: Path):
        path = tmp_path / "config.json"
        small_config.to_json(path)
        restored = TransformerConfig.from_json(path)
        assert restored.d_ff == small_config.d_ff
        assert restored.vocab_size == small_config.vocab_size


# ---------------------------------------------------------------------------
# RMSNorm tests
# ---------------------------------------------------------------------------

class TestRMSNorm:

    def test_output_shape(self, small_config: TransformerConfig):
        norm = RMSNorm(small_config.d_model)
        x = torch.randn(2, 16, small_config.d_model)
        out = norm(x)
        assert out.shape == x.shape

    def test_no_nan(self, small_config: TransformerConfig):
        norm = RMSNorm(small_config.d_model)
        x = torch.randn(2, 16, small_config.d_model)
        out = norm(x)
        assert not torch.isnan(out).any()

    def test_normalises(self, small_config: TransformerConfig):
        """RMSNorm should make the RMS of each vector approximately 1."""
        norm = RMSNorm(small_config.d_model)
        x = torch.randn(2, 16, small_config.d_model) * 10  # large values
        out = norm(x)
        rms = torch.sqrt(torch.mean(out * out, dim=-1))
        # After norm (with weight=1), RMS should be close to 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


# ---------------------------------------------------------------------------
# SwiGLUFeedForward tests
# ---------------------------------------------------------------------------

class TestSwiGLUFeedForward:

    def test_output_shape(self, small_config: TransformerConfig):
        ff = SwiGLUFeedForward(small_config)
        x = torch.randn(2, 16, small_config.d_model)
        out = ff(x)
        assert out.shape == x.shape

    def test_no_nan(self, small_config: TransformerConfig):
        ff = SwiGLUFeedForward(small_config)
        x = torch.randn(2, 16, small_config.d_model)
        out = ff(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# RoPE tests
# ---------------------------------------------------------------------------

class TestRoPE:

    def test_precompute_shape(self):
        cos, sin = precompute_rope_frequencies(d_head=64, max_seq_len=128)
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

    def test_apply_rope_shape(self):
        B, H, T, D = 2, 4, 16, 64
        x = torch.randn(B, H, T, D)
        cos, sin = precompute_rope_frequencies(d_head=D, max_seq_len=T)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_apply_rope_no_nan(self):
        B, H, T, D = 2, 4, 16, 64
        x = torch.randn(B, H, T, D)
        cos, sin = precompute_rope_frequencies(d_head=D, max_seq_len=T)
        out = apply_rope(x, cos, sin)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# CausalSelfAttention tests
# ---------------------------------------------------------------------------

class TestCausalSelfAttention:

    def test_output_shape(self, small_config: TransformerConfig):
        attn = CausalSelfAttention(small_config)
        x = torch.randn(2, small_config.max_seq_len, small_config.d_model)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_mask(self, small_config: TransformerConfig):
        """Verify that changing future tokens does not affect past outputs.

        If the causal mask works correctly, the output at position i should
        be identical regardless of what tokens appear at positions > i.
        """
        attn = CausalSelfAttention(small_config)
        attn.eval()

        x1 = torch.randn(1, small_config.max_seq_len, small_config.d_model)
        x2 = x1.clone()
        # Modify the last 8 positions in x2
        x2[:, -8:, :] = torch.randn(1, 8, small_config.d_model)

        out1 = attn(x1)
        out2 = attn(x2)

        # Positions before the modification should be identical
        torch.testing.assert_close(
            out1[:, :-8, :],
            out2[:, :-8, :],
            atol=1e-5,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# TransformerBlock tests
# ---------------------------------------------------------------------------

class TestTransformerBlock:

    def test_output_shape(self, small_config: TransformerConfig):
        block = TransformerBlock(small_config)
        x = torch.randn(2, small_config.max_seq_len, small_config.d_model)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self, small_config: TransformerConfig):
        """Output should differ from input (non-trivial transformation)."""
        block = TransformerBlock(small_config)
        x = torch.randn(2, small_config.max_seq_len, small_config.d_model)
        out = block(x)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Full Transformer tests
# ---------------------------------------------------------------------------

class TestTransformer:

    def test_output_shape(self, small_config: TransformerConfig, batch: torch.Tensor):
        model = Transformer(small_config)
        logits, loss = model(batch)
        assert logits.shape == (2, small_config.max_seq_len, small_config.vocab_size)
        assert loss is None  # no targets

    def test_with_targets(self, small_config: TransformerConfig, batch: torch.Tensor):
        model = Transformer(small_config)
        targets = torch.randint(0, small_config.vocab_size, batch.shape)
        logits, loss = model(batch, targets=targets)
        assert logits.shape == (2, small_config.max_seq_len, small_config.vocab_size)
        assert loss is not None
        assert loss.ndim == 0  # scalar
        assert not torch.isnan(loss)

    def test_weight_tying(self, small_config: TransformerConfig):
        model = Transformer(small_config)
        # Embedding and LM head should share the same data pointer
        assert model.token_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()

    def test_no_weight_tying(self):
        cfg = TransformerConfig(
            n_layers=2, d_model=64, n_heads=4, d_ff=128,
            vocab_size=256, max_seq_len=32, weight_tying=False,
        )
        model = Transformer(cfg)
        assert model.token_emb.weight.data_ptr() != model.lm_head.weight.data_ptr()

    def test_no_nan_no_inf(self, small_config: TransformerConfig, batch: torch.Tensor):
        model = Transformer(small_config)
        logits, _ = model(batch)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_gradient_flow(self, small_config: TransformerConfig, batch: torch.Tensor):
        """Every trainable parameter should receive a non-zero gradient."""
        model = Transformer(small_config)
        targets = torch.randint(0, small_config.vocab_size, batch.shape)
        _, loss = model(batch, targets=targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_determinism(self, small_config: TransformerConfig, batch: torch.Tensor):
        """Same seed → same output."""
        torch.manual_seed(42)
        m1 = Transformer(small_config)
        out1, _ = m1(batch)

        torch.manual_seed(42)
        m2 = Transformer(small_config)
        out2, _ = m2(batch)

        torch.testing.assert_close(out1, out2)

    def test_param_count(self):
        """Verify the 25M target (with weight tying) is in the right ballpark."""
        cfg = TransformerConfig.from_name("tinystories-25m")
        model = Transformer(cfg)
        n_params = model.count_parameters()
        # Should be roughly 25M ± 5M
        assert 15_000_000 < n_params < 35_000_000, (
            f"Expected ~25M params, got {n_params:,}"
        )

    def test_generate_shape(self, small_config: TransformerConfig):
        model = Transformer(small_config)
        prompt = torch.randint(0, small_config.vocab_size, (1, 5))
        output = model.generate(prompt, max_new_tokens=10)
        assert output.shape == (1, 15)  # 5 prompt + 10 generated

    def test_generate_stays_in_vocab(self, small_config: TransformerConfig):
        model = Transformer(small_config)
        prompt = torch.randint(0, small_config.vocab_size, (1, 5))
        output = model.generate(prompt, max_new_tokens=10)
        assert (output >= 0).all()
        assert (output < small_config.vocab_size).all()
