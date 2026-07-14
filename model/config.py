"""
TransformerConfig — all hyperparameters for the from-scratch transformer.

Centralises every architectural knob in one dataclass so that:
  1. Checkpoint metadata is self-describing (config saved alongside weights).
  2. Tests and benchmarks can construct arbitrary configurations.
  3. Phase 2 (Triton kernel) can override attention backend without touching model code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TransformerConfig:
    """Full configuration for the decoder-only transformer."""

    # --- architecture ---
    n_layers: int = 12
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024          # feed-forward intermediate dim (4× d_model)
    vocab_size: int = 50257   # GPT-2 tokenizer vocabulary
    max_seq_len: int = 512    # maximum context window
    dropout: float = 0.1

    # --- RoPE ---
    rope_theta: float = 10_000.0  # base frequency for rotary embeddings

    # --- training ---
    weight_tying: bool = True  # share embedding ↔ LM-head weights

    # --- attention backend (for Phase 2 swapability) ---
    attn_backend: str = "sdpa"  # "sdpa" | "triton" (Phase 2)

    # --- derived ---
    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads

    # --- presets ---
    _PRESETS: dict[str, dict] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_name(cls, name: str) -> "TransformerConfig":
        """Load a named preset configuration."""
        presets = {
            "tinystories-25m": dict(
                n_layers=12, d_model=256, n_heads=4, d_ff=1024,
                vocab_size=50257, max_seq_len=512, dropout=0.1,
            ),
            "tinystories-small": dict(
                n_layers=6, d_model=384, n_heads=6, d_ff=1536,
                vocab_size=50257, max_seq_len=512, dropout=0.1,
            ),
        }
        if name not in presets:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(presets.keys())}"
            )
        return cls(**presets[name])

    # --- serialisation ---
    def to_json(self, path: Optional[str | Path] = None) -> str:
        """Serialise config to JSON string. Optionally write to file."""
        d = asdict(self)
        d.pop("_PRESETS", None)
        json_str = json.dumps(d, indent=2)
        if path is not None:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> "TransformerConfig":
        """Load config from a JSON file or a raw JSON string."""
        p = Path(path_or_str)
        if p.exists():
            raw = p.read_text(encoding="utf-8")
        else:
            raw = str(path_or_str)
        d = json.loads(raw)
        d.pop("_PRESETS", None)
        return cls(**d)
