# Architecture Patterns

Several standard software engineering design patterns are applied in this codebase to ensure extensibility and security.

## Patterns Applied

### Strategy Pattern — Attention Backend Selection
- **Where:** `model/attention.py`
- **What:** Abstracted the attention forward pass into an `AttentionBackend` protocol with `SDPAAttentionBackend` and `TritonAttentionBackend` implementations.
- **Why:** Replaces rigid `if/else` logic in the core module. Makes it trivial to plug in future backends (e.g. FlashAttention-3) without modifying the `CausalSelfAttention` class.

### Factory Pattern — Model/Checkpoint Instantiation
- **Where:** `model/factory.py`
- **What:** Created a `ModelFactory` with static methods to instantiate models from scratch or load them from checkpoints.
- **Why:** Centralizes config parsing and weight loading. Critically, it provides a single choke-point to enforce `safetensors` usage, preventing any arbitrary code execution vulnerabilities from PyTorch's native `pickle` loading.

### Observer Pattern — Training Loop Callbacks
- **Where:** `model/train.py`
- **What:** Decoupled logging, validation, and checkpointing from the core optimization step by introducing a `Trainer` class and a `TrainerCallback` base class. Implemented `LoggingCallback`, `CheckpointCallback`, `ValidationCallback`, and `WandbCallback`.
- **Why:** The previous training loop was a 400-line monolithic function. The Observer pattern isolates side-effects from the math, making the core loop readable and highly extensible.

### Security Hardening
- **Input Validation:** Added strict bounds checking to `Transformer.forward` in `model/transformer.py` to gracefully reject inputs exceeding `max_seq_len`, preventing GPU OOM crashes in production.
- **Dependency Auditing:** Created `requirements-dev.txt` including `pip-audit` to automate vulnerability scanning of python packages.
- **Weight Serialization:** Exclusively using `safetensors.torch.save_model` over `torch.save`.
