"""
Training loop for the from-scratch transformer on TinyStories.

Features:
  - AdamW optimiser with cosine LR schedule + linear warmup
  - Mixed-precision training (fp16 autocast) for T4 performance
  - Gradient clipping (max_norm=1.0) for training stability
  - Periodic validation loss evaluation
  - Checkpoint saving in safetensors format (ADR requirement)
  - Weights & Biases logging (loss curves, LR, grad norms, GPU memory)
  - Full reproducibility via seed control

Usage:
  python -m model.train                          # defaults
  python -m model.train --epochs 5 --lr 3e-4     # override
  python -m model.train --max_stories 1000        # quick debug run
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import TransformerConfig
from model.transformer import Transformer
from model.dataset import TinyStoriesDataset, SyntheticDataset


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic ops where possible (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay from max_lr to min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(
    model: Transformer,
    config: TransformerConfig,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: Path,
) -> None:
    """Save model checkpoint in safetensors format + training state as .pt."""
    from safetensors.torch import save_model

    path.mkdir(parents=True, exist_ok=True)

    # Model weights → safetensors (ADR: never pickle)
    # Use save_model instead of save_file to handle weight-tied shared tensors
    save_model(model, path / "model.safetensors")

    # Config → JSON
    config.to_json(path / "config.json")

    # Optimiser + training state → torch (this is our own code, not untrusted)
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss,
        },
        path / "training_state.pt",
    )
    print(f"  💾 Checkpoint saved to {path}")


def load_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
) -> int:
    """Load a checkpoint. Returns the step number to resume from."""
    from safetensors.torch import load_model

    if not (path / "model.safetensors").exists():
        return 0

    # Load model weights (handles shared tensors from weight tying)
    load_model(model, path / "model.safetensors")

    # Load training state
    if (path / "training_state.pt").exists():
        training_state = torch.load(
            path / "training_state.pt", map_location=device, weights_only=True
        )
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        step = training_state["step"]
        print(f"  📂 Resumed from checkpoint at step {step}")
        return step

    return 0


@torch.no_grad()
def evaluate(
    model: Transformer,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Evaluate model on validation set. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i, (input_ids, targets) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model(input_ids, targets=targets)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Config ---
    config = TransformerConfig.from_name("tinystories-25m")
    print(f"\nModel config: {config}")

    # --- Dataset ---
    if args.synthetic:
        print("\nUsing synthetic dataset (pipeline validation mode)...")
        n_samples = args.max_stories or 2000
        train_ds = SyntheticDataset(
            n_samples=n_samples,
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
        )
        val_ds = SyntheticDataset(
            n_samples=max(n_samples // 10, 50),
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
        )
    else:
        print("\nLoading TinyStories dataset...")
        train_ds = TinyStoriesDataset(
            split="train",
            max_seq_len=config.max_seq_len,
            max_stories=args.max_stories,
        )
        val_ds = TinyStoriesDataset(
            split="validation",
            max_seq_len=config.max_seq_len,
            max_stories=args.max_stories // 10 if args.max_stories else None,
        )
    print(f"  Train: {len(train_ds):,} chunks")
    print(f"  Val:   {len(val_ds):,} chunks")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    model = Transformer(config).to(device)
    n_params = model.count_parameters()
    print(f"\nModel parameters: {n_params:,}")

    # --- Optimiser ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",  # fused AdamW on CUDA
    )

    # --- LR schedule ---
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 5)
    min_lr = args.lr * 0.1

    # --- Mixed precision ---
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # --- Checkpoint ---
    ckpt_dir = Path(args.checkpoint_dir)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, ckpt_dir / "latest", device)

    # --- W&B ---
    if args.wandb:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    **vars(args),
                    "model_config": config.to_json(),
                    "n_params": n_params,
                    "total_steps": total_steps,
                },
            )
        except ImportError:
            print("  ⚠️  wandb not installed, skipping logging")
            args.wandb = False

    # --- Training loop ---
    print(f"\nTraining for {args.epochs} epochs ({total_steps:,} steps)")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"  LR: {args.lr} → {min_lr}")
    print(f"  Batch size: {args.batch_size}")
    print()

    model.train()
    global_step = start_step
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # LR schedule
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward + backward with mixed precision
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(input_ids, targets=targets)

            scaler.scale(loss).backward()

            # Gradient clipping (unscale first for accurate norm)
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            global_step += 1

            # --- Logging ---
            if global_step % args.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - epoch_start
                tokens_per_sec = (
                    (batch_idx + 1) * args.batch_size * config.max_seq_len / elapsed
                )
                gpu_mem = (
                    torch.cuda.max_memory_allocated() / 1e9
                    if device.type == "cuda"
                    else 0
                )

                print(
                    f"  Step {global_step:>6d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Avg {avg_loss:.4f} | "
                    f"LR {lr:.2e} | "
                    f"Grad {grad_norm:.2f} | "
                    f"GPU {gpu_mem:.1f}GB | "
                    f"{tokens_per_sec:.0f} tok/s"
                )

                if args.wandb:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/avg_loss": avg_loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/gpu_mem_gb": gpu_mem,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            # --- Validation ---
            if global_step % args.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"  📊 Val loss: {val_loss:.4f}")

                if args.wandb:
                    import wandb

                    wandb.log({"val/loss": val_loss}, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, config, optimizer, global_step, val_loss,
                        ckpt_dir / "best",
                    )

            # --- Periodic checkpoint ---
            if global_step % args.save_interval == 0:
                save_checkpoint(
                    model, config, optimizer, global_step, loss.item(),
                    ckpt_dir / "latest",
                )

        # End of epoch
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss / len(train_loader)
        print(
            f"\n  Epoch {epoch + 1}/{args.epochs} done — "
            f"Avg loss: {epoch_avg_loss:.4f} — "
            f"Time: {epoch_time:.0f}s\n"
        )

    # Final checkpoint
    save_checkpoint(
        model, config, optimizer, global_step, epoch_avg_loss,
        ckpt_dir / "final",
    )

    # Final validation
    val_loss = evaluate(model, val_loader, device)
    print(f"\n✅ Training complete. Final val loss: {val_loss:.4f}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Total steps: {global_step}")

    if args.wandb:
        import wandb

        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch transformer on TinyStories"
    )
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=2)

    # Data
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Cap on stories to load (for debugging)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no downloads, for pipeline validation)")

    # Logging
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")

    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="llm-systems-phase1")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
