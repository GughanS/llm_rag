"""
Training loop for the from-scratch transformer.
Utilizes the Observer Pattern (Callbacks) and Factory Pattern for modularity.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import TransformerConfig
from model.dataset import PackedTextDataset, SyntheticDataset
from model.factory import ModelFactory


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# Observer Pattern: Trainer Callbacks
# =============================================================================

class TrainerCallback:
    """Base class for training callbacks (Observer Pattern)."""
    def on_train_start(self, trainer: "Trainer"): pass
    def on_epoch_start(self, trainer: "Trainer", epoch: int): pass
    def on_step_end(self, trainer: "Trainer", step: int, loss: float, lr: float, grad_norm: float): pass
    def on_epoch_end(self, trainer: "Trainer", epoch: int, avg_loss: float): pass
    def on_train_end(self, trainer: "Trainer"): pass


class LoggingCallback(TrainerCallback):
    def __init__(self, log_interval: int, max_seq_len: int, batch_size: int):
        self.log_interval = log_interval
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.epoch_start_time = 0

    def on_epoch_start(self, trainer, epoch):
        self.epoch_start_time = time.time()

    def on_step_end(self, trainer, step, loss, lr, grad_norm):
        if step % self.log_interval == 0:
            elapsed = time.time() - self.epoch_start_time
            # approximation of batch index in epoch
            batch_idx = trainer.batch_idx
            avg_loss = trainer.epoch_loss / (batch_idx + 1)
            tokens_per_sec = ((batch_idx + 1) * self.batch_size * self.max_seq_len) / elapsed
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if trainer.device.type == "cuda" else 0
            
            print(
                f"  Step {step:>6d} | Loss {loss:.4f} | Avg {avg_loss:.4f} | "
                f"LR {lr:.2e} | Grad {grad_norm:.2f} | GPU {gpu_mem:.1f}GB | "
                f"{tokens_per_sec:.0f} tok/s"
            )

    def on_epoch_end(self, trainer, epoch, avg_loss):
        epoch_time = time.time() - self.epoch_start_time
        print(f"\n  Epoch {epoch + 1} done — Avg loss: {avg_loss:.4f} — Time: {epoch_time:.0f}s\n")


class CheckpointCallback(TrainerCallback):
    def __init__(self, save_interval: int, ckpt_dir: Path):
        self.save_interval = save_interval
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, trainer, name: str, loss: float):
        path = self.ckpt_dir / name
        path.mkdir(parents=True, exist_ok=True)
        import safetensors.torch
        safetensors.torch.save_model(trainer.model, path / "model.safetensors")
        trainer.config.to_json(path / "config.json")
        torch.save({
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "step": trainer.global_step,
            "loss": loss,
        }, path / "training_state.pt")
        print(f"  💾 Checkpoint saved to {path}")

    def on_step_end(self, trainer, step, loss, lr, grad_norm):
        if step > 0 and step % self.save_interval == 0:
            self._save(trainer, "latest", loss)

    def on_train_end(self, trainer):
        self._save(trainer, "final", trainer.epoch_loss / max(1, trainer.batch_idx))


class ValidationCallback(TrainerCallback):
    def __init__(self, eval_interval: int, val_loader: DataLoader, ckpt_callback: CheckpointCallback):
        self.eval_interval = eval_interval
        self.val_loader = val_loader
        self.ckpt_callback = ckpt_callback
        self.best_val_loss = float("inf")

    @torch.no_grad()
    def evaluate(self, trainer, max_batches: int = 50) -> float:
        trainer.model.eval()
        total_loss = 0.0
        n_batches = 0
        for i, (input_ids, targets) in enumerate(self.val_loader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(trainer.device)
            targets = targets.to(trainer.device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=trainer.device.type=="cuda"):
                _, loss = trainer.model(input_ids, targets=targets)
            total_loss += loss.item()
            n_batches += 1
        trainer.model.train()
        return total_loss / max(n_batches, 1)

    def on_step_end(self, trainer, step, loss, lr, grad_norm):
        if step > 0 and step % self.eval_interval == 0:
            val_loss = self.evaluate(trainer)
            print(f"  📊 Val loss: {val_loss:.4f}")
            if hasattr(trainer, "wandb_run") and trainer.wandb_run:
                import wandb
                wandb.log({"val/loss": val_loss}, step=step)
                
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.ckpt_callback._save(trainer, "best", val_loss)

    def on_train_end(self, trainer):
        val_loss = self.evaluate(trainer)
        print(f"\n✅ Training complete. Final val loss: {val_loss:.4f}")
        print(f"   Best val loss: {self.best_val_loss:.4f}")


class WandbCallback(TrainerCallback):
    def on_step_end(self, trainer, step, loss, lr, grad_norm):
        if step % 50 == 0 and hasattr(trainer, "wandb_run") and trainer.wandb_run:
            import wandb
            avg_loss = trainer.epoch_loss / max(1, trainer.batch_idx + 1)
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if trainer.device.type == "cuda" else 0
            wandb.log({
                "train/loss": loss,
                "train/avg_loss": avg_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm,
                "train/gpu_mem_gb": gpu_mem,
                "train/epoch": trainer.current_epoch,
            }, step=step)

    def on_train_end(self, trainer):
        if hasattr(trainer, "wandb_run") and trainer.wandb_run:
            import wandb
            wandb.finish()


# =============================================================================
# Trainer Core
# =============================================================================

class Trainer:
    def __init__(self, model, config, optimizer, train_loader, device, args, callbacks: List[TrainerCallback] = None):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.args = args
        self.callbacks = callbacks or []
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        
        self.global_step = 0
        self.current_epoch = 0
        self.epoch_loss = 0.0
        self.batch_idx = 0
        self.total_steps = len(train_loader) * args.epochs
        self.warmup_steps = min(args.warmup_steps, self.total_steps // 5)
        self.min_lr = args.lr * 0.1
        
        if args.wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name,
                    config={**vars(args), "model_config": config.to_json()}
                )
            except ImportError:
                print("  ⚠️  wandb not installed, skipping logging")
                self.wandb_run = None
        else:
            self.wandb_run = None

    def fit(self):
        for cb in self.callbacks:
            cb.on_train_start(self)
            
        self.model.train()
        
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            self.epoch_loss = 0.0
            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)
                
            for batch_idx, (input_ids, targets) in enumerate(self.train_loader):
                self.batch_idx = batch_idx
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                lr = get_lr(self.global_step, self.warmup_steps, self.total_steps, self.args.lr, self.min_lr)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                    _, loss = self.model(input_ids, targets=targets)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self.epoch_loss += loss.item()
                
                for cb in self.callbacks:
                    cb.on_step_end(self, self.global_step, loss.item(), lr, grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    
                self.global_step += 1

            avg_loss = self.epoch_loss / len(self.train_loader)
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, avg_loss)
                
        for cb in self.callbacks:
            cb.on_train_end(self)

# =============================================================================
# Main
# =============================================================================

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Factory Pattern
    config_dict = TransformerConfig.from_name("tinystories-25m").__dict__
    model = ModelFactory.create_from_config(config_dict).to(device)
    config = model.config
    
    # Checkpoint resumption (using Factory Pattern structure implicitly inside CheckpointCallback logic later, 
    # but for initial resume we handle here or trust start_step = 0 for now to keep refactor clean).
    # Since Phase 4.5 is about patterns, we'll keep it simple: no resume implemented here for brevity, 
    # or just use ModelFactory if resume is requested.
    if args.resume:
        print("Resuming from checkpoint using factory...")
        model = ModelFactory.load_from_checkpoint(str(Path(args.checkpoint_dir) / "latest")).to(device)
        config = model.config

    if args.synthetic:
        n_samples = args.max_stories or 2000
        train_ds = SyntheticDataset(n_samples=n_samples, max_seq_len=config.max_seq_len, vocab_size=config.vocab_size)
        val_ds = SyntheticDataset(n_samples=max(n_samples // 10, 50), max_seq_len=config.max_seq_len, vocab_size=config.vocab_size)
    else:
        train_ds = PackedTextDataset(dataset_name=args.dataset, split="train", max_seq_len=config.max_seq_len, max_samples=args.max_stories)
        val_ds = PackedTextDataset(dataset_name=args.dataset, split="validation", max_seq_len=config.max_seq_len, max_samples=args.max_stories // 10 if args.max_stories else None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=device.type == "cuda")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_cb = CheckpointCallback(args.save_interval, ckpt_dir)
    
    callbacks = [
        LoggingCallback(args.log_interval, config.max_seq_len, args.batch_size),
        ckpt_cb,
        ValidationCallback(args.eval_interval, val_loader, ckpt_cb),
        WandbCallback()
    ]

    trainer = Trainer(model, config, optimizer, train_loader, device, args, callbacks)
    trainer.fit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch transformer on TinyStories")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "tinyshakespeare"])
    parser.add_argument("--max_stories", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="llm-systems-phase1")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
