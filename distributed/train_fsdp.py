"""
Distributed fine-tuning with PyTorch FSDP (Fully Sharded Data Parallel).

This script demonstrates training a model (TinyLlama-1.1B) that is too large
to fit on a single GPU using standard FP32 Adam (requires ~16-18GB just for optimizer states).
FSDP shards the model parameters, gradients, and optimizer states across multiple GPUs.
"""
import argparse
import os
import time
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


class TinyShakespeareDataset(Dataset):
    """Simple dataset bypassing HuggingFace to avoid CDN issues."""
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Download data
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_path = Path("/tmp/tinyshakespeare.txt")
        if not data_path.exists():
            print(f"Downloading dataset from {url}...")
            urllib.request.urlretrieve(url, data_path)
            
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Tokenize everything and chunk it
        print("Tokenizing dataset...")
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        
        # Create blocks
        self.examples = []
        for i in range(0, len(tokens) - max_length, max_length):
            self.examples.append(tokens[i : i + max_length])
            
        print(f"Created {len(self.examples)} examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Shift inputs for causal LM (x -> y)
        x = self.examples[idx][:-1]
        y = self.examples[idx][1:]
        return {"input_ids": x, "labels": y}


def setup():
    """Initialize the process group."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # Fallback to single GPU if not run with torchrun
        print("Not running under torchrun, falling back to single GPU.")
        torch.cuda.set_device(0)
        return 0, 0, 1


def cleanup():
    """Clean up the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    rank, local_rank, world_size = setup()
    
    if rank == 0:
        print(f"Training on {world_size} GPUs with FSDP={args.fsdp}")
        
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    
    # We load tokenizer on all ranks
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = TinyShakespeareDataset(tokenizer, max_length=args.seq_len)
    
    # In distributed training, we need a DistributedSampler
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank
        )
        
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        shuffle=(sampler is None)
    )

    if rank == 0:
        print(f"Loading model {model_name} (random weights to bypass 4GB download)...")
        
    # Load model config and initialize with random weights on CPU first
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    
    if args.fsdp and world_size > 1:
        # FSDP auto-wrap policy for transformer layers
        import functools
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
        
        # Wrap model in FSDP
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sync_module_states=True, # Sync weights from rank 0
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            use_orig_params=False,
        )
        
        # Apply activation checkpointing to save memory
        if args.activation_checkpointing:
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
            )
            if rank == 0:
                print("Applied activation checkpointing.")
    else:
        # Single GPU mode (will likely OOM on 1.1B params)
        model = model.to(local_rank)

    # Optimizer (AdamW full state is huge, FSDP shards it)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= args.max_steps:
            break
            
        input_ids = batch["input_ids"].to(local_rank)
        labels = batch["labels"].to(local_rank)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
    # Record peak memory (Constraint Verification)
    peak_mem_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 * 1024)
    print(f"[Rank {rank}] Peak VRAM: {peak_mem_mb:.2f} MB")
    
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP")
    parser.add_argument("--activation_checkpointing", action="store_true", help="Enable Activation Checkpointing")
    
    args = parser.parse_args()
    train(args)
