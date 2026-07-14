"""
Memory and speed benchmark for Triton Fused Attention vs PyTorch SDPA.

Measures peak VRAM allocated and execution time across sequence lengths,
proving the memory constraints defined in Phase 2 ADR.
"""
import time
import argparse

import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from kernels.attention import triton_flash_attention


def benchmark_memory_and_speed(batch=4, nheads=8, d_head=64, seq_lengths=[128, 256, 512, 1024, 2048, 4096]):
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run benchmark.")
        return

    print(f"Benchmarking Attention (Batch={batch}, Heads={nheads}, d_head={d_head})")
    print("-" * 75)
    print(f"{'Seq Len':<10} | {'SDPA Mem (MB)':<15} | {'Triton Mem (MB)':<15} | {'SDPA (ms)':<10} | {'Triton (ms)':<10}")
    print("-" * 75)

    results = {
        "seq_len": seq_lengths,
        "sdpa_mem": [],
        "triton_mem": [],
        "sdpa_time": [],
        "triton_time": [],
    }

    for seq_len in seq_lengths:
        # Create inputs
        q = torch.randn(batch, nheads, seq_len, d_head, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(batch, nheads, seq_len, d_head, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(batch, nheads, seq_len, d_head, dtype=torch.float16, device="cuda", requires_grad=True)
        
        do = torch.randn_like(q)

        # --- SDPA Benchmark ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        for _ in range(3):
            out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out_sdpa.backward(do, retain_graph=True)
            
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out_sdpa.backward(do, retain_graph=True)
        torch.cuda.synchronize()
        sdpa_time = (time.time() - start) * 100  # ms per run
        sdpa_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # Reset gradients
        q.grad = k.grad = v.grad = None
        
        # --- Triton Benchmark ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        for _ in range(3):
            out_tri = triton_flash_attention(q, k, v, causal=True)
            out_tri.backward(do, retain_graph=True)
            
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            out_tri = triton_flash_attention(q, k, v, causal=True)
            out_tri.backward(do, retain_graph=True)
        torch.cuda.synchronize()
        tri_time = (time.time() - start) * 100
        tri_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        results["sdpa_mem"].append(sdpa_mem)
        results["triton_mem"].append(tri_mem)
        results["sdpa_time"].append(sdpa_time)
        results["triton_time"].append(tri_time)
        
        print(f"{seq_len:<10} | {sdpa_mem:<15.1f} | {tri_mem:<15.1f} | {sdpa_time:<10.2f} | {tri_time:<10.2f}")

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 5))
        
        # Memory Plot
        plt.subplot(1, 2, 1)
        plt.plot(seq_lengths, results["sdpa_mem"], marker='o', label='PyTorch SDPA')
        plt.plot(seq_lengths, results["triton_mem"], marker='s', label='Triton FlashAttn')
        plt.title('Peak VRAM Allocation (Forward+Backward)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Peak Memory (MB)')
        plt.legend()
        plt.grid(True)
        
        # Speed Plot
        plt.subplot(1, 2, 2)
        plt.plot(seq_lengths, results["sdpa_time"], marker='o', label='PyTorch SDPA')
        plt.plot(seq_lengths, results["triton_time"], marker='s', label='Triton FlashAttn')
        plt.title('Execution Time')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("benchmark_results.png")
        print("\nPlot saved to benchmark_results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d_head", type=int, default=64)
    args = parser.parse_args()
    
    benchmark_memory_and_speed(args.batch, args.heads, args.d_head)
