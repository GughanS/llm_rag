#!/bin/bash
# Benchmark script to prove Phase 3 ADR constraints

set -e

MODE=$1

if [ "$MODE" = "single" ]; then
    echo "========================================================="
    echo "Running Single GPU Baseline (Expected to OOM)"
    echo "========================================================="
    # Run on a single GPU without FSDP
    python -m distributed.train_fsdp --batch_size 2 --seq_len 512
elif [ "$MODE" = "distributed" ]; then
    echo "========================================================="
    echo "Running Distributed FSDP (Expected to Succeed)"
    echo "========================================================="
    # Run on 2 GPUs with FSDP and activation checkpointing
    torchrun --nproc_per_node=2 -m distributed.train_fsdp --fsdp --activation_checkpointing --batch_size 2 --seq_len 512
else
    echo "Usage: bash distributed/run_benchmark.sh [single|distributed]"
    exit 1
fi
