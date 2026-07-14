# LLM-RAG System

A high-performance, from-scratch implementation of a Large Language Model serving and training pipeline.

This repository demonstrates a complete, production-ready LLM system. It includes a custom transformer implementation, optimized GPU kernels, distributed fine-tuning capabilities, and a scalable serving infrastructure.

## Features

- **Custom Transformer Architecture:** A full transformer implementation (~25-60M parameters) built from scratch in PyTorch, featuring Rotary Positional Embeddings (RoPE).
- **Triton Fused Attention:** A highly optimized, custom Triton kernel for causal self-attention that significantly reduces VRAM usage and increases throughput compared to standard PyTorch implementations.
- **Distributed Training (FSDP):** Scalable fine-tuning scripts utilizing PyTorch Fully Sharded Data Parallel (FSDP) to train models like TinyLlama-1.1B across multiple GPUs.
- **DPO Alignment:** Built-in Direct Preference Optimization (DPO) implementation for aligning models with human preferences using custom loss functions.
- **Production Serving API:** A high-throughput FastAPI inference server utilizing the Factory Pattern for secure checkpoint loading (safetensors only) and the Strategy Pattern for attention backend selection.
- **Observability:** Prometheus integration for granular metrics (e.g., token generation latency, GPU memory usage) and rate-limiting via ElastiCache Redis.
- **Infrastructure as Code:** Terraform configurations (`infra/terraform/`) for deploying the serving API to AWS ECS Fargate.

## Architecture

Please see the [`docs/adr/`](docs/adr/) directory for comprehensive Architecture Decision Records. 

Key architectural components include:
- **Serving:** Monolithic FastAPI application handling both routing and generation.
- **Storage:** SQLite for robust, single-writer request logging without the overhead of heavy observability stacks.
- **Rate Limiting:** Token-bucket rate limiting backed by Redis.
- **Deployment:** Containerized via Docker and orchestrated via GitHub Actions to GHCR.

## Quick Start

### 1. Installation

Ensure you have Python 3.10+ and PyTorch installed.

```bash
git clone https://github.com/GughanS/llm_rag.git
cd llm_rag
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Running the Tests

To verify the custom Triton kernels against PyTorch's native SDPA:

```bash
pytest tests/unit/test_kernel.py
```

### 3. Starting the Inference Server

Start the FastAPI application locally:

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

Or run the full stack (API, Redis, Prometheus, Grafana) via Docker Compose:

```bash
docker-compose up -d
```

## Documentation

- [Architecture Decision Records](docs/adr/)
- [Architecture Patterns](docs/architecture-patterns.md)
- [Proposed vLLM Kernel Enhancements](docs/vllm-geglu-kernel-pr.md)

## License

MIT License
