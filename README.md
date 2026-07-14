# LLM Systems Portfolio

End-to-end LLM engineering: custom Triton attention kernel → distributed fine-tuning → DPO alignment → production serving — built on free-tier compute, scoped with deliberate engineering trade-offs.

## Phase Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Requirements Engineering & ADR | ✅ Complete |
| 1 | Transformer from scratch (~25-60M params) | ✅ Complete |
| 2 | Triton fused attention kernel | ✅ Complete |
| 3 | Distributed fine-tune with FSDP (TinyLlama-1.1B) | ✅ Complete |
| 4 | DPO alignment | ✅ Complete |
| 4.5 | Engineering standards hardening | ✅ Complete |
| 5 | CI/CD, serving, security, observability | ✅ Complete |
| 5.5 | Cloud & IaC (documented, not deployed) | ✅ Complete |
| 6 | vLLM open-source contribution | ⬜ Not started |

## Key Architecture Decisions

See [`docs/adr/`](docs/adr/) for full decision records. Summary:

- **Architecture:** Monolith — single developer, two routes, low traffic.
- **Database:** SQLite — file-based, zero infra, single-writer workload.
- **Caching:** Redis for rate-limiting only — LLM sampling is stochastic, generation caching doesn't apply at temperature > 0.
- **Registry:** GHCR — free, no cloud credentials, same CD mechanic as ECR.
- **Monitoring:** Prometheus/Grafana + webhook alerting — not PagerDuty (no on-call rotation), not ELK (one service).
- **Message brokers:** Document-only — no cross-service async work exists to justify Kafka/RabbitMQ.

## Testable Claims

| Claim | Constraint | How verified |
|-------|-----------|--------------|
| Custom Triton kernel correctness | max abs diff < 1e-2 vs SDPA (fp16) | `pytest tests/unit/test_kernel.py` |
| Kernel VRAM advantage | Peak VRAM ≤ SDPA at seq_len ≥ 512 | Benchmark sweep plot |
| FSDP distributed training | 1.1B full fine-tune on 2×T4; single-GPU OOMs | Memory logs + OOM evidence |
| DPO alignment | Win-rate improvement on held-out prompts | Before/after eval |
| Serving latency | p95 < 2s for 64-token gen (from-scratch model, T4) | Load test with `httpx` |

## Repo Structure

```
llm-systems/
├── docs/adr/           # architecture decision records
├── model/              # from-scratch transformer (Phase 1)
├── kernels/            # Triton fused attention (Phase 2)
├── distributed/        # FSDP fine-tuning (Phase 3)
├── alignment/          # DPO training (Phase 4)
├── serving/            # FastAPI + Docker (Phase 5)
├── tests/{unit,integration,e2e}/
├── monitoring/         # Prometheus/Grafana configs
├── infra/              # Terraform (plan-only, never applied)
└── .github/workflows/  # CI + CD
```

## Environment

- **Phases 0–2:** Google Colab free tier (single T4, 16GB)
- **Phases 3–4:** Kaggle Notebooks (2×T4, 16GB each)
- **No paid cloud compute** — nothing in this project incurs a cloud bill.

## Interview Readiness

Every architecture decision in this project has a "why I chose this" and a "what would change my mind" answer. See the [ADR](docs/adr/0001-architecture.md) and the interview appendix in the project spec for the full list.

## License

MIT
