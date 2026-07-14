# ADR-0001: Architecture Decisions — LLM Systems Portfolio

**Status:** Accepted
**Date:** 2026-07-12
**Author:** Gugha (solo developer)

---

## 1. Context

This project builds an end-to-end LLM engineering portfolio: custom Triton attention kernel,
distributed fine-tuning with FSDP, DPO alignment, and production-style serving. It runs
entirely on free-tier compute (Colab T4, Kaggle 2×T4) with zero cloud spend.

Before writing any code, this document turns the project's resume claims into testable
constraints and records every architecture decision with a justification and a scale-trigger
for reversal. Decisions are explicitly marked **[BUILD]** (implemented as running code) or
**[DOCUMENT ONLY]** (justified as not worth building at this scale).

---

## 2. Testable Requirements

Each resume claim is rewritten as a measurable, reproducible constraint.

### 2.1 Custom Triton Fused-Attention Kernel

| Requirement | Constraint | Verification |
|---|---|---|
| Correctness | Max absolute difference < 1e-2 vs `F.scaled_dot_product_attention` in fp16, across sequence lengths 128–4096 and random Q/K/V tensors | `pytest tests/unit/test_kernel.py` with `torch.allclose(atol=1e-2)` |
| VRAM advantage | Peak `torch.cuda.max_memory_allocated()` ≤ SDPA at seq_len ≥ 512; strictly less at seq_len ≥ 1024 | Benchmark sweep script, plotted |
| Reproducibility | Benchmark produces identical numerical results given a fixed seed | `--seed 42` flag, deterministic CUDA ops where possible |

**Why atol=1e-2 for fp16:** Block-tiled attention accumulates partial softmax results across
SRAM tiles. In fp16, each tile introduces rounding error that compounds multiplicatively
through the online softmax correction. A tolerance of 1e-2 on the absolute difference is
standard practice for fp16 fused attention implementations (FlashAttention's own tests use
similar thresholds). We also report max and mean absolute difference numerically alongside
the pass/fail check, so a reviewer can judge the distribution of errors, not just the worst
case.

### 2.2 FSDP Distributed Fine-Tuning

| Requirement | Constraint | Verification |
|---|---|---|
| Solves a real memory problem | Full fp32 Adam state for TinyLlama-1.1B (~16-18GB) exceeds single T4 (16GB VRAM) | Single-GPU OOM logged; 2-GPU FSDP run completes |
| Correct sharding | Training completes without OOM on 2×T4 with activation checkpointing | `torch.cuda.max_memory_allocated()` per GPU logged and graphed |

### 2.3 DPO Alignment

| Requirement | Constraint | Verification |
|---|---|---|
| Measurable behavior change | Win-rate or rubric-score improvement on held-out prompts (n ≥ 20) | Before/after eval on identical prompt set |

### 2.4 Serving

| Requirement | Constraint | Verification |
|---|---|---|
| Latency (from-scratch model) | p95 < 2s for 64-token generation on T4 | `httpx`-based load test, 50 sequential requests |
| Latency (TinyLlama-1.1B DPO) | p95 < 5s for 64-token generation on T4 | Same load test, separate route |
| Input validation | Reject payloads with empty prompt or token count > 512 | Integration test with invalid payloads |
| Rate limiting | Token-bucket rate limiter enforced (Redis-backed) | Test that exceeding rate returns 429 |
| Auth | API-key header required; missing/invalid key returns 401 | Integration test |

### 2.5 Reproducibility

All benchmarks and evaluations are reproducible from a single script invocation with a
fixed random seed. Checkpoint and evaluation artifacts are saved to Hugging Face Hub.

---

## 3. Architecture Decisions

### 3.1 Application Architecture — Monolith [BUILD]

**Decision:** Single FastAPI process serving both model routes (from-scratch + DPO-aligned
TinyLlama), packaged in one Docker container.

**Justification:** Single developer, two model routes, demo-only request volume. Microservices
would add:
- Inter-service networking and service discovery
- Separate deployment pipelines per service
- Distributed tracing to debug cross-service calls
- Container orchestration (Kubernetes) to manage multiple services

None of these costs have a matching benefit when one developer serves two routes from one
process. The request volume is effectively zero (demo/portfolio traffic). Microservices at
this scale is cargo-culting, not engineering.

**Scale-trigger to revisit:** If models needed independent scaling (e.g., a batch-eval worker
that should autoscale separately from an interactive endpoint), or if a second developer owned
one of the routes, microservices would begin to earn their deployment complexity.

---

### 3.2 Database — SQLite [BUILD]

**Decision:** SQLite for request logging, eval results, and serving metadata.

**Justification:**
- **File-based, zero infrastructure.** No database process to start, configure, or monitor.
  The database is a single file that ships with the container.
- **Single-writer, low-volume workload.** One FastAPI process, demo traffic only. SQLite
  handles hundreds of writes/second with WAL mode — orders of magnitude more than this
  project will ever see.
- **Fully structured data.** Every logged record has a known schema: timestamp, prompt,
  response, latency_ms, token_count, model_name.

**Alternatives considered:**

| Alternative | Why not |
|---|---|
| **PostgreSQL** | Requires a running database process. Value is concurrent-writer support and advanced queries — neither needed here. Adds operational complexity (connection pooling, migrations, backups) with no matching benefit for a single-writer demo. |
| **MongoDB** | Value is schema flexibility for heterogeneous documents. Our data is fully structured and uniform — a relational schema is simpler and more appropriate. |
| **DynamoDB** | Value is managed scalability and multi-region availability. Costs money, requires AWS credentials, and solves a scale problem that doesn't exist here. |

**Scale-trigger to revisit:** Once there are concurrent writers (multiple service replicas
writing simultaneously), migrate to PostgreSQL. The SQLite schema translates directly — this
is a deployment change, not a data-model change.

---

### 3.3 Caching — Redis for Rate Limiting Only [BUILD]

**Decision:** Redis is used **exclusively** for token-bucket rate limiting on the FastAPI
endpoint. It is **not** used for caching LLM generations.

**Why rate limiting (BUILD):**
A token-bucket rate limiter is a real, cheap, demonstrable backend skill. Implementation is
~20 lines of code plus a single Redis instance. It protects the GPU endpoint from abuse and
demonstrates understanding of a standard API-gateway pattern.

**Why NOT caching generations:**
LLM text generation is stochastic. With temperature > 0 (the default for creative text),
identical prompts produce different outputs on every call. Caching these responses would:
1. Return stale, non-representative outputs
2. Mask actual model behavior from the user
3. Add cache-invalidation complexity for zero benefit

Caching makes sense **only** for `temperature=0` deterministic requests (e.g., structured
extraction, classification). This is a narrow use case that doesn't justify the complexity
for a demo service.

**Scale-trigger for generation caching:** If this served production traffic with a meaningful
fraction of `temperature=0` requests, a Redis-backed semantic cache keyed on
`(prompt_hash, temperature, top_p, max_tokens)` would be justified. The cache key must
include all sampling parameters — not just the prompt — to avoid serving wrong results.

---

### 3.4 Message Brokers — Kafka/RabbitMQ/SQS [DOCUMENT ONLY]

**Decision:** No message broker is built or deployed.

**Justification:** A single-service, low-QPS demo has no asynchronous cross-service
communication need. The only async work (generation requests) is handled by FastAPI's
built-in async support. Message brokers solve three problems that don't exist here:
1. **Cross-service decoupling** — there is one service
2. **Workload buffering/backpressure** — request volume is negligible
3. **Event replay/ordering guarantees** — no downstream consumers need replay

Running Kafka for zero real throughput would read as cargo-culting in an interview, not
engineering judgment.

**Scale-trigger:**
- **Task queue (Celery + Redis, or SQS):** If async batch evaluation became a separate worker
  service — e.g., "run this eval suite on 1000 prompts and notify me when done."
- **Full Kafka/RabbitMQ:** If there were multiple consumer services needing replay, ordering
  guarantees, or fan-out (e.g., a logging pipeline, an analytics service, and a monitoring
  service all consuming generation events).

---

### 3.5 Container Registry — GHCR [BUILD]

**Decision:** Push Docker images to GitHub Container Registry (GHCR), not AWS ECR or
Google Artifact Registry.

**Justification:**
- **Free.** GHCR provides free container image hosting for public repositories.
- **No cloud credentials.** No AWS access keys, no IAM roles, no billing alerts. The CD
  pipeline authenticates with `GITHUB_TOKEN`, which GitHub Actions provides automatically.
- **Functionally identical CD mechanic.** The CI/CD pipeline does: build → tag with commit
  SHA → push to registry. The push target is a one-line config change (`docker push
  ghcr.io/...` → `docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/...`). The skill
  demonstrated — building a tagged image and pushing it from CI — is the same regardless
  of registry.

**Scale-trigger:** When actually deploying to AWS/GCP, swap the push target to ECR or
Artifact Registry. This is a 5-minute config change, not an architecture change.

---

### 3.6 Monitoring — Prometheus + Grafana [BUILD]

**Decision:** Prometheus scraping application metrics; Grafana for dashboards and alerting.

**Metrics exposed:**
- Request latency (histogram, by route/model)
- Request throughput (counter, by route/model)
- GPU memory utilization (`torch.cuda.max_memory_allocated()`)
- Error rate (counter, by status code)
- Active requests (gauge)

**Justification:** Prometheus + Grafana are free, self-hosted, and the industry standard for
application metrics. They run alongside the application in Docker Compose with minimal
resource overhead.

---

### 3.7 Logging — Structured JSON, not ELK [BUILD, right-sized]

**Decision:** FastAPI emits structured JSON logs to stdout. No Elasticsearch, Logstash, or
Kibana.

**Justification:** ELK (Elasticsearch + Logstash + Kibana) aggregates and searches logs
**across many services**. This project has one service. Structured JSON logs to stdout are:
- Searchable with `grep`/`jq` for a single container
- Ingestible by Grafana Loki (free tier) if a searchable dashboard is desired
- Zero additional infrastructure

**Scale-trigger:** If this grew to 3+ services where correlating logs across services became
necessary, ELK or a managed equivalent (Datadog, CloudWatch Logs Insights) would be
justified.

---

### 3.8 Alerting — Grafana Webhook, not PagerDuty [BUILD, right-sized]

**Decision:** Grafana's built-in alerting sends webhooks to a personal Discord/Slack channel
when thresholds are crossed. Not PagerDuty.

**Alert rules (two only, to avoid alert fatigue):**
1. **p95 latency > 3s** for either route, sustained over 5 minutes
2. **Error rate > 5%** over a 5-minute window

**Why two alerts, not twenty:** Alert fatigue is the primary failure mode of monitoring
systems. Two high-signal alerts that each require action are better than twenty low-signal
alerts that get ignored. Each alert has a clear action:
- High latency → check GPU utilization, check for OOM, check batch size
- High error rate → check logs for stack traces, check model loading

**Why not PagerDuty:**
- Costs money ($21+/user/month)
- Assumes an on-call rotation — there is one developer, always on-call by default
- The value of PagerDuty is escalation policies and multi-person rotation scheduling,
  neither of which applies to a solo project

**Scale-trigger:** When there is a team with an actual on-call rotation and SLA obligations,
PagerDuty (or Opsgenie) earns its cost through escalation policies and schedule management.

---

### 3.9 Security Posture

| Control | Scope | Implementation | Justification |
|---|---|---|---|
| API-key auth | **[BUILD]** | `X-API-Key` header checked against env var | Single-consumer demo; simplest viable auth that demonstrates the pattern |
| OAuth2/JWT | **[DOCUMENT ONLY]** | Noted in README as the multi-tenant extension | Single consumer, no user management needed; OAuth2 adds token refresh, JWKS rotation, and identity-provider integration that serve no purpose here |
| Input validation | **[BUILD]** | Pydantic models: reject empty prompts, cap `max_tokens` at 512, validate types | OWASP-relevant boundary control; prevents resource exhaustion on GPU |
| `pip-audit` in CI | **[BUILD]** | GitHub Actions step on every push | Dependency vulnerability scanning, free, zero configuration |
| `safetensors` format | **[BUILD]** | All checkpoints saved/loaded via `safetensors` | Never `pickle.load` untrusted data — pickle deserialization is arbitrary code execution |
| Pinned dependencies | **[BUILD]** | `requirements.txt` with exact versions | Reproducibility + supply-chain hygiene; prevents silent breakage from upstream updates |
| HTTPS/TLS | **[DOCUMENT ONLY]** | Noted in cloud ADR (Phase 5.5) | TLS termination happens at the load balancer in production, not in the application; demo runs on localhost |

---

## 4. Explicit Non-Goals

These are technologies and patterns that are **deliberately not built** in this project.
Each has a stated reason and a trigger that would change the decision.

| Technology | Why not built | Scale-trigger |
|---|---|---|
| **Kubernetes** | One container, one service, one developer. K8s solves multi-service orchestration, rolling updates across a fleet, and auto-scaling — none of which apply. | Multiple services with independent scaling needs and a team to operate the cluster. |
| **Terraform `apply`** | No cloud resources to provision. `terraform plan` proves IaC fluency without risking a forgotten instance billing at 2am. | Actual cloud deployment with real traffic and a budget. |
| **Blue-Green / Canary deployments** | A demo endpoint with no real traffic has no blast-radius problem. There are no users to protect from a bad deploy. | Multiple paying users where a bad deploy has business impact. |
| **Full ELK stack** | One service — `grep`/`jq` on structured JSON logs is sufficient. | 3+ services where cross-service log correlation is needed. |
| **PagerDuty** | Costs money, assumes an on-call rotation. Solo developer is always on-call by default. | A team with SLA obligations and a rotation schedule. |
| **OAuth2/JWT** | Single-consumer demo, no user management. API-key auth demonstrates the pattern. | Multi-tenant access with user-level permissions. |

---

## 5. Technology Stack Summary

| Component | Choice | Phase |
|---|---|---|
| Language | Python 3.11+ | All |
| Deep Learning | PyTorch, Triton | 1, 2 |
| Distributed Training | `torch.distributed`, FSDP | 3 |
| Alignment | DPO (custom loss + trl plumbing) | 4 |
| Tokenizer | HF `tokenizers` (BPE) or GPT-2 | 1 |
| Dataset | TinyStories | 1 |
| Serving | FastAPI | 5 |
| Containerization | Docker | 5 |
| CI/CD | GitHub Actions → GHCR | 5 |
| Database | SQLite | 5 |
| Rate Limiting | Redis (token-bucket) | 5 |
| Monitoring | Prometheus + Grafana | 5 |
| Alerting | Grafana → Discord/Slack webhook | 5 |
| IaC | Terraform (plan only) | 5.5 |
| Experiment Tracking | Weights & Biases (free tier) | 1–4 |
| Checkpointing | HF Hub + safetensors | 1–4 |

---

## 6. Compute Environment

| Phase | Environment | GPU | VRAM | Constraint |
|---|---|---|---|---|
| 0–2 | Google Colab (free) | 1× T4 | 16 GB | Session disconnects; checkpoint to HF Hub |
| 3–4 | Kaggle Notebooks | 2× T4 | 16 GB each | 30 GPU-hrs/week quota |
| 5–5.5 | Local / GitHub Actions | CPU | — | No GPU needed for serving setup/CI |
| 6 | N/A | — | — | Contribution to upstream repo |

**Hard constraint:** Nothing in this project ever incurs a cloud bill.
