import time
import json
import logging
import sys
import os
from fastapi import FastAPI, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from serving.auth import verify_api_key
from serving.rate_limiter import limiter
from serving.db import log_request_async

# Setup structured JSON logging
logger = logging.getLogger("fastapi_serving")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'))
logger.addHandler(handler)

# Prometheus Metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_latency_seconds", "HTTP request latency", ["endpoint", "model"])
ACTIVE_REQUESTS = Gauge("http_active_requests", "Currently active requests")
GPU_MEMORY = Gauge("gpu_memory_allocated_bytes", "GPU memory allocated by PyTorch")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="LLM Serving API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2"

print("Loading models at startup...")
try:
    tokenizer = AutoTokenizer.from_pretrained("dpo_aligned_model")
    aligned_model = AutoModelForCausalLM.from_pretrained("dpo_aligned_model").to(device)
    aligned_model.eval()
    print("Loaded aligned model.")
except Exception as e:
    print(f"Aligned model not found locally ({e}). Falling back to random weights.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    aligned_model = AutoModelForCausalLM.from_config(config).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

hf_token = os.environ.get("HF_TOKEN")
print(f"Loading base model ({model_name}) from HuggingFace...")
base_model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(device)
base_model.eval()
print("Models loaded successfully.")

# --- API Models ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(50, le=512)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

class GenerateResponse(BaseModel):
    response: str
    latency_ms: float
    token_count: int

# --- Middleware ---
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    response = await call_next(request)
    
    latency = time.time() - start_time
    ACTIVE_REQUESTS.dec()
    
    # Update metrics
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
    
    if torch.cuda.is_available():
        GPU_MEMORY.set(torch.cuda.max_memory_allocated())
        
    return response

# --- Endpoints ---
@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def _generate(model, req: GenerateRequest, model_name_str: str) -> GenerateResponse:
    start_time = time.time()
    
    # Input validation (ADR: reject token count > 512)
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
    input_tokens = inputs["input_ids"].shape[1]
    
    if input_tokens > 512:
        raise HTTPException(status_code=400, detail="Prompt exceeds 512 tokens.")
        
    with torch.no_grad():
        with REQUEST_LATENCY.labels(endpoint=f"/generate/{model_name_str}", model=model_name_str).time():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=req.temperature > 0,
                temperature=req.temperature if req.temperature > 0 else 1.0,
            )
            
    response_ids = outputs[0][input_tokens:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    latency_ms = (time.time() - start_time) * 1000
    token_count = len(response_ids)
    
    # Async SQLite logging
    log_request_async(model_name_str, req.prompt, response_text, latency_ms, token_count, 200)
    
    # Structured JSON log
    logger.info(json.dumps({
        "event": "generation_complete",
        "model": model_name_str,
        "latency_ms": latency_ms,
        "tokens": token_count
    }))
    
    return GenerateResponse(response=response_text, latency_ms=latency_ms, token_count=token_count)


@app.post("/generate/base", response_model=GenerateResponse)
def generate_base(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    """Generate text using the unaligned base model."""
    if not limiter.is_allowed(api_key):
        raise HTTPException(status_code=429, detail="Too Many Requests")
    return _generate(base_model, req, "base")


@app.post("/generate/aligned", response_model=GenerateResponse)
def generate_aligned(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    """Generate text using the DPO-aligned model."""
    if not limiter.is_allowed(api_key):
        raise HTTPException(status_code=429, detail="Too Many Requests")
    return _generate(aligned_model, req, "aligned")
