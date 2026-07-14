import pytest
from fastapi.testclient import TestClient
from serving.app import app
from serving.auth import VALID_API_KEY
from serving.rate_limiter import limiter

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter for each test."""
    if limiter.enabled:
        limiter.redis.flushall()
    yield

def test_missing_api_key():
    response = client.post("/generate/base", json={"prompt": "Hello"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API Key"

def test_invalid_api_key():
    response = client.post("/generate/base", json={"prompt": "Hello"}, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401

def test_valid_api_key():
    response = client.post("/generate/base", json={"prompt": "Hello"}, headers={"X-API-Key": VALID_API_KEY})
    assert response.status_code == 200
    assert "response" in response.json()

def test_input_validation_empty_prompt():
    response = client.post("/generate/base", json={"prompt": ""}, headers={"X-API-Key": VALID_API_KEY})
    assert response.status_code == 422 # FastAPI Pydantic validation (min_length=1)

def test_input_validation_too_many_tokens():
    # Generate a massive prompt > 512 tokens
    massive_prompt = "hello " * 600
    response = client.post("/generate/base", json={"prompt": massive_prompt}, headers={"X-API-Key": VALID_API_KEY})
    assert response.status_code == 400
    assert "exceeds 512 tokens" in response.json()["detail"]

def test_rate_limiting():
    # If Redis is not available locally, skip the test
    if not limiter.enabled:
        pytest.skip("Redis not available for rate limiter testing")
        
    original_rate = limiter.refill_rate
    limiter.refill_rate = 0.0 # Prevent refill during the slow test requests
    
    try:
        # Capacity is 5, so 5 requests should succeed
        for _ in range(5):
            response = client.post("/generate/base", json={"prompt": "Hello"}, headers={"X-API-Key": VALID_API_KEY})
            assert response.status_code == 200
            
        # The 6th request should be rate limited (429)
        response = client.post("/generate/base", json={"prompt": "Hello"}, headers={"X-API-Key": VALID_API_KEY})
        assert response.status_code == 429
        assert response.json()["detail"] == "Too Many Requests"
    finally:
        limiter.refill_rate = original_rate

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests_total" in response.content
