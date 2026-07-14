import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

# In production, this would be injected via secrets manager or env var
# For this portfolio demo, we'll accept a dummy key or fall back to an env var
VALID_API_KEY = os.getenv("API_KEY", "test_sk_12345")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key == VALID_API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )
