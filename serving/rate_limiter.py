import time
from typing import Optional
import redis
import os

class RateLimiter:
    """Redis-backed Token Bucket Rate Limiter."""
    
    def __init__(self, capacity: int, refill_rate: float, redis_url: str = "redis://localhost:6379/0"):
        self.capacity = capacity
        self.refill_rate = refill_rate # tokens per second
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
            self.enabled = True
        except redis.ConnectionError:
            print("⚠️ Redis not available. Rate limiting disabled.")
            self.enabled = False

    def is_allowed(self, user_id: str) -> bool:
        if not self.enabled:
            return True
            
        key = f"rate_limit:{user_id}"
        now = time.time()
        
        # Redis transaction (pipeline)
        pipe = self.redis.pipeline()
        
        try:
            # 1. Get current token count and last update time
            tokens, last_update = self.redis.hmget(key, ["tokens", "last_update"])
            
            if tokens is None:
                tokens = self.capacity
                last_update = now
            else:
                tokens = float(tokens)
                last_update = float(last_update)
                
            # 2. Refill bucket based on time passed
            time_passed = now - last_update
            new_tokens = min(self.capacity, tokens + time_passed * self.refill_rate)
            
            # 3. Check if we have enough tokens (1 request = 1 token)
            if new_tokens >= 1.0:
                new_tokens -= 1.0
                allowed = True
            else:
                allowed = False
                
            # 4. Save state back to Redis
            pipe.hset(key, mapping={"tokens": new_tokens, "last_update": now})
            pipe.expire(key, int(self.capacity / self.refill_rate) + 60) # TTL to clean up inactive users
            pipe.execute()
            
            return allowed
            
        except redis.RedisError as e:
            print(f"Redis error during rate limiting: {e}")
            return True # Fail open if Redis drops

# Global instance for the FastAPI app
# Capacity of 5 requests, refilling 1 request per second
limiter = RateLimiter(capacity=5, refill_rate=1.0, redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
