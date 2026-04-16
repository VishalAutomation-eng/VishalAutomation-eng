import hashlib
import json

import redis

from app.core.config import settings

redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


def cache_key(user_id: str, query: str, filters: dict | None) -> str:
    payload = json.dumps({'u': user_id, 'q': query, 'f': filters}, sort_keys=True)
    return f"rag:{hashlib.sha256(payload.encode()).hexdigest()}"


def get_cached_response(key: str) -> str | None:
    return redis_client.get(key)


def set_cached_response(key: str, value: str, ttl: int = 1800) -> None:
    redis_client.setex(key, ttl, value)
