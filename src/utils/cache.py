from functools import wraps
import hashlib
import json
from typing import Any, Callable, Dict, Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis

from ..observability.logging import get_logger
from ..utils.config import get_settings
from ..utils.exceptions import CacheError

logger = get_logger(__name__)


def cache_key(*args: Any, **kwargs: Any) -> str:
    key_data = {
        "args": [str(arg) for arg in args],
        "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode("utf-8")).hexdigest()


class CacheManager:
   
    def __init__(self, redis_url: Optional[str] = None):
        settings = get_settings()
        self.redis_url = redis_url or getattr(settings, "redis_url", "redis://localhost:6379/0")
        self.default_ttl = getattr(settings, "cache_ttl", 3600)
        self._client: Optional[Redis] = None
        self._memory_store: Dict[str, str] = {}
        self.use_fallback = False

    async def get_client(self) -> Redis:
        if self._client is None and not self.use_fallback:
            try:
                self._client = await aioredis.from_url(self.redis_url,encoding="utf-8",decode_responses=True,)
                await self._client.ping()
                logger.info("Redis cache client connected successfully")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis at {self.redis_url} ({e}). Falling back to memory cache.")
                self.use_fallback = True
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        try:
            if self.use_fallback:
                val = self._memory_store.get(key)
                return json.loads(val) if val else None

            client = await self.get_client()
            if self.use_fallback or client is None:
                val = self._memory_store.get(key)
                return json.loads(val) if val else None

            val = await client.get(key)
            if val:
                logger.debug(f"Cache hit: {key}")
                return json.loads(val)
            logger.debug(f"Cache miss: {key}")
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
    
        try:
            serialized = json.dumps(value)
            expire = ttl or self.default_ttl

            if self.use_fallback:
                self._memory_store[key] = serialized
                return True

            client = await self.get_client()
            if self.use_fallback or client is None:
                self._memory_store[key] = serialized
                return True

            await client.setex(key, expire, serialized)
            logger.debug(f"Cache set: {key}")
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
       
        try:
            if self.use_fallback:
                return self._memory_store.pop(key, None) is not None

            client = await self.get_client()
            if self.use_fallback or client is None:
                return self._memory_store.pop(key, None) is not None

            await client.delete(key)
            logger.debug(f"Cache evict: {key}")
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        try:
            if self.use_fallback:
                prefix = pattern.replace("*", "")
                to_delete = [k for k in self._memory_store if k.startswith(prefix)]
                for k in to_delete:
                    del self._memory_store[k]
                return len(to_delete)

            client = await self.get_client()
            if self.use_fallback or client is None:
                prefix = pattern.replace("*", "")
                to_delete = [k for k in self._memory_store if k.startswith(prefix)]
                for k in to_delete:
                    del self._memory_store[k]
                return len(to_delete)

            keys = []
            async for k in client.scan_iter(match=pattern):
                keys.append(k)
            if keys:
                return await client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache clear_pattern error for {pattern}: {e}")
            return 0

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("Redis cache client closed")


_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_key = f"{key_prefix}{func.__name__}"
            arg_key = cache_key(*args, **kwargs)
            full_key = f"{func_key}:{arg_key}"

            cache = get_cache_manager()
            cached_val = await cache.get(full_key)
            if cached_val is not None:
                return cached_val

            result = await func(*args, **kwargs)
            await cache.set(full_key, result, ttl=ttl)
            return result

        return wrapper
    return decorator