import pytest

from src.utils.cache import CacheManager, cache_key, cached, get_cache_manager


def test_cache_key_generation():
    key1 = cache_key("user_123", limit=10, sort="asc")
    key2 = cache_key("user_123", sort="asc", limit=10)
    key3 = cache_key("user_456", limit=10, sort="asc")

    # Order of kwargs should not affect the hash
    assert key1 == key2
    assert key1 != key3


@pytest.mark.asyncio
async def test_cache_manager_memory_fallback():
    manager = CacheManager(redis_url="redis://invalid-host:9999/0")
    manager.use_fallback = True  # force in-memory behavior

    # Set and Get
    assert await manager.set("query:1", {"result": "retrieved data"}) is True
    val = await manager.get("query:1")
    assert val == {"result": "retrieved data"}

    # Cache miss
    assert await manager.get("query:nonexistent") is None

    # Delete
    assert await manager.delete("query:1") is True
    assert await manager.get("query:1") is None


@pytest.mark.asyncio
async def test_cache_clear_pattern():
    manager = CacheManager()
    manager.use_fallback = True

    await manager.set("doc:1", "data1")
    await manager.set("doc:2", "data2")
    await manager.set("user:1", "data3")

    deleted_count = await manager.clear_pattern("doc:*")
    assert deleted_count == 2
    assert await manager.get("doc:1") is None
    assert await manager.get("user:1") == "data3"


@pytest.mark.asyncio
async def test_cached_decorator():
    call_counter = 0
    manager = get_cache_manager()
    manager.use_fallback = True

    @cached(ttl=60, key_prefix="test_fn:")
    async def expensive_computation(x: int, y: int) -> int:
        nonlocal call_counter
        call_counter += 1
        return x + y

    res1 = await expensive_computation(5, 10)
    assert res1 == 15
    assert call_counter == 1

    # Second call with same args hits cache; execution count does not increase
    res2 = await expensive_computation(5, 10)
    assert res2 == 15
    assert call_counter == 1

    # Different arguments trigger re-execution
    res3 = await expensive_computation(10, 20)
    assert res3 == 30
    assert call_counter == 2