from __future__ import annotations

import asyncio
import time

import pytest

from app.graph_memory import inference_priority


class FakePipeline:
    def __init__(self, client) -> None:
        self.client = client
        self.now = 0.0

    def zremrangebyscore(self, key, _minimum, maximum):
        self.now = float(maximum)
        return self

    def zcard(self, key):
        return self

    async def execute(self):
        expired = [token for token, score in self.client.leases.items() if score <= self.now]
        for token in expired:
            self.client.leases.pop(token, None)
        return [len(expired), len(self.client.leases)]


class FakeRedis:
    def __init__(self) -> None:
        self.leases = {}

    async def zadd(self, key, values):
        assert key.endswith(":leases")
        self.leases.update(values)
        return 1

    async def eval(self, script, _key_count, key, token, *args):
        assert key.endswith(":leases")
        if "zrem" in script:
            return int(self.leases.pop(token, None) is not None)
        if token not in self.leases:
            return 0
        self.leases[token] = float(args[0])
        return 1

    def pipeline(self, *, transaction):
        assert transaction is True
        return FakePipeline(self)

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_multiple_foreground_leases_block_background_until_all_exit(monkeypatch) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(inference_priority, "_redis_client", lambda: redis)

    assert await inference_priority.background_inference_allowed() is True
    async with inference_priority.foreground_inference_lease(ttl_seconds=30):
        assert await inference_priority.background_inference_allowed() is False
        async with inference_priority.foreground_inference_lease(ttl_seconds=30):
            assert len(redis.leases) == 2
            assert await inference_priority.background_inference_allowed() is False
        assert len(redis.leases) == 1
        assert await inference_priority.background_inference_allowed() is False
    assert await inference_priority.background_inference_allowed() is True

    redis.leases["stale"] = time.time() - 1
    assert await inference_priority.background_inference_allowed() is True
    assert redis.leases == {}


@pytest.mark.asyncio
async def test_hung_redis_acquisition_never_materially_delays_foreground(monkeypatch) -> None:
    class HungRedis(FakeRedis):
        async def zadd(self, _key, _values):
            await asyncio.Event().wait()

    redis = HungRedis()
    monkeypatch.setattr(inference_priority, "_redis_client", lambda: redis)
    started = time.monotonic()

    async with inference_priority.foreground_inference_lease(ttl_seconds=30):
        assert time.monotonic() - started < 0.5

    assert redis.leases == {}


@pytest.mark.asyncio
async def test_refresh_retries_and_republishes_after_transient_redis_failure() -> None:
    class FlakyRedis(FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self.refresh_calls = 0
            self.republished = asyncio.Event()

        async def eval(self, script, key_count, key, token, *args):
            if "zscore" in script:
                self.refresh_calls += 1
                if self.refresh_calls == 1:
                    self.leases.pop(token, None)
                    raise ConnectionError("transient refresh failure")
            return await super().eval(script, key_count, key, token, *args)

        async def zadd(self, key, values):
            result = await super().zadd(key, values)
            if self.refresh_calls:
                self.republished.set()
            return result

    redis = FlakyRedis()
    token = "still-open-chat"
    redis.leases[token] = time.time() + 1
    stop = asyncio.Event()
    task = asyncio.create_task(inference_priority._refresh(redis, token, 1, stop))
    try:
        await asyncio.wait_for(redis.republished.wait(), timeout=3)
        assert token in redis.leases
        assert redis.refresh_calls >= 2
    finally:
        stop.set()
        await task
