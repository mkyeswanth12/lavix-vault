"""Best-effort foreground lease that keeps background Ollama work deferential."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

from app.config import settings

logger = logging.getLogger(__name__)

_FOREGROUND_KEY = "lavix:inference:foreground:leases"
_ACQUIRE_TIMEOUT_SECONDS = 0.25
_RELEASE_TIMEOUT_SECONDS = 0.25
_RELEASE_IF_OWNED = """
return redis.call('zrem', KEYS[1], ARGV[1])
"""
_REFRESH_IF_OWNED = """
if redis.call('zscore', KEYS[1], ARGV[1]) then
  return redis.call('zadd', KEYS[1], ARGV[2], ARGV[1])
end
return 0
"""


class BackgroundInferenceDeferred(RuntimeError):
    """Background model work must yield without consuming a job attempt."""


def _redis_client():
    from redis.asyncio import from_url

    return from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=0.2,
        socket_timeout=0.5,
    )


async def _refresh(client, token: str, ttl_seconds: int, stop: asyncio.Event) -> None:
    interval = max(1, ttl_seconds // 3)
    delay = interval
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
            return
        except TimeoutError:
            try:
                expires_at = time.time() + ttl_seconds
                owned = await client.eval(
                    _REFRESH_IF_OWNED,
                    1,
                    _FOREGROUND_KEY,
                    token,
                    expires_at,
                )
                if not owned:
                    # A Redis outage may outlive the previous TTL. This token is
                    # unique to the still-open context, so republishing it is
                    # safe until the context explicitly stops the refresher.
                    await client.zadd(_FOREGROUND_KEY, {token: expires_at})
                delay = interval
            except Exception:
                logger.warning("Unable to refresh foreground inference lease")
                delay = min(1.0, max(0.1, ttl_seconds / 20))


@asynccontextmanager
async def foreground_inference_lease(*, ttl_seconds: int = 240):
    """Advertise an interactive Ollama run without making chat depend on Redis."""

    client = None
    token = str(uuid4())
    stop = asyncio.Event()
    refresher: asyncio.Task | None = None
    acquired = False
    try:
        client = _redis_client()
        acquired = bool(
            await asyncio.wait_for(
                client.zadd(_FOREGROUND_KEY, {token: time.time() + ttl_seconds}),
                timeout=_ACQUIRE_TIMEOUT_SECONDS,
            )
        )
        if acquired:
            refresher = asyncio.create_task(_refresh(client, token, ttl_seconds, stop))
    except Exception:
        # Scheduling telemetry must never make foreground chat unavailable.
        logger.warning("Unable to publish foreground inference lease")
    try:
        yield
    finally:
        stop.set()
        if refresher is not None:
            refresher.cancel()
            with suppress(asyncio.CancelledError):
                await refresher
        if acquired and client is not None:
            with suppress(Exception):
                await asyncio.wait_for(
                    client.eval(_RELEASE_IF_OWNED, 1, _FOREGROUND_KEY, token),
                    timeout=_RELEASE_TIMEOUT_SECONDS,
                )
        if client is not None:
            with suppress(Exception):
                await client.aclose()


async def background_inference_allowed() -> bool:
    """Fail closed when the scheduler cannot prove Ollama is free."""

    client = None
    try:
        client = _redis_client()
        now = time.time()
        pipeline = client.pipeline(transaction=True)
        pipeline.zremrangebyscore(_FOREGROUND_KEY, "-inf", now)
        pipeline.zcard(_FOREGROUND_KEY)
        _, active = await pipeline.execute()
        return int(active) == 0
    except Exception:
        logger.warning("Deferring background inference because its priority gate is unavailable")
        return False
    finally:
        if client is not None:
            with suppress(Exception):
                await client.aclose()


__all__ = [
    "BackgroundInferenceDeferred",
    "background_inference_allowed",
    "foreground_inference_lease",
]
