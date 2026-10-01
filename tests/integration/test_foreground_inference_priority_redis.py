from __future__ import annotations

import ipaddress
import os
import re
from urllib.parse import urlparse

import pytest
from redis.asyncio import from_url

from app.graph_memory import inference_priority
from tests.unit.test_foreground_inference_priority_integration import (
    run_concurrent_priority_drill,
)

pytestmark = pytest.mark.integration

_URL_ENV = "LAVIX_PRIORITY_TEST_REDIS_URL"
_DISPOSABLE_ENV = "LAVIX_PRIORITY_TEST_REDIS_DISPOSABLE"
_PROJECT_ENV = "COMPOSE_PROJECT_NAME"
_GENERATED_E2E_PROJECT = re.compile(r"lavix-e2e-[a-z0-9_-]+-[0-9a-f]{8}")


def _disposable_redis_url() -> str:
    url = os.environ.get(_URL_ENV, "").strip()
    if os.environ.get(_DISPOSABLE_ENV) != "1" or not url:
        pytest.skip(f"set {_DISPOSABLE_ENV}=1 and {_URL_ENV} to an empty disposable Redis database")
    parsed = urlparse(url)
    try:
        loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = False
    project = os.environ.get(_PROJECT_ENV, "").strip()
    generated_e2e_project = _GENERATED_E2E_PROJECT.fullmatch(project) is not None
    isolated_e2e_redis = parsed.hostname == "redis" and parsed.port == 6379 and generated_e2e_project
    if (
        parsed.scheme != "redis"
        or (not isolated_e2e_redis and (not loopback or parsed.port in {None, 6379}))
        or parsed.path != "/15"
        or parsed.username is not None
        or parsed.password is not None
    ):
        pytest.fail(
            f"{_URL_ENV} must be an unauthenticated Redis DB 15 URL that is either "
            "loopback on a non-default port or the redis:6379 service in an explicit "
            f"generated {_PROJECT_ENV}=lavix-e2e-...-<8 hex> project"
        )
    return url


def test_disposable_redis_url_accepts_generated_e2e_service_target(monkeypatch) -> None:
    url = "redis://redis:6379/15"
    monkeypatch.setenv(_DISPOSABLE_ENV, "1")
    monkeypatch.setenv(_URL_ENV, url)
    monkeypatch.setenv(_PROJECT_ENV, "lavix-e2e-local-0-job-smoke-12345-1b2e0ac5")

    assert _disposable_redis_url() == url


@pytest.mark.parametrize(
    "project",
    [
        "",
        "lavix-e2e-local-0-job-smoke-12345",
        "lavix-e2e-local-0-job-smoke-12345-nothex00",
        "lavix-vault",
    ],
)
def test_disposable_redis_url_rejects_service_target_without_generated_project(
    monkeypatch, project: str
) -> None:
    monkeypatch.setenv(_DISPOSABLE_ENV, "1")
    monkeypatch.setenv(_URL_ENV, "redis://redis:6379/15")
    if project:
        monkeypatch.setenv(_PROJECT_ENV, project)
    else:
        monkeypatch.delenv(_PROJECT_ENV, raising=False)

    with pytest.raises(pytest.fail.Exception, match="explicit generated"):
        _disposable_redis_url()


def test_disposable_redis_url_retains_loopback_fence_outside_e2e(monkeypatch) -> None:
    url = "redis://127.0.0.1:16379/15"
    monkeypatch.setenv(_DISPOSABLE_ENV, "1")
    monkeypatch.setenv(_URL_ENV, url)
    monkeypatch.delenv(_PROJECT_ENV, raising=False)

    assert _disposable_redis_url() == url


@pytest.mark.asyncio
async def test_real_redis_protocol_coordinates_chat_and_both_background_roles(monkeypatch) -> None:
    url = _disposable_redis_url()

    def redis_factory():
        return from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )

    observer = redis_factory()
    confirmed_empty = False
    try:
        assert await observer.ping() is True
        assert await observer.dbsize() == 0
        confirmed_empty = True

        async def active_lease_count() -> int:
            client = redis_factory()
            try:
                return int(await client.zcard(inference_priority._FOREGROUND_KEY))
            finally:
                await client.aclose()

        timeline = await run_concurrent_priority_drill(
            monkeypatch,
            redis_factory=redis_factory,
            active_lease_count=active_lease_count,
        )

        assert "intelligence:model:deferred" in timeline
        assert "memory:model:deferred" in timeline
        assert "intelligence:model:called" in timeline
        assert "memory:model:called" in timeline
        assert await observer.zcard(inference_priority._FOREGROUND_KEY) == 0
        assert await observer.dbsize() == 0
    finally:
        # Never clean a database that failed the initial empty-database fence.
        # Once fenced, the test may remove only keys it could have created in
        # its explicitly disposable database.
        if confirmed_empty:
            await observer.flushdb()
        await observer.aclose()
