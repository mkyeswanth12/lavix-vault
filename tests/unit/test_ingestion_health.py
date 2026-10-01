from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ingestion import health


def test_health_payload_is_strictly_validated() -> None:
    payload = {
        "status": "ready",
        "worker_id": "worker-1",
        "observed_at": "2026-07-15T10:00:00+00:00",
        "pdf_verified_at": "2026-07-15T09:59:00+00:00",
        "detail": None,
        "consecutive_failures": 0,
        "verification_source": "probe",
    }
    assert health.decode_worker_health(json.dumps(payload).encode()) == payload
    assert health.decode_worker_health(b"not-json") is None
    assert health.decode_worker_health(json.dumps({**payload, "status": "forged"})) is None


@pytest.mark.asyncio
async def test_odl_probe_uses_a_real_ephemeral_pdf(tmp_path: Path) -> None:
    seen = []

    class Registry:
        settings = SimpleNamespace(opendataloader_hybrid="off")

        async def parse(self, route, request):
            seen.append((route, request.path.read_bytes()))
            assert request.path.parent.parent == tmp_path
            return SimpleNamespace(elements=(object(),), metadata={"parser": "opendataloader"})

    await health.OpenDataLoaderReadinessProbe(Registry(), tmp_path)()

    assert seen and seen[0][1].startswith(b"%PDF-1.4")
    assert seen[0][1].endswith(b"%%EOF\n")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_odl_probe_runs_exactly_one_end_to_end_parser_conversion(tmp_path: Path) -> None:
    calls = []

    class Registry:
        settings = SimpleNamespace(
            opendataloader_hybrid="docling-fast",
            opendataloader_hybrid_url="http://odl-hybrid:5002",
            opendataloader_hybrid_timeout_ms=90_000,
        )

        async def parse(self, route, request):
            calls.append((route, request.path.read_bytes()))
            return SimpleNamespace(elements=(object(),), metadata={"parser": "opendataloader"})

    await health.OpenDataLoaderReadinessProbe(Registry(), tmp_path)()

    assert len(calls) == 1
    assert calls[0][1].startswith(b"%PDF-1.4")


@pytest.mark.asyncio
async def test_odl_probe_rejects_a_pdf_that_needed_the_ocr_fallback(tmp_path: Path) -> None:
    class Registry:
        settings = SimpleNamespace(opendataloader_hybrid="off")

        async def parse(self, _route, _request):
            return SimpleNamespace(elements=(object(),), metadata={"parser": "docling"})

    with pytest.raises(RuntimeError, match="required the OCR fallback"):
        await health.OpenDataLoaderReadinessProbe(Registry(), tmp_path)()


@pytest.mark.asyncio
async def test_reporter_requires_two_failures_before_degraded(monkeypatch) -> None:
    class RedisClient:
        def __init__(self) -> None:
            self.values = []

        async def set(self, key, value, ex):
            self.values.append((key, json.loads(value), ex))

        async def aclose(self):
            return None

    client = RedisClient()
    monkeypatch.setattr(health.Redis, "from_url", lambda *_args, **_kwargs: client)

    async def failed_probe():
        raise TimeoutError("converter queue is saturated")

    reporter = health.RedisWorkerHealthReporter(
        "redis://example.invalid/0",
        "worker-1",
        failed_probe,
        heartbeat_seconds=0.01,
        probe_seconds=1,
        ttl_seconds=2,
    )
    await reporter._run_probe_once()
    assert reporter.status == "busy"
    assert reporter.consecutive_failures == 1

    await reporter._run_probe_once()
    assert reporter.status == "degraded"
    assert reporter.consecutive_failures == 2
    assert [item[1]["status"] for item in client.values] == ["busy", "degraded"]


@pytest.mark.asyncio
async def test_recent_real_pdf_success_keeps_transient_probe_failures_busy(monkeypatch) -> None:
    class RedisClient:
        async def set(self, _key, _value, ex):
            assert ex == 2

        async def aclose(self):
            return None

    monkeypatch.setattr(health.Redis, "from_url", lambda *_args, **_kwargs: RedisClient())

    async def failed_probe():
        raise TimeoutError("converter queue is saturated")

    reporter = health.RedisWorkerHealthReporter(
        "redis://example.invalid/0",
        "worker-1",
        failed_probe,
        heartbeat_seconds=0.01,
        probe_seconds=1,
        ttl_seconds=2,
        recent_success_seconds=60,
    )
    await reporter.record_pdf_success()
    verified_at = reporter.pdf_verified_at
    await reporter._run_probe_once()
    await reporter._run_probe_once()

    assert reporter.status == "busy"
    assert reporter.pdf_verified_at == verified_at
    assert reporter.consecutive_failures == 2
    assert "last verified" in str(reporter.detail)


@pytest.mark.asyncio
async def test_recent_real_pdf_success_skips_an_unnecessary_synthetic_probe(monkeypatch) -> None:
    class RedisClient:
        async def set(self, _key, _value, ex):
            assert ex == 2

        async def aclose(self):
            return None

    monkeypatch.setattr(health.Redis, "from_url", lambda *_args, **_kwargs: RedisClient())
    calls = 0

    async def probe():
        nonlocal calls
        calls += 1

    reporter = health.RedisWorkerHealthReporter(
        "redis://example.invalid/0",
        "worker-1",
        probe,
        heartbeat_seconds=0.01,
        probe_seconds=1,
        ttl_seconds=2,
        recent_success_seconds=60,
    )
    await reporter.record_pdf_success()
    await reporter._refresh_or_probe_once()

    assert calls == 0
    assert reporter.status == "ready"
    assert reporter.verification_source == "real_pdf"
