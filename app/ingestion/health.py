"""Worker-owned readiness evidence for the network-isolated PDF converter."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from redis.asyncio import Redis

from .parsers.base import ParseRequest
from .parsers.registry import ParserRegistry
from .routing import route_file
from .security import SecureTempLease

logger = logging.getLogger(__name__)

WORKER_HEALTH_KEY = "lavix:ingestion:worker:health"
_HEALTH_STATES = frozenset({"checking", "ready", "busy", "degraded"})


def decode_worker_health(raw: bytes | str | None) -> dict[str, Any] | None:
    """Validate untrusted Redis health data before exposing it through the API."""

    if raw is None:
        return None
    try:
        payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (UnicodeError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("status") not in _HEALTH_STATES:
        return None
    worker_id = payload.get("worker_id")
    observed_at = payload.get("observed_at")
    if not isinstance(worker_id, str) or not worker_id or not isinstance(observed_at, str):
        return None
    return {
        "status": str(payload["status"]),
        "worker_id": worker_id[:200],
        "observed_at": observed_at[:64],
        "pdf_verified_at": (
            str(payload["pdf_verified_at"])[:64] if isinstance(payload.get("pdf_verified_at"), str) else None
        ),
        "detail": str(payload["detail"])[:500] if isinstance(payload.get("detail"), str) else None,
        "consecutive_failures": (
            min(1000, max(0, int(payload["consecutive_failures"])))
            if isinstance(payload.get("consecutive_failures"), int)
            and not isinstance(payload.get("consecutive_failures"), bool)
            else 0
        ),
        "verification_source": (
            str(payload["verification_source"])
            if payload.get("verification_source") in {"probe", "real_pdf"}
            else None
        ),
    }


class OpenDataLoaderReadinessProbe:
    """Run a tiny real PDF through the same parser used for user documents."""

    def __init__(self, registry: ParserRegistry, temp_root: Path) -> None:
        self.registry = registry
        self.temp_root = temp_root

    async def __call__(self) -> None:
        payload = _minimal_pdf("Lavix OpenDataLoader readiness")
        with SecureTempLease(self.temp_root, prefix="odl-health-") as lease:
            source = lease.allocate("readiness.pdf")
            await asyncio.to_thread(source.write_bytes, payload)
            document = await self.registry.parse(
                route_file("readiness.pdf", "application/pdf"),
                ParseRequest(
                    path=source,
                    source_name="readiness.pdf",
                    media_type="application/pdf",
                ),
            )
        if not document.elements:
            raise RuntimeError("OpenDataLoader readiness PDF produced no elements")
        if document.metadata.get("parser") != "opendataloader":
            raise RuntimeError("OpenDataLoader readiness PDF required the OCR fallback")


class RedisWorkerHealthReporter:
    """Publish expiring liveness plus verified ODL readiness to Redis."""

    def __init__(
        self,
        redis_url: str,
        worker_id: str,
        probe: Callable[[], Awaitable[None]],
        *,
        heartbeat_seconds: float = 20.0,
        probe_seconds: float = 900.0,
        ttl_seconds: int = 75,
        degrade_after_failures: int = 2,
        recent_success_seconds: float | None = None,
    ) -> None:
        if (
            heartbeat_seconds <= 0
            or probe_seconds <= 0
            or ttl_seconds <= heartbeat_seconds
            or degrade_after_failures < 2
        ):
            raise ValueError("invalid worker health timing")
        self.client = Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        self.worker_id = worker_id
        self.probe = probe
        self.heartbeat_seconds = heartbeat_seconds
        self.probe_seconds = probe_seconds
        self.ttl_seconds = ttl_seconds
        self.degrade_after_failures = degrade_after_failures
        self.recent_success_seconds = (
            float(recent_success_seconds)
            if recent_success_seconds is not None
            else max(1800.0, probe_seconds * 2)
        )
        if self.recent_success_seconds <= 0:
            raise ValueError("recent_success_seconds must be positive")
        self.status = "checking"
        self.pdf_verified_at: str | None = None
        self.real_pdf_verified_at: str | None = None
        self.verification_source: str | None = None
        self.detail: str | None = None
        self.consecutive_failures = 0

    async def run(self, stop_event: asyncio.Event) -> None:
        probe_task = asyncio.create_task(self._probe_loop(stop_event))
        try:
            while not stop_event.is_set():
                await self._publish()
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.heartbeat_seconds)
                except TimeoutError:
                    pass
        finally:
            probe_task.cancel()
            await asyncio.gather(probe_task, return_exceptions=True)
            await self.client.aclose()

    async def record_pdf_success(self) -> None:
        verified_at = _utc_now()
        self.status = "ready"
        self.pdf_verified_at = verified_at
        self.real_pdf_verified_at = verified_at
        self.verification_source = "real_pdf"
        self.detail = None
        self.consecutive_failures = 0
        await self._publish()

    async def _probe_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await self._refresh_or_probe_once()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.probe_seconds)
            except TimeoutError:
                pass

    async def _refresh_or_probe_once(self) -> None:
        if self._has_recent_real_pdf_success():
            # A user's successful conversion exercises more of the real path
            # than another synthetic document and avoids adding load to a busy
            # converter queue.
            self.status = "ready"
            self.pdf_verified_at = self.real_pdf_verified_at
            self.verification_source = "real_pdf"
            self.detail = None
            self.consecutive_failures = 0
            await self._publish()
            return
        await self._run_probe_once()

    async def _run_probe_once(self) -> None:
        try:
            await self.probe()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.consecutive_failures += 1
            raw_detail = f"{type(exc).__name__}: {exc}"[:400]
            recently_verified = self._has_recent_pdf_success()
            if self.consecutive_failures < self.degrade_after_failures or recently_verified:
                self.status = "busy"
                suffix = f"; last verified {self.pdf_verified_at}" if self.pdf_verified_at else ""
                self.detail = f"Temporary probe failure ({raw_detail}){suffix}"[:500]
            else:
                self.status = "degraded"
                self.detail = raw_detail
            logger.warning(
                "OpenDataLoader readiness probe failed (%s/%s): %s",
                self.consecutive_failures,
                self.degrade_after_failures,
                raw_detail,
            )
        else:
            self.status = "ready"
            self.pdf_verified_at = _utc_now()
            self.verification_source = "probe"
            self.detail = None
            self.consecutive_failures = 0
        await self._publish()

    def _has_recent_pdf_success(self) -> bool:
        verified = _parse_utc(self.pdf_verified_at)
        if verified is None:
            return False
        return datetime.now(UTC) - verified <= timedelta(seconds=self.recent_success_seconds)

    def _has_recent_real_pdf_success(self) -> bool:
        verified = _parse_utc(self.real_pdf_verified_at)
        if verified is None:
            return False
        return datetime.now(UTC) - verified <= timedelta(seconds=self.recent_success_seconds)

    async def _publish(self) -> None:
        payload = json.dumps(
            {
                "status": self.status,
                "worker_id": self.worker_id,
                "observed_at": _utc_now(),
                "pdf_verified_at": self.pdf_verified_at,
                "detail": self.detail,
                "consecutive_failures": self.consecutive_failures,
                "verification_source": self.verification_source,
            },
            separators=(",", ":"),
        )
        try:
            await self.client.set(WORKER_HEALTH_KEY, payload, ex=self.ttl_seconds)
        except Exception as exc:
            logger.warning("Could not publish ingestion-worker health: %s", type(exc).__name__)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _minimal_pdf(text: str) -> bytes:
    """Create one deterministic, dependency-free PDF page for a health probe."""

    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 36 72 Td ({escaped}) Tj ET".encode("ascii")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode("ascii") + b">>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    )
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(body)
        document.extend(b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(document)
