from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from agent_runtime.events import AUTHORITATIVE_STREAM_PROVENANCE
from app.agent.capability import CapabilitySigner
from app.agent.evidence import RunEvidenceStore
from app.graph_memory import inference_priority
from app.graph_memory.worker import (
    GraphMemoryJob,
    GraphMemoryWorker,
    PersistedUserMessage,
)
from app.ingestion.intelligence import (
    ConfiguredDocumentIntelligenceService,
    DocumentIntelligenceService,
    IntelligenceSettings,
)
from app.ingestion.models import CanonicalChunk, CanonicalDocument, Provenance
from app.routers.chat.orchestrator import ChatCoordinator
from app.routers.chat.schemas import ChatRequest


class _PriorityPipeline:
    def __init__(self, redis: _PriorityRedis) -> None:
        self.redis = redis
        self.now = 0.0

    def zremrangebyscore(self, key: str, _minimum: str, maximum: float):
        assert key.endswith(":leases")
        self.now = float(maximum)
        return self

    def zcard(self, key: str):
        assert key.endswith(":leases")
        return self

    async def execute(self) -> list[int]:
        expired = [token for token, score in self.redis.leases.items() if score <= self.now]
        for token in expired:
            self.redis.leases.pop(token, None)
        return [len(expired), len(self.redis.leases)]


class _PriorityRedis:
    """One event-loop-local Redis double shared by all three production paths."""

    def __init__(self, timeline: list[str]) -> None:
        self.leases: dict[str, float] = {}
        self.timeline = timeline

    async def zadd(self, key: str, values: dict[str, float]) -> int:
        assert key.endswith(":leases")
        self.leases.update(values)
        self.timeline.append("lease:published")
        return 1

    async def eval(
        self,
        script: str,
        _key_count: int,
        key: str,
        token: str,
        *args: float,
    ) -> int:
        assert key.endswith(":leases")
        if "zrem" in script:
            self.timeline.append("lease:released")
            return int(self.leases.pop(token, None) is not None)
        if token not in self.leases:
            return 0
        self.leases[token] = float(args[0])
        return 1

    def pipeline(self, *, transaction: bool) -> _PriorityPipeline:
        assert transaction is True
        return _PriorityPipeline(self)

    async def aclose(self) -> None:
        return None


class _Persistence:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def prepare(self, **_values):
        return [], None, None, ([], {}, False)

    async def add_message(self, **values):
        self.messages.append(values)
        return uuid4()


class _BlockingRuntime:
    def __init__(self, signer: CapabilitySigner, timeline: list[str]) -> None:
        self.signer = signer
        self.timeline = timeline
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request, capability):
        scope = self.signer.verify(capability, expected_run_id=request["run_id"])
        assert scope.user_id == 77
        self.timeline.append("chat:model:start")
        self.entered.set()
        await self.release.wait()
        self.timeline.append("chat:model:finish")
        yield {
            "type": "answer_delta",
            "delta": "priority-ok",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        }
        yield {
            "type": "final",
            "answer": "priority-ok",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        }


class _MemoryJobs:
    """Faithfully model claim/defer attempt accounting without a database."""

    def __init__(self, timeline: list[str]) -> None:
        now = datetime.now(UTC)
        self.template = GraphMemoryJob(
            id=uuid4(),
            user_id=77,
            tenant_uuid=uuid4(),
            generation=1,
            job_type="extract",
            source_chat_id=uuid4(),
            source_message_id=uuid4(),
            payload={},
            attempts=0,
            max_attempts=5,
            lease_owner="",
            lease_expires_at=now + timedelta(minutes=3),
        )
        self.timeline = timeline
        self.state = "queued"
        self.attempts = 0
        self.claims = 0
        self.deferrals = 0

    def claim_next(self, worker_id, _lease_seconds):
        if self.state != "queued":
            return None
        self.state = "running"
        self.attempts += 1
        self.claims += 1
        return replace(self.template, attempts=self.attempts, lease_owner=worker_id)

    def heartbeat(self, _job, _worker_id, _lease_seconds):
        return self.state == "running"

    def current_memory_model(self):
        return True, "memory-model"

    def load_user_message(self, job):
        return PersistedUserMessage(
            chat_id=job.source_chat_id,
            message_id=job.source_message_id,
            content="I prefer concise answers.",
        )

    def defer(self, _job, _worker_id, *, delay_seconds):
        assert self.state == "running"
        assert delay_seconds == 5
        self.attempts = max(0, self.attempts - 1)
        self.state = "queued"
        self.deferrals += 1
        self.timeline.append("memory:model:deferred")
        return True

    def complete(self, _job, _worker_id):
        assert self.state == "running"
        self.state = "complete"
        return True

    def cancel(self, *_args):  # pragma: no cover - a call fails the test
        raise AssertionError("memory job was unexpectedly cancelled")

    def requeue_or_fail(self, *_args, **_kwargs):  # pragma: no cover - a call fails the test
        raise AssertionError("memory job unexpectedly consumed a retry")

    def release_completed_lease(self, *_args):  # pragma: no cover - extract does not use it
        raise AssertionError("extract job used the projection completion path")


class _MemoryService:
    def expire_due(self, *, limit):
        assert limit == 500
        return 0

    def accept_candidate(self, *_args, **_kwargs):  # pragma: no cover - extractor returns none
        raise AssertionError("empty extraction unexpectedly produced a candidate")


class _MemoryExtractor:
    def __init__(self, timeline: list[str]) -> None:
        self.timeline = timeline
        self.calls: list[tuple[str, str]] = []

    async def extract(self, model: str, source: str):
        self.timeline.append("memory:model:called")
        self.calls.append((model, source))
        return ()


def _document() -> CanonicalDocument:
    return CanonicalDocument(
        source_name="invoice.pdf",
        source_sha256="a" * 64,
        media_type="application/pdf",
        parser_fingerprint="test:parser",
        elements=(),
    )


def _chunks() -> tuple[CanonicalChunk, ...]:
    text = "Tax invoice 1042 records consulting services billed to Example Company."
    return (
        CanonicalChunk(
            chunk_id="chk_" + "1" * 64,
            ordinal=0,
            text=text,
            embedding_text=text,
            element_ids=("el_" + "1" * 64,),
            provenance=(Provenance(page_number=1),),
            token_count=len(text.split()),
        ),
    )


@pytest.mark.asyncio
async def test_real_chat_lease_defers_both_background_roles_then_they_resume(monkeypatch) -> None:
    timeline: list[str] = []
    redis = _PriorityRedis(timeline)

    async def active_lease_count() -> int:
        return len(redis.leases)

    await run_concurrent_priority_drill(
        monkeypatch,
        redis_factory=lambda: redis,
        active_lease_count=active_lease_count,
        timeline=timeline,
    )


async def run_concurrent_priority_drill(
    monkeypatch,
    *,
    redis_factory,
    active_lease_count,
    timeline: list[str] | None = None,
) -> list[str]:
    """Exercise all production priority callers against one Redis implementation."""

    timeline = timeline if timeline is not None else []
    monkeypatch.setattr(inference_priority, "_redis_client", redis_factory)

    signer = CapabilitySigner("p" * 32)
    runtime = _BlockingRuntime(signer, timeline)
    persistence = _Persistence()
    coordinator = ChatCoordinator(
        signer=signer,
        runtime=runtime,  # type: ignore[arg-type]
        persistence=persistence,  # type: ignore[arg-type]
        evidence_store=RunEvidenceStore(),
        foreground_lease=inference_priority.foreground_inference_lease,
    )

    async def collect_chat():
        return [
            event
            async for event in coordinator.stream(
                ChatRequest(message="Keep interactive chat responsive"),
                user_id=77,
            )
        ]

    chat_task = asyncio.create_task(collect_chat())
    await asyncio.wait_for(runtime.entered.wait(), timeout=1)
    assert await active_lease_count() == 1
    assert "chat:model:start" in timeline

    intelligence_gate_checked = asyncio.Event()

    async def intelligence_gate() -> bool:
        allowed = await inference_priority.background_inference_allowed()
        if not allowed:
            intelligence_gate_checked.set()
            timeline.append("intelligence:model:deferred")
        return allowed

    intelligence_calls: list[str] = []

    def intelligence_post(_self, _document, _chunks):
        timeline.append("intelligence:model:called")
        intelligence_calls.append("intelligence-model")
        return {
            "doc_type": "invoice",
            "summary": "Tax invoice 1042 bills Example Company for consulting services.",
            "tags": ["consulting services"],
        }

    monkeypatch.setattr(DocumentIntelligenceService, "_post", intelligence_post)
    intelligence = ConfiguredDocumentIntelligenceService(
        lambda: (True, "intelligence-model"),
        IntelligenceSettings(model="unused-environment-model"),
        priority_gate=intelligence_gate,
        priority_poll_seconds=0.005,
    )
    intelligence_task = asyncio.create_task(intelligence.analyze(_document(), _chunks()))
    await asyncio.wait_for(intelligence_gate_checked.wait(), timeout=1)

    memory_jobs = _MemoryJobs(timeline)
    memory_extractor = _MemoryExtractor(timeline)
    memory_worker = GraphMemoryWorker(
        _MemoryService(),  # type: ignore[arg-type]
        jobs=memory_jobs,  # type: ignore[arg-type]
        extractor=memory_extractor,  # type: ignore[arg-type]
        priority_gate=inference_priority.background_inference_allowed,
        worker_id="priority-test",
        maintenance_interval_seconds=3600,
    )
    assert await memory_worker.run_once() is True

    assert not intelligence_task.done()
    assert intelligence_calls == []
    assert memory_extractor.calls == []
    assert memory_jobs.deferrals == 1
    assert memory_jobs.attempts == 0
    assert await active_lease_count() == 1

    runtime.release.set()
    chat_events = await asyncio.wait_for(chat_task, timeout=1)
    intelligence_result = await asyncio.wait_for(intelligence_task, timeout=1)
    assert await active_lease_count() == 0
    assert [message["role"] for message in persistence.messages] == ["user", "assistant"]
    assert "priority-ok" == "".join(
        str(event["content"]) for event in chat_events if event["type"] == "token"
    )

    assert intelligence_result.status == "model"
    assert intelligence_calls == ["intelligence-model"]
    assert await memory_worker.run_once() is True
    assert memory_extractor.calls == [("memory-model", "I prefer concise answers.")]
    assert memory_jobs.claims == 2
    assert memory_jobs.attempts == 1
    assert memory_jobs.state == "complete"
    return timeline


@pytest.mark.asyncio
async def test_redis_client_construction_failure_keeps_chat_open_and_background_closed(
    monkeypatch,
) -> None:
    def unavailable_redis():
        raise ConnectionError("isolated Redis outage")

    monkeypatch.setattr(inference_priority, "_redis_client", unavailable_redis)
    signer = CapabilitySigner("r" * 32)

    class ImmediateRuntime:
        async def stream(self, request, capability):
            scope = signer.verify(capability, expected_run_id=request["run_id"])
            assert scope.user_id == 88
            yield {
                "type": "answer_delta",
                "delta": "still-online",
                "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
            }
            yield {
                "type": "final",
                "answer": "still-online",
                "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
            }

    coordinator = ChatCoordinator(
        signer=signer,
        runtime=ImmediateRuntime(),  # type: ignore[arg-type]
        persistence=_Persistence(),  # type: ignore[arg-type]
        evidence_store=RunEvidenceStore(),
        foreground_lease=inference_priority.foreground_inference_lease,
    )

    async def collect_chat():
        return [
            event
            async for event in coordinator.stream(
                ChatRequest(message="Does chat survive Redis?"),
                user_id=88,
            )
        ]

    started = time.monotonic()
    events = await asyncio.wait_for(collect_chat(), timeout=1)
    assert time.monotonic() - started < 0.5
    assert "still-online" == "".join(
        str(event["content"]) for event in events if event["type"] == "token"
    )
    assert await inference_priority.background_inference_allowed() is False
