from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from app.database import DatabaseUnavailableError
from app.graph_memory.models import MemoryCandidate
from app.graph_memory.repository import GraphMemoryConflictError, GraphMemoryValidationError
from app.graph_memory.worker import (
    GraphMemoryJob,
    GraphMemoryWorker,
    GraphProjectionReconciler,
    OllamaMemoryExtractor,
    PersistedUserMessage,
    ReconciliationFence,
)


def graph_job(*, job_type: str = "extract", payload=None, generation: int = 3) -> GraphMemoryJob:
    now = datetime.now(UTC)
    return GraphMemoryJob(
        id=uuid4(),
        user_id=41,
        tenant_uuid=uuid4(),
        generation=generation,
        job_type=job_type,
        source_chat_id=uuid4(),
        source_message_id=uuid4(),
        payload=payload or {},
        attempts=1,
        max_attempts=5,
        lease_owner="worker-test",
        lease_expires_at=now + timedelta(minutes=3),
    )


def candidate(**overrides) -> MemoryCandidate:
    values = {
        "kind": "preference",
        "subject": "I",
        "predicate": "prefer",
        "object_value": "concise answers",
        "confidence": 0.96,
        "source_excerpt": "I prefer concise answers.",
        "source_role": "user",
        "explicit_user_assertion": True,
        "contains_sensitive_data": False,
    }
    values.update(overrides)
    return MemoryCandidate(**values)


@pytest.mark.asyncio
async def test_ollama_extractor_accepts_only_strict_bounded_json_from_user_message() -> None:
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps({"candidates": [candidate().model_dump(mode="json")]})}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    extractor = OllamaMemoryExtractor("http://ollama:11434", client=client)
    result = await extractor.extract("memory-model", "I prefer concise answers.")

    assert result == (candidate(),)
    assert captured["model"] == "memory-model"
    assert captured["stream"] is False
    assert captured["format"]["additionalProperties"] is False
    assert captured["options"] == {
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 1200,
        "num_gpu": 0,
    }
    assert [message["role"] for message in captured["messages"]] == ["system", "user"]
    assert "I prefer concise answers." in captured["messages"][1]["content"]
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_extractor_rejects_extra_model_fields() -> None:
    raw = candidate().model_dump(mode="json")
    raw["user_id"] = 999

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": json.dumps({"candidates": [raw]})}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    extractor = OllamaMemoryExtractor("http://ollama:11434", client=client)
    with pytest.raises(ValueError, match="strict validation"):
        await extractor.extract("memory-model", "I prefer concise answers.")
    await client.aclose()


class FakeJobs:
    def __init__(self, job: GraphMemoryJob, *, model_enabled: bool = True) -> None:
        self.job = job
        self.model_enabled = model_enabled
        self.claimed = False
        self.completed = []
        self.deferred = []
        self.cancelled = []
        self.requeued = []

    def claim_next(self, worker_id, lease_seconds):
        if self.claimed:
            return None
        self.claimed = True
        return self.job

    def heartbeat(self, job, worker_id, lease_seconds):
        return True

    def current_memory_model(self):
        return self.model_enabled, "memory-model" if self.model_enabled else None

    def load_user_message(self, job):
        return PersistedUserMessage(
            chat_id=job.source_chat_id,
            message_id=job.source_message_id,
            content="I prefer concise answers.",
        )

    def complete(self, job, worker_id):
        self.completed.append(job.id)
        return True

    def defer(self, job, worker_id, *, delay_seconds):
        self.deferred.append((job.id, delay_seconds))
        return True

    def cancel(self, job, worker_id, code):
        self.cancelled.append((job.id, code))
        return True

    def requeue_or_fail(self, job, worker_id, **values):
        self.requeued.append((job.id, values))
        return True

    def release_completed_lease(self, job, worker_id):
        return True


class FakeService:
    def __init__(self) -> None:
        self.accepted = []
        self.expiry_runs = 0

    def expire_due(self, *, limit):
        self.expiry_runs += 1
        return 0

    def accept_candidate(self, user_id, **values):
        if values["candidate"].object_value == "reject me":
            raise GraphMemoryValidationError("candidate_rejected")
        self.accepted.append((user_id, values))


class FakeExtractor:
    def __init__(self, values) -> None:
        self.values = values
        self.calls = []

    async def extract(self, model, source):
        self.calls.append((model, source))
        return self.values


@pytest.mark.asyncio
async def test_worker_resolves_role_per_job_and_validates_candidates_independently() -> None:
    job = graph_job()
    jobs = FakeJobs(job)
    service = FakeService()
    extractor = FakeExtractor((candidate(), candidate(object_value="reject me")))

    async def allowed():
        return True

    worker = GraphMemoryWorker(
        service,  # type: ignore[arg-type]
        jobs=jobs,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        priority_gate=allowed,
        worker_id="worker-test",
    )
    assert await worker.run_once() is True

    assert extractor.calls == [("memory-model", "I prefer concise answers.")]
    assert len(service.accepted) == 1
    assert service.accepted[0][1]["source_message_id"] == job.source_message_id
    assert service.accepted[0][1]["lease_job_id"] == job.id
    assert service.accepted[0][1]["lease_owner"] == "worker-test"
    assert jobs.completed == [job.id]
    assert jobs.requeued == []


@pytest.mark.asyncio
async def test_worker_defers_model_call_while_foreground_inference_is_active() -> None:
    job = graph_job()
    jobs = FakeJobs(job)
    extractor = FakeExtractor((candidate(),))

    async def blocked():
        return False

    worker = GraphMemoryWorker(
        FakeService(),  # type: ignore[arg-type]
        jobs=jobs,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        priority_gate=blocked,
        worker_id="worker-test",
    )
    assert await worker.run_once() is True

    assert extractor.calls == []
    assert jobs.deferred == [(job.id, 5)]
    assert jobs.completed == []


@pytest.mark.asyncio
async def test_worker_discards_extraction_output_after_heartbeat_loses_lease() -> None:
    job = graph_job()
    jobs = FakeJobs(job)
    service = FakeService()
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingExtractor(FakeExtractor):
        async def extract(self, model, source):
            self.calls.append((model, source))
            entered.set()
            await release.wait()
            return (candidate(),)

    async def allowed():
        return True

    worker = GraphMemoryWorker(
        service,  # type: ignore[arg-type]
        jobs=jobs,  # type: ignore[arg-type]
        extractor=BlockingExtractor(()),  # type: ignore[arg-type]
        priority_gate=allowed,
        worker_id="worker-test",
    )

    async def lose_lease(_job, _stop, lease_lost):
        await entered.wait()
        lease_lost.set()
        release.set()

    worker._heartbeat = lose_lease  # type: ignore[method-assign]
    assert await worker.run_once() is True

    assert service.accepted == []
    assert jobs.completed == []
    assert jobs.deferred == []
    assert jobs.cancelled == []
    assert jobs.requeued == []


@pytest.mark.asyncio
async def test_run_forever_recovers_from_a_transient_database_startup_failure() -> None:
    requested_stop = asyncio.Event()

    class RecoveringWorker(GraphMemoryWorker):
        def __init__(self) -> None:
            super().__init__(FakeService(), jobs=object())  # type: ignore[arg-type]
            self.calls = 0

        async def run_once(self) -> bool:
            self.calls += 1
            if self.calls == 1:
                raise DatabaseUnavailableError("database unavailable")
            requested_stop.set()
            return False

    worker = RecoveringWorker()
    await asyncio.wait_for(
        worker.run_forever(requested_stop, idle_seconds=0.001),
        timeout=1,
    )

    assert worker.calls == 2


@pytest.mark.asyncio
async def test_run_forever_stop_interrupts_dependency_retry_backoff() -> None:
    requested_stop = asyncio.Event()
    failed_once = asyncio.Event()

    class OfflineWorker(GraphMemoryWorker):
        def __init__(self) -> None:
            super().__init__(FakeService(), jobs=object())  # type: ignore[arg-type]
            self.calls = 0

        async def run_once(self) -> bool:
            self.calls += 1
            failed_once.set()
            raise DatabaseUnavailableError("database unavailable")

    worker = OfflineWorker()
    task = asyncio.create_task(worker.run_forever(requested_stop, idle_seconds=30))
    await failed_once.wait()
    requested_stop.set()
    await asyncio.wait_for(task, timeout=0.5)

    assert worker.calls == 1


@pytest.mark.asyncio
async def test_projection_rejects_a_job_from_old_generation_before_projection() -> None:
    job = graph_job(
        job_type="project",
        payload={"operation": "delete", "memory_id": str(uuid4())},
        generation=2,
    )
    current = SimpleNamespace(
        user_id=job.user_id,
        tenant_uuid=job.tenant_uuid,
        generation=3,
    )
    service = SimpleNamespace(repository=SimpleNamespace(get_tenant=lambda _user_id: current))
    worker = GraphMemoryWorker(service, jobs=FakeJobs(job), worker_id="worker-test")

    with pytest.raises(GraphMemoryConflictError, match="generation"):
        await worker._project(job)


class ReconcileControl:
    def __init__(self, fence, *, fail_complete=False) -> None:
        self.fence = fence
        self.fail_complete = fail_complete
        self.completed = []
        self.failed = []

    def tenant_user_ids(self, user_id):
        return (self.fence.user_id,)

    def begin(self, user_id):
        assert user_id == self.fence.user_id
        return self.fence

    def complete(self, fence):
        if self.fail_complete:
            raise GraphMemoryConflictError("memory_changed_during_reconciliation")
        self.completed.append(fence)

    def fail(self, fence):
        self.failed.append(fence)


class ReconcileCanonicalRepository:
    def __init__(self, items) -> None:
        self.items = tuple(items)

    def aggregate(self, user_id):
        return {"active": len(self.items)}

    def active_items_for_profile(self, user_id, *, limit):
        return self.items[:limit]


class ReconcileProjection:
    def __init__(self, *, fail_upsert=False) -> None:
        self.fail_upsert = fail_upsert
        self.deleted = []
        self.upserted = []

    def delete_tenant(self, tenant_uuid):
        self.deleted.append(tenant_uuid)

    def upsert_item(self, tenant_uuid, item):
        if self.fail_upsert:
            raise RuntimeError("projection unavailable")
        self.upserted.append((tenant_uuid, item.id))


def test_reconcile_rebuilds_projection_from_pg_authority_behind_a_fence() -> None:
    fence = ReconciliationFence(
        user_id=41,
        tenant_uuid=uuid4(),
        generation=3,
        graph_revision=9,
        updated_at=datetime.now(UTC),
    )
    items = [SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())]
    projection = ReconcileProjection()
    service = SimpleNamespace(
        repository=ReconcileCanonicalRepository(items),
        projection=projection,
    )
    control = ReconcileControl(fence)

    result = GraphProjectionReconciler(
        service,  # type: ignore[arg-type]
        control=control,  # type: ignore[arg-type]
    ).reconcile()

    assert result == {"tenants": 1, "projected_items": 2, "failures": []}
    assert projection.deleted == [fence.tenant_uuid]
    assert projection.upserted == [(fence.tenant_uuid, item.id) for item in items]
    assert control.completed == [fence]
    assert control.failed == []


@pytest.mark.parametrize("failure", ["projection", "revision_fence"])
def test_reconcile_fails_closed_and_marks_tenant_unavailable(failure) -> None:
    fence = ReconciliationFence(
        user_id=41,
        tenant_uuid=uuid4(),
        generation=3,
        graph_revision=9,
        updated_at=datetime.now(UTC),
    )
    service = SimpleNamespace(
        repository=ReconcileCanonicalRepository([SimpleNamespace(id=uuid4())]),
        projection=ReconcileProjection(fail_upsert=failure == "projection"),
    )
    control = ReconcileControl(fence, fail_complete=failure == "revision_fence")

    result = GraphProjectionReconciler(
        service,  # type: ignore[arg-type]
        control=control,  # type: ignore[arg-type]
    ).reconcile(user_id=41)

    assert result == {"tenants": 1, "projected_items": 0, "failures": [41]}
    assert control.completed == []
    assert control.failed == [fence]

@pytest.mark.asyncio
async def test_worker_never_rewrites_candidate_subjects() -> None:
    job = graph_job()
    jobs = FakeJobs(job)
    service = FakeService()
    extractor = FakeExtractor(
        (
            candidate(
                subject="user",
                predicate="loves",
                object_value="sushi",
                source_excerpt="I love sushi.",
            ),
        )
    )

    async def allowed():
        return True

    worker = GraphMemoryWorker(
        service,  # type: ignore[arg-type]
        jobs=jobs,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        priority_gate=allowed,
        worker_id="worker-test",
    )
    assert await worker.run_once() is True
    subjects = [values["candidate"].subject for _, values in service.accepted]
    assert subjects == ["user"]
