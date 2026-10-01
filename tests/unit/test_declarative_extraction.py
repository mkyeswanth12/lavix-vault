"""Declarative-only extraction: interrogative spans never reach the model."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.graph_memory.models import MemoryCandidate
from app.graph_memory.worker import (
    GraphMemoryJob,
    GraphMemoryWorker,
    PersistedUserMessage,
    attribute_span,
    split_declarative_spans,
)

WORK_MESSAGE = (
    "I normally work from 10:30 AM to 6 PM on weekdays. "
    "How could I organize my evenings around work and personal projects?"
)
SPICY_MESSAGE = (
    "I usually prefer spicy food and I make butter chicken at home. "
    "What spices would work well with it?"
)
BHEEM_MESSAGE = (
    "I really love dogs. I had a Golden Retriever named Bheem. "
    "What are some good ways to preserve memories of a dog that was important to me?"
)


def test_work_message_keeps_schedule_drops_question() -> None:
    assert split_declarative_spans(WORK_MESSAGE) == [
        "I normally work from 10:30 AM to 6 PM on weekdays."
    ]


def test_spicy_message_keeps_statements_drops_question() -> None:
    assert split_declarative_spans(SPICY_MESSAGE) == [
        "I usually prefer spicy food and I make butter chicken at home."
    ]


def test_bheem_message_keeps_two_statements() -> None:
    assert split_declarative_spans(BHEEM_MESSAGE) == [
        "I really love dogs.",
        "I had a Golden Retriever named Bheem.",
    ]


def test_question_only_message_yields_no_spans() -> None:
    assert split_declarative_spans("What spices would work well with it?") == []
    assert split_declarative_spans("How could I organize my evenings?") == []
    assert split_declarative_spans("   ") == []


def test_auxiliary_inversion_dropped() -> None:
    assert split_declarative_spans("Should I learnRust? Do you agree? I code daily.") == [
        "I code daily."
    ]


def test_mid_sentence_question_words_kept() -> None:
    spans = split_declarative_spans("My dog, who is a retriever, died in 2025.")
    assert spans == ["My dog, who is a retriever, died in 2025."]


def test_attribution_picks_supporting_span() -> None:
    spans = split_declarative_spans(WORK_MESSAGE)
    question_derived = MemoryCandidate(
        kind="relationship",
        subject="I",
        predicate="work and personal projects",
        object_value="organize evenings around work and personal projects",
        confidence=0.8,
        source_excerpt="placeholder",
        source_role="user",
        explicit_user_assertion=True,
    )
    excerpt = attribute_span(question_derived, spans, "\n".join(spans))
    assert excerpt == spans[0]
    assert "?" not in excerpt
    assert "How could I" not in excerpt


def _extract_job(content: str) -> tuple[GraphMemoryJob, dict[str, object]]:
    job = GraphMemoryJob(
        id=uuid4(),
        user_id=41,
        tenant_uuid=uuid4(),
        generation=0,
        job_type="extract",
        source_chat_id=uuid4(),
        source_message_id=uuid4(),
        payload={},
        attempts=1,
        max_attempts=3,
        lease_owner="",
        lease_expires_at=None,  # type: ignore[arg-type]
    )
    state: dict[str, object] = {"content": content}
    return job, state


class _Jobs:
    def __init__(self, job: GraphMemoryJob, content: str) -> None:
        self._job = job
        self._content = content

    def current_memory_model(self):  # type: ignore[no-untyped-def]
        return True, "memory-model"

    def load_user_message(self, job):  # type: ignore[no-untyped-def]
        assert job.id == self._job.id
        return PersistedUserMessage(
            chat_id=self._job.source_chat_id,
            message_id=self._job.source_message_id,
            content=self._content,
        )


class _Extractor:
    def __init__(self, values: tuple) -> None:
        self.values = values
        self.calls: list[tuple[str, str]] = []

    async def extract(self, model: str, source: str):  # type: ignore[no-untyped-def]
        self.calls.append((model, source))
        return self.values


class _Service:
    def __init__(self) -> None:
        self.accepted: list[MemoryCandidate] = []

    def record_question_topics(self, user_id: int, text: str):  # type: ignore[no-untyped-def]
        return ()

    def accept_candidate(self, user_id: int, **values):  # type: ignore[no-untyped-def]
        self.accepted.append(values["candidate"])


def _worker(content: str, extractor: _Extractor, service: _Service, job: GraphMemoryJob) -> GraphMemoryWorker:
    async def allowed() -> bool:
        return True

    return GraphMemoryWorker(
        service,  # type: ignore[arg-type]
        jobs=_Jobs(job, content),  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        priority_gate=allowed,
        worker_id="test",
    )


def _candidate(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "kind": "fact",
        "subject": "I",
        "predicate": "work",
        "object_value": "10:30 AM to 6 PM on weekdays",
        "confidence": 0.9,
        "source_excerpt": "model excerpt",
        "source_role": "user",
        "explicit_user_assertion": True,
    }
    values.update(overrides)
    return MemoryCandidate(**values)


@pytest.mark.asyncio
async def test_extract_feeds_declaratives_and_attributes_excerpt() -> None:
    job, _ = _extract_job(WORK_MESSAGE)
    extractor = _Extractor((_candidate(),))
    service = _Service()
    worker = _worker(WORK_MESSAGE, extractor, service, job)

    await worker._extract(job, asyncio.Event())

    assert len(extractor.calls) == 1
    fed = extractor.calls[0][1]
    assert "How could I" not in fed
    assert "?" not in fed
    assert "I normally work from 10:30 AM to 6 PM on weekdays." in fed
    assert len(service.accepted) == 1
    assert service.accepted[0].source_excerpt == "I normally work from 10:30 AM to 6 PM on weekdays."


@pytest.mark.asyncio
async def test_question_only_message_completes_without_model_call() -> None:
    job, _ = _extract_job("What spices would work well with it?")
    extractor = _Extractor((_candidate(),))
    service = _Service()
    worker = _worker("What spices would work well with it?", extractor, service, job)

    await worker._extract(job, asyncio.Event())

    assert extractor.calls == []
    assert service.accepted == []
