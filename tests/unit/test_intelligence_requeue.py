"""Regression tests for BUG-007: deferred enrichment must not strand rows.

Bounded inline retry (3 attempts, fixed delay) then an explicit terminal
state for the operator backfill. No infinite loops, no silent dead-ends.
"""
import asyncio
from types import SimpleNamespace

import pytest

import app.ingestion.worker as worker_mod
from app.ingestion.worker import IngestionWorker


class ScriptedSink:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def enrich(self, job, document, chunks, *, cancel_event):
        self.calls += 1
        return self.outcomes.pop(0)


class ExplodingSink:
    def __init__(self):
        self.calls = 0

    async def enrich(self, job, document, chunks, *, cancel_event):
        self.calls += 1
        raise RuntimeError("model down")


def make_worker(sink):
    return IngestionWorker(
        repository=object(),
        source_provider=object(),
        sink=sink,
    )


@pytest.mark.asyncio
async def test_deferred_then_done_retries(monkeypatch):
    monkeypatch.setattr(worker_mod, "_ENRICH_RETRY_DELAY_SECONDS", 0.0)
    sink = ScriptedSink(["deferred", "done"])
    worker = make_worker(sink)
    stages = {}
    await worker._enrich_best_effort(SimpleNamespace(job_id="test-job"), object(), (), asyncio.Event(), stages)
    assert sink.calls == 2
    assert "enrich_s" in stages


@pytest.mark.asyncio
async def test_deferred_exhausts_bounded_attempts(monkeypatch):
    monkeypatch.setattr(worker_mod, "_ENRICH_RETRY_DELAY_SECONDS", 0.0)
    sink = ScriptedSink(["deferred"] * 10)
    worker = make_worker(sink)
    await worker._enrich_best_effort(SimpleNamespace(job_id="test-job"), object(), (), asyncio.Event(), {})
    assert sink.calls == 3, "must stop after _ENRICH_MAX_ATTEMPTS, never loop"


@pytest.mark.asyncio
async def test_exception_counts_as_deferred_then_terminal(monkeypatch):
    monkeypatch.setattr(worker_mod, "_ENRICH_RETRY_DELAY_SECONDS", 0.0)
    sink = ExplodingSink()
    worker = make_worker(sink)
    await worker._enrich_best_effort(SimpleNamespace(job_id="test-job"), object(), (), asyncio.Event(), {})
    assert sink.calls == 3


@pytest.mark.asyncio
async def test_skipped_and_done_do_not_retry():
    for outcome in ("skipped", "done", None):
        sink = ScriptedSink([outcome])
        worker = make_worker(sink)
        await worker._enrich_best_effort(SimpleNamespace(job_id="test-job"), object(), (), asyncio.Event(), {})
        assert sink.calls == 1


def test_publisher_enrich_returns_terminal_outcome():
    import asyncio as _asyncio

    from app.ingestion.publisher import PgVectorIndexSink

    async def go(outcome_fn, intelligence):
        job = object()
        sink = PgVectorIndexSink(lambda: None, None, expected_dimension=3, intelligence=intelligence)
        return await sink.enrich(job, object(), (), cancel_event=_asyncio.Event())

    class OkIntel:
        async def analyze(self, *a, **k):
            from app.ingestion.intelligence import DocumentIntelligence

            return DocumentIntelligence("guide", "s", (), "model", "m", None)

    # intelligence=None -> skipped without touching anything
    sink = PgVectorIndexSink(lambda: None, None, expected_dimension=3, intelligence=None)
    assert _asyncio.run(sink.enrich(object(), object(), (), cancel_event=_asyncio.Event())) == "skipped"
