import asyncio
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.graph_memory.inference_priority import BackgroundInferenceDeferred
from app.ingestion.errors import ProcessExecutionError
from app.ingestion.models import (
    CanonicalChunk,
    CanonicalDocument,
    CanonicalElement,
    ElementType,
    IngestionJob,
    JobState,
    Provenance,
    PublicationResult,
)
from app.ingestion.parsers.base import ParseRequest
from app.ingestion.worker import IngestionWorker, _load_factory


class FakeRepository:
    def __init__(self, job):
        self.job = job
        self.claimed = False
        self.transitions = []
        self.finishes = []
        self.requeues = []
        self.deferrals = []

    def claim_next(self, worker_id, lease_seconds):
        if self.claimed:
            return None
        self.claimed = True
        return self.job

    def transition(self, job, worker_id, current, target, **kwargs):
        self.transitions.append((current, target, kwargs))
        return True

    def heartbeat(self, job, worker_id, lease_seconds):
        return True

    def should_cancel(self, job, worker_id):
        return False

    def finish(self, job, worker_id, current, target, **kwargs):
        self.finishes.append((current, target, kwargs))
        return True

    def requeue(self, job, worker_id, current, **kwargs):
        self.requeues.append((current, kwargs))
        return True

    def defer(self, job, worker_id, current, **kwargs):
        self.deferrals.append((current, kwargs))
        return True


class FakeSource:
    def __init__(self, request):
        self.request = request
        self.cancel_event = None

    @asynccontextmanager
    async def open(self, job, *, cancel_event=None):
        self.cancel_event = cancel_event
        yield self.request


class FakeParsers:
    def __init__(self, document):
        self.document = document
        self.calls = 0

    async def parse(self, route, request, *, cancel_event=None):
        self.calls += 1
        return self.document


class FakeChunker:
    def __init__(self, chunk):
        self.chunk_value = chunk

    def chunk(self, document):
        return (self.chunk_value,)


class FakeSink:
    def __init__(self, *, atomic=False):
        self.atomic = atomic
        self.embedded = False
        self.published = False
        self.enriched = False

    async def embed(self, job, document, chunks, *, cancel_event):
        self.embedded = True
        return "prepared"

    async def publish(self, job, document, chunks, prepared, *, cancel_event):
        self.published = True
        return PublicationResult(job_completed=True) if self.atomic else None

    async def enrich(self, job, document, chunks, *, cancel_event):
        self.enriched = True


def fixture(source_name="report.pdf", media_type="application/pdf"):
    provenance = (Provenance(page_number=1),)
    element = CanonicalElement(
        element_id="el_" + "e" * 64,
        element_type=ElementType.PARAGRAPH,
        text="content",
        provenance=provenance,
    )
    document = CanonicalDocument(
        source_name=source_name,
        source_sha256="a" * 64,
        media_type=media_type,
        parser_fingerprint="parser:test",
        elements=(element,),
    )
    chunk = CanonicalChunk(
        chunk_id="chk_" + "c" * 64,
        ordinal=0,
        text="content",
        embedding_text="content",
        element_ids=(element.element_id,),
        provenance=provenance,
        token_count=1,
    )
    job = IngestionJob(
        job_id="00000000-0000-0000-0000-000000000001",
        file_id=1,
        user_id=2,
        revision=1,
        state=JobState.DECRYPTING,
        source_name=source_name,
        media_type=media_type,
        source_sha256="a" * 64,
        attempts=1,
        lease_owner="worker-test",
    )
    request = ParseRequest(
        path=Path("/unused/source"),
        source_name=source_name,
        media_type=media_type,
        source_sha256="a" * 64,
    )
    return job, request, document, chunk


class IngestionWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def _worker(self, *, atomic=False, source_name="report.pdf", media_type="application/pdf"):
        job, request, document, chunk = fixture(source_name, media_type)
        repository = FakeRepository(job)
        source = FakeSource(request)
        parsers = FakeParsers(document)
        sink = FakeSink(atomic=atomic)
        worker = IngestionWorker(
            repository=repository,
            source_provider=source,
            sink=sink,
            parser_registry=parsers,
            chunker=FakeChunker(chunk),
            worker_id="worker-test",
            heartbeat_seconds=100,
            cancellation_poll_seconds=100,
        )
        await worker.run_once()
        return repository, source, parsers, sink

    async def test_success_moves_through_all_durable_stages(self):
        repository, source, parsers, sink = await self._worker()
        targets = [target for _current, target, _kwargs in repository.transitions]
        self.assertEqual(
            targets,
            [
                JobState.PARSING,
                JobState.CHUNKING,
                JobState.EMBEDDING,
                JobState.PUBLISHING,
            ],
        )
        self.assertEqual(repository.finishes[0][1], JobState.READY)
        self.assertIsNotNone(source.cancel_event)
        self.assertEqual(parsers.calls, 1)
        self.assertTrue(sink.embedded and sink.published)

    async def test_atomic_sink_does_not_double_finish_job(self):
        repository, _source, _parsers, sink = await self._worker(atomic=True)
        self.assertTrue(sink.published)
        self.assertEqual(repository.finishes, [])

    async def test_background_inference_deferral_is_attempt_neutral(self):
        class DeferredSink(FakeSink):
            async def publish(self, job, document, chunks, prepared, *, cancel_event):
                self.published = True
                raise BackgroundInferenceDeferred()

        job, request, document, chunk = fixture()
        repository = FakeRepository(job)
        sink = DeferredSink()
        worker = IngestionWorker(
            repository=repository,
            source_provider=FakeSource(request),
            sink=sink,
            parser_registry=FakeParsers(document),
            chunker=FakeChunker(chunk),
            worker_id="worker-test",
            heartbeat_seconds=100,
            cancellation_poll_seconds=100,
            retry_base_seconds=5,
        )

        await worker.run_once()

        self.assertTrue(sink.embedded and sink.published)
        self.assertEqual(repository.finishes, [])
        self.assertEqual(repository.requeues, [])
        self.assertEqual(
            repository.deferrals,
            [(JobState.PUBLISHING, {"delay_seconds": 5})],
        )

    async def test_unsupported_media_is_terminal_without_parser_call(self):
        repository, _source, parsers, sink = await self._worker(
            source_name="clip.mp4", media_type="video/mp4"
        )
        self.assertEqual(parsers.calls, 0)
        self.assertFalse(sink.embedded)
        self.assertEqual(repository.finishes[0][1], JobState.UNSUPPORTED)

    async def test_parser_failures_requeue_with_safe_bounded_diagnostics(self):
        class FailingParsers:
            def __init__(self, error):
                self.error = error

            async def parse(self, _route, _request, *, cancel_event=None):
                raise self.error

        unsafe_detail = (
            "x" * 2_100 + "\n\x1b[31mprivate-report.pdf /unused/source token=supersecret "
            "backend worker exited unexpectedly\x1b[0m"
        )
        cases = (
            (
                ProcessExecutionError(
                    "OpenDataLoader exited with status 1",
                    detail=unsafe_detail,
                ),
                "process_failed",
                "Retrying ingestion job",
            ),
            (RuntimeError(unsafe_detail), "internal_error", "Unhandled ingestion failure"),
        )
        for error, expected_code, expected_log in cases:
            with self.subTest(error=type(error).__name__):
                job, request, _document, chunk = fixture(source_name="private-report.pdf")
                repository = FakeRepository(job)
                worker = IngestionWorker(
                    repository=repository,
                    source_provider=FakeSource(request),
                    sink=FakeSink(),
                    parser_registry=FailingParsers(error),
                    chunker=FakeChunker(chunk),
                    worker_id="worker-test",
                    heartbeat_seconds=100,
                    cancellation_poll_seconds=100,
                )

                with self.assertLogs("app.ingestion.worker", level="WARNING") as logs:
                    await worker.run_once()

                self.assertEqual(repository.finishes, [])
                self.assertEqual(len(repository.requeues), 1)
                current, persisted = repository.requeues[0]
                self.assertEqual(current, JobState.PARSING)
                self.assertEqual(persisted["error_code"], expected_code)
                detail = persisted["error_detail"]
                self.assertLessEqual(len(detail), 2_000)
                self.assertIn("backend worker exited unexpectedly", detail)
                self.assertIn("token=<redacted>", detail)
                self.assertNotIn("supersecret", detail)
                self.assertNotIn("private-report.pdf", detail)
                self.assertNotIn("/unused/source", detail)
                self.assertNotIn("\x1b", detail)
                log_output = "\n".join(logs.output)
                self.assertIn(expected_log, log_output)
                self.assertNotIn("supersecret", log_output)

    async def test_health_remains_live_until_a_graceful_drain_finishes(self):
        class BlockingWorker(IngestionWorker):
            def __init__(self, health_reporter):
                super().__init__(
                    repository=object(),
                    source_provider=object(),
                    sink=object(),
                    health_reporter=health_reporter,
                )
                self.job_started = asyncio.Event()
                self.release_job = asyncio.Event()

            async def run_once(self):
                self.job_started.set()
                await self.release_job.wait()
                return True

        class HealthReporter:
            def __init__(self):
                self.started = asyncio.Event()
                self.stop_event = None

            async def run(self, stop_event):
                self.stop_event = stop_event
                self.started.set()
                await stop_event.wait()

        reporter = HealthReporter()
        worker = BlockingWorker(reporter)
        requested_stop = asyncio.Event()
        task = asyncio.create_task(worker.run_forever(requested_stop))
        await worker.job_started.wait()
        await reporter.started.wait()

        requested_stop.set()
        await asyncio.sleep(0)
        self.assertIsNot(reporter.stop_event, requested_stop)
        self.assertFalse(reporter.stop_event.is_set())

        worker.release_job.set()
        await task
        self.assertTrue(reporter.stop_event.is_set())

    async def test_run_forever_retries_a_transient_iteration_failure(self):
        requested_stop = asyncio.Event()

        class FlakyWorker(IngestionWorker):
            def __init__(self):
                super().__init__(
                    repository=object(),
                    source_provider=object(),
                    sink=object(),
                )
                self.calls = 0

            async def run_once(self):
                self.calls += 1
                if self.calls == 1:
                    raise ConnectionError("temporary PostgreSQL outage")
                requested_stop.set()
                return False

        worker = FlakyWorker()
        await asyncio.wait_for(
            worker.run_forever(requested_stop, idle_seconds=0.001),
            timeout=2,
        )
        self.assertEqual(worker.calls, 2)


class WorkerFactoryTests(unittest.TestCase):
    def test_factory_accepts_the_narrow_runtime_interface(self):
        class StructuralWorker:
            async def run_forever(self, stop_event, *, idle_seconds=1.0):
                return None

        runtime = StructuralWorker()
        module = SimpleNamespace(create_worker=lambda: runtime)
        with patch("app.ingestion.worker.importlib.import_module", return_value=module):
            self.assertIs(_load_factory("factory.module:create_worker"), runtime)

    def test_factory_rejects_objects_without_run_forever(self):
        module = SimpleNamespace(create_worker=lambda: object())
        with patch("app.ingestion.worker.importlib.import_module", return_value=module):
            with self.assertRaisesRegex(TypeError, "worker runtime"):
                _load_factory("factory.module:create_worker")


if __name__ == "__main__":
    unittest.main()
