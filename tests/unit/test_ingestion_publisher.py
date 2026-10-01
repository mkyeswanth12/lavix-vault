import asyncio
import unittest

import httpx

from app.graph_memory.inference_priority import BackgroundInferenceDeferred
from app.ingestion.embedding import (
    EmbeddingSettings,
    OllamaEmbeddingClient,
    PreparedEmbeddings,
)
from app.ingestion.errors import EmbeddingDimensionError, IngestionCancelled
from app.ingestion.models import (
    CanonicalChunk,
    CanonicalDocument,
    IngestionJob,
    JobState,
    Provenance,
)
from app.ingestion.publisher import PgVectorIndexSink


class RecordingCursor:
    def __init__(self, *, fence=True):
        self.fence = fence
        self.rowcount = 0
        self.calls = []
        self.many = []
        self._fetch = None

    def execute(self, query, params=None):
        normalized = " ".join(query.split())
        self.calls.append((normalized, params))
        if normalized.startswith(("SELECT f.id", "SELECT j.id")):
            self._fetch = {"id": "job"} if self.fence else None
            self.rowcount = 1 if self.fence else 0
        elif "SET status = 'superseded'" in normalized:
            self.rowcount = 0
        else:
            self.rowcount = 1

    def executemany(self, query, rows):
        self.many = list(rows)
        self.calls.append((" ".join(query.split()), self.many))
        self.rowcount = len(self.many)

    def fetchone(self):
        return self._fetch


class RecordingConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class ConnectionContext:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, *_args):
        return False


class FakeEmbeddings:
    def __init__(self, prepared):
        self.prepared = prepared
        self.texts = None

    async def embed(self, texts, *, cancel_event=None):
        self.texts = list(texts)
        return self.prepared


def fixtures():
    job = IngestionJob(
        job_id="00000000-0000-0000-0000-000000000001",
        file_id=5,
        user_id=9,
        revision=2,
        state=JobState.DECRYPTING,
        source_name="report.pdf",
        media_type="application/pdf",
        source_sha256="a" * 64,
        lease_owner="worker-a",
        metadata={"file_size_bytes": 123},
    )
    document = CanonicalDocument(
        source_name="report.pdf",
        source_sha256="a" * 64,
        media_type="application/pdf",
        parser_fingerprint="odl:test",
        elements=(),
        metadata={"parser": "opendataloader"},
    )
    chunk = CanonicalChunk(
        chunk_id="chk_" + "c" * 64,
        ordinal=0,
        text="Visible content",
        embedding_text="Context\nVisible content",
        element_ids=("el_" + "d" * 64,),
        provenance=(Provenance(page_number=1),),
        section_path=("Context",),
        token_count=2,
    )
    prepared = PreparedEmbeddings(
        vectors=((0.1, 0.2, 0.3),),
        model="test",
        dimension=3,
        fingerprint="emb_test",
    )
    return job, document, chunk, prepared


class PgVectorPublisherTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_and_publication_are_end_to_end_and_atomic(self):
        job, document, chunk, prepared = fixtures()
        cursor = RecordingCursor()
        connection = RecordingConnection(cursor)
        embeddings = FakeEmbeddings(prepared)
        sink = PgVectorIndexSink(
            lambda: ConnectionContext(connection),
            embeddings,
            chunker_fingerprint="chunker:test",
            expected_dimension=3,
        )
        event = asyncio.Event()
        actual = await sink.embed(job, document, (chunk,), cancel_event=event)
        receipt = await sink.publish(job, document, (chunk,), actual, cancel_event=event)
        self.assertEqual(embeddings.texts, [chunk.embedding_text])
        self.assertTrue(receipt.job_completed)
        self.assertEqual(connection.commits, 1)
        self.assertEqual(connection.rollbacks, 0)
        self.assertEqual(cursor.many[0][0], chunk.chunk_id)
        self.assertEqual(cursor.many[0][-1], "[0.10000000000000001,0.20000000000000001,0.29999999999999999]")
        sql = " ".join(query for query, _params in cursor.calls)
        self.assertIn("SET current_revision = %s", sql)
        self.assertIn("SET state = 'ready'", sql)

    async def test_failed_fence_rolls_back_everything(self):
        job, document, chunk, prepared = fixtures()
        cursor = RecordingCursor(fence=False)
        connection = RecordingConnection(cursor)
        sink = PgVectorIndexSink(
            lambda: ConnectionContext(connection),
            FakeEmbeddings(prepared),
            expected_dimension=3,
        )
        with self.assertRaises(IngestionCancelled):
            await sink.publish(
                job,
                document,
                (chunk,),
                prepared,
                cancel_event=asyncio.Event(),
            )
        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)
        self.assertEqual(cursor.many, [])

    async def test_dimension_mismatch_fails_before_database_work(self):
        job, document, chunk, prepared = fixtures()
        cursor = RecordingCursor()
        connection = RecordingConnection(cursor)
        sink = PgVectorIndexSink(
            lambda: ConnectionContext(connection),
            FakeEmbeddings(prepared),
            expected_dimension=1024,
        )
        with self.assertRaises(EmbeddingDimensionError):
            await sink.publish(
                job,
                document,
                (chunk,),
                prepared,
                cancel_event=asyncio.Event(),
            )
        self.assertEqual(cursor.calls, [])


class OllamaEmbeddingShapeTests(unittest.TestCase):
    def test_cpu_embedding_defaults_are_bounded_for_large_documents(self):
        settings = EmbeddingSettings()
        self.assertEqual(settings.batch_size, 16)
        self.assertEqual(settings.timeout_seconds, 300.0)

    def test_openai_response_is_sorted_and_dimension_checked(self):
        client = OllamaEmbeddingClient(EmbeddingSettings(dimension=2, batch_size=2, max_attempts=1))
        vectors = client._vectors(
            {
                "data": [
                    {"index": 1, "embedding": [3, 4]},
                    {"index": 0, "embedding": [1, 2]},
                ]
            },
            2,
        )
        self.assertEqual(vectors, [(1.0, 2.0), (3.0, 4.0)])
        with self.assertRaises(EmbeddingDimensionError):
            client._vectors({"embeddings": [[1.0]]}, 1)

    def test_openai_response_rejects_duplicate_indices(self):
        client = OllamaEmbeddingClient(EmbeddingSettings(dimension=1, batch_size=2, max_attempts=1))
        with self.assertRaisesRegex(Exception, "contiguous"):
            client._vectors(
                {
                    "data": [
                        {"index": 0, "embedding": [1]},
                        {"index": 0, "embedding": [2]},
                    ]
                },
                2,
            )


class PgVectorEnrichTests(unittest.IsolatedAsyncioTestCase):
    async def test_enrich_upgrades_metadata_with_revision_fence(self):
        from app.ingestion.intelligence import DocumentIntelligence

        job, document, chunk, prepared = fixtures()
        cursor = RecordingCursor()
        connection = RecordingConnection(cursor)

        class FakeIntelligence:
            async def analyze(self, _document, _chunks, *, cancel_event=None):
                return DocumentIntelligence("guide", "Model summary", ("t1",), "model", "m", None)

        sink = PgVectorIndexSink(
            lambda: ConnectionContext(connection),
            FakeEmbeddings(prepared),
            expected_dimension=3,
            intelligence=FakeIntelligence(),
        )
        await sink.enrich(job, document, (chunk,), cancel_event=asyncio.Event())
        self.assertEqual(connection.commits, 1)
        updates = [call for call in cursor.calls if "SET quick_summary" in call[0]]
        self.assertEqual(len(updates), 1)
        query, params = updates[0]
        self.assertIn("current_revision = %s", query)
        self.assertEqual(params[0], "Model summary")
        self.assertEqual(params[3], "model")
        self.assertEqual(params[4:], (job.file_id, job.user_id, job.revision))

    async def test_enrich_never_raises_and_skips_superseded_revisions(self):
        job, document, chunk, prepared = fixtures()
        cursor = RecordingCursor()
        cursor.rowcount = 0  # fence miss: revision superseded or revoked
        connection = RecordingConnection(cursor)

        class FakeIntelligence:
            async def analyze(self, _document, _chunks, *, cancel_event=None):
                from app.ingestion.intelligence import DocumentIntelligence

                return DocumentIntelligence("guide", "s", (), "model", "m", None)

        sink = PgVectorIndexSink(
            lambda: ConnectionContext(connection),
            FakeEmbeddings(prepared),
            expected_dimension=3,
            intelligence=FakeIntelligence(),
        )
        await sink.enrich(job, document, (chunk,), cancel_event=asyncio.Event())

        class BrokenIntelligence:
            async def analyze(self, _document, _chunks, *, cancel_event=None):
                raise RuntimeError("model down")

        sink_broken = PgVectorIndexSink(
            lambda: ConnectionContext(connection),
            FakeEmbeddings(prepared),
            expected_dimension=3,
            intelligence=BrokenIntelligence(),
        )
        await sink_broken.enrich(job, document, (chunk,), cancel_event=asyncio.Event())

        sink_none = PgVectorIndexSink(
            lambda: ConnectionContext(connection),
            FakeEmbeddings(prepared),
            expected_dimension=3,
        )
        await sink_none.enrich(job, document, (chunk,), cancel_event=asyncio.Event())


class OllamaEmbeddingPriorityTests(unittest.IsolatedAsyncioTestCase):
    async def test_closed_priority_gate_defers_before_the_first_model_batch(self):
        calls = []

        async def blocked():
            calls.append("gate")
            return False

        client = OllamaEmbeddingClient(
            EmbeddingSettings(dimension=1, batch_size=1, max_attempts=3),
            priority_gate=blocked,
            priority_timeout_seconds=0.1,
        )
        client._post = lambda _texts: self.fail("closed priority gate called the embedding model")

        with self.assertRaises(BackgroundInferenceDeferred):
            await client.embed(["one", "two"])

        self.assertEqual(calls, ["gate"])

    async def test_cancellation_closes_inflight_requests_and_starts_nothing_new(self):
        started = asyncio.Event()
        never_finishes = asyncio.Event()
        calls = 0
        lock = asyncio.Lock()

        async def handler(_request):
            nonlocal calls
            async with lock:
                calls += 1
                if calls >= 2:
                    started.set()
            try:
                await never_finishes.wait()
            finally:
                pass
            raise AssertionError("cancelled request unexpectedly resumed")

        client = OllamaEmbeddingClient(
            EmbeddingSettings(dimension=1, batch_size=1, max_attempts=3, concurrency=2),
            transport=httpx.MockTransport(handler),
        )
        cancelled = asyncio.Event()
        task = asyncio.create_task(
            client.embed(["one", "two", "three", "four"], cancel_event=cancelled)
        )
        await asyncio.wait_for(started.wait(), timeout=2.0)

        cancelled.set()
        with self.assertRaises(IngestionCancelled):
            await asyncio.wait_for(task, timeout=2.0)
        # The initial concurrency window may start, but cancellation must
        # prevent every batch beyond it from reaching the model.
        self.assertEqual(calls, 2)

    async def test_parallel_batches_preserve_input_order(self):
        import json as jsonlib

        async def handler(request):
            body = jsonlib.loads(request.content.decode("utf-8"))
            data = [
                {"index": i, "embedding": [float(len(text))]}
                for i, text in enumerate(body["input"])
            ]
            await asyncio.sleep(0.01)
            return httpx.Response(200, json={"data": data})

        client = OllamaEmbeddingClient(
            EmbeddingSettings(dimension=1, batch_size=1, max_attempts=1, concurrency=4),
            transport=httpx.MockTransport(handler),
        )
        try:
            prepared = await client.embed(["a", "bb", "ccc", "dddd"])
        finally:
            await client.aclose()
        self.assertEqual(
            prepared.vectors, ((1.0,), (2.0,), (3.0,), (4.0,))
        )


if __name__ == "__main__":
    unittest.main()
