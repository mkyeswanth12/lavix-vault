import inspect
import unittest
from dataclasses import dataclass

import app.ingestion.commands as commands_module
from app.ingestion.commands import IngestionCommands
from app.ingestion.errors import UnsupportedFormatError
from app.ingestion.models import JobState


@dataclass
class Step:
    row: object = None
    rows: object = None
    rowcount: int = 0


class ScriptedCursor:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []
        self.rowcount = 0
        self._row = None
        self._rows = []

    def execute(self, query, params=None):
        self.calls.append((" ".join(query.split()), params))
        if not self.steps:
            raise AssertionError(f"unexpected SQL: {query}")
        step = self.steps.pop(0)
        self.rowcount = step.rowcount
        self._row = step.row
        self._rows = list(step.rows or [])

    def executemany(self, query, params_seq):
        raise AssertionError("commands service must not need executemany")

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class FakeConnection:
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

    def __exit__(self, exc_type, *_args):
        if exc_type is not None:
            self.connection.rollback()
        return False


def file_row(
    file_id=7,
    *,
    name="report.pdf",
    media_type="application/pdf",
    revision=0,
    consent=False,
):
    return {
        "id": file_id,
        "user_id": 3,
        "original_filename": name,
        "mime_type": media_type,
        "desired_revision": revision,
        "current_revision": revision or None,
        "user_granted_ai_access": consent,
    }


class IngestionCommandTests(unittest.TestCase):
    def service(self, steps):
        cursor = ScriptedCursor(steps)
        connection = FakeConnection(cursor)
        service = IngestionCommands(lambda: ConnectionContext(connection))
        return service, cursor, connection

    def test_grant_increments_revision_and_enqueues_once(self):
        service, cursor, connection = self.service(
            [
                Step(row=file_row(), rowcount=1),
                Step(row=None, rowcount=0),
                Step(rowcount=0),
                Step(row={"desired_revision": 1}, rowcount=1),
                Step(row={"id": "job-1", "state": "queued"}, rowcount=1),
            ]
        )
        result = service.grant_and_enqueue(3, 7, idempotency_key="request-1", priority=4)
        self.assertTrue(result.created)
        self.assertEqual(result.revision, 1)
        self.assertEqual(result.job_id, "job-1")
        self.assertIs(result.state, JobState.QUEUED)
        self.assertEqual(connection.commits, 1)
        sql = " ".join(query for query, _params in cursor.calls)
        self.assertIn("desired_revision = desired_revision + 1", sql)
        self.assertIn("INSERT INTO ingestion_jobs", sql)
        insert_params = cursor.calls[-1][1]
        self.assertEqual(insert_params[:4], (3, 7, 1, 4))
        self.assertIn('"idempotency_key":"request-1"', insert_params[4])

        update_sql = cursor.calls[-2][0]
        self.assertIn("WHEN current_revision IS NULL THEN 'pending'", update_sql)
        self.assertIn("WHEN current_revision IS NULL THEN NULL ELSE quick_summary", update_sql)
        self.assertIn("WHEN current_revision IS NULL THEN NULL ELSE quick_tags", update_sql)
        self.assertIn("WHEN current_revision IS NULL THEN NULL ELSE doc_type", update_sql)

    def test_retry_preserves_published_intelligence_until_replacement_is_ready(self):
        failed_reindex = file_row(revision=3, consent=True)
        failed_reindex["current_revision"] = 2
        service, cursor, connection = self.service(
            [
                Step(row=failed_reindex, rowcount=1),
                Step(row=None, rowcount=0),
                Step(row=None, rowcount=0),
                Step(rowcount=0),
                Step(row={"desired_revision": 4}, rowcount=1),
                Step(row={"id": "job-4", "state": "queued"}, rowcount=1),
            ]
        )

        result = service.grant_and_enqueue(3, 7, idempotency_key="retry-after-failure")

        self.assertTrue(result.created)
        self.assertEqual(result.revision, 4)
        self.assertEqual(connection.commits, 1)
        update_sql = cursor.calls[-2][0]
        for column in ("quick_summary", "quick_tags", "doc_type", "intelligence_status"):
            self.assertIn(f"ELSE {column}", update_sql)
        self.assertNotIn("intelligence_status = 'pending'", update_sql)

    def test_same_request_key_returns_existing_job_without_increment(self):
        service, cursor, connection = self.service(
            [
                Step(row=file_row(revision=2, consent=True), rowcount=1),
                Step(
                    row={"id": "job-existing", "revision": 2, "state": "ready"},
                    rowcount=1,
                ),
            ]
        )
        result = service.grant_and_enqueue(3, 7, idempotency_key="same-key")
        self.assertFalse(result.created)
        self.assertEqual(result.revision, 2)
        self.assertIs(result.state, JobState.READY)
        self.assertEqual(len(cursor.calls), 2)
        self.assertEqual(connection.commits, 1)

    def test_grant_is_state_idempotent_across_different_request_keys(self):
        service, cursor, _connection = self.service(
            [
                Step(row=file_row(revision=2, consent=True), rowcount=1),
                Step(row=None, rowcount=0),
                Step(
                    row={
                        "id": "job-running",
                        "revision": 2,
                        "state": "embedding",
                    },
                    rowcount=1,
                ),
                Step(rowcount=1),
            ]
        )
        result = service.grant_and_enqueue(3, 7, idempotency_key="new-key")
        self.assertFalse(result.created)
        self.assertIs(result.state, JobState.EMBEDDING)
        self.assertEqual(len(cursor.calls), 4)
        self.assertIn("idempotency_keys", cursor.calls[-1][0])
        self.assertEqual(cursor.calls[-1][1], ("new-key", "job-running", "new-key"))

    def test_unsupported_file_rolls_back_without_job(self):
        service, cursor, connection = self.service(
            [
                Step(
                    row=file_row(name="movie.mp4", media_type="video/mp4"),
                    rowcount=1,
                )
            ]
        )
        with self.assertRaises(UnsupportedFormatError):
            service.grant_and_enqueue(3, 7, idempotency_key="media")
        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)
        self.assertEqual(len(cursor.calls), 1)

    def test_foreign_or_deleted_file_is_indistinguishable_from_missing(self):
        service, cursor, connection = self.service([Step(row=None, rowcount=0)])
        with self.assertRaises(FileNotFoundError):
            service.grant_and_enqueue(3, 7, idempotency_key="request")
        self.assertIn("user_id = %s", cursor.calls[0][0])
        self.assertIn("is_deleted = FALSE", cursor.calls[0][0])
        self.assertEqual(connection.rollbacks, 1)

    def test_bulk_skips_media_and_does_not_reindex_ready_files(self):
        ready = file_row(9, name="ready.txt", media_type="text/plain", revision=1, consent=True)
        service, cursor, connection = self.service(
            [
                Step(
                    rows=[
                        file_row(7),
                        file_row(8, name="movie.mp4", media_type="video/mp4"),
                        ready,
                    ],
                    rowcount=3,
                ),
                Step(row=None, rowcount=0),
                Step(rowcount=0),
                Step(row={"desired_revision": 1}, rowcount=1),
                Step(row={"id": "job-new", "state": "queued"}, rowcount=1),
                Step(row=None, rowcount=0),
                Step(
                    row={"id": "job-ready", "revision": 1, "state": "ready"},
                    rowcount=1,
                ),
                Step(rowcount=1),
            ]
        )
        result = service.grant_and_enqueue_all(3, idempotency_key="batch-1")
        self.assertEqual(result.skipped_unsupported, 1)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.created_count, 1)
        self.assertTrue(result.items[0].created)
        self.assertFalse(result.items[1].created)
        self.assertEqual(connection.commits, 1)
        self.assertEqual(cursor.steps, [])

    def test_revoke_cancels_jobs_and_removes_only_derived_rows(self):
        service, cursor, connection = self.service(
            [
                Step(row=file_row(revision=2, consent=True), rowcount=1),
                Step(rowcount=2),
                Step(rowcount=14),
                Step(rowcount=2),
                Step(rowcount=1),
            ]
        )
        result = service.revoke_and_remove(3, 7)
        self.assertEqual(result.cancelled_job_count, 2)
        self.assertEqual(result.removed_chunk_count, 14)
        self.assertEqual(result.removed_revision_count, 2)
        self.assertEqual(connection.commits, 1)
        sql = " ".join(query for query, _params in cursor.calls)
        self.assertIn("DELETE FROM document_chunks", sql)
        self.assertIn("DELETE FROM document_revisions", sql)
        self.assertIn("current_revision = NULL", sql)
        self.assertNotIn("DELETE FROM files", sql)

    def test_cancel_indexing_preserves_consent_current_revision_and_chunks(self):
        service, cursor, connection = self.service(
            [
                Step(rows=[file_row(7, revision=2, consent=True), file_row(8, revision=1, consent=True)]),
                Step(rows=[{"file_id": 7}, {"file_id": 7}, {"file_id": 8}], rowcount=3),
                Step(rows=[{"current_revision": 2}, {"current_revision": None}], rowcount=2),
            ]
        )

        result = service.cancel_indexing(3)

        self.assertEqual(result.file_count, 2)
        self.assertEqual(result.cancelled_job_count, 3)
        self.assertEqual(result.ready_file_count, 1)
        self.assertEqual(connection.commits, 1)
        sql = " ".join(query for query, _params in cursor.calls)
        self.assertIn("error_code = 'cancelled_by_user'", sql)
        self.assertIn("WHEN current_revision IS NOT NULL THEN 'ready'", sql)
        self.assertNotIn("SET user_granted_ai_access = FALSE", sql)
        self.assertNotIn("DELETE FROM document_", sql)
        cancel_file_update = cursor.calls[-1][0]
        self.assertNotIn("quick_summary", cancel_file_update)
        self.assertNotIn("quick_tags", cancel_file_update)
        self.assertNotIn("doc_type", cancel_file_update)
        self.assertNotIn("intelligence_status", cancel_file_update)

    def test_cancel_one_file_is_owner_scoped_and_preserves_published_metadata(self):
        service, cursor, connection = self.service(
            [
                Step(row=file_row(7, revision=2, consent=True), rowcount=1),
                Step(rowcount=1),
                Step(row={"current_revision": 2}, rowcount=1),
            ]
        )

        result = service.cancel_file_indexing(3, 7)

        self.assertEqual(result.file_count, 1)
        self.assertEqual(result.cancelled_job_count, 1)
        self.assertEqual(result.ready_file_count, 1)
        self.assertEqual(connection.commits, 1)
        sql = " ".join(query for query, _params in cursor.calls)
        self.assertIn("file_id = %s AND user_id = %s", sql)
        self.assertIn("error_code = %s", sql)
        self.assertIn("WHEN current_revision IS NOT NULL THEN 'ready'", sql)
        self.assertNotIn("SET user_granted_ai_access = FALSE", sql)
        self.assertNotIn("DELETE FROM document_", sql)
        file_update = cursor.calls[-1][0]
        self.assertNotIn("quick_summary", file_update)
        self.assertNotIn("quick_tags", file_update)
        self.assertNotIn("doc_type", file_update)
        self.assertNotIn("intelligence_status", file_update)

    def test_soft_trash_cancels_work_without_revoking_or_removing_index_data(self):
        service, cursor, connection = self.service(
            [
                Step(row=file_row(revision=2, consent=True), rowcount=1),
                Step(rowcount=1),
                Step(rowcount=1),
            ]
        )

        result = service.trash_file(3, 7)

        self.assertEqual(result.cancelled_job_count, 1)
        self.assertEqual(result.ready_file_count, 1)
        self.assertEqual(connection.commits, 1)
        sql = " ".join(query for query, _params in cursor.calls)
        self.assertIn("is_deleted = TRUE", sql)
        self.assertIn("WHEN current_revision IS NOT NULL THEN 'ready'", sql)
        self.assertNotIn("SET user_granted_ai_access = FALSE", sql)
        self.assertNotIn("current_revision = NULL", sql)
        self.assertNotIn("DELETE FROM document_", sql)

    def test_status_is_owner_scoped_and_reports_current_chunk_count(self):
        service, cursor, _connection = self.service(
            [
                Step(
                    row={
                        "file_id": 7,
                        "consent_granted": True,
                        "desired_revision": 2,
                        "current_revision": 2,
                        "state": "ready",
                        "job_id": "job-2",
                        "chunk_count": 18,
                        "error_code": None,
                        "error_detail": None,
                    },
                    rowcount=1,
                )
            ]
        )
        status = service.status(3, 7)
        self.assertEqual(status.chunk_count, 18)
        self.assertEqual(status.state, "ready")
        self.assertEqual(status.current_revision, 2)
        self.assertIn("f.user_id = %s", cursor.calls[0][0])
        self.assertEqual(cursor.calls[0][1], (7, 3))

    def test_ready_publication_does_not_inherit_a_cancelled_replacement_error(self):
        normalized = " ".join(commands_module.STATUS_SQL.lower().split())

        self.assertEqual(
            normalized.count(
                "when f.ai_status = 'ready' and f.current_revision is not null then f.ai_error_"
            ),
            2,
        )
        self.assertIn("then f.ai_error_code else coalesce(f.ai_error_code, j.error_code)", normalized)
        self.assertIn(
            "then f.ai_error_detail else coalesce(f.ai_error_detail, j.error_detail)",
            normalized,
        )

    def test_module_has_no_legacy_queue_or_vector_store_dependency(self):
        source = inspect.getsource(commands_module).lower()
        self.assertNotIn("redis", source)
        self.assertNotIn("chroma", source)
        self.assertNotIn("backgroundtask", source)


if __name__ == "__main__":
    unittest.main()
