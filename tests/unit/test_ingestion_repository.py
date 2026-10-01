import unittest

from app.ingestion.models import JobState
from app.ingestion.repository import CLAIM_NEXT_SQL, PostgresJobRepository


class FakeCursor:
    def __init__(self, row=None, rowcount=1):
        self.row = row
        self.rowcount = rowcount
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append((query, params))

    def fetchone(self):
        return self.row


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

    def __exit__(self, *_args):
        return False


class IngestionRepositoryTests(unittest.TestCase):
    def test_claim_is_skip_locked_and_revision_consent_fenced(self):
        normalized = " ".join(CLAIM_NEXT_SQL.split())
        self.assertIn("FOR UPDATE OF j SKIP LOCKED", normalized)
        self.assertIn("f.desired_revision = j.revision", normalized)
        self.assertIn("f.user_granted_ai_access = TRUE", normalized)
        self.assertIn("'file_size_bytes', f.file_size_bytes", normalized)

    def test_claim_atomically_terminalizes_expired_final_attempts(self):
        normalized = " ".join(CLAIM_NEXT_SQL.split())
        self.assertIn("WITH exhausted AS ( UPDATE ingestion_jobs", normalized)
        self.assertIn("attempts >= max_attempts", normalized)
        self.assertIn("error_code = 'worker_lease_expired'", normalized)
        self.assertIn("lease_owner = NULL", normalized)
        self.assertIn("FROM exhausted AS e", normalized)
        self.assertIn("f.desired_revision = e.revision", normalized)

    def test_claim_maps_source_and_lease_fields(self):
        row = {
            "id": "00000000-0000-0000-0000-000000000001",
            "file_id": 7,
            "user_id": 3,
            "revision": 2,
            "state": "decrypting",
            "priority": 4,
            "attempts": 1,
            "max_attempts": 3,
            "lease_owner": "worker-a",
            "lease_expires_at": None,
            "cancel_requested": False,
            "metadata": {"file_size_bytes": 42},
            "source_name": "report.pdf",
            "media_type": "application/pdf",
            "source_sha256": "a" * 64,
        }
        cursor = FakeCursor(row)
        connection = FakeConnection(cursor)
        repository = PostgresJobRepository(lambda: ConnectionContext(connection))
        job = repository.claim_next("worker-a", 90)
        self.assertEqual(job.file_id, 7)
        self.assertIs(job.state, JobState.DECRYPTING)
        self.assertEqual(job.metadata["file_size_bytes"], 42)
        self.assertEqual(cursor.calls[0][1], ("worker-a", 90))
        self.assertIn("SET ai_status = %s", " ".join(cursor.calls[1][0].split()))
        self.assertEqual(cursor.calls[1][1][0], "decrypting")
        self.assertEqual(connection.commits, 1)

    def test_transition_rejects_terminal_target(self):
        row = {
            "id": "job",
            "file_id": 1,
            "user_id": 1,
            "revision": 1,
            "state": "decrypting",
        }
        job = PostgresJobRepository._job(row)
        repository = PostgresJobRepository(lambda: ConnectionContext(FakeConnection(FakeCursor())))
        with self.assertRaises(ValueError):
            repository.transition(job, "worker", JobState.DECRYPTING, JobState.FAILED)

    def test_transition_updates_visible_file_state_in_same_transaction(self):
        job = PostgresJobRepository._job(
            {
                "id": "job",
                "file_id": 1,
                "user_id": 2,
                "revision": 3,
                "state": "decrypting",
            }
        )
        cursor = FakeCursor(rowcount=1)
        connection = FakeConnection(cursor)
        repository = PostgresJobRepository(lambda: ConnectionContext(connection))
        changed = repository.transition(job, "worker", JobState.DECRYPTING, JobState.PARSING)
        self.assertTrue(changed)
        self.assertEqual(len(cursor.calls), 2)
        self.assertIn("UPDATE ingestion_jobs", cursor.calls[0][0])
        self.assertIn("UPDATE files", cursor.calls[1][0])
        self.assertEqual(cursor.calls[1][1], ("parsing", None, None, 1, 2, 3))
        self.assertEqual(connection.commits, 1)

    def test_requeue_and_terminal_failure_propagate_status_and_error(self):
        job = PostgresJobRepository._job(
            {
                "id": "job",
                "file_id": 1,
                "user_id": 2,
                "revision": 3,
                "state": "parsing",
            }
        )
        requeue_cursor = FakeCursor(rowcount=1)
        requeue_connection = FakeConnection(requeue_cursor)
        repository = PostgresJobRepository(lambda: ConnectionContext(requeue_connection))
        self.assertTrue(
            repository.requeue(
                job,
                "worker",
                JobState.PARSING,
                delay_seconds=5,
                error_code="parser_unavailable",
                error_detail="try again",
            )
        )
        self.assertEqual(
            requeue_cursor.calls[1][1],
            ("queued", "parser_unavailable", "try again", 1, 2, 3),
        )

        finish_cursor = FakeCursor(rowcount=1)
        finish_connection = FakeConnection(finish_cursor)
        repository = PostgresJobRepository(lambda: ConnectionContext(finish_connection))
        self.assertTrue(
            repository.finish(
                job,
                "worker",
                JobState.PARSING,
                JobState.FAILED,
                error_code="parse_failed",
                error_detail="bad input",
            )
        )
        self.assertEqual(
            finish_cursor.calls[1][1],
            ("failed", "parse_failed", "bad input", 1, 2, 3),
        )

    def test_inference_deferral_releases_lease_and_refunds_claim_attempt(self):
        job = PostgresJobRepository._job(
            {
                "id": "job",
                "file_id": 1,
                "user_id": 2,
                "revision": 3,
                "state": "publishing",
                "attempts": 3,
                "max_attempts": 3,
            }
        )
        cursor = FakeCursor(rowcount=1)
        connection = FakeConnection(cursor)
        repository = PostgresJobRepository(lambda: ConnectionContext(connection))

        self.assertTrue(
            repository.defer(
                job,
                "worker",
                JobState.PUBLISHING,
                delay_seconds=5,
            )
        )

        normalized = " ".join(cursor.calls[0][0].split())
        self.assertIn("attempts = GREATEST(0, j.attempts - 1)", normalized)
        self.assertNotIn("j.attempts < j.max_attempts", normalized)
        self.assertIn("f.desired_revision = j.revision", normalized)
        self.assertEqual(cursor.calls[1][1], ("queued", None, None, 1, 2, 3))
        self.assertEqual(connection.commits, 1)


if __name__ == "__main__":
    unittest.main()
