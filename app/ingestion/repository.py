"""PostgreSQL-backed durable job leasing with revision fencing.

The connection factory is injected so this module has no import-time dependency
on psycopg or on the legacy database singleton.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol

from .models import IngestionJob, JobState, validate_job_transition


class CursorLike(Protocol):
    rowcount: int

    def execute(self, query: str, params: object = None) -> Any: ...

    def executemany(self, query: str, params_seq: object) -> Any: ...

    def fetchone(self) -> Any: ...

    def fetchall(self) -> list[Any]: ...


class ConnectionLike(Protocol):
    def cursor(self) -> CursorLike: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


ConnectionFactory = Callable[[], AbstractContextManager[ConnectionLike]]


class JobRepository(Protocol):
    def claim_next(self, worker_id: str, lease_seconds: int) -> IngestionJob | None: ...

    def transition(
        self,
        job: IngestionJob,
        worker_id: str,
        current: JobState,
        target: JobState,
        *,
        parser_fingerprint: str | None = None,
    ) -> bool: ...

    def heartbeat(self, job: IngestionJob, worker_id: str, lease_seconds: int) -> bool: ...

    def should_cancel(self, job: IngestionJob, worker_id: str) -> bool: ...

    def finish(
        self,
        job: IngestionJob,
        worker_id: str,
        current: JobState,
        target: JobState,
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> bool: ...

    def requeue(
        self,
        job: IngestionJob,
        worker_id: str,
        current: JobState,
        *,
        delay_seconds: int,
        error_code: str,
        error_detail: str,
    ) -> bool: ...

    def defer(
        self,
        job: IngestionJob,
        worker_id: str,
        current: JobState,
        *,
        delay_seconds: int,
    ) -> bool: ...


CLAIM_NEXT_SQL = """
WITH exhausted AS (
    UPDATE ingestion_jobs
    SET state = 'failed',
        lease_owner = NULL,
        lease_expires_at = NULL,
        error_code = 'worker_lease_expired',
        error_detail = 'Worker lease expired after the final attempt',
        finished_at = COALESCE(finished_at, NOW()),
        updated_at = NOW()
    WHERE state IN (
            'decrypting', 'converting', 'parsing', 'chunking',
            'embedding', 'publishing'
        )
      AND lease_expires_at < NOW()
      AND attempts >= max_attempts
    RETURNING file_id, user_id, revision
), failed_files AS (
    UPDATE files AS f
    SET ai_status = 'failed',
        ai_error_code = 'worker_lease_expired',
        ai_error_detail = 'Worker lease expired after the final attempt'
    FROM exhausted AS e
    WHERE f.id = e.file_id
      AND f.user_id = e.user_id
      AND f.desired_revision = e.revision
    RETURNING f.id
), stale_files AS (
    UPDATE files AS f
    SET ai_status = 'ready'
    WHERE f.ai_status IN (
            'queued', 'decrypting', 'converting', 'parsing',
            'chunking', 'embedding', 'publishing'
        )
      AND f.current_revision IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM ingestion_jobs ij
          WHERE ij.file_id = f.id
            AND ij.user_id = f.user_id
            AND ij.revision = f.desired_revision
            AND ij.state IN (
                'queued', 'decrypting', 'converting', 'parsing',
                'chunking', 'embedding', 'publishing'
            )
      )
    RETURNING f.id
), candidate AS (
    SELECT j.id
    FROM ingestion_jobs AS j
    JOIN files AS f ON f.id = j.file_id AND f.user_id = j.user_id
    WHERE (
        (j.state = 'queued' AND j.available_at <= NOW())
        OR (
            j.state IN (
                'decrypting', 'converting', 'parsing', 'chunking',
                'embedding', 'publishing'
            )
            AND j.lease_expires_at < NOW()
        )
    )
      AND j.cancel_requested_at IS NULL
      AND j.attempts < j.max_attempts
      AND f.is_deleted = FALSE
      AND f.user_granted_ai_access = TRUE
      AND f.desired_revision = j.revision
    -- Shortest-job-first within a priority: small files stop queueing
    -- behind multi-hour giants (bulk backfill drains fastest this way).
    -- Explicit priority still wins; available_at keeps FIFO among equals.
    ORDER BY j.priority DESC, f.file_size_bytes ASC NULLS LAST, j.available_at ASC, j.created_at ASC
    FOR UPDATE OF j SKIP LOCKED
    LIMIT 1
)
UPDATE ingestion_jobs AS j
SET state = 'decrypting',
    attempts = j.attempts + 1,
    lease_owner = %s,
    lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
    started_at = COALESCE(j.started_at, NOW()),
    error_code = NULL,
    error_detail = NULL,
    updated_at = NOW()
FROM candidate AS c, files AS f
WHERE j.id = c.id
  AND f.id = j.file_id
RETURNING
    j.id, j.file_id, j.user_id, j.revision, j.state, j.priority,
    j.attempts, j.max_attempts, j.lease_owner, j.lease_expires_at,
    (j.cancel_requested_at IS NOT NULL) AS cancel_requested,
    j.metadata || jsonb_build_object(
        's3_bucket_name', f.s3_bucket_name,
        's3_path', f.s3_path,
        's3_key_path', f.s3_key_path,
        'encrypted_filename', f.encrypted_filename,
        'rsa_key_filename', f.rsa_key_filename,
        'file_size_bytes', f.file_size_bytes
    ) AS metadata,
    f.original_filename AS source_name,
    f.mime_type AS media_type,
    COALESCE(f.sha256_hash, '') AS source_sha256
"""


class PostgresJobRepository:
    """Small SQL repository; publication remains an injected sink concern."""

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        self.connection_factory = connection_factory

    def claim_next(self, worker_id: str, lease_seconds: int = 90) -> IngestionJob | None:
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(CLAIM_NEXT_SQL, (worker_id, lease_seconds))
            row = cursor.fetchone()
            if not row:
                connection.commit()
                return None
            job = self._job(row)
            if not self._set_file_status(cursor, job, JobState.DECRYPTING):
                connection.rollback()
                return None
            connection.commit()
        return job

    def transition(
        self,
        job: IngestionJob,
        worker_id: str,
        current: JobState,
        target: JobState,
        *,
        parser_fingerprint: str | None = None,
    ) -> bool:
        validate_job_transition(current, target)
        if target.terminal:
            raise ValueError("use finish() for a terminal transition")
        query = """
            UPDATE ingestion_jobs AS j
            SET state = %s,
                parser_fingerprint = COALESCE(%s, j.parser_fingerprint),
                updated_at = NOW()
            FROM files AS f
            WHERE j.id = %s
              AND j.file_id = %s
              AND j.user_id = %s
              AND j.revision = %s
              AND j.state = %s
              AND j.lease_owner = %s
              AND j.lease_expires_at > NOW()
              AND j.cancel_requested_at IS NULL
              AND f.id = j.file_id
              AND f.is_deleted = FALSE
              AND f.user_granted_ai_access = TRUE
              AND f.desired_revision = j.revision
        """
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                query,
                (
                    target.value,
                    parser_fingerprint,
                    job.job_id,
                    job.file_id,
                    job.user_id,
                    job.revision,
                    current.value,
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.commit()
                return False
            if not self._set_file_status(cursor, job, target):
                connection.rollback()
                return False
            connection.commit()
            return True

    def heartbeat(self, job: IngestionJob, worker_id: str, lease_seconds: int = 90) -> bool:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        query = """
            UPDATE ingestion_jobs AS j
            SET lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                updated_at = NOW()
            FROM files AS f
            WHERE j.id = %s
              AND j.file_id = %s
              AND j.user_id = %s
              AND j.revision = %s
              AND j.lease_owner = %s
              AND j.state IN (
                  'decrypting', 'converting', 'parsing', 'chunking',
                  'embedding', 'publishing'
              )
              AND j.cancel_requested_at IS NULL
              AND f.id = j.file_id
              AND f.is_deleted = FALSE
              AND f.user_granted_ai_access = TRUE
              AND f.desired_revision = j.revision
        """
        return self._execute_count(
            query,
            (
                lease_seconds,
                job.job_id,
                job.file_id,
                job.user_id,
                job.revision,
                worker_id,
            ),
        )

    def should_cancel(self, job: IngestionJob, worker_id: str) -> bool:
        query = """
            SELECT (
                j.cancel_requested_at IS NOT NULL
                OR j.lease_owner IS DISTINCT FROM %s
                OR j.lease_expires_at <= NOW()
                OR f.is_deleted = TRUE
                OR f.user_granted_ai_access = FALSE
                OR f.desired_revision IS DISTINCT FROM j.revision
            ) AS should_cancel
            FROM ingestion_jobs AS j
            JOIN files AS f ON f.id = j.file_id AND f.user_id = j.user_id
            WHERE j.id = %s
              AND j.file_id = %s
              AND j.user_id = %s
              AND j.revision = %s
        """
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                query,
                (worker_id, job.job_id, job.file_id, job.user_id, job.revision),
            )
            row = cursor.fetchone()
        if not row:
            return True
        if isinstance(row, Mapping):
            return bool(row["should_cancel"])
        return bool(row[0])

    def finish(
        self,
        job: IngestionJob,
        worker_id: str,
        current: JobState,
        target: JobState,
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> bool:
        validate_job_transition(current, target)
        if not target.terminal:
            raise ValueError("finish target must be terminal")
        # READY is revision-fenced and consent-fenced. Failure/cancellation must
        # still be recordable after a revoke or desired-revision change.
        ready_fence = target is JobState.READY
        query = """
            UPDATE ingestion_jobs AS j
            SET state = %s,
                error_code = %s,
                error_detail = %s,
                lease_owner = NULL,
                lease_expires_at = NULL,
                finished_at = NOW(),
                updated_at = NOW()
            FROM files AS f
            WHERE j.id = %s
              AND j.file_id = %s
              AND j.user_id = %s
              AND j.revision = %s
              AND j.state = %s
              AND j.lease_owner = %s
              AND f.id = j.file_id
              AND (
                  %s = FALSE
                  OR (
                      j.cancel_requested_at IS NULL
                      AND f.is_deleted = FALSE
                      AND f.user_granted_ai_access = TRUE
                      AND f.desired_revision = j.revision
                  )
              )
        """
        detail = (error_detail or "")[:2_000] or None
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                query,
                (
                    target.value,
                    error_code,
                    detail,
                    job.job_id,
                    job.file_id,
                    job.user_id,
                    job.revision,
                    current.value,
                    worker_id,
                    ready_fence,
                ),
            )
            if cursor.rowcount != 1:
                connection.commit()
                return False
            file_changed = self._set_file_status(
                cursor,
                job,
                target,
                error_code=error_code,
                error_detail=detail,
            )
            if ready_fence and not file_changed:
                connection.rollback()
                return False
            connection.commit()
            return True

    def requeue(
        self,
        job: IngestionJob,
        worker_id: str,
        current: JobState,
        *,
        delay_seconds: int,
        error_code: str,
        error_detail: str,
    ) -> bool:
        validate_job_transition(current, JobState.QUEUED)
        if delay_seconds < 0:
            raise ValueError("delay_seconds cannot be negative")
        query = """
            UPDATE ingestion_jobs AS j
            SET state = 'queued',
                available_at = NOW() + (%s * INTERVAL '1 second'),
                lease_owner = NULL,
                lease_expires_at = NULL,
                error_code = %s,
                error_detail = %s,
                updated_at = NOW()
            FROM files AS f
            WHERE j.id = %s
              AND j.file_id = %s
              AND j.user_id = %s
              AND j.revision = %s
              AND j.state = %s
              AND j.lease_owner = %s
              AND j.cancel_requested_at IS NULL
              AND j.attempts < j.max_attempts
              AND f.id = j.file_id
              AND f.user_id = j.user_id
              AND f.is_deleted = FALSE
              AND f.user_granted_ai_access = TRUE
              AND f.desired_revision = j.revision
        """
        detail = error_detail[:2_000]
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                query,
                (
                    delay_seconds,
                    error_code,
                    detail,
                    job.job_id,
                    job.file_id,
                    job.user_id,
                    job.revision,
                    current.value,
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.commit()
                return False
            if not self._set_file_status(
                cursor,
                job,
                JobState.QUEUED,
                error_code=error_code,
                error_detail=detail,
            ):
                connection.rollback()
                return False
            connection.commit()
            return True

    def defer(
        self,
        job: IngestionJob,
        worker_id: str,
        current: JobState,
        *,
        delay_seconds: int,
    ) -> bool:
        """Release model-deferred work without charging the claim attempt."""

        validate_job_transition(current, JobState.QUEUED)
        if delay_seconds < 1:
            raise ValueError("delay_seconds must be positive")
        query = """
            UPDATE ingestion_jobs AS j
            SET state = 'queued',
                attempts = GREATEST(0, j.attempts - 1),
                available_at = NOW() + (%s * INTERVAL '1 second'),
                lease_owner = NULL,
                lease_expires_at = NULL,
                error_code = NULL,
                error_detail = NULL,
                updated_at = NOW()
            FROM files AS f
            WHERE j.id = %s
              AND j.file_id = %s
              AND j.user_id = %s
              AND j.revision = %s
              AND j.state = %s
              AND j.lease_owner = %s
              AND j.cancel_requested_at IS NULL
              AND f.id = j.file_id
              AND f.user_id = j.user_id
              AND f.is_deleted = FALSE
              AND f.user_granted_ai_access = TRUE
              AND f.desired_revision = j.revision
        """
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                query,
                (
                    delay_seconds,
                    job.job_id,
                    job.file_id,
                    job.user_id,
                    job.revision,
                    current.value,
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.commit()
                return False
            if not self._set_file_status(cursor, job, JobState.QUEUED):
                connection.rollback()
                return False
            connection.commit()
            return True

    @staticmethod
    def _set_file_status(
        cursor: CursorLike,
        job: IngestionJob,
        state: JobState,
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> bool:
        cursor.execute(
            """
            UPDATE files
            SET ai_status = %s,
                ai_error_code = %s,
                ai_error_detail = %s
            WHERE id = %s
              AND user_id = %s
              AND desired_revision = %s
              AND is_deleted = FALSE
              AND user_granted_ai_access = TRUE
            """,
            (
                state.value,
                error_code,
                (error_detail or "")[:2_000] or None,
                job.file_id,
                job.user_id,
                job.revision,
            ),
        )
        return cursor.rowcount == 1

    def _execute_count(self, query: str, params: tuple[Any, ...]) -> bool:
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(query, params)
            changed = cursor.rowcount == 1
            connection.commit()
        return changed

    @staticmethod
    def _job(row: Mapping[str, Any]) -> IngestionJob:
        if not isinstance(row, Mapping):
            raise TypeError("ingestion repository requires mapping database rows")
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return IngestionJob(
            job_id=str(row["id"]),
            file_id=int(row["file_id"]),
            user_id=int(row["user_id"]),
            revision=int(row["revision"]),
            state=(row["state"] if isinstance(row["state"], JobState) else JobState(str(row["state"]))),
            source_name=str(row.get("source_name") or ""),
            media_type=str(row.get("media_type") or "application/octet-stream"),
            source_sha256=str(row.get("source_sha256") or ""),
            priority=int(row.get("priority") or 0),
            attempts=int(row.get("attempts") or 0),
            max_attempts=int(row.get("max_attempts") or 3),
            lease_owner=row.get("lease_owner"),
            lease_expires_at=row.get("lease_expires_at"),
            cancel_requested=bool(row.get("cancel_requested", False)),
            metadata=metadata,
        )
