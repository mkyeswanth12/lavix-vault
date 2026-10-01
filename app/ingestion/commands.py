"""Transactional API-side commands for the durable ingestion lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import UnsupportedFormatError
from .models import JobState
from .repository import ConnectionFactory, CursorLike
from .routing import route_file


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    file_id: int
    revision: int
    job_id: str
    state: JobState
    created: bool


@dataclass(frozen=True, slots=True)
class BulkEnqueueResult:
    items: tuple[EnqueueResult, ...]
    skipped_unsupported: int = 0

    @property
    def created_count(self) -> int:
        return sum(item.created for item in self.items)


@dataclass(frozen=True, slots=True)
class RemovalResult:
    file_count: int
    cancelled_job_count: int
    removed_revision_count: int
    removed_chunk_count: int


@dataclass(frozen=True, slots=True)
class CancellationResult:
    file_count: int
    cancelled_job_count: int
    ready_file_count: int


@dataclass(frozen=True, slots=True)
class IngestionStatus:
    file_id: int
    consent_granted: bool
    desired_revision: int
    current_revision: int | None
    state: str
    job_id: str | None
    chunk_count: int
    error_code: str | None = None
    error_detail: str | None = None


LOCK_FILE_SQL = """
SELECT id, user_id, original_filename, mime_type, desired_revision,
       current_revision, user_granted_ai_access
FROM files
WHERE id = %s
  AND user_id = %s
  AND is_deleted = FALSE
FOR UPDATE
"""


LOCK_USER_FILES_SQL = """
SELECT id, user_id, original_filename, mime_type, desired_revision,
       current_revision, user_granted_ai_access
FROM files
WHERE user_id = %s
  AND is_deleted = FALSE
ORDER BY id
FOR UPDATE
"""


STATUS_SQL = """
SELECT
    f.id AS file_id,
    f.user_granted_ai_access AS consent_granted,
    f.desired_revision,
    f.current_revision,
    CASE
        WHEN f.user_granted_ai_access = FALSE THEN 'not_granted'
        WHEN f.ai_status IN (
                 'queued', 'decrypting', 'converting', 'parsing',
                 'chunking', 'embedding', 'publishing'
             )
             AND j.state IS NULL
             AND f.current_revision IS NOT NULL THEN 'ready'
        ELSE COALESCE(
            f.ai_status,
            j.state,
            CASE WHEN f.current_revision IS NOT NULL THEN 'ready' ELSE 'queued' END
        )
    END AS state,
    j.id AS job_id,
    COALESCE((
        SELECT COUNT(*)
        FROM document_chunks AS dc
        WHERE dc.file_id = f.id
          AND dc.user_id = f.user_id
          AND dc.revision = f.current_revision
    ), 0) AS chunk_count,
    CASE
        WHEN f.user_granted_ai_access = FALSE THEN NULL
        WHEN f.ai_status = 'ready' AND f.current_revision IS NOT NULL
        THEN f.ai_error_code
        ELSE COALESCE(f.ai_error_code, j.error_code)
    END AS error_code,
    CASE
        WHEN f.user_granted_ai_access = FALSE THEN NULL
        WHEN f.ai_status = 'ready' AND f.current_revision IS NOT NULL
        THEN f.ai_error_detail
        ELSE COALESCE(f.ai_error_detail, j.error_detail)
    END AS error_detail
FROM files AS f
LEFT JOIN ingestion_jobs AS j
  ON j.file_id = f.id
 AND j.user_id = f.user_id
 AND j.revision = f.desired_revision
WHERE f.id = %s
  AND f.user_id = %s
  AND f.is_deleted = FALSE
"""


class IngestionCommands:
    """Synchronous commands intended to be called from an API thread pool."""

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        self.connection_factory = connection_factory

    def grant_and_enqueue(
        self,
        user_id: int,
        file_id: int,
        *,
        idempotency_key: str,
        priority: int = 0,
    ) -> EnqueueResult:
        key = self._key(idempotency_key)
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(LOCK_FILE_SQL, (file_id, user_id))
            row = cursor.fetchone()
            if row is None:
                raise FileNotFoundError("file not found")
            result = self._enqueue_locked(cursor, self._row(row), key, priority)
            connection.commit()
            return result

    def grant_and_enqueue_all(
        self,
        user_id: int,
        *,
        idempotency_key: str,
        priority: int = 0,
    ) -> BulkEnqueueResult:
        batch_key = self._key(idempotency_key, max_length=160)
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(LOCK_USER_FILES_SQL, (user_id,))
            rows = [self._row(row) for row in cursor.fetchall()]
            results: list[EnqueueResult] = []
            skipped = 0
            for row in rows:
                route = route_file(
                    str(row.get("original_filename") or ""),
                    str(row.get("mime_type") or ""),
                )
                if not route.supported:
                    skipped += 1
                    continue
                results.append(
                    self._enqueue_locked(
                        cursor,
                        row,
                        f"{batch_key}:{int(row['id'])}",
                        priority,
                        route_checked=True,
                    )
                )
            connection.commit()
            return BulkEnqueueResult(tuple(results), skipped)

    def revoke_and_remove(self, user_id: int, file_id: int) -> RemovalResult:
        """Revoke one file and physically remove derived data, never the source."""

        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(LOCK_FILE_SQL, (file_id, user_id))
            if cursor.fetchone() is None:
                raise FileNotFoundError("file not found")
            cancelled = self._cancel_jobs(
                cursor,
                "file_id = %s AND user_id = %s",
                (file_id, user_id),
                code="access_revoked",
            )
            cursor.execute(
                "DELETE FROM document_chunks WHERE file_id = %s AND user_id = %s",
                (file_id, user_id),
            )
            removed_chunks = cursor.rowcount
            cursor.execute(
                "DELETE FROM document_revisions WHERE file_id = %s AND user_id = %s",
                (file_id, user_id),
            )
            removed_revisions = cursor.rowcount
            cursor.execute(
                """
                UPDATE files
                SET user_granted_ai_access = FALSE,
                    current_revision = NULL,
                    ai_status = 'not_granted',
                    ai_error_code = NULL,
                    ai_error_detail = NULL,
                    quick_summary = NULL,
                    quick_tags = NULL,
                    doc_type = NULL,
                    intelligence_status = 'pending'
                WHERE id = %s
                  AND user_id = %s
                  AND is_deleted = FALSE
                """,
                (file_id, user_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError("file ownership fence changed during revoke")
            connection.commit()
            return RemovalResult(1, cancelled, removed_revisions, removed_chunks)

    def cancel_indexing(self, user_id: int) -> CancellationResult:
        """Cancel unfinished work without revoking consent or deleting an index."""

        with self.connection_factory() as connection:
            cursor = connection.cursor()
            # Serialize with publication before changing a job. This prevents a
            # late publisher from replacing the revision selected below.
            cursor.execute(LOCK_USER_FILES_SQL, (user_id,))
            cursor.fetchall()
            cursor.execute(
                """
                UPDATE ingestion_jobs
                SET state = 'cancelled',
                    cancel_requested_at = COALESCE(cancel_requested_at, NOW()),
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    error_code = 'cancelled_by_user',
                    error_detail = NULL,
                    finished_at = COALESCE(finished_at, NOW()),
                    updated_at = NOW()
                WHERE user_id = %s
                  AND state IN (
                      'queued', 'decrypting', 'converting', 'parsing',
                      'chunking', 'embedding', 'publishing'
                  )
                RETURNING file_id
                """,
                (user_id,),
            )
            cancelled_rows = cursor.fetchall()
            cancelled_job_count = len(cancelled_rows)
            file_ids = sorted({int(self._value(row, "file_id")) for row in cancelled_rows})
            ready_file_count = 0
            if file_ids:
                cursor.execute(
                    """
                    UPDATE files
                    SET ai_status = CASE
                            WHEN current_revision IS NOT NULL THEN 'ready'
                            ELSE 'cancelled'
                        END,
                        ai_error_code = CASE
                            WHEN current_revision IS NULL THEN 'cancelled_by_user'
                            ELSE NULL
                        END,
                        ai_error_detail = NULL
                    WHERE user_id = %s
                      AND id = ANY(%s)
                      AND is_deleted = FALSE
                      AND user_granted_ai_access = TRUE
                    RETURNING current_revision
                    """,
                    (user_id, file_ids),
                )
                changed = cursor.fetchall()
                ready_file_count = sum(self._value(row, "current_revision") is not None for row in changed)
            connection.commit()
            return CancellationResult(
                file_count=len(file_ids),
                cancelled_job_count=cancelled_job_count,
                ready_file_count=ready_file_count,
            )

    def cancel_file_indexing(self, user_id: int, file_id: int) -> CancellationResult:
        """Cancel one file's unfinished work while retaining its published revision."""

        with self.connection_factory() as connection:
            cursor = connection.cursor()
            # Serialize with publication and enforce ownership before touching
            # either the job or the file state.
            cursor.execute(LOCK_FILE_SQL, (file_id, user_id))
            file_row = cursor.fetchone()
            if file_row is None:
                raise FileNotFoundError("file not found")

            cancelled = self._cancel_jobs(
                cursor,
                "file_id = %s AND user_id = %s",
                (file_id, user_id),
                code="cancelled_by_user",
            )
            ready_file_count = 0
            if cancelled:
                cursor.execute(
                    """
                    UPDATE files
                    SET ai_status = CASE
                            WHEN current_revision IS NOT NULL THEN 'ready'
                            ELSE 'cancelled'
                        END,
                        ai_error_code = CASE
                            WHEN current_revision IS NULL THEN 'cancelled_by_user'
                            ELSE NULL
                        END,
                        ai_error_detail = NULL
                    WHERE id = %s
                      AND user_id = %s
                      AND is_deleted = FALSE
                      AND user_granted_ai_access = TRUE
                    RETURNING current_revision
                    """,
                    (file_id, user_id),
                )
                changed = cursor.fetchone()
                ready_file_count = int(
                    changed is not None and self._value(changed, "current_revision") is not None
                )

            connection.commit()
            return CancellationResult(
                file_count=int(cancelled > 0),
                cancelled_job_count=cancelled,
                ready_file_count=ready_file_count,
            )

    def trash_file(self, user_id: int, file_id: int) -> CancellationResult:
        """Soft-delete a source while preserving consent and published intelligence."""

        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(LOCK_FILE_SQL, (file_id, user_id))
            file_row = cursor.fetchone()
            if file_row is None:
                raise FileNotFoundError("file not found")
            row = self._row(file_row)
            cancelled = self._cancel_jobs(
                cursor,
                "file_id = %s AND user_id = %s",
                (file_id, user_id),
                code="file_trashed",
            )
            cursor.execute(
                """
                UPDATE files
                SET is_deleted = TRUE,
                    deleted_at = NOW(),
                    ai_status = CASE
                        WHEN user_granted_ai_access = FALSE THEN 'not_granted'
                        WHEN current_revision IS NOT NULL THEN 'ready'
                        WHEN ai_status IN (
                            'queued', 'decrypting', 'converting', 'parsing',
                            'chunking', 'embedding', 'publishing'
                        ) THEN 'cancelled'
                        ELSE ai_status
                    END,
                    ai_error_code = CASE
                        WHEN user_granted_ai_access = TRUE
                             AND current_revision IS NULL
                             AND ai_status IN (
                                 'queued', 'decrypting', 'converting', 'parsing',
                                 'chunking', 'embedding', 'publishing'
                             )
                        THEN 'file_trashed'
                        WHEN current_revision IS NOT NULL THEN NULL
                        ELSE ai_error_code
                    END,
                    ai_error_detail = CASE
                        WHEN current_revision IS NOT NULL OR ai_status IN (
                            'queued', 'decrypting', 'converting', 'parsing',
                            'chunking', 'embedding', 'publishing'
                        ) THEN NULL
                        ELSE ai_error_detail
                    END
                WHERE id = %s
                  AND user_id = %s
                  AND is_deleted = FALSE
                """,
                (file_id, user_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError("file ownership fence changed during trash")
            connection.commit()
            ready = bool(row.get("user_granted_ai_access")) and row.get("current_revision") is not None
            return CancellationResult(1, cancelled, int(ready))

    def status(self, user_id: int, file_id: int) -> IngestionStatus | None:
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(STATUS_SQL, (file_id, user_id))
            raw = cursor.fetchone()
        if raw is None:
            return None
        row = self._row(raw)
        return IngestionStatus(
            file_id=int(row["file_id"]),
            consent_granted=bool(row["consent_granted"]),
            desired_revision=int(row.get("desired_revision") or 0),
            current_revision=(
                int(row["current_revision"]) if row.get("current_revision") is not None else None
            ),
            state=str(row["state"]),
            job_id=str(row["job_id"]) if row.get("job_id") is not None else None,
            chunk_count=int(row.get("chunk_count") or 0),
            error_code=(str(row["error_code"]) if row.get("error_code") is not None else None),
            error_detail=(str(row["error_detail"]) if row.get("error_detail") is not None else None),
        )

    def _enqueue_locked(
        self,
        cursor: CursorLike,
        file_row: Mapping[str, Any],
        key: str,
        priority: int,
        *,
        route_checked: bool = False,
    ) -> EnqueueResult:
        file_id = int(file_row["id"])
        user_id = int(file_row["user_id"])
        if not route_checked:
            route = route_file(
                str(file_row.get("original_filename") or ""),
                str(file_row.get("mime_type") or ""),
            )
            if not route.supported:
                raise UnsupportedFormatError(route.reason)

        cursor.execute(
            """
            SELECT id, revision, state
            FROM ingestion_jobs
            WHERE file_id = %s
              AND user_id = %s
              AND (
                  metadata ->> 'idempotency_key' = %s
                  OR COALESCE(metadata -> 'idempotency_keys', '[]'::jsonb) ? %s
              )
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (file_id, user_id, key, key),
        )
        existing = cursor.fetchone()
        if existing is not None:
            return self._existing_result(file_id, existing)

        # Grant is a state-setting command, not an implicit reindex command.
        # A different request key must not create another revision while the
        # currently desired revision is already queued, running, or ready.
        if bool(file_row.get("user_granted_ai_access")):
            cursor.execute(
                """
                SELECT id, revision, state
                FROM ingestion_jobs
                WHERE file_id = %s
                  AND user_id = %s
                  AND revision = %s
                  AND state IN (
                      'queued', 'decrypting', 'converting', 'parsing',
                      'chunking', 'embedding', 'publishing', 'ready'
                  )
                LIMIT 1
                """,
                (file_id, user_id, int(file_row.get("desired_revision") or 0)),
            )
            current = cursor.fetchone()
            if current is not None:
                self._remember_key(cursor, str(self._row(current)["id"]), key)
                return self._existing_result(file_id, current)

        self._cancel_jobs(
            cursor,
            "file_id = %s AND user_id = %s",
            (file_id, user_id),
            code="superseded_request",
        )
        cursor.execute(
            """
            UPDATE files
            SET user_granted_ai_access = TRUE,
                desired_revision = desired_revision + 1,
                ai_status = 'queued',
                ai_error_code = NULL,
                ai_error_detail = NULL,
                quick_summary = CASE
                    WHEN current_revision IS NULL THEN NULL
                    ELSE quick_summary
                END,
                quick_tags = CASE
                    WHEN current_revision IS NULL THEN NULL
                    ELSE quick_tags
                END,
                doc_type = CASE
                    WHEN current_revision IS NULL THEN NULL
                    ELSE doc_type
                END,
                intelligence_status = CASE
                    WHEN current_revision IS NULL THEN 'pending'
                    ELSE intelligence_status
                END
            WHERE id = %s
              AND user_id = %s
              AND is_deleted = FALSE
            RETURNING desired_revision
            """,
            (file_id, user_id),
        )
        revision_row = cursor.fetchone()
        if revision_row is None:
            raise RuntimeError("file ownership fence changed during enqueue")
        revision = int(self._value(revision_row, "desired_revision"))
        cursor.execute(
            """
            INSERT INTO ingestion_jobs (
                user_id, file_id, revision, state, priority, metadata
            )
            VALUES (%s, %s, %s, 'queued', %s, %s::jsonb)
            RETURNING id, state
            """,
            (
                user_id,
                file_id,
                revision,
                priority,
                json.dumps(
                    {"idempotency_key": key, "idempotency_keys": [key]},
                    separators=(",", ":"),
                ),
            ),
        )
        job_row = cursor.fetchone()
        if job_row is None:
            raise RuntimeError("ingestion job insert returned no identity")
        job = self._row(job_row)
        return EnqueueResult(
            file_id=file_id,
            revision=revision,
            job_id=str(job["id"]),
            state=JobState(str(job["state"])),
            created=True,
        )

    @staticmethod
    def _cancel_jobs(
        cursor: CursorLike,
        predicate: str,
        params: tuple[Any, ...],
        *,
        code: str,
    ) -> int:
        allowed = {
            "file_id = %s AND user_id = %s",
            "user_id = %s",
        }
        if predicate not in allowed:
            raise ValueError("invalid cancellation predicate")
        cursor.execute(
            f"""
            UPDATE ingestion_jobs
            SET state = 'cancelled',
                cancel_requested_at = COALESCE(cancel_requested_at, NOW()),
                lease_owner = NULL,
                lease_expires_at = NULL,
                error_code = %s,
                error_detail = NULL,
                finished_at = COALESCE(finished_at, NOW()),
                updated_at = NOW()
            WHERE {predicate}
              AND state IN (
                  'queued', 'decrypting', 'converting', 'parsing',
                  'chunking', 'embedding', 'publishing'
              )
            """,
            (code, *params),
        )
        return cursor.rowcount

    @staticmethod
    def _key(value: str, *, max_length: int = 200) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > max_length:
            raise ValueError(f"idempotency_key must contain 1-{max_length} characters")
        return normalized

    @classmethod
    def _existing_result(cls, file_id: int, value: Any) -> EnqueueResult:
        row = cls._row(value)
        return EnqueueResult(
            file_id=file_id,
            revision=int(row["revision"]),
            job_id=str(row["id"]),
            state=JobState(str(row["state"])),
            created=False,
        )

    @staticmethod
    def _remember_key(cursor: CursorLike, job_id: str, key: str) -> None:
        cursor.execute(
            """
            UPDATE ingestion_jobs
            SET metadata = jsonb_set(
                metadata,
                '{idempotency_keys}',
                COALESCE(metadata -> 'idempotency_keys', '[]'::jsonb)
                    || jsonb_build_array(%s::text),
                TRUE
            ),
                updated_at = NOW()
            WHERE id = %s
              AND NOT (
                  COALESCE(metadata -> 'idempotency_keys', '[]'::jsonb) ? %s
              )
            """,
            (key, job_id, key),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("failed to attach idempotency key to current job")

    @staticmethod
    def _row(value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("ingestion commands require mapping database rows")
        return value

    @staticmethod
    def _value(row: Any, key: str) -> Any:
        if isinstance(row, Mapping):
            return row[key]
        if isinstance(row, (tuple, list)) and row:
            return row[0]
        raise TypeError("database RETURNING row has an unsupported shape")
