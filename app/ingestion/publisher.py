"""Atomic PostgreSQL/pgvector publication for canonical ingestion output."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from app.graph_memory.inference_priority import BackgroundInferenceDeferred

from .chunking import ChunkingPolicy
from .embedding import OllamaEmbeddingClient, PreparedEmbeddings
from .errors import (
    EmbeddingDimensionError,
    IngestionCancelled,
    PublicationError,
)
from .intelligence import (
    DocumentIntelligence,
    DocumentIntelligenceService,
    deterministic_intelligence,
)
from .models import (
    CanonicalChunk,
    CanonicalDocument,
    IngestionJob,
    PublicationResult,
)
from .repository import ConnectionFactory

logger = logging.getLogger(__name__)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


LOCK_PUBLICATION_FILE_SQL = """
SELECT f.id
FROM files AS f
WHERE f.id = %s
  AND f.user_id = %s
  AND f.desired_revision = %s
  AND f.is_deleted = FALSE
  AND f.user_granted_ai_access = TRUE
FOR UPDATE
"""


LOCK_PUBLICATION_JOB_SQL = """
SELECT j.id
FROM ingestion_jobs AS j
WHERE j.id = %s
  AND j.file_id = %s
  AND j.user_id = %s
  AND j.revision = %s
  AND j.state = 'publishing'
  AND j.lease_owner = %s
  AND j.lease_expires_at > NOW()
  AND j.cancel_requested_at IS NULL
FOR UPDATE
"""


UPSERT_REVISION_SQL = """
INSERT INTO document_revisions (
    user_id, file_id, revision, job_id, source_sha256,
    parser_fingerprint, chunker_fingerprint, embedding_fingerprint,
    embedding_dimension, status, chunk_count, source_size_bytes, metadata,
    completed_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'building', 0, %s, %s::jsonb, NULL)
ON CONFLICT (file_id, revision) DO UPDATE
SET user_id = EXCLUDED.user_id,
    job_id = EXCLUDED.job_id,
    source_sha256 = EXCLUDED.source_sha256,
    parser_fingerprint = EXCLUDED.parser_fingerprint,
    chunker_fingerprint = EXCLUDED.chunker_fingerprint,
    embedding_fingerprint = EXCLUDED.embedding_fingerprint,
    embedding_dimension = EXCLUDED.embedding_dimension,
    status = 'building',
    chunk_count = 0,
    source_size_bytes = EXCLUDED.source_size_bytes,
    metadata = EXCLUDED.metadata,
    completed_at = NULL
WHERE document_revisions.user_id = EXCLUDED.user_id
  AND document_revisions.job_id = EXCLUDED.job_id
"""


INSERT_CHUNK_SQL = """
INSERT INTO document_chunks (
    chunk_id, user_id, file_id, revision, ordinal, content, embedding_text,
    element_ids, provenance, section_path, token_count, metadata, embedding
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s::jsonb, %s, %s, %s::jsonb, %s::vector
)
"""


class PgVectorIndexSink:
    """Embed chunks and atomically move one fenced revision to ``current``."""

    def __init__(
        self,
        connection_factory: ConnectionFactory,
        embeddings: OllamaEmbeddingClient | None = None,
        *,
        chunker_fingerprint: str | None = None,
        expected_dimension: int = 1024,
        intelligence: DocumentIntelligenceService | None = None,
    ) -> None:
        if expected_dimension < 1:
            raise ValueError("expected_dimension must be positive")
        self.connection_factory = connection_factory
        self.embeddings = embeddings or OllamaEmbeddingClient()
        self.chunker_fingerprint = chunker_fingerprint or ChunkingPolicy().fingerprint
        self.expected_dimension = expected_dimension
        self.intelligence = intelligence

    async def embed(
        self,
        job: IngestionJob,
        document: CanonicalDocument,
        chunks: Sequence[CanonicalChunk],
        *,
        cancel_event: asyncio.Event,
    ) -> PreparedEmbeddings:
        del job, document
        if not chunks:
            raise PublicationError("cannot embed an empty revision")
        return await self.embeddings.embed(
            [chunk.embedding_text for chunk in chunks],
            cancel_event=cancel_event,
        )

    async def publish(
        self,
        job: IngestionJob,
        document: CanonicalDocument,
        chunks: Sequence[CanonicalChunk],
        prepared: PreparedEmbeddings,
        *,
        cancel_event: asyncio.Event,
    ) -> PublicationResult:
        if cancel_event.is_set():
            raise IngestionCancelled("publication cancelled before commit")
        self._validate(document, chunks, prepared)
        # Commit path stays LLM-free: deterministic intelligence is instant,
        # so chunks become searchable without waiting 30-120s for the model.
        # The model upgrade runs afterwards via enrich() (best-effort) and the
        # intelligence backfill covers anything left pending.
        intel_start = time.perf_counter()
        intelligence = deterministic_intelligence(document, chunks)
        intelligence_s = round(time.perf_counter() - intel_start, 2)
        if cancel_event.is_set():
            raise IngestionCancelled("publication cancelled before commit")
        commit_start = time.perf_counter()
        await asyncio.to_thread(
            self._publish_sync,
            job,
            document,
            tuple(chunks),
            prepared,
            intelligence,
        )
        logger.info(
            "Publish split job=%s intelligence_s=%s commit_s=%s",
            job.job_id,
            intelligence_s,
            round(time.perf_counter() - commit_start, 2),
        )
        return PublicationResult(job_completed=True)

    async def enrich(
        self,
        job: IngestionJob,
        document: CanonicalDocument,
        chunks: Sequence[CanonicalChunk],
        *,
        cancel_event: asyncio.Event,
    ) -> str:
        """Upgrade quick_* metadata with the model. Best-effort: never raises.

        Returns a terminal outcome for bounded retry drivers: "done" when
        metadata persisted, "deferred" when foreground inference owned the
        model (retryable), "skipped" for cancel/disabled/superseded rows.
        """
        if self.intelligence is None:
            return "skipped"
        if cancel_event.is_set():
            return "skipped"
        try:
            intelligence = await self.intelligence.analyze(
                document,
                chunks,
                cancel_event=cancel_event,
            )
        except BackgroundInferenceDeferred as exc:
            logger.warning(
                "Intelligence enrich deferred job=%s (%s); bounded retry applies",
                job.job_id,
                type(exc).__name__,
            )
            return "deferred"
        except Exception as exc:
            logger.warning(
                "Intelligence enrich deferred job=%s (%s); backfill covers pending rows",
                job.job_id,
                type(exc).__name__,
            )
            return "deferred"
        if cancel_event.is_set():
            return "skipped"
        try:
            updated = await asyncio.to_thread(
                self._enrich_sync,
                job,
                intelligence,
            )
        except Exception as exc:
            logger.warning(
                "Intelligence enrich update failed job=%s: %s",
                job.job_id,
                type(exc).__name__,
            )
            return "deferred"
        if not updated:
            logger.info(
                "Intelligence enrich skipped job=%s (revision superseded or revoked)",
                job.job_id,
            )
            return "skipped"
        return "done"

    def _enrich_sync(
        self,
        job: IngestionJob,
        intelligence: DocumentIntelligence,
    ) -> bool:
        with self.connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE files
                SET quick_summary = %s,
                    quick_tags = %s,
                    doc_type = %s,
                    intelligence_status = %s
                WHERE id = %s
                  AND user_id = %s
                  AND current_revision = %s
                  AND is_deleted = FALSE
                  AND user_granted_ai_access = TRUE
                """,
                (
                    intelligence.summary,
                    list(intelligence.tags),
                    intelligence.doc_type,
                    intelligence.status,
                    job.file_id,
                    job.user_id,
                    job.revision,
                ),
            )
            updated = cursor.rowcount == 1
            connection.commit()
            return updated

    def _publish_sync(
        self,
        job: IngestionJob,
        document: CanonicalDocument,
        chunks: tuple[CanonicalChunk, ...],
        prepared: PreparedEmbeddings,
        intelligence: DocumentIntelligence,
    ) -> None:
        try:
            with self.connection_factory() as connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute(
                        LOCK_PUBLICATION_FILE_SQL,
                        (job.file_id, job.user_id, job.revision),
                    )
                    if cursor.fetchone() is None:
                        raise IngestionCancelled("consent or revision fence rejected publication")
                    cursor.execute(
                        LOCK_PUBLICATION_JOB_SQL,
                        (
                            job.job_id,
                            job.file_id,
                            job.user_id,
                            job.revision,
                            job.lease_owner,
                        ),
                    )
                    if cursor.fetchone() is None:
                        raise IngestionCancelled(
                            "lease, consent, cancellation, or revision fence rejected publication"
                        )

                    metadata = self._revision_metadata(document, intelligence)
                    cursor.execute(
                        UPSERT_REVISION_SQL,
                        (
                            job.user_id,
                            job.file_id,
                            job.revision,
                            job.job_id,
                            document.source_sha256.lower(),
                            document.parser_fingerprint,
                            self.chunker_fingerprint,
                            prepared.fingerprint,
                            prepared.dimension,
                            self._source_size(job.metadata),
                            metadata,
                        ),
                    )
                    self._require_one(cursor.rowcount, "revision identity changed")

                    cursor.execute(
                        "DELETE FROM document_chunks WHERE file_id = %s AND revision = %s",
                        (job.file_id, job.revision),
                    )
                    rows = [
                        self._chunk_row(job, chunk, vector)
                        for chunk, vector in zip(chunks, prepared.vectors, strict=True)
                    ]
                    from app.db.embedding_store import resolve_vector_cast

                    insert_sql = INSERT_CHUNK_SQL.replace(
                        "::vector", f"::{resolve_vector_cast(connection, 'document_chunks')}"
                    )
                    cursor.executemany(insert_sql, rows)
                    if cursor.rowcount != len(rows):
                        raise PublicationError("not all chunks were written")

                    cursor.execute(
                        """
                        UPDATE document_revisions
                        SET status = 'superseded'
                        WHERE file_id = %s
                          AND revision <> %s
                          AND status = 'current'
                        """,
                        (job.file_id, job.revision),
                    )
                    cursor.execute(
                        """
                        UPDATE document_revisions
                        SET status = 'current', chunk_count = %s, completed_at = NOW()
                        WHERE file_id = %s
                          AND revision = %s
                          AND user_id = %s
                          AND job_id = %s
                          AND status = 'building'
                        """,
                        (
                            len(chunks),
                            job.file_id,
                            job.revision,
                            job.user_id,
                            job.job_id,
                        ),
                    )
                    self._require_one(cursor.rowcount, "revision could not become current")

                    cursor.execute(
                        """
                        UPDATE files
                        SET current_revision = %s,
                            ai_status = 'ready',
                            ai_error_code = NULL,
                            ai_error_detail = NULL,
                            quick_summary = %s,
                            quick_tags = %s,
                            doc_type = %s,
                            intelligence_status = 'pending'
                        WHERE id = %s
                          AND user_id = %s
                          AND desired_revision = %s
                          AND is_deleted = FALSE
                          AND user_granted_ai_access = TRUE
                        """,
                        (
                            job.revision,
                            intelligence.summary,
                            list(intelligence.tags),
                            intelligence.doc_type,
                            job.file_id,
                            job.user_id,
                            job.revision,
                        ),
                    )
                    self._require_one(cursor.rowcount, "file revision fence changed")

                    cursor.execute(
                        """
                        UPDATE ingestion_jobs
                        SET state = 'ready',
                            progress_current = %s,
                            progress_total = %s,
                            error_code = NULL,
                            error_detail = NULL,
                            lease_owner = NULL,
                            lease_expires_at = NULL,
                            finished_at = NOW(),
                            updated_at = NOW()
                        WHERE id = %s
                          AND file_id = %s
                          AND user_id = %s
                          AND revision = %s
                          AND state = 'publishing'
                          AND lease_owner = %s
                          AND lease_expires_at > NOW()
                          AND cancel_requested_at IS NULL
                        """,
                        (
                            len(chunks),
                            len(chunks),
                            job.job_id,
                            job.file_id,
                            job.user_id,
                            job.revision,
                            job.lease_owner,
                        ),
                    )
                    self._require_one(cursor.rowcount, "job publication fence changed")
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
        except IngestionCancelled:
            raise
        except PublicationError:
            raise
        except Exception as exc:
            raise PublicationError("atomic pgvector publication failed") from exc

    def _validate(
        self,
        document: CanonicalDocument,
        chunks: Sequence[CanonicalChunk],
        prepared: PreparedEmbeddings,
    ) -> None:
        if not chunks:
            raise PublicationError("cannot publish an empty revision")
        if not _SHA256.fullmatch(document.source_sha256.lower()):
            raise PublicationError("document source hash is invalid")
        if prepared.dimension != self.expected_dimension:
            raise EmbeddingDimensionError(f"expected embedding dimension {self.expected_dimension}")
        if len(prepared.vectors) != len(chunks):
            raise PublicationError("embedding count does not match chunk count")
        for vector in prepared.vectors:
            if len(vector) != prepared.dimension or not all(math.isfinite(float(value)) for value in vector):
                raise EmbeddingDimensionError("embedding vector is malformed")

    @staticmethod
    def _require_one(rowcount: int, message: str) -> None:
        if rowcount != 1:
            raise IngestionCancelled(message)

    @staticmethod
    def _source_size(metadata: Mapping[str, Any]) -> int | None:
        value = metadata.get("file_size_bytes")
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    @staticmethod
    def _revision_metadata(
        document: CanonicalDocument,
        intelligence: DocumentIntelligence,
    ) -> str:
        value = {
            "source_name": document.source_name,
            "media_type": document.media_type,
            "parser": dict(document.metadata),
            "intelligence": intelligence.metadata(),
        }
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _chunk_row(
        job: IngestionJob,
        chunk: CanonicalChunk,
        vector: Sequence[float],
    ) -> tuple[Any, ...]:
        provenance = json.dumps(
            [item.stable_dict() for item in chunk.provenance],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        metadata = json.dumps(
            {"chunk_type": chunk.chunk_type},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        vector_text = "[" + ",".join(format(float(value), ".17g") for value in vector) + "]"
        return (
            chunk.chunk_id,
            job.user_id,
            job.file_id,
            job.revision,
            chunk.ordinal,
            chunk.text,
            chunk.embedding_text,
            list(chunk.element_ids),
            provenance,
            list(chunk.section_path),
            chunk.token_count,
            metadata,
            vector_text,
        )
