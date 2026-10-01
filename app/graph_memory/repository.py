"""PostgreSQL authority for relationship-memory lifecycle and provenance."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from app.database import get_db

from .models import (
    DEFAULT_RETENTION_DAYS,
    PENDING_RETENTION_DAYS,
    GraphMemoryItem,
    MemoryCandidate,
    MemoryStatus,
)
from .validation import CandidateDecision

logger = logging.getLogger(__name__)


#: Every job_type string this module can INSERT into graph_memory_jobs.
#: Single source of truth for the contract test; must stay a subset of
#: the graph_memory_jobs_type_check constraint (see migration 0022).
JOB_TYPES = frozenset({"extract", "project", "expire", "purge", "summarize"})


class GraphMemoryConflictError(RuntimeError):
    """An optimistic revision or generation no longer matches."""


class GraphMemoryJobLeaseLost(GraphMemoryConflictError):
    """A worker no longer owns the extraction job that authorizes a write."""


class GraphMemoryDisabledError(RuntimeError):
    """Learning or recall was requested without current user consent."""


class GraphMemoryNotFoundError(RuntimeError):
    """A tenant-owned record was not found."""


class GraphMemoryValidationError(RuntimeError):
    """A mutation violates a lifecycle or provenance invariant."""


@dataclass(frozen=True, slots=True)
class TenantRecord:
    user_id: int
    tenant_uuid: UUID
    enabled: bool
    retention_days: int
    generation: int
    revision: int
    graph_revision: int
    purge_state: str
    last_learned_at: datetime | None


@dataclass(frozen=True, slots=True)
class ItemPageRecord:
    items: tuple[GraphMemoryItem, ...]
    total: int


@dataclass(frozen=True, slots=True)
class ItemMutation:
    tenant: TenantRecord
    item: GraphMemoryItem
    job_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class ClearResult:
    tenant: TenantRecord
    graph_deleted_count: int
    saved_preferences_deleted_count: int
    purge_job_id: UUID | None


@dataclass(frozen=True, slots=True)
class ExpiredItem:
    user_id: int
    memory_id: UUID
    fingerprint: str = ""
    tenant_uuid: UUID | None = None


@dataclass(frozen=True, slots=True)
class ProjectionSnapshotRow:
    memory_id: UUID
    subject: str
    predicate: str
    object_value: str
    projection_error: str | None = None


ConnectionFactory = Callable[[], AbstractContextManager[Any]]


def _tenant_from_row(row: dict[str, Any]) -> TenantRecord:
    return TenantRecord(
        user_id=int(row["user_id"]),
        tenant_uuid=UUID(str(row["tenant_uuid"])),
        enabled=bool(row["enabled"]),
        retention_days=int(row["retention_days"]),
        generation=int(row["generation"]),
        revision=int(row["revision"]),
        graph_revision=int(row["graph_revision"]),
        purge_state=str(row["purge_state"]),
        last_learned_at=row.get("last_learned_at"),
    )


def _item_from_row(row: dict[str, Any]) -> GraphMemoryItem:
    return GraphMemoryItem.model_validate(
        {
            "id": row["id"],
            "kind": row["kind"],
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object_value": row["object_value"],
            "confidence": float(row["confidence"]),
            "status": row["status"],
            "source_chat_id": row.get("source_chat_id"),
            "source_message_id": row.get("source_message_id"),
            "source_excerpt": row["source_excerpt"],
            "source_created_at": row.get("source_created_at"),
            "last_confirmed_at": row["last_confirmed_at"],
            "expires_at": row["expires_at"],
            "revision": row["revision"],
            "projection_state": row["projection_state"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    )


_TENANT_COLUMNS = """
user_id, tenant_uuid, enabled, retention_days, generation, revision,
graph_revision, purge_state, last_learned_at
"""

_ITEM_COLUMNS = """
i.id, i.kind, i.subject, i.predicate, i.object_value, i.confidence,
i.status, i.source_chat_id, i.source_message_id, i.source_excerpt,
i.last_confirmed_at, i.expires_at, i.revision, i.projection_state,
i.created_at, i.updated_at
"""


class PostgresGraphMemoryRepository:
    def __init__(self, connection_factory: ConnectionFactory = get_db) -> None:
        self._connection_factory = connection_factory

    def get_tenant(self, user_id: int) -> TenantRecord | None:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"SELECT {_TENANT_COLUMNS} FROM graph_memory_tenants WHERE user_id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
        return _tenant_from_row(row) if row else None

    def update_settings(
        self,
        user_id: int,
        *,
        enabled: bool,
        retention_days: int | None,
        expected_revision: int | None,
    ) -> TenantRecord:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (user_id,))
            if not cursor.fetchone():
                raise GraphMemoryNotFoundError("user_not_found")
            # Safe Automatic memory and explicit Saved Preferences share one
            # user-facing consent switch. Keep the legacy preference flag in
            # the same transaction as the revisioned graph setting so the UI
            # can never leave them half-enabled.
            cursor.execute(
                "UPDATE users SET memory_enabled = %s WHERE id = %s",
                (enabled, user_id),
            )
            cursor.execute(
                f"SELECT {_TENANT_COLUMNS} FROM graph_memory_tenants WHERE user_id = %s FOR UPDATE",
                (user_id,),
            )
            current_row = cursor.fetchone()
            if current_row is None:
                if expected_revision not in (None, 0):
                    raise GraphMemoryConflictError("settings_revision_conflict")
                effective_retention_days = (
                    DEFAULT_RETENTION_DAYS if retention_days is None else retention_days
                )
                cursor.execute(
                    f"""
                    INSERT INTO graph_memory_tenants (user_id, enabled, retention_days, revision)
                    VALUES (%s, %s, %s, 1)
                    RETURNING {_TENANT_COLUMNS}
                    """,
                    (user_id, enabled, effective_retention_days),
                )
                return _tenant_from_row(cursor.fetchone())

            current = _tenant_from_row(current_row)
            if expected_revision is not None and current.revision != expected_revision:
                raise GraphMemoryConflictError("settings_revision_conflict")

            effective_retention_days = (
                current.retention_days if retention_days is None else retention_days
            )

            if effective_retention_days != current.retention_days:
                # Pending review always has its independent 14-day lifetime.
                # The WHERE clause prevents longer retention from resurrecting
                # an item which has already expired.
                cursor.execute(
                    """
                    UPDATE graph_memory_items
                    SET expires_at = last_confirmed_at + make_interval(days => %s),
                        projection_state = 'pending',
                        projection_error = NULL,
                        updated_at = NOW()
                    WHERE user_id = %s
                      AND status = 'active'
                      AND expires_at > NOW()
                    RETURNING id
                    """,
                    (effective_retention_days, user_id),
                )
                for item_row in cursor.fetchall():
                    self._record_item_projection(
                        cursor,
                        current,
                        UUID(str(item_row["id"])),
                        "upsert",
                    )

            cursor.execute(
                f"""
                UPDATE graph_memory_tenants
                SET enabled = %s,
                    retention_days = %s,
                    revision = revision + 1,
                    updated_at = NOW()
                WHERE user_id = %s
                RETURNING {_TENANT_COLUMNS}
                """,
                (enabled, effective_retention_days, user_id),
            )
            return _tenant_from_row(cursor.fetchone())

    def aggregate(self, user_id: int) -> dict[str, Any]:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE status = 'active' AND expires_at > NOW()
                    ) AS active_count,
                    COUNT(*) FILTER (
                        WHERE status = 'pending' AND expires_at > NOW()
                    ) AS pending_count,
                    COUNT(*) FILTER (WHERE expires_at <= NOW()) AS expired_count,
                    MIN(expires_at) FILTER (
                        WHERE status IN ('active', 'pending') AND expires_at > NOW()
                    ) AS next_expiry_at
                FROM graph_memory_items
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cursor.fetchone()
        return {
            "active": int(row["active_count"] or 0),
            "pending": int(row["pending_count"] or 0),
            "expired": int(row["expired_count"] or 0),
            "next_expiry_at": row["next_expiry_at"],
        }

    def list_items(
        self,
        user_id: int,
        *,
        status: MemoryStatus | None,
        kind: str | None,
        limit: int,
        offset: int,
    ) -> ItemPageRecord:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT {_ITEM_COLUMNS}, source.created_at AS source_created_at,
                       COUNT(*) OVER() AS total_count
                FROM graph_memory_items AS i
                LEFT JOIN chat_messages AS source ON source.id = i.source_message_id
                WHERE i.user_id = %s
                  AND i.expires_at > NOW()
                  AND (%s::text IS NULL OR i.status = %s::text)
                  AND (%s::text IS NULL OR i.kind = %s::text)
                ORDER BY i.updated_at DESC, i.id DESC
                LIMIT %s OFFSET %s
                """,
                (
                    user_id,
                    status.value if status else None,
                    status.value if status else None,
                    kind,
                    kind,
                    limit,
                    offset,
                ),
            )
            rows = cursor.fetchall()
        return ItemPageRecord(
            items=tuple(_item_from_row(row) for row in rows),
            total=int(rows[0]["total_count"]) if rows else 0,
        )

    def active_items_for_profile(self, user_id: int, *, limit: int = 100) -> tuple[GraphMemoryItem, ...]:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT {_ITEM_COLUMNS}, source.created_at AS source_created_at
                FROM graph_memory_items AS i
                LEFT JOIN chat_messages AS source ON source.id = i.source_message_id
                WHERE i.user_id = %s
                  AND i.status = 'active'
                  AND i.expires_at > NOW()
                ORDER BY i.last_confirmed_at DESC, i.id DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            return tuple(_item_from_row(row) for row in cursor.fetchall())

    def authorized_recall_items(
        self,
        user_id: int,
        memory_ids: Sequence[UUID],
    ) -> tuple[GraphMemoryItem, ...]:
        if not memory_ids:
            return ()
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT {_ITEM_COLUMNS}, source.created_at AS source_created_at
                FROM graph_memory_items AS i
                JOIN graph_memory_tenants AS tenant ON tenant.user_id = i.user_id
                LEFT JOIN chat_messages AS source ON source.id = i.source_message_id
                WHERE i.user_id = %s
                  AND i.id = ANY(%s::uuid[])
                  AND i.status = 'active'
                  AND i.expires_at > NOW()
                  AND i.generation = tenant.generation
                  AND tenant.enabled = TRUE
                  AND tenant.purge_state = 'ready'
                """,
                (user_id, [str(memory_id) for memory_id in memory_ids]),
            )
            by_id = {
                item.id: item for item in (_item_from_row(row) for row in cursor.fetchall())
            }
        return tuple(by_id[memory_id] for memory_id in memory_ids if memory_id in by_id)

    def expire_due(self, *, limit: int = 500) -> tuple[ExpiredItem, ...]:
        """Delete due canonical rows and report what expired."""

        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                WITH due AS (
                    SELECT id
                    FROM graph_memory_items
                    WHERE expires_at <= NOW()
                    ORDER BY expires_at, id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                DELETE FROM graph_memory_items AS item
                USING due
                WHERE item.id = due.id
                RETURNING item.user_id, item.id, item.normalized_fingerprint,
                          item.tenant_uuid
                """,
                (max(1, min(limit, 2_000)),),
            )
            deleted_rows = cursor.fetchall()
            expired: list[ExpiredItem] = []
            affected_users: set[int] = set()
            summarize_targets: list[tuple[int, UUID, int]] = []
            for row in deleted_rows:
                user_id = int(row["user_id"])
                memory_id = UUID(str(row["id"]))
                expired.append(
                    ExpiredItem(
                        user_id=user_id,
                        memory_id=memory_id,
                        fingerprint=str(row.get("normalized_fingerprint") or ""),
                        tenant_uuid=UUID(str(row["tenant_uuid"]))
                        if row.get("tenant_uuid") is not None
                        else None,
                    )
                )
                affected_users.add(user_id)

            if affected_users:
                cursor.execute(
                    """
                    UPDATE graph_memory_tenants
                    SET graph_revision = graph_revision + 1,
                        updated_at = NOW()
                    WHERE user_id = ANY(%s::int[])
                    RETURNING user_id, tenant_uuid, generation
                    """,
                    (list(affected_users),),
                )
                summarize_targets = [
                    (
                        int(_row["user_id"]),
                        UUID(str(_row["tenant_uuid"])),
                        int(_row["generation"]),
                    )
                    for _row in cursor.fetchall()
                ]
            connection.commit()
            # Post-commit trigger: the revision bump is durable even if the
            # summarize enqueue fails (best-effort, never raises).
            for _uid, _tuuid, _gen in summarize_targets:
                self.enqueue_summarize_best_effort(_uid, _tuuid, _gen)
            return tuple(expired)

    def delete_items_by_tenant(self, user_id: int, generation: int) -> None:
        """Delete all graph_memory_items for a given user and generation."""
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                DELETE FROM graph_memory_items
                WHERE user_id = %s AND generation = %s
                """,
                (user_id, generation),
            )

    def get_item(self, user_id: int, memory_id: UUID) -> ItemMutation:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT {_ITEM_COLUMNS}, source.created_at AS source_created_at
                FROM graph_memory_items AS i
                LEFT JOIN chat_messages AS source ON source.id = i.source_message_id
                WHERE i.user_id = %s AND i.id = %s AND i.expires_at > NOW()
                """,
                (user_id, memory_id),
            )
            item_row = cursor.fetchone()
            if not item_row:
                raise GraphMemoryNotFoundError("memory_item_not_found")
            cursor.execute(
                f"SELECT {_TENANT_COLUMNS} FROM graph_memory_tenants WHERE user_id = %s",
                (user_id,),
            )
            tenant_row = cursor.fetchone()
        return ItemMutation(_tenant_from_row(tenant_row), _item_from_row(item_row))

    def get_projectable_item(self, user_id: int, memory_id: UUID) -> Any | None:
        """Fetch the projection boundary row (fingerprint included).

        Returns None when the row is missing or expired: the projection
        job is then trivially complete. Never raises for absent rows.
        """
        from .neo4j_projection import ProjectableItem

        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT tenant_uuid, kind, subject, predicate, object_value,
                       status, normalized_fingerprint
                FROM graph_memory_items
                WHERE user_id = %s AND id = %s AND expires_at > NOW()
                """,
                (user_id, memory_id),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return ProjectableItem(
            user_id=user_id,
            tenant_uuid=str(row["tenant_uuid"]),
            fingerprint=str(row["normalized_fingerprint"] or ""),
            kind=str(row["kind"]),
            subject=str(row["subject"] or ""),
            predicate=str(row["predicate"] or ""),
            object_value=str(row["object_value"] or ""),
            status=str(row["status"] or ""),
        )

    def fetch_active_by_fingerprints(        self, user_id: int, fingerprints: Sequence[str]
    ) -> tuple[GraphMemoryItem, ...]:
        """Resolve graph-suggested fingerprints to live active rows.

        PostgreSQL stays authoritative: unknown, expired, or non-active
        fingerprints resolve to nothing.
        """
        wanted = [str(fp) for fp in (fingerprints or []) if str(fp or "").strip()]
        if not wanted:
            return ()
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT {_ITEM_COLUMNS}
                FROM graph_memory_items AS i
                WHERE i.user_id = %s
                  AND i.status = 'active'
                  AND i.expires_at > NOW()
                  AND i.normalized_fingerprint = ANY(%s)
                LIMIT 25
                """,
                (user_id, wanted),
            )
            rows = cursor.fetchall() or []
        return tuple(_item_from_row(row) for row in rows)

    def update_embedding(
        self, user_id: int, memory_id: UUID, vector: Sequence[float]
    ) -> bool:
        """Store an embedding shaped for the configured dimensions."""
        from app.config import settings
        from app.db.embedding_store import resolve_vector_cast

        values = [float(value) for value in (vector or [])]
        if len(values) != settings.embedding_dimension:
            return False
        literal = "[" + ",".join(repr(value) for value in values) + "]"
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cast = resolve_vector_cast(connection, "graph_memory_items")
            cursor.execute(
                f"""
                UPDATE graph_memory_items
                SET embedding = %s::{cast}, updated_at = NOW()
                WHERE id = %s AND user_id = %s
                """,
                (literal, memory_id, user_id),
            )
            touched = (cursor.rowcount or 0) == 1
            connection.commit()
            return touched

    def search_similar(
        self,
        user_id: int,
        vector: Sequence[float],
        *,
        limit: int = 4,
        exclude_ids: Sequence[UUID] = (),
    ) -> tuple[GraphMemoryItem, ...]:
        """Cosine-nearest live active rows; PostgreSQL stays authoritative."""
        from app.config import settings
        from app.db.embedding_store import resolve_vector_cast

        values = [float(value) for value in (vector or [])]
        if len(values) != settings.embedding_dimension:
            return ()
        literal = "[" + ",".join(repr(value) for value in values) + "]"
        excluded = [str(item) for item in (exclude_ids or [])]
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cast = resolve_vector_cast(connection, "graph_memory_items")
            cursor.execute(
                f"""
                SELECT {_ITEM_COLUMNS}
                FROM graph_memory_items AS i
                WHERE i.user_id = %s
                  AND i.status = 'active'
                  AND i.expires_at > NOW()
                  AND i.embedding IS NOT NULL
                  AND NOT (i.id = ANY(%s::uuid[]))
                ORDER BY i.embedding <=> %s::{cast}
                LIMIT %s
                """,
                (user_id, excluded, literal, max(1, min(int(limit or 1), 25))),
            )
            rows = cursor.fetchall() or []
        return tuple(_item_from_row(row) for row in rows)

    def active_projection_snapshot(
        self, user_id: int
    ) -> dict[str, ProjectionSnapshotRow]:
        """Active fingerprints with content for drift comparison.

        Maps normalized_fingerprint -> ProjectionSnapshotRow.
        """
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT id, normalized_fingerprint, subject, predicate,
                       object_value, projection_error
                FROM graph_memory_items
                WHERE user_id = %s
                  AND status = 'active'
                  AND expires_at > NOW()
                """,
                (user_id,),
            )
            rows = cursor.fetchall() or []
        snapshot: dict[str, ProjectionSnapshotRow] = {}
        for row in rows:
            snapshot[str(row["normalized_fingerprint"])] = ProjectionSnapshotRow(
                memory_id=UUID(str(row["id"])),
                subject=str(row["subject"] or ""),
                predicate=str(row["predicate"] or ""),
                object_value=str(row["object_value"] or ""),
                projection_error=str(row.get("projection_error") or "") or None,
            )
        return snapshot

    def mark_item_projection_skipped(
        self, user_id: int, generation: int, memory_id: UUID, code: str
    ) -> None:
        """Deterministic projection skip: failed state, no retry loop."""
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE graph_memory_items
                SET projection_state = 'failed', projection_error = %s, updated_at = NOW()
                WHERE id = %s AND user_id = %s AND generation = %s
                """,
                (code[:80], memory_id, user_id, generation),
            )
            connection.commit()

    def queue_projection_job(
        self, user_id: int, memory_id: UUID, operation: str = "upsert"
    ) -> UUID | None:
        """Queue a project job for drift healing; marks the row stale."""
        if operation not in ("upsert", "delete"):
            raise ValueError("invalid_projection_operation")
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            tenant = self._locked_tenant(cursor, user_id)
            cursor.execute(
                """
                UPDATE graph_memory_items
                SET projection_state = 'stale', projection_error = NULL, updated_at = NOW()
                WHERE id = %s AND user_id = %s AND generation = %s
                  AND status = 'active'
                """,
                (memory_id, user_id, tenant.generation),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            job_id = self._record_item_projection(cursor, tenant, memory_id, operation)
            connection.commit()
            return job_id

    def edit_item(
        self,
        user_id: int,
        memory_id: UUID,
        *,
        subject: str,
        predicate: str,
        object_value: str,
        fingerprint: str,
        expected_revision: int,
    ) -> ItemMutation:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            tenant = self._locked_tenant(cursor, user_id)
            cursor.execute(
                """
                SELECT 1
                FROM graph_memory_items
                WHERE user_id = %s
                  AND normalized_fingerprint = %s
                  AND id <> %s
                  AND status IN ('pending', 'active')
                """,
                (user_id, fingerprint, memory_id),
            )
            if cursor.fetchone():
                raise GraphMemoryValidationError("memory_item_already_exists")
            cursor.execute(
                f"""
                UPDATE graph_memory_items AS i
                SET subject = %s,
                    predicate = %s,
                    object_value = %s,
                    normalized_fingerprint = %s,
                    confidence = 1,
                    status = 'active',
                    approved_at = COALESCE(approved_at, NOW()),
                    last_confirmed_at = NOW(),
                    expires_at = NOW() + make_interval(days => %s),
                    revision = revision + 1,
                    projection_state = 'pending',
                    projection_error = NULL,
                    updated_at = NOW()
                WHERE i.user_id = %s
                  AND i.id = %s
                  AND i.revision = %s
                  AND i.expires_at > NOW()
                RETURNING {_ITEM_COLUMNS}
                """,
                (
                    subject,
                    predicate,
                    object_value,
                    fingerprint,
                    tenant.retention_days,
                    user_id,
                    memory_id,
                    expected_revision,
                ),
            )
            row = cursor.fetchone()
            if not row:
                self._raise_item_mutation_error(cursor, user_id, memory_id, expected_revision)
            job_id = self._record_item_projection(cursor, tenant, memory_id, "upsert")
            tenant = self._bump_graph_revision(cursor, user_id)
            summarize_target = (tenant.user_id, tenant.tenant_uuid, tenant.generation)
            mutation = ItemMutation(tenant, _item_from_row(row), job_id)
        self.enqueue_summarize_best_effort(*summarize_target)
        return mutation

    def approve_item(self, user_id: int, memory_id: UUID) -> ItemMutation:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            tenant = self._locked_tenant(cursor, user_id)
            cursor.execute(
                f"""
                UPDATE graph_memory_items AS i
                SET status = 'active',
                    confidence = GREATEST(confidence, 0.88),
                    approved_at = NOW(),
                    last_confirmed_at = NOW(),
                    expires_at = NOW() + make_interval(days => %s),
                    revision = revision + 1,
                    projection_state = 'pending',
                    projection_error = NULL,
                    updated_at = NOW()
                WHERE i.user_id = %s
                  AND i.id = %s
                  AND i.status = 'pending'
                  AND i.expires_at > NOW()
                RETURNING {_ITEM_COLUMNS}
                """,
                (tenant.retention_days, user_id, memory_id),
            )
            row = cursor.fetchone()
            if not row:
                raise GraphMemoryNotFoundError("pending_memory_item_not_found")
            job_id = self._record_item_projection(cursor, tenant, memory_id, "upsert")
            tenant = self._bump_graph_revision(cursor, user_id)
            summarize_target = (tenant.user_id, tenant.tenant_uuid, tenant.generation)
            mutation = ItemMutation(tenant, _item_from_row(row), job_id)
        self.enqueue_summarize_best_effort(*summarize_target)
        return mutation

    def renew_item(self, user_id: int, memory_id: UUID) -> ItemMutation:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            tenant = self._locked_tenant(cursor, user_id)
            cursor.execute(
                f"""
                UPDATE graph_memory_items AS i
                SET last_confirmed_at = NOW(),
                    expires_at = NOW() + make_interval(days => %s),
                    revision = revision + 1,
                    projection_state = 'pending',
                    projection_error = NULL,
                    updated_at = NOW()
                WHERE i.user_id = %s
                  AND i.id = %s
                  AND i.status = 'active'
                  AND i.expires_at > NOW()
                RETURNING {_ITEM_COLUMNS}
                """,
                (tenant.retention_days, user_id, memory_id),
            )
            row = cursor.fetchone()
            if not row:
                raise GraphMemoryNotFoundError("active_memory_item_not_found")
            job_id = self._record_item_projection(cursor, tenant, memory_id, "upsert")
            tenant = self._bump_graph_revision(cursor, user_id)
            summarize_target = (tenant.user_id, tenant.tenant_uuid, tenant.generation)
            mutation = ItemMutation(tenant, _item_from_row(row), job_id)
        self.enqueue_summarize_best_effort(*summarize_target)
        return mutation

    def expire_superseded(
        self,
        user_id: int,
        *,
        kind: str,
        subject: str,
        predicate: str,
        keep_fingerprint: str,
    ) -> int:
        """Expire live rows sharing an identity key, except the keeper.

        Corrections ("now Sam" vs "mk") match on (kind, subject,
        predicate) with normalized comparison; the newly inserted row
        (keep_fingerprint) survives. Only live rows (pending/active,
        unexpired) are touched. Returns the expired count.
        """
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE graph_memory_items
                SET expires_at = NOW(), updated_at = NOW()
                WHERE user_id = %s
                  AND LOWER(kind) = LOWER(%s)
                  AND LOWER(TRIM(subject)) = LOWER(TRIM(%s))
                  AND LOWER(predicate) = LOWER(%s)
                  AND normalized_fingerprint <> %s
                  AND status IN ('pending', 'active')
                  AND expires_at > NOW()
                RETURNING id
                """,
                (user_id, kind, subject, predicate, keep_fingerprint),
            )
            count = len(cursor.fetchall())
            connection.commit()
            return count

    def delete_item(self, user_id: int, memory_id: UUID) -> ItemMutation:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            tenant = self._locked_tenant(cursor, user_id)
            cursor.execute(
                """
                SELECT normalized_fingerprint
                FROM graph_memory_items
                WHERE user_id = %s AND id = %s
                """,
                (user_id, memory_id),
            )
            row = cursor.fetchone()
            fingerprint = str((row or {}).get("normalized_fingerprint") or "")
            cursor.execute(
                f"""
                DELETE FROM graph_memory_items AS i
                WHERE i.user_id = %s AND i.id = %s
                RETURNING {_ITEM_COLUMNS}
                """,
                (user_id, memory_id),
            )
            row = cursor.fetchone()
            if not row:
                raise GraphMemoryNotFoundError("memory_item_not_found")
            job_id = self._record_item_projection(
                cursor, tenant, memory_id, "delete", fingerprint=fingerprint or None
            )
            cursor.execute(
                f"""
                UPDATE graph_memory_tenants
                SET graph_revision = graph_revision + 1,
                    purge_state = 'pending',
                    purge_requested_at = NOW(),
                    updated_at = NOW()
                WHERE user_id = %s
                RETURNING {_TENANT_COLUMNS}
                """,
                (user_id,),
            )
            _t_row = cursor.fetchone()
            _t_tenant = _tenant_from_row(_t_row)
            summarize_target = (_t_tenant.user_id, _t_tenant.tenant_uuid, _t_tenant.generation)
            mutation = ItemMutation(_t_tenant, _item_from_row(row), job_id)
        self.enqueue_summarize_best_effort(*summarize_target)
        return mutation

    def extraction_job_exists(self, user_id: int, source_message_id: UUID) -> bool:
        """True when an extract job was already queued for this message.

        Used to distinguish a benign duplicate enqueue (silent) from a
        failure to file anything (honest notice). Read-only.
        """
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT 1 FROM graph_memory_jobs
                WHERE user_id = %s
                  AND job_type = 'extract'
                  AND source_message_id = %s
                LIMIT 1
                """,
                (user_id, source_message_id),
            )
            return cursor.fetchone() is not None

    def enqueue_extraction(self, user_id: int, chat_id: UUID, user_message_id: UUID) -> UUID | None:
        """Queue once only after a user message and later assistant reply exist."""

        with self._connection_factory() as connection:
            cursor = connection.cursor()
            # Self-healing (not strict): a valid user with no tenant row yet
            # gets one created with schema defaults (disabled) instead of a
            # throw. Still returns None while opted out — honest, storable,
            # and never a false promise downstream.
            tenant = self._ensure_locked_tenant(cursor, user_id)
            if not tenant.enabled or tenant.purge_state != "ready":
                return None
            cursor.execute(
                """
                SELECT source.id
                FROM chat_messages AS source
                WHERE source.id = %s
                  AND source.chat_id = %s
                  AND source.user_id = %s
                  AND source.role = 'user'
                  AND EXISTS (
                      SELECT 1
                      FROM chat_messages AS answer
                      WHERE answer.chat_id = source.chat_id
                        AND answer.user_id = source.user_id
                        AND answer.role = 'assistant'
                        AND answer.sequence_number > source.sequence_number
                  )
                """,
                (user_message_id, chat_id, user_id),
            )
            if not cursor.fetchone():
                raise GraphMemoryValidationError("saved_user_and_assistant_messages_required")
            cursor.execute(
                """
                INSERT INTO graph_memory_jobs (
                    user_id, tenant_uuid, generation, job_type,
                    source_chat_id, source_message_id
                )
                VALUES (%s, %s, %s, 'extract', %s, %s)
                ON CONFLICT (user_id, job_type, source_message_id)
                    WHERE job_type = 'extract' AND source_message_id IS NOT NULL
                DO NOTHING
                RETURNING id
                """,
                (user_id, tenant.tenant_uuid, tenant.generation, chat_id, user_message_id),
            )
            row = cursor.fetchone()
            return UUID(str(row["id"])) if row else None

    def upsert_candidate(
        self,
        user_id: int,
        *,
        expected_generation: int,
        source_chat_id: UUID,
        source_message_id: UUID,
        candidate: MemoryCandidate,
        decision: CandidateDecision,
        lease_job_id: UUID | None = None,
        lease_owner: str | None = None,
    ) -> ItemMutation:
        if not decision.accepted or decision.status is None or decision.fingerprint is None:
            raise GraphMemoryValidationError(decision.reason or "candidate_rejected")
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            if (lease_job_id is None) != (lease_owner is None):
                raise ValueError("candidate lease fence requires both job ID and owner")
            if lease_job_id is not None:
                cursor.execute(
                    """
                    SELECT id
                    FROM graph_memory_jobs
                    WHERE id = %s
                      AND user_id = %s
                      AND generation = %s
                      AND job_type = 'extract'
                      AND state = 'running'
                      AND lease_owner = %s
                      AND lease_expires_at > NOW()
                    FOR UPDATE
                    """,
                    (lease_job_id, user_id, expected_generation, lease_owner),
                )
                if not cursor.fetchone():
                    raise GraphMemoryJobLeaseLost("graph_job_lease_lost")
            tenant = self._locked_tenant(cursor, user_id)
            if not tenant.enabled:
                raise GraphMemoryDisabledError("graph_memory_disabled")
            if tenant.purge_state != "ready" or tenant.generation != expected_generation:
                raise GraphMemoryConflictError("memory_generation_changed")
            cursor.execute(
                """
                SELECT content
                FROM chat_messages
                WHERE id = %s AND chat_id = %s AND user_id = %s AND role = 'user'
                """,
                (source_message_id, source_chat_id, user_id),
            )
            source = cursor.fetchone()
            if not source:
                raise GraphMemoryValidationError("invalid_user_message_provenance")
            normalized_content = " ".join(str(source["content"]).casefold().split())
            normalized_excerpt = " ".join(candidate.source_excerpt.casefold().split())
            if normalized_excerpt not in normalized_content:
                raise GraphMemoryValidationError("source_excerpt_not_grounded")

            expiry_days = (
                tenant.retention_days
                if decision.status is MemoryStatus.ACTIVE
                else PENDING_RETENTION_DAYS
            )
            cursor.execute(
                f"""
                INSERT INTO graph_memory_items AS i (
                    user_id, tenant_uuid, generation, kind, subject, predicate,
                    object_value, normalized_fingerprint, confidence, status,
                    source_chat_id, source_message_id, source_excerpt,
                    expires_at, approved_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    NOW() + make_interval(days => %s),
                    CASE WHEN %s = 'active' THEN NOW() ELSE NULL END
                )
                ON CONFLICT (user_id, normalized_fingerprint)
                    WHERE status IN ('pending', 'active')
                DO UPDATE SET
                    confidence = GREATEST(i.confidence, EXCLUDED.confidence),
                    status = CASE
                        WHEN i.status = 'active' OR EXCLUDED.status = 'active'
                            THEN 'active'
                        ELSE 'pending'
                    END,
                    source_chat_id = EXCLUDED.source_chat_id,
                    source_message_id = EXCLUDED.source_message_id,
                    source_excerpt = EXCLUDED.source_excerpt,
                    last_confirmed_at = NOW(),
                    expires_at = NOW() + make_interval(days => CASE
                        WHEN i.status = 'active' OR EXCLUDED.status = 'active' THEN %s
                        ELSE %s
                    END),
                    approved_at = CASE
                        WHEN i.status = 'active' OR EXCLUDED.status = 'active'
                            THEN COALESCE(i.approved_at, NOW())
                        ELSE NULL
                    END,
                    revision = i.revision + 1,
                    projection_state = 'pending',
                    projection_error = NULL,
                    updated_at = NOW()
                RETURNING {_ITEM_COLUMNS}
                """,
                (
                    user_id,
                    tenant.tenant_uuid,
                    tenant.generation,
                    candidate.kind.value,
                    candidate.subject,
                    candidate.predicate,
                    candidate.object_value,
                    decision.fingerprint,
                    candidate.confidence,
                    decision.status.value,
                    source_chat_id,
                    source_message_id,
                    candidate.source_excerpt,
                    expiry_days,
                    decision.status.value,
                    tenant.retention_days,
                    PENDING_RETENTION_DAYS,
                ),
            )
            row = cursor.fetchone()
            memory_id = UUID(str(row["id"]))
            job_id = self._record_item_projection(cursor, tenant, memory_id, "upsert")
            cursor.execute(
                f"""
                UPDATE graph_memory_tenants
                SET graph_revision = graph_revision + 1,
                    last_learned_at = NOW(),
                    updated_at = NOW()
                WHERE user_id = %s
                  AND generation = %s
                  AND enabled = TRUE
                  AND purge_state = 'ready'
                RETURNING {_TENANT_COLUMNS}
                """,
                (user_id, expected_generation),
            )
            updated_tenant = cursor.fetchone()
            if not updated_tenant:
                raise GraphMemoryConflictError("memory_generation_changed")
            updated = _tenant_from_row(updated_tenant)
            summarize_target = (updated.user_id, updated.tenant_uuid, updated.generation)
            mutation = ItemMutation(updated, _item_from_row(row), job_id)
        # Post-commit trigger (1a): the item write and revision bump are
        # durable before the summarize job is queued. Best-effort (1b).
        self.enqueue_summarize_best_effort(*summarize_target)
        return mutation

    def clear(self, user_id: int, *, include_saved_preferences: bool) -> ClearResult:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            self._ensure_locked_tenant(cursor, user_id)
            cursor.execute(
                """
                UPDATE graph_memory_jobs
                SET state = 'cancelled',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    finished_at = NOW(),
                    updated_at = NOW()
                WHERE user_id = %s AND state IN ('queued', 'running')
                """,
                (user_id,),
            )
            cursor.execute("DELETE FROM graph_memory_items WHERE user_id = %s", (user_id,))
            graph_deleted = max(0, int(cursor.rowcount))
            saved_deleted = 0
            if include_saved_preferences:
                cursor.execute("DELETE FROM user_memories WHERE user_id = %s", (user_id,))
                saved_deleted = max(0, int(cursor.rowcount))

            cursor.execute(
                f"""
                UPDATE graph_memory_tenants
                SET generation = generation + 1,
                    graph_revision = graph_revision + 1,
                    revision = revision + 1,
                    purge_state = 'pending',
                    purge_requested_at = NOW(),
                    updated_at = NOW()
                WHERE user_id = %s
                RETURNING {_TENANT_COLUMNS}
                """,
                (user_id,),
            )
            updated = _tenant_from_row(cursor.fetchone())
            cursor.execute(
                """
                INSERT INTO graph_memory_jobs (
                    user_id, tenant_uuid, generation, job_type, payload
                )
                VALUES (%s, %s, %s, 'purge', %s::jsonb)
                RETURNING id
                """,
                (
                    user_id,
                    updated.tenant_uuid,
                    updated.generation,
                    json.dumps({"operation": "purge_tenant"}),
                ),
            )
            purge_job_id = UUID(str(cursor.fetchone()["id"]))
            summarize_target = (updated.user_id, updated.tenant_uuid, updated.generation)
            result = ClearResult(updated, graph_deleted, saved_deleted, purge_job_id)
        self.enqueue_summarize_best_effort(*summarize_target)
        return result

    def mark_purge_complete(self, user_id: int, generation: int, job_id: UUID | None) -> TenantRecord:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            if job_id is not None:
                cursor.execute(
                    """
                    UPDATE graph_memory_jobs
                    SET state = 'complete', finished_at = NOW(), updated_at = NOW()
                    WHERE id = %s AND user_id = %s AND generation = %s
                    """,
                    (job_id, user_id, generation),
                )
            cursor.execute(
                f"""
                UPDATE graph_memory_tenants
                SET purge_state = 'ready', purge_requested_at = NULL, updated_at = NOW()
                WHERE user_id = %s AND generation = %s
                RETURNING {_TENANT_COLUMNS}
                """,
                (user_id, generation),
            )
            row = cursor.fetchone()
            if not row:
                raise GraphMemoryConflictError("memory_generation_changed")
            return _tenant_from_row(row)

    def mark_item_projection_complete(
        self,
        user_id: int,
        generation: int,
        memory_id: UUID,
        job_id: UUID | None,
        *,
        deleted: bool,
        projected: bool = True,
    ) -> TenantRecord:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            if not deleted and projected:
                # 'projected' is written ONLY here, and only when the caller
                # passes projected=True after a real driver success (or a
                # deliberate PG-only completion). Driver failures propagate
                # before this point, so the state truly moves.
                cursor.execute(
                    """
                    UPDATE graph_memory_items
                    SET projection_state = 'projected', projection_error = NULL, updated_at = NOW()
                    WHERE id = %s AND user_id = %s AND generation = %s
                    """,
                    (memory_id, user_id, generation),
                )
            if job_id is not None:
                cursor.execute(
                    """
                    UPDATE graph_memory_jobs
                    SET state = 'complete', finished_at = NOW(), updated_at = NOW()
                    WHERE id = %s AND user_id = %s AND generation = %s
                    """,
                    (job_id, user_id, generation),
                )
            cursor.execute(
                f"""
                UPDATE graph_memory_tenants
                SET purge_state = CASE
                        WHEN NOT EXISTS (
                            SELECT 1 FROM graph_memory_jobs
                            WHERE user_id = %s
                              AND generation = %s
                              AND state IN ('queued', 'running', 'failed')
                              AND job_type IN ('project', 'purge')
                              AND payload->>'operation' IN ('delete', 'purge_tenant')
                        ) THEN 'ready'
                        ELSE purge_state
                    END,
                    purge_requested_at = CASE
                        WHEN NOT EXISTS (
                            SELECT 1 FROM graph_memory_jobs
                            WHERE user_id = %s
                              AND generation = %s
                              AND state IN ('queued', 'running', 'failed')
                              AND job_type IN ('project', 'purge')
                              AND payload->>'operation' IN ('delete', 'purge_tenant')
                        ) THEN NULL
                        ELSE purge_requested_at
                    END,
                    updated_at = NOW()
                WHERE user_id = %s AND generation = %s
                RETURNING {_TENANT_COLUMNS}
                """,
                (user_id, generation, user_id, generation, user_id, generation),
            )
            tenant_row = cursor.fetchone()
            if tenant_row is None:
                raise GraphMemoryConflictError("memory_generation_changed")
            return _tenant_from_row(tenant_row)

    def get_summary(self, user_id: int) -> dict[str, Any] | None:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT summary, source_memory_ids, graph_revision, generated_at, model, is_fallback
                FROM memory_summaries
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "summary": str(row["summary"]),
                "source_memory_ids": [UUID(str(uid)) for uid in row["source_memory_ids"]],
                "graph_revision": int(row["graph_revision"]),
                "generated_at": row["generated_at"],
                "model": str(row["model"]),
                "is_fallback": bool(row["is_fallback"]),
            }

    def store_summary(
        self,
        user_id: int,
        summary: str,
        source_memory_ids: list[UUID],
        graph_revision: int,
        model: str,
        is_fallback: bool,
    ) -> None:
        with self._connection_factory() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO memory_summaries (
                    user_id, summary, source_memory_ids, graph_revision, model, is_fallback, generated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (user_id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    source_memory_ids = EXCLUDED.source_memory_ids,
                    graph_revision = EXCLUDED.graph_revision,
                    model = EXCLUDED.model,
                    is_fallback = EXCLUDED.is_fallback,
                    generated_at = NOW()
                """,
                (
                    user_id,
                    summary,
                    [str(uid) for uid in source_memory_ids],
                    graph_revision,
                    model,
                    is_fallback,
                ),
            )
            connection.commit()

    @staticmethod
    def _enqueue_summarize(cursor: Any, user_id: int, tenant_uuid: UUID, generation: int) -> None:
        cursor.execute(
            """
            INSERT INTO graph_memory_jobs (
                user_id, tenant_uuid, generation, job_type
            )
            VALUES (%s, %s, %s, 'summarize')
            ON CONFLICT (user_id) WHERE job_type = 'summarize' AND state IN ('queued', 'running')
            DO NOTHING
            """,
            (user_id, tenant_uuid, generation),
        )

    def _ensure_locked_tenant(self, cursor: Any, user_id: int) -> TenantRecord:
        cursor.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (user_id,))
        if not cursor.fetchone():
            raise GraphMemoryNotFoundError("user_not_found")
        cursor.execute(
            """
            INSERT INTO graph_memory_tenants (user_id, retention_days)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO NOTHING
            """,
            (user_id, DEFAULT_RETENTION_DAYS),
        )
        return self._locked_tenant(cursor, user_id)

    def _locked_tenant(self, cursor: Any, user_id: int) -> TenantRecord:
        cursor.execute(
            f"SELECT {_TENANT_COLUMNS} FROM graph_memory_tenants WHERE user_id = %s FOR UPDATE",
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise GraphMemoryNotFoundError("graph_memory_not_found")
        return _tenant_from_row(row)

    def enqueue_summarize_best_effort(
        self, user_id: int, tenant_uuid: UUID, generation: int
    ) -> bool:
        """Queue a summarize job in its own transaction; never raises.

        The summarize job is a secondary trigger: the primary memory write
        and its graph_revision bump already committed before this runs. A
        failure here (stale constraint, lease race, DB blip) only logs a
        warning. Read-path revalidation (service.get_status) re-enqueues on
        the next read when the stored summary is stale or a fallback, so a
        skipped enqueue self-heals.
        """
        try:
            with self._connection_factory() as connection:
                cursor = connection.cursor()
                self._enqueue_summarize(cursor, user_id, tenant_uuid, generation)
            return True
        except Exception:
            logger.warning(
                "summarize enqueue failed open for user %s", user_id, exc_info=True
            )
            return False

    @staticmethod
    def _bump_graph_revision(cursor: Any, user_id: int) -> TenantRecord:
        """Bump graph_revision inside the caller's transaction (no enqueue).

        The post-commit summarize trigger is the caller's job via
        enqueue_summarize_best_effort, so a trigger failure can never roll
        back the primary write.
        """
        cursor.execute(
            f"""
            UPDATE graph_memory_tenants
            SET graph_revision = graph_revision + 1, updated_at = NOW()
            WHERE user_id = %s
            RETURNING {_TENANT_COLUMNS}
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        return _tenant_from_row(row)

    @staticmethod
    def _record_item_projection(
        cursor: Any,
        tenant: TenantRecord,
        memory_id: UUID,
        operation: str,
        *,
        fingerprint: str | None = None,
    ) -> UUID:
        payload: dict[str, Any] = {"operation": operation, "memory_id": str(memory_id)}
        if fingerprint:
            # Delete jobs carry the fingerprint: the canonical row is gone
            # before the worker runs, and Neo4j scoping needs it.
            payload["fingerprint"] = fingerprint
        cursor.execute(
            """
            INSERT INTO graph_memory_jobs (
                user_id, tenant_uuid, generation, job_type, payload
            )
            VALUES (%s, %s, %s, 'project', %s::jsonb)
            RETURNING id
            """,
            (
                tenant.user_id,
                tenant.tenant_uuid,
                tenant.generation,
                json.dumps(payload),
            ),
        )
        return UUID(str(cursor.fetchone()["id"]))

    @staticmethod
    def _raise_item_mutation_error(
        cursor: Any,
        user_id: int,
        memory_id: UUID,
        expected_revision: int,
    ) -> None:
        cursor.execute(
            "SELECT revision FROM graph_memory_items WHERE user_id = %s AND id = %s",
            (user_id, memory_id),
        )
        row = cursor.fetchone()
        if not row:
            raise GraphMemoryNotFoundError("memory_item_not_found")
        if int(row["revision"]) != expected_revision:
            raise GraphMemoryConflictError("memory_item_revision_conflict")
        raise GraphMemoryValidationError("memory_item_expired")
