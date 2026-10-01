"""Save-path isolation: a summarize-trigger failure never kills the item write.

Uses a fake connection factory (repository takes it as a constructor arg)
to prove, without a database:
1. upsert_candidate commits the item write on connection #1, then queues
   the summarize job on a separate post-commit connection #2.
2. When the summarize INSERT itself raises (e.g. a stale CHECK
   constraint on an unmigrated DB), the mutation is still returned and
   the item write stays committed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.graph_memory.models import MemoryCandidate
from app.graph_memory.repository import PostgresGraphMemoryRepository
from app.graph_memory.validation import validate_candidate

psycopg_errors: Any = pytest.importorskip("psycopg.errors")


def _tenant_row(generation: int = 0) -> dict[str, Any]:
    return {
        "user_id": 41,
        "tenant_uuid": uuid4(),
        "enabled": True,
        "retention_days": 90,
        "generation": generation,
        "revision": 2,
        "graph_revision": 7,
        "purge_state": "ready",
        "last_learned_at": None,
    }


def _item_row(fingerprint: str, chat_id: UUID, message_id: UUID) -> dict[str, Any]:
    now = datetime(2026, 7, 16, tzinfo=UTC)
    return {
        "id": uuid4(),
        "kind": "relationship",
        "subject": "I",
        "predicate": "has_pet",
        "object_value": "Bheem",
        "confidence": 0.95,
        "status": "active",
        "source_chat_id": chat_id,
        "source_message_id": message_id,
        "source_excerpt": "My dog Bheem is a Labrador.",
        "last_confirmed_at": now,
        "expires_at": now + timedelta(days=90),
        "revision": 1,
        "projection_state": "pending",
        "created_at": now,
        "updated_at": now,
    }


class FakeCursor:
    def __init__(self, fetchone_queue: list[Any], on_execute: Any | None = None) -> None:
        self.statements: list[str] = []
        self._queue = list(fetchone_queue)
        self._on_execute = on_execute

    def execute(self, sql: str, params: Any = None) -> None:
        self.statements.append(" ".join(str(sql).split()))
        if self._on_execute is not None:
            self._on_execute(str(sql))

    def fetchone(self) -> Any:
        assert self._queue, "fake cursor ran out of scripted rows"
        return self._queue.pop(0)

    def fetchall(self) -> list[Any]:
        return []


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is None:
            self.commit()
        else:
            self.rollback()


class FakeFactory:
    """get_db-shaped factory: each call opens a fresh connection."""

    def __init__(self, connections: list[FakeConnection]) -> None:
        self._connections = list(connections)
        self.calls = 0

    def __call__(self) -> FakeConnection:
        self.calls += 1
        assert self._connections, "factory ran out of scripted connections"
        return self._connections.pop(0)


def _candidate() -> MemoryCandidate:
    return MemoryCandidate(
        kind="relationship",
        subject="I",
        predicate="has_pet",
        object_value="Bheem",
        confidence=0.95,
        source_excerpt="My dog Bheem is a Labrador.",
        source_role="user",
        explicit_user_assertion=True,
    )


def _decision() -> Any:
    decision = validate_candidate(_candidate())
    assert decision.accepted and decision.status is not None and decision.fingerprint
    return decision


def test_upsert_commits_item_before_queuing_summarize() -> None:
    chat_id, message_id = uuid4(), uuid4()
    decision = _decision()
    tenant = _tenant_row()
    primary = FakeConnection(
        FakeCursor(
            [
                dict(tenant),  # _locked_tenant
                {"content": "My dog Bheem is a Labrador."},  # provenance
                _item_row(decision.fingerprint or "", chat_id, message_id),
                {"id": uuid4()},  # project job
                dict(tenant),  # tenant revision bump
            ]
        )
    )
    trigger = FakeConnection(FakeCursor([]))
    repo = PostgresGraphMemoryRepository(connection_factory=FakeFactory([primary, trigger]))

    mutation = repo.upsert_candidate(
        41,
        expected_generation=0,
        source_chat_id=chat_id,
        source_message_id=message_id,
        candidate=_candidate(),
        decision=decision,
    )

    assert mutation.item.object_value == "Bheem"
    # Primary write committed on its own connection, with no summarize
    # INSERT inside that transaction.
    assert primary.committed and not primary.rolled_back
    primary_sql = "\n".join(primary._cursor.statements)
    assert "INSERT INTO graph_memory_items" in primary_sql
    assert "summarize" not in primary_sql
    # Summarize trigger ran post-commit on a second connection.
    trigger_sql = "\n".join(trigger._cursor.statements)
    assert "'summarize'" in trigger_sql
    assert trigger.committed


def test_summarize_trigger_failure_keeps_the_saved_item() -> None:
    chat_id, message_id = uuid4(), uuid4()
    decision = _decision()
    tenant = _tenant_row()
    primary = FakeConnection(
        FakeCursor(
            [
                dict(tenant),
                {"content": "My dog Bheem is a Labrador."},
                _item_row(decision.fingerprint or "", chat_id, message_id),
                {"id": uuid4()},
                dict(tenant),
            ]
        )
    )

    def _boom(sql: str) -> None:
        if "summarize" in sql:
            raise psycopg_errors.CheckViolation("stale job_type check")

    trigger = FakeConnection(FakeCursor([], on_execute=_boom))
    repo = PostgresGraphMemoryRepository(connection_factory=FakeFactory([primary, trigger]))

    # Must not raise: the secondary trigger failure only logs.
    mutation = repo.upsert_candidate(
        41,
        expected_generation=0,
        source_chat_id=chat_id,
        source_message_id=message_id,
        candidate=_candidate(),
        decision=decision,
    )

    assert mutation.item.object_value == "Bheem"
    assert primary.committed and not primary.rolled_back
