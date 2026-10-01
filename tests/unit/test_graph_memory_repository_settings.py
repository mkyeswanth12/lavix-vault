from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.graph_memory.repository import PostgresGraphMemoryRepository


def tenant_row(*, enabled: bool, retention_days: int, revision: int) -> dict[str, Any]:
    return {
        "user_id": 41,
        "tenant_uuid": uuid4(),
        "enabled": enabled,
        "retention_days": retention_days,
        "generation": 3,
        "revision": revision,
        "graph_revision": 7,
        "purge_state": "ready",
        "last_learned_at": datetime(2026, 7, 16, tzinfo=UTC),
    }


class SettingsCursor:
    def __init__(self) -> None:
        self.current = tenant_row(enabled=False, retention_days=365, revision=4)
        self.executed: list[tuple[str, Any]] = []
        self._one: dict[str, Any] | None = None

    def execute(self, sql: str, params: Any = None) -> None:
        normalized = " ".join(sql.lower().split())
        self.executed.append((normalized, params))
        if normalized.startswith("select id from users"):
            self._one = {"id": 41}
        elif normalized.startswith("update users set memory_enabled"):
            self._one = None
        elif normalized.startswith("select user_id, tenant_uuid"):
            self._one = dict(self.current)
        elif normalized.startswith("update graph_memory_tenants set enabled"):
            enabled, retention_days, _user_id = params
            self.current = {
                **self.current,
                "enabled": enabled,
                "retention_days": retention_days,
                "revision": int(self.current["revision"]) + 1,
            }
            self._one = dict(self.current)
        else:
            raise AssertionError(f"Unexpected SQL: {normalized}")

    def fetchone(self) -> dict[str, Any] | None:
        return self._one

    def fetchall(self) -> list[dict[str, Any]]:
        return []


class SettingsConnection:
    def __init__(self) -> None:
        self.cursor_value = SettingsCursor()

    def cursor(self) -> SettingsCursor:
        return self.cursor_value


def test_compatibility_consent_write_preserves_retention_in_one_transaction() -> None:
    connection = SettingsConnection()

    @contextmanager
    def connection_factory():
        yield connection

    repository = PostgresGraphMemoryRepository(connection_factory=connection_factory)

    updated = repository.update_settings(
        41,
        enabled=True,
        retention_days=None,
        expected_revision=None,
    )

    assert updated.enabled is True
    assert updated.retention_days == 365
    assert updated.revision == 5
    assert connection.cursor_value.executed[1] == (
        "update users set memory_enabled = %s where id = %s",
        (True, 41),
    )
    assert not any(
        "update graph_memory_items" in sql for sql, _params in connection.cursor_value.executed
    )
    assert connection.cursor_value.executed[-1][1] == (True, 365, 41)


class RowlessEnqueueCursor:
    """First tenant read finds no row; users row exists; insert then re-read."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self._one: dict[str, Any] | None = None
        self._inserted = False

    def execute(self, sql: str, params: Any = None) -> None:
        normalized = " ".join(sql.lower().split())
        self.executed.append((normalized, params))
        if normalized.startswith("select user_id, tenant_uuid"):
            self._one = (
                tenant_row(enabled=False, retention_days=90, revision=1)
                if self._inserted
                else None
            )
        elif normalized.startswith("select id from users"):
            self._one = {"id": 41}
        elif normalized.startswith("insert into graph_memory_tenants"):
            self._inserted = True
            self._one = None
        else:
            raise AssertionError(f"Unexpected SQL: {normalized}")

    def fetchone(self) -> dict[str, Any] | None:
        return self._one

    def fetchall(self) -> list[dict[str, Any]]:
        return []


class RowlessEnqueueConnection:
    def __init__(self) -> None:
        self.cursor_value = RowlessEnqueueCursor()

    def cursor(self) -> RowlessEnqueueCursor:
        return self.cursor_value


def test_rowless_enqueue_self_heals_instead_of_throwing() -> None:
    """A valid user with no tenant row gets one (disabled) — no throw, no job."""
    from uuid import UUID

    connection = RowlessEnqueueConnection()

    @contextmanager
    def connection_factory():
        yield connection

    repository = PostgresGraphMemoryRepository(connection_factory=connection_factory)

    assert (
        repository.enqueue_extraction(
            41,
            UUID("08c51abc-6fff-44f0-970b-73b82870e22c"),
            UUID("b7a34f10-1708-42a8-b448-eb8ab4ae6bde"),
        )
        is None
    )
    assert any(
        "insert into graph_memory_tenants" in sql
        for sql, _params in connection.cursor_value.executed
    )
