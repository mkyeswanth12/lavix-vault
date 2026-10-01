from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.chat import memory_routes
from app.routers.chat.memory_routes import PostgresRuntimeMemoryReader


class ScriptCursor:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.executed: list[tuple[str, object]] = []
        self.current: dict[str, Any] = {}
        self.rowcount = -1

    def execute(self, sql: str, params: object = None) -> None:
        normalized = " ".join(sql.lower().split())
        self.executed.append((normalized, params))
        assert self.script, f"Unexpected SQL: {normalized}"
        self.current = self.script.pop(0)
        assert self.current["contains"] in normalized
        self.rowcount = int(self.current.get("rowcount", -1))

    def fetchone(self) -> dict[str, Any] | None:
        return self.current.get("one")

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self.current.get("many", []))


class ScriptConnection:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.cursor_value = ScriptCursor(script)

    def cursor(self) -> ScriptCursor:
        return self.cursor_value


def database_context(connection: ScriptConnection):
    @contextmanager
    def fake_get_db() -> Iterator[ScriptConnection]:
        yield connection

    return fake_get_db


class MemorySettingsService:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, Any]]] = []

    def update_settings(self, user_id: int, **values: Any) -> SimpleNamespace:
        self.calls.append((user_id, values))
        return SimpleNamespace(enabled=values["enabled"])


def memory_client(
    monkeypatch,
    connection: ScriptConnection,
    *,
    user_id: int = 41,
    settings_service: MemorySettingsService | None = None,
) -> TestClient:
    monkeypatch.setattr(memory_routes, "get_db", database_context(connection))
    app = FastAPI()
    app.include_router(memory_routes.router, prefix="/api/ai")
    app.dependency_overrides[memory_routes.require_ai_permission] = lambda: {
        "id": user_id,
        "perm_ai": True,
    }
    app.dependency_overrides[memory_routes.get_graph_memory_service] = lambda: (
        settings_service or MemorySettingsService()
    )
    return TestClient(app)


def test_memory_http_contract_and_default_state_are_explicit(monkeypatch) -> None:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    connection = ScriptConnection(
        [
            {"contains": "select memory_enabled from users", "one": {"memory_enabled": False}},
            {
                "contains": "from user_memories",
                "many": [
                    {
                        "id": 7,
                        "memory_text": "Use metric units.",
                        "created_at": now,
                        "updated_at": now,
                    }
                ],
            },
        ]
    )
    client = memory_client(monkeypatch, connection)

    response = client.get("/api/ai/memory")

    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "items": [
            {
                "id": 7,
                "text": "Use metric units.",
                "created_at": "2026-07-15T00:00:00Z",
                "updated_at": "2026-07-15T00:00:00Z",
            }
        ],
        "max_items": 20,
        "max_text_length": 500,
    }
    assert all(params == (41,) or params == (41, 20) for _, params in connection.cursor_value.executed)

    paths = client.app.openapi()["paths"]
    assert {"get", "put", "post", "delete"}.issubset(paths["/api/ai/memory"])
    assert "delete" in paths["/api/ai/memory/{memory_id}"]


def test_memory_enable_and_add_are_tenant_scoped_and_normalized(monkeypatch) -> None:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    enable_connection = ScriptConnection([])
    settings_service = MemorySettingsService()
    response = memory_client(
        monkeypatch,
        enable_connection,
        settings_service=settings_service,
    ).put("/api/ai/memory", json={"enabled": True})
    assert response.status_code == 200
    assert response.json() == {"enabled": True}
    assert enable_connection.cursor_value.executed == []
    assert settings_service.calls == [
        (
            41,
            {
                "enabled": True,
                "retention_days": None,
                "expected_revision": None,
            },
        )
    ]

    add_connection = ScriptConnection(
        [
            {
                "contains": "left join graph_memory_tenants",
                "one": {"memory_enabled": True, "graph_memory_enabled": True},
            },
            {"contains": "select count(*)", "one": {"count": 1}},
            {
                "contains": "insert into user_memories",
                "one": {
                    "id": 8,
                    "memory_text": "Prefer concise answers.",
                    "created_at": now,
                    "updated_at": now,
                },
            },
        ]
    )
    response = memory_client(monkeypatch, add_connection).post(
        "/api/ai/memory", json={"text": "  Prefer\n concise   answers.  "}
    )
    assert response.status_code == 201
    assert response.json()["item"]["text"] == "Prefer concise answers."
    assert add_connection.cursor_value.executed[-1][1] == (41, "Prefer concise answers.")


def test_memory_rejects_disabled_full_duplicate_and_invalid_items(monkeypatch) -> None:
    disabled = ScriptConnection(
        [
            {
                "contains": "left join graph_memory_tenants",
                "one": {"memory_enabled": False, "graph_memory_enabled": False},
            }
        ]
    )
    assert (
        memory_client(monkeypatch, disabled).post("/api/ai/memory", json={"text": "Remember me"}).status_code
        == 409
    )

    mismatched = ScriptConnection(
        [
            {
                "contains": "left join graph_memory_tenants",
                "one": {"memory_enabled": True, "graph_memory_enabled": False},
            }
        ]
    )
    assert (
        memory_client(monkeypatch, mismatched)
        .post("/api/ai/memory", json={"text": "Do not accept split consent"})
        .status_code
        == 409
    )

    full = ScriptConnection(
        [
            {
                "contains": "left join graph_memory_tenants",
                "one": {"memory_enabled": True, "graph_memory_enabled": True},
            },
            {"contains": "select count(*)", "one": {"count": 20}},
        ]
    )
    assert (
        memory_client(monkeypatch, full).post("/api/ai/memory", json={"text": "One too many"}).status_code
        == 409
    )

    duplicate = ScriptConnection(
        [
            {
                "contains": "left join graph_memory_tenants",
                "one": {"memory_enabled": True, "graph_memory_enabled": True},
            },
            {"contains": "select count(*)", "one": {"count": 3}},
            {"contains": "insert into user_memories", "one": None},
        ]
    )
    assert (
        memory_client(monkeypatch, duplicate).post("/api/ai/memory", json={"text": "Existing"}).status_code
        == 409
    )

    client = memory_client(monkeypatch, ScriptConnection([]))
    assert client.post("/api/ai/memory", json={"text": " \n "}).status_code == 422
    assert client.post("/api/ai/memory", json={"text": "x" * 501}).status_code == 422


def test_memory_delete_and_clear_never_cross_tenants(monkeypatch) -> None:
    delete_connection = ScriptConnection(
        [{"contains": "delete from user_memories where id = %s and user_id = %s", "rowcount": 1}]
    )
    response = memory_client(monkeypatch, delete_connection, user_id=77).delete("/api/ai/memory/9")
    assert response.json() == {"deleted_id": 9}
    assert delete_connection.cursor_value.executed[0][1] == (9, 77)

    missing_connection = ScriptConnection(
        [{"contains": "delete from user_memories where id = %s and user_id = %s", "rowcount": 0}]
    )
    assert (
        memory_client(monkeypatch, missing_connection, user_id=77).delete("/api/ai/memory/9").status_code
        == 404
    )

    clear_connection = ScriptConnection(
        [{"contains": "delete from user_memories where user_id = %s", "rowcount": 4}]
    )
    response = memory_client(monkeypatch, clear_connection, user_id=77).delete("/api/ai/memory")
    assert response.json() == {"deleted_count": 4}
    assert clear_connection.cursor_value.executed[0][1] == (77,)


def test_runtime_reader_bounds_enabled_items(monkeypatch) -> None:
    enabled = ScriptConnection(
        [
            {
                "contains": "from user_memories",
                "many": [
                    {"memory_text": "a" * 500},
                    {"memory_text": "b" * 500},
                    {"memory_text": "c" * 500},
                    {"memory_text": "d" * 500},
                ],
            },
        ]
    )
    monkeypatch.setattr(memory_routes, "get_db", database_context(enabled))
    result = PostgresRuntimeMemoryReader._for_runtime(41)
    assert [len(value) for value in result] == [500, 500, 500]
    assert enabled.cursor_value.executed[0][1] == (41, 6)
    runtime_sql = enabled.cursor_value.executed[0][0]
    assert "join graph_memory_tenants as tenant" in runtime_sql
    assert "u.memory_enabled = true" in runtime_sql
    assert "tenant.enabled = true" in runtime_sql
