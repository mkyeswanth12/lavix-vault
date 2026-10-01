from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.routers.chat import session_routes
from app.routers.chat.schemas import ChatPatchRequest

CHAT_ID = "efbd931f-8308-4eec-9208-f1778e372bdd"
USER = {"id": 17}


@dataclass
class Step:
    contains: str
    one: Any = None
    all: list[Any] = field(default_factory=list)


class ScriptedCursor:
    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.current: Step | None = None
        self.executed: list[tuple[str, Any]] = []

    def execute(self, sql: str, params=None):
        assert self.steps, f"unexpected SQL: {sql}"
        step = self.steps.pop(0)
        normalized = " ".join(sql.lower().split())
        assert step.contains.lower() in normalized
        self.current = step
        self.executed.append((normalized, params))
        return self

    def fetchone(self):
        assert self.current is not None
        return self.current.one

    def fetchall(self):
        assert self.current is not None
        return list(self.current.all)


class FakeConnection:
    def __init__(self, steps: list[Step]) -> None:
        self.cursor_value = ScriptedCursor(steps)

    def cursor(self) -> ScriptedCursor:
        return self.cursor_value

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


def install_database(monkeypatch, steps: list[Step]) -> FakeConnection:
    connection = FakeConnection(steps)

    @contextmanager
    def fake_get_db() -> Iterator[FakeConnection]:
        yield connection

    monkeypatch.setattr(session_routes, "get_db", fake_get_db)
    return connection


def test_get_chat_prefers_durable_messages_and_preserves_frontend_fields(monkeypatch):
    connection = install_database(
        monkeypatch,
        [
            Step(
                "select id, title, messages",
                one={
                    "id": CHAT_ID,
                    "title": "Durable",
                    "messages": [{"role": "assistant", "content": "legacy"}],
                    "created_at": "created",
                    "updated_at": "updated",
                    "pinned": False,
                },
            ),
            Step(
                "from chat_messages",
                all=[
                    {
                        "role": "assistant",
                        "content": "durable answer",
                        "content_json": {"followups": ["What changed next?"], "unverified": True},
                        "provider": "ollama",
                        "model": "local-model",
                        "tool_name": None,
                        "tool_call_id": None,
                        "sources": [{"id": "V1"}],
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                    }
                ],
            ),
        ],
    )

    result = asyncio.run(session_routes.get_chat(CHAT_ID, USER))

    assert result["id"] == CHAT_ID
    assert result["messages"] == [
        {
            "role": "assistant",
            "content": "durable answer",
            "sources": [{"id": "V1", "kind": "vault", "match_percentage": 0}],
            # Stored followups are now trusted verbatim (persisted == streamed).
            "followups": ["What changed next?"],
            "unverified": True,
            "provider": "ollama",
            "model": "local-model",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }
    ]
    assert connection.cursor_value.executed[1][1] == (CHAT_ID, USER["id"])
    assert connection.cursor_value.steps == []


def test_get_chat_uses_sanitized_legacy_json_only_when_durable_history_is_empty(monkeypatch):
    install_database(
        monkeypatch,
        [
            Step(
                "select id, title, messages",
                one={
                    "id": CHAT_ID,
                    "title": "Legacy",
                    "messages": """[
                        {"role":"user","content":"old question","sources":"bad"},
                        {"role":"admin","content":"must be dropped"},
                        {"role":"assistant","content":""},
                        "not a message"
                    ]""",
                    "created_at": "created",
                    "updated_at": "updated",
                    "pinned": False,
                },
            ),
            Step("from chat_messages", all=[]),
        ],
    )

    result = asyncio.run(session_routes.get_chat(CHAT_ID, USER))

    assert result["messages"] == [{"role": "user", "content": "old question"}]


def test_history_accepts_legacy_pgchat_session_id_and_reads_durable_rows(monkeypatch):
    connection = install_database(
        monkeypatch,
        [
            Step(
                "select id, messages from chats",
                one={"id": CHAT_ID, "messages": []},
            ),
            Step(
                "from chat_messages",
                all=[
                    {
                        "role": "user",
                        "content": "question",
                        "content_json": {},
                        "provider": None,
                        "model": None,
                        "tool_name": None,
                        "tool_call_id": None,
                        "sources": [],
                        "prompt_tokens": None,
                        "completion_tokens": None,
                    }
                ],
            ),
        ],
    )

    result = asyncio.run(
        session_routes.get_chat_history(
            USER,
            session_id=f"user_17_pgchat_{CHAT_ID}",
        )
    )

    assert result == {"history": [{"role": "user", "content": "question", "sources": []}]}
    assert connection.cursor_value.executed[0][1] == (CHAT_ID, USER["id"])


def test_patch_replaces_chat_messages_without_copying_history_back_to_legacy_jsonb(monkeypatch):
    connection = install_database(
        monkeypatch,
        [
            Step(
                "select id, title from chats",
                one={"id": CHAT_ID, "title": "New Chat"},
            ),
            Step("delete from chat_messages"),
            Step("insert into chat_messages"),
            Step("insert into chat_messages"),
            Step(
                "update chats set",
                one={"id": CHAT_ID, "title": "question", "updated_at": "updated"},
            ),
        ],
    )
    body = ChatPatchRequest(
        messages=[
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "answer",
                "sources": [{"id": "V1"}],
                "followups": ["continue?"],
            },
        ]
    )

    result = asyncio.run(session_routes.patch_chat(CHAT_ID, body, USER))

    assert result == {"id": CHAT_ID, "title": "question", "updated_at": "updated"}
    statements = connection.cursor_value.executed
    assert statements[2][1][2:5] == (0, "user", "question")
    assert statements[3][1][2:5] == (1, "assistant", "answer")
    assert statements[3][1][10] == '[{"id":"V1","kind":"vault","match_percentage":0}]'
    update_sql = statements[-1][0]
    assert "messages = '[]'::jsonb" in update_sql
    assert "messages = %s" not in update_sql
    assert connection.cursor_value.steps == []


def test_legacy_and_patch_projection_strip_private_answer_and_source_fields():
    message = session_routes._normalize_public_message(
        {
            "role": "assistant",
            "content": (
                "Public answer [V1].\n"
                '< LAVIX_FOLLOWUPS>{"followups":["What changed next?",'
                '"Which evidence matters most?"]}</ LAVIX_FOLLOWUPS>'
            ),
            "sources": [
                {
                    "id": "V1",
                    "file_id": 9,
                    "filename": "report.pdf",
                    "score": 0.61,
                    "revision": 4,
                    "chunk_id": "private",
                    "provenance": [{"page": 1}],
                    "temperature": 0.7,
                }
            ],
            "tool_name": "search_vault",
            "tool_call_id": "private-call",
        }
    )

    assert message == {
        "role": "assistant",
        "content": "Public answer.",
        "sources": [
            {
                "id": "V1",
                "kind": "vault",
                "match_percentage": 61,
                "file_id": 9,
                "filename": "report.pdf",
            }
        ],
    }


def test_public_route_contract_is_preserved_without_redis_or_embedding_imports():
    routes = {(method, route.path) for route in session_routes.router.routes for method in route.methods}
    assert {
        ("GET", "/chats"),
        ("POST", "/chats"),
        ("GET", "/chats/{chat_id}"),
        ("PATCH", "/chats/{chat_id}"),
        ("DELETE", "/chats/{chat_id}"),
        ("GET", "/chat/sessions"),
        ("GET", "/chat/history"),
        ("DELETE", "/chat/history"),
        ("DELETE", "/chat/history/{session_id}"),
    }.issubset(routes)

    source = inspect.getsource(session_routes).lower()
    for forbidden in (
        "redis_service",
        "ai_embeddings",
        "delete_embeddings",
        "embeddings_generated",
        "chromadb",
    ):
        assert forbidden not in source


def test_get_chat_projects_scoped_file_ids_from_session_scope(monkeypatch):
    install_database(
        monkeypatch,
        [
            Step(
                "select id, title, messages",
                one={
                    "id": CHAT_ID,
                    "title": "Scoped",
                    "messages": {"scoped_file_ids": [8, 9]},
                    "created_at": "created",
                    "updated_at": "updated",
                    "pinned": False,
                },
            ),
            Step("from chat_messages", all=[]),
        ],
    )

    result = asyncio.run(session_routes.get_chat(CHAT_ID, USER))

    assert result["scoped_file_ids"] == [8, 9]
    assert result["messages"] == []


def test_get_chat_projects_empty_scope_for_legacy_rows(monkeypatch):
    install_database(
        monkeypatch,
        [
            Step(
                "select id, title, messages",
                one={
                    "id": CHAT_ID,
                    "title": "Legacy",
                    "messages": [],
                    "created_at": "created",
                    "updated_at": "updated",
                    "pinned": False,
                },
            ),
            Step("from chat_messages", all=[]),
        ],
    )

    result = asyncio.run(session_routes.get_chat(CHAT_ID, USER))

    assert result["scoped_file_ids"] == []


def test_set_scope_endpoint_replaces_and_echoes_scope(monkeypatch):
    import asyncio

    from app.routers.chat.schemas import ChatScopeRequest

    connection = install_database(
        monkeypatch,
        [
            Step("SELECT id FROM chats", one={"id": CHAT_ID}),
            Step("UPDATE chats", one=None),
            Step(
                "SELECT messages FROM chats",
                one={"messages": {"scoped_file_ids": [8]}},
            ),
            Step(
                "SELECT messages FROM chats",
                one={"messages": {"scoped_file_ids": [8]}},
            ),
        ],
    )

    result = asyncio.run(
        session_routes.set_chat_file_scope(
            CHAT_ID, ChatScopeRequest(file_ids=[8, 8]), USER
        )
    )

    assert result == {"id": CHAT_ID, "scoped_file_ids": [8], "scoped_folder_ids": []}
    statements = [sql for sql, _ in connection.cursor_value.executed]
    assert any("jsonb_set" in sql for sql in statements)


def test_set_scope_endpoint_clears_on_empty_list(monkeypatch):
    import asyncio

    from app.routers.chat.schemas import ChatScopeRequest

    install_database(
        monkeypatch,
        [
            Step("SELECT id FROM chats", one={"id": CHAT_ID}),
            Step("UPDATE chats", one=None),
            Step("SELECT messages FROM chats", one={"messages": {}}),
            Step("SELECT messages FROM chats", one={"messages": {}}),
        ],
    )

    result = asyncio.run(
        session_routes.set_chat_file_scope(CHAT_ID, ChatScopeRequest(file_ids=[]), USER)
    )

    assert result == {"id": CHAT_ID, "scoped_file_ids": [], "scoped_folder_ids": []}


def test_set_scope_endpoint_404s_unknown_chat(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from app.routers.chat.schemas import ChatScopeRequest

    install_database(monkeypatch, [Step("SELECT id FROM chats", one=None)])

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            session_routes.set_chat_file_scope(
                CHAT_ID, ChatScopeRequest(file_ids=[8]), USER
            )
        )
    assert exc_info.value.status_code == 404


def test_scope_request_schema_rejects_garbage():
    from pydantic import ValidationError

    from app.routers.chat.schemas import ChatScopeRequest

    assert ChatScopeRequest(file_ids=[3, 3]).file_ids == [3]
    for bad in ([0], [-2], [True], ["x"]):
        with pytest.raises(ValidationError):
            ChatScopeRequest(file_ids=bad)
