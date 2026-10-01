from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import auth


class RecordingCursor:
    def __init__(self, rows: list[dict[str, Any] | None] | None = None) -> None:
        self.rows = list(rows or [])
        self.executed: list[tuple[str, object]] = []

    def execute(self, sql: str, params: object = None) -> None:
        self.executed.append((" ".join(sql.lower().split()), params))

    def fetchone(self) -> dict[str, Any] | None:
        return self.rows.pop(0) if self.rows else None


class RecordingConnection:
    def __init__(self, rows: list[dict[str, Any] | None] | None = None) -> None:
        self.cursor_value = RecordingCursor(rows)

    def cursor(self) -> RecordingCursor:
        return self.cursor_value


def database_context(connection: RecordingConnection):
    @contextmanager
    def get_db() -> Iterator[RecordingConnection]:
        yield connection

    return get_db


def account_client(user: dict[str, Any] | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(auth.router, prefix="/api/auth")
    app.dependency_overrides[auth.get_current_user] = lambda: (
        user
        or {
            "id": 41,
            "perm_ai": True,
        }
    )
    return TestClient(app)


def test_profile_update_apis_persist_only_the_authenticated_users_values(monkeypatch) -> None:
    connection = RecordingConnection(rows=[None])
    monkeypatch.setattr(auth, "get_db", database_context(connection))
    client = account_client()

    persona = client.put("/api/auth/persona", json={"persona_prompt": "Answer in short bullets."})
    avatar = client.put(
        "/api/auth/avatar",
        json={"avatar_data": "data:image/png;base64,iVBORw0KGgo="},
    )
    username = client.put("/api/auth/username", json={"username": "  atlas.reader  "})

    assert persona.status_code == 200
    assert persona.json() == {"message": "Persona updated"}
    assert avatar.status_code == 200
    assert avatar.json() == {"message": "Avatar updated"}
    assert username.status_code == 200
    assert username.json() == {"message": "Username updated", "username": "atlas.reader"}
    assert connection.cursor_value.executed == [
        (
            "update users set persona_prompt = %s where id = %s",
            ("Answer in short bullets.", 41),
        ),
        (
            "update users set avatar_data = %s where id = %s",
            ("data:image/png;base64,iVBORw0KGgo=", 41),
        ),
        (
            "select 1 from users where username = %s and id != %s",
            ("atlas.reader", 41),
        ),
        (
            "update users set username = %s where id = %s",
            ("atlas.reader", 41),
        ),
    ]


def test_profile_update_apis_reject_invalid_values_without_writing(monkeypatch) -> None:
    connection = RecordingConnection()
    monkeypatch.setattr(auth, "get_db", database_context(connection))
    client = account_client()

    assert client.put("/api/auth/persona", json={"persona_prompt": "x" * 4_001}).status_code == 422
    invalid_avatar = client.put("/api/auth/avatar", json={"avatar_data": "https://example.com/me.png"})
    invalid_username = client.put("/api/auth/username", json={"username": "not allowed"})

    assert invalid_avatar.status_code == 400
    assert invalid_avatar.json()["detail"] == "Invalid image format"
    assert invalid_username.status_code == 400
    assert "letters, numbers" in invalid_username.json()["detail"]
    assert connection.cursor_value.executed == []


def test_username_update_rejects_an_existing_name_without_overwriting(monkeypatch) -> None:
    connection = RecordingConnection(rows=[{"exists": 1}])
    monkeypatch.setattr(auth, "get_db", database_context(connection))

    response = account_client().put("/api/auth/username", json={"username": "already-taken"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Username already taken"
    assert connection.cursor_value.executed == [
        (
            "select 1 from users where username = %s and id != %s",
            ("already-taken", 41),
        )
    ]


def test_delete_session_is_scoped_to_the_authenticated_user(monkeypatch) -> None:
    connection = RecordingConnection()
    monkeypatch.setattr(auth, "get_db", database_context(connection))

    response = account_client().delete("/api/auth/sessions/271bd703-ea37-42d6-b1ea-960ee6ddc914")

    assert response.status_code == 200
    assert response.json() == {"message": "Session revoked"}
    assert connection.cursor_value.executed == [
        (
            "update refresh_tokens set is_revoked = true where session_id = %s and user_id = %s",
            ("271bd703-ea37-42d6-b1ea-960ee6ddc914", 41),
        )
    ]


class LocalModelService:
    def __init__(self) -> None:
        self.resolve_calls: list[dict[str, Any]] = []
        self.openrouter_calls: list[str] = []

    def resolve(self, **kwargs: Any) -> tuple[str, str, int]:
        self.resolve_calls.append(kwargs)
        return ("http://ollama.test/v1/chat/completions", "local-chat:latest", 120)

    def get_available_models(self) -> list[dict[str, Any]]:
        return [
            {"name": "local-chat:latest"},
            {"name": "local-reasoner:latest"},
            {"name": "local-vision:latest"},
            {"name": "local-embed:latest"},
        ]

    def fetch_openrouter_free_models(self, *, api_key: str) -> list[str]:
        self.openrouter_calls.append(api_key)
        return ["cloud/model"]

    @staticmethod
    def is_service_available(url: str) -> bool:
        return url == "http://reranker.test/health"


def test_model_config_reports_the_authenticated_users_local_stack(monkeypatch) -> None:
    connection = RecordingConnection(
        rows=[
            {
                "preferred_chat_model": "local-reasoner:latest",
            }
        ]
    )
    model_service = LocalModelService()
    monkeypatch.setattr(auth, "get_db", database_context(connection))
    monkeypatch.setattr(auth, "get_model_service", lambda: model_service)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")
    monkeypatch.setenv("LLM_MODEL", "local-chat:latest")
    monkeypatch.setenv("ALLOWED_CHAT_MODELS", "local-chat:latest,local-reasoner:latest")
    monkeypatch.setenv("VISION_MODEL", "local-vision:latest")
    monkeypatch.setenv("INTELLIGENCE_MODEL", "local-vision:latest")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "local-embed:latest")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("EMBEDDING_API_URL", "http://ollama.test/v1/embeddings")
    monkeypatch.setenv("RERANK_BASE_URL", "http://reranker.test")
    monkeypatch.setenv("RERANK_MODEL", "local-reranker")
    monkeypatch.setenv("ENABLE_RERANKING", "true")

    response = account_client().get("/api/auth/model-config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "ollama"
    assert payload["model"] == "local-reasoner:latest"
    assert payload["preferred_chat_model"] == "local-reasoner:latest"
    assert payload["active_chat_model"] == "local-reasoner:latest"
    assert payload["allowed_chat_models"] == ["local-chat:latest", "local-reasoner:latest"]
    assert payload["available_models"] == [
        "local-chat:latest",
        "local-reasoner:latest",
        "local-vision:latest",
        "local-embed:latest",
    ]
    assert payload["ollama_base_url"] == "http://ollama.test"
    assert payload["embedding_model"] == "local-embed:latest"
    assert payload["embedding_api_url"] == "http://ollama.test/v1/embeddings"
    assert payload["reranker_configured"] is True
    assert payload["reranker_model"] == "local-reranker"
    assert payload["reranker_url"] == "http://reranker.test"
    assert payload["enable_reranking"] is True
    assert payload["chat"] == {
        "model": "local-reasoner:latest",
        "preferred_model": "local-reasoner:latest",
        "allowed_models": ["local-chat:latest", "local-reasoner:latest"],
        "default_model": "local-chat:latest",
        "enabled": True,
        "configured": True,
        "available": True,
        "optional": False,
    }
    assert payload["vision"] == {
        "model": "local-vision:latest",
        "configured": True,
        "enabled": True,
        "available": True,
        "optional": True,
    }
    assert payload["intelligence"]["model"] == "local-vision:latest"
    assert payload["embedding"]["dimension"] == 1024
    assert payload["embedding"]["available"] is True
    assert payload["embedding"]["optional"] is False
    assert payload["reranker"]["available"] is True
    assert payload["reranker"]["optional"] is True
    assert model_service.openrouter_calls == []
    assert connection.cursor_value.executed == [
        (
            "select preferred_chat_model from users where id = %s",
            (41,),
        ),
        (
            "select revision, chat_enabled, chat_default_model, chat_allowed_models, "
            "vision_enabled, vision_model, intelligence_enabled, intelligence_model, "
            "memory_extraction_enabled, memory_extraction_model, reranker_enabled, "
            "clock_timezone, model_max_num_ctx, fallback_chat_model, file_scope, top_k, search_depth, session_timeout_minutes, registration_enabled, updated_by, created_at, updated_at from system_ai_configuration "
            "where singleton_id = 1",
            None,
        ),
    ]


def test_chat_model_preference_is_allowlisted_account_scoped_and_nullable(monkeypatch) -> None:
    connection = RecordingConnection()
    monkeypatch.setattr(auth, "get_db", database_context(connection))
    monkeypatch.setattr(auth, "get_model_service", LocalModelService)
    monkeypatch.setenv("LLM_MODEL", "local-chat:latest")
    monkeypatch.setenv("ALLOWED_CHAT_MODELS", "local-chat:latest,local-reasoner:latest")
    client = account_client()

    selected = client.put("/api/auth/chat-model", json={"model": " local-reasoner:latest "})
    reset = client.put("/api/auth/chat-model", json={"model": None})
    rejected = client.put("/api/auth/chat-model", json={"model": "untrusted:latest"})

    assert selected.status_code == 200
    assert selected.json() == {
        "preferred_chat_model": "local-reasoner:latest",
        "active_chat_model": "local-reasoner:latest",
        "allowed_chat_models": ["local-chat:latest", "local-reasoner:latest"],
        "chat_enabled": True,
        "revision": 0,
    }
    assert reset.status_code == 200
    assert reset.json()["preferred_chat_model"] is None
    assert reset.json()["active_chat_model"] == "local-chat:latest"
    assert rejected.status_code == 422
    updates = [
        statement
        for statement in connection.cursor_value.executed
        if statement[0].startswith("update users set preferred_chat_model")
    ]
    assert updates == [
        (
            "update users set preferred_chat_model = %s where id = %s",
            ("local-reasoner:latest", 41),
        ),
        (
            "update users set preferred_chat_model = %s where id = %s",
            (None, 41),
        ),
    ]


def test_model_config_marks_an_unconfigured_vlm_as_optional_and_disabled(monkeypatch) -> None:
    connection = RecordingConnection(
        rows=[
            {
                "openrouter_api_key": None,
                "openrouter_provider": "ollama",
                "preferred_chat_model": None,
            }
        ]
    )
    monkeypatch.setattr(auth, "get_db", database_context(connection))
    monkeypatch.setattr(auth, "get_model_service", LocalModelService)
    monkeypatch.setenv("LLM_MODEL", "local-chat:latest")
    monkeypatch.setenv("ALLOWED_CHAT_MODELS", "local-chat:latest")
    monkeypatch.setenv("VISION_MODEL", "")

    response = account_client().get("/api/auth/model-config")

    assert response.status_code == 200
    assert response.json()["vision"] == {
        "model": "",
        "configured": False,
        "enabled": False,
        "available": False,
        "optional": True,
    }


def test_model_config_falls_back_from_a_preference_disabled_by_the_system_revision(
    monkeypatch,
) -> None:
    connection = RecordingConnection(
        rows=[
            {
                "openrouter_api_key": None,
                "openrouter_provider": "ollama",
                "preferred_chat_model": "local-chat:latest",
            },
            {
                "revision": 12,
                "chat_enabled": True,
                "chat_default_model": "local-reasoner:latest",
                "chat_allowed_models": ["local-reasoner:latest"],
                "vision_enabled": False,
                "vision_model": None,
                "intelligence_enabled": True,
                "intelligence_model": "local-reasoner:latest",
                "memory_extraction_enabled": False,
                "memory_extraction_model": "local-reasoner:latest",
                "reranker_enabled": False,
                "updated_by": 1,
                "created_at": None,
                "updated_at": None,
            },
        ]
    )
    monkeypatch.setattr(auth, "get_db", database_context(connection))
    monkeypatch.setattr(auth, "get_model_service", LocalModelService)
    monkeypatch.setenv("ALLOWED_CHAT_MODELS", "local-chat:latest,local-reasoner:latest")

    response = account_client().get("/api/auth/model-config")

    assert response.status_code == 200
    assert response.json()["revision"] == 12
    assert response.json()["configuration_source"] == "database"
    assert response.json()["preferred_chat_model"] is None
    assert response.json()["active_chat_model"] == "local-reasoner:latest"
    assert response.json()["allowed_chat_models"] == ["local-reasoner:latest"]
    assert response.json()["reranker"]["enabled"] is False


def test_model_config_falls_back_from_an_allowlisted_model_removed_from_ollama(
    monkeypatch,
) -> None:
    connection = RecordingConnection(
        rows=[
            {
                "openrouter_api_key": None,
                "openrouter_provider": "ollama",
                "preferred_chat_model": "local-reasoner:latest",
            }
        ]
    )
    model_service = LocalModelService()
    model_service.get_available_models = lambda: [{"name": "local-chat:latest"}]
    monkeypatch.setattr(auth, "get_db", database_context(connection))
    monkeypatch.setattr(auth, "get_model_service", lambda: model_service)
    monkeypatch.setenv("LLM_MODEL", "local-chat:latest")
    monkeypatch.setenv(
        "ALLOWED_CHAT_MODELS",
        "local-chat:latest,local-reasoner:latest",
    )

    response = account_client().get("/api/auth/model-config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["preferred_chat_model"] is None
    assert payload["active_chat_model"] == "local-chat:latest"
    assert payload["chat"]["model"] == "local-chat:latest"
    assert payload["chat"]["available"] is True


def test_chat_model_selection_reports_a_typed_error_when_chat_is_disabled(monkeypatch) -> None:
    connection = RecordingConnection(
        rows=[
            {
                "revision": 3,
                "chat_enabled": False,
                "chat_default_model": "local-chat:latest",
                "chat_allowed_models": ["local-chat:latest"],
                "vision_enabled": False,
                "vision_model": None,
                "intelligence_enabled": False,
                "intelligence_model": None,
                "memory_extraction_enabled": False,
                "memory_extraction_model": None,
                "reranker_enabled": False,
                "updated_by": 1,
                "created_at": None,
                "updated_at": None,
            }
        ]
    )
    monkeypatch.setattr(auth, "get_db", database_context(connection))
    monkeypatch.setattr(auth, "get_model_service", LocalModelService)
    monkeypatch.setenv("ALLOWED_CHAT_MODELS", "local-chat:latest")

    response = account_client().put(
        "/api/auth/chat-model",
        json={"model": "local-chat:latest"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "ai_chat_disabled",
        "message": "AI Chat is disabled",
    }
    assert not any(
        statement.startswith("update users")
        for statement, _params in connection.cursor_value.executed
    )


def test_disabled_google_oauth_is_not_advertised_and_fails_before_exchange(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GOOGLE_REDIRECT_URI", raising=False)
    client = account_client()

    capabilities = client.get("/api/auth/capabilities")
    login = client.get("/api/auth/google/login", follow_redirects=False)
    callback = client.get(
        "/api/auth/google/callback?code=untrusted&state=untrusted",
        follow_redirects=False,
    )

    assert capabilities.status_code == 200
    assert capabilities.json()["google_oauth_enabled"] is False
    assert login.status_code == 503
    assert login.json()["detail"] == "Google OAuth not configured"
    assert callback.status_code == 503
    assert callback.json()["detail"] == "Google OAuth not configured"
