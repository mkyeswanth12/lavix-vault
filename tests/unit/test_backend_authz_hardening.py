from __future__ import annotations

import inspect
from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from app import auth as auth_helpers
from app.config import Settings
from app.routers import admin
from app.routers import auth as auth_routes
from app.routers import files as file_routes
from app.routers.chat import router as ai_router
from app.routers.chat.schemas import (
    ChatCreateRequest,
    ChatPatchRequest,
    ChatRequest,
    SemanticSearchRequest,
)


def test_all_public_ai_routes_fail_closed_without_ai_permission() -> None:
    app = FastAPI()
    app.include_router(ai_router, prefix="/api/ai")
    app.dependency_overrides[auth_helpers.get_current_user] = lambda: {
        "id": 7,
        "perm_ai": False,
    }
    client = TestClient(app)

    requests = (
        client.post("/api/ai/chat", json={"message": "hello"}),
        client.post("/api/ai/search", json={"query": "hello"}),
        client.get("/api/ai/chats"),
    )

    assert {response.status_code for response in requests} == {403}
    assert all(response.json()["detail"] == "AI access permission denied" for response in requests)

    auth_app = FastAPI()
    auth_app.include_router(auth_routes.router, prefix="/api/auth")
    auth_app.dependency_overrides[auth_helpers.get_current_user] = lambda: {
        "id": 7,
        "perm_ai": False,
    }
    assert TestClient(auth_app).get("/api/auth/models").status_code == 403


def test_file_move_fails_closed_without_folder_permission() -> None:
    with pytest.raises(file_routes.HTTPException) as caught:
        file_routes.move_file(
            1,
            file_routes.MoveFileRequest(folder_id=None),
            {"id": 7, "perm_folders": False},
        )

    assert caught.value.status_code == 403
    assert caught.value.detail == "Folder move permission denied"


def test_chat_web_search_defaults_on_and_removed_fields_are_rejected() -> None:
    assert ChatRequest(message="hello").web_search_enabled is True
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "hello", "confirm_search": True})
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "hello", "max_context_chunks": 4})
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "hello", "session_id": "legacy"})


@pytest.mark.parametrize(
    ("model", "payload"),
    (
        (SemanticSearchRequest, {"query": "", "max_results": 10}),
        (SemanticSearchRequest, {"query": "x" * 1_001}),
        (SemanticSearchRequest, {"query": "hello", "max_results": 0}),
        (SemanticSearchRequest, {"query": "hello", "max_results": 21}),
        (ChatCreateRequest, {"title": "x" * 201}),
        (ChatPatchRequest, {"title": "x" * 201}),
        (ChatPatchRequest, {"messages": [{}] * 1_001}),
    ),
)
def test_public_chat_utility_schemas_reject_out_of_bounds_values(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize(
    ("model", "payload"),
    (
        (SemanticSearchRequest, {"query": "hello", "unexpected": True}),
        (ChatCreateRequest, {"title": None, "unexpected": True}),
        (ChatPatchRequest, {"pinned": True, "unexpected": True}),
    ),
)
def test_public_chat_utility_schemas_forbid_unknown_fields(model, payload) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate(payload)


def test_public_chat_utility_schemas_preserve_valid_optional_payloads() -> None:
    assert SemanticSearchRequest(query="hello", max_results=None).max_results is None
    assert ChatCreateRequest(title=None).title is None
    assert ChatPatchRequest(title=None, messages=None, pinned=None).messages is None


@pytest.mark.parametrize(
    ("model", "payload"),
    (
        (
            auth_routes.RegisterRequest,
            {"username": "user", "email": "user@example.com", "password": "x" * 73},
        ),
        (auth_routes.LoginRequest, {"username": "user", "password": "é" * 37}),
        (
            auth_routes.ChangePasswordRequest,
            {"current_password": "current-pass", "new_password": "x" * 73},
        ),
        (admin.ResetPasswordRequest, {"new_password": "é" * 37}),
    ),
)
def test_auth_schemas_reject_passwords_over_bcrypt_byte_limit(model, payload) -> None:
    with pytest.raises(ValidationError, match="72 UTF-8 bytes|string_too_long"):
        model.model_validate(payload)


def test_password_helpers_never_hash_or_verify_truncated_values(monkeypatch) -> None:
    assert auth_helpers.validate_bcrypt_password("é" * 36) == "é" * 36
    with pytest.raises(ValueError, match="72 UTF-8 bytes"):
        auth_helpers.get_password_hash("é" * 37)

    monkeypatch.setattr(
        auth_helpers.bcrypt,
        "checkpw",
        lambda *_args: pytest.fail("bcrypt must not receive an oversized password"),
    )
    assert auth_helpers.verify_password("x" * 73, "unused") is False


def test_existing_bcrypt_password_hashes_remain_compatible() -> None:
    existing_hash = "$2b$04$4soDeWz41D9M.o/NHKQDiuNoIFYhHHnhrtqIEQjnucDqvf4y4pHc2"

    assert auth_helpers.verify_password("existing-password", existing_hash) is True
    assert auth_helpers.verify_password("wrong-password", existing_hash) is False

    new_hash = auth_helpers.get_password_hash("new-password")
    assert new_hash.startswith("$2b$")
    assert auth_helpers.verify_password("new-password", new_hash) is True


def test_legacy_refresh_verification_remains_available_for_one_time_rotation() -> None:
    token = "legacy-refresh-jwt-" + ("x" * 100)
    legacy_hash = "$2b$04$5BzpmVBA29lKd/G.AOMsOeSGRoFaGrMERSBnIcH6mWva.t2JSp0PK"

    assert auth_helpers.verify_refresh_token(token, legacy_hash) is True


class RecordingCursor:
    def __init__(self, rows: list[dict | None]) -> None:
        self.rows = list(rows)
        self.executed: list[tuple[str, object]] = []

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((" ".join(sql.lower().split()), params))

    def fetchone(self):
        return self.rows.pop(0)


class RecordingConnection:
    def __init__(self, rows: list[dict | None]) -> None:
        self.cursor_value = RecordingCursor(rows)

    def cursor(self) -> RecordingCursor:
        return self.cursor_value


def database_context(connection: RecordingConnection):
    @contextmanager
    def get_db():
        yield connection

    return get_db


def test_password_change_updates_hash_and_revokes_sessions_in_one_transaction(monkeypatch) -> None:
    connection = RecordingConnection([{"password_hash": "old-hash"}])
    monkeypatch.setattr(auth_routes, "get_db", database_context(connection))
    monkeypatch.setattr(
        auth_routes, "verify_password", lambda plain, hashed: (plain, hashed) == ("old-pass", "old-hash")
    )
    monkeypatch.setattr(auth_routes, "get_password_hash", lambda _value: "new-hash")

    result = auth_routes.change_password(
        auth_routes.ChangePasswordRequest(current_password="old-pass", new_password="new-pass"),
        {"id": 23},
    )

    assert result == {"message": "Password changed successfully"}
    statements = connection.cursor_value.executed
    assert "for update" in statements[0][0]
    assert statements[1] == (
        "update users set password_hash = %s where id = %s",
        ("new-hash", 23),
    )
    assert statements[2] == (
        "update refresh_tokens set is_revoked = true where user_id = %s",
        (23,),
    )


def test_admin_password_reset_revokes_target_sessions_in_same_transaction(monkeypatch) -> None:
    connection = RecordingConnection([{"id": 31}])
    monkeypatch.setattr(admin, "get_db", database_context(connection))
    monkeypatch.setattr(admin, "get_password_hash", lambda _value: "reset-hash")

    result = admin.reset_user_password(
        31,
        admin.ResetPasswordRequest(new_password="new-pass"),
        {"id": 1, "is_admin": True},
    )

    assert result == {"message": "Password reset successfully"}
    assert connection.cursor_value.executed == [
        (
            "update users set password_hash = %s where id = %s returning id",
            ("reset-hash", 31),
        ),
        (
            "update refresh_tokens set is_revoked = true where user_id = %s",
            (31,),
        ),
    ]


def test_production_requires_explicit_independent_non_placeholder_secrets(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("AGENT_CAPABILITY_SECRET", raising=False)
    with pytest.raises(ValueError, match="explicit non-placeholder secrets"):
        Settings().validate_runtime_secrets()

    monkeypatch.setenv("SECRET_KEY", "replace-with-at-least-32-random-bytes")
    monkeypatch.setenv("AGENT_CAPABILITY_SECRET", "a" * 40)
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings().validate_runtime_secrets()

    monkeypatch.setenv("SECRET_KEY", "lavix-vault-secret-key-change-in-production-2024")
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings().validate_runtime_secrets()

    monkeypatch.setenv("SECRET_KEY", "s" * 40)
    monkeypatch.setenv("AGENT_CAPABILITY_SECRET", "s" * 40)
    with pytest.raises(ValueError, match="must be independent"):
        Settings().validate_runtime_secrets()

    monkeypatch.setenv("AGENT_CAPABILITY_SECRET", "a" * 40)
    Settings().validate_runtime_secrets()


def test_e2e_environment_can_use_isolated_harness_configuration(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "e2e")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("AGENT_CAPABILITY_SECRET", raising=False)
    Settings().validate_runtime_secrets()


def test_authenticated_user_and_auth_router_have_no_removed_model_metrics() -> None:
    assert "model_preferences" not in inspect.getsource(auth_helpers.get_current_user)
    routes = {(method, route.path) for route in auth_routes.router.routes for method in route.methods}
    assert ("GET", "/model-perf") not in routes


def test_auth_router_has_no_removed_clerk_surface() -> None:
    routes = {(method, route.path) for route in auth_routes.router.routes for method in route.methods}
    assert ("GET", "/config") not in routes
    assert ("POST", "/clerk-signin") not in routes
    assert not hasattr(auth_routes, "ClerkSigninRequest")
    assert "clerk" not in inspect.getsource(auth_routes).lower()


def test_registration_is_closed_by_default_in_production(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("REGISTRATION_ENABLED", raising=False)
    app = FastAPI()
    app.include_router(auth_routes.router, prefix="/api/auth")

    response = TestClient(app).post(
        "/api/auth/register",
        json={
            "username": "closed-user",
            "email": "closed@example.com",
            "password": "strong-password",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Account registration is disabled"


def test_auth_capabilities_fail_closed_and_only_advertise_configured_oauth(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("REGISTRATION_ENABLED", raising=False)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "https://vault.example/api/auth/google/callback")

    assert auth_routes.authentication_capabilities() == {
        "registration_enabled": False,
        "google_oauth_enabled": True,
    }
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET")
    assert auth_routes.authentication_capabilities()["google_oauth_enabled"] is False


def test_production_auth_rate_limit_fails_closed_when_redis_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("AUTH_RATE_LIMIT_FAIL_CLOSED", raising=False)
    monkeypatch.setattr(
        auth_routes,
        "_get_rl_client",
        lambda: (_ for _ in ()).throw(ConnectionError("redis unavailable")),
    )
    request = Request({"type": "http", "client": ("127.0.0.1", 1234), "headers": []})

    with pytest.raises(auth_routes.HTTPException) as caught:
        auth_routes._rate_limit(request, "login", max_hits=10, window=60)

    assert caught.value.status_code == 503
