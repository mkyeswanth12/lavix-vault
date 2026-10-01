from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from app import auth as auth_helpers
from app.config import settings
from app.routers import auth as auth_routes


class ScriptedCursor:
    def __init__(
        self,
        *,
        one: list[dict[str, Any] | None] | None = None,
        many: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.one = list(one or [])
        self.many = list(many or [])
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql: str, params: object = None) -> None:
        self.calls.append((" ".join(sql.lower().split()), params))

    def fetchone(self) -> dict[str, Any] | None:
        return self.one.pop(0) if self.one else None

    def fetchall(self) -> list[dict[str, Any]]:
        return self.many.pop(0) if self.many else []


class FakeConnection:
    def __init__(self, cursor: ScriptedCursor) -> None:
        self.cursor_value = cursor
        self.exited_cleanly = False

    def cursor(self) -> ScriptedCursor:
        return self.cursor_value


def database(connection: FakeConnection):
    @contextmanager
    def get_db() -> Iterator[FakeConnection]:
        yield connection
        connection.exited_cleanly = True

    return get_db


def credentials(token: str = "access-token") -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/auth/refresh",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )


def session_row(last_activity_at: datetime, server_now: datetime) -> dict[str, datetime]:
    return {"last_activity_at": last_activity_at, "server_now": server_now}


def test_session_timeout_configuration_defaults_to_120_minutes(monkeypatch) -> None:
    monkeypatch.delenv("ACCESS_TOKEN_EXPIRE_MINUTES", raising=False)
    monkeypatch.delenv("SESSION_IDLE_TIMEOUT_MINUTES", raising=False)

    assert settings.access_token_expire_minutes == 120
    assert settings.session_idle_timeout_minutes == 120

    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "0")
    with pytest.raises(ValueError, match="greater than zero"):
        _ = settings.session_idle_timeout_minutes


def test_idle_boundary_is_active_before_120_minutes_and_idle_at_120(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "120")
    started = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)

    active = auth_helpers.SessionActivity(
        last_activity_at=started,
        server_now=started + timedelta(minutes=119, seconds=59),
        idle_expires_at=auth_helpers.session_idle_deadline(started),
    )
    boundary = auth_helpers.SessionActivity(
        last_activity_at=started,
        server_now=started + timedelta(minutes=120),
        idle_expires_at=auth_helpers.session_idle_deadline(started),
    )

    assert active.is_idle is False
    assert boundary.is_idle is True
    assert auth_helpers.session_idle_timeout_seconds() == 7_200


def test_protected_request_validates_without_extending_activity(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "120")
    session_id = str(uuid4())
    now = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    user = {
        "id": 17,
        "username": "reader",
        "email": "reader@example.test",
        "is_active": True,
        "perm_ai": True,
    }
    cursor = ScriptedCursor(one=[session_row(now - timedelta(minutes=20), now), user])
    connection = FakeConnection(cursor)
    monkeypatch.setattr(auth_helpers, "get_db", database(connection))
    monkeypatch.setattr(
        auth_helpers,
        "decode_token",
        lambda _token: {"sub": "17", "sid": session_id, "type": "access"},
    )

    assert auth_helpers.get_current_user(credentials()) == user
    assert connection.exited_cleanly is True
    assert all("update refresh_tokens" not in sql for sql, _params in cursor.calls)
    assert any("max(last_activity_at)" in sql for sql, _params in cursor.calls)


def test_idle_protected_request_commits_current_session_revocation(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "120")
    session_id = str(uuid4())
    now = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    cursor = ScriptedCursor(one=[session_row(now - timedelta(minutes=120), now)])
    connection = FakeConnection(cursor)
    monkeypatch.setattr(auth_helpers, "get_db", database(connection))
    monkeypatch.setattr(
        auth_helpers,
        "decode_token",
        lambda _token: {"sub": "17", "sid": session_id, "type": "access"},
    )

    with pytest.raises(HTTPException) as exc_info:
        auth_helpers.get_current_user(credentials())

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["code"] == "session_idle_timeout"
    assert connection.exited_cleanly is True
    assert any(
        "where user_id = %s and session_id = %s and is_revoked = false" in sql and params == (17, session_id)
        for sql, params in cursor.calls
    )
    assert all(
        not (sql == "update refresh_tokens set is_revoked = true where user_id = %s")
        for sql, _params in cursor.calls
    )


def test_activity_endpoint_is_the_only_path_that_extends_the_session(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "120")
    session_id = str(uuid4())
    previous = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)
    now = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    cursor = ScriptedCursor(
        one=[session_row(previous, now)],
        many=[[{"last_activity_at": now}, {"last_activity_at": now}]],
    )
    connection = FakeConnection(cursor)
    monkeypatch.setattr(auth_routes, "get_db", database(connection))
    monkeypatch.setattr(
        auth_routes,
        "decode_token",
        lambda _token: {"sub": "17", "sid": session_id, "type": "access"},
    )

    response = auth_routes.record_activity({"id": 17}, credentials())

    assert response.last_activity_at == now
    assert response.idle_expires_at == now + timedelta(minutes=120)
    assert response.idle_timeout_seconds == 7_200
    update = next(sql for sql, _params in cursor.calls if sql.startswith("update refresh_tokens"))
    assert "set last_activity_at = now()" in update
    assert "is_revoked = false" not in update


def test_refresh_rotation_preserves_activity_and_does_not_extend_idle_deadline(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "120")
    user_id = 29
    session_id = str(uuid4())
    last_activity_at = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)
    server_now = last_activity_at + timedelta(minutes=40)
    cursor = ScriptedCursor(
        one=[session_row(last_activity_at, server_now)],
        many=[
            [
                {
                    "id": 51,
                    "token_hash": "stored",
                    "is_revoked": False,
                    "last_activity_at": last_activity_at,
                    "server_now": server_now,
                }
            ]
        ],
    )
    connection = FakeConnection(cursor)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(auth_routes, "get_db", database(connection))
    monkeypatch.setattr(auth_routes, "_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        auth_routes,
        "decode_token",
        lambda _token: {"sub": str(user_id), "sid": session_id, "type": "refresh"},
    )
    monkeypatch.setattr(auth_routes, "verify_refresh_token", lambda *_args: True)
    monkeypatch.setattr(auth_routes, "create_access_token", lambda *_args: "new-access")

    def create_refresh(*_args, **kwargs):
        captured.update(kwargs)
        return "new-refresh"

    monkeypatch.setattr(auth_routes, "create_refresh_token", create_refresh)

    response = auth_routes.refresh_token(
        auth_routes.RefreshRequest(refresh_token="old-refresh"),
        request(),
    )

    assert response.access_token == "new-access"
    assert response.refresh_token == "new-refresh"
    assert response.last_activity_at == last_activity_at
    assert response.idle_expires_at == last_activity_at + timedelta(minutes=120)
    assert captured["last_activity_at"] == last_activity_at
    assert all("set last_activity_at" not in sql for sql, _params in cursor.calls)


def test_refresh_at_idle_boundary_revokes_only_that_session(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "120")
    user_id = 29
    session_id = str(uuid4())
    last_activity_at = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
    server_now = last_activity_at + timedelta(minutes=120)
    cursor = ScriptedCursor(
        one=[session_row(last_activity_at, server_now)],
        many=[
            [
                {
                    "id": 51,
                    "token_hash": "stored",
                    "is_revoked": False,
                    "last_activity_at": last_activity_at,
                    "server_now": server_now,
                }
            ]
        ],
    )
    connection = FakeConnection(cursor)
    monkeypatch.setattr(auth_routes, "get_db", database(connection))
    monkeypatch.setattr(auth_routes, "_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        auth_routes,
        "decode_token",
        lambda _token: {"sub": str(user_id), "sid": session_id, "type": "refresh"},
    )
    monkeypatch.setattr(auth_routes, "verify_refresh_token", lambda *_args: True)

    with pytest.raises(HTTPException) as exc_info:
        auth_routes.refresh_token(
            auth_routes.RefreshRequest(refresh_token="old-refresh"),
            request(),
        )

    assert exc_info.value.detail["code"] == "session_idle_timeout"
    assert connection.exited_cleanly is True
    assert any(params == (user_id, session_id) for _sql, params in cursor.calls)
    assert all(
        not (sql == "update refresh_tokens set is_revoked = true where user_id = %s")
        for sql, _params in cursor.calls
    )


def test_retrying_an_idle_revoked_token_does_not_revoke_other_sessions(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "120")
    user_id = 29
    session_id = str(uuid4())
    last_activity_at = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
    server_now = last_activity_at + timedelta(minutes=121)
    cursor = ScriptedCursor(
        many=[
            [
                {
                    "id": 51,
                    "token_hash": "stored",
                    "is_revoked": True,
                    "last_activity_at": last_activity_at,
                    "server_now": server_now,
                }
            ]
        ]
    )
    connection = FakeConnection(cursor)
    monkeypatch.setattr(auth_routes, "get_db", database(connection))
    monkeypatch.setattr(auth_routes, "_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        auth_routes,
        "decode_token",
        lambda _token: {"sub": str(user_id), "sid": session_id, "type": "refresh"},
    )
    monkeypatch.setattr(auth_routes, "verify_refresh_token", lambda *_args: True)

    with pytest.raises(HTTPException) as exc_info:
        auth_routes.refresh_token(
            auth_routes.RefreshRequest(refresh_token="idle-refresh"),
            request(),
        )

    assert exc_info.value.detail["code"] == "session_idle_timeout"
    assert all(
        not (sql == "update refresh_tokens set is_revoked = true where user_id = %s")
        for sql, _params in cursor.calls
    )


def test_login_and_google_exchange_expose_server_idle_metadata(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "120")
    monkeypatch.setattr(auth_routes, "_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        auth_routes,
        "authenticate_user",
        lambda *_args: {"id": 7, "is_active": True},
    )
    monkeypatch.setattr(auth_routes, "create_access_token", lambda *_args: "access")
    monkeypatch.setattr(auth_routes, "create_refresh_token", lambda *_args, **_kwargs: "refresh")

    login = auth_routes.login(
        auth_routes.LoginRequest(username="reader", password="password123"),
        request(),
    )
    assert login.idle_timeout_seconds == 7_200
    assert login.idle_expires_at - login.last_activity_at == timedelta(minutes=120)

    class RedisPayload:
        def getdel(self, _key: str) -> str:
            return json.dumps(
                {
                    "access_token": "google-access",
                    "refresh_token": "google-refresh",
                    "session_id": str(uuid4()),
                    "last_activity_at": "2026-07-15T10:00:00+00:00",
                    "idle_expires_at": "2026-07-15T12:00:00+00:00",
                    "idle_timeout_seconds": 7_200,
                }
            )

    monkeypatch.setattr(auth_routes, "_get_rl_client", lambda: RedisPayload())
    google = auth_routes.google_get_token("one-time-key")
    assert google["idle_timeout_seconds"] == 7_200
    assert google["idle_expires_at"] == "2026-07-15T12:00:00+00:00"


def test_session_listing_reports_activity_and_deadline(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_IDLE_TIMEOUT_MINUTES", "120")
    session_id = uuid4()
    created_at = datetime(2026, 7, 15, 8, 30, tzinfo=UTC)
    last_activity_at = datetime(2026, 7, 15, 9, 45, tzinfo=UTC)
    cursor = ScriptedCursor(
        many=[
            [
                {
                    "session_id": session_id,
                    "created_at": created_at,
                    "last_activity_at": last_activity_at,
                }
            ]
        ]
    )
    monkeypatch.setattr(auth_routes, "get_db", database(FakeConnection(cursor)))

    sessions = auth_routes.list_sessions({"id": 7})

    assert sessions == [
        {
            "session_id": str(session_id),
            "created_at": created_at.isoformat(),
            "last_activity_at": last_activity_at.isoformat(),
            "idle_expires_at": (last_activity_at + timedelta(minutes=120)).isoformat(),
            "idle_timeout_seconds": 7_200,
        }
    ]
    assert cursor.calls[0][1] == (7, 120)
