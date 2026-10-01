from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ingestion.commands import (
    BulkEnqueueResult,
    CancellationResult,
    EnqueueResult,
)
from app.ingestion.models import JobState
from app.routers import auth as auth_routes
from app.routers import files
from app.routers.chat import session_routes


class FakeConnection:
    def __init__(self, cursor) -> None:
        self.cursor_value = cursor

    def cursor(self):
        return self.cursor_value


def database(connection: FakeConnection):
    @contextmanager
    def get_db():
        yield connection

    return get_db


class PreviewCursor:
    def __init__(self, row=None) -> None:
        self.row = row
        self.sql = ""
        self.params = None

    def execute(self, sql, params=None) -> None:
        self.sql = " ".join(sql.split())
        self.params = params

    def fetchone(self):
        return self.row


def test_preview_capability_rejects_invalid_revoked_and_changed_client(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(files.router, prefix="/api/files")
    client = TestClient(app)

    monkeypatch.setattr(
        files,
        "get_db",
        lambda: (_ for _ in ()).throw(AssertionError("invalid tokens must not reach the database")),
    )
    invalid = client.get("/api/files/preview/not-a-uuid")
    assert invalid.status_code == 404

    revoked_cursor = PreviewCursor()
    monkeypatch.setattr(files, "get_db", database(FakeConnection(revoked_cursor)))
    revoked = client.get(f"/api/files/preview/{uuid4()}")
    assert revoked.status_code == 404
    assert "u.is_active = TRUE AND u.perm_download = TRUE" in revoked_cursor.sql
    assert "s.purpose = 'preview'" in revoked_cursor.sql
    assert "s.expires_at > NOW()" in revoked_cursor.sql

    monkeypatch.setenv("VALIDATE_PREVIEW_IP", "true")
    changed_client_cursor = PreviewCursor({"client_ip": "203.0.113.9"})
    monkeypatch.setattr(files, "get_db", database(FakeConnection(changed_client_cursor)))
    changed_client = client.get(f"/api/files/preview/{uuid4()}")
    assert changed_client.status_code == 403
    assert changed_client.json()["detail"] == "Preview session client changed"


def test_preview_capability_supports_bounded_http_ranges(monkeypatch, tmp_path) -> None:
    payload = b"%PDF-1.7\nfirst-page-bytes\n%%EOF"
    materialized = tmp_path / "preview.pdf"
    materialized.write_bytes(payload)

    class ObjectAccess:
        def __init__(self) -> None:
            self.cleaned = []

        async def materialize(self, _ref):
            return materialized

        def cleanup(self, path) -> None:
            self.cleaned.append(path)

    object_access = ObjectAccess()
    row = {
        "filename": "preview.pdf",
        "mime_type": "application/pdf",
        "s3_bucket_name": "bucket",
        "s3_path": "payload.enc",
        "s3_key_path": "key.enc",
        "sha256_hash": "",
        "client_ip": None,
    }
    monkeypatch.setattr(files, "get_db", database(FakeConnection(PreviewCursor(row))))
    monkeypatch.setattr(files, "object_access", object_access)

    app = FastAPI()
    app.include_router(files.router, prefix="/api/files")
    response = TestClient(app).get(
        f"/api/files/preview/{uuid4()}",
        headers={"Range": "bytes=0-8"},
    )

    assert response.status_code == 206
    assert response.content == payload[:9]
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-range"] == f"bytes 0-8/{len(payload)}"
    assert object_access.cleaned == [materialized]


class RefreshReuseCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql, params=None) -> None:
        self.calls.append((" ".join(sql.lower().split()), params))

    def fetchall(self):
        return [{"id": 9, "token_hash": "stored-hash", "is_revoked": True}]

    def fetchone(self):
        # No unrevoked successor exists: a genuine replay with nothing to
        # inherit, so the global revoke path must still fire.
        return None


def test_refresh_token_reuse_revokes_every_session_for_the_user(monkeypatch) -> None:
    user_id = 41
    session_id = str(uuid4())
    cursor = RefreshReuseCursor()
    monkeypatch.setattr(auth_routes, "get_db", database(FakeConnection(cursor)))
    monkeypatch.setattr(auth_routes, "_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        auth_routes,
        "decode_token",
        lambda _token: {"sub": str(user_id), "sid": session_id, "type": "refresh"},
    )
    monkeypatch.setattr(auth_routes, "verify_refresh_token", lambda *_args: True)

    app = FastAPI()
    app.include_router(auth_routes.router, prefix="/api/auth")
    response = TestClient(app).post("/api/auth/refresh", json={"refresh_token": "reused-token"})

    assert response.status_code == 401
    assert "Token reuse detected" in response.json()["detail"]
    assert any(
        sql == "update refresh_tokens set is_revoked = true where user_id = %s" and params == (user_id,)
        for sql, params in cursor.calls
    )


class FakeBulkCommands:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def grant_and_enqueue_all(self, user_id: int, *, idempotency_key: str) -> BulkEnqueueResult:
        self.calls.append(("enable", user_id, idempotency_key))
        return BulkEnqueueResult(
            items=(
                EnqueueResult(7, 2, "job-7", JobState.QUEUED, True),
                EnqueueResult(8, 1, "job-8", JobState.READY, False),
            ),
            skipped_unsupported=2,
        )

    def cancel_indexing(self, user_id: int) -> CancellationResult:
        self.calls.append(("cancel", user_id))
        return CancellationResult(file_count=2, cancelled_job_count=3, ready_file_count=1)

    def cancel_file_indexing(self, user_id: int, file_id: int) -> CancellationResult:
        self.calls.append(("cancel-file", user_id, file_id))
        return CancellationResult(file_count=1, cancelled_job_count=1, ready_file_count=1)


def test_bulk_index_wrappers_preserve_scope_idempotency_and_ai_permission(monkeypatch) -> None:
    commands = FakeBulkCommands()
    monkeypatch.setattr(files, "ingestion_commands", commands)
    app = FastAPI()
    app.include_router(files.router, prefix="/api/files")
    app.dependency_overrides[files.get_current_user] = lambda: {"id": 73, "perm_ai": True}
    client = TestClient(app)

    disabled = client.post("/api/files/disable-all-embeddings")
    assert disabled.status_code == 200
    assert disabled.json() == {
        "message": "Deprecated endpoint: unfinished indexing cancelled; published index data was preserved.",
        "deprecated": True,
        "replacement": "/api/files/cancel-indexing",
        "affected_files": 2,
        "cancelled_jobs": 3,
        "ready_files_preserved": 1,
        "consent_preserved": True,
        "index_data_preserved": True,
    }

    enabled = client.post(
        "/api/files/enable-all-embeddings",
        headers={"Idempotency-Key": "browser-batch"},
    )
    assert enabled.status_code == 200
    assert enabled.json()["created_count"] == 1
    assert enabled.json()["count"] == 1
    assert enabled.json()["skipped_unsupported"] == 2
    assert enabled.json()["items"] == [
        {"file_id": 7, "revision": 2, "job_id": "job-7", "state": "queued", "created": True},
        {"file_id": 8, "revision": 1, "job_id": "job-8", "state": "ready", "created": False},
    ]
    assert commands.calls == [("cancel", 73), ("enable", 73, "grant-all:browser-batch")]

    cancelled = client.post("/api/files/cancel-indexing")
    assert cancelled.status_code == 200
    assert cancelled.json() == {
        "message": "Unfinished indexing cancelled; published index data was preserved.",
        "affected_files": 2,
        "cancelled_jobs": 3,
        "ready_files_preserved": 1,
        "consent_preserved": True,
        "index_data_preserved": True,
    }

    cancelled_file = client.post("/api/files/cancel-indexing/7")
    assert cancelled_file.status_code == 200
    assert cancelled_file.json() == {
        "message": "Unfinished indexing cancelled; published index data was preserved.",
        "file_id": 7,
        "affected_files": 1,
        "cancelled_jobs": 1,
        "ready_files_preserved": 1,
        "consent_preserved": True,
        "index_data_preserved": True,
    }

    app.dependency_overrides[files.get_current_user] = lambda: {"id": 73, "perm_ai": False}
    denied = client.post("/api/files/enable-all-embeddings")
    assert denied.status_code == 403
    safety_cancel = client.post("/api/files/cancel-indexing")
    assert safety_cancel.status_code == 200
    assert commands.calls == [
        ("cancel", 73),
        ("enable", 73, "grant-all:browser-batch"),
        ("cancel", 73),
        ("cancel-file", 73, 7),
        ("cancel", 73),
    ]


class HistoryCursor:
    def __init__(self, chat_id: str) -> None:
        self.chat_id = chat_id
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql, params=None) -> None:
        self.calls.append((" ".join(sql.lower().split()), params))

    def fetchone(self):
        return {"id": self.chat_id, "messages": []}


def test_chat_clear_and_revoke_http_routes_keep_mutations_owner_scoped(monkeypatch) -> None:
    chat_id = str(uuid4())
    cursors: list[HistoryCursor] = []

    @contextmanager
    def get_db():
        cursor = HistoryCursor(chat_id)
        cursors.append(cursor)
        yield FakeConnection(cursor)

    monkeypatch.setattr(session_routes, "get_db", get_db)
    app = FastAPI()
    app.include_router(session_routes.router, prefix="/api/ai")
    app.dependency_overrides[session_routes.require_ai_permission] = lambda: {
        "id": 73,
        "perm_ai": True,
    }
    client = TestClient(app)

    cleared = client.delete(f"/api/ai/chat/history/{chat_id}")
    assert cleared.status_code == 200
    assert cleared.json()["history_cleared"] is True

    assert len(cursors) == 1
    for cursor in cursors:
        assert cursor.calls[0][1] == (chat_id, 73)
        assert all(params == (chat_id, 73) for _sql, params in cursor.calls)
        mutation_sql = " ".join(sql for sql, _params in cursor.calls[1:])
        assert "delete from chat_messages" in mutation_sql
        assert "update chats" in mutation_sql
        assert "document_chunks" not in mutation_sql
        assert "files" not in mutation_sql
