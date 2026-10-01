from __future__ import annotations

import inspect
from contextlib import contextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ingestion.commands import EnqueueResult, IngestionStatus
from app.ingestion.models import JobState
from app.routers import files


class FakeCommands:
    def __init__(self) -> None:
        self.grants = []

    def grant_and_enqueue(self, user_id, file_id, **kwargs):
        self.grants.append((user_id, file_id, kwargs))
        return EnqueueResult(file_id, 2, "job-2", JobState.QUEUED, True)

    def status(self, user_id, file_id):
        return IngestionStatus(
            file_id=file_id,
            consent_granted=True,
            desired_revision=2,
            current_revision=None,
            state="embedding",
            job_id="job-2",
            chunk_count=0,
        )


def _client(monkeypatch, commands=None, user=None) -> TestClient:
    monkeypatch.setattr(files, "ingestion_commands", commands or FakeCommands())
    app = FastAPI()
    app.include_router(files.router, prefix="/api/files")
    app.dependency_overrides[files.get_current_user] = lambda: (
        user
        or {
            "id": 5,
            "perm_ai": True,
            "perm_download": True,
        }
    )
    return TestClient(app)


def test_grant_and_status_preserve_routes_while_exposing_v2_state(monkeypatch) -> None:
    commands = FakeCommands()
    client = _client(monkeypatch, commands)
    response = client.post(
        "/api/files/grant-ai-access/17",
        headers={"Idempotency-Key": "browser-request"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "message": "AI access granted; durable ingestion is queued.",
        "file_id": 17,
        "revision": 2,
        "job_id": "job-2",
        "state": "queued",
        "status": "processing",
        "created": True,
    }
    assert commands.grants == [(5, 17, {"idempotency_key": "grant:17:browser-request"})]

    status = client.get("/api/files/ai-status/17")
    assert status.status_code == 200
    assert status.json()["state"] == "embedding"
    assert status.json()["status"] == "processing"
    assert status.json()["chunks_indexed"] == 0


def test_status_reports_published_embeddings_during_reindex_or_failure() -> None:
    for state in ("embedding", "failed"):
        payload = files._status_payload(
            IngestionStatus(
                file_id=17,
                consent_granted=True,
                desired_revision=3,
                current_revision=2,
                state=state,
                job_id="job-3",
                chunk_count=9,
            )
        )
        assert payload["state"] == state
        assert payload["embeddings_ready"] is True

    revoked = files._status_payload(
        IngestionStatus(
            file_id=17,
            consent_granted=False,
            desired_revision=3,
            current_revision=2,
            state="not_granted",
            job_id=None,
            chunk_count=9,
        )
    )
    assert revoked["embeddings_ready"] is False


def test_grant_ai_access_fails_closed_without_ai_permission(monkeypatch) -> None:
    commands = FakeCommands()
    client = _client(monkeypatch, commands, user={"id": 5, "perm_ai": False})

    response = client.post("/api/files/grant-ai-access/17")

    assert response.status_code == 403
    assert response.json()["detail"] == "AI access permission denied"
    assert commands.grants == []


class ScriptedCursor:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.current = None
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append((query, params))
        self.current = self.results.pop(0)

    def fetchall(self):
        return list(self.current)

    def fetchone(self):
        return self.current


class FakeConnection:
    def __init__(self, cursor) -> None:
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def test_list_maps_granular_worker_state_without_losing_compatibility(monkeypatch) -> None:
    now = datetime.now(UTC)
    row = {
        "id": 7,
        "original_filename": "report.docx",
        "filename": "Quarterly report.docx",
        "file_size_bytes": 100,
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "uploaded_at": now,
        "quick_tags": ["report"],
        "quick_summary": "summary",
        "doc_type": "report",
        "intelligence_status": "model",
        "user_granted_ai_access": True,
        "ai_status": "embedding",
        "desired_revision": 2,
        "current_revision": None,
        "ai_error_code": None,
        "ai_error_detail": None,
        "is_deleted": False,
        "deleted_at": None,
        "folder_id": None,
    }
    cursor = ScriptedCursor([[row], {"storage_quota_bytes": 1000, "storage_used_bytes": 100}])

    @contextmanager
    def fake_db():
        yield FakeConnection(cursor)

    monkeypatch.setattr(files, "get_db", fake_db)
    payload = files.list_files(user={"id": 5}, show_deleted=False)
    assert payload["files"][0]["ai_status"] == "processing"
    assert payload["files"][0]["ingestion_state"] == "embedding"
    assert payload["files"][0]["doc_type"] == "report"
    assert payload["files"][0]["intelligence_status"] == "model"
    assert payload["storage_quota_bytes"] == 1000


def test_list_hides_pending_filename_placeholders(monkeypatch) -> None:
    now = datetime.now(UTC)
    row = {
        "id": 8,
        "original_filename": "invoice-acme.pdf",
        "filename": "invoice-acme.pdf",
        "file_size_bytes": 100,
        "mime_type": "application/pdf",
        "uploaded_at": now,
        "quick_tags": ["invoice", "acme"],
        "quick_summary": "File: invoice-acme.pdf",
        "doc_type": None,
        "intelligence_status": "pending",
        "user_granted_ai_access": True,
        "ai_status": "ready",
        "desired_revision": 1,
        "current_revision": 1,
        "ai_error_code": None,
        "ai_error_detail": None,
        "is_deleted": False,
        "deleted_at": None,
        "folder_id": None,
    }
    cursor = ScriptedCursor([[row], {"storage_quota_bytes": 1000, "storage_used_bytes": 100}])

    @contextmanager
    def fake_db():
        yield FakeConnection(cursor)

    monkeypatch.setattr(files, "get_db", fake_db)
    item = files.list_files(user={"id": 5}, show_deleted=False)["files"][0]

    assert item["ai_searchable"] is True
    assert item["intelligence_status"] == "pending"
    assert item["tags"] == []
    assert item["summary"] is None
    assert item["doc_type"] is None


def test_list_keeps_published_intelligence_visible_during_retry_or_failure(monkeypatch) -> None:
    now = datetime.now(UTC)
    base = {
        "original_filename": "report.pdf",
        "filename": "report.pdf",
        "file_size_bytes": 100,
        "mime_type": "application/pdf",
        "uploaded_at": now,
        "quick_tags": ["revenue", "quarterly"],
        "quick_summary": "The published revision contains a quarterly revenue report.",
        "doc_type": "report",
        "intelligence_status": "model",
        "user_granted_ai_access": True,
        "desired_revision": 3,
        "current_revision": 2,
        "ai_error_code": None,
        "ai_error_detail": None,
        "is_deleted": False,
        "deleted_at": None,
        "folder_id": None,
    }
    rows = [
        {**base, "id": 9, "ai_status": "embedding"},
        {
            **base,
            "id": 10,
            "ai_status": "failed",
            "ai_error_code": "embedding_failed",
        },
    ]
    cursor = ScriptedCursor([rows, {"storage_quota_bytes": 1000, "storage_used_bytes": 200}])

    @contextmanager
    def fake_db():
        yield FakeConnection(cursor)

    monkeypatch.setattr(files, "get_db", fake_db)
    items = files.list_files(user={"id": 5}, show_deleted=False)["files"]

    assert [item["ingestion_state"] for item in items] == ["embedding", "failed"]
    for item in items:
        assert item["ai_searchable"] is True
        assert item["summary"] == base["quick_summary"]
        assert item["tags"] == base["quick_tags"]
        assert item["doc_type"] == "report"
        assert item["intelligence_status"] == "model"





def test_files_router_has_no_legacy_vector_or_background_ingestion_dependency() -> None:
    source = inspect.getsource(files).lower()
    for forbidden in (
        "chroma",
        "ai_embeddings",
        "redis_service",
        "process_file_for_embeddings",
        "asyncio.create_task(",
        "embeddings_generated",
        "ai_access_granted_at",
        "last_ai_access",
    ):
        assert forbidden not in source


def test_active_document_media_is_never_served_as_executable_content() -> None:
    for media_type in (
        "text/html",
        "Text/HTML; charset=utf-8",
        "application/xhtml+xml",
        "application/xml",
        "text/xml",
        "image/svg+xml",
    ):
        assert files._safe_served_media_type(media_type) == "text/plain; charset=utf-8"
    assert files._safe_served_media_type("application/pdf") == "application/pdf"
    assert files._safe_served_media_type(None) == "application/octet-stream"
    assert files._safe_served_media_type("application/octet-stream", "UPPER.PDF") == "application/pdf"
    assert files._safe_served_media_type("application/octet-stream", "not-a-pdf.bin") == "application/octet-stream"


def test_restore_reconciles_status_to_the_preserved_current_revision(monkeypatch) -> None:
    class RestoreCursor:
        def __init__(self) -> None:
            self.calls = []
            self.index = 0

        def execute(self, query, params=None):
            self.calls.append((" ".join(query.split()), params))
            self.index += 1

        def fetchone(self):
            if self.index == 1:
                return {"file_size_bytes": 100, "folder_id": None, "filename": "report.pdf"}
            if self.index == 2:
                return {"storage_quota_bytes": 1000, "storage_used_bytes": 100}
            return None

    cursor = RestoreCursor()

    @contextmanager
    def fake_db():
        yield FakeConnection(cursor)

    monkeypatch.setattr(files, "get_db", fake_db)
    assert files.restore_file(7, {"id": 5}) == {"status": "restored"}
    update = cursor.calls[-1][0]
    assert "WHEN current_revision IS NOT NULL THEN 'ready'" in update
    assert "user_granted_ai_access = FALSE THEN 'not_granted'" in update
    assert "current_revision = NULL" not in update


def test_bulk_revoke_reports_per_file_success_and_failure(monkeypatch) -> None:
    from app.ingestion.commands import RemovalResult

    class BulkCommands(FakeCommands):
        def revoke_and_remove(self, user_id, file_id):
            if file_id == 9:
                raise FileNotFoundError("file not found")
            return RemovalResult(1, 0, 1, 3)

    client = _client(monkeypatch, BulkCommands())
    response = client.post("/api/files/revoke-ai-access/bulk", json={"file_ids": [7, 9, 7]})
    assert response.status_code == 200
    body = response.json()
    assert body["revoked_count"] == 1
    assert body["failed_count"] == 1
    assert body["items"] == [
        {"file_id": 7, "status": "locked", "cancelled_jobs": 0, "removed_revisions": 1, "removed_chunks": 3}
    ]
    assert body["failed"] == [{"file_id": 9, "error": "File not found"}]


def test_bulk_revoke_rejects_empty_and_oversized_lists(monkeypatch) -> None:
    client = _client(monkeypatch)
    assert client.post("/api/files/revoke-ai-access/bulk", json={"file_ids": []}).status_code == 422
    assert client.post("/api/files/revoke-ai-access/bulk", json={"file_ids": list(range(1, 102))}).status_code == 422


def test_bulk_grant_reports_per_file_success_and_failure(monkeypatch) -> None:
    from app.ingestion.errors import UnsupportedFormatError

    class BulkGrantCommands(FakeCommands):
        def grant_and_enqueue(self, user_id, file_id, **kwargs):
            if file_id == 11:
                raise UnsupportedFormatError("no parser for media type")
            return super().grant_and_enqueue(user_id, file_id, **kwargs)

    client = _client(monkeypatch, BulkGrantCommands())
    response = client.post("/api/files/grant-ai-access/bulk", json={"file_ids": [17, 11]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["created_count"] == 1
    assert body["items"][0]["file_id"] == 17
    assert body["failed"] == [{"file_id": 11, "error": "no parser for media type"}]


def test_bulk_grant_requires_ai_permission(monkeypatch) -> None:
    client = _client(monkeypatch, user={"id": 5, "perm_ai": False})
    response = client.post("/api/files/grant-ai-access/bulk", json={"file_ids": [17]})
    assert response.status_code in (401, 403)


def test_list_exposes_intelligence_fallback_reason(monkeypatch) -> None:
    from datetime import UTC, datetime

    row = {
        "id": 7,
        "original_filename": "report.docx",
        "filename": "Quarterly report.docx",
        "file_size_bytes": 100,
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "uploaded_at": datetime.now(UTC),
        "quick_tags": ["report"],
        "quick_summary": "summary",
        "doc_type": "report",
        "intelligence_status": "fallback",
        "intelligence_fallback_reason": "invalid_model_response",
        "user_granted_ai_access": True,
        "ai_status": "ready",
        "desired_revision": 2,
        "current_revision": 2,
        "ai_error_code": None,
        "ai_error_detail": None,
        "is_deleted": False,
        "deleted_at": None,
        "folder_id": None,
    }
    cursor = ScriptedCursor([[row], {"storage_quota_bytes": 1000, "storage_used_bytes": 100}])

    @contextmanager
    def fake_db():
        yield FakeConnection(cursor)

    monkeypatch.setattr(files, "get_db", fake_db)
    payload = files.list_files(user={"id": 5}, show_deleted=False)
    assert payload["files"][0]["intelligence_status"] == "fallback"
    assert payload["files"][0]["intelligence_fallback_reason"] == "invalid_model_response"


def test_list_fallback_reason_defaults_to_none(monkeypatch) -> None:
    from datetime import UTC, datetime

    row = {
        "id": 7,
        "original_filename": "report.docx",
        "filename": "Quarterly report.docx",
        "file_size_bytes": 100,
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "uploaded_at": datetime.now(UTC),
        "quick_tags": ["report"],
        "quick_summary": "summary",
        "doc_type": "report",
        "intelligence_status": "model",
        "user_granted_ai_access": True,
        "ai_status": "ready",
        "desired_revision": 2,
        "current_revision": 2,
        "ai_error_code": None,
        "ai_error_detail": None,
        "is_deleted": False,
        "deleted_at": None,
        "folder_id": None,
    }
    cursor = ScriptedCursor([[row], {"storage_quota_bytes": 1000, "storage_used_bytes": 100}])

    @contextmanager
    def fake_db():
        yield FakeConnection(cursor)

    monkeypatch.setattr(files, "get_db", fake_db)
    payload = files.list_files(user={"id": 5}, show_deleted=False)
    assert payload["files"][0]["intelligence_fallback_reason"] is None


def test_list_exposes_parser_supported_flag(monkeypatch) -> None:
    from datetime import UTC, datetime

    def row(name, mime, index):
        return {
            "id": index,
            "original_filename": name,
            "filename": name,
            "file_size_bytes": 100,
            "mime_type": mime,
            "uploaded_at": datetime.now(UTC),
            "quick_tags": [],
            "quick_summary": None,
            "doc_type": None,
            "intelligence_status": "pending",
            "user_granted_ai_access": False,
            "ai_status": "not_granted",
            "desired_revision": 0,
            "current_revision": None,
            "ai_error_code": None,
            "ai_error_detail": None,
            "is_deleted": False,
            "deleted_at": None,
            "folder_id": None,
        }

    rows = [row("doc.pdf", "application/pdf", 1), row("archive.zip", "application/zip", 2)]
    cursor = ScriptedCursor([rows, {"storage_quota_bytes": 1000, "storage_used_bytes": 100}])

    @contextmanager
    def fake_db():
        yield FakeConnection(cursor)

    monkeypatch.setattr(files, "get_db", fake_db)
    payload = files.list_files(user={"id": 5}, show_deleted=False)
    by_id = {item["id"]: item for item in payload["files"]}
    assert by_id[1]["parser_supported"] is True
    assert by_id[2]["parser_supported"] is False
