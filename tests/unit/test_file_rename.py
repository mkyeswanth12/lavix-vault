"""Regression tests for BUG-008: PATCH /api/files/rename/{file_id}.

Display-name-only rename: ownership + sibling-conflict fenced, stored
object name / MIME / revision identity untouched, perm_rename gated.
"""
from __future__ import annotations

from contextlib import contextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import files
from app.routers.files import RenameFileRequest


class ScriptedCursor:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append((query, params))

    def fetchone(self):
        return self.results.pop(0) if self.results else None


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _client(monkeypatch, cursor, user=None):
    @contextmanager
    def fake_db():
        yield FakeConnection(cursor)

    monkeypatch.setattr(files, "get_db", fake_db)
    app = FastAPI()
    app.include_router(files.router, prefix="/api/files")
    app.dependency_overrides[files.get_current_user] = lambda: (
        user or {"id": 5, "perm_rename": True}
    )
    return TestClient(app)


def test_rename_happy_path_updates_display_name_only(monkeypatch):
    cursor = ScriptedCursor([{"folder_id": 11}, None])
    client = _client(monkeypatch, cursor)
    response = client.patch("/api/files/rename/305", json={"filename": "renamed.pdf"})
    assert response.status_code == 200, response.text
    assert response.json() == {"message": "File renamed", "file_id": 305, "filename": "renamed.pdf"}
    updates = [c for c in cursor.calls if c[0].strip().upper().startswith("UPDATE")]
    assert len(updates) == 1
    assert "display_name" in updates[0][0]
    assert "original_filename" not in updates[0][0]
    assert updates[0][1] == ("renamed.pdf", 305, 5)


def test_rename_requires_permission(monkeypatch):
    cursor = ScriptedCursor([])
    client = _client(monkeypatch, cursor, user={"id": 5, "perm_rename": False})
    response = client.patch("/api/files/rename/305", json={"filename": "x.pdf"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Rename permission denied"


def test_rename_missing_file_404(monkeypatch):
    cursor = ScriptedCursor([None])
    client = _client(monkeypatch, cursor)
    assert client.patch("/api/files/rename/999", json={"filename": "x.pdf"}).status_code == 404


def test_rename_sibling_conflict_409(monkeypatch):
    cursor = ScriptedCursor([{"folder_id": 11}, {"id": 306}])
    client = _client(monkeypatch, cursor)
    assert client.patch("/api/files/rename/305", json={"filename": "invoice.pdf"}).status_code == 409


def test_rename_rejects_empty_name(monkeypatch):
    cursor = ScriptedCursor([])
    client = _client(monkeypatch, cursor)
    assert client.patch("/api/files/rename/305", json={"filename": ""}).status_code == 422


def test_rename_schema_bounds():
    assert RenameFileRequest(filename="a.pdf").filename == "a.pdf"


def test_delete_denied_names_admin_grant_path(monkeypatch):
    """BUG-011: the 403 must stay a denial but tell the user what to do."""
    cursor = ScriptedCursor([])
    client = _client(monkeypatch, cursor, user={"id": 5})
    response = client.delete("/api/files/delete/305")
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail.startswith("Delete permission denied.")
    assert "administrator" in detail
    assert "Permissions" in detail
