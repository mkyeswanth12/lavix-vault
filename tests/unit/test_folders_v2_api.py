from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi import HTTPException

from app.routers import folders


class FolderDeleteCursor:
    def __init__(self) -> None:
        self.query = ""
        self.params = None
        self.executions: list[tuple[str, object]] = []
        self.rowcount = 0

    def execute(self, query, params=None) -> None:
        self.query = " ".join(query.split())
        self.params = params
        self.executions.append((self.query, params))
        if self.query.startswith("UPDATE ingestion_jobs"):
            self.rowcount = 2
        elif self.query.startswith("DELETE FROM document_revisions"):
            self.rowcount = 3
        else:
            self.rowcount = 0

    def fetchone(self):
        if "FROM folders" in self.query and "FOR UPDATE" in self.query:
            return {"id": 10, "name": "Root", "parent_id": None}
        return None

    def fetchall(self):
        if "WITH RECURSIVE descendants" in self.query:
            return [{"id": 10}, {"id": 11}]
        if self.query.startswith("SELECT id FROM files"):
            return [{"id": 20}, {"id": 21}]
        return []


class FakeConnection:
    def __init__(self, cursor: FolderDeleteCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> FolderDeleteCursor:
        return self._cursor


def test_folder_delete_preserves_index_data_and_trashes_tree_atomically(monkeypatch) -> None:
    cursor = FolderDeleteCursor()

    @contextmanager
    def fake_db():
        yield FakeConnection(cursor)

    monkeypatch.setattr(folders, "get_db", fake_db)
    result = folders.delete_folder(10, {"id": 7, "perm_folders": True})

    assert result == {
        "message": "Folder moved to trash",
        "folder_ids": [10, 11],
        "file_count": 2,
        "cancelled_jobs": 2,
        "removed_revisions": 0,
        "index_data_preserved": True,
    }
    statements = [query for query, _params in cursor.executions]
    assert any(query.startswith("UPDATE ingestion_jobs") for query in statements)
    assert not any(query.startswith("DELETE FROM document_revisions") for query in statements)
    file_update = next(query for query in statements if query.startswith("UPDATE files"))
    assert "SET user_granted_ai_access = FALSE" not in file_update
    assert "current_revision = NULL" not in file_update
    assert "WHEN current_revision IS NOT NULL THEN 'ready'" in file_update
    assert "is_deleted = TRUE" in file_update
    assert cursor.executions[-1][1] == (7, [10, 11])


def test_folder_delete_permission_is_checked_before_database_access(monkeypatch) -> None:
    monkeypatch.setattr(
        folders,
        "get_db",
        lambda: pytest.fail("database must not be opened when permission is denied"),
    )
    with pytest.raises(HTTPException) as exc_info:
        folders.delete_folder(10, {"id": 7, "perm_folders": False})
    assert exc_info.value.status_code == 403


def test_folder_names_reject_whitespace_only_values() -> None:
    with pytest.raises(HTTPException) as exc_info:
        folders._folder_name("   ")
    assert exc_info.value.status_code == 422


def test_folder_name_conflict_query_never_binds_an_untyped_null_exclusion() -> None:
    cursor = FolderDeleteCursor()

    assert not folders._name_conflicts(
        cursor,
        user_id=7,
        name="Root",
        parent_id=None,
    )
    assert cursor.params == (7, "Root", None)
    assert "id <>" not in cursor.query

    assert not folders._name_conflicts(
        cursor,
        user_id=7,
        name="Root",
        parent_id=None,
        exclude_id=10,
    )
    assert cursor.params == (7, "Root", None, 10)
    assert "AND id <> %s" in cursor.query
