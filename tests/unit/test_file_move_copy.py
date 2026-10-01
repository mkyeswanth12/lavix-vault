"""Dashboard bulk copy/move contract: file routes, folder picker list.

Covers the exact user flow that regressed in the field:
create nested folder -> it appears in the destination picker list
(GET /folders keeps parent_id) -> bulk copy/move files into it.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.routers import files, folders
from app.routers.files import MoveFileRequest
from app.routers.folders import FolderCreate


class ScriptedCursor:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, object]] = []

    def execute(self, query, params=None):
        self.calls.append((" ".join(str(query).split()), params))
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


def fake_db(cursor):
    @contextmanager
    def _db():
        yield FakeConnection(cursor)

    return _db


def _update_calls(cursor):
    return [params for sql, params in cursor.calls if "UPDATE files SET folder_id" in sql]


# ---------------------------------------------------------------------------
# PATCH /files/move/{file_id}
# ---------------------------------------------------------------------------


def test_move_file_to_folder_updates_folder_id(monkeypatch) -> None:
    cursor = ScriptedCursor(
        [
            {"id": 9},  # _require_folder: destination exists
            {"filename": "report.pdf"},  # source file
            None,  # no name collision
            None,  # UPDATE issues no fetch
        ]
    )
    monkeypatch.setattr(files, "get_db", fake_db(cursor))

    response = files.move_file(
        7, MoveFileRequest(folder_id=9), {"id": 2, "perm_folders": True}
    )

    assert response == {"message": "File moved", "file_id": 7, "folder_id": 9}
    assert _update_calls(cursor) == [(9, 7, 2)]


def test_move_file_to_root_skips_folder_lookup(monkeypatch) -> None:
    cursor = ScriptedCursor([{"filename": "report.pdf"}, None, None])
    monkeypatch.setattr(files, "get_db", fake_db(cursor))

    response = files.move_file(
        7, MoveFileRequest(folder_id=None), {"id": 2, "perm_folders": True}
    )

    assert response == {"message": "File moved", "file_id": 7, "folder_id": None}
    assert _update_calls(cursor) == [(None, 7, 2)]
    assert not [sql for sql, _ in cursor.calls if "FROM folders" in sql]


def test_move_file_rejects_unknown_folder(monkeypatch) -> None:
    cursor = ScriptedCursor([None])
    monkeypatch.setattr(files, "get_db", fake_db(cursor))

    with pytest.raises(HTTPException) as exc_info:
        files.move_file(7, MoveFileRequest(folder_id=999), {"id": 2, "perm_folders": True})

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Folder not found"


def test_move_file_rejects_name_collision(monkeypatch) -> None:
    cursor = ScriptedCursor([{"id": 9}, {"filename": "report.pdf"}, {"id": 11}])
    monkeypatch.setattr(files, "get_db", fake_db(cursor))

    with pytest.raises(HTTPException) as exc_info:
        files.move_file(7, MoveFileRequest(folder_id=9), {"id": 2, "perm_folders": True})

    assert exc_info.value.status_code == 409
    assert _update_calls(cursor) == []


def test_move_file_missing_source_is_404(monkeypatch) -> None:
    cursor = ScriptedCursor([{"id": 9}, None])
    monkeypatch.setattr(files, "get_db", fake_db(cursor))

    with pytest.raises(HTTPException) as exc_info:
        files.move_file(7, MoveFileRequest(folder_id=9), {"id": 2, "perm_folders": True})

    assert exc_info.value.status_code == 404


def test_move_file_requires_folder_permission(monkeypatch) -> None:
    cursor = ScriptedCursor([])
    monkeypatch.setattr(files, "get_db", fake_db(cursor))

    with pytest.raises(HTTPException) as exc_info:
        files.move_file(7, MoveFileRequest(folder_id=9), {"id": 2, "perm_folders": False})

    assert exc_info.value.status_code == 403
    assert cursor.calls == []


# ---------------------------------------------------------------------------
# POST /files/copy/{file_id}
# ---------------------------------------------------------------------------


class FakeObjectAccess:
    def __init__(self) -> None:
        self.copies = []
        self.removed = []

    async def copy(self, source, destination):
        self.copies.append((source, destination))
        return destination

    async def remove(self, ref):
        self.removed.append(ref)
        return 1


def _source_row(**overrides):
    row = {
        "original_filename": "report.pdf",
        "display_name": "",
        "mime_type": "application/pdf",
        "file_size_bytes": 100,
        "sha256_hash": "abc",
        "s3_bucket_name": "vault",
        "s3_path": "p.enc",
        "s3_key_path": "p.key",
    }
    row.update(overrides)
    return row


def _patch_copy_storage(monkeypatch, access):
    monkeypatch.setattr(files, "object_access", access)
    monkeypatch.setattr(
        files,
        "object_keys_for_file",
        lambda user_id, file_uuid: SimpleNamespace(payload="p.enc", wrapped_key="p.key"),
    )


async def test_copy_file_into_folder_returns_new_id(monkeypatch) -> None:
    access = FakeObjectAccess()
    _patch_copy_storage(monkeypatch, access)
    cursor = ScriptedCursor(
        [
            {"id": 9},  # destination folder pre-check
            _source_row(),  # source lookup
            None,  # advisory lock (no fetch)
            {"storage_quota_bytes": 10_000, "storage_used_bytes": 100},  # owner
            {"id": 9},  # destination folder re-check
            {"id": 7},  # source still present
            None,  # no name collision
            {"id": 99},  # INSERT ... RETURNING id
        ]
    )
    monkeypatch.setattr(files, "get_db", fake_db(cursor))

    response = await files.copy_file(
        7, MoveFileRequest(folder_id=9), {"id": 2, "perm_share": True}
    )

    assert response["file_id"] == 99
    assert response["message"] == "File copied"
    assert len(access.copies) == 1
    assert access.removed == []


async def test_copy_file_missing_source_never_touches_storage(monkeypatch) -> None:
    access = FakeObjectAccess()
    _patch_copy_storage(monkeypatch, access)
    cursor = ScriptedCursor([None])
    monkeypatch.setattr(files, "get_db", fake_db(cursor))

    with pytest.raises(HTTPException) as exc_info:
        await files.copy_file(7, MoveFileRequest(folder_id=9), {"id": 2, "perm_share": True})

    assert exc_info.value.status_code == 404
    assert access.copies == []


async def test_copy_file_over_quota_rejects_and_cleans_up(monkeypatch) -> None:
    access = FakeObjectAccess()
    _patch_copy_storage(monkeypatch, access)
    cursor = ScriptedCursor(
        [
            {"id": 9},
            _source_row(file_size_bytes=950),
            None,
            {"storage_quota_bytes": 1000, "storage_used_bytes": 100},
            {"id": 9},
            {"id": 7},
            None,
        ]
    )
    monkeypatch.setattr(files, "get_db", fake_db(cursor))

    with pytest.raises(HTTPException) as exc_info:
        await files.copy_file(7, MoveFileRequest(folder_id=9), {"id": 2, "perm_share": True})

    assert exc_info.value.status_code == 413
    assert len(access.removed) == 1


# ---------------------------------------------------------------------------
# Folder list contract behind the dashboard destination picker
# ---------------------------------------------------------------------------


def test_folder_list_keeps_nested_folder_with_parent(monkeypatch) -> None:
    rows = [
        {
            "id": 29,
            "name": "2",
            "parent_id": 9,
            "created_at": datetime(2026, 9, 27, tzinfo=UTC),
            "file_count": 0,
            "total_size_bytes": 0,
        },
        {
            "id": 9,
            "name": "images",
            "parent_id": None,
            "created_at": datetime(2026, 9, 27, tzinfo=UTC),
            "file_count": 0,
            "total_size_bytes": 0,
        },
    ]
    cursor = ScriptedCursor([rows])
    monkeypatch.setattr(folders, "get_db", fake_db(cursor))

    listed = folders.list_folders({"id": 2})

    nested = next(item for item in listed if item["id"] == 29)
    assert nested["name"] == "2"
    assert nested["parent_id"] == 9


def test_create_nested_folder_returns_parent_for_picker_path(monkeypatch) -> None:
    cursor = ScriptedCursor(
        [
            {"id": 9, "name": "images", "parent_id": None},  # parent exists
            None,  # no name conflict
            None,  # no deleted folder to resurrect
            {"id": 29, "name": "2", "parent_id": 9, "created_at": datetime(2026, 9, 27, tzinfo=UTC)},  # INSERT
        ]
    )
    monkeypatch.setattr(folders, "get_db", fake_db(cursor))

    created = folders.create_folder(
        FolderCreate(name="2", parent_id=9), {"id": 2, "perm_folders": True}
    )

    assert created["id"] == 29
    assert created["name"] == "2"
    assert created["parent_id"] == 9
