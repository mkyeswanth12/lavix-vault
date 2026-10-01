from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import file_upload
from app.storage.service import StoredFile, UploadConflict


class FakeStorage:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def upload(self, source, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


def _client(monkeypatch, storage, user=None):
    monkeypatch.setattr(file_upload, "storage_service", storage)
    app = FastAPI()
    app.include_router(file_upload.router, prefix="/api/files")
    app.dependency_overrides[file_upload.get_current_user] = lambda: user or {"id": 5, "perm_upload": True}
    return TestClient(app)


def test_upload_api_delegates_to_collision_safe_storage(monkeypatch):
    stored = StoredFile(
        file_id=12,
        file_uuid=UUID("8d1d312b-ff3f-4d5f-9f64-6685d880640b"),
        filename="report.docx",
        size_bytes=4,
        sha256="a" * 64,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        tags=(),
    )
    storage = FakeStorage(stored)
    response = _client(monkeypatch, storage).post(
        "/api/files/upload",
        data={"folder_id": "9", "replace": "true"},
        files={"file": ("report.docx", b"test", "application/octet-stream")},
    )
    assert response.status_code == 200
    assert response.json()["file_uuid"] == str(stored.file_uuid)
    assert response.json()["tags"] == []
    assert response.json()["summary"] is None
    assert response.json()["intelligence_status"] == "pending"
    assert storage.calls == [
        {
            "user_id": 5,
            "filename": "report.docx",
            "declared_mime_type": "application/octet-stream",
            "folder_id": 9,
            "replace": True,
        }
    ]


def test_upload_api_preserves_conflict_contract_and_permissions(monkeypatch):
    storage = FakeStorage(error=UploadConflict("Filename already exists", existing_id=21))
    response = _client(monkeypatch, storage).post(
        "/api/files/upload",
        files={"file": ("report.pdf", b"pdf", "application/pdf")},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["existing_id"] == 21

    denied = _client(monkeypatch, storage, user={"id": 5, "perm_upload": False}).post(
        "/api/files/upload",
        files={"file": ("report.pdf", b"pdf", "application/pdf")},
    )
    assert denied.status_code == 403


def test_upload_provisions_named_folder_when_no_id(monkeypatch):
    from app.routers import file_upload as module

    stored = StoredFile(
        file_id=21,
        file_uuid=UUID("8d1d312b-ff3f-4d5f-9f64-6685d880640b"),
        filename="pasted-image-20260920-120000.png",
        size_bytes=8,
        sha256="b" * 64,
        mime_type="image/png",
        tags=(),
    )
    storage = FakeStorage(stored)
    resolved = {}

    def _fake_provision(user_id, name, user):
        resolved["args"] = (user_id, name)
        return 77

    monkeypatch.setattr(module, "_get_or_create_folder_id", _fake_provision)
    response = _client(monkeypatch, storage).post(
        "/api/files/upload",
        data={"folder_name": "chat_uploads"},
        files={"file": ("pasted-image-20260920-120000.png", b"pngdata!", "image/png")},
    )
    assert response.status_code == 200
    assert resolved["args"] == (5, "chat_uploads")
    assert storage.calls[0]["folder_id"] == 77


def test_upload_explicit_id_wins_over_folder_name(monkeypatch):
    from app.routers import file_upload as module

    stored = StoredFile(
        file_id=22,
        file_uuid=UUID("8d1d312b-ff3f-4d5f-9f64-6685d880640b"),
        filename="a.png",
        size_bytes=4,
        sha256="c" * 64,
        mime_type="image/png",
        tags=(),
    )
    storage = FakeStorage(stored)

    def _boom(user_id, name, user):
        raise AssertionError("must not provision when folder_id is explicit")

    monkeypatch.setattr(module, "_get_or_create_folder_id", _boom)
    response = _client(monkeypatch, storage).post(
        "/api/files/upload",
        data={"folder_id": "9", "folder_name": "chat_uploads"},
        files={"file": ("a.png", b"pngdata!", "image/png")},
    )
    assert response.status_code == 200
    assert storage.calls[0]["folder_id"] == 9


def test_get_or_create_folder_id_reuses_restores_and_inserts(monkeypatch):
    from fastapi import HTTPException

    from app.routers import file_upload as module

    script = iter([
        {"id": 11},
        None, {"id": 12},
        None, None, {"id": 13},
    ])
    statements = []

    class FakeCursor:
        def execute(self, sql, params=None):
            statements.append(" ".join(str(sql).split())[:24])

        def fetchone(self):
            return next(script)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(module, "get_db", lambda: FakeConnection())
    user = {"id": 5, "perm_upload": True, "perm_folders": True}

    assert module._get_or_create_folder_id(5, "chat_uploads", user) == 11
    assert statements[0].startswith("SELECT id FROM folders")
    assert module._get_or_create_folder_id(5, "  chat_uploads  ", user) == 12
    assert any(s.startswith("UPDATE folders") for s in statements)
    assert module._get_or_create_folder_id(5, "chat_uploads", user) == 13
    assert any(s.startswith("INSERT INTO folders") for s in statements)

    try:
        module._get_or_create_folder_id(5, "chat_uploads", {"id": 5, "perm_folders": False})
    except HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("expected 403 without folder permission")
