from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

import pytest

from app.storage.service import StorageServiceSettings, VaultStorageService
from app.storage.upload import (
    QuotaExceeded,
    UploadRejected,
    UploadTooLarge,
    object_keys_for_file,
    sanitize_filename,
    stage_upload,
)


class AsyncBytes:
    def __init__(self, value: bytes) -> None:
        self.value = value
        self.offset = 0

    async def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.value):
            return b""
        end = len(self.value) if size < 0 else self.offset + size
        chunk = self.value[self.offset : end]
        self.offset += len(chunk)
        return chunk


def test_object_keys_are_uuid_scoped_and_filename_independent() -> None:
    keys = object_keys_for_file(17, UUID("7d849e91-4da9-47ed-b3b4-3cdf9f508e55"))
    assert keys.payload == "users/17/files/7d849e91-4da9-47ed-b3b4-3cdf9f508e55/payload.enc"
    assert keys.wrapped_key.endswith("/key.wrap")


def test_sanitize_filename_removes_paths_and_control_characters() -> None:
    assert sanitize_filename("../../reports/quarter\x00ly.pdf") == "quarterly.pdf"
    assert sanitize_filename(r"..\..\reports\windows.docx") == "windows.docx"
    with pytest.raises(UploadRejected):
        sanitize_filename("../..")


async def test_stage_upload_hashes_and_writes_private_file(tmp_path) -> None:
    payload = b"lavix-vault" * 32
    result = await stage_upload(
        AsyncBytes(payload),
        tmp_path / "private" / "payload",
        max_size_bytes=len(payload),
        quota_remaining_bytes=len(payload),
        chunk_size=17,
    )
    assert result.path.read_bytes() == payload
    assert result.size_bytes == len(payload)
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert result.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("maximum", "quota", "error"),
    [(3, 100, UploadTooLarge), (100, 3, QuotaExceeded)],
)
async def test_stage_upload_rejects_before_persisting_partial_file(tmp_path, maximum, quota, error) -> None:
    destination = tmp_path / "payload"
    with pytest.raises(error):
        await stage_upload(
            AsyncBytes(b"0123456789"),
            destination,
            max_size_bytes=maximum,
            quota_remaining_bytes=quota,
            chunk_size=4,
        )
    assert not destination.exists()


class FailingSecondUploadMinio:
    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.removed: list[str] = []

    def fput_object(self, bucket: str, object_name: str, source: str) -> None:
        assert Path(source).is_file()
        if self.uploaded:
            raise RuntimeError("wrapped-key upload failed")
        self.uploaded.append(object_name)

    def remove_object(self, bucket: str, object_name: str) -> None:
        self.removed.append(object_name)


async def test_storage_service_compensates_when_second_object_upload_fails(tmp_path, monkeypatch) -> None:
    public_key = tmp_path / "public.pem"
    public_key.write_text("test", encoding="utf-8")
    minio = FailingSecondUploadMinio()
    service = VaultStorageService(
        minio,
        service_settings=StorageServiceSettings(
            temp_root=tmp_path / "temp",
            public_key=public_key,
            encryption_script=tmp_path / "encrypt.py",
            bucket="vault-test",
            max_file_size=1024,
        ),
    )
    monkeypatch.setattr(service, "_early_quota_remaining", lambda *_args: 1024)

    async def fake_encrypt(lease, staged):
        encrypted = staged.path.with_suffix(staged.path.suffix + ".enc")
        wrapped_key = staged.path.with_suffix(staged.path.suffix + ".key.rsa4096")
        encrypted.write_bytes(b"encrypted")
        wrapped_key.write_bytes(b"key")
        return encrypted, wrapped_key

    monkeypatch.setattr(service, "_encrypt", fake_encrypt)

    with pytest.raises(RuntimeError, match="wrapped-key upload failed"):
        await service.upload(
            AsyncBytes(b"plain"),
            user_id=7,
            filename="report.docx",
            declared_mime_type=None,
            folder_id=None,
            replace=False,
        )

    assert len(minio.uploaded) == 1
    assert minio.removed == [
        minio.uploaded[0],
        minio.uploaded[0].replace("payload.enc", "key.wrap"),
    ]
