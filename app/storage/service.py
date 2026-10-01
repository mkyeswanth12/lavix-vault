"""Transactional encrypted upload service with collision-free object keys."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from app.config import settings
from app.database import get_db
from app.ingestion.security import AsyncProcessRunner, SecureTempLease

from .upload import (
    QuotaExceeded,
    StagedUpload,
    UploadRejected,
    object_keys_for_file,
    sanitize_filename,
    stage_upload,
)

logger = logging.getLogger(__name__)


class ConnectionLike(Protocol):
    def cursor(self) -> Any: ...


ConnectionFactory = Callable[[], AbstractContextManager[ConnectionLike]]


class UploadConflict(UploadRejected):
    def __init__(self, message: str, *, existing_id: int | None = None) -> None:
        super().__init__(message)
        self.existing_id = existing_id


@dataclass(frozen=True, slots=True)
class StorageServiceSettings:
    temp_root: Path
    public_key: Path
    encryption_script: Path
    bucket: str
    max_file_size: int
    encryption_timeout_seconds: float = 300.0

    @classmethod
    def from_app_settings(cls) -> StorageServiceSettings:
        return cls(
            temp_root=settings.temp_dir,
            public_key=settings.public_key,
            encryption_script=settings.encryption_script,
            bucket=settings.s3_bucket_name,
            max_file_size=settings.max_file_size,
        )


@dataclass(frozen=True, slots=True)
class StoredFile:
    file_id: int
    file_uuid: UUID
    filename: str
    size_bytes: int
    sha256: str
    mime_type: str
    tags: tuple[str, ...]


class VaultStorageService:
    def __init__(
        self,
        minio_client: Any,
        *,
        connection_factory: ConnectionFactory = get_db,
        service_settings: StorageServiceSettings | None = None,
        process_runner: AsyncProcessRunner | None = None,
    ) -> None:
        self._minio = minio_client
        self._connections = connection_factory
        self._settings = service_settings or StorageServiceSettings.from_app_settings()
        self._runner = process_runner or AsyncProcessRunner(max_output_bytes=2 * 1024 * 1024)

    async def upload(
        self,
        source: Any,
        *,
        user_id: int,
        filename: str,
        declared_mime_type: str | None,
        folder_id: int | None,
        replace: bool,
    ) -> StoredFile:
        safe_name = sanitize_filename(filename)
        quota_remaining = await asyncio.to_thread(
            self._early_quota_remaining,
            user_id,
            folder_id,
            safe_name,
            replace,
        )
        file_uuid = uuid4()
        keys = object_keys_for_file(user_id, file_uuid)
        old_objects: tuple[str, str, str] | None = None
        objects_uploaded = False

        with SecureTempLease(self._settings.temp_root, prefix="upload-") as lease:
            suffix = Path(safe_name).suffix[:32]
            staged = await stage_upload(
                source,
                lease.allocate(f"payload{suffix}"),
                max_size_bytes=self._settings.max_file_size,
                quota_remaining_bytes=quota_remaining,
            )
            if staged.size_bytes == 0:
                raise UploadRejected("empty files are not accepted")
            mime_type = await asyncio.to_thread(
                self._detect_mime_type,
                staged.path,
                declared_mime_type,
            )
            encrypted_path, key_path = await self._encrypt(lease, staged)
            try:
                await asyncio.to_thread(
                    self._minio.fput_object,
                    self._settings.bucket,
                    keys.payload,
                    str(encrypted_path),
                )
                # From this point onward, attempt best-effort compensation.
                # Removing both keys is safe even when the wrapped-key upload
                # has not completed yet.
                objects_uploaded = True
                await asyncio.to_thread(
                    self._minio.fput_object,
                    self._settings.bucket,
                    keys.wrapped_key,
                    str(key_path),
                )
                file_id, old_objects = await asyncio.to_thread(
                    self._commit_file,
                    user_id,
                    file_uuid,
                    safe_name,
                    mime_type,
                    folder_id,
                    replace,
                    staged,
                    encrypted_path.name,
                    key_path.name,
                    keys.payload,
                    keys.wrapped_key,
                )
            except BaseException:
                if objects_uploaded:
                    await self._remove_objects(
                        self._settings.bucket,
                        keys.payload,
                        keys.wrapped_key,
                    )
                raise

        if old_objects:
            await self._remove_objects(*old_objects)
        return StoredFile(
            file_id=file_id,
            file_uuid=file_uuid,
            filename=safe_name,
            size_bytes=staged.size_bytes,
            sha256=staged.sha256,
            mime_type=mime_type,
            tags=(),
        )

    def _early_quota_remaining(
        self,
        user_id: int,
        folder_id: int | None,
        filename: str,
        replace: bool,
    ) -> int:
        with self._connections() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT storage_quota_bytes, storage_used_bytes FROM users WHERE id = %s",
                (user_id,),
            )
            user = cursor.fetchone()
            if not user:
                raise UploadRejected("upload owner does not exist")
            self._require_folder(cursor, user_id, folder_id)
            cursor.execute(
                """
                SELECT id, file_size_bytes
                FROM files
                WHERE user_id = %s AND original_filename = %s
                  AND folder_id IS NOT DISTINCT FROM %s AND is_deleted = FALSE
                ORDER BY id LIMIT 1
                """,
                (user_id, filename, folder_id),
            )
            duplicate = cursor.fetchone()
            if duplicate and not replace:
                raise UploadConflict("Filename already exists", existing_id=int(duplicate["id"]))
            reclaimable = int(duplicate["file_size_bytes"]) if duplicate and replace else 0
            return max(
                0,
                int(user["storage_quota_bytes"]) - int(user["storage_used_bytes"]) + reclaimable,
            )

    async def _encrypt(
        self,
        lease: SecureTempLease,
        staged: StagedUpload,
    ) -> tuple[Path, Path]:
        if not self._settings.public_key.is_file():
            raise UploadRejected("encryption key is unavailable")
        await self._runner.run(
            [
                sys.executable,
                str(self._settings.encryption_script),
                "encrypt",
                str(staged.path),
                "--delete",
            ],
            cwd=lease.path,
            timeout_seconds=self._settings.encryption_timeout_seconds,
            check=True,
        )
        encrypted_path = staged.path.with_suffix(staged.path.suffix + ".enc")
        key_path = staged.path.with_suffix(staged.path.suffix + ".key.rsa4096")
        if not encrypted_path.is_file() or not key_path.is_file():
            raise UploadRejected("encryption produced incomplete output")
        return encrypted_path, key_path

    def _commit_file(
        self,
        user_id: int,
        file_uuid: UUID,
        filename: str,
        mime_type: str,
        folder_id: int | None,
        replace: bool,
        staged: StagedUpload,
        encrypted_filename: str,
        rsa_key_filename: str,
        object_path: str,
        key_path: str,
    ) -> tuple[int, tuple[str, str, str] | None]:
        name_lock = self._advisory_key(f"name:{folder_id}:{filename}")
        hash_lock = self._advisory_key(f"sha:{folder_id}:{staged.sha256}")
        with self._connections() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT pg_advisory_xact_lock(%s, %s)", (user_id, name_lock))
            cursor.execute("SELECT pg_advisory_xact_lock(%s, %s)", (user_id, hash_lock))
            cursor.execute(
                """
                SELECT storage_quota_bytes, storage_used_bytes
                FROM users WHERE id = %s FOR UPDATE
                """,
                (user_id,),
            )
            owner = cursor.fetchone()
            if not owner:
                raise UploadRejected("upload owner does not exist")
            self._require_folder(cursor, user_id, folder_id)
            cursor.execute(
                """
                SELECT id, file_size_bytes, s3_bucket_name, s3_path, s3_key_path
                FROM files
                WHERE user_id = %s AND original_filename = %s
                  AND folder_id IS NOT DISTINCT FROM %s AND is_deleted = FALSE
                ORDER BY id LIMIT 1 FOR UPDATE
                """,
                (user_id, filename, folder_id),
            )
            duplicate = cursor.fetchone()
            if duplicate and not replace:
                raise UploadConflict("Filename already exists", existing_id=int(duplicate["id"]))
            replacement_id = int(duplicate["id"]) if duplicate and replace else None
            replacement_size = int(duplicate["file_size_bytes"]) if duplicate and replace else 0

            cursor.execute(
                """
                SELECT id, original_filename
                FROM files
                WHERE user_id = %s AND sha256_hash = %s
                  AND folder_id IS NOT DISTINCT FROM %s AND is_deleted = FALSE
                  AND (%s::INTEGER IS NULL OR id <> %s)
                ORDER BY id LIMIT 1
                """,
                (user_id, staged.sha256, folder_id, replacement_id, replacement_id),
            )
            content_duplicate = cursor.fetchone()
            if content_duplicate:
                raise UploadConflict(
                    f"Duplicate content already exists as {content_duplicate['original_filename']}",
                    existing_id=int(content_duplicate["id"]),
                )

            projected = int(owner["storage_used_bytes"]) - replacement_size + staged.size_bytes
            if projected > int(owner["storage_quota_bytes"]):
                raise QuotaExceeded("upload exceeds remaining storage quota")

            cursor.execute(
                """
                INSERT INTO files (
                    uuid, user_id, original_filename, display_name, mime_type,
                    file_size_bytes, encrypted_filename, rsa_key_filename,
                    sha256_hash, s3_bucket_name, s3_path, s3_key_path,
                    quick_tags, quick_summary, folder_id, ai_status
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, 'not_granted'
                )
                RETURNING id
                """,
                (
                    file_uuid,
                    user_id,
                    filename,
                    filename,
                    mime_type,
                    staged.size_bytes,
                    encrypted_filename,
                    rsa_key_filename,
                    staged.sha256,
                    self._settings.bucket,
                    object_path,
                    key_path,
                    [],
                    None,
                    folder_id,
                ),
            )
            file_id = int(cursor.fetchone()["id"])
            old_objects = None
            if replacement_id is not None:
                old_objects = (
                    str(duplicate["s3_bucket_name"] or self._settings.bucket),
                    str(duplicate["s3_path"]),
                    str(duplicate["s3_key_path"]),
                )
                cursor.execute(
                    "DELETE FROM files WHERE id = %s AND user_id = %s",
                    (replacement_id, user_id),
                )
        return file_id, old_objects

    async def _remove_objects(self, bucket: str, object_path: str, key_path: str) -> None:
        for path in (object_path, key_path):
            try:
                await asyncio.to_thread(self._minio.remove_object, bucket, path)
            except Exception as exc:
                logger.warning("Object cleanup failed for %s: %s", path, type(exc).__name__)

    @staticmethod
    def _require_folder(cursor: Any, user_id: int, folder_id: int | None) -> None:
        if folder_id is None:
            return
        cursor.execute(
            "SELECT id FROM folders WHERE id = %s AND user_id = %s AND is_deleted = FALSE",
            (folder_id, user_id),
        )
        if not cursor.fetchone():
            raise UploadRejected("upload folder does not exist")

    @staticmethod
    def _detect_mime_type(path: Path, declared: str | None) -> str:
        try:
            import magic

            detected = str(magic.from_file(str(path), mime=True) or "").strip()
        except (ImportError, OSError):
            detected = ""
        return (detected or declared or "application/octet-stream")[:255]

    @staticmethod
    def _advisory_key(value: str) -> int:
        raw = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(raw, "big", signed=True)
