"""Concrete encrypted MinIO source lease for end-to-end worker wiring."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import (
    ParseError,
    SourceIntegrityError,
    SourceUnavailableError,
)
from .models import IngestionJob
from .parsers.base import ParseRequest, sha256_file
from .security import AsyncProcessRunner, SecureTempLease


@dataclass(frozen=True, slots=True)
class VaultSourceSettings:
    temp_root: Path = Path("/app/temp")
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_secure: bool = False
    private_key: Path = Path("/run/secrets/private_key")
    encryption_script: Path = Path("/app/enc/encryption.py")
    decrypt_timeout_seconds: float = 300.0

    @classmethod
    def from_environment(cls) -> VaultSourceSettings:
        from app.config import settings as app_settings

        # Values resolve through the standard config surface (env var ->
        # config.yml -> default); the explicit env check below only exists to
        # produce a precise missing-setting message.
        required = {
            "S3_ENDPOINT": app_settings.s3_endpoint,
            "S3_ACCESS_KEY": app_settings.s3_access_key,
            "S3_SECRET_KEY": app_settings.s3_secret_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"missing worker setting(s): {', '.join(missing)}")
        return cls(
            temp_root=app_settings.ingestion_temp_root,
            s3_endpoint=required["S3_ENDPOINT"],
            s3_access_key=required["S3_ACCESS_KEY"],
            s3_secret_key=required["S3_SECRET_KEY"],
            s3_secure=app_settings.s3_secure,
            private_key=app_settings.private_key,
            encryption_script=app_settings.encryption_script,
            decrypt_timeout_seconds=float(
                os.environ.get("INGESTION_DECRYPT_TIMEOUT_SECONDS", "300")
            ),
        )


class VaultEncryptedSourceProvider:
    """Download, decrypt, verify and securely remove one vault object."""

    def __init__(
        self,
        settings: VaultSourceSettings | None = None,
        *,
        runner: AsyncProcessRunner | None = None,
        minio_client: Any = None,
    ) -> None:
        self.settings = settings or VaultSourceSettings.from_environment()
        self.runner = runner or AsyncProcessRunner(max_output_bytes=2 * 1024 * 1024)
        self._minio_client = minio_client

    @asynccontextmanager
    async def open(
        self,
        job: IngestionJob,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[ParseRequest]:
        if cancel_event and cancel_event.is_set():
            from .errors import IngestionCancelled

            raise IngestionCancelled("source acquisition cancelled before start")
        metadata = job.metadata
        bucket = self._required(metadata, "s3_bucket_name")
        object_path = self._required(metadata, "s3_path")
        key_path = self._required(metadata, "s3_key_path")
        encrypted_name = self._basename(str(metadata.get("encrypted_filename") or Path(object_path).name))
        rsa_name = self._basename(str(metadata.get("rsa_key_filename") or Path(key_path).name))
        with SecureTempLease(self.settings.temp_root, prefix=f"source-{job.file_id}-") as lease:
            encrypted_local = lease.allocate(encrypted_name)
            key_local = lease.allocate(rsa_name)
            try:
                client = self._client()
                downloads = await asyncio.gather(
                    asyncio.to_thread(client.fget_object, bucket, object_path, str(encrypted_local)),
                    asyncio.to_thread(client.fget_object, bucket, key_path, str(key_local)),
                    return_exceptions=True,
                )
                failure = next(
                    (result for result in downloads if isinstance(result, BaseException)),
                    None,
                )
                if failure is not None:
                    raise SourceUnavailableError("failed to download encrypted source") from failure
            except SourceUnavailableError:
                raise
            except Exception as exc:
                raise SourceUnavailableError("failed to download encrypted source") from exc

            if cancel_event and cancel_event.is_set():
                from .errors import IngestionCancelled

                raise IngestionCancelled("source acquisition cancelled")

            if not self.settings.private_key.is_file():
                raise SourceUnavailableError("private key is unavailable")
            unlock_dir = lease.mkdir("unlock")
            await asyncio.to_thread(shutil.copy2, self.settings.private_key, unlock_dir / "private_4096.pem")
            result = await self.runner.run(
                [
                    sys.executable,
                    str(self.settings.encryption_script),
                    "decrypt",
                    str(encrypted_local),
                ],
                cwd=lease.path,
                timeout_seconds=self.settings.decrypt_timeout_seconds,
                cancel_event=cancel_event,
            )
            if result.returncode != 0:
                raise SourceUnavailableError("source decryption failed")
            decrypted = encrypted_local.with_suffix("")
            if not decrypted.is_file() or decrypted.stat().st_size == 0:
                raise ParseError("decryption produced no source file")
            actual_sha = await asyncio.to_thread(sha256_file, decrypted)
            if job.source_sha256 and actual_sha.lower() != job.source_sha256.lower():
                raise SourceIntegrityError("decrypted source hash does not match its vault record")
            yield ParseRequest(
                path=decrypted,
                source_name=job.source_name,
                media_type=job.media_type,
                source_sha256=actual_sha,
            )

    def _client(self) -> Any:
        if self._minio_client is not None:
            return self._minio_client
        try:
            from minio import Minio  # imported only in a wired worker
        except ModuleNotFoundError as exc:
            raise SourceUnavailableError("MinIO client is not installed") from exc
        endpoint = self.settings.s3_endpoint.removeprefix("https://").removeprefix("http://")
        self._minio_client = Minio(
            endpoint,
            access_key=self.settings.s3_access_key,
            secret_key=self.settings.s3_secret_key,
            secure=self.settings.s3_secure,
        )
        return self._minio_client

    @staticmethod
    def _required(metadata: Any, key: str) -> str:
        value = metadata.get(key) if hasattr(metadata, "get") else None
        if not isinstance(value, str) or not value:
            raise SourceUnavailableError(f"source metadata lacks {key}")
        return value

    @staticmethod
    def _basename(name: str) -> str:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise SourceUnavailableError("source metadata contains an unsafe filename")
        return name
