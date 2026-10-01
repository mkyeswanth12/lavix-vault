"""Bounded decryption and object-store operations for vault file delivery."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from minio.commonconfig import CopySource

from app.config import settings
from app.ingestion.security import AsyncProcessRunner, SecureTempLease

_VERIFY_STREAM_CHUNK_BYTES = 1024 * 1024


class StoredObjectError(RuntimeError):
    """A safe public base error for encrypted object operations."""


class StoredObjectUnavailable(StoredObjectError):
    pass


class StoredObjectIntegrityError(StoredObjectError):
    pass


@dataclass(frozen=True, slots=True)
class StoredObjectRef:
    bucket: str
    payload_path: str
    wrapped_key_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class StoredObjectExpectation:
    """Capture-manifest identity for one encrypted payload/key pair."""

    payload_size: int
    payload_sha256: str
    wrapped_key_size: int
    wrapped_key_sha256: str
    max_object_bytes: int = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ObjectAccessSettings:
    temp_root: Path
    private_key: Path
    encryption_script: Path
    decrypt_timeout_seconds: float = 300.0

    @classmethod
    def from_app_settings(cls) -> ObjectAccessSettings:
        return cls(
            temp_root=settings.temp_dir,
            private_key=settings.private_key,
            encryption_script=settings.encryption_script,
        )


class VaultObjectAccess:
    def __init__(
        self,
        minio_client: Any,
        *,
        access_settings: ObjectAccessSettings | None = None,
        process_runner: AsyncProcessRunner | None = None,
    ) -> None:
        self._minio = minio_client
        self._settings = access_settings or ObjectAccessSettings.from_app_settings()
        self._runner = process_runner or AsyncProcessRunner(max_output_bytes=2 * 1024 * 1024)

    async def materialize(self, ref: StoredObjectRef) -> Path:
        """Decrypt one object to a private response file owned by the caller."""

        response_root = self._settings.temp_root / "responses"
        await asyncio.to_thread(response_root.mkdir, mode=0o700, parents=True, exist_ok=True)
        await asyncio.to_thread(os.chmod, response_root, 0o700)
        output_path = response_root / f"response-{uuid4().hex}"

        try:
            with SecureTempLease(self._settings.temp_root, prefix="delivery-") as lease:
                encrypted = lease.allocate("payload.enc")
                wrapped_key = lease.allocate("payload.key.rsa4096")
                downloads = await asyncio.gather(
                    asyncio.to_thread(
                        self._minio.fget_object,
                        ref.bucket,
                        ref.payload_path,
                        str(encrypted),
                    ),
                    asyncio.to_thread(
                        self._minio.fget_object,
                        ref.bucket,
                        ref.wrapped_key_path,
                        str(wrapped_key),
                    ),
                    return_exceptions=True,
                )
                failure = next((result for result in downloads if isinstance(result, BaseException)), None)
                if failure is not None:
                    raise StoredObjectUnavailable("encrypted source is unavailable") from failure

                if not self._settings.private_key.is_file():
                    raise StoredObjectUnavailable("decryption key is unavailable")
                unlock = lease.mkdir("unlock")
                await asyncio.to_thread(
                    shutil.copy2,
                    self._settings.private_key,
                    unlock / "private_4096.pem",
                )
                try:
                    await self._runner.run(
                        [
                            sys.executable,
                            str(self._settings.encryption_script),
                            "decrypt",
                            str(encrypted),
                        ],
                        cwd=lease.path,
                        timeout_seconds=self._settings.decrypt_timeout_seconds,
                        check=True,
                    )
                except Exception as exc:
                    raise StoredObjectIntegrityError("encrypted source could not be decrypted") from exc

                decrypted = encrypted.with_suffix("")
                if not decrypted.is_file():
                    raise StoredObjectIntegrityError("decryption produced no file")
                actual_hash = await asyncio.to_thread(self._sha256_file, decrypted)
                if ref.sha256 and actual_hash.lower() != ref.sha256.lower():
                    raise StoredObjectIntegrityError("decrypted source failed its integrity check")
                await asyncio.to_thread(os.replace, decrypted, output_path)
                await asyncio.to_thread(os.chmod, output_path, 0o600)
            return output_path
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise

    async def verify(
        self,
        ref: StoredObjectRef,
        expectation: StoredObjectExpectation,
    ) -> None:
        """Verify captured ciphertext and plaintext without materializing a response file."""

        sizes = (expectation.payload_size, expectation.wrapped_key_size)
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in sizes):
            raise StoredObjectIntegrityError("captured object size is invalid")
        if (
            isinstance(expectation.max_object_bytes, bool)
            or not isinstance(expectation.max_object_bytes, int)
            or expectation.max_object_bytes < 1
        ):
            raise StoredObjectIntegrityError("captured object limit is invalid")
        if (
            expectation.payload_size > expectation.max_object_bytes
            or expectation.wrapped_key_size > expectation.max_object_bytes
        ):
            raise StoredObjectIntegrityError("captured object exceeds the verification limit")

        try:
            payload_stat, key_stat = await asyncio.gather(
                asyncio.to_thread(self._minio.stat_object, ref.bucket, ref.payload_path),
                asyncio.to_thread(self._minio.stat_object, ref.bucket, ref.wrapped_key_path),
            )
        except Exception as exc:
            raise StoredObjectUnavailable("encrypted source is unavailable") from exc
        if (
            self._stat_size(payload_stat) != expectation.payload_size
            or self._stat_size(key_stat) != expectation.wrapped_key_size
        ):
            raise StoredObjectIntegrityError("encrypted source size differs from the capture")

        with SecureTempLease(self._settings.temp_root, prefix="restore-verify-") as lease:
            encrypted = lease.allocate("payload.enc")
            wrapped_key = lease.allocate("payload.key.rsa4096")
            await asyncio.to_thread(
                self._download_verified_object,
                bucket=ref.bucket,
                object_name=ref.payload_path,
                destination=encrypted,
                expected_size=expectation.payload_size,
                expected_sha256=expectation.payload_sha256,
            )
            await asyncio.to_thread(
                self._download_verified_object,
                bucket=ref.bucket,
                object_name=ref.wrapped_key_path,
                destination=wrapped_key,
                expected_size=expectation.wrapped_key_size,
                expected_sha256=expectation.wrapped_key_sha256,
            )
            if not self._settings.private_key.is_file():
                raise StoredObjectUnavailable("decryption key is unavailable")
            unlock = lease.mkdir("unlock")
            await asyncio.to_thread(
                shutil.copy2,
                self._settings.private_key,
                unlock / "private_4096.pem",
            )
            try:
                await self._runner.run(
                    [
                        sys.executable,
                        str(self._settings.encryption_script),
                        "decrypt",
                        str(encrypted),
                    ],
                    cwd=lease.path,
                    timeout_seconds=self._settings.decrypt_timeout_seconds,
                    check=True,
                )
            except Exception as exc:
                raise StoredObjectIntegrityError("encrypted source could not be decrypted") from exc
            decrypted = encrypted.with_suffix("")
            if not decrypted.is_file():
                raise StoredObjectIntegrityError("decryption produced no file")
            actual_plaintext_hash = await asyncio.to_thread(self._sha256_file, decrypted)
            if ref.sha256 and actual_plaintext_hash.lower() != ref.sha256.lower():
                raise StoredObjectIntegrityError("decrypted source failed its integrity check")

    def _download_verified_object(
        self,
        *,
        bucket: str,
        object_name: str,
        destination: Path,
        expected_size: int,
        expected_sha256: str,
    ) -> None:
        """Stream one object to disk without ever writing beyond its captured size."""

        response: Any | None = None
        digest = hashlib.sha256()
        observed_size = 0
        try:
            response = self._minio.get_object(bucket, object_name)
            with destination.open("xb") as output:
                os.chmod(destination, 0o600)
                while True:
                    remaining = expected_size - observed_size
                    chunk = response.read(min(_VERIFY_STREAM_CHUNK_BYTES, remaining + 1))
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise StoredObjectIntegrityError("encrypted source returned invalid bytes")
                    if len(chunk) > remaining:
                        raise StoredObjectIntegrityError("encrypted source differs from the capture")
                    output.write(chunk)
                    digest.update(chunk)
                    observed_size += len(chunk)
            if observed_size != expected_size or digest.hexdigest().lower() != expected_sha256.lower():
                raise StoredObjectIntegrityError("encrypted source differs from the capture")
        except StoredObjectIntegrityError:
            raise
        except Exception as exc:
            raise StoredObjectUnavailable("encrypted source is unavailable") from exc
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
                try:
                    response.release_conn()
                except Exception:
                    pass

    @staticmethod
    def _stat_size(metadata: Any) -> int:
        try:
            size = metadata.size
        except Exception as exc:
            raise StoredObjectIntegrityError("encrypted source size is invalid") from exc
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise StoredObjectIntegrityError("encrypted source size is invalid")
        return size

    async def remove(self, ref: StoredObjectRef) -> tuple[str, ...]:
        """Best-effort removal; return object names that could not be removed."""

        failed: list[str] = []
        for path in (ref.payload_path, ref.wrapped_key_path):
            try:
                await asyncio.to_thread(self._minio.remove_object, ref.bucket, path)
            except Exception:
                failed.append(path)
        return tuple(failed)

    async def copy(
        self,
        source: StoredObjectRef,
        destination: StoredObjectRef,
    ) -> None:
        """Copy both encrypted objects and attempt best-effort partial cleanup."""

        copied = False
        try:
            await asyncio.to_thread(
                self._minio.copy_object,
                destination.bucket,
                destination.payload_path,
                CopySource(source.bucket, source.payload_path),
            )
            copied = True
            await asyncio.to_thread(
                self._minio.copy_object,
                destination.bucket,
                destination.wrapped_key_path,
                CopySource(source.bucket, source.wrapped_key_path),
            )
        except BaseException:
            if copied:
                await self.remove(destination)
            raise

    @staticmethod
    def cleanup(path: Path) -> None:
        path.unlink(missing_ok=True)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
