"""Safe, deterministic staging for encrypted vault uploads."""

from __future__ import annotations

import hashlib
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Protocol
from uuid import UUID


class AsyncReadable(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


class UploadRejected(ValueError):
    """Base class for upload validation failures."""


class UploadTooLarge(UploadRejected):
    pass


class QuotaExceeded(UploadRejected):
    pass


@dataclass(frozen=True, slots=True)
class ObjectKeys:
    payload: str
    wrapped_key: str


@dataclass(frozen=True, slots=True)
class StagedUpload:
    path: Path
    size_bytes: int
    sha256: str


def object_keys_for_file(user_id: int, file_uuid: UUID) -> ObjectKeys:
    """Return collision-resistant keys that never contain a user filename."""

    if user_id <= 0:
        raise ValueError("user_id must be positive")
    prefix = f"users/{user_id}/files/{file_uuid}"
    return ObjectKeys(
        payload=f"{prefix}/payload.enc",
        wrapped_key=f"{prefix}/key.wrap",
    )


def sanitize_filename(value: str, *, max_length: int = 255) -> str:
    """Normalize a display filename without using it as an object-store key."""

    # Normalize both separator styles even when the server itself is POSIX.
    candidate = unicodedata.normalize("NFC", PurePath((value or "").replace("\\", "/")).name)
    candidate = "".join(character for character in candidate if character >= " " and character != "\x7f")
    candidate = candidate.replace("\x00", "").strip().strip(".")
    if not candidate:
        raise UploadRejected("filename is empty after normalization")
    encoded = candidate.encode("utf-8")
    if len(encoded) <= max_length:
        return candidate

    suffix = Path(candidate).suffix
    suffix_bytes = suffix.encode("utf-8")
    budget = max(1, max_length - len(suffix_bytes))
    stem_bytes = Path(candidate).stem.encode("utf-8")[:budget]
    while stem_bytes:
        try:
            stem = stem_bytes.decode("utf-8")
            return f"{stem}{suffix}"
        except UnicodeDecodeError:
            stem_bytes = stem_bytes[:-1]
    raise UploadRejected("filename cannot be represented safely")


async def stage_upload(
    source: AsyncReadable,
    destination: Path,
    *,
    max_size_bytes: int,
    quota_remaining_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> StagedUpload:
    """Stream an upload to a private path while enforcing limits before writing.

    A partial file is removed on every failure.  The caller must provide a
    destination inside its private temporary job directory.
    """

    if max_size_bytes < 0 or quota_remaining_bytes < 0:
        raise ValueError("size limits must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as staged:
            while True:
                chunk = await source.read(chunk_size)
                if not chunk:
                    break
                next_size = size + len(chunk)
                if next_size > max_size_bytes:
                    raise UploadTooLarge(f"upload exceeds {max_size_bytes} bytes")
                if next_size > quota_remaining_bytes:
                    raise QuotaExceeded("upload exceeds remaining storage quota")
                staged.write(chunk)
                digest.update(chunk)
                size = next_size
            staged.flush()
            os.fsync(staged.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    return StagedUpload(path=destination, size_bytes=size, sha256=digest.hexdigest())
