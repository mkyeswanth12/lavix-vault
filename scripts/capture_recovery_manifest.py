#!/usr/bin/env python3
"""Capture one immutable, verifier-compatible Lavix recovery manifest.

This utility is read-only against PostgreSQL, MinIO, and the RSA private key.
The sole write is an atomic, mode-0600 manifest at the operator-selected path.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.storage.access import StoredObjectExpectation  # noqa: E402
from scripts.verify_restored_vault import (  # noqa: E402
    MAX_CAPTURED_OBJECT_BYTES,
    RECOVERY_ID_PATTERN,
    SOURCE_REVISION_PATTERN,
    CapturedFile,
    FileReference,
    RecoveryVerificationError,
    build_minio_client,
    encode_recovery_manifest,
    load_database_snapshot,
    load_manifest,
    rsa_public_key_sha256,
    validate_database_snapshot,
)

_STREAM_CHUNK_BYTES = 1024 * 1024
_RECEIPT_SCHEMA = "lavix.recovery-manifest-capture.v1"


class RecoveryCaptureError(RuntimeError):
    """A bounded capture failure whose category is safe to print."""


class ObjectResponse(Protocol):
    def read(self, amount: int) -> bytes: ...

    def close(self) -> None: ...

    def release_conn(self) -> None: ...


class ObjectStore(Protocol):
    def stat_object(self, bucket: str, object_name: str) -> Any: ...

    def get_object(self, bucket: str, object_name: str) -> ObjectResponse: ...


@dataclass(frozen=True, slots=True)
class StreamedObjectDigest:
    size: int
    sha256: str
    generation: tuple[int, str | None, str | None, str | None]


@dataclass(frozen=True, slots=True)
class CapturedObjectGeneration:
    bucket: str
    object_name: str
    generation: tuple[int, str | None, str | None, str | None]


@dataclass(frozen=True, slots=True)
class CapturedObjectSet:
    files: tuple[CapturedFile, ...]
    generations: tuple[CapturedObjectGeneration, ...]


def _stat_size(metadata: Any) -> int:
    try:
        size = metadata.size
    except Exception as exc:
        raise RecoveryCaptureError("object_stat_invalid") from exc
    if isinstance(size, bool) or not isinstance(size, int):
        raise RecoveryCaptureError("object_stat_invalid")
    if size < 0:
        raise RecoveryCaptureError("object_stat_invalid")
    if size > MAX_CAPTURED_OBJECT_BYTES:
        raise RecoveryCaptureError("object_too_large")
    return size


def _generation_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    return normalized or None


def _object_generation(
    metadata: Any,
) -> tuple[int, str | None, str | None, str | None]:
    """Return stable StatObject identity; size alone is not a generation."""

    size = _stat_size(metadata)
    try:
        generation = tuple(
            _generation_value(getattr(metadata, attribute, None))
            for attribute in ("etag", "version_id", "last_modified")
        )
    except Exception as exc:
        raise RecoveryCaptureError("object_stat_invalid") from exc
    if not any(value is not None for value in generation):
        raise RecoveryCaptureError("object_generation_unavailable")
    return (size, *generation)


def stream_object_digest(
    client: ObjectStore,
    *,
    bucket: str,
    object_name: str,
) -> StreamedObjectDigest:
    """Hash one exact GetObject byte stream under the verifier's hard limit."""

    try:
        before = client.stat_object(bucket, object_name)
    except Exception as exc:
        raise RecoveryCaptureError("object_stat_failed") from exc
    before_generation = _object_generation(before)
    expected_size = before_generation[0]

    response: ObjectResponse | None = None
    digest = hashlib.sha256()
    observed_size = 0
    try:
        response = client.get_object(bucket, object_name)
        while True:
            chunk = response.read(_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise RecoveryCaptureError("object_stream_invalid")
            observed_size += len(chunk)
            if observed_size > expected_size or observed_size > MAX_CAPTURED_OBJECT_BYTES:
                raise RecoveryCaptureError("object_changed_during_capture")
            digest.update(chunk)
    except RecoveryCaptureError:
        raise
    except Exception as exc:
        raise RecoveryCaptureError("object_read_failed") from exc
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

    if observed_size != expected_size:
        raise RecoveryCaptureError("object_changed_during_capture")
    try:
        after = client.stat_object(bucket, object_name)
    except Exception as exc:
        raise RecoveryCaptureError("object_stat_failed") from exc
    after_generation = _object_generation(after)
    if before_generation != after_generation:
        raise RecoveryCaptureError("object_changed_during_capture")
    return StreamedObjectDigest(
        size=observed_size,
        sha256=digest.hexdigest(),
        generation=after_generation,
    )


def capture_file_objects(
    client: ObjectStore,
    records: Sequence[FileReference],
) -> CapturedObjectSet:
    """Capture exact encrypted payload and wrapped-key identities, sequentially."""

    captured: list[CapturedFile] = []
    generations: list[CapturedObjectGeneration] = []
    for record in records:
        payload = stream_object_digest(
            client,
            bucket=record.bucket,
            object_name=record.payload_path,
        )
        wrapped_key = stream_object_digest(
            client,
            bucket=record.bucket,
            object_name=record.wrapped_key_path,
        )
        generations.extend(
            (
                CapturedObjectGeneration(
                    bucket=record.bucket,
                    object_name=record.payload_path,
                    generation=payload.generation,
                ),
                CapturedObjectGeneration(
                    bucket=record.bucket,
                    object_name=record.wrapped_key_path,
                    generation=wrapped_key.generation,
                ),
            )
        )
        captured.append(
            CapturedFile(
                reference=record,
                expectation=StoredObjectExpectation(
                    payload_size=payload.size,
                    payload_sha256=payload.sha256,
                    wrapped_key_size=wrapped_key.size,
                    wrapped_key_sha256=wrapped_key.sha256,
                    max_object_bytes=MAX_CAPTURED_OBJECT_BYTES,
                ),
            )
        )
    return CapturedObjectSet(files=tuple(captured), generations=tuple(generations))


def verify_object_generations(
    client: ObjectStore,
    generations: Sequence[CapturedObjectGeneration],
) -> None:
    """Recheck every object after the complete streaming pass."""

    for captured in generations:
        try:
            current = client.stat_object(captured.bucket, captured.object_name)
        except Exception as exc:
            raise RecoveryCaptureError("object_stat_failed") from exc
        if _object_generation(current) != captured.generation:
            raise RecoveryCaptureError("object_changed_during_capture")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_create_private(
    destination: Path,
    raw: bytes,
    *,
    validate_staged: Callable[[Path], None],
) -> None:
    """Publish complete bytes atomically without replacing an existing path."""

    parent = destination.parent
    if not destination.name:
        raise RecoveryCaptureError("manifest_output_parent_invalid")
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise RecoveryCaptureError("manifest_output_parent_invalid") from exc
    if not stat.S_ISDIR(parent_metadata.st_mode) or parent.is_symlink():
        raise RecoveryCaptureError("manifest_output_parent_invalid")
    if parent_metadata.st_mode & 0o077:
        raise RecoveryCaptureError("manifest_output_parent_permissions_too_open")
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise RecoveryCaptureError("manifest_output_unavailable") from exc
    else:
        raise RecoveryCaptureError("manifest_output_exists")

    staged = parent / f".{destination.name}.capture-{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    published = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(staged, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        validate_staged(staged)
        try:
            os.link(staged, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise RecoveryCaptureError("manifest_output_exists") from exc
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise RecoveryCaptureError("manifest_output_exists") from exc
            raise RecoveryCaptureError("manifest_publish_failed") from exc
        published = True
        _fsync_directory(parent)
        metadata = destination.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o777 != 0o600:
            raise RecoveryCaptureError("manifest_permissions_invalid")
    except RecoveryCaptureError:
        if published:
            destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if published:
            destination.unlink(missing_ok=True)
        raise RecoveryCaptureError("manifest_publish_failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        staged.unlink(missing_ok=True)
        try:
            _fsync_directory(parent)
        except OSError:
            if published and destination.exists():
                destination.unlink(missing_ok=True)
            raise RecoveryCaptureError("manifest_publish_failed") from None


def _path_sha256(path: Path) -> str:
    normalized = os.fsencode(str(path.expanduser().absolute()))
    return hashlib.sha256(normalized).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a read-only, manifest-bound Lavix recovery identity.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument(
        "--source-revision",
        required=True,
        help=(
            "Digest-form release-tree/OCI identity obtained independently from "
            "signed build or deployment evidence."
        ),
    )
    parser.add_argument(
        "--confirm-writers-stopped",
        action="store_true",
        required=True,
        help="Assert that all PostgreSQL and MinIO writers are stopped for this capture window.",
    )
    args = parser.parse_args(argv)
    if not RECOVERY_ID_PATTERN.fullmatch(args.recovery_id):
        parser.error("--recovery-id contains unsupported characters")
    if not SOURCE_REVISION_PATTERN.fullmatch(args.source_revision):
        parser.error(
            "--source-revision must be release-tree-sha256:<64 lowercase hex> "
            "or oci-image-sha256:<64 lowercase hex>"
        )
    return args


def _failure_receipt(recovery_id: str, output: Path, category: str) -> dict[str, Any]:
    return {
        "schema": _RECEIPT_SCHEMA,
        "status": "failed",
        "recovery_id": recovery_id,
        "output_path_sha256": _path_sha256(output),
        "issue": category,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        initial = load_database_snapshot()
        validate_database_snapshot(initial)
        initial_key_sha256 = rsa_public_key_sha256(settings.private_key)
        object_client = build_minio_client()
        captured = capture_file_objects(object_client, initial.files)
        raw = encode_recovery_manifest(
            recovery_id=args.recovery_id,
            source_revision=args.source_revision,
            rsa_public_key_sha256=initial_key_sha256,
            snapshot=initial,
            captured_files=captured.files,
        )
        manifest_sha256 = hashlib.sha256(raw).hexdigest()

        def validate_staged(path: Path) -> None:
            load_manifest(
                path,
                expected_sha256=manifest_sha256,
                expected_recovery_id=args.recovery_id,
                expected_source_revision=args.source_revision,
            )
            verify_object_generations(object_client, captured.generations)
            final = load_database_snapshot()
            if final != initial:
                raise RecoveryCaptureError("database_changed_during_capture")
            final_key_sha256 = rsa_public_key_sha256(settings.private_key)
            if final_key_sha256 != initial_key_sha256:
                raise RecoveryCaptureError("private_key_changed_during_capture")

        atomic_create_private(args.output, raw, validate_staged=validate_staged)
    except (RecoveryCaptureError, RecoveryVerificationError) as exc:
        print(
            json.dumps(
                _failure_receipt(args.recovery_id, args.output, str(exc)),
                sort_keys=True,
            )
        )
        return 2
    except Exception:
        print(
            json.dumps(
                _failure_receipt(
                    args.recovery_id,
                    args.output,
                    "unexpected_capture_failure",
                ),
                sort_keys=True,
            )
        )
        return 2

    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "status": "captured",
        "recovery_id": args.recovery_id,
        "manifest_sha256": manifest_sha256,
        "output_path_sha256": _path_sha256(args.output),
        "file_count": len(initial.files),
        "live_file_count": sum(not record.is_deleted for record in initial.files),
        "trashed_file_count": sum(record.is_deleted for record in initial.files),
        "tenant_count": len(initial.tenants),
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
