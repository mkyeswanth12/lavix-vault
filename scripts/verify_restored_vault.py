#!/usr/bin/env python3
"""Verify an isolated Lavix restore against one immutable capture manifest.

The verifier is read-only. It binds PostgreSQL rows, tenant/trash ownership,
encrypted MinIO bytes, the RSA public key, migration head, and decrypted hashes
to a caller-supplied manifest digest. Plaintext never leaves a per-object secure
temporary lease.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import stat
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryptography.hazmat.primitives import serialization  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import get_connection, put_connection  # noqa: E402
from app.storage.access import (  # noqa: E402
    StoredObjectExpectation,
    StoredObjectIntegrityError,
    StoredObjectRef,
    StoredObjectUnavailable,
    VaultObjectAccess,
)
from minio import Minio  # noqa: E402

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RECOVERY_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SOURCE_REVISION_RE = re.compile(
    r"(?:release-tree|oci-image)-sha256:[0-9a-f]{64}\Z"
)
_MANIFEST_SCHEMA = "lavix.recovery-manifest.v2"
_REPORT_SCHEMA = "lavix.restore-verification.v3"
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_CAPTURED_OBJECT_BYTES = 2 * 1024 * 1024 * 1024

# Public recovery-contract constants. Capture tooling imports these aliases so
# schema names and safety bounds have one authority: this verifier.
MANIFEST_SCHEMA = _MANIFEST_SCHEMA
MAX_MANIFEST_BYTES = _MAX_MANIFEST_BYTES
MAX_CAPTURED_OBJECT_BYTES = _MAX_CAPTURED_OBJECT_BYTES
SHA256_PATTERN = _SHA256_RE
RECOVERY_ID_PATTERN = _RECOVERY_ID_RE
SOURCE_REVISION_PATTERN = _SOURCE_REVISION_RE


class RecoveryVerificationError(RuntimeError):
    """A bounded verifier failure safe to expose in recovery evidence."""


class ObjectAccess(Protocol):
    async def verify(
        self,
        ref: StoredObjectRef,
        expectation: StoredObjectExpectation,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class FileReference:
    file_id: int
    user_id: int
    is_deleted: bool
    bucket: str
    payload_path: str
    wrapped_key_path: str
    plaintext_sha256: str
    file_row_sha256: str

    def validation_error(self) -> str | None:
        if self.file_id <= 0:
            return "invalid_file_id"
        if self.user_id <= 0:
            return "invalid_user_id"
        if not self.bucket:
            return "missing_bucket"
        if not self.payload_path:
            return "missing_payload_reference"
        if not self.wrapped_key_path:
            return "missing_wrapped_key_reference"
        if not _SHA256_RE.fullmatch(self.plaintext_sha256):
            return "invalid_plaintext_sha256"
        if not _SHA256_RE.fullmatch(self.file_row_sha256):
            return "invalid_file_row_sha256"
        return None

    def object_ref(self) -> StoredObjectRef:
        return StoredObjectRef(
            bucket=self.bucket,
            payload_path=self.payload_path,
            wrapped_key_path=self.wrapped_key_path,
            sha256=self.plaintext_sha256,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "user_id": self.user_id,
            "is_deleted": self.is_deleted,
            "bucket": self.bucket,
            "payload_path": self.payload_path,
            "wrapped_key_path": self.wrapped_key_path,
            "plaintext_sha256": self.plaintext_sha256.lower(),
            "file_row_sha256": self.file_row_sha256.lower(),
        }


@dataclass(frozen=True, slots=True)
class TenantIdentity:
    """Redacted binding for one canonical user/tenant identity."""

    user_id: int
    stable_identity_sha256: str

    def validation_error(self) -> str | None:
        if self.user_id <= 0:
            return "invalid_user_id"
        if not _SHA256_RE.fullmatch(self.stable_identity_sha256):
            return "invalid_stable_identity_sha256"
        return None

    def identity(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "stable_identity_sha256": self.stable_identity_sha256.lower(),
        }


@dataclass(frozen=True, slots=True)
class CapturedFile:
    reference: FileReference
    expectation: StoredObjectExpectation


@dataclass(frozen=True, slots=True)
class DatabaseSnapshot:
    files: tuple[FileReference, ...]
    tenants: tuple[TenantIdentity, ...]
    schema_head: int

    @property
    def rowset_sha256(self) -> str:
        return _rowset_sha256(self.files)

    @property
    def tenantset_sha256(self) -> str:
        return _tenantset_sha256(self.tenants)


@dataclass(frozen=True, slots=True)
class RecoveryManifest:
    recovery_id: str
    source_revision: str
    schema_head: int
    rsa_public_key_sha256: str
    captured_files: tuple[CapturedFile, ...]
    tenants: tuple[TenantIdentity, ...]
    manifest_sha256: str

    @property
    def files(self) -> tuple[FileReference, ...]:
        return tuple(item.reference for item in self.captured_files)

    @property
    def rowset_sha256(self) -> str:
        return _rowset_sha256(self.files)

    @property
    def tenantset_sha256(self) -> str:
        return _tenantset_sha256(self.tenants)


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    category: str
    file_id: int | None = None


@dataclass(frozen=True, slots=True)
class VerificationReport:
    recovery_id: str
    manifest_sha256: str
    source_revision: str
    expected_files: int
    referenced_files: int
    verified_files: int
    live_files: int
    trashed_files: int
    tenant_count: int
    expected_schema_head: int
    observed_schema_head: int
    database_rowset_sha256: str
    final_database_rowset_sha256: str
    database_tenantset_sha256: str
    final_database_tenantset_sha256: str
    issues: tuple[VerificationIssue, ...]

    @property
    def ok(self) -> bool:
        return self.referenced_files == self.expected_files and not self.issues

    def public_payload(self) -> dict[str, Any]:
        counts = Counter(issue.category for issue in self.issues)
        return {
            "schema": _REPORT_SCHEMA,
            "recovery_id": self.recovery_id,
            "manifest_sha256": self.manifest_sha256,
            "source_revision": self.source_revision,
            "status": "passed" if self.ok else "failed",
            "expected_files": self.expected_files,
            "referenced_files": self.referenced_files,
            "verified_files": self.verified_files,
            "live_files": self.live_files,
            "trashed_files": self.trashed_files,
            "tenant_count": self.tenant_count,
            "expected_schema_head": self.expected_schema_head,
            "observed_schema_head": self.observed_schema_head,
            "database_rowset_sha256": self.database_rowset_sha256,
            "final_database_rowset_sha256": self.final_database_rowset_sha256,
            "database_tenantset_sha256": self.database_tenantset_sha256,
            "final_database_tenantset_sha256": self.final_database_tenantset_sha256,
            "issue_counts": dict(sorted(counts.items())),
            "issues": [asdict(issue) for issue in self.issues],
        }


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rowset_sha256(records: Sequence[FileReference]) -> str:
    payload = json.dumps(
        [record.identity() for record in records],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _tenantset_sha256(records: Sequence[TenantIdentity]) -> str:
    payload = json.dumps(
        [record.identity() for record in records],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _required_mapping(value: Any, category: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryVerificationError(category)
    return value


def _required_sha256(value: Any, category: str) -> str:
    normalized = _text(value).lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise RecoveryVerificationError(category)
    return normalized


def _required_positive_int(value: Any, category: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise RecoveryVerificationError(category)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryVerificationError(category) from exc
    if parsed < (0 if allow_zero else 1):
        raise RecoveryVerificationError(category)
    return parsed


def validate_database_snapshot(snapshot: DatabaseSnapshot) -> None:
    """Reject a snapshot that could not produce a verifier-valid manifest."""

    if isinstance(snapshot.schema_head, bool) or snapshot.schema_head < 0:
        raise RecoveryVerificationError("database_schema_head_invalid")
    if snapshot.files != tuple(sorted(snapshot.files, key=lambda item: item.file_id)):
        raise RecoveryVerificationError("database_files_not_canonical")
    if snapshot.tenants != tuple(sorted(snapshot.tenants, key=lambda item: item.user_id)):
        raise RecoveryVerificationError("database_tenants_not_canonical")

    seen_file_ids: set[int] = set()
    seen_tenant_ids: set[int] = set()
    seen_objects: set[tuple[str, str]] = set()
    for tenant in snapshot.tenants:
        metadata_error = tenant.validation_error()
        if metadata_error is not None:
            raise RecoveryVerificationError(f"database_tenant_{metadata_error}")
        if tenant.user_id in seen_tenant_ids:
            raise RecoveryVerificationError("database_duplicate_tenant_user_id")
        seen_tenant_ids.add(tenant.user_id)
    for record in snapshot.files:
        metadata_error = record.validation_error()
        if metadata_error is not None:
            raise RecoveryVerificationError(f"database_{metadata_error}")
        if record.file_id in seen_file_ids:
            raise RecoveryVerificationError("database_duplicate_file_id")
        seen_file_ids.add(record.file_id)
        for object_ref in (
            (record.bucket, record.payload_path),
            (record.bucket, record.wrapped_key_path),
        ):
            if object_ref in seen_objects:
                raise RecoveryVerificationError("database_duplicate_object_reference")
            seen_objects.add(object_ref)
    if not {record.user_id for record in snapshot.files}.issubset(seen_tenant_ids):
        raise RecoveryVerificationError("database_file_tenant_missing")


def encode_recovery_manifest(
    *,
    recovery_id: str,
    source_revision: str,
    rsa_public_key_sha256: str,
    snapshot: DatabaseSnapshot,
    captured_files: Sequence[CapturedFile],
) -> bytes:
    """Encode deterministic v2 bytes using the verifier's canonical contract."""

    if not RECOVERY_ID_PATTERN.fullmatch(recovery_id):
        raise RecoveryVerificationError("manifest_recovery_id_invalid")
    if not SOURCE_REVISION_PATTERN.fullmatch(source_revision):
        raise RecoveryVerificationError("manifest_source_revision_invalid")
    rsa_fingerprint = _required_sha256(
        rsa_public_key_sha256,
        "manifest_rsa_fingerprint_invalid",
    )
    validate_database_snapshot(snapshot)
    captured = tuple(captured_files)
    if tuple(item.reference for item in captured) != snapshot.files:
        raise RecoveryVerificationError("manifest_capture_rowset_mismatch")

    files: list[dict[str, Any]] = []
    for item in captured:
        expectation = item.expectation
        sizes = (expectation.payload_size, expectation.wrapped_key_size)
        if any(isinstance(size, bool) or size < 0 for size in sizes):
            raise RecoveryVerificationError("manifest_object_size_invalid")
        if any(size > MAX_CAPTURED_OBJECT_BYTES for size in sizes):
            raise RecoveryVerificationError("manifest_object_too_large")
        payload_sha256 = _required_sha256(
            expectation.payload_sha256,
            "manifest_payload_sha256_invalid",
        )
        wrapped_key_sha256 = _required_sha256(
            expectation.wrapped_key_sha256,
            "manifest_wrapped_key_sha256_invalid",
        )
        files.append(
            {
                **item.reference.identity(),
                "payload_size": expectation.payload_size,
                "payload_sha256": payload_sha256,
                "wrapped_key_size": expectation.wrapped_key_size,
                "wrapped_key_sha256": wrapped_key_sha256,
            }
        )

    payload = {
        "schema": MANIFEST_SCHEMA,
        "recovery_id": recovery_id,
        "source_revision": source_revision,
        "schema_head": snapshot.schema_head,
        "rsa_public_key_sha256": rsa_fingerprint,
        "tenants": [tenant.identity() for tenant in snapshot.tenants],
        "files": files,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise RecoveryVerificationError("manifest_too_large")
    return raw


def load_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_recovery_id: str,
    expected_source_revision: str,
) -> RecoveryManifest:
    """Load one manifest and bind it to independent recovery/source evidence."""

    if not _SOURCE_REVISION_RE.fullmatch(expected_source_revision):
        raise RecoveryVerificationError("observed_source_revision_invalid")

    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise RecoveryVerificationError("manifest_not_regular")
        if metadata.st_mode & 0o077:
            raise RecoveryVerificationError("manifest_permissions_too_open")
        if metadata.st_size > _MAX_MANIFEST_BYTES:
            raise RecoveryVerificationError("manifest_too_large")
        raw = path.read_bytes()
    except RecoveryVerificationError:
        raise
    except OSError as exc:
        raise RecoveryVerificationError("manifest_unavailable") from exc
    manifest_sha256 = _sha256_bytes(raw)
    if manifest_sha256 != expected_sha256:
        raise RecoveryVerificationError("manifest_digest_mismatch")
    try:
        payload = _required_mapping(json.loads(raw), "manifest_invalid_json")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryVerificationError("manifest_invalid_json") from exc
    if payload.get("schema") != _MANIFEST_SCHEMA:
        raise RecoveryVerificationError("manifest_schema_unsupported")
    recovery_id = _text(payload.get("recovery_id"))
    if recovery_id != expected_recovery_id or not _RECOVERY_ID_RE.fullmatch(recovery_id):
        raise RecoveryVerificationError("manifest_recovery_id_mismatch")
    source_revision = _text(payload.get("source_revision"))
    if not _SOURCE_REVISION_RE.fullmatch(source_revision):
        raise RecoveryVerificationError("manifest_source_revision_invalid")
    if source_revision != expected_source_revision:
        raise RecoveryVerificationError("source_revision_mismatch")
    schema_head = _required_positive_int(
        payload.get("schema_head"),
        "manifest_schema_head_invalid",
        allow_zero=True,
    )
    rsa_fingerprint = _required_sha256(
        payload.get("rsa_public_key_sha256"),
        "manifest_rsa_fingerprint_invalid",
    )
    raw_tenants = payload.get("tenants")
    if not isinstance(raw_tenants, list):
        raise RecoveryVerificationError("manifest_tenants_invalid")
    tenants: list[TenantIdentity] = []
    seen_tenant_ids: set[int] = set()
    for raw_tenant in raw_tenants:
        item = _required_mapping(raw_tenant, "manifest_tenant_invalid")
        tenant = TenantIdentity(
            user_id=_required_positive_int(
                item.get("user_id"),
                "manifest_tenant_user_id_invalid",
            ),
            stable_identity_sha256=_required_sha256(
                item.get("stable_identity_sha256"),
                "manifest_tenant_identity_sha256_invalid",
            ),
        )
        metadata_error = tenant.validation_error()
        if metadata_error is not None:
            raise RecoveryVerificationError(f"manifest_tenant_{metadata_error}")
        if tenant.user_id in seen_tenant_ids:
            raise RecoveryVerificationError("manifest_duplicate_tenant_user_id")
        seen_tenant_ids.add(tenant.user_id)
        tenants.append(tenant)
    tenants.sort(key=lambda item: item.user_id)

    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise RecoveryVerificationError("manifest_files_invalid")
    captured: list[CapturedFile] = []
    seen_ids: set[int] = set()
    seen_objects: set[tuple[str, str]] = set()
    for raw_file in raw_files:
        item = _required_mapping(raw_file, "manifest_file_invalid")
        reference = FileReference(
            file_id=_required_positive_int(item.get("file_id"), "manifest_file_id_invalid"),
            user_id=_required_positive_int(item.get("user_id"), "manifest_user_id_invalid"),
            is_deleted=item.get("is_deleted") if isinstance(item.get("is_deleted"), bool) else False,
            bucket=_text(item.get("bucket")),
            payload_path=_text(item.get("payload_path")),
            wrapped_key_path=_text(item.get("wrapped_key_path")),
            plaintext_sha256=_required_sha256(
                item.get("plaintext_sha256"),
                "manifest_plaintext_sha256_invalid",
            ),
            file_row_sha256=_required_sha256(
                item.get("file_row_sha256"),
                "manifest_file_row_sha256_invalid",
            ),
        )
        if not isinstance(item.get("is_deleted"), bool):
            raise RecoveryVerificationError("manifest_deleted_state_invalid")
        metadata_error = reference.validation_error()
        if metadata_error is not None:
            raise RecoveryVerificationError(f"manifest_{metadata_error}")
        if reference.file_id in seen_ids:
            raise RecoveryVerificationError("manifest_duplicate_file_id")
        object_refs = (
            (reference.bucket, reference.payload_path),
            (reference.bucket, reference.wrapped_key_path),
        )
        if any(value in seen_objects for value in object_refs):
            raise RecoveryVerificationError("manifest_duplicate_object_reference")
        seen_ids.add(reference.file_id)
        seen_objects.update(object_refs)
        expectation = StoredObjectExpectation(
            payload_size=_required_positive_int(
                item.get("payload_size"),
                "manifest_payload_size_invalid",
                allow_zero=True,
            ),
            payload_sha256=_required_sha256(
                item.get("payload_sha256"),
                "manifest_payload_sha256_invalid",
            ),
            wrapped_key_size=_required_positive_int(
                item.get("wrapped_key_size"),
                "manifest_wrapped_key_size_invalid",
                allow_zero=True,
            ),
            wrapped_key_sha256=_required_sha256(
                item.get("wrapped_key_sha256"),
                "manifest_wrapped_key_sha256_invalid",
            ),
            max_object_bytes=_MAX_CAPTURED_OBJECT_BYTES,
        )
        if (
            expectation.payload_size > _MAX_CAPTURED_OBJECT_BYTES
            or expectation.wrapped_key_size > _MAX_CAPTURED_OBJECT_BYTES
        ):
            raise RecoveryVerificationError("manifest_object_too_large")
        captured.append(CapturedFile(reference=reference, expectation=expectation))
    captured.sort(key=lambda item: item.reference.file_id)
    if not {item.reference.user_id for item in captured}.issubset(seen_tenant_ids):
        raise RecoveryVerificationError("manifest_file_tenant_missing")
    return RecoveryManifest(
        recovery_id=recovery_id,
        source_revision=source_revision,
        schema_head=schema_head,
        rsa_public_key_sha256=rsa_fingerprint,
        captured_files=tuple(captured),
        tenants=tuple(tenants),
        manifest_sha256=manifest_sha256,
    )


def load_database_snapshot(
    *,
    connection_factory: Callable[[], Any] = get_connection,
) -> DatabaseSnapshot:
    """Read all canonical file rows, stable tenant identities, and schema head."""

    connection = None
    try:
        connection = connection_factory()
        cursor = connection.cursor()
        # One snapshot must bind all three queries, and timestamp JSON must not
        # vary with the database role/server session timezone after restore.
        cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        cursor.execute("SET LOCAL TIME ZONE 'UTC'")
        cursor.execute(
            """
            -- Hash the whole physical row instead of maintaining a lossy
            -- verifier-side column allowlist. The manifest stores only this
            -- digest plus the object-delivery fields required below.
            SELECT f.id, f.user_id, f.is_deleted, f.s3_bucket_name, f.s3_path,
                   f.s3_key_path, f.sha256_hash,
                   encode(
                       digest(convert_to(to_jsonb(f)::text, 'UTF8'), 'sha256'),
                       'hex'
                   ) AS file_row_sha256
            FROM files AS f
            ORDER BY f.id
            """
        )
        rows = cursor.fetchall()
        cursor.execute(
            """
            -- Bind stable ownership without exporting password hashes, API
            -- keys, avatar data, permission state, or the identity text.
            SELECT u.id AS user_id,
                   encode(
                       digest(
                           convert_to(
                               jsonb_build_object(
                                   'id', u.id,
                                   'username', u.username,
                                   'email', u.email,
                                   'created_at', u.created_at,
                                   'google_user_id', u.google_user_id
                               )::text,
                               'UTF8'
                           ),
                           'sha256'
                       ),
                       'hex'
                   ) AS stable_identity_sha256
            FROM users AS u
            ORDER BY u.id
            """
        )
        tenant_rows = cursor.fetchall()
        cursor.execute("SELECT COALESCE(MAX(version), 0) AS schema_head FROM public.schema_migrations")
        schema_row = cursor.fetchone()
        files = tuple(
            FileReference(
                file_id=int(row["id"]),
                user_id=int(row["user_id"]),
                is_deleted=bool(row["is_deleted"]),
                bucket=_text(row["s3_bucket_name"]),
                payload_path=_text(row["s3_path"]),
                wrapped_key_path=_text(row["s3_key_path"]),
                plaintext_sha256=_text(row["sha256_hash"]).lower(),
                file_row_sha256=_text(row["file_row_sha256"]).lower(),
            )
            for row in rows
        )
        tenants = tuple(
            TenantIdentity(
                user_id=int(row["user_id"]),
                stable_identity_sha256=_text(row["stable_identity_sha256"]).lower(),
            )
            for row in tenant_rows
        )
        schema_head = int(schema_row["schema_head"] if schema_row else 0)
        return DatabaseSnapshot(files=files, tenants=tenants, schema_head=schema_head)
    except Exception as exc:
        raise RecoveryVerificationError("database_read_failed") from exc
    finally:
        if connection is not None:
            try:
                connection.rollback()
            finally:
                put_connection(connection)


def load_file_references(
    *,
    connection_factory: Callable[[], Any] = get_connection,
) -> tuple[FileReference, ...]:
    """Compatibility helper returning the complete read-only snapshot row set."""

    return load_database_snapshot(connection_factory=connection_factory).files


async def verify_file_references(
    records: tuple[FileReference, ...],
    *,
    captured_files: tuple[CapturedFile, ...],
    observed_tenants: tuple[TenantIdentity, ...],
    captured_tenants: tuple[TenantIdentity, ...],
    access: ObjectAccess,
    recovery_id: str,
    manifest_sha256: str,
    source_revision: str,
    expected_schema_head: int,
    observed_schema_head: int,
) -> VerificationReport:
    """Verify only references already proven identical to the capture manifest."""

    issues: list[VerificationIssue] = []
    verified = 0
    expected_records = tuple(item.reference for item in captured_files)
    if records != expected_records:
        issues.append(VerificationIssue(category="database_rowset_mismatch"))
    if observed_tenants != captured_tenants:
        issues.append(VerificationIssue(category="database_tenantset_mismatch"))
    if observed_schema_head != expected_schema_head:
        issues.append(VerificationIssue(category="database_schema_head_mismatch"))
    for record in records:
        metadata_error = record.validation_error()
        if metadata_error is not None:
            issues.append(VerificationIssue(category=metadata_error, file_id=record.file_id))
    for tenant in observed_tenants:
        metadata_error = tenant.validation_error()
        if metadata_error is not None:
            issues.append(VerificationIssue(category=f"tenant_{metadata_error}"))
    if not issues:
        for record, captured in zip(records, captured_files, strict=True):
            try:
                await access.verify(record.object_ref(), captured.expectation)
            except StoredObjectUnavailable:
                issues.append(VerificationIssue(category="object_unavailable", file_id=record.file_id))
            except StoredObjectIntegrityError:
                issues.append(
                    VerificationIssue(category="decrypt_or_hash_failed", file_id=record.file_id)
                )
            except Exception:
                issues.append(
                    VerificationIssue(category="unexpected_object_error", file_id=record.file_id)
                )
            else:
                verified += 1

    return VerificationReport(
        recovery_id=recovery_id,
        manifest_sha256=manifest_sha256,
        source_revision=source_revision,
        expected_files=len(captured_files),
        referenced_files=len(records),
        verified_files=verified,
        live_files=sum(not item.is_deleted for item in records),
        trashed_files=sum(item.is_deleted for item in records),
        tenant_count=len(observed_tenants),
        expected_schema_head=expected_schema_head,
        observed_schema_head=observed_schema_head,
        database_rowset_sha256=_rowset_sha256(records),
        final_database_rowset_sha256="",
        database_tenantset_sha256=_tenantset_sha256(observed_tenants),
        final_database_tenantset_sha256="",
        issues=tuple(issues),
    )


def rsa_public_key_sha256(private_key_path: Path) -> str:
    try:
        private_key = serialization.load_pem_private_key(private_key_path.read_bytes(), password=None)
        public_der = private_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    except Exception as exc:
        raise RecoveryVerificationError("private_key_invalid") from exc
    return _sha256_bytes(public_der)


def build_minio_client() -> Minio:
    """Build a MinIO client without performing any object-store mutation."""

    missing = [
        name
        for name, value in (
            ("S3_ACCESS_KEY", settings.s3_access_key),
            ("S3_SECRET_KEY", settings.s3_secret_key),
        )
        if not value
    ]
    if missing:
        raise RecoveryVerificationError(f"missing_configuration:{','.join(missing)}")
    endpoint = settings.s3_endpoint.removeprefix("https://").removeprefix("http://")
    try:
        return Minio(
            endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_secure,
        )
    except Exception as exc:
        raise RecoveryVerificationError("object_client_configuration_failed") from exc


def build_object_access() -> VaultObjectAccess:
    if not settings.private_key.is_file():
        raise RecoveryVerificationError("private_key_unavailable")
    if not settings.encryption_script.is_file():
        raise RecoveryVerificationError("encryption_script_unavailable")
    return VaultObjectAccess(build_minio_client())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manifest-bound read-only verification for an isolated Lavix restore.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument(
        "--observed-source-revision",
        required=True,
        help=(
            "Trusted release-tree/OCI SHA-256 identity supplied independently "
            "of the recovery manifest."
        ),
    )
    parser.add_argument(
        "--confirm-isolated-restore",
        action="store_true",
        required=True,
        help="Confirm this restore has no production traffic or production data endpoints.",
    )
    args = parser.parse_args(argv)
    args.manifest_sha256 = args.manifest_sha256.lower()
    if not _SHA256_RE.fullmatch(args.manifest_sha256):
        parser.error("--manifest-sha256 must be a lowercase SHA-256 digest")
    if not _RECOVERY_ID_RE.fullmatch(args.recovery_id):
        parser.error("--recovery-id contains unsupported characters")
    if not _SOURCE_REVISION_RE.fullmatch(args.observed_source_revision):
        parser.error(
            "--observed-source-revision must be "
            "release-tree-sha256:<64 lowercase hex> or oci-image-sha256:<64 lowercase hex>"
        )
    return args


def _startup_failure(recovery_id: str, manifest_sha256: str, category: str) -> dict[str, Any]:
    return VerificationReport(
        recovery_id=recovery_id,
        manifest_sha256=manifest_sha256,
        source_revision="",
        expected_files=0,
        referenced_files=0,
        verified_files=0,
        live_files=0,
        trashed_files=0,
        tenant_count=0,
        expected_schema_head=0,
        observed_schema_head=0,
        database_rowset_sha256="",
        final_database_rowset_sha256="",
        database_tenantset_sha256="",
        final_database_tenantset_sha256="",
        issues=(VerificationIssue(category=category),),
    ).public_payload()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(
            args.manifest,
            expected_sha256=args.manifest_sha256,
            expected_recovery_id=args.recovery_id,
            expected_source_revision=args.observed_source_revision,
        )
        observed_key_fingerprint = rsa_public_key_sha256(settings.private_key)
        if observed_key_fingerprint != manifest.rsa_public_key_sha256:
            raise RecoveryVerificationError("rsa_public_key_mismatch")
        initial = load_database_snapshot()
        access = build_object_access()
        report = asyncio.run(
            verify_file_references(
                initial.files,
                captured_files=manifest.captured_files,
                observed_tenants=initial.tenants,
                captured_tenants=manifest.tenants,
                access=access,
                recovery_id=manifest.recovery_id,
                manifest_sha256=manifest.manifest_sha256,
                source_revision=manifest.source_revision,
                expected_schema_head=manifest.schema_head,
                observed_schema_head=initial.schema_head,
            )
        )
        final = load_database_snapshot()
        final_issues = list(report.issues)
        if (
            final.schema_head != initial.schema_head
            or final.rowset_sha256 != initial.rowset_sha256
            or final.tenantset_sha256 != initial.tenantset_sha256
        ):
            final_issues.append(VerificationIssue(category="database_changed_during_verification"))
        report = replace(
            report,
            final_database_rowset_sha256=final.rowset_sha256,
            final_database_tenantset_sha256=final.tenantset_sha256,
            issues=tuple(final_issues),
        )
    except RecoveryVerificationError as exc:
        print(
            json.dumps(
                _startup_failure(args.recovery_id, args.manifest_sha256, str(exc)),
                sort_keys=True,
            )
        )
        return 2
    except Exception:
        print(
            json.dumps(
                _startup_failure(
                    args.recovery_id,
                    args.manifest_sha256,
                    "unexpected_verifier_failure",
                ),
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report.public_payload(), sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
