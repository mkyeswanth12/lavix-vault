from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import scripts.verify_restored_vault as verifier
from app.storage.access import StoredObjectExpectation, StoredObjectIntegrityError, StoredObjectUnavailable
from scripts.verify_restored_vault import (
    CapturedFile,
    FileReference,
    RecoveryVerificationError,
    TenantIdentity,
    load_database_snapshot,
    load_manifest,
    verify_file_references,
)


class FakeCursor:
    def __init__(self, rows, tenants=None, schema_head: int = 9):
        self.rows = rows
        self.tenants = tenants or []
        self.schema_head = schema_head
        self.commands: list[str] = []
        self.fetchall_calls = 0

    def execute(self, command):
        self.commands.append(" ".join(command.split()))

    def fetchall(self):
        self.fetchall_calls += 1
        return self.rows if self.fetchall_calls == 1 else self.tenants

    def fetchone(self):
        return {"schema_head": self.schema_head}


class FakeConnection:
    def __init__(self, rows, tenants=None, schema_head: int = 9):
        self.cursor_instance = FakeCursor(rows, tenants, schema_head)
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class FakeAccess:
    def __init__(self, outcomes: dict[str, str] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.verified: list[str] = []

    async def verify(self, ref, expectation):
        outcome = self.outcomes.get(ref.payload_path, "ok")
        if outcome == "unavailable":
            raise StoredObjectUnavailable("redacted")
        if outcome == "integrity":
            raise StoredObjectIntegrityError("redacted")
        assert expectation.payload_size == 9
        self.verified.append(ref.payload_path)


def _record(
    file_id: int,
    payload: str | None = None,
    *,
    user_id: int = 3,
    is_deleted: bool = False,
) -> FileReference:
    payload = payload or f"objects/{file_id}/payload.enc"
    return FileReference(
        file_id=file_id,
        user_id=user_id,
        is_deleted=is_deleted,
        bucket="restored-vault",
        payload_path=payload,
        wrapped_key_path=f"{payload}.key.rsa4096",
        plaintext_sha256=hashlib.sha256(b"verified").hexdigest(),
        file_row_sha256=hashlib.sha256(f"file-row:{file_id}".encode()).hexdigest(),
    )


def _tenant(user_id: int) -> TenantIdentity:
    return TenantIdentity(
        user_id=user_id,
        stable_identity_sha256=hashlib.sha256(f"tenant:{user_id}".encode()).hexdigest(),
    )


def _captured(record: FileReference) -> CapturedFile:
    return CapturedFile(
        reference=record,
        expectation=StoredObjectExpectation(
            payload_size=9,
            payload_sha256="a" * 64,
            wrapped_key_size=9,
            wrapped_key_sha256="b" * 64,
        ),
    )


def _manifest_payload(records: tuple[FileReference, ...]) -> dict:
    tenants = tuple(_tenant(user_id) for user_id in sorted({record.user_id for record in records}))
    return {
        "schema": "lavix.recovery-manifest.v2",
        "recovery_id": "lavix-20260716T120000Z",
        "source_revision": f"release-tree-sha256:{'d' * 64}",
        "schema_head": 9,
        "rsa_public_key_sha256": "c" * 64,
        "tenants": [tenant.identity() for tenant in tenants],
        "files": [
            {
                **record.identity(),
                "payload_size": 9,
                "payload_sha256": "a" * 64,
                "wrapped_key_size": 9,
                "wrapped_key_sha256": "b" * 64,
            }
            for record in records
        ],
    }


def _write_manifest(path: Path, records: tuple[FileReference, ...]) -> str:
    raw = json.dumps(_manifest_payload(records), sort_keys=True).encode()
    path.write_bytes(raw)
    os.chmod(path, 0o600)
    return hashlib.sha256(raw).hexdigest()


def test_load_database_snapshot_is_read_only_and_includes_tenant_trash_state() -> None:
    connection = FakeConnection(
        [
            {
                "id": 7,
                "user_id": 11,
                "is_deleted": True,
                "s3_bucket_name": "restored-vault",
                "s3_path": "users/11/payload.enc",
                "s3_key_path": "users/11/payload.key.rsa4096",
                "sha256_hash": "a" * 64,
                "file_row_sha256": "b" * 64,
            }
        ],
        tenants=[{"user_id": 11, "stable_identity_sha256": "c" * 64}],
    )

    snapshot = load_database_snapshot(connection_factory=lambda: connection)

    assert snapshot.files[0].file_id == 7
    assert snapshot.files[0].user_id == 11
    assert snapshot.files[0].is_deleted is True
    assert snapshot.files[0].file_row_sha256 == "b" * 64
    assert snapshot.tenants == (TenantIdentity(11, "c" * 64),)
    assert snapshot.schema_head == 9
    assert connection.cursor_instance.commands[0] == (
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    )
    assert connection.cursor_instance.commands[1] == "SET LOCAL TIME ZONE 'UTC'"
    assert "to_jsonb(f)" in connection.cursor_instance.commands[2]
    assert "FROM files AS f" in connection.cursor_instance.commands[2]
    assert "WHERE" not in connection.cursor_instance.commands[2]
    assert "jsonb_build_object" in connection.cursor_instance.commands[3]
    assert "FROM users AS u" in connection.cursor_instance.commands[3]
    assert "password_hash" not in connection.cursor_instance.commands[3]
    assert "schema_migrations" in connection.cursor_instance.commands[4]
    assert connection.rolled_back is True


def test_manifest_is_bound_to_digest_recovery_id_permissions_and_exact_rows(tmp_path) -> None:
    manifest_path = tmp_path / "capture.json"
    records = (_record(1), _record(2, user_id=4, is_deleted=True))
    digest = _write_manifest(manifest_path, records)

    manifest = load_manifest(
        manifest_path,
        expected_sha256=digest,
        expected_recovery_id="lavix-20260716T120000Z",
        expected_source_revision=f"release-tree-sha256:{'d' * 64}",
    )

    assert manifest.files == records
    assert manifest.manifest_sha256 == digest
    assert manifest.schema_head == 9
    assert manifest.tenants == (_tenant(3), _tenant(4))
    with pytest.raises(RecoveryVerificationError, match="source_revision_mismatch"):
        load_manifest(
            manifest_path,
            expected_sha256=digest,
            expected_recovery_id="lavix-20260716T120000Z",
            expected_source_revision=f"oci-image-sha256:{'e' * 64}",
        )
    with pytest.raises(RecoveryVerificationError, match="manifest_digest_mismatch"):
        load_manifest(
            manifest_path,
            expected_sha256="f" * 64,
            expected_recovery_id="lavix-20260716T120000Z",
            expected_source_revision=f"release-tree-sha256:{'d' * 64}",
        )
    os.chmod(manifest_path, 0o644)
    with pytest.raises(RecoveryVerificationError, match="manifest_permissions_too_open"):
        load_manifest(
            manifest_path,
            expected_sha256=digest,
            expected_recovery_id="lavix-20260716T120000Z",
            expected_source_revision=f"release-tree-sha256:{'d' * 64}",
        )


def test_source_mismatch_fails_before_private_key_database_or_object_access(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    manifest_path = tmp_path / "capture.json"
    digest = _write_manifest(manifest_path, (_record(1),))
    touched: list[str] = []
    monkeypatch.setattr(
        verifier,
        "rsa_public_key_sha256",
        lambda _path: touched.append("private-key"),
    )
    monkeypatch.setattr(
        verifier,
        "load_database_snapshot",
        lambda: touched.append("database"),
    )
    monkeypatch.setattr(
        verifier,
        "build_object_access",
        lambda: touched.append("object-store"),
    )

    exit_code = verifier.main(
        [
            "--manifest",
            str(manifest_path),
            "--manifest-sha256",
            digest,
            "--recovery-id",
            "lavix-20260716T120000Z",
            "--observed-source-revision",
            f"oci-image-sha256:{'e' * 64}",
            "--confirm-isolated-restore",
        ]
    )

    assert exit_code == 2
    assert touched == []
    assert json.loads(capsys.readouterr().out)["issue_counts"] == {
        "source_revision_mismatch": 1
    }


async def test_bulk_verifier_checks_every_exact_manifest_reference() -> None:
    access = FakeAccess()
    records = (_record(1, "one.enc"), _record(2, "two.enc", is_deleted=True))

    report = await verify_file_references(
        records,
        captured_files=tuple(_captured(record) for record in records),
        observed_tenants=(_tenant(3),),
        captured_tenants=(_tenant(3),),
        access=access,
        recovery_id="lavix-20260716T120000Z",
        manifest_sha256="d" * 64,
        source_revision=f"release-tree-sha256:{'d' * 64}",
        expected_schema_head=9,
        observed_schema_head=9,
    )

    assert report.ok is True
    assert report.verified_files == 2
    assert report.live_files == 1
    assert report.trashed_files == 1
    assert access.verified == ["one.enc", "two.enc"]


async def test_rowset_mismatch_fails_before_accessing_any_database_object() -> None:
    captured = _record(1, "captured.enc", user_id=3)
    restored = _record(1, "other.enc", user_id=4)
    access = FakeAccess()

    report = await verify_file_references(
        (restored,),
        captured_files=(_captured(captured),),
        observed_tenants=(_tenant(4),),
        captured_tenants=(_tenant(3),),
        access=access,
        recovery_id="lavix-20260716T120000Z",
        manifest_sha256="d" * 64,
        source_revision=f"release-tree-sha256:{'d' * 64}",
        expected_schema_head=9,
        observed_schema_head=9,
    )

    assert report.ok is False
    assert report.public_payload()["issue_counts"] == {
        "database_rowset_mismatch": 1,
        "database_tenantset_mismatch": 1,
    }
    assert access.verified == []


async def test_full_file_row_digest_mismatch_fails_before_object_access() -> None:
    captured = _record(1, "same.enc")
    restored = FileReference(
        **{
            **captured.identity(),
            "file_row_sha256": "f" * 64,
        }
    )
    access = FakeAccess()

    report = await verify_file_references(
        (restored,),
        captured_files=(_captured(captured),),
        observed_tenants=(_tenant(3),),
        captured_tenants=(_tenant(3),),
        access=access,
        recovery_id="lavix-20260716T120000Z",
        manifest_sha256="d" * 64,
        source_revision=f"release-tree-sha256:{'d' * 64}",
        expected_schema_head=9,
        observed_schema_head=9,
    )

    assert report.public_payload()["issue_counts"] == {"database_rowset_mismatch": 1}
    assert access.verified == []


async def test_bulk_verifier_continues_with_redacted_failure_evidence() -> None:
    access = FakeAccess({"missing.enc": "unavailable", "damaged.enc": "integrity"})
    records = (
        _record(1, "ok.enc"),
        _record(2, "missing.enc"),
        _record(3, "damaged.enc"),
    )

    report = await verify_file_references(
        records,
        captured_files=tuple(_captured(record) for record in records),
        observed_tenants=(_tenant(3),),
        captured_tenants=(_tenant(3),),
        access=access,
        recovery_id="lavix-20260716T120000Z",
        manifest_sha256="d" * 64,
        source_revision=f"release-tree-sha256:{'d' * 64}",
        expected_schema_head=9,
        observed_schema_head=9,
    )
    payload = report.public_payload()

    assert report.ok is False
    assert report.verified_files == 1
    assert payload["issue_counts"] == {
        "decrypt_or_hash_failed": 1,
        "object_unavailable": 1,
    }
    serialized = str(payload)
    assert "missing.enc" not in serialized
    assert "damaged.enc" not in serialized
