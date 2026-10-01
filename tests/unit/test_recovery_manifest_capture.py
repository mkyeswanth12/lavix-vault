from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.capture_recovery_manifest as capture
from scripts.verify_restored_vault import (
    MAX_CAPTURED_OBJECT_BYTES,
    DatabaseSnapshot,
    FileReference,
    TenantIdentity,
    load_manifest,
)


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)
        self.closed = False
        self.released = False

    def read(self, amount: int) -> bytes:
        return self._stream.read(amount)

    def close(self) -> None:
        self.closed = True
        self._stream.close()

    def release_conn(self) -> None:
        self.released = True


class FakeMinio:
    def __init__(
        self,
        objects: dict[tuple[str, str], bytes],
        *,
        changed_objects: set[tuple[str, str]] | None = None,
        change_after_calls: int = 1,
        size_override: int | None = None,
    ) -> None:
        self.objects = objects
        self.changed_objects = changed_objects or set()
        self.change_after_calls = change_after_calls
        self.size_override = size_override
        self.stat_calls: dict[tuple[str, str], int] = {}
        self.get_calls: list[tuple[str, str]] = []
        self.responses: list[FakeResponse] = []

    def stat_object(self, bucket: str, object_name: str):
        key = (bucket, object_name)
        calls = self.stat_calls.get(key, 0) + 1
        self.stat_calls[key] = calls
        suffix = (
            "changed"
            if key in self.changed_objects and calls > self.change_after_calls
            else "stable"
        )
        return SimpleNamespace(
            size=self.size_override if self.size_override is not None else len(self.objects[key]),
            etag=f"{key[1]}-{suffix}",
            version_id="v1",
        )

    def get_object(self, bucket: str, object_name: str) -> FakeResponse:
        key = (bucket, object_name)
        self.get_calls.append(key)
        response = FakeResponse(self.objects[key])
        self.responses.append(response)
        return response


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _record(file_id: int = 7, *, deleted: bool = False) -> FileReference:
    return FileReference(
        file_id=file_id,
        user_id=3,
        is_deleted=deleted,
        bucket="vault-private",
        payload_path=f"users/private-name/{file_id}/payload.enc",
        wrapped_key_path=f"users/private-name/{file_id}/payload.key.rsa4096",
        plaintext_sha256=_sha(f"plain:{file_id}"),
        file_row_sha256=_sha(f"row:{file_id}"),
    )


def _snapshot(*records: FileReference, schema_head: int = 9) -> DatabaseSnapshot:
    return DatabaseSnapshot(
        files=tuple(records),
        tenants=(TenantIdentity(user_id=3, stable_identity_sha256=_sha("private tenant")),),
        schema_head=schema_head,
    )


def _objects(records: tuple[FileReference, ...]) -> dict[tuple[str, str], bytes]:
    result: dict[tuple[str, str], bytes] = {}
    for record in records:
        result[(record.bucket, record.payload_path)] = f"encrypted:{record.file_id}".encode()
        result[(record.bucket, record.wrapped_key_path)] = f"wrapped:{record.file_id}".encode()
    return result


def test_stream_object_digest_hashes_exact_bytes_and_closes_response() -> None:
    client = FakeMinio({("vault", "object.enc"): b"exact encrypted bytes"})

    observed = capture.stream_object_digest(
        client,
        bucket="vault",
        object_name="object.enc",
    )

    assert observed.size == len(b"exact encrypted bytes")
    assert observed.sha256 == hashlib.sha256(b"exact encrypted bytes").hexdigest()
    assert client.stat_calls[("vault", "object.enc")] == 2
    assert client.responses[0].closed is True
    assert client.responses[0].released is True


def test_stream_object_digest_rejects_oversize_before_get_object() -> None:
    client = FakeMinio(
        {("vault", "object.enc"): b"small"},
        size_override=MAX_CAPTURED_OBJECT_BYTES + 1,
    )

    with pytest.raises(capture.RecoveryCaptureError, match="object_too_large"):
        capture.stream_object_digest(client, bucket="vault", object_name="object.enc")

    assert client.get_calls == []


def test_stream_object_digest_rejects_generation_change() -> None:
    client = FakeMinio(
        {("vault", "object.enc"): b"bytes"},
        changed_objects={("vault", "object.enc")},
    )

    with pytest.raises(capture.RecoveryCaptureError, match="object_changed_during_capture"):
        capture.stream_object_digest(client, bucket="vault", object_name="object.enc")


def test_stream_object_digest_rejects_disappearing_generation_metadata() -> None:
    class DisappearingEtagMinio(FakeMinio):
        def stat_object(self, bucket: str, object_name: str):
            metadata = super().stat_object(bucket, object_name)
            if self.stat_calls[(bucket, object_name)] > 1:
                metadata.etag = None
            return metadata

    client = DisappearingEtagMinio({("vault", "object.enc"): b"bytes"})

    with pytest.raises(capture.RecoveryCaptureError, match="object_changed_during_capture"):
        capture.stream_object_digest(client, bucket="vault", object_name="object.enc")


def test_stream_object_digest_requires_generation_metadata() -> None:
    class NoGenerationMinio(FakeMinio):
        def stat_object(self, bucket: str, object_name: str):
            key = (bucket, object_name)
            self.stat_calls[key] = self.stat_calls.get(key, 0) + 1
            return SimpleNamespace(size=len(self.objects[key]))

    client = NoGenerationMinio({("vault", "object.enc"): b"bytes"})

    with pytest.raises(capture.RecoveryCaptureError, match="object_generation_unavailable"):
        capture.stream_object_digest(client, bucket="vault", object_name="object.enc")

    assert client.get_calls == []


def test_capture_main_publishes_verifier_compatible_private_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = (_record(7), _record(8, deleted=True))
    snapshot = _snapshot(*records)
    client = FakeMinio(_objects(records))
    snapshots = iter((snapshot, snapshot))
    monkeypatch.setattr(capture, "load_database_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(capture, "build_minio_client", lambda: client)
    monkeypatch.setattr(capture, "rsa_public_key_sha256", lambda _path: "c" * 64)
    output = tmp_path / "operator-private-capture.json"
    source_revision = f"release-tree-sha256:{'d' * 64}"

    exit_code = capture.main(
        [
            "--output",
            str(output),
            "--recovery-id",
            "lavix-20260716T140000Z",
            "--source-revision",
            source_revision,
            "--confirm-writers-stopped",
        ]
    )

    assert exit_code == 0
    assert output.stat().st_mode & 0o777 == 0o600
    raw = output.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = load_manifest(
        output,
        expected_sha256=digest,
        expected_recovery_id="lavix-20260716T140000Z",
        expected_source_revision=source_revision,
    )
    assert manifest.files == records
    assert manifest.schema_head == 9
    assert manifest.captured_files[0].expectation.payload_sha256 == hashlib.sha256(
        b"encrypted:7"
    ).hexdigest()

    receipt = json.loads(capsys.readouterr().out)
    assert set(receipt) == {
        "schema",
        "status",
        "recovery_id",
        "manifest_sha256",
        "output_path_sha256",
        "file_count",
        "live_file_count",
        "trashed_file_count",
        "tenant_count",
    }
    assert receipt["manifest_sha256"] == digest
    assert receipt["file_count"] == 2
    assert receipt["live_file_count"] == 1
    assert receipt["trashed_file_count"] == 1
    serialized_receipt = json.dumps(receipt)
    assert "private-name" not in serialized_receipt
    assert output.name not in serialized_receipt
    assert source_revision not in serialized_receipt
    assert all(response.closed and response.released for response in client.responses)


def test_capture_fails_closed_when_database_changes_and_writes_no_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = (_record(),)
    initial = _snapshot(*records)
    changed = _snapshot(*records, schema_head=10)
    snapshots = iter((initial, changed))
    monkeypatch.setattr(capture, "load_database_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(capture, "build_minio_client", lambda: FakeMinio(_objects(records)))
    monkeypatch.setattr(capture, "rsa_public_key_sha256", lambda _path: "c" * 64)
    output = tmp_path / "must-not-exist.json"

    exit_code = capture.main(
        [
            "--output",
            str(output),
            "--recovery-id",
            "lavix-20260716T140000Z",
            "--source-revision",
            f"oci-image-sha256:{'e' * 64}",
            "--confirm-writers-stopped",
        ]
    )

    assert exit_code == 2
    assert not output.exists()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["issue"] == "database_changed_during_capture"
    assert output.name not in json.dumps(receipt)
    assert "private-name" not in json.dumps(receipt)


def test_capture_fails_closed_when_rsa_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = (_record(),)
    snapshot = _snapshot(*records)
    snapshots = iter((snapshot, snapshot))
    key_fingerprints = iter(("c" * 64, "d" * 64))
    monkeypatch.setattr(capture, "load_database_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(capture, "build_minio_client", lambda: FakeMinio(_objects(records)))
    monkeypatch.setattr(
        capture,
        "rsa_public_key_sha256",
        lambda _path: next(key_fingerprints),
    )
    output = tmp_path / "must-not-exist.json"

    exit_code = capture.main(
        [
            "--output",
            str(output),
            "--recovery-id",
            "lavix-20260716T140000Z",
            "--source-revision",
            f"release-tree-sha256:{'d' * 64}",
            "--confirm-writers-stopped",
        ]
    )

    assert exit_code == 2
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["issue"] == (
        "private_key_changed_during_capture"
    )


def test_capture_fails_closed_when_object_changes_after_its_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = (_record(),)
    snapshot = _snapshot(*records)
    object_key = (records[0].bucket, records[0].payload_path)
    client = FakeMinio(
        _objects(records),
        changed_objects={object_key},
        change_after_calls=2,
    )
    monkeypatch.setattr(capture, "load_database_snapshot", lambda: snapshot)
    monkeypatch.setattr(capture, "build_minio_client", lambda: client)
    monkeypatch.setattr(capture, "rsa_public_key_sha256", lambda _path: "c" * 64)
    output = tmp_path / "must-not-exist.json"

    exit_code = capture.main(
        [
            "--output",
            str(output),
            "--recovery-id",
            "lavix-20260716T140000Z",
            "--source-revision",
            f"release-tree-sha256:{'d' * 64}",
            "--confirm-writers-stopped",
        ]
    )

    assert exit_code == 2
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["issue"] == (
        "object_changed_during_capture"
    )


def test_atomic_create_private_never_replaces_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_bytes(b"operator evidence")
    os.chmod(output, 0o600)

    with pytest.raises(capture.RecoveryCaptureError, match="manifest_output_exists"):
        capture.atomic_create_private(output, b"replacement", validate_staged=lambda _path: None)

    assert output.read_bytes() == b"operator evidence"
    assert not list(tmp_path.glob(".*.capture-*.tmp"))


def test_atomic_create_private_requires_private_real_output_directory(tmp_path: Path) -> None:
    public_parent = tmp_path / "public"
    public_parent.mkdir(mode=0o700)
    public_parent.chmod(0o755)

    with pytest.raises(
        capture.RecoveryCaptureError,
        match="manifest_output_parent_permissions_too_open",
    ):
        capture.atomic_create_private(
            public_parent / "capture.json",
            b"private",
            validate_staged=lambda _path: None,
        )

    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(private_parent, target_is_directory=True)
    with pytest.raises(capture.RecoveryCaptureError, match="manifest_output_parent_invalid"):
        capture.atomic_create_private(
            linked_parent / "capture.json",
            b"private",
            validate_staged=lambda _path: None,
        )


def test_capture_requires_explicit_writers_stopped_confirmation(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        capture.parse_args(
            [
                "--output",
                str(tmp_path / "capture.json"),
                "--recovery-id",
                "lavix-20260716T140000Z",
                "--source-revision",
                f"release-tree-sha256:{'d' * 64}",
            ]
        )

    assert exc.value.code == 2
