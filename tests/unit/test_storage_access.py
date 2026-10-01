from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.storage.access import (
    ObjectAccessSettings,
    StoredObjectExpectation,
    StoredObjectIntegrityError,
    StoredObjectRef,
    StoredObjectUnavailable,
    VaultObjectAccess,
)


class FakeRunner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, args, **_kwargs):
        self.calls += 1
        encrypted = Path(args[-1])
        encrypted.with_suffix("").write_bytes(b"plain text")
        return SimpleNamespace(returncode=0)


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
        *,
        fail_download: bool = False,
        fail_second_copy: bool = False,
        streamed_bytes: bytes = b"encrypted",
        stat_size: int | None = None,
    ) -> None:
        self.fail_download = fail_download
        self.fail_second_copy = fail_second_copy
        self.streamed_bytes = streamed_bytes
        self.stat_size = len(b"encrypted") if stat_size is None else stat_size
        self.copies: list[str] = []
        self.removed: list[str] = []
        self.responses: list[FakeResponse] = []

    def fget_object(self, _bucket, object_name, destination):
        if self.fail_download and object_name.endswith("payload.enc"):
            raise RuntimeError("missing")
        Path(destination).write_bytes(b"encrypted")

    def stat_object(self, _bucket, _object_name):
        return SimpleNamespace(size=self.stat_size)

    def get_object(self, _bucket, _object_name):
        if self.fail_download:
            raise RuntimeError("missing")
        response = FakeResponse(self.streamed_bytes)
        self.responses.append(response)
        return response

    def copy_object(self, _bucket, object_name, _source):
        if self.fail_second_copy and self.copies:
            raise RuntimeError("copy failed")
        self.copies.append(object_name)

    def remove_object(self, _bucket, object_name):
        self.removed.append(object_name)


def _settings(tmp_path) -> ObjectAccessSettings:
    private_key = tmp_path / "private.pem"
    private_key.write_text("test", encoding="utf-8")
    return ObjectAccessSettings(
        temp_root=tmp_path / "temp",
        private_key=private_key,
        encryption_script=tmp_path / "encryption.py",
    )


def _ref(prefix: str = "source") -> StoredObjectRef:
    return StoredObjectRef(
        bucket="vault",
        payload_path=f"{prefix}/payload.enc",
        wrapped_key_path=f"{prefix}/key.wrap",
        sha256=hashlib.sha256(b"plain text").hexdigest(),
    )


async def test_materialize_verifies_hash_and_returns_private_file(tmp_path) -> None:
    access = VaultObjectAccess(
        FakeMinio(),
        access_settings=_settings(tmp_path),
        process_runner=FakeRunner(),
    )
    path = await access.materialize(_ref())
    assert path.read_bytes() == b"plain text"
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list((tmp_path / "temp").glob("delivery-*"))
    access.cleanup(path)
    assert not path.exists()


async def test_materialize_maps_object_download_failure_and_cleans_temp(tmp_path) -> None:
    access = VaultObjectAccess(
        FakeMinio(fail_download=True),
        access_settings=_settings(tmp_path),
        process_runner=FakeRunner(),
    )
    with pytest.raises(StoredObjectUnavailable):
        await access.materialize(_ref())
    assert not list((tmp_path / "temp").glob("delivery-*"))
    assert not list((tmp_path / "temp" / "responses").glob("response-*"))


async def test_verify_binds_encrypted_bytes_and_leaves_no_plaintext(tmp_path) -> None:
    encrypted_hash = hashlib.sha256(b"encrypted").hexdigest()
    access = VaultObjectAccess(
        FakeMinio(),
        access_settings=_settings(tmp_path),
        process_runner=FakeRunner(),
    )
    expectation = StoredObjectExpectation(
        payload_size=len(b"encrypted"),
        payload_sha256=encrypted_hash,
        wrapped_key_size=len(b"encrypted"),
        wrapped_key_sha256=encrypted_hash,
    )

    await access.verify(_ref(), expectation)

    assert not list((tmp_path / "temp").glob("restore-verify-*"))
    assert not list((tmp_path / "temp").rglob("payload"))


async def test_verify_rejects_a_capture_digest_mismatch(tmp_path) -> None:
    access = VaultObjectAccess(
        FakeMinio(),
        access_settings=_settings(tmp_path),
        process_runner=FakeRunner(),
    )
    expectation = StoredObjectExpectation(
        payload_size=len(b"encrypted"),
        payload_sha256="f" * 64,
        wrapped_key_size=len(b"encrypted"),
        wrapped_key_sha256=hashlib.sha256(b"encrypted").hexdigest(),
    )

    with pytest.raises(StoredObjectIntegrityError):
        await access.verify(_ref(), expectation)

    assert not list((tmp_path / "temp").glob("restore-verify-*"))


@pytest.mark.parametrize(
    "streamed_bytes",
    [
        pytest.param(b"encrypted-extra", id="response-larger-than-capture"),
        pytest.param(b"short", id="response-shorter-than-capture"),
    ],
)
async def test_verify_bounds_a_lying_object_stream_and_cleans_up(
    tmp_path,
    streamed_bytes: bytes,
) -> None:
    minio = FakeMinio(streamed_bytes=streamed_bytes)
    runner = FakeRunner()
    access = VaultObjectAccess(
        minio,
        access_settings=_settings(tmp_path),
        process_runner=runner,
    )
    encrypted_hash = hashlib.sha256(b"encrypted").hexdigest()
    expectation = StoredObjectExpectation(
        payload_size=len(b"encrypted"),
        payload_sha256=encrypted_hash,
        wrapped_key_size=len(b"encrypted"),
        wrapped_key_sha256=encrypted_hash,
    )

    with pytest.raises(StoredObjectIntegrityError, match="differs from the capture"):
        await access.verify(_ref(), expectation)

    assert runner.calls == 0
    assert len(minio.responses) == 1
    assert minio.responses[0].closed is True
    assert minio.responses[0].released is True
    assert not list((tmp_path / "temp").glob("restore-verify-*"))
    assert not list((tmp_path / "temp").rglob("payload*"))


async def test_copy_compensates_payload_when_wrapped_key_copy_fails(tmp_path) -> None:
    minio = FakeMinio(fail_second_copy=True)
    access = VaultObjectAccess(minio, access_settings=_settings(tmp_path))
    with pytest.raises(RuntimeError, match="copy failed"):
        await access.copy(_ref("source"), _ref("destination"))
    assert minio.copies == ["destination/payload.enc"]
    assert minio.removed == ["destination/payload.enc", "destination/key.wrap"]
