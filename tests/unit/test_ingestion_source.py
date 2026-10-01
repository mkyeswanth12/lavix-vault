import asyncio
import hashlib
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion.errors import IngestionCancelled, SourceUnavailableError
from app.ingestion.models import IngestionJob, JobState
from app.ingestion.security import ProcessResult
from app.ingestion.source import VaultEncryptedSourceProvider, VaultSourceSettings


class FakeMinio:
    def __init__(self):
        self.calls = []

    def fget_object(self, bucket, object_path, destination):
        self.calls.append((bucket, object_path, destination))
        Path(destination).write_bytes(b"key" if object_path.endswith("key") else b"encrypted")


class FakeDecryptRunner:
    def __init__(self, plaintext):
        self.plaintext = plaintext
        self.cancel_event = None

    async def run(self, args, **kwargs):
        self.cancel_event = kwargs.get("cancel_event")
        encrypted = Path(args[-1])
        encrypted.with_suffix("").write_bytes(self.plaintext)
        return ProcessResult(tuple(str(item) for item in args), 0, b"", b"", 0.01)


class PartiallyFailingMinio:
    def __init__(self):
        self.second_started = threading.Event()
        self.second_finished = False

    def fget_object(self, _bucket, object_path, _destination):
        if object_path.endswith("key"):
            self.second_started.set()
            time.sleep(0.05)
            self.second_finished = True
            return
        self.second_started.wait(timeout=1)
        raise OSError("synthetic download failure")


def job_for(plaintext):
    return IngestionJob(
        job_id="00000000-0000-0000-0000-000000000001",
        file_id=1,
        user_id=2,
        revision=1,
        state=JobState.DECRYPTING,
        source_name="report.pdf",
        media_type="application/pdf",
        source_sha256=hashlib.sha256(plaintext).hexdigest(),
        metadata={
            "s3_bucket_name": "vault",
            "s3_path": "user_2/report.pdf.enc",
            "s3_key_path": "user_2/report.pdf.key",
            "encrypted_filename": "report.pdf.enc",
            "rsa_key_filename": "report.pdf.key.rsa4096",
        },
    )


class VaultSourceProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_environment_requires_object_store_credentials(self):
        from app.config import reload_config

        with patch.dict(os.environ, {}, clear=True):
            # patch.dict also removes the fixture's LAVIX_CONFIG_FILE pointer,
            # so re-isolate explicitly before reading the config surface.
            os.environ["LAVIX_CONFIG_FILE"] = "/nonexistent/lavix-config.yml"
            reload_config()
            try:
                # The endpoint has a safe topology default; credentials do not.
                with self.assertRaisesRegex(ValueError, "S3_ACCESS_KEY"):
                    VaultSourceSettings.from_environment()
            finally:
                reload_config()

    async def test_plaintext_exists_only_for_source_context(self):
        plaintext = b"pdf bytes"
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            private_key = root / "private.pem"
            private_key.write_bytes(b"private")
            minio = FakeMinio()
            runner = FakeDecryptRunner(plaintext)
            provider = VaultEncryptedSourceProvider(
                VaultSourceSettings(
                    temp_root=root / "leases",
                    private_key=private_key,
                    encryption_script=root / "encryption.py",
                ),
                runner=runner,
                minio_client=minio,
            )
            event = asyncio.Event()
            async with provider.open(job_for(plaintext), cancel_event=event) as request:
                plaintext_path = request.path
                self.assertTrue(plaintext_path.is_file())
                self.assertEqual(plaintext_path.read_bytes(), plaintext)
                self.assertEqual(request.source_sha256, hashlib.sha256(plaintext).hexdigest())
            self.assertFalse(plaintext_path.exists())
            self.assertIs(runner.cancel_event, event)
            self.assertEqual(len(minio.calls), 2)

    async def test_preexisting_cancellation_avoids_download(self):
        plaintext = b"data"
        minio = FakeMinio()
        provider = VaultEncryptedSourceProvider(VaultSourceSettings(), minio_client=minio)
        event = asyncio.Event()
        event.set()
        with self.assertRaises(IngestionCancelled):
            async with provider.open(job_for(plaintext), cancel_event=event):
                pass
        self.assertEqual(minio.calls, [])

    async def test_partial_download_failure_waits_for_both_threads(self):
        with tempfile.TemporaryDirectory() as raw:
            minio = PartiallyFailingMinio()
            provider = VaultEncryptedSourceProvider(
                VaultSourceSettings(temp_root=Path(raw)),
                minio_client=minio,
            )
            with self.assertRaises(SourceUnavailableError):
                async with provider.open(job_for(b"data")):
                    pass
            self.assertTrue(minio.second_finished)


if __name__ == "__main__":
    unittest.main()

