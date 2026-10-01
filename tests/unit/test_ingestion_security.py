import asyncio
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from app.ingestion.errors import (
    IngestionCancelled,
    ProcessOutputLimitError,
    ProcessTimeoutError,
)
from app.ingestion.security import AsyncProcessRunner, SecureTempLease


class SecureTempLeaseTests(unittest.TestCase):
    def test_private_directory_cleanup_and_path_validation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "leases"
            with SecureTempLease(root) as lease:
                path = lease.path
                self.assertIsNotNone(path)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
                lease.allocate("payload.bin").write_bytes(b"secret")
                with self.assertRaises(ValueError):
                    lease.allocate("../escape")
            self.assertFalse(path.exists())
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)


@unittest.skipIf(os.name != "posix", "process-group assertions require POSIX")
class AsyncProcessRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_success(self):
        result = await AsyncProcessRunner().run([sys.executable, "-c", "print('ok')"], check=True)
        self.assertEqual(result.stdout_text.strip(), "ok")

    async def test_timeout_terminates_process(self):
        with self.assertRaises(ProcessTimeoutError):
            await AsyncProcessRunner(terminate_grace_seconds=0.1).run(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                timeout_seconds=0.05,
            )

    async def test_cancellation_terminates_process(self):
        event = asyncio.Event()

        async def cancel():
            await asyncio.sleep(0.05)
            event.set()

        task = asyncio.create_task(cancel())
        with self.assertRaises(IngestionCancelled):
            await AsyncProcessRunner(terminate_grace_seconds=0.1).run(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                cancel_event=event,
            )
        await task

    async def test_output_is_bounded(self):
        with self.assertRaises(ProcessOutputLimitError):
            await AsyncProcessRunner(max_output_bytes=32).run(
                [sys.executable, "-c", "print('x' * 10000)"],
                timeout_seconds=2,
            )


if __name__ == "__main__":
    unittest.main()
