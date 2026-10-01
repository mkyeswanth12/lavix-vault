"""Private temporary storage and cancellable subprocess execution."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .errors import (
    IngestionCancelled,
    ProcessExecutionError,
    ProcessOutputLimitError,
    ProcessTimeoutError,
)


class SecureTempLease:
    """A mode-0700 job directory that is recursively removed on every exit path."""

    def __init__(self, root: Path | str, *, prefix: str = "ingestion-") -> None:
        self.root = Path(root)
        self.prefix = prefix
        self.path: Path | None = None

    def __enter__(self) -> SecureTempLease:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        raw = tempfile.mkdtemp(prefix=self.prefix, dir=str(self.root))
        self.path = Path(raw)
        os.chmod(self.path, 0o700)
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.cleanup()

    def _require_path(self) -> Path:
        if self.path is None:
            raise RuntimeError("temporary lease has not been entered")
        return self.path

    def allocate(self, name: str) -> Path:
        """Return a safe direct child path without creating it."""

        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("temporary child names must be plain basenames")
        return self._require_path() / name

    def mkdir(self, name: str) -> Path:
        path = self.allocate(name)
        path.mkdir(mode=0o700)
        return path

    def cleanup(self) -> None:
        path, self.path = self.path, None
        if path is None:
            return
        try:
            resolved_root = self.root.resolve()
            resolved_parent = path.resolve(strict=False).parent
            if resolved_parent != resolved_root:
                raise RuntimeError("refusing to clean a path outside the temp root")
            shutil.rmtree(path)
        except FileNotFoundError:
            return


@dataclass(frozen=True, slots=True)
class ProcessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    duration_seconds: float

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")


class AsyncProcessRunner:
    """Run an argv vector without a shell, bounding output, time and descendants."""

    def __init__(
        self,
        *,
        max_output_bytes: int = 8 * 1024 * 1024,
        terminate_grace_seconds: float = 3.0,
    ) -> None:
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        self.max_output_bytes = max_output_bytes
        self.terminate_grace_seconds = terminate_grace_seconds

    async def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
        cancel_event: asyncio.Event | None = None,
        check: bool = False,
    ) -> ProcessResult:
        argv = tuple(os.fspath(arg) for arg in args)
        if not argv or any(not arg or "\x00" in arg for arg in argv):
            raise ValueError("process argv contains an invalid value")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if cancel_event and cancel_event.is_set():
            raise IngestionCancelled("process cancelled before start")

        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=os.fspath(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=(os.name == "posix"),
        )
        overflow = asyncio.Event()
        stdout_task = asyncio.create_task(self._read_limited(process.stdout, overflow))
        stderr_task = asyncio.create_task(self._read_limited(process.stderr, overflow))
        wait_task = asyncio.create_task(process.wait())
        overflow_task = asyncio.create_task(overflow.wait())
        cancel_task = asyncio.create_task(cancel_event.wait()) if cancel_event else None
        watchers = {wait_task, overflow_task}
        if cancel_task:
            watchers.add(cancel_task)

        reason: str | None = None
        try:
            done, _pending = await asyncio.wait(
                watchers,
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait_task in done:
                pass
            elif (
                cancel_task is not None
                and cancel_task in done
                and cancel_event is not None
                and cancel_event.is_set()
            ):
                reason = "cancelled"
            elif overflow_task in done and overflow.is_set():
                reason = "output_limit"
            elif not done:
                reason = "timeout"

            if reason:
                await self._terminate_group(process)
            returncode = await wait_task
            stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
            if reason is None and overflow.is_set():
                reason = "output_limit"
        except asyncio.CancelledError:
            await self._terminate_group(process)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise
        finally:
            for task in (overflow_task, cancel_task):
                if task is not None and not task.done():
                    task.cancel()

        result = ProcessResult(
            args=argv,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
        )
        if reason == "cancelled":
            raise IngestionCancelled("process cancelled")
        if reason == "timeout":
            raise ProcessTimeoutError(f"process exceeded {timeout_seconds} seconds")
        if reason == "output_limit":
            raise ProcessOutputLimitError("process output exceeded the configured limit")
        if check and result.returncode != 0:
            raise ProcessExecutionError(f"process exited with status {result.returncode}", result=result)
        return result

    async def _read_limited(
        self,
        stream: asyncio.StreamReader | None,
        overflow: asyncio.Event,
    ) -> bytes:
        if stream is None:
            return b""
        kept = bytearray()
        total = 0
        while True:
            chunk = await stream.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if len(kept) < self.max_output_bytes:
                remaining = self.max_output_bytes - len(kept)
                kept.extend(chunk[:remaining])
            if total > self.max_output_bytes:
                overflow.set()
        return bytes(kept)

    async def _terminate_group(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=self.terminate_grace_seconds)
            return
        except TimeoutError:
            pass
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return
        await process.wait()
