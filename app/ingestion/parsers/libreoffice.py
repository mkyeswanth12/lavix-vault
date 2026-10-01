"""Sandbox-friendly LibreOffice normalization for legacy binary Office files."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import (
    CorruptDocumentError,
    ParseError,
    ParserUnavailableError,
    PasswordRequiredError,
)
from ..models import CanonicalDocument
from ..security import AsyncProcessRunner, SecureTempLease
from .base import ParseRequest
from .docling import DoclingParser

_FILTERS = {
    ".docx": "docx:Office Open XML Text",
    ".xlsx": "xlsx:Calc MS Excel 2007 XML",
    ".pptx": "pptx:Impress MS PowerPoint 2007 XML",
}


@dataclass(frozen=True, slots=True)
class LibreOfficeSettings:
    executable: str = "soffice"
    version: str = "runtime"
    timeout_seconds: float = 300.0

    @property
    def fingerprint(self) -> str:
        return f"libreoffice:{self.version}:headless-safe-profile"


class LibreOfficeDoclingParser:
    def __init__(
        self,
        *,
        target_suffix: str,
        temp_root: Path,
        docling: DoclingParser,
        runner: AsyncProcessRunner | None = None,
        settings: LibreOfficeSettings | None = None,
    ) -> None:
        if target_suffix not in _FILTERS:
            raise ValueError(f"unsupported LibreOffice target: {target_suffix}")
        self.target_suffix = target_suffix
        self.temp_root = temp_root
        self.docling = docling
        self.runner = runner or AsyncProcessRunner()
        self.settings = settings or LibreOfficeSettings()
        self._runtime_fingerprint: str | None = None

    async def parse(
        self,
        request: ParseRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> CanonicalDocument:
        with SecureTempLease(self.temp_root, prefix="libreoffice-") as lease:
            output_dir = lease.mkdir("output")
            profile_dir = lease.mkdir("profile")
            parser_fingerprint = await self._fingerprint(profile_dir, cancel_event)
            args = [
                self.settings.executable,
                "--headless",
                "--safe-mode",
                "--nologo",
                "--nodefault",
                "--nofirststartwizard",
                "--nolockcheck",
                "--norestore",
                f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
                "--convert-to",
                _FILTERS[self.target_suffix],
                "--outdir",
                str(output_dir),
                str(request.path),
            ]
            try:
                result = await self.runner.run(
                    args,
                    cwd=lease.path,
                    env=self._environment(profile_dir),
                    timeout_seconds=self.settings.timeout_seconds,
                    cancel_event=cancel_event,
                )
            except FileNotFoundError as exc:
                raise ParserUnavailableError("LibreOffice is unavailable") from exc
            if result.returncode != 0:
                diagnostic = f"{result.stdout_text}\n{result.stderr_text}".lower()
                if "password" in diagnostic or "encrypted" in diagnostic:
                    raise PasswordRequiredError("document requires a password")
                raise CorruptDocumentError(
                    f"LibreOffice conversion exited with status {result.returncode}",
                    detail=result.stderr_text[-2_000:],
                )
            candidates = [
                path
                for path in output_dir.iterdir()
                if path.is_file() and not path.is_symlink() and path.suffix.lower() == self.target_suffix
            ]
            if len(candidates) != 1 or candidates[0].stat().st_size == 0:
                raise ParseError("LibreOffice did not produce one non-empty normalized file")
            converted = request.with_path(candidates[0], parser=parser_fingerprint)
            return await self.docling.parse(converted, cancel_event=cancel_event)

    async def _fingerprint(
        self,
        profile_dir: Path,
        cancel_event: asyncio.Event | None,
    ) -> str:
        if self.settings.version != "runtime":
            return self.settings.fingerprint
        if self._runtime_fingerprint is not None:
            return self._runtime_fingerprint
        try:
            result = await self.runner.run(
                [self.settings.executable, "--version"],
                env=self._environment(profile_dir),
                timeout_seconds=15,
                cancel_event=cancel_event,
            )
        except FileNotFoundError as exc:
            raise ParserUnavailableError("LibreOffice is unavailable") from exc
        if result.returncode != 0:
            raise ParserUnavailableError("LibreOffice version probe failed")
        raw_version = result.stdout_text.strip() or result.stderr_text.strip()
        normalized = re.sub(r"[^A-Za-z0-9._+-]+", "_", raw_version).strip("_")[:120]
        if not normalized:
            raise ParserUnavailableError("LibreOffice returned no version")
        self._runtime_fingerprint = f"libreoffice:{normalized}:headless-safe-profile"
        return self._runtime_fingerprint

    @staticmethod
    def _environment(profile_dir: Path) -> dict[str, str]:
        # Do not inherit proxy or model credentials into a document converter.
        keep = ("PATH", "LANG", "LC_ALL", "TZ")
        env = {key: os.environ[key] for key in keep if key in os.environ}
        env.update(
            {
                "HOME": str(profile_dir),
                "TMPDIR": str(profile_dir.parent),
                "SAL_USE_VCLPLUGIN": "svp",
            }
        )
        return env
