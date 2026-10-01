"""Lazy construction of the parser selected by the pure routing layer."""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path

from ..errors import UnsupportedFormatError
from ..models import CanonicalDocument
from ..routing import ParserKind, ParserRoute
from ..security import AsyncProcessRunner, SecureTempLease
from .base import DocumentParser, ParseRequest
from .docling import DoclingParser, DoclingSettings
from .fastpdf import FastPdfFirstParser, FastPdfParser, FastPdfSettings
from .libreoffice import LibreOfficeDoclingParser, LibreOfficeSettings
from .opendataloader import (
    OpenDataLoaderEmptyFallbackParser,
    OpenDataLoaderPdfParser,
    OpenDataLoaderSettings,
)
from .text import DeterministicTextParser
from .vision import (
    ConfiguredOllamaVisionParser,
    OllamaVisionParser,
    VisionFallbackParser,
    VisionSettings,
)

_VISION_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True, slots=True)
class ParserSettings:
    temp_root: Path = Path("/tmp/vault_ingestion")
    opendataloader_executable: str = "opendataloader-pdf"
    opendataloader_version: str = "2.4.7"
    opendataloader_hybrid: str = "docling-fast"
    opendataloader_hybrid_url: str = "http://odl-hybrid:5002"
    opendataloader_hybrid_timeout_ms: int = 3_600_000
    parser_timeout_seconds: float = 7_200.0
    fastpdf_enabled: bool = True
    fastpdf_min_total_chars: int = 500
    fastpdf_min_chars_per_page: int = 100
    docling_version: str = "2.112.0"
    libreoffice_executable: str = "soffice"
    libreoffice_version: str = "runtime"

    @classmethod
    def from_environment(cls) -> ParserSettings:
        return cls(
            temp_root=Path(os.environ.get("INGESTION_TEMP_ROOT", "/tmp/vault_ingestion")),
            opendataloader_executable=os.environ.get("OPENDATALOADER_EXECUTABLE", "opendataloader-pdf"),
            opendataloader_version=os.environ.get("OPENDATALOADER_VERSION", "2.4.7"),
            opendataloader_hybrid=os.environ.get("OPENDATALOADER_HYBRID", "docling-fast"),
            opendataloader_hybrid_url=os.environ.get("OPENDATALOADER_HYBRID_URL", "http://odl-hybrid:5002"),
            opendataloader_hybrid_timeout_ms=int(
                os.environ.get("OPENDATALOADER_HYBRID_TIMEOUT_MS", "3600000")
            ),
            parser_timeout_seconds=float(os.environ.get("INGESTION_PARSER_TIMEOUT_SECONDS", "7200")),
            fastpdf_enabled=os.environ.get("FASTPDF_ENABLED", "true").lower() not in {"0", "false", "no"},
            fastpdf_min_total_chars=int(os.environ.get("FASTPDF_MIN_TOTAL_CHARS", "500")),
            fastpdf_min_chars_per_page=int(os.environ.get("FASTPDF_MIN_CHARS_PER_PAGE", "100")),
            docling_version=os.environ.get("DOCLING_VERSION", "2.112.0"),
            libreoffice_executable=os.environ.get("LIBREOFFICE_EXECUTABLE", "soffice"),
            libreoffice_version=os.environ.get("LIBREOFFICE_VERSION", "runtime"),
        )


class ParserRegistry:
    def __init__(
        self,
        settings: ParserSettings | None = None,
        *,
        runner: AsyncProcessRunner | None = None,
        vision_role_resolver: Callable[[], tuple[bool, str | None]] | None = None,
        vision_priority_gate: Callable[[], Awaitable[bool]] | None = None,
        vision_priority_timeout_seconds: float = 2.0,
    ) -> None:
        self.settings = settings or ParserSettings.from_environment()
        self.runner = runner or AsyncProcessRunner()
        self.vision_role_resolver = vision_role_resolver
        self.vision_priority_gate = vision_priority_gate
        self.vision_priority_timeout_seconds = vision_priority_timeout_seconds
        self._parsers: dict[tuple[ParserKind, str | None], DocumentParser] = {}

    def get(self, route: ParserRoute) -> DocumentParser:
        key = (route.parser, route.normalized_extension)
        if key in self._parsers:
            return self._parsers[key]
        parser: DocumentParser
        if route.parser is ParserKind.OPEN_DATALOADER_PDF:
            primary = OpenDataLoaderPdfParser(
                temp_root=self.settings.temp_root,
                runner=self.runner,
                settings=OpenDataLoaderSettings(
                    executable=self.settings.opendataloader_executable,
                    hybrid=self.settings.opendataloader_hybrid,
                    hybrid_url=self.settings.opendataloader_hybrid_url,
                    hybrid_timeout_ms=self.settings.opendataloader_hybrid_timeout_ms,
                    process_timeout_seconds=self.settings.parser_timeout_seconds,
                    version=self.settings.opendataloader_version,
                ),
            )
            parser = OpenDataLoaderEmptyFallbackParser(primary, self._docling())
            if self.settings.fastpdf_enabled:
                parser = FastPdfFirstParser(
                    FastPdfParser(
                        FastPdfSettings(
                            min_total_chars=self.settings.fastpdf_min_total_chars,
                            min_chars_per_page=self.settings.fastpdf_min_chars_per_page,
                        )
                    ),
                    parser,
                )
        elif route.parser is ParserKind.DOCLING:
            if route.normalized_extension in _VISION_EXTENSIONS:
                vision = (
                    ConfiguredOllamaVisionParser(
                        self.vision_role_resolver,
                        priority_gate=self.vision_priority_gate,
                        priority_timeout_seconds=self.vision_priority_timeout_seconds,
                    )
                    if self.vision_role_resolver is not None
                    else OllamaVisionParser(
                        VisionSettings.from_environment(),
                        priority_gate=self.vision_priority_gate,
                        priority_timeout_seconds=self.vision_priority_timeout_seconds,
                    )
                )
                parser = VisionFallbackParser(
                    self._docling(),
                    vision,
                )
            else:
                parser = self._docling()
        elif route.parser is ParserKind.LIBREOFFICE_DOCLING:
            if route.normalized_extension is None:
                raise ValueError("LibreOffice route lacks a target extension")
            parser = LibreOfficeDoclingParser(
                target_suffix=route.normalized_extension,
                temp_root=self.settings.temp_root,
                docling=self._docling(),
                runner=self.runner,
                settings=LibreOfficeSettings(
                    executable=self.settings.libreoffice_executable,
                    version=self.settings.libreoffice_version,
                ),
            )
        elif route.parser is ParserKind.TEXT:
            parser = DeterministicTextParser()
        else:
            raise UnsupportedFormatError(route.reason or "format is not indexable")
        self._parsers[key] = parser
        return parser

    async def parse(
        self,
        route: ParserRoute,
        request: ParseRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> CanonicalDocument:
        parser = self.get(route)
        normalized_extension = route.normalized_extension
        if (
            route.parser is not ParserKind.DOCLING
            or normalized_extension is None
            or request.path.suffix.lower() == normalized_extension
        ):
            return await parser.parse(request, cancel_event=cancel_event)
        if (
            not normalized_extension.startswith(".")
            or Path(f"source{normalized_extension}").suffix != normalized_extension
        ):
            raise ValueError("invalid normalized parser extension")
        with SecureTempLease(self.settings.temp_root, prefix="normalized-") as lease:
            normalized_path = lease.allocate(f"source{normalized_extension}")
            try:
                await asyncio.to_thread(os.link, request.path, normalized_path)
            except OSError:
                await asyncio.to_thread(shutil.copyfile, request.path, normalized_path)
            normalized_request = replace(request, path=normalized_path)
            return await parser.parse(normalized_request, cancel_event=cancel_event)

    def _docling(self) -> DoclingParser:
        key = (ParserKind.DOCLING, None)
        existing = self._parsers.get(key)
        if isinstance(existing, DoclingParser):
            return existing
        parser = DoclingParser(
            DoclingSettings(version=self.settings.docling_version),
            temp_root=self.settings.temp_root,
        )
        self._parsers[key] = parser
        return parser
