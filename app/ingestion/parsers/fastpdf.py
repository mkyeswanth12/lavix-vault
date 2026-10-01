"""Fast-path PDF text extraction for digital PDFs.

Tries dependency-free text extraction (pypdfium2, already pinned via the
Docling/ODL stack) before the heavy OpenDataLoader/Docling OCR chain. If the
extracted text is too thin to index (scanned PDF), raises FastPdfThinError so
the caller falls through to the full OCR chain unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from ..errors import (
    CorruptDocumentError,
    EmptyDocumentError,
    IngestionCancelled,
    ParseError,
    ParserUnavailableError,
    PasswordRequiredError,
)
from ..models import (
    CanonicalDocument,
    CanonicalElement,
    ElementType,
    Provenance,
    build_element_id,
    normalize_text,
)
from .base import DocumentParser, ParseRequest

logger = logging.getLogger(__name__)


class FastPdfThinError(ParseError):
    """Extracted text is too thin; fall through to the OCR chain."""


@dataclass(frozen=True, slots=True)
class FastPdfSettings:
    min_total_chars: int = 500
    min_chars_per_page: int = 100

    @property
    def fingerprint(self) -> str:
        try:
            from importlib.metadata import version

            pdfium_version = version("pypdfium2")
        except Exception:  # pragma: no cover - pinned dep is always installed
            pdfium_version = "unknown"
        return f"fastpdf:{pdfium_version}"

    @classmethod
    def from_environment(cls) -> FastPdfSettings:
        return cls(
            min_total_chars=int(os.environ.get("FASTPDF_MIN_TOTAL_CHARS", "500")),
            min_chars_per_page=int(os.environ.get("FASTPDF_MIN_CHARS_PER_PAGE", "100")),
        )


def _extract_pages(path: str, cancel_event: asyncio.Event | None) -> list[str]:
    """Blocking pypdfium2 extraction; runs inside a worker thread."""
    try:
        import pypdfium2 as pdfium
    except ModuleNotFoundError as exc:
        raise ParserUnavailableError("pypdfium2 is unavailable") from exc
    try:
        document = pdfium.PdfDocument(path)
    except Exception as exc:
        message = str(exc).lower()
        if "password" in message or "encrypted" in message:
            raise PasswordRequiredError("document requires a password") from exc
        raise CorruptDocumentError("fast PDF open failed", detail=str(exc)[:2_000]) from exc
    pages: list[str] = []
    try:
        count = len(document)
        for index in range(count):
            if cancel_event is not None and index % 16 == 0 and cancel_event.is_set():
                raise IngestionCancelled()
            page = document[index]
            try:
                textpage = page.get_textpage()
                try:
                    pages.append(textpage.get_text_range() or "")
                finally:
                    textpage.close()
            finally:
                page.close()
    finally:
        document.close()
    return pages


class FastPdfParser:
    """Extract indexable text from digital PDFs without OCR."""

    def __init__(self, settings: FastPdfSettings | None = None) -> None:
        self.settings = settings or FastPdfSettings.from_environment()

    async def parse(
        self,
        request: ParseRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> CanonicalDocument:
        if cancel_event and cancel_event.is_set():
            raise IngestionCancelled()
        if not request.path.exists() or request.path.stat().st_size == 0:
            raise EmptyDocumentError("document is empty")
        raw_pages = await asyncio.to_thread(_extract_pages, str(request.path), cancel_event)
        if cancel_event and cancel_event.is_set():
            raise IngestionCancelled()

        source_sha = request.resolved_sha256()
        fingerprint = "+".join((*request.parser_chain, self.settings.fingerprint))
        elements: list[CanonicalElement] = []
        total_chars = 0
        nonempty_pages = 0
        for page_number, raw in enumerate(raw_pages, start=1):
            page_text = normalize_text(raw)
            if not page_text:
                continue
            page_chars = sum(1 for ch in page_text if ch.isalnum())
            if page_chars == 0:
                continue
            nonempty_pages += 1
            total_chars += page_chars
            offset = 0
            for block in page_text.split("\n\n"):
                text = normalize_text(block)
                if not text:
                    continue
                start = page_text.find(text, offset)
                if start < 0:
                    start = offset
                end = start + len(text)
                offset = end
                provenance = (
                    Provenance(
                        page_number=page_number,
                        char_start=start,
                        char_end=end,
                    ),
                )
                ordinal = len(elements)
                elements.append(
                    CanonicalElement(
                        element_id=build_element_id(
                            source_sha256=source_sha,
                            parser_fingerprint=fingerprint,
                            element_type=ElementType.PARAGRAPH,
                            text=text,
                            provenance=provenance,
                            ordinal=ordinal,
                        ),
                        element_type=ElementType.PARAGRAPH,
                        text=text,
                        provenance=provenance,
                        metadata={"parser": "fastpdf"},
                    )
                )
        avg_per_page = total_chars / nonempty_pages if nonempty_pages else 0
        if (
            not elements
            or total_chars < self.settings.min_total_chars
            or avg_per_page < self.settings.min_chars_per_page
        ):
            raise FastPdfThinError(
                f"fast PDF text too thin ({total_chars} alnum chars, "
                f"{avg_per_page:.0f}/page, {len(elements)} blocks); "
                "falling through to OCR chain"
            )
        logger.info(
            "FastPdf accepted %s (%d pages, %d chars)",
            request.source_name,
            len(raw_pages),
            total_chars,
        )
        return CanonicalDocument(
            source_name=request.source_name,
            source_sha256=source_sha,
            media_type=request.media_type,
            parser_fingerprint=fingerprint,
            elements=tuple(elements),
            metadata={"parser": "fastpdf"},
        )


class FastPdfFirstParser:
    """Try fast digital-PDF extraction first, then the full OCR chain."""

    def __init__(self, fast: FastPdfParser, fallback: DocumentParser) -> None:
        self.fast = fast
        self.fallback = fallback

    async def parse(
        self,
        request: ParseRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> CanonicalDocument:
        try:
            return await self.fast.parse(request, cancel_event=cancel_event)
        except FastPdfThinError as exc:
            logger.info("FastPdf thin for %s: %s", request.source_name, exc)
            if cancel_event and cancel_event.is_set():
                raise IngestionCancelled() from None
            attempted = f"{self.fast.settings.fingerprint}:result=thin"
            fallback_request = request.with_path(request.path, parser=attempted)
            return await self.fallback.parse(fallback_request, cancel_event=cancel_event)
