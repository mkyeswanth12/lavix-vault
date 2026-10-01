"""Deterministic plain-text parsing with line provenance."""

from __future__ import annotations

import asyncio

from ..errors import EmptyDocumentError, IngestionCancelled, ParseError
from ..models import (
    CanonicalDocument,
    CanonicalElement,
    ElementType,
    Provenance,
    build_element_id,
    normalize_text,
)
from .base import ParseRequest


class DeterministicTextParser:
    fingerprint = "text:v1:utf-bom-utf8-cp1252"

    def __init__(self, *, max_bytes: int = 64 * 1024 * 1024) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes

    async def parse(
        self,
        request: ParseRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> CanonicalDocument:
        if cancel_event and cancel_event.is_set():
            raise IngestionCancelled()
        try:
            size = request.path.stat().st_size
        except OSError as exc:
            raise ParseError("text input is unavailable") from exc
        if size == 0:
            raise EmptyDocumentError("text file is empty")
        if size > self.max_bytes:
            raise ParseError(f"text file exceeds {self.max_bytes} bytes")
        data = await asyncio.to_thread(request.path.read_bytes)
        if cancel_event and cancel_event.is_set():
            raise IngestionCancelled()
        if len(data) > self.max_bytes:
            raise ParseError(f"text file exceeds {self.max_bytes} bytes")
        text, encoding = self._decode(data)
        source_sha = request.resolved_sha256()
        fingerprint = "+".join((*request.parser_chain, self.fingerprint))
        elements = self._elements(text, source_sha, fingerprint)
        if not elements:
            raise EmptyDocumentError("text file contains no indexable content")
        return CanonicalDocument(
            source_name=request.source_name,
            source_sha256=source_sha,
            media_type=request.media_type,
            parser_fingerprint=fingerprint,
            elements=tuple(elements),
            metadata={"encoding": encoding},
        )

    @staticmethod
    def _decode(data: bytes) -> tuple[str, str]:
        candidates: list[tuple[str, str]] = []
        if data.startswith(b"\xef\xbb\xbf"):
            candidates.append(("utf-8-sig", "utf-8-sig"))
        elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
            candidates.append(("utf-16", "utf-16"))
        candidates.extend((("utf-8", "utf-8"), ("cp1252", "cp1252")))
        for codec, label in candidates:
            try:
                return data.decode(codec), label
            except UnicodeDecodeError:
                continue
        raise ParseError("unable to decode text file")

    @staticmethod
    def _elements(text: str, source_sha: str, fingerprint: str) -> list[CanonicalElement]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        elements: list[CanonicalElement] = []
        paragraph: list[str] = []
        start_line = 1

        def flush(end_line: int) -> None:
            nonlocal paragraph, start_line
            content = normalize_text("\n".join(paragraph))
            paragraph = []
            if not content:
                return
            provenance = (Provenance(line_start=start_line, line_end=end_line),)
            ordinal = len(elements)
            elements.append(
                CanonicalElement(
                    element_id=build_element_id(
                        source_sha256=source_sha,
                        parser_fingerprint=fingerprint,
                        element_type=ElementType.PARAGRAPH,
                        text=content,
                        provenance=provenance,
                        ordinal=ordinal,
                    ),
                    element_type=ElementType.PARAGRAPH,
                    text=content,
                    provenance=provenance,
                )
            )

        for line_number, line in enumerate(lines, start=1):
            if line.strip():
                if not paragraph:
                    start_line = line_number
                paragraph.append(line)
            else:
                flush(line_number - 1)
        flush(len(lines))
        return elements
