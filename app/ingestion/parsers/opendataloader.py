"""OpenDataLoader PDF adapter producing canonical elements and PDF provenance."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import (
    CorruptDocumentError,
    EmptyDocumentError,
    IngestionCancelled,
    ParseError,
    ParserUnavailableError,
    PasswordRequiredError,
    ProcessExecutionError,
    bounded_safe_error_detail,
)
from ..models import (
    BoundingBox,
    CanonicalDocument,
    CanonicalElement,
    CoordinateSystem,
    ElementType,
    Provenance,
    build_element_id,
    normalize_text,
)
from ..security import AsyncProcessRunner, SecureTempLease
from .base import DocumentParser, ParseRequest

logger = logging.getLogger(__name__)

_ODL_TYPES: Mapping[str, ElementType] = {
    "title": ElementType.TITLE,
    "heading": ElementType.HEADING,
    "paragraph": ElementType.PARAGRAPH,
    "text": ElementType.PARAGRAPH,
    "list": ElementType.LIST,
    "list-item": ElementType.LIST,
    "table": ElementType.TABLE,
    "formula": ElementType.FORMULA,
    "code": ElementType.CODE,
    "caption": ElementType.CAPTION,
    "image": ElementType.IMAGE,
    "picture": ElementType.IMAGE,
}

_SCANNER_WATERMARKS = (
    "scanned with camscanner",
    "scanned by tap scanner",
    "microsoft lens",
    "adobe scan",
    "camscanner",
)


class OpenDataLoaderNoElementsError(EmptyDocumentError):
    """ODL converted successfully, but its JSON contained no useful elements."""


@dataclass(frozen=True, slots=True)
class OpenDataLoaderSettings:
    executable: str = "opendataloader-pdf"
    hybrid: str = "docling-fast"
    hybrid_url: str = "http://odl-hybrid:5002"
    hybrid_timeout_ms: int = 3_600_000
    process_timeout_seconds: float = 7_200.0
    max_json_bytes: int = 128 * 1024 * 1024
    version: str = "2.4.7"

    @property
    def fingerprint(self) -> str:
        return (
            f"opendataloader:{self.version}:json:xycut:safety-on:sanitize-off:"
            f"images-off:struct-off:threads=1:hybrid={self.hybrid}:fallback=off"
        )


class OpenDataLoaderPdfParser:
    def __init__(
        self,
        *,
        temp_root: Path,
        runner: AsyncProcessRunner | None = None,
        settings: OpenDataLoaderSettings | None = None,
    ) -> None:
        self.temp_root = temp_root
        self.runner = runner or AsyncProcessRunner()
        self.settings = settings or OpenDataLoaderSettings()

    async def parse(
        self,
        request: ParseRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> CanonicalDocument:
        if not request.path.exists():
            raise ParseError("PDF input does not exist")
        if request.path.stat().st_size == 0:
            raise EmptyDocumentError("PDF is empty")
        with SecureTempLease(self.temp_root, prefix="odl-") as lease:
            output_dir = lease.mkdir("output")
            args = [
                self.settings.executable,
                "--output-dir",
                str(output_dir),
                "--format",
                "json",
                "--image-output",
                "off",
                "--reading-order",
                "xycut",
                "--threads",
                "1",
            ]
            if self.settings.hybrid and self.settings.hybrid != "off":
                args.extend(
                    [
                        "--hybrid",
                        self.settings.hybrid,
                        "--hybrid-url",
                        self.settings.hybrid_url,
                        "--hybrid-timeout",
                        str(self.settings.hybrid_timeout_ms),
                    ]
                )
            args.append(str(request.path))
            try:
                result = await self.runner.run(
                    args,
                    cwd=lease.path,
                    timeout_seconds=self.settings.process_timeout_seconds,
                    cancel_event=cancel_event,
                )
            except FileNotFoundError as exc:
                raise ParserUnavailableError("OpenDataLoader executable is unavailable") from exc

            if result.returncode != 0:
                diagnostic = f"{result.stdout_text}\n{result.stderr_text}".lower()
                if "password" in diagnostic or "encrypted" in diagnostic:
                    raise PasswordRequiredError("PDF requires a password")
                if "corrupt" in diagnostic or "malformed" in diagnostic:
                    raise CorruptDocumentError("PDF is corrupt")
                stderr_detail = bounded_safe_error_detail(
                    result.stderr_text,
                    replacements=(str(request.path), str(lease.path), str(output_dir)),
                    limit=1_600,
                )
                raise ProcessExecutionError(
                    f"OpenDataLoader exited with status {result.returncode}",
                    result=result,
                    detail=(
                        f"OpenDataLoader exited with status {result.returncode}: "
                        f"{stderr_detail or 'no stderr output'}"
                    ),
                )

            candidates = sorted(
                path for path in output_dir.rglob("*.json") if path.is_file() and not path.is_symlink()
            )
            if len(candidates) != 1:
                raise ParseError(f"OpenDataLoader produced {len(candidates)} JSON documents; expected one")
            output = candidates[0]
            if output.stat().st_size > self.settings.max_json_bytes:
                raise ParseError("OpenDataLoader JSON exceeds the configured limit")
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ParseError("OpenDataLoader returned invalid JSON") from exc

        source_sha = request.resolved_sha256()
        fingerprint = "+".join((*request.parser_chain, self.settings.fingerprint))
        elements = odl_json_to_elements(payload, source_sha, fingerprint)
        if not elements or _only_scanner_watermark_text(elements):
            raise OpenDataLoaderNoElementsError("PDF contains no useful indexable elements")
        return CanonicalDocument(
            source_name=request.source_name,
            source_sha256=source_sha,
            media_type=request.media_type,
            parser_fingerprint=fingerprint,
            elements=tuple(elements),
            metadata={"parser": "opendataloader", "hybrid": self.settings.hybrid},
        )


class OpenDataLoaderEmptyFallbackParser:
    """Use local PDF OCR only after OpenDataLoader returns no indexable elements."""

    def __init__(
        self,
        primary: OpenDataLoaderPdfParser,
        fallback: DocumentParser,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    async def parse(
        self,
        request: ParseRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> CanonicalDocument:
        try:
            return await self.primary.parse(request, cancel_event=cancel_event)
        except OpenDataLoaderNoElementsError:
            if cancel_event and cancel_event.is_set():
                raise IngestionCancelled() from None
            attempted = f"{self.primary.settings.fingerprint}:result=empty"
            fallback_request = request.with_path(request.path, parser=attempted)
            return await self.fallback.parse(fallback_request, cancel_event=cancel_event)
        except ProcessExecutionError as exc:
            # ODL CLI crashes (malformed PDFs the hybrid service chokes on)
            # must not be terminal: Docling handles these directly.
            logger.warning(
                "OpenDataLoader crashed for %s (%s); falling through to Docling",
                request.source_name,
                str(exc.detail or exc)[:160],
            )
            if cancel_event and cancel_event.is_set():
                raise IngestionCancelled() from None
            attempted = f"{self.primary.settings.fingerprint}:result=crashed"
            fallback_request = request.with_path(request.path, parser=attempted)
            return await self.fallback.parse(fallback_request, cancel_event=cancel_event)


def _only_scanner_watermark_text(elements: Iterable[CanonicalElement]) -> bool:
    """Identify a successful parse whose only payload is a scan-app watermark."""

    combined = " ".join(element.text for element in elements).casefold()
    if not any(marker in combined for marker in _SCANNER_WATERMARKS):
        return False
    remainder = combined
    for marker in _SCANNER_WATERMARKS:
        remainder = remainder.replace(marker, " ")
    return len("".join(character for character in remainder if character.isalnum())) < 12


def _children(node: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    value = node.get("kids", node.get("children", node.get("elements", [])))
    if isinstance(value, list):
        for child in value:
            if isinstance(child, Mapping):
                yield child


def _node_text(node: Mapping[str, Any]) -> str:
    for key in ("content", "text", "description"):
        value = node.get(key)
        if isinstance(value, str):
            return normalize_text(value)
        if isinstance(value, (dict, list)) and value:
            return normalize_text(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
    return ""


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _raw_type(node: Mapping[str, Any]) -> str:
    return (
        str(node.get("type", node.get("level", "text"))).lower().replace("_", "-").replace(" ", "-")
    )


def _collect_text(node: Mapping[str, Any]) -> str:
    """Recursive text for cells: join real text runs, never stringify structure."""
    kids = [
        kid
        for kid in _children(node)
        if _raw_type(kid) not in {"table", "table-row", "table-cell"}
    ]
    if kids:
        return normalize_text(" ".join(part for part in (_collect_text(kid) for kid in kids) if part))
    return _node_text(node)


def _expand_cells(cells: list[Mapping[str, Any]]) -> tuple[list[str], list[int | None]]:
    """Expand a row's cells across columns honoring column number/span.

    Returns (texts, pages) indexed by 0-based column. Missing columns are "".
    """
    texts: list[str] = []
    pages: list[int | None] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        start = _positive_int(cell.get("column number")) or (len(texts) + 1)
        span = _positive_int(cell.get("column span")) or 1
        while len(texts) < start - 1:
            texts.append("")
            pages.append(None)
        text = _collect_text(cell)
        page = _positive_int(cell.get("page number", cell.get("page_number")))
        for _ in range(span):
            texts.append(text)
            pages.append(page)
    return texts, pages


def _table_row_elements(
    node: Mapping[str, Any],
    raw_type: str,
    heading_stack: list[str],
    source_sha256: str,
    parser_fingerprint: str,
    elements: list[CanonicalElement],
    stats: dict[str, Any],
) -> int:
    """Emit one TABLE element per data row as 'Header: value | ...'.

    Returns the number of rows emitted. The first row is the header; every
    data row carries its own page number so multi-page tables stay attributed.
    """
    rows = node.get("rows")
    if not isinstance(rows, list):
        return 0
    parsed: list[tuple[list[str], list[int | None]]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cells = row.get("cells")
        if not isinstance(cells, list):
            continue
        parsed.append(_expand_cells([c for c in cells if isinstance(c, Mapping)]))
    if not parsed:
        return 0
    headers = parsed[0][0]
    if not any(headers):
        headers = [f"Column {i + 1}" for i in range(len(headers))]
    table_page = _positive_int(node.get("page number", node.get("page_number")))
    table_bbox = _odl_bbox(node.get("bounding box", node.get("bounding_box")))
    table_id = str(node["id"]) if node.get("id") is not None else None
    emitted = 0
    for index, (values, pages) in enumerate(parsed[1:], start=2):
        pairs = []
        for col, value in enumerate(values):
            if not value:
                continue
            header = headers[col] if col < len(headers) else f"Column {col + 1}"
            if not header:
                header = f"Column {col + 1}"
            pairs.append(f"{header}: {value}")
        if not pairs:
            stats["empty_rows"] += 1
            continue
        page = next((p for p in pages if p is not None), table_page)
        provenance = (
            Provenance(
                page_number=page,
                section_path=tuple(heading_stack),
                block_id=f"{table_id}#row={index}" if table_id else None,
                bbox=table_bbox,
            ),
        )
        ordinal = len(elements)
        elements.append(
            CanonicalElement(
                element_id=build_element_id(
                    source_sha256=source_sha256,
                    parser_fingerprint=parser_fingerprint,
                    element_type=ElementType.TABLE,
                    text=" | ".join(pairs),
                    provenance=provenance,
                    ordinal=ordinal,
                ),
                element_type=ElementType.TABLE,
                text=" | ".join(pairs),
                provenance=provenance,
                heading_level=None,
                metadata={"odl_type": raw_type, "table_row": index},
            )
        )
        emitted += 1
    return emitted


def _odl_bbox(value: Any) -> BoundingBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, bottom, right, top = (float(item) for item in value)
        return BoundingBox(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            coordinate_system=CoordinateSystem.PDF_POINTS_BOTTOM_LEFT,
        )
    except (TypeError, ValueError):
        return None


def odl_json_to_elements(
    payload: Any,
    source_sha256: str,
    parser_fingerprint: str,
) -> list[CanonicalElement]:
    """Convert ODL's recursive JSON format without importing its Python SDK."""

    roots: list[Mapping[str, Any]]
    if isinstance(payload, list):
        roots = [item for item in payload if isinstance(item, Mapping)]
    elif isinstance(payload, Mapping):
        roots = list(_children(payload)) or [payload]
    else:
        raise ParseError("OpenDataLoader JSON root is not an object or list")

    elements: list[CanonicalElement] = []
    heading_stack: list[str] = []
    stats: dict[str, Any] = {
        "nodes": 0,
        "skipped_textless": 0,
        "table_nodes": 0,
        "table_rows": 0,
        "empty_rows": 0,
        "empty_tables": 0,
    }

    def visit(node: Mapping[str, Any]) -> None:
        nonlocal heading_stack
        stats["nodes"] += 1
        raw_type = _raw_type(node)
        element_type = _ODL_TYPES.get(raw_type, ElementType.OTHER)
        if element_type is ElementType.TABLE and isinstance(node.get("rows"), list):
            stats["table_nodes"] += 1
            emitted = _table_row_elements(
                node,
                raw_type,
                heading_stack,
                source_sha256,
                parser_fingerprint,
                elements,
                stats,
            )
            stats["table_rows"] += emitted
            if emitted == 0:
                stats["empty_tables"] += 1
                logger.warning(
                    "OpenDataLoader table on page %s yielded no text rows; dropping table node",
                    node.get("page number", node.get("page_number")),
                )
            return
        text = _node_text(node)
        heading_level = _positive_int(node.get("heading level", node.get("heading_level")))
        if element_type in {ElementType.TITLE, ElementType.HEADING} and text:
            level = heading_level or 1
            heading_stack = heading_stack[: max(0, level - 1)]
            heading_stack.append(text)
        page_number = _positive_int(node.get("page number", node.get("page_number")))
        provenance = (
            Provenance(
                page_number=page_number,
                section_path=tuple(heading_stack),
                block_id=str(node["id"]) if node.get("id") is not None else None,
                bbox=_odl_bbox(node.get("bounding box", node.get("bounding_box"))),
            ),
        )
        if text:
            ordinal = len(elements)
            elements.append(
                CanonicalElement(
                    element_id=build_element_id(
                        source_sha256=source_sha256,
                        parser_fingerprint=parser_fingerprint,
                        element_type=element_type,
                        text=text,
                        provenance=provenance,
                        ordinal=ordinal,
                    ),
                    element_type=element_type,
                    text=text,
                    provenance=provenance,
                    heading_level=heading_level,
                    metadata={"odl_type": raw_type},
                )
            )
        else:
            stats["skipped_textless"] += 1
        for child in _children(node):
            visit(child)

    for root in roots:
        visit(root)
    if isinstance(payload, Mapping):
        total_pages = _positive_int(payload.get("number of pages"))
        if total_pages is not None:
            covered = {
                prov.page_number
                for item in elements
                for prov in item.provenance
                if prov.page_number is not None
            }
            missing = [page for page in range(1, total_pages + 1) if page not in covered]
            if missing:
                logger.warning(
                    "OpenDataLoader parse yielded zero text units for pages %s; "
                    "these pages contributed no indexable content",
                    missing,
                )
    logger.debug(
        "odl elements=%d tables=%d rows=%d skipped_textless=%d empty_tables=%d",
        len(elements),
        stats["table_nodes"],
        stats["table_rows"],
        stats["skipped_textless"],
        stats["empty_tables"],
    )
    return elements
