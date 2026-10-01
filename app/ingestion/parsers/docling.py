"""Lazy Docling adapter for Office, HTML, CSV, raster, and scanned-PDF OCR."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from contextlib import contextmanager
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
from ..security import SecureTempLease
from .base import ParseRequest

_DOCLING_TYPES: Mapping[str, ElementType] = {
    "title": ElementType.TITLE,
    "section_header": ElementType.HEADING,
    "heading": ElementType.HEADING,
    "paragraph": ElementType.PARAGRAPH,
    "text": ElementType.PARAGRAPH,
    "list_item": ElementType.LIST,
    "table": ElementType.TABLE,
    "formula": ElementType.FORMULA,
    "code": ElementType.CODE,
    "caption": ElementType.CAPTION,
    "picture": ElementType.IMAGE,
    "image": ElementType.IMAGE,
}
_RASTER_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True, slots=True)
class DoclingSettings:
    version: str = "2.112.0"
    max_input_bytes: int = 512 * 1024 * 1024
    tesseract_executable: str = "tesseract"
    # Keep the installed language set deliberately small. Kannada land records
    # and bilingual Arabic purchase orders both occur in the production corpus.
    tesseract_languages: tuple[str, ...] = ("eng", "kan", "ara")
    pdf_ocr_psm: int = 6

    @property
    def fingerprint(self) -> str:
        return f"docling:{self.version}:targeted:local"


class DoclingParser:
    def __init__(
        self,
        settings: DoclingSettings | None = None,
        *,
        temp_root: Path | None = None,
    ) -> None:
        self.settings = settings or DoclingSettings()
        self.temp_root = temp_root or Path(os.environ.get("INGESTION_TEMP_ROOT", "/tmp/vault_ingestion"))
        self._converter: Any = None
        self._converter_lock = threading.Lock()
        self._tesseract_identity: str | None = None

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
        if request.path.stat().st_size > self.settings.max_input_bytes:
            raise ParseError("document exceeds the Docling input limit")
        is_raster = request.media_type.lower().startswith("image/") or (
            request.path.suffix.lower() in _RASTER_SUFFIXES
        )
        uses_tesseract = (
            is_raster
            or request.media_type.lower() == "application/pdf"
            or (request.path.suffix.lower() == ".pdf")
        )
        if uses_tesseract:
            ocr_fingerprint = await asyncio.to_thread(self._ocr_fingerprint)
        else:
            ocr_fingerprint = None
        try:
            payload = await asyncio.to_thread(self._convert, request.path)
        except ModuleNotFoundError as exc:
            missing = str(exc.name or "unknown")[:120]
            raise ParserUnavailableError(f"Docling runtime dependency is unavailable: {missing}") from exc
        if cancel_event and cancel_event.is_set():
            raise IngestionCancelled()

        source_sha = request.resolved_sha256()
        parser_fingerprint = self.settings.fingerprint
        if ocr_fingerprint is not None:
            parser_fingerprint = f"{parser_fingerprint}+{ocr_fingerprint}"
            if not is_raster:
                parser_fingerprint = f"{parser_fingerprint}:psm={self.settings.pdf_ocr_psm}"
        fingerprint = "+".join((*request.parser_chain, parser_fingerprint))
        elements = docling_dict_to_elements(
            payload,
            source_sha,
            fingerprint,
            source_suffix=request.path.suffix.lower(),
        )
        if not elements:
            raise EmptyDocumentError("Docling returned no indexable elements")
        return CanonicalDocument(
            source_name=request.source_name,
            source_sha256=source_sha,
            media_type=request.media_type,
            parser_fingerprint=fingerprint,
            elements=tuple(elements),
            metadata={"parser": "docling"},
        )

    def _convert(self, path: Path) -> Mapping[str, Any]:
        # Heavy import is deliberately inside the adapter.
        from docling.datamodel.base_models import InputFormat  # type: ignore
        from docling.datamodel.pipeline_options import (  # type: ignore
            PdfPipelineOptions,
            TesseractCliOcrOptions,
        )
        from docling.document_converter import (  # type: ignore
            DocumentConverter,
            ImageFormatOption,
            PdfFormatOption,
        )

        with self._converter_lock, _scoped_docling_temp(self.temp_root):
            if self._converter is None:
                image_options = PdfPipelineOptions(
                    do_ocr=True,
                    do_table_structure=False,
                    ocr_options=TesseractCliOcrOptions(
                        lang=list(self.settings.tesseract_languages),
                        tesseract_cmd=self.settings.tesseract_executable,
                    ),
                )
                pdf_ocr_options = PdfPipelineOptions(
                    # OpenDataLoader remains the PDF parser. Docling reaches this
                    # path only after ODL returns a typed empty-document result.
                    # Force OCR for full-page scans and leave TableFormer disabled.
                    do_ocr=True,
                    do_table_structure=False,
                    ocr_options=TesseractCliOcrOptions(
                        lang=list(self.settings.tesseract_languages),
                        force_full_page_ocr=True,
                        psm=self.settings.pdf_ocr_psm,
                        tesseract_cmd=self.settings.tesseract_executable,
                    ),
                )
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.IMAGE: ImageFormatOption(
                            pipeline_options=image_options,
                        ),
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pdf_ocr_options,
                        ),
                    }
                )
            try:
                result = self._converter.convert(str(path), raises_on_error=False)
            except TypeError:
                # Compatibility with Docling releases that do not expose the keyword.
                result = self._converter.convert(str(path))
            status = str(getattr(result, "status", "success")).lower()
            if "failure" in status or status.endswith("failed"):
                errors = getattr(result, "errors", ())
                diagnostic = str(errors).lower()
                if "password" in diagnostic or "encrypted" in diagnostic:
                    raise PasswordRequiredError("document requires a password")
                raise CorruptDocumentError("Docling conversion failed", detail=str(errors)[:2_000])
            document = getattr(result, "document", None)
            if document is None:
                raise ParseError("Docling returned no document")
            payload = document.export_to_dict()
            if not isinstance(payload, Mapping):
                raise ParseError("Docling returned a non-object document")
            return payload

    def _ocr_fingerprint(self) -> str:
        with self._converter_lock:
            if self._tesseract_identity is not None:
                return self._tesseract_identity
            try:
                result = subprocess.run(
                    [self.settings.tesseract_executable, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ParserUnavailableError("Tesseract CLI is unavailable") from exc
            if result.returncode != 0:
                raise ParserUnavailableError("Tesseract CLI version probe failed")
            version_line = (result.stdout or result.stderr).splitlines()[0].strip()
            normalized = re.sub(r"[^A-Za-z0-9._+-]+", "_", version_line).strip("_")[:120]
            if not normalized:
                raise ParserUnavailableError("Tesseract CLI returned no version")
            languages = ",".join(self.settings.tesseract_languages)
            self._tesseract_identity = f"ocr:tesseract-cli:{normalized}:lang={languages}"
            return self._tesseract_identity


@contextmanager
def _scoped_docling_temp(root: Path):
    """Contain third-party temporary files in a lease that is always removed."""

    with SecureTempLease(root, prefix="docling-") as lease:
        if lease.path is None:  # pragma: no cover - guarded by SecureTempLease
            raise RuntimeError("Docling temporary lease was not created")
        previous = tempfile.tempdir
        tempfile.tempdir = os.fspath(lease.path)
        try:
            yield
        finally:
            tempfile.tempdir = previous


def _docling_bbox(value: Any) -> BoundingBox | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return BoundingBox(
            left=float(value.get("l", value.get("left"))),
            top=float(value.get("t", value.get("top"))),
            right=float(value.get("r", value.get("right"))),
            bottom=float(value.get("b", value.get("bottom"))),
            coordinate_system=CoordinateSystem.DOCLING,
        )
    except (TypeError, ValueError):
        return None


def _as_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _table_text(node: Mapping[str, Any]) -> str:
    direct = node.get("text")
    if isinstance(direct, str) and direct.strip():
        return normalize_text(direct)
    data = node.get("data")
    if not isinstance(data, Mapping):
        return ""
    cells = data.get("table_cells", data.get("cells", []))
    if not isinstance(cells, list):
        return ""
    grid: dict[int, dict[int, str]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        try:
            row = int(cell.get("start_row_offset_idx", cell.get("row", 0)))
            column = int(cell.get("start_col_offset_idx", cell.get("column", 0)))
        except (TypeError, ValueError):
            continue
        text = normalize_text(str(cell.get("text", ""))).replace("\n", " ")
        grid.setdefault(row, {})[column] = text
    rows: list[str] = []
    for row_index in sorted(grid):
        row_values = grid[row_index]
        rows.append(" | ".join(row_values.get(index, "") for index in range(max(row_values) + 1)))
    return normalize_text("\n".join(rows))


def _node_text(node: Mapping[str, Any], label: str) -> str:
    if label == "table":
        return _table_text(node)
    for key in ("text", "orig", "description", "caption"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    return ""


def _all_nodes(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    ordered: list[Mapping[str, Any]] = []
    by_ref: dict[str, Mapping[str, Any]] = {}
    for key in ("texts", "tables", "pictures", "key_value_items", "form_items", "groups"):
        values = payload.get(key, [])
        if not isinstance(values, list):
            continue
        for node in values:
            if not isinstance(node, Mapping):
                continue
            ordered.append(node)
            reference = node.get("self_ref")
            if isinstance(reference, str):
                by_ref[reference] = node
    return ordered, by_ref


def _ordered_nodes(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    fallback, by_ref = _all_nodes(payload)
    body = payload.get("body")
    if not isinstance(body, Mapping) or not isinstance(body.get("children"), list):
        return fallback
    result: list[Mapping[str, Any]] = []
    visited: set[str] = set()

    def visit_ref(item: Any) -> None:
        if not isinstance(item, Mapping):
            return
        reference = item.get("$ref", item.get("ref"))
        node = by_ref.get(reference) if isinstance(reference, str) else item
        if not isinstance(node, Mapping):
            return
        node_ref = str(node.get("self_ref", reference or id(node)))
        if node_ref in visited:
            return
        visited.add(node_ref)
        children = node.get("children")
        label = str(node.get("label", "")).lower()
        if label not in {"group", "section"}:
            result.append(node)
        if isinstance(children, list):
            for child in children:
                visit_ref(child)

    for child in body["children"]:
        visit_ref(child)
    for node in fallback:
        reference = str(node.get("self_ref", id(node)))
        if reference not in visited and str(node.get("label", "")).lower() not in {"group", "section"}:
            result.append(node)
    return result


def _provenance(
    node: Mapping[str, Any],
    *,
    section_path: tuple[str, ...],
    source_suffix: str,
) -> tuple[Provenance, ...]:
    raw = node.get("prov", [])
    entries = raw if isinstance(raw, list) else []
    result: list[Provenance] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        page = _as_positive_int(entry.get("page_no", entry.get("page_number")))
        charspan = entry.get("charspan")
        char_start = char_end = None
        if isinstance(charspan, (list, tuple)) and len(charspan) == 2:
            try:
                char_start, char_end = int(charspan[0]), int(charspan[1])
            except (TypeError, ValueError):
                pass
        result.append(
            Provenance(
                page_number=None if source_suffix == ".pptx" else page,
                slide_number=page if source_suffix == ".pptx" else None,
                sheet_name=str(entry.get("sheet_name")) if entry.get("sheet_name") else None,
                cell_range=str(entry.get("cell_range")) if entry.get("cell_range") else None,
                char_start=char_start,
                char_end=char_end,
                section_path=section_path,
                block_id=str(node.get("self_ref")) if node.get("self_ref") else None,
                bbox=_docling_bbox(entry.get("bbox")),
            )
        )
    if not result:
        result.append(
            Provenance(
                section_path=section_path,
                block_id=str(node.get("self_ref")) if node.get("self_ref") else None,
            )
        )
    return tuple(result)


def docling_dict_to_elements(
    payload: Mapping[str, Any],
    source_sha256: str,
    parser_fingerprint: str,
    *,
    source_suffix: str,
) -> list[CanonicalElement]:
    elements: list[CanonicalElement] = []
    headings: list[str] = []
    for node in _ordered_nodes(payload):
        label = str(node.get("label", "text")).lower()
        element_type = _DOCLING_TYPES.get(label, ElementType.OTHER)
        text = _node_text(node, label)
        if not text:
            continue
        heading_level = _as_positive_int(node.get("level"))
        if element_type in {ElementType.TITLE, ElementType.HEADING}:
            level = heading_level or 1
            headings = headings[: max(0, level - 1)]
            headings.append(text)
        provenance = _provenance(
            node,
            section_path=tuple(headings),
            source_suffix=source_suffix,
        )
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
                metadata={"docling_label": label},
            )
        )
    return elements
