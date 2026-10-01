"""Deterministic, side-effect-free parser routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ParserKind(StrEnum):
    OPEN_DATALOADER_PDF = "opendataloader_pdf"
    DOCLING = "docling"
    LIBREOFFICE_DOCLING = "libreoffice_docling"
    TEXT = "text"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ParserRoute:
    parser: ParserKind
    normalized_extension: str | None = None
    reason: str = ""

    @property
    def supported(self) -> bool:
        return self.parser is not ParserKind.UNSUPPORTED


_MODERN_OFFICE = {".docx", ".pptx", ".xlsx"}
_WEB_DATA = {".html", ".htm", ".csv"}
_DOCLING_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_IMAGE_MIMES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}
_LEGACY_TARGETS = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}
_TEXT = {
    ".bash",
    ".c",
    ".cfg",
    ".conf",
    ".cpp",
    ".css",
    ".fish",
    ".go",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".log",
    ".md",
    ".markdown",
    ".py",
    ".rs",
    ".rst",
    ".sh",
    ".sql",
    ".text",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}

_TEXT_MIMES = {
    "application/json",
    "application/ld+json",
    "application/toml",
    "application/xml",
    "application/x-ndjson",
    "application/x-yaml",
    "text/markdown",
    "text/x-python",
    "text/xml",
    "text/yaml",
}

_MODERN_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/csv": ".csv",
}
_LEGACY_MIMES = {
    "application/msword": ".docx",
    "application/vnd.ms-excel": ".xlsx",
    "application/vnd.ms-powerpoint": ".pptx",
}


def route_file(source_name: str, media_type: str | None = None) -> ParserRoute:
    """Choose one parser from a trusted/sniffed media type and sanitized name.

    A definitive audio/video media type always wins over a misleading extension.
    For generic ZIP/octet-stream uploads, an Office extension is allowed to select
    Docling so corrupt OOXML reaches a typed parser failure instead of being hidden
    as merely unsupported.
    """

    suffix = Path(source_name).suffix.lower()
    mime = (media_type or "").split(";", 1)[0].strip().lower()

    if mime.startswith(("audio/", "video/")):
        return ParserRoute(ParserKind.UNSUPPORTED, reason="audio_video_disabled")
    if mime == "application/pdf":
        return ParserRoute(ParserKind.OPEN_DATALOADER_PDF, ".pdf")
    if mime in _LEGACY_MIMES:
        return ParserRoute(ParserKind.LIBREOFFICE_DOCLING, _LEGACY_MIMES[mime])
    if mime in _MODERN_MIMES:
        return ParserRoute(ParserKind.DOCLING, _MODERN_MIMES[mime])
    if mime.startswith("image/"):
        if mime in _IMAGE_MIMES:
            return ParserRoute(ParserKind.DOCLING, _IMAGE_MIMES[mime])
        return ParserRoute(ParserKind.UNSUPPORTED, reason="unsupported_image_encoding")
    if mime == "text/plain":
        # libmagic commonly classifies ordinary CSV as text/plain. The extension
        # remains necessary here so arbitrary text cannot impersonate another
        # binary format, while real CSV still reaches the supported Docling path.
        if suffix == ".csv":
            return ParserRoute(ParserKind.DOCLING, suffix)
        if suffix in _TEXT or not suffix:
            return ParserRoute(ParserKind.TEXT, suffix or ".txt")
    if (mime in _TEXT_MIMES or mime.startswith("text/")) and suffix in _TEXT:
        return ParserRoute(ParserKind.TEXT, suffix)

    # Extension fallback is intentionally limited to absent/generic MIME values.
    # CFB container MIMEs (password-protected OOXML, some legacy Office saves)
    # are included: sniffing cannot see inside them, so a matching extension is
    # the only routing signal — and LibreOffice's typed PasswordRequiredError
    # beats a misleading terminal "unsupported".
    generic = not mime or mime in {
        "application/octet-stream",
        "application/zip",
        "application/x-ole-storage",
        "application/x-cfb",
        "application/cdfv2",
        "application/cdfv2-corrupt",
        "application/vnd.ms-office",
    }
    if generic:
        if suffix == ".pdf":
            return ParserRoute(ParserKind.OPEN_DATALOADER_PDF, ".pdf")
        if suffix in _MODERN_OFFICE | _WEB_DATA | _DOCLING_IMAGES:
            return ParserRoute(ParserKind.DOCLING, suffix)
        if suffix in _LEGACY_TARGETS:
            return ParserRoute(ParserKind.LIBREOFFICE_DOCLING, _LEGACY_TARGETS[suffix])
        if suffix in _TEXT:
            return ParserRoute(ParserKind.TEXT, suffix)

    return ParserRoute(ParserKind.UNSUPPORTED, reason="no_parser_for_media_type")
