"""Typed failures shared by ingestion workers and parser adapters."""

from __future__ import annotations

import re
from collections.abc import Iterable

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-_]|\[[0-?]*[ -/]*[@-~])")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|token|secret|api[ _-]?key|authorization)\b"
    r"(?:\s*[:=]\s*|\s+)(?:bearer\s+)?[^\s,;]+"
)


def bounded_safe_error_detail(
    value: object,
    *,
    replacements: Iterable[str] = (),
    limit: int = 2_000,
) -> str:
    """Return a log-safe, single-line diagnostic with secrets and known paths removed."""

    if limit < 1:
        raise ValueError("error detail limit must be positive")
    text = _ANSI_ESCAPE_RE.sub("", str(value or ""))
    for replacement in sorted({item for item in replacements if item}, key=len, reverse=True):
        text = text.replace(replacement, "<redacted-path>")
    text = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = "".join(character if character.isprintable() else " " for character in text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = "…" if limit == 1 else "…" + text[-(limit - 1) :]
    return text


class IngestionError(Exception):
    """Base failure with a stable machine-readable code."""

    code = "ingestion_error"
    retryable = False

    def __init__(self, message: str = "", *, detail: str = "") -> None:
        super().__init__(message or self.code)
        self.detail = detail


class RetryableIngestionError(IngestionError):
    code = "retryable_ingestion_error"
    retryable = True


class ParseError(IngestionError):
    code = "parse_failed"


class ParserUnavailableError(RetryableIngestionError):
    code = "parser_unavailable"


class UnsupportedFormatError(IngestionError):
    code = "unsupported_format"


class PasswordRequiredError(IngestionError):
    code = "password_required"


class CorruptDocumentError(IngestionError):
    code = "corrupt_document"


class EmptyDocumentError(IngestionError):
    code = "empty_document"


class IngestionCancelled(IngestionError):
    code = "cancelled"


class ProcessExecutionError(RetryableIngestionError):
    code = "process_failed"

    def __init__(
        self,
        message: str,
        result: object | None = None,
        *,
        detail: str = "",
    ) -> None:
        super().__init__(message, detail=detail)
        self.result = result


class ProcessTimeoutError(RetryableIngestionError):
    code = "process_timeout"


class ProcessOutputLimitError(IngestionError):
    code = "process_output_limit"


class SourceUnavailableError(RetryableIngestionError):
    code = "source_unavailable"


class SourceIntegrityError(IngestionError):
    code = "source_integrity_failed"


class EmbeddingError(RetryableIngestionError):
    code = "embedding_failed"


class EmbeddingDimensionError(IngestionError):
    code = "embedding_dimension_mismatch"


class PublicationError(RetryableIngestionError):
    code = "publication_failed"
