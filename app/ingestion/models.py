"""Canonical ingestion models and durable job-state semantics."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class JobState(StrEnum):
    QUEUED = "queued"
    DECRYPTING = "decrypting"
    CONVERTING = "converting"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    PUBLISHING = "publishing"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNSUPPORTED = "unsupported"
    PASSWORD_REQUIRED = "password_required"

    @property
    def terminal(self) -> bool:
        return self in {
            JobState.READY,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.UNSUPPORTED,
            JobState.PASSWORD_REQUIRED,
        }


_FAILURE_TARGETS = {
    JobState.FAILED,
    JobState.CANCELLED,
    JobState.UNSUPPORTED,
    JobState.PASSWORD_REQUIRED,
}

ALLOWED_JOB_TRANSITIONS: Mapping[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.DECRYPTING, *_FAILURE_TARGETS}),
    JobState.DECRYPTING: frozenset(
        {JobState.CONVERTING, JobState.PARSING, JobState.QUEUED, *_FAILURE_TARGETS}
    ),
    JobState.CONVERTING: frozenset({JobState.PARSING, JobState.QUEUED, *_FAILURE_TARGETS}),
    JobState.PARSING: frozenset({JobState.CHUNKING, JobState.QUEUED, *_FAILURE_TARGETS}),
    JobState.CHUNKING: frozenset({JobState.EMBEDDING, JobState.QUEUED, *_FAILURE_TARGETS}),
    JobState.EMBEDDING: frozenset({JobState.PUBLISHING, JobState.QUEUED, *_FAILURE_TARGETS}),
    JobState.PUBLISHING: frozenset({JobState.READY, JobState.QUEUED, JobState.FAILED, JobState.CANCELLED}),
    JobState.READY: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
    JobState.UNSUPPORTED: frozenset(),
    JobState.PASSWORD_REQUIRED: frozenset(),
}


def validate_job_transition(current: JobState, target: JobState) -> None:
    """Raise when a state change would violate the durable state machine."""

    if target not in ALLOWED_JOB_TRANSITIONS[current]:
        raise ValueError(f"invalid ingestion transition: {current.value} -> {target.value}")


class CoordinateSystem(StrEnum):
    PDF_POINTS_BOTTOM_LEFT = "pdf_points_bottom_left"
    PIXELS_TOP_LEFT = "pixels_top_left"
    DOCLING = "docling"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float
    coordinate_system: CoordinateSystem = CoordinateSystem.UNKNOWN

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bounding-box coordinates must be finite")
        if self.right < self.left:
            raise ValueError("bounding-box right must be >= left")
        # PDF coordinates use a bottom-left origin and commonly report top > bottom;
        # top-left image coordinates report bottom > top.  Preserve either ordering.

    def stable_dict(self) -> dict[str, Any]:
        return {
            "left": round(self.left, 4),
            "top": round(self.top, 4),
            "right": round(self.right, 4),
            "bottom": round(self.bottom, 4),
            "coordinate_system": self.coordinate_system.value,
        }


@dataclass(frozen=True, slots=True)
class Provenance:
    """A source locator. Fields not meaningful for a format remain ``None``."""

    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    section_path: tuple[str, ...] = ()
    block_id: str | None = None
    bbox: BoundingBox | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("page_number", self.page_number),
            ("slide_number", self.slide_number),
            ("line_start", self.line_start),
            ("line_end", self.line_end),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{label} must be one-indexed")
        if self.line_start is not None and self.line_end is not None and self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        if self.char_start is not None and self.char_end is not None and self.char_end < self.char_start:
            raise ValueError("char_end must be >= char_start")
        for label, value in (("char_start", self.char_start), ("char_end", self.char_end)):
            if value is not None and value < 0:
                raise ValueError(f"{label} must be non-negative")

    def stable_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "slide_number": self.slide_number,
            "sheet_name": self.sheet_name,
            "cell_range": self.cell_range,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "section_path": list(self.section_path),
            "block_id": self.block_id,
            "bbox": self.bbox.stable_dict() if self.bbox else None,
        }

    def boundary_key(self) -> tuple[Any, ...]:
        """Boundary used to keep chunks within a citeable source unit."""

        if self.slide_number is not None:
            return ("slide", self.slide_number)
        if self.sheet_name is not None:
            return ("sheet", self.sheet_name)
        if self.page_number is not None:
            return ("page", self.page_number)
        return ("section", *self.section_path)


class ElementType(StrEnum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FORMULA = "formula"
    CODE = "code"
    CAPTION = "caption"
    IMAGE = "image"
    TEXT = "text"
    OTHER = "other"


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without changing meaningful newlines."""

    text = unicodedata.normalize("NFC", text or "").replace("\x00", "")
    lines = [re.sub(r"[\t \r\f\v]+", " ", line).strip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(namespace: str, value: Any) -> str:
    material = f"{namespace}\x00{_stable_json(value)}".encode()
    return hashlib.sha256(material).hexdigest()


def build_element_id(
    *,
    source_sha256: str,
    parser_fingerprint: str,
    element_type: ElementType,
    text: str,
    provenance: Sequence[Provenance],
    ordinal: int,
) -> str:
    payload = {
        "schema": 1,
        "source_sha256": source_sha256.lower(),
        "parser_fingerprint": parser_fingerprint,
        "element_type": element_type.value,
        "text": normalize_text(text),
        "provenance": [item.stable_dict() for item in provenance],
        "ordinal": ordinal,
    }
    return f"el_{_digest('lavix-element-v1', payload)}"


@dataclass(frozen=True, slots=True)
class CanonicalElement:
    element_id: str
    element_type: ElementType
    text: str
    provenance: tuple[Provenance, ...]
    heading_level: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        normalized = normalize_text(self.text)
        object.__setattr__(self, "text", normalized)
        if not normalized and self.element_type is not ElementType.IMAGE:
            raise ValueError("canonical text elements cannot be empty")
        if self.heading_level is not None and self.heading_level < 1:
            raise ValueError("heading_level must be >= 1")


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    source_name: str
    source_sha256: str
    media_type: str
    parser_fingerprint: str
    elements: tuple[CanonicalElement, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)


def build_chunk_id(
    *,
    source_sha256: str,
    parser_fingerprint: str,
    chunker_fingerprint: str,
    ordinal: int,
    text: str,
    element_ids: Sequence[str],
    provenance: Sequence[Provenance],
) -> str:
    payload = {
        "schema": 1,
        "source_sha256": source_sha256.lower(),
        "parser_fingerprint": parser_fingerprint,
        "chunker_fingerprint": chunker_fingerprint,
        "ordinal": ordinal,
        "text": normalize_text(text),
        "element_ids": list(element_ids),
        "provenance": [item.stable_dict() for item in provenance],
    }
    return f"chk_{_digest('lavix-chunk-v1', payload)}"


@dataclass(frozen=True, slots=True)
class CanonicalChunk:
    chunk_id: str
    ordinal: int
    text: str
    embedding_text: str
    element_ids: tuple[str, ...]
    provenance: tuple[Provenance, ...]
    section_path: tuple[str, ...] = ()
    token_count: int = 0
    # OG SmartChunker label: single, markdown_section, code_block, paragraph,
    # sentence, list, table, or mixed. Carried into the chunk metadata JSON.
    chunk_type: str = "mixed"

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("chunk ordinal must be non-negative")
        object.__setattr__(self, "text", normalize_text(self.text))
        object.__setattr__(self, "embedding_text", normalize_text(self.embedding_text))
        if not self.text:
            raise ValueError("canonical chunks cannot be empty")


@dataclass(frozen=True, slots=True)
class PublicationResult:
    """Signals that a sink atomically completed the durable job transition."""

    job_completed: bool = False


@dataclass(frozen=True, slots=True)
class IngestionJob:
    job_id: str
    file_id: int
    user_id: int
    revision: int
    state: JobState
    source_name: str
    media_type: str
    source_sha256: str = ""
    priority: int = 0
    attempts: int = 0
    max_attempts: int = 3
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    cancel_requested: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts


def deduplicate_provenance(items: Iterable[Provenance]) -> tuple[Provenance, ...]:
    seen: set[str] = set()
    result: list[Provenance] = []
    for item in items:
        key = _stable_json(item.stable_dict())
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)
