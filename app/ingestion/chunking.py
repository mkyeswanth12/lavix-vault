"""Deterministic, citation-preserving canonical chunking with type-aware routing."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import (
    CanonicalChunk,
    CanonicalDocument,
    CanonicalElement,
    ElementType,
    Provenance,
    build_chunk_id,
    deduplicate_provenance,
    normalize_text,
)

logger = logging.getLogger(__name__)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def split(self, text: str, max_tokens: int) -> list[str]: ...


class RegexTokenCounter:
    """Dependency-free deterministic counter used until a model tokenizer is injected."""

    _token = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count(self, text: str) -> int:
        return len(self._token.findall(text or ""))

    def split(self, text: str, max_tokens: int) -> list[str]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        normalized = normalize_text(text)
        matches = list(self._token.finditer(normalized))
        if not matches:
            return []
        pieces: list[str] = []
        for start in range(0, len(matches), max_tokens):
            group = matches[start : start + max_tokens]
            raw = normalized[group[0].start() : group[-1].end()]
            pieces.append(normalize_text(raw))
        return [piece for piece in pieces if piece]


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    target_tokens: int = 450
    max_tokens: int = 700
    overlap_tokens: int = 60

    def __post_init__(self) -> None:
        if self.target_tokens < 1 or self.max_tokens < self.target_tokens:
            raise ValueError("chunk token bounds are invalid")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be between zero and max_tokens")

    @property
    def fingerprint(self) -> str:
        return f"canonical-v3:target={self.target_tokens}:max={self.max_tokens}:overlap={self.overlap_tokens}"


# Content shapes ported from the OG v0.1.0 SmartChunker router
# (extension-first, then content scoring). Shapes only change *grouping*;
# budgets, provenance, and deterministic IDs are untouched.
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go",
    ".rs", ".rb", ".php", ".sh", ".bash",
}
_HEADED_EXTENSIONS = {".md", ".markdown", ".rst"}

_MARKDOWN_PATTERNS = (
    r"^#{1,6}\s+",
    r"^\*{3,}$",
    r"^[-*+]\s+",
    r"^\d+\.\s+",
    r"^```",
    r"^\[.+\]\(.+\)",
)
_CODE_PATTERNS = (
    r"^\s*(def|class|function|const|let|var|import|from|public|private)\s+",
    r"^\s*(if|for|while|return|try|except|raise)\s*[\(:)]",
    r"^\s*[{}();]",
    r"^\s*#.*coding[:=]",
)


@dataclass(slots=True)
class _Unit:
    text: str
    element_ids: tuple[str, ...]
    provenance: tuple[Provenance, ...]
    section_path: tuple[str, ...]
    boundary: tuple[object, ...]
    is_heading: bool = False
    is_code: bool = False
    chunk_type: str = "paragraph"


# OG SmartChunker labels per element kind. Groups take the unanimous unit
# label, "mixed" for blended groups, "single" for tiny whole documents.
_ELEMENT_CHUNK_TYPES = {
    ElementType.TITLE: "markdown_section",
    ElementType.HEADING: "markdown_section",
    ElementType.PARAGRAPH: "paragraph",
    ElementType.LIST: "paragraph",
    ElementType.TABLE: "table",
    ElementType.FORMULA: "paragraph",
    ElementType.CODE: "code_block",
    ElementType.CAPTION: "paragraph",
    ElementType.IMAGE: "paragraph",
    ElementType.TEXT: "sentence",
    ElementType.OTHER: "sentence",
}

# OG sentence splitter (SmartChunker._chunk_sentences): greedy sentence
# packing is also the overflow strategy for oversize prose elements.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# OG oversize back-scan: step back at most this many chars for a
# sentence/line boundary before accepting a hard cut.
_BACKSCAN_CHARS = 200


class CanonicalChunker:
    def __init__(
        self,
        policy: ChunkingPolicy | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.policy = policy or ChunkingPolicy()
        self.tokens = token_counter or RegexTokenCounter()

    def chunk(self, document: CanonicalDocument) -> tuple[CanonicalChunk, ...]:
        units = self._units(document.elements)
        if not units:
            return ()
        # OG single fast path: tiny documents stay one chunk, labelled single.
        if len(units) == 1 and self.tokens.count(units[0].text) < self.policy.target_tokens:
            return (self._build(document, units, 0, "single", "", None),)
        shape = self._detect_shape(document)
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        current_tokens = 0
        current_boundary: tuple[object, ...] | None = None

        for unit in units:
            unit_tokens = self.tokens.count(unit.text)
            boundary_changed = current and unit.boundary != current_boundary
            target_reached = current and current_tokens >= self.policy.target_tokens
            would_overflow = current and current_tokens + unit_tokens > self.policy.max_tokens
            # OG headed: a new heading starts a new group AND a section is
            # never cut for size below max — sections stay whole.
            section_break = bool(current and shape != "prose" and unit.is_heading)
            atomic_section = shape == "headed" and not unit.is_heading
            # OG code: every code block is its own chunk, never packed.
            code_block_break = bool(current and shape == "code" and unit.is_code)
            if (
                boundary_changed
                or section_break
                or code_block_break
                or would_overflow
                or (target_reached and not atomic_section)
            ):
                groups.append(current)
                current = []
                current_tokens = 0
            if not current:
                current_boundary = unit.boundary
            current.append(unit)
            current_tokens += unit_tokens
        if current:
            groups.append(current)
        logger.info(
            "Chunk router shape=%s groups=%d units=%d source=%s",
            shape,
            len(groups),
            len(units),
            document.source_name,
        )

        chunks: list[CanonicalChunk] = []
        previous_tail = ""
        previous_boundary: tuple[object, ...] | None = None
        for ordinal, group in enumerate(groups):
            types = {unit.chunk_type for unit in group}
            chunk_type = types.pop() if len(types) == 1 else "mixed"
            chunk, previous_tail, previous_boundary = self._assemble(
                document, group, ordinal, chunk_type, previous_tail, previous_boundary
            )
            chunks.append(chunk)
        return tuple(chunks)

    def _build(
        self,
        document: CanonicalDocument,
        units: list[_Unit],
        ordinal: int,
        chunk_type: str,
        previous_tail: str,
        previous_boundary: tuple[object, ...] | None,
    ) -> CanonicalChunk:
        chunk, _, _ = self._assemble(
            document, units, ordinal, chunk_type, previous_tail, previous_boundary
        )
        return chunk

    def _assemble(
        self,
        document: CanonicalDocument,
        group: list[_Unit],
        ordinal: int,
        chunk_type: str,
        previous_tail: str,
        previous_boundary: tuple[object, ...] | None,
    ) -> tuple[CanonicalChunk, str, tuple[object, ...] | None]:
        text = normalize_text("\n\n".join(unit.text for unit in group))
        provenance = deduplicate_provenance(item for unit in group for item in unit.provenance)
        element_ids = tuple(dict.fromkeys(item for unit in group for item in unit.element_ids))
        section = group[-1].section_path
        boundary = group[0].boundary
        breadcrumb = " > ".join(section)
        prefix_parts = [
            part for part in (breadcrumb, previous_tail if boundary == previous_boundary else "") if part
        ]
        embedding_text = "\n".join([*prefix_parts, text])
        chunk_id = build_chunk_id(
            source_sha256=document.source_sha256,
            parser_fingerprint=document.parser_fingerprint,
            chunker_fingerprint=self.policy.fingerprint,
            ordinal=ordinal,
            text=text,
            element_ids=element_ids,
            provenance=provenance,
        )
        chunk = CanonicalChunk(
            chunk_id=chunk_id,
            ordinal=ordinal,
            text=text,
            embedding_text=embedding_text,
            element_ids=element_ids,
            provenance=provenance,
            section_path=section,
            token_count=self.tokens.count(text),
            chunk_type=chunk_type,
        )
        return chunk, self._tail(text), boundary

    def _detect_shape(self, document: CanonicalDocument) -> str:
        """Classify content shape: 'code', 'headed', or 'prose' (OG router port)."""
        ext = Path(document.source_name or "").suffix.lower()
        if ext in _CODE_EXTENSIONS:
            return "code"
        if ext in _HEADED_EXTENSIONS:
            return "headed"
        code_chars = sum(
            len(element.text)
            for element in document.elements
            if element.element_type is ElementType.CODE and element.text
        )
        total_chars = sum(len(element.text) for element in document.elements if element.text)
        if total_chars and code_chars / total_chars >= 0.2:
            return "code"
        if any(
            element.element_type in {ElementType.TITLE, ElementType.HEADING}
            for element in document.elements
        ):
            return "headed"
        lines: list[str] = []
        for element in document.elements:
            if element.text:
                lines.extend(element.text.split("\n"))
            if len(lines) >= 50:
                break
        lines = lines[:50]
        markdown_score = sum(
            1 for line in lines for pattern in _MARKDOWN_PATTERNS if re.match(pattern, line)
        )
        code_score = sum(
            1 for line in lines for pattern in _CODE_PATTERNS if re.match(pattern, line)
        )
        if markdown_score >= 3:
            return "headed"
        if code_score >= 3:
            return "code"
        return "prose"

    def _units(self, elements: Sequence[CanonicalElement]) -> list[_Unit]:
        units: list[_Unit] = []
        section: list[str] = []
        for element in elements:
            if not element.text:
                continue
            if element.element_type in {ElementType.TITLE, ElementType.HEADING}:
                level = element.heading_level or 1
                section = section[: max(0, level - 1)]
                section.append(element.text)
            explicit_section = next(
                (prov.section_path for prov in element.provenance if prov.section_path),
                tuple(section),
            )
            provenance = element.provenance or (Provenance(section_path=tuple(explicit_section)),)
            boundary = provenance[0].boundary_key()
            pieces = self._split_element(element)
            is_heading = element.element_type in {ElementType.TITLE, ElementType.HEADING}
            is_code = element.element_type is ElementType.CODE
            chunk_type = _ELEMENT_CHUNK_TYPES.get(element.element_type, "sentence")
            for piece in pieces:
                units.append(
                    _Unit(
                        text=piece,
                        element_ids=(element.element_id,),
                        provenance=provenance,
                        section_path=tuple(explicit_section),
                        boundary=boundary,
                        is_heading=is_heading,
                        is_code=is_code,
                        chunk_type=chunk_type,
                    )
                )
        return units

    def _split_element(self, element: CanonicalElement) -> list[str]:
        if self.tokens.count(element.text) <= self.policy.max_tokens:
            return [element.text]
        if element.element_type is ElementType.TABLE and "\n" in element.text:
            return self._split_table(element.text)
        if element.element_type is ElementType.CODE:
            # OG code: split oversize blocks on blank lines, never mid-line.
            blocks = [b for b in element.text.split("\n\n") if b.strip()]
            if len(blocks) > 1:
                return self._pack_sentences(blocks, "\n\n")
            return self._backscan_split(element.text)
        return self._pack_sentences(
            [s for s in _SENTENCE_SPLIT.split(element.text) if s.strip()], " "
        )

    def _pack_sentences(self, parts: list[str], joiner: str) -> list[str]:
        """OG plain strategy: greedy-pack sentences to max_tokens (no mid-sentence cuts)."""
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for part in parts:
            part_tokens = self.tokens.count(part)
            if part_tokens > self.policy.max_tokens:
                if current:
                    result.append(normalize_text(joiner.join(current)))
                    current = []
                    current_tokens = 0
                result.extend(self._backscan_split(part))
                continue
            if current and current_tokens + part_tokens > self.policy.max_tokens:
                result.append(normalize_text(joiner.join(current)))
                current = []
                current_tokens = 0
            current.append(part)
            current_tokens += part_tokens
        if current:
            result.append(normalize_text(joiner.join(current)))
        return [piece for piece in result if piece]

    def _backscan_split(self, text: str) -> list[str]:
        """OG oversize strategy: hard-cut at max, step back for a boundary."""
        pieces: list[str] = []
        rest = normalize_text(text)
        while rest and self.tokens.count(rest) > self.policy.max_tokens:
            cut = self.tokens.split(rest, self.policy.max_tokens)[0]
            scan_from = max(0, len(cut) - _BACKSCAN_CHARS)
            break_at = max(
                cut.rfind("\n", scan_from),
                cut.rfind(". ", scan_from),
                cut.rfind("? ", scan_from),
                cut.rfind("! ", scan_from),
            )
            if break_at > 0:
                head, rest = cut[: break_at + 1], (cut[break_at + 1 :] + rest[len(cut) :])
            else:
                head, rest = cut, rest[len(cut) :]
            head = normalize_text(head)
            rest = normalize_text(rest)
            if head:
                pieces.append(head)
        if rest:
            pieces.append(rest)
        return pieces

    def _split_table(self, text: str) -> list[str]:
        rows = [row for row in text.splitlines() if row.strip()]
        if len(rows) < 2:
            return self.tokens.split(text, self.policy.max_tokens)
        header = rows[0]
        result: list[str] = []
        current = [header]
        for row in rows[1:]:
            proposed = "\n".join([*current, row])
            if len(current) > 1 and self.tokens.count(proposed) > self.policy.max_tokens:
                result.append(normalize_text("\n".join(current)))
                current = [header, row]
            else:
                current.append(row)
        if current:
            result.append(normalize_text("\n".join(current)))
        bounded: list[str] = []
        for piece in result:
            if self.tokens.count(piece) <= self.policy.max_tokens:
                bounded.append(piece)
            else:
                bounded.extend(self.tokens.split(piece, self.policy.max_tokens))
        return bounded

    def _tail(self, text: str) -> str:
        if self.policy.overlap_tokens == 0:
            return ""
        tokens = RegexTokenCounter._token.findall(text)
        return " ".join(tokens[-self.policy.overlap_tokens :])
