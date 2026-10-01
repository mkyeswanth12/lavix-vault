"""Embedding-model presets: prompt prefixes, context caps, chunk guidance.

Different embedding families expect different input shaping (e5/nomic want
query/document prefixes; bge/snowflake want raw text). Unknown models fall
back to no prefix and a conservative truncation bound with a warning, so a
new model degrades to plain behavior instead of failing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EmbeddingPreset:
    query_prefix: str = ""
    document_prefix: str = ""
    # Conservative char cap applied before the request (keeps us inside the
    # model's context even for unknown tokenizers; ~4 chars per token).
    max_input_chars: int = 2000
    recommended_chunk_tokens: int = 450


_DEFAULT_PRESET = EmbeddingPreset()

# Substring match on the (lowercased) model name, first hit wins.
_PRESETS: tuple[tuple[str, EmbeddingPreset], ...] = (
    ("nomic-embed", EmbeddingPreset(
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        max_input_chars=8000,
        recommended_chunk_tokens=500,
    )),
    ("mxbai-embed", EmbeddingPreset(
        query_prefix="Represent this sentence for searching relevant passages: ",
        document_prefix="",
        max_input_chars=1800,
        recommended_chunk_tokens=450,
    )),
    ("bge-m3", EmbeddingPreset(
        max_input_chars=30000,
        recommended_chunk_tokens=2000,
    )),
    ("bge-", EmbeddingPreset(
        max_input_chars=1800,
        recommended_chunk_tokens=450,
    )),
    ("e5-", EmbeddingPreset(
        query_prefix="query: ",
        document_prefix="passage: ",
        max_input_chars=1800,
        recommended_chunk_tokens=450,
    )),
    ("snowflake-arctic-embed", EmbeddingPreset(
        max_input_chars=30000,
        recommended_chunk_tokens=2000,
    )),
)


def preset_for(model: str) -> EmbeddingPreset:
    """Return the preset for a model name, or the conservative default."""
    name = (model or "").strip().lower()
    for marker, preset in _PRESETS:
        if marker in name:
            return preset
    if name:
        logger.warning("no embedding preset for model %r; using no-prefix default", model)
    return _DEFAULT_PRESET


def prepare_texts(model: str, texts: list[str], *, kind: str = "document") -> list[str]:
    """Apply the model preset: prefix + conservative truncation."""
    preset = preset_for(model)
    prefix = preset.query_prefix if kind == "query" else preset.document_prefix
    prepared: list[str] = []
    for text in texts:
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) > preset.max_input_chars:
            logger.warning(
                "truncating %d-char %s input for model %r (cap %d)",
                len(cleaned), kind, model, preset.max_input_chars,
            )
            cleaned = cleaned[: preset.max_input_chars]
        prepared.append(f"{prefix}{cleaned}" if prefix else cleaned)
    return prepared
