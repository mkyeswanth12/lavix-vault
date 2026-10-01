"""Tenant-fenced pgvector/FTS retrieval and optional Infinity reranking."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any, Protocol

import httpx

from app.config import settings
from app.database import get_db
from app.ingestion.embedding import OllamaEmbeddingClient
from app.services.model_config import DEFAULT_FILE_SCOPE, ModelConfigurationRepository

logger = logging.getLogger(__name__)


# Reranker input bounds: full-size payloads exceed any interactive
# timeout on CPU inference. The historic 24-row window is now the floor,
# not the ceiling: the head scales with the requested top_k up to the hard
# cap, while the interactive timeout plus fail-open ordering remains the
# independent latency backstop.
_RERANK_CANDIDATE_CAP = 24
_RERANK_HARD_CAP = 500
_RERANK_DOC_CHAR_CAP = 600


class ConnectionLike(Protocol):
    def cursor(self) -> Any: ...


ConnectionFactory = Callable[[], AbstractContextManager[ConnectionLike]]


HYBRID_SEARCH_SQL = """
WITH eligible AS MATERIALIZED (
    SELECT
        dc.chunk_id, dc.file_id, dc.revision, dc.ordinal, dc.content,
        dc.element_ids, dc.provenance, dc.section_path, dc.token_count,
        f.display_name, f.original_filename, f.mime_type,
        f.quick_summary, f.quick_tags,
        to_jsonb(f)->>'doc_type' AS doc_type,
        to_jsonb(f)->>'intelligence_status' AS intelligence_status
    FROM document_chunks AS dc
    JOIN document_revisions AS dr
      ON dr.file_id = dc.file_id AND dr.revision = dc.revision
    JOIN files AS f ON f.id = dc.file_id
    WHERE dc.user_id = %s
      AND dr.user_id = %s
      AND f.user_id = %s
      AND f.is_deleted = FALSE
      AND f.user_granted_ai_access = TRUE
      AND f.current_revision = dc.revision
      AND dr.status = 'current'
      AND (%s::INTEGER[] IS NULL OR dc.file_id = ANY(%s::INTEGER[]))
),
vector_candidates AS MATERIALIZED (
    SELECT
        chunk_id, file_id, revision,
        GREATEST(0.0, 1.0 - (embedding <=> %s::vector)) AS vector_score,
        0.0::REAL AS lexical_score,
        FALSE AS literal_match,
        0.0::REAL AS summary_tag_score
    FROM document_chunks
    WHERE user_id = %s
      AND embedding IS NOT NULL
      AND EXISTS (
          SELECT 1 FROM eligible
          WHERE eligible.file_id = document_chunks.file_id
            AND eligible.revision = document_chunks.revision
            AND eligible.chunk_id = document_chunks.chunk_id
      )
    ORDER BY embedding <=> %s::vector
    LIMIT %s
),
lexical_candidates AS MATERIALIZED (
    SELECT
        dc.chunk_id, dc.file_id, dc.revision,
        0.0::REAL AS vector_score,
        ts_rank_cd(dc.search_vector, websearch_to_tsquery('english', %s)) AS lexical_score,
        FALSE AS literal_match,
        0.0::REAL AS summary_tag_score
    FROM document_chunks AS dc
    WHERE dc.user_id = %s
      AND dc.search_vector @@ websearch_to_tsquery('english', %s)
      AND EXISTS (
          SELECT 1 FROM eligible
          WHERE eligible.file_id = dc.file_id
            AND eligible.revision = dc.revision
            AND eligible.chunk_id = dc.chunk_id
      )
    ORDER BY lexical_score DESC
    LIMIT %s
),
literal_candidates AS MATERIALIZED (
    SELECT DISTINCT ON (file_id)
        chunk_id, file_id, revision,
        0.0::REAL AS vector_score,
        0.0::REAL AS lexical_score,
        TRUE AS literal_match,
        0.0::REAL AS summary_tag_score
    FROM eligible
    WHERE %s::TEXT IS NOT NULL
      AND STRPOS(LOWER(content), %s::TEXT) > 0
    ORDER BY file_id, ordinal
    LIMIT %s
),
summary_tag_candidates AS MATERIALIZED (
    SELECT DISTINCT ON (eligible.file_id)
        eligible.chunk_id, eligible.file_id, eligible.revision,
        0.0::REAL AS vector_score,
        0.0::REAL AS lexical_score,
        FALSE AS literal_match,
        0.50::REAL AS summary_tag_score
    FROM eligible
    LEFT JOIN vector_candidates AS vc
      ON vc.chunk_id = eligible.chunk_id
     AND vc.file_id = eligible.file_id
     AND vc.revision = eligible.revision
    WHERE LOWER(COALESCE(eligible.quick_summary, '')) LIKE ANY(
        SELECT '%%' || word || '%%' FROM regexp_split_to_table(LOWER(%s), '[[:space:]]+') AS word
        WHERE LENGTH(word) > 2
    )
    OR EXISTS (
        SELECT 1 FROM unnest(COALESCE(eligible.quick_tags, ARRAY[]::TEXT[])) AS tag
        WHERE LOWER(tag) LIKE ANY(
            SELECT '%%' || word || '%%' FROM regexp_split_to_table(LOWER(%s), '[[:space:]]+') AS word
            WHERE LENGTH(word) > 2
        )
    )
    -- Deterministic representative: the file's best vector chunk carries the
    -- file-level summary signal, not an arbitrary first row.
    ORDER BY eligible.file_id, vc.vector_score DESC NULLS LAST
    LIMIT %s
),
rare_file_counts AS MATERIALIZED (
    SELECT w.word, COUNT(DISTINCT dc.file_id) AS file_count
    FROM unnest(%s::TEXT[]) AS w(word)
    JOIN document_chunks AS dc ON dc.search_vector @@ plainto_tsquery('english', w.word)
    JOIN files AS f ON f.id = dc.file_id
    WHERE dc.user_id = %s
      AND f.user_id = %s
      AND f.is_deleted = FALSE
      AND f.user_granted_ai_access = TRUE
      AND f.current_revision = dc.revision
      AND (%s::INTEGER[] IS NULL OR dc.file_id = ANY(%s::INTEGER[]))
    GROUP BY w.word
),
pin_candidates AS MATERIALIZED (
    SELECT
        eligible.chunk_id, eligible.file_id, eligible.revision,
        0.0::REAL AS vector_score,
        0.0::REAL AS lexical_score,
        TRUE AS literal_match,
        -- A rare identifier visibly present is summary-grade evidence: score
        -- it like a file-level summary hit so pinned rows display honest
        -- nonzero scores even when the reranker is down.
        0.50::REAL AS summary_tag_score
    FROM eligible
    WHERE EXISTS (
        SELECT 1 FROM rare_file_counts AS rfc
        WHERE rfc.file_count BETWEEN 1 AND 5
          AND STRPOS(LOWER(eligible.content), rfc.word) > 0
    )
    LIMIT 12
),
scores AS (
    SELECT
        chunk_id, file_id, revision,
        MAX(vector_score) AS vector_score,
        MAX(lexical_score) AS lexical_score,
        BOOL_OR(literal_match) AS literal_match,
        MAX(summary_tag_score) AS summary_tag_score
    FROM (
        SELECT * FROM vector_candidates
        UNION ALL
        SELECT * FROM lexical_candidates
        UNION ALL
        SELECT * FROM literal_candidates
        UNION ALL
        SELECT * FROM summary_tag_candidates
        UNION ALL
        SELECT * FROM pin_candidates
    ) AS candidates
    GROUP BY chunk_id, file_id, revision
)
SELECT
    eligible.*,
    scores.vector_score,
    scores.lexical_score,
    scores.literal_match,
    scores.summary_tag_score,
    -- Blend weights: vector dominates; lexical is scaled (ts_rank_cd runs
    -- small); summary_tag_score carries 0.14 base + 0.30 file-match bonus so
    -- a file-level summary/tag hit clears the vector noise floor (~0.2).
    (0.68 * scores.vector_score + 0.18 * LEAST(1.0, scores.lexical_score * 4.0) + 0.44 * scores.summary_tag_score) AS score
FROM scores
JOIN eligible USING (chunk_id, file_id, revision)
ORDER BY literal_match DESC, score DESC, file_id, ordinal
LIMIT %s
"""


def _vector_literal(vector: Sequence[float]) -> str:
    if not vector or any(not math.isfinite(float(value)) for value in vector):
        raise ValueError("embedding contains non-finite values")
    return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"


def _literal_query(query: str) -> str | None:
    """Return literals worth pinning ahead of approximate semantic matches."""

    value = query.strip()
    if len(value) < 3:
        return None
    if len(value.split()) > 1 or any(not character.isalpha() for character in value):
        return value.casefold()
    return None


# Rare-term pinning: a distinctive word living in a handful of files is an
# identifier, not a keyword. The whole-query literal pin cannot fire on
# natural questions ("did we have RAJESH invoice? ..."), so single rare
# words get their own deterministic pin path (same prepend machinery).
_RARE_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_RARE_MIN_WORD_LEN = 3
_RARE_MAX_WORDS = 10
_RARE_DF_MAX = 5


def _rare_query_words(query: str) -> list[str]:
    """Distinctive candidate words: lowercase alphanumerics, length-gated."""
    words: list[str] = []
    for word in _RARE_WORD_RE.findall(query.lower()):
        if len(word) >= _RARE_MIN_WORD_LEN and word not in words:
            words.append(word)
        if len(words) >= _RARE_MAX_WORDS:
            break
    return words


class InfinityReranker:
    def __init__(self, base_url: str | None = None, *, timeout_seconds: float | None = None) -> None:
        self._url = (base_url or settings.rerank_base_url).rstrip("/") + "/rerank"
        self._timeout = timeout_seconds or settings.interactive_rerank_timeout

    async def rerank(
        self,
        query: str,
        rows: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        documents = [
            str(row.get("content") or "")[: min(settings.rerank_max_input_chars, _RERANK_DOC_CHAR_CAP)]
            for row in rows
        ]
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._url,
                    json={"model": settings.rerank_model, "query": query, "documents": documents},
                )
                response.raise_for_status()
                result = response.json().get("results", [])
        except (httpx.HTTPError, ValueError, TypeError):
            return rows[:top_k]

        ranked: list[tuple[float, dict[str, Any]]] = []
        for value in result:
            try:
                index = int(value["index"])
                score = float(value["relevance_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < len(rows) and math.isfinite(score):
                item = dict(rows[index])
                item["rerank_score"] = score
                ranked.append((score, item))
        if not ranked:
            return rows[:top_k]
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[:top_k]]


def _ensure_scoped_file_representation(
    ranked: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    file_ids: Sequence[int] | None,
    *,
    top_k: int,
    min_score: float,
) -> list[dict[str, Any]]:
    """Keep one best chunk per explicitly requested file (BUG-003).

    Ranking is global, so one file can consume the whole candidate budget
    and a second explicitly scoped file vanishes entirely. When the caller
    scoped multiple files, re-attach each missing file's best hybrid-score
    chunk. The hybrid floor (same default as the rerank floor) stays the
    relevance bar, so an irrelevant file still stays out and existing
    refusal paths handle the empty case. Appends at most one row per
    missing file; global thresholds and ordering are otherwise untouched.
    """
    if not ranked or top_k <= 0 or not file_ids:
        return ranked
    try:
        wanted = list(dict.fromkeys(int(fid) for fid in file_ids))
    except (TypeError, ValueError):
        return ranked
    if len(wanted) < 2:
        return ranked
    present = set()
    for row in ranked:
        try:
            present.add(int(row.get("file_id")))
        except (TypeError, ValueError):
            continue
    missing = [fid for fid in wanted if fid not in present]
    if not missing:
        return ranked
    missing_set = set(missing)
    best: dict[int, dict[str, Any]] = {}
    for row in rows:  # rows are score-ordered; first hit per file wins
        try:
            fid = int(row.get("file_id"))
        except (TypeError, ValueError):
            continue
        if fid in missing_set and fid not in best:
            try:
                score = min(1.0, float(row.get("score", 0.0)))
            except (TypeError, ValueError):
                continue
            if score >= min_score:
                best[fid] = row
    if not best:
        return ranked
    have_chunks = {str(row.get("chunk_id")) for row in ranked}
    extra = [row for row in best.values() if str(row.get("chunk_id")) not in have_chunks]
    if not extra:
        return ranked
    return (ranked + extra)[: top_k + len(extra)]


def _effective_score(row: dict[str, Any]) -> float | None:
    """Best available relevance score for a candidate row, if any."""
    best: float | None = None
    for key in ("rerank_score", "score"):
        try:
            value = float(row.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if best is None or value > best:
            best = value
    return best


def _apply_discovery_floor(
    ranked: list[dict[str, Any]], *, floor: float
) -> list[dict[str, Any]]:
    """Drop sub-floor rows from unscoped (discovery) retrieval.

    When no chunk reaches the discovery floor, return only exact literal
    pins (possibly empty) instead of best-effort junk: with no file scope
    and no genuinely relevant chunk, the caller answers via the
    unverified_explanatory path rather than grounding in noise. Scoped
    (@-tagged) retrieval never passes through here.
    """
    survivors = [
        row
        for row in ranked
        if row.get("literal_match") or (_effective_score(row) or -1.0) >= floor
    ]
    return survivors


def _apply_file_scope_cap(
    ranked: list[dict[str, Any]],
    *,
    max_files: int,
) -> list[dict[str, Any]]:
    """Cap distinct files in discovery/unscoped results (admin file_scope).

    Keeps rank order; every file that survives keeps all of its ranked
    chunks. Files beyond the cap are dropped wholesale — no partial file
    representation. A non-positive cap is treated as unset (no truncation).
    """
    if max_files <= 0:
        return ranked
    kept: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in ranked:
        try:
            file_id = int(row.get("file_id"))
        except (TypeError, ValueError):
            kept.append(row)
            continue
        if file_id in seen:
            kept.append(row)
        elif len(seen) < max_files:
            seen.add(file_id)
            kept.append(row)
    return kept


class VaultRetriever:
    def __init__(
        self,
        connection_factory: ConnectionFactory = get_db,
        *,
        embedding_client: OllamaEmbeddingClient | None = None,
        reranker: InfinityReranker | None = None,
        reranking_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._connections = connection_factory
        self._embeddings = embedding_client or OllamaEmbeddingClient()
        self._reranker = reranker or InfinityReranker()
        self._reranking_enabled = reranking_enabled or self._configured_reranking_enabled

    async def search(
        self,
        *,
        user_id: int,
        query: str,
        file_ids: Sequence[int] | None,
        top_k: int,
        deep_search: bool,
    ) -> list[dict[str, Any]]:
        prepared = await self._embeddings.embed([query], kind="query")
        vector = _vector_literal(prepared.vectors[0])
        candidate_limit = min(500, max(top_k * (6 if deep_search else 4), top_k))
        rows = await asyncio.to_thread(
            self._query,
            user_id,
            query,
            list(file_ids) if file_ids is not None else None,
            vector,
            candidate_limit,
        )
        literal_rows = [row for row in rows if row.get("literal_match")]
        reranking_enabled = await asyncio.to_thread(self._reranking_enabled)
        # Rerank headroom scales with the requested top_k (bounded by the
        # hard cap) so larger candidate pools are actually reranked instead
        # of silently falling back to the historic 24-row window. The
        # interactive timeout plus fail-open ordering remains the
        # independent latency backstop.
        rerank_head = rows[: min(max(top_k, _RERANK_CANDIDATE_CAP), _RERANK_HARD_CAP)]
        ranked = (
            await self._reranker.rerank(query, rerank_head, top_k=top_k)
            if reranking_enabled
            else rows[:top_k]
        )
        scored_rows = [row for row in ranked if "rerank_score" in row]
        has_relevant_score = any(
            float(row["rerank_score"]) >= settings.rerank_min_score for row in scored_rows
        )
        if has_relevant_score:
            ranked = [
                row
                for row in ranked
                if "rerank_score" not in row or float(row["rerank_score"]) >= settings.rerank_min_score
            ]
        if literal_rows:
            # Exact identifiers and phrases must survive approximate ranking and
            # an optional reranker timeout/threshold. SQL contributes at most one
            # pinned chunk per file, so repeated text cannot consume every slot.
            ranked_by_chunk = {str(row["chunk_id"]): row for row in ranked}
            pinned = [
                ranked_by_chunk.get(str(row["chunk_id"]), row)
                for row in literal_rows
            ]
            pinned_ids = {str(row["chunk_id"]) for row in pinned}
            ranked = (
                pinned
                + [row for row in ranked if str(row["chunk_id"]) not in pinned_ids]
            )[:top_k]
        ranked = _ensure_scoped_file_representation(
            ranked,
            rows,
            list(file_ids) if file_ids is not None else None,
            top_k=top_k,
            min_score=float(settings.rerank_min_score),
        )
        if file_ids is None:
            # Discovery/unscoped retrieval: first apply the discovery
            # floor — sub-threshold rows are junk, not evidence — then
            # enforce the admin file-scope cap on distinct files
            # (first-occurrence order). Explicitly scoped runs bypass both
            # the floor entirely and the cap: tagged files are always fully
            # represented (up to the per-request scope ceiling).
            ranked = _apply_discovery_floor(
                ranked, floor=float(settings.discovery_min_score)
            )
            ranked = _apply_file_scope_cap(
                ranked,
                max_files=await asyncio.to_thread(self._configured_file_scope),
            )
        return [self._evidence(row) for row in ranked]

    def _configured_reranking_enabled(self) -> bool:
        """Resolve the admin setting for every search without caching a stale role.
        Environment configuration remains the availability fallback while the
        singleton row has not been created.  If configuration cannot be read,
        retrieval remains available using that same deployment fallback.
        RERANKER_MODE=off always wins: no reranker call is made and search
        degrades gracefully to hybrid order.
        """

        try:
            if settings.rerank_mode == "off":
                return False
        except Exception as exc:  # misconfiguration must not kill retrieval
            logger.warning("Could not read reranker mode; assuming enabled (%s)", exc)
        try:
            with self._connections() as connection:
                return ModelConfigurationRepository(connection).get().reranker_enabled
        except Exception as exc:  # availability boundary; never fail all retrieval
            logger.warning(
                "Could not read system reranker configuration; using deployment default (%s)",
                type(exc).__name__,
            )
            return settings.enable_reranking

    def _configured_file_scope(self) -> int:
        """Resolve the admin file-scope cap per search (same pattern).

        Falls back to the default (5) when unreadable: retrieval stays
        available and discovery keeps its historical effective width.
        """

        try:
            with self._connections() as connection:
                return int(ModelConfigurationRepository(connection).get().file_scope)
        except Exception as exc:  # availability boundary; never fail all retrieval
            logger.warning(
                "Could not read system file-scope configuration; using default (%s)",
                type(exc).__name__,
            )
            return int(DEFAULT_FILE_SCOPE)

    def _query(
        self,
        user_id: int,
        query: str,
        file_ids: list[int] | None,
        vector: str,
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        literal_query = _literal_query(query)
        params = (
            user_id,
            user_id,
            user_id,
            file_ids,
            file_ids,
            vector,
            user_id,
            vector,
            candidate_limit,
            query,
            user_id,
            query,
            candidate_limit,
            literal_query,
            literal_query,
            candidate_limit,
            query,
            query,
            candidate_limit,
            _rare_query_words(query),
            user_id,
            user_id,
            file_ids,
            file_ids,
            candidate_limit,
        )
        with self._connections() as connection:
            from app.db.embedding_store import resolve_vector_cast

            cast = resolve_vector_cast(connection, "document_chunks")
            sql = HYBRID_SEARCH_SQL.replace("::vector", f"::{cast}")
            cursor = connection.cursor()
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def _evidence(row: dict[str, Any]) -> dict[str, Any]:
        provenance = row.get("provenance") or []
        if isinstance(provenance, str):
            try:
                provenance = json.loads(provenance)
            except json.JSONDecodeError:
                provenance = []
        intelligence_status = str(row.get("intelligence_status") or "pending")
        intelligence_ready = intelligence_status in {"model", "fallback"}
        tags = (row.get("quick_tags") or []) if intelligence_ready else []
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = []
        score = max(
            0.0,
            min(1.0, float(row.get("score", 0.0))),
            min(1.0, float(row.get("rerank_score", 0.0))),
        )
        return {
            "kind": "vault",
            "chunk_id": str(row["chunk_id"]),
            "file_id": int(row["file_id"]),
            "revision": int(row["revision"]),
            "filename": str(row.get("display_name") or row.get("original_filename") or "document"),
            "mime_type": str(row.get("mime_type") or "application/octet-stream"),
            "summary": (str(row.get("quick_summary") or "").strip() or None if intelligence_ready else None),
            "tags": [str(tag) for tag in tags[:12]] if isinstance(tags, (list, tuple)) else [],
            "doc_type": (str(row.get("doc_type") or "").strip() or None if intelligence_ready else None),
            "content": str(row.get("content") or "")[:8_000],
            "section_path": list(row.get("section_path") or []),
            "provenance": provenance,
            "score": round(score, 6),
            "match_percentage": round(score * 100),
            "reranked": "rerank_score" in row,
        }
