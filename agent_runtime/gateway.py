"""Capability-authenticated client for Lavix-owned agent tools."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

import httpx

from .events import qlog as _qlog
from .scope import current_run_scope

logger = logging.getLogger(__name__)

MAX_VAULT_RESULTS = 500
# Ceiling for per-leg web results. Raised from 5 to 12 so the admin
# search_depth "pro" tier (12 results/leg) can pass through; lower tiers
# pass their own explicit values and are unaffected by the ceiling.
MAX_WEB_RESULTS = 12
MAX_GRAPH_RESULTS = 8
MAX_QUERY_CHARS = 1_000
MAX_TOOL_EVIDENCE = 6
# Vault excerpt budget (also the graph-memory compaction budget below).
# Unchanged: vault retrieval is not governed by the web search_depth setting.
MAX_TOOL_CONTENT_CHARS = 2_000
# Default web excerpt budget — the conservative search_depth tier. The
# per-request tier value rides search_web(excerpt_chars=...); this default
# only guards direct callers that omit it.
MAX_WEB_TOOL_CONTENT_CHARS = 2_000


_LAVIX_BRACKET_MARKER = re.compile(r"\[lavix[^\[\]]*?\]", re.IGNORECASE)
_LAVIX_TAG_MARKER = re.compile(r"</?lavix[^<>]*>", re.IGNORECASE)


def neutralize_lavix_markers(value: Any) -> Any:
    """Strip LAVIX-directive-shaped markers from untrusted text.

    Removes ``[LAVIX ...]`` bracket shapes and ``<LAVIX_...>`` tag shapes
    (any case) so directive-looking smuggling in user input or retrieved
    content cannot reach a model prompt intact. Non-LAVIX brackets
    (``[REMINDER]``, ``[1]``, markdown links) pass through untouched, as
    does an unclosed ``<lavix`` without a closing angle bracket.
    """

    if not isinstance(value, str) or not value:
        return value
    cleaned = _LAVIX_BRACKET_MARKER.sub("", value)
    cleaned = _LAVIX_TAG_MARKER.sub("", cleaned)
    return cleaned


def _bounded(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _dedupe_positive(values: Iterable[int] | None) -> list[int] | None:
    if values is None:
        return None
    return list(dict.fromkeys(int(value) for value in values if int(value) > 0))


_NUMBER_ATOM = re.compile(r"[\u20b9$\u20ac\u00a3\u00a5]?\d[\d,]*\.?\d*%?")


def _snap_out_of_number(text: str, pos: int, *, edge: str) -> int:
    """Move a cut position out of a numeric atom (BUG-004).

    Slicing inside ``7,799`` orphans digits (``799``), and small models
    concatenate the orphan with a neighbouring digit (``9`` + ``799`` =
    ``97,799``). A cut strictly inside an atom snaps to the atom's safe
    edge instead; cuts on atom boundaries are untouched.
    """
    for match in _NUMBER_ATOM.finditer(text):
        start, end = match.span()
        if start < pos < end:
            return start if edge == "start" else min(end, len(text))
    return pos


def _evidence_excerpt(query: str, content: Any, *, max_chars: int = MAX_TOOL_CONTENT_CHARS) -> str:
    text = str(content or "")
    if len(text) <= max_chars:
        return text
    lowered = text.lower()
    position = -1
    for term in sorted(set(re.findall(r"\w{4,}", query)), key=len, reverse=True):
        position = lowered.find(term.lower())
        if position >= 0:
            break
    start = max(0, position - max_chars // 4) if position >= 0 else 0
    start = min(start, len(text) - max_chars)
    end = start + max_chars
    start = _snap_out_of_number(text, start, edge="start")
    end = _snap_out_of_number(text, end, edge="end")
    # Sentence-integrity snap (BUG-003 follow-up): pairing and grounding
    # bind entities to numbers clause-locally. A window edge through a
    # sentence orphans the pairing on both sides ("growth of 4%" split
    # from its "Qatar" subject), manufacturing false contradictions.
    # Snap to sentence boundaries within a bounded slop so windows stay
    # whole claims without runaway growth on unpunctuated OCR text.
    start = _snap_to_sentence(text, start, edge="start")
    end = _snap_to_sentence(text, end, edge="end")
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


_SENTENCE_END = re.compile(r"[.?!](?:\s|$)")
_SENTENCE_SLOP_CHARS = 200


def _snap_to_sentence(text: str, pos: int, *, edge: str) -> int:
    """Snap a cut to the nearest sentence boundary within slop."""
    if pos <= 0 or pos >= len(text):
        return pos
    if edge == "start":
        window = text[max(0, pos - _SENTENCE_SLOP_CHARS):pos]
        matches = list(_SENTENCE_END.finditer(window))
        if matches:
            candidate = max(0, pos - _SENTENCE_SLOP_CHARS) + matches[-1].end()
            return candidate
        return pos
    window = text[pos:pos + _SENTENCE_SLOP_CHARS]
    match = _SENTENCE_END.search(window)
    if match:
        return pos + match.end()
    return pos


def _compact_tool_result(
    result: dict[str, Any],
    query: str,
    *,
    kind: str,
    content_chars: int = MAX_WEB_TOOL_CONTENT_CHARS,
) -> dict[str, Any]:
    """Compact a tool result for model visibility.

    ``content_chars`` is the excerpt budget for WEB evidence (the admin
    search_depth tier value). Vault evidence always uses
    MAX_TOOL_CONTENT_CHARS — vault retrieval is not governed by the web
    search_depth setting, so a wider web tier must never widen vault
    excerpts.
    """
    if not result.get("ok"):
        return result
    raw_evidence = result.get("evidence")
    if not isinstance(raw_evidence, list):
        return {"ok": True, "evidence": [], "count": 0}
    # The API retains complete provenance in its run evidence store for the
    # public source cards. The model only needs citation IDs, human labels,
    # and bounded excerpts; geometry and retrieval internals waste context
    # and can cause small models to reproduce the tool payload verbatim.
    # The relevance flag and citation ID survive compaction: downstream
    # filters (relevance enforcement, card membership) key on them, while
    # _synthesis_evidence still projects only title/url/content to the model.
    # PART A (BUG-003): file_id/chunk_id/revision survive compaction as the
    # authoritative source identity. The model must bind facts to file_id,
    # never to filename prose (filenames are non-unique and renameable).
    allowed = (
        ("filename", "section_path", "file_id", "chunk_id", "revision")
        if kind == "vault"
        else ("title", "url")
    )
    allowed = (*allowed, "id", "relevant")
    # Truncation preserves per-file representation (BUG-003): the
    # retrieval fairness repair appends a missing scoped file's best chunk
    # past the rank cutoff; a naive [:6] slice would strand it there every
    # time. When truncation would erase a file entirely, its first item
    # displaces the lowest-ranked kept item. Rank order is otherwise kept.
    trimmed: list[dict[str, Any]] = [
        raw for raw in raw_evidence[:MAX_TOOL_EVIDENCE] if isinstance(raw, dict)
    ]
    kept_files = set()
    for item in trimmed:
        try:
            kept_files.add(int(item.get("file_id")))
        except (TypeError, ValueError):
            continue
    for raw in raw_evidence[MAX_TOOL_EVIDENCE:]:
        if not isinstance(raw, dict) or not trimmed:
            continue
        try:
            fid = int(raw.get("file_id"))
        except (TypeError, ValueError):
            continue
        if fid in kept_files:
            continue
        trimmed[-1] = raw
        kept_files.add(fid)
    evidence: list[dict[str, Any]] = []
    truncated_excerpts = 0
    excerpt_chars = content_chars if kind == "web" else MAX_TOOL_CONTENT_CHARS
    for raw in trimmed:
        if not isinstance(raw, dict):
            continue
        excerpt = _evidence_excerpt(
            query, neutralize_lavix_markers(raw.get("content")), max_chars=excerpt_chars
        )
        item = {
            key: (
                neutralize_lavix_markers(raw[key])
                if key in ("filename", "section_path", "title", "url")
                else raw[key]
            )
            for key in allowed
            if key in raw
        }
        item.update(
            {
                "content": excerpt,
            }
        )
        if len(str(raw.get("content") or "")) > excerpt_chars:
            truncated_excerpts += 1
        evidence.append(item)
    try:
        run = current_run_scope().run_id
    except Exception:
        run = "-"
    raw_n = sum(1 for raw in raw_evidence if isinstance(raw, dict))
    if raw_n != len(evidence) or truncated_excerpts:
        logger.warning(
            "web_rag stage=5 run=%s query=%.80s kind=%s compact_raw=%d kept=%d excerpt_trunc=%d",
            run,
            _qlog((query or "")[:80]),
            kind,
            raw_n,
            len(evidence),
            truncated_excerpts,
        )
    return {"ok": True, "evidence": evidence, "count": len(evidence)}


def strip_tool_routing_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Remove routing-only fields before model-visible tool output.

    Citation IDs and relevance flags must survive in the stored gateway
    result (downstream relevance enforcement and card membership key on
    them), but the model-visible tool payload keeps its lean historical
    shape so small models are not tempted to reproduce citation markers.
    """

    if not isinstance(result.get("evidence"), list):
        return result
    return {
        **result,
        "evidence": [
            {key: value for key, value in item.items() if key not in ("id", "relevant")}
            if isinstance(item, dict)
            else item
            for item in result["evidence"]
        ],
    }


def _compact_graph_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep relationship memory bounded and separate from evidence."""

    if not result.get("ok"):
        return result
    raw_memories = result.get("memories")
    if not isinstance(raw_memories, list):
        return {"ok": True, "memories": [], "count": 0}
    memories: list[dict[str, Any]] = []
    used_chars = 0
    for raw in raw_memories[:MAX_GRAPH_RESULTS]:
        if not isinstance(raw, dict):
            continue
        item = {
            key: (
                neutralize_lavix_markers(raw[key])
                if key in ("subject", "predicate", "object_value")
                else raw[key]
            )
            for key in ("kind", "subject", "predicate", "object_value")
            if key in raw
        }
        size = sum(
            len(str(item.get(key) or ""))
            for key in ("subject", "predicate", "object_value")
        )
        if memories and used_chars + size > MAX_TOOL_CONTENT_CHARS:
            break
        memories.append(item)
        used_chars += size
    return {"ok": True, "memories": memories, "count": len(memories)}


class ToolGatewayClient:
    """Calls the main API without possessing database or storage credentials."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 20,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        scope = current_run_scope()
        headers = {
            "X-Lavix-Capability": scope.capability_token,
            "X-Lavix-Run-ID": scope.run_id,
        }
        try:
            response = await self._client.post(f"{self._base_url}/{tool_name}", json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("tool gateway returned a non-object response")
            return body
        except httpx.HTTPStatusError as exc:
            code = (
                "capability_denied"
                if exc.response.status_code in {401, 403}
                else "tool_request_rejected"
                if 400 <= exc.response.status_code < 500
                else "tool_gateway_unavailable"
            )
            logger.warning(
                "Tool gateway rejected %s for run %s with status %s",
                tool_name,
                scope.run_id,
                exc.response.status_code,
            )
            return {
                "ok": False,
                "error": {"code": code, "message": "Tool request could not be completed"},
            }
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "Tool gateway %s failed for run %s: %s",
                tool_name,
                scope.run_id,
                type(exc).__name__,
            )
            return {
                "ok": False,
                "error": {
                    "code": "tool_gateway_unavailable",
                    "message": "Tool gateway is temporarily unavailable",
                },
            }

    async def search_vault(
        self,
        query: str,
        file_ids: list[int] | None = None,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """Search only within the capability's server-defined vault scope."""

        scope = current_run_scope()
        await scope.emit({"type": "status", "step": "searching_vault"})

        model_ids = _dedupe_positive(file_ids)
        if scope.requested_file_ids is None:
            effective_ids = model_ids
        else:
            # The signed, server-selected scope is authoritative. Ignore any
            # model-proposed narrowing so a hallucinated ID cannot erase valid
            # selected-file evidence (and can never widen the scope).
            effective_ids = list(scope.requested_file_ids)

        result = await self._post(
            "search-vault",
            {
                "query": query.strip()[:MAX_QUERY_CHARS],
                "file_ids": effective_ids,
                "top_k": _bounded(top_k, minimum=1, maximum=MAX_VAULT_RESULTS),
                "deep_search": scope.deep_search,
            },
        )
        await scope.emit({"type": "status", "step": "reranking"})
        return _compact_tool_result(result, query, kind="vault")

    async def search_web(
        self,
        query: str,
        max_results: int = 3,
        *,
        excerpt_chars: int = MAX_WEB_TOOL_CONTENT_CHARS,
    ) -> dict[str, Any]:
        """Search through the gateway only when the request capability permits it.

        ``max_results``/``excerpt_chars`` come from the admin search_depth
        tier (conservative default when a caller omits them).
        """

        scope = current_run_scope()
        if not scope.web_search_enabled:
            return {
                "ok": False,
                "error": {
                    "code": "web_search_disabled",
                    "message": "Web search is disabled for this request",
                },
            }
        await scope.emit({"type": "status", "step": "web_search"})
        result = await self._post(
            "search-web",
            {
                "query": query.strip()[:MAX_QUERY_CHARS],
                "max_results": _bounded(max_results, minimum=1, maximum=MAX_WEB_RESULTS),
            },
        )
        return _compact_tool_result(
            result, query, kind="web", content_chars=excerpt_chars
        )

    async def recall_graph(
        self,
        query: str,
        max_results: int = 6,
    ) -> dict[str, Any]:
        """Recall only the capability subject's active relationship memory."""

        result = await self._post(
            "recall-graph",
            {
                "query": query.strip()[:MAX_QUERY_CHARS],
                "max_results": _bounded(
                    max_results,
                    minimum=1,
                    maximum=MAX_GRAPH_RESULTS,
                ),
            },
        )
        return _compact_graph_result(result)
