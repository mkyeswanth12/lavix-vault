"""Private capability-authenticated tool gateway consumed by CUGA."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_runtime.events import qlog as _qlog
from app.config import settings
from app.security.redact import redact_credentials
from app.services.web_search import search_web

from .capability import CapabilityError, CapabilityScope, CapabilitySigner
from .citation_verify import domain_drop_reason, domain_sanity_check
from .evidence import RunEvidenceStore, run_evidence_store
from .retrieval import InfinityReranker, VaultRetriever

logger = logging.getLogger(__name__)


class VaultSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1_000)
    file_ids: list[int] | None = Field(default=None, max_length=50)
    top_k: int = Field(default=20, ge=1, le=500)
    deep_search: bool = False

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class WebSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1_000)
    # Ceiling 12 = the admin search_depth "pro" tier's per-leg budget.
    max_results: int = Field(default=3, ge=1, le=12)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class GraphRecallRequest(BaseModel):
    """The model may choose only a query and a small result bound."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1_000)
    max_results: int = Field(default=6, ge=1, le=8)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


CapabilityHeader = Annotated[str, Header(alias="X-Lavix-Capability", min_length=16, max_length=8_192)]
RunHeader = Annotated[str, Header(alias="X-Lavix-Run-ID", min_length=36, max_length=36)]
WebSearch = Callable[[str, int | None], Awaitable[list[dict[str, Any]]]]
GraphRecall = Callable[[int, str, int], Awaitable[Sequence[dict[str, Any]]]]


async def _recall_graph_memory(
    user_id: int,
    query: str,
    max_results: int,
) -> Sequence[dict[str, Any]]:
    """Recall through the API-owned service; CUGA never receives a graph driver."""

    from app.graph_memory.runtime import graph_memory_runtime_service

    service = graph_memory_runtime_service()
    records = await asyncio.to_thread(service.recall, user_id, query, limit=max_results)
    return [
        {
            "kind": record.kind.value,
            "subject": record.subject,
            "predicate": record.predicate,
            "object_value": record.object_value,
            "confidence": record.confidence,
        }
        for record in records
    ]


def _scope(signer: CapabilitySigner, token: str, run_id: str) -> CapabilityScope:
    try:
        return signer.verify(token, expected_run_id=run_id)
    except CapabilityError as exc:
        # Reason code only (expired/signature/mismatch/malformed); never the token.
        logger.warning("agent capability denied: %s", str(exc))
        raise HTTPException(status_code=403, detail="Agent capability denied") from None
    except ValueError:
        logger.warning("agent capability denied: malformed run id")
        raise HTTPException(status_code=403, detail="Agent capability denied") from None


def _effective_file_ids(
    allowed: tuple[int, ...] | None,
    requested: list[int] | None,
) -> list[int] | None:
    requested_ids = None if requested is None else list(dict.fromkeys(requested))
    if requested_ids is not None and any(value <= 0 for value in requested_ids):
        raise HTTPException(status_code=422, detail="Invalid file selection")
    if allowed is None:
        return requested_ids
    allowed_set = set(allowed)
    if requested_ids is None:
        return list(allowed)
    return [value for value in requested_ids if value in allowed_set]


def _plural_stem(word: str) -> str:
    """Strip a trailing plural 's' only for true plural shapes.

    The naive rule (len>4, ends in s) stemmed interrogatives like
    "whats"->"what", donating half the gate overlap for free. Require a
    longer consonant-final stem ("stocks"->"stock"); vowel stems
    ("price") and short words ("whats"/"thats") match literally only.
    """

    folded = word.casefold()
    # consonant + "ies" plurals ("countries"->"country"); short words
    # ("dies", "ties") and vowel stems stay literal.
    if len(folded) > 5 and folded.endswith("ies") and folded[-4] not in "aeiou":
        return folded[:-3] + "y"
    if len(folded) > 5 and folded.endswith("s") and folded[-2] not in "aeiou":
        return folded[:-1]
    return folded


def _distinct_substantive_matches(
    query_words: list[str], hay: str, *, stopwords: set[str] | frozenset[str] = frozenset()
) -> int:
    """Count distinct query words present in a title+content haystack.

    Singular/plural variants of one stem ("stock"/"stocks") count once, so
    a single repeated word can never satisfy the relevance gate by itself
    (e.g. a duplicated entity string matching one word twice). Both sides
    are token-matched (no substring hits: "whats" must not match
    "WhatsApp"), with symmetric singular/plural fallback so "iphone"
    still finds "iphones" and vice versa. Stemmed stopwords ("whats" is
    already one) contribute nothing.
    """

    hay_tokens = set(re.findall(r"[\w']+", hay.casefold()))
    matched: set[str] = set()
    for word in query_words:
        folded = word.casefold()
        key = _plural_stem(folded)
        if key in stopwords:
            continue
        candidates = {folded, key}
        if not folded.endswith("s"):
            candidates.add(folded + "s")
            # reverse ies-direction ("country" query vs "countries" haystack)
            if len(key) > 2 and key.endswith("y") and key[-2] not in "aeiou":
                candidates.add(key[:-1] + "ies")
        if any(candidate in hay_tokens for candidate in candidates):
            matched.add(key)
    return len(matched)


# URL substrings marking wire-service / press-release distribution.
_PROMO_URL_PARTS = (
    "prnewswire",
    "globenewswire",
    "businesswire",
    "newswire",
    "/press-release",
    "/press-releases",
    "/pressreleases",
    "/newsroom/",
)
# Title phrases marking vendor partnership announcements. Kept tight and
# literal: analytical pieces merely mentioning partnerships rarely use
# these exact announcement framings.
_PROMO_TITLE_PHRASES = (
    "announces partnership",
    "announce partnership",
    "partners with",
    "partnership with",
    "collaborates with",
    "proud to announce",
)


def _is_promotional(title: str, url: str) -> bool:
    """Flag press-release/marketing-pattern content for deprioritization.

    Deprioritize only, never exclude: a flagged item still surfaces when
    nothing better exists in the same result set.
    """

    if any(part in (url or "").casefold() for part in _PROMO_URL_PARTS):
        return True
    hay_title = (title or "").casefold()
    return any(phrase in hay_title for phrase in _PROMO_TITLE_PHRASES)


def _web_result_sort_key(item: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Order web results: relevant first, non-promotional first, specific first.

    Stable sort preserves SearXNG order within each group. Items without
    flags (legacy shapes) sort as relevant, non-promotional, and specific.
    """

    return (
        not item.get("relevant", True),
        bool(item.get("promotional", False)),
        bool(item.get("generic", False)),
    )


# Navigation/footer chrome phrases: when two or more appear in one snippet,
# the "content" is site chrome rather than subject-specific prose (e.g. a
# Wikipedia nav block matched only via a boilerplate word). Deliberately
# chrome-only: topical words are never listed here, so genuinely topical
# snippets cannot match no matter their shape.
_BOILERPLATE_PHRASES = (
    "jump to content",
    "main menu",
    "terms of use",
    "privacy policy",
    "all rights reserved",
    "move to sidebar",
    "toggle the table",
    "skip to main content",
    "cookie policy",
    "sign in",
    "current events",
)

# Dictionary-definition shape: title names the reference work and the
# snippet defines a common word rather than answering anything topical.
_DEFINITION_TITLE = re.compile(r"defin|meaning|dictionary", re.IGNORECASE)
_DEFINITION_BODY = re.compile(
    r"the meaning of\b.{1,60}\bis\b|\bdefinition\s*:",
    re.IGNORECASE,
)


def _is_generic_content(title: str, content: str) -> bool:
    """Flag boilerplate or dictionary-definition snippets for deprioritization.

    Deprioritize only, never exclude: a flagged item still surfaces when
    nothing better exists in the same result set. Rank-only by design —
    the relevance gate above remains the sole exclusion authority besides
    the domain filter.
    """

    hay = f"{title or ''}\n{content or ''}".casefold()
    if _DEFINITION_TITLE.search(title or "") and _DEFINITION_BODY.search(hay):
        return True
    hits = sum(1 for phrase in _BOILERPLATE_PHRASES if phrase in hay)
    return hits >= 2


def create_agent_gateway_router(
    *,
    signer: CapabilitySigner | None = None,
    retriever: VaultRetriever | None = None,
    evidence_store: RunEvidenceStore = run_evidence_store,
    web_search: WebSearch = search_web,
    graph_recall: GraphRecall = _recall_graph_memory,
    reranker: InfinityReranker | None = None,
) -> APIRouter:
    capability_signer = signer or CapabilitySigner(
        settings.agent_capability_secret,
        ttl_seconds=settings.agent_capability_ttl_seconds,
    )
    vault_retriever = retriever or VaultRetriever()
    # Dedicated short timeout: SearXNG already spends ~8s of the 20s tool
    # budget. Fail-open (SearXNG order) on any Infinity outage.
    web_reranker = reranker or InfinityReranker(timeout_seconds=2.5)
    router = APIRouter()

    @router.post("/search-vault")
    async def search_vault_tool(
        body: VaultSearchRequest,
        capability: CapabilityHeader,
        run_id: RunHeader,
    ) -> dict[str, Any]:
        scope = _scope(capability_signer, capability, run_id)
        file_ids = _effective_file_ids(scope.file_ids, body.file_ids)
        if file_ids == []:
            return {"ok": True, "evidence": [], "count": 0}
        try:
            found = await vault_retriever.search(
                user_id=scope.user_id,
                query=body.query,
                file_ids=file_ids,
                top_k=body.top_k,
                deep_search=scope.deep_search and body.deep_search,
            )
        except Exception:
            logger.warning("Vault retrieval failed for run %s", run_id, exc_info=True)
            raise HTTPException(status_code=503, detail="Vault retrieval unavailable") from None
        evidence = await evidence_store.record(run_id, "vault", found)
        return {"ok": True, "evidence": evidence, "count": len(evidence), "untrusted": True}

    def _diverse_engine_pick(
        items: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        """Round-robin across search engines up to limit, order-stable.

        The merged fan-out sorts one engine first, so a flat [:limit]
        slice can crowd out every other engine's results (bing filler
        burying bing-news gems). Interleaving keeps each engine's best
        items reachable for rerank/verify; relative order within an
        engine is preserved. Items without attribution ride along under
        "unknown" instead of being dropped.
        """
        try:
            count = max(0, int(limit))
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            return []
        buckets: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            engine = str(item.get("engine") or "unknown")
            if engine not in buckets:
                buckets[engine] = []
                order.append(engine)
            buckets[engine].append(item)
        picked: list[dict[str, Any]] = []
        while len(picked) < count and any(buckets[engine] for engine in order):
            for engine in order:
                if len(picked) >= count:
                    break
                if buckets[engine]:
                    picked.append(buckets[engine].pop(0))
        return picked

    @router.post("/search-web")
    async def search_web_tool(
        body: WebSearchRequest,
        capability: CapabilityHeader,
        run_id: RunHeader,
    ) -> dict[str, Any]:
        scope = _scope(capability_signer, capability, run_id)
        if not scope.web_search_enabled:
            raise HTTPException(status_code=403, detail="Web search is not permitted for this run")
        # A secret pasted into a question must never flow out to search
        # vendors. Scrubbing also improves retrieval (valueless noise
        # tokens never matched anything useful).
        searched_query = redact_credentials(body.query)
        if searched_query != body.query:
            logger.warning("Web query scrubbed of credentials for run %s", run_id)
        try:
            found = await web_search(searched_query, body.max_results, run_id=run_id)
        except Exception as exc:
            logger.warning("Web search failed for run %s: %s", run_id, type(exc).__name__)
            raise HTTPException(status_code=503, detail="Web search unavailable") from None
        selected = _diverse_engine_pick(found, body.max_results)
        # Cross-encoder rerank over the merged fan-out (log-only signal for
        # now: no min-score drops — the relevance gate stays the sole
        # exclusion authority). Rerank scores replace the unbounded SearXNG
        # aggregates downstream; originals are kept as search_score.
        reranked: list[dict[str, Any]] = []
        rerank_outcome = "skip"
        if selected:
            try:
                reranked = await web_reranker.rerank(body.query, selected, top_k=len(selected))
            except Exception:
                reranked = []
            if reranked and any(isinstance(_r.get("rerank_score"), (int, float)) for _r in reranked):
                rerank_outcome = "ok"
                reordered: list[dict[str, Any]] = []
                for _r in reranked:
                    _item = dict(_r)
                    try:
                        _score = float(_item.get("rerank_score") or 0.0)
                    except (TypeError, ValueError):
                        _score = 0.0
                    _item["search_score"] = _item.get("score")
                    _item["score"] = _score if math.isfinite(_score) else 0.0
                    reordered.append(_item)
                selected = reordered
            else:
                rerank_outcome = "unavailable"
                selected = list(selected)
        logger.warning(
            "web_rag stage=2 run=%s query=%.80s rerank=%s kept=%d",
            run_id,
            _qlog((body.query or "")[:80]),
            rerank_outcome,
            len(selected),
        )
        scores = []
        for item in selected:
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            scores.append(max(0.0, score) if math.isfinite(score) else 0.0)
        best_score = max(scores, default=0.0)

        # Relevance gate inputs: the query's meaningful words. A result whose
        # title+snippet shares too few of them is recorded as irrelevant — the
        # model still sees it, but no source card is rendered for it.
        _STOP = {
            "the", "and", "for", "with", "what", "when", "how", "why", "who",
            "which", "that", "this", "those", "these", "about", "from",
            "into", "over", "after", "before", "are", "was", "were", "has",
            "have", "had", "you", "your", "our", "their", "its", "can",
            "could", "should", "would", "will", "did", "does", "do",
        }
        query_words = [
            word.casefold()
            for word in re.findall(r"[\w'-]+", body.query or "")
            if len(word) >= 3 and word.casefold() not in _STOP
        ]

        def _match_count(item: dict[str, Any]) -> int:
            # Distinct substantive word matches (with fuzzy plural matching).
            # Callers require >= 2 to prevent car forums from passing — and to
            # stop one repeated word from satisfying the gate by itself.
            if not query_words:
                return -1  # gate open; no anchor words to count against
            hay = f"{item.get('title') or ''} {item.get('content') or ''}".casefold()
            return _distinct_substantive_matches(query_words, hay, stopwords=_STOP)

        normalized = []
        for rank, (item, score) in enumerate(zip(selected, scores, strict=True), start=1):
            # SearXNG engine scores are unbounded aggregates, not probabilities.
            # Normalize them within this result set and retain a rank fallback.
            match_percentage = (
                round(score / best_score * 100)
                if best_score > 0
                else max(1, round((len(selected) - rank + 1) / max(1, len(selected)) * 100))
            )
            title = str(item.get("title") or "")
            url = str(item.get("url") or "")
            # Web results only (this is the /search-web tool endpoint; vault
            # chunks never pass through here). Raised 4000 -> 6000 so the
            # search_depth "pro" tier's 6000-char excerpt budget can actually
            # materialize; lower tiers still excerpt at their own smaller
            # budgets downstream.
            content = str(item.get("content") or "")[:6_000]
            match_count = _match_count(item)
            # Single-entity queries (one substantive word) can never score 2,
            # so require all of their words instead of an absolute floor.
            # Multi-word queries keep the >= 2 floor so one repeated word
            # cannot satisfy the gate by itself.
            required_matches = min(2, len(query_words))
            is_relevant = True if not query_words else match_count >= required_matches
            if query_words:
                logger.warning(
                    "web_rag stage=4 run=%s query=%.80s matches=%d relevant=%s promo=%s generic=%s url=%.50s",
                    run_id,
                    _qlog((body.query or "")[:80]),
                    match_count,
                    is_relevant,
                    _is_promotional(title, url),
                    _is_generic_content(title, content),
                    url[:50],
                )
            normalized.append(
                {
                    "kind": "web",
                    "filename": str(item.get("title") or item.get("url") or "Web result"),
                    "mime_type": "web",
                    "title": title,
                    "url": url,
                    "content": content,
                    "score": score,
                    "match_percentage": match_percentage,
                    **(
                        {"search_score": item.get("search_score")}
                        if item.get("search_score") is not None
                        else {}
                    ),
                    "relevant": is_relevant,
                    "promotional": _is_promotional(title, url),
                    "generic": _is_generic_content(title, content),
                }
            )
        # Pre-injection domain sanity filter: drop results from clearly
        # mismatched domains (job portals, e-commerce, exam-result sites)
        # before the model sees them.  This prevents the model from citing
        # topically irrelevant sources.
        if getattr(settings, "citation_relevance_enabled", True):
            kept: list[dict[str, Any]] = []
            for item in normalized:
                url = str(item.get("url", ""))
                if domain_sanity_check(url, body.query):
                    kept.append(item)
                else:
                    logger.warning(
                        "web_rag stage=3 run=%s query=%.80s drop=domain rule=%s url=%.50s",
                        run_id,
                        _qlog((body.query or "")[:80]),
                        domain_drop_reason(url),
                        url[:50],
                    )
            normalized = kept
        # Deprioritize promotional content behind relevant, non-promotional
        # results. Stable sort preserves SearXNG order within each group and
        # drops nothing: exclusion stays with the relevance gate and the
        # domain filter above.
        normalized.sort(key=_web_result_sort_key)
        evidence = await evidence_store.record(run_id, "web", normalized)
        return {"ok": True, "evidence": evidence, "count": len(evidence), "untrusted": True}

    @router.post("/recall-graph")
    async def recall_graph_tool(
        body: GraphRecallRequest,
        capability: CapabilityHeader,
        run_id: RunHeader,
    ) -> dict[str, Any]:
        scope = _scope(capability_signer, capability, run_id)
        try:
            recalled = await graph_recall(scope.user_id, body.query, body.max_results)
        except Exception as exc:
            # Relationship memory is optional personalization. Its private
            # operational failure must not leak graph details or break chat.
            # Return the same empty, non-evidence contract as disabled memory
            # instead of turning an optional projection outage into a failed
            # agent tool call.
            logger.warning("Graph recall failed for run %s: %s", run_id, type(exc).__name__)
            return {
                "ok": True,
                "memories": [],
                "count": 0,
                "untrusted": True,
            }

        memories: list[dict[str, Any]] = []
        used_chars = 0
        for raw in recalled[: body.max_results]:
            if not isinstance(raw, dict):
                continue
            item = {
                key: raw[key]
                for key in ("kind", "subject", "predicate", "object_value", "confidence")
                if key in raw
            }
            size = sum(len(str(item.get(key) or "")) for key in ("subject", "predicate", "object_value"))
            if memories and used_chars + size > 1_600:
                break
            memories.append(item)
            used_chars += size
        # Deliberately not recorded in RunEvidenceStore: memory can personalize
        # synthesis, but can never become a public source or citation.
        return {"ok": True, "memories": memories, "count": len(memories), "untrusted": True}

    return router


router = create_agent_gateway_router()
