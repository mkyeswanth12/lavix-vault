"""Semantic citation relevance verification and domain sanity checking.

Prevents the model from citing topically-mismatched sources by verifying
that each cited source actually supports the generated claim.  Uses the
same Ollama embedding endpoint (snowflake-arctic-embed2:cpu) as the
ingestion pipeline for consistency.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cosine similarity (pure Python, no external deps)
# ---------------------------------------------------------------------------

def cosine_similarity(a: tuple[float, ...] | list[float], b: tuple[float, ...] | list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Returns a value in [-1.0, 1.0].  For normalised embeddings the result
    is in [0.0, 1.0].
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Embedding client (calls Ollama directly — Option B)
# ---------------------------------------------------------------------------

async def _embed_texts(
    texts: list[str],
    *,
    embedding_url: str,
    embedding_model: str,
    timeout_seconds: float = 30.0,
) -> list[tuple[float, ...]]:
    """Embed one or more texts via the Ollama /v1/embeddings endpoint."""
    if not texts:
        return []
    payload = {"model": embedding_model, "input": texts}
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(
                embedding_url,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.warning("Citation verification embedding failed: %s", type(exc).__name__)
        return []

    # Parse response — handle both OpenAI-compatible and Ollama-native formats
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        rows = data["data"]
    elif isinstance(data, dict) and isinstance(data.get("embeddings"), list):
        return [tuple(v) for v in data["embeddings"]]

    vectors: list[tuple[float, ...]] = []
    for row in rows:
        emb = row.get("embedding") if isinstance(row, dict) else None
        if isinstance(emb, list):
            vectors.append(tuple(float(x) for x in emb))
    return vectors


# ---------------------------------------------------------------------------
# Domain sanity checking
# ---------------------------------------------------------------------------

# Domains that never contain article content — social share widgets,
# URL shorteners, utility endpoints, and exam/job portals.
_DOMAIN_BLOCKLIST: set[str] = {
    # Social share widget domains
    "wa.me",
    "web.whatsapp.com",
    "whatsapp.com",
    "t.me",
    "telegram.me",
    # URL shorteners (content is behind redirect, not scrapable)
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "bl.ink",
    "lnkd.in",
    "db.tt",
    "qr.ae",
    "rb.gy",
    "cutt.ly",
    "shorturl.at",
    "tiny.cc",
    # Social/commerce utility endpoints
    "pinterest.com",       # pin-it widget, not article content
    # Indian exam/job portals (topically mismatched for general queries)
    "sarkariresult.com",
    "sarkari-exam.com",
    "governmentexams.co.in",
    "governmentexams.net",
    "fresherslive.com",
    "freshersnow.com",
    "recruitmentforall.in",
    "jobrasta.com",
    "sarkari naukri.com",
    "indiagovtexam.com",
    "examresult.net",
    "resultservals.com",
    "myaipapers.com",
}

# URL substrings that signal non-informational content
_SUSPICIOUS_URL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"/(recruitment|admit-card|answer-key|exam-results)", re.IGNORECASE),
    # Social share intent URLs
    re.compile(r"/(intent/tweet|sharer/share|share\?|send/?to/?device|addthis|sharethis)", re.IGNORECASE),
    re.compile(r"/(status/\d+/intent|messages/compose|share/v\d|refer/?a/?friend)", re.IGNORECASE),
    # Social submit/sharing widgets
    re.compile(r"/submit\b", re.IGNORECASE),
    re.compile(r"linkedin\.com/sharing/", re.IGNORECASE),
]

# Commerce paths (buy/cart/price/...) signal coupon/affiliate junk on
# general queries — but they are the whole point of manufacturer-direct
# pages on price questions, so allowlisted official stores skip them.
_COMMERCE_URL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"/(buy|cart|checkout|shop|product|price|discount|coupon)", re.IGNORECASE),
]

# Manufacturer-direct hosts: commerce paths are legitimate content here
# (e.g. apple.com/shop/buy-iphone on a price question). Subdomains match
# too (store.apple.com). The domain blocklist and non-commerce patterns
# still apply to these hosts.
_COMMERCE_PATH_ALLOWLIST: frozenset[str] = frozenset({
    "apple.com",
    "samsung.com",
    "motorola.com",
    "oneplus.com",
    "xiaomi.com",
    "mi.com",
    "vivo.com",
    "oppo.com",
    "realme.com",
    "store.google.com",
})


def _host_allowlisted_commerce(host: str) -> bool:
    """True when a hostname is (or is under) an allowlisted store domain."""
    return any(host == base or host.endswith("." + base) for base in _COMMERCE_PATH_ALLOWLIST)


def is_utility_url(url: str) -> bool:
    """Return True if the URL is a share widget, shortener, or non-content endpoint.

    This catches URL patterns that the domain blocklist alone cannot handle,
    such as `twitter.com/intent/tweet?url=...` or `facebook.com/sharer/sharer.php?u=...`.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().removeprefix("www.")
    except (ValueError, TypeError):
        return False

    # Domain-level check
    if host in _DOMAIN_BLOCKLIST:
        return True

    # Path-level check for share intent URLs (applies everywhere)
    for pattern in _SUSPICIOUS_URL_PATTERNS:
        if pattern.search(url):
            return True

    # Commerce paths are legitimate on allowlisted official stores.
    if _host_allowlisted_commerce(host):
        return False
    for pattern in _COMMERCE_URL_PATTERNS:
        if pattern.search(url):
            return True

    return False


def domain_sanity_check(url: str, query: str) -> bool:
    """Return True if the domain is plausible for the query topic.

    Flags social share widgets, URL shorteners, exam-result sites, and
    other utility/non-content domains.
    """
    if not url:
        return True
    if is_utility_url(url):
        return False
    return True


def domain_drop_reason(url: str) -> str:
    """Classify a domain-sanity drop for stage-3 observability."""
    if not url:
        return "utility"
    try:
        host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    except (ValueError, TypeError):
        return "utility"
    if host in _DOMAIN_BLOCKLIST:
        return "blocklist"
    for pattern in (*_SUSPICIOUS_URL_PATTERNS, *_COMMERCE_URL_PATTERNS):
        try:
            if pattern.search(url):
                return "suspicious"
        except (TypeError, ValueError):
            continue
    return "utility"


# ---------------------------------------------------------------------------
# Semantic relevance verification
# ---------------------------------------------------------------------------

async def verify_citation_relevance(
    claim: str,
    source_content: str,
    *,
    embedding_url: str,
    embedding_model: str,
    threshold: float = 0.5,
    timeout_seconds: float = 30.0,
) -> float:
    """Compute semantic similarity between a claim and source content.

    Returns the cosine similarity score (0.0–1.0).  A score >= threshold
    means the source is considered relevant to the claim.
    """
    if not claim.strip() or not source_content.strip():
        return 0.0

    # Truncate to avoid token overflow (embeddings have context limits)
    claim_text = claim.strip()[:2_000]
    content_text = source_content.strip()[:3_000]

    vectors = await _embed_texts(
        [claim_text, content_text],
        embedding_url=embedding_url,
        embedding_model=embedding_model,
        timeout_seconds=timeout_seconds,
    )
    if len(vectors) < 2:
        return 0.0

    score = cosine_similarity(vectors[0], vectors[1])
    # Clamp to [0, 1] for the relevance interpretation
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Combined citation filter
# ---------------------------------------------------------------------------

async def filter_citations(
    evidence: list[dict[str, Any]],
    *,
    query: str,
    claim: str | None = None,
    threshold: float = 0.5,
    embedding_url: str,
    embedding_model: str,
    run_id: str | None = None,
    leg_queries: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Filter web evidence by domain sanity and semantic relevance.

    Phase 1 (fast, no embedding call): domain sanity check + keyword
    relevance signal. The keyword check is LOG-ONLY: a lexical mismatch
    never drops a candidate — the bounded pool always reaches Phase 2,
    where the embedding verdict decides. Domain drops still apply
    (safety control, unchanged).
    Phase 2 (one batched embedding call): semantic similarity between
    claim and each source, evaluated per research leg (each item is
    keyword-checked against its own leg query when leg_queries maps its
    leg_id; otherwise the fallback query). Semantically kept items get
    their ``relevant`` flag upgraded to True so cards, synthesis
    projection, and evidence IDs treat verify as overriding lexical.

    If claim is None, only Phase 1 runs (pre-injection filter path).
    Every candidate is logged at WARNING (sidecar root logger discards INFO)
    so near-miss score distributions are observable per run.
    """
    run = run_id or "-"
    short_query = (query or "")[:80]
    filtered: list[dict[str, Any]] = []
    # Pre-compute query words for keyword relevance (fast path)
    stop = {
        "the", "and", "for", "with", "what", "when", "how", "why", "who",
        "which", "that", "this", "those", "these", "about", "from",
        "into", "over", "after", "before", "are", "was", "were", "has",
        "have", "had", "you", "your", "our", "their", "its", "can",
        "could", "should", "would", "will", "did", "does", "do",
    }

    def _words(text: str) -> list[str]:
        return [
            w.casefold()
            for w in re.findall(r"[\w'-]+", text or "")
            if len(w) >= 3 and w.casefold() not in stop
        ]

    fallback_words = _words(query)
    leg_word_cache: dict[str, list[str]] = {}

    def _leg_words(item: dict[str, Any]) -> tuple[list[str], str]:
        leg_id = str(item.get("leg_id") or "")
        leg_query = (leg_queries or {}).get(leg_id, "")
        if leg_query and leg_id:
            if leg_query not in leg_word_cache:
                leg_word_cache[leg_query] = _words(leg_query)
            return leg_word_cache[leg_query], leg_id
        return fallback_words, ""

    for item in evidence:
        url = str(item.get("url") or "")
        content = str(item.get("content") or "")
        title = str(item.get("title") or "")

        # Phase 1: Domain sanity check (still drops — safety control)
        if not domain_sanity_check(url, query):
            logger.warning(
                "web_rag stage=6 run=%s query=%.80s drop=domain url=%.50s",
                run,
                short_query,
                url[:50],
            )
            continue

        # Phase 1: Keyword relevance is a logged signal only. A mismatch
        # must not eliminate the candidate before semantic verification.
        words, leg_id = _leg_words(item)
        keyword_ok = True
        if words:
            hay = f"{title} {content}".casefold()
            keyword_ok = any(w in hay for w in words)
            if not keyword_ok:
                if claim is None:
                    # Pre-injection path (no claim to verify against):
                    # keyword gate still applies here as before.
                    logger.warning(
                        "web_rag stage=6 run=%s query=%.80s drop=keyword url=%.50s",
                        run,
                        short_query,
                        (url or title)[:50],
                    )
                    continue
                logger.warning(
                    "web_rag stage=6 run=%s query=%.80s lexical_miss leg=%s url=%.50s",
                    run,
                    short_query,
                    leg_id or "-",
                    (url or title)[:50],
                )

        filtered.append(item)

    # Phase 2: Semantic relevance (only when we have a claim to verify)
    if claim is not None and filtered:
        texts: list[str] = []
        batch_items: list[dict[str, Any]] = []
        for item in filtered:
            content = str(item.get("content") or "")
            title = str(item.get("title") or "")
            source_text = f"{title}\n{content}".strip()
            if not source_text:
                continue
            texts.append(source_text)
            batch_items.append(item)
        verified: list[dict[str, Any]] = []
        embedding_available = False
        vectors: list[tuple[float, ...]] = []
        if batch_items:
            claim_text = claim.strip()[:2_000]
            source_texts = [text.strip()[:3_000] for text in texts]
            try:
                vectors = await _embed_texts(
                    [claim_text, *source_texts],
                    embedding_url=embedding_url,
                    embedding_model=embedding_model,
                    timeout_seconds=30.0,
                )
            except (httpx.HTTPError, ValueError, TypeError):
                vectors = []
        if vectors and len(vectors) == len(batch_items) + 1:
            claim_vector = vectors[0]
            for item, source_vector in zip(batch_items, vectors[1:], strict=True):
                score = cosine_similarity(claim_vector, source_vector)
                score = max(0.0, min(1.0, score))
                if score > 0:
                    embedding_available = True
                kept = score >= threshold
                _, leg_id = _leg_words(item)
                logger.warning(
                    "web_rag stage=6 run=%s query=%.80s score=%.3f thr=%.2f keep=%s reason=%s leg=%s url=%.50s",
                    run,
                    short_query,
                    score,
                    threshold,
                    kept,
                    "semantic" if score > 0 else "infra",
                    leg_id or "-",
                    str(item.get("url") or item.get("title"))[:50],
                )
                if kept:
                    item["relevance_score"] = round(score, 4)
                    # Semantic verify overrides the lexical flag so cards,
                    # synthesis projection, and evidence IDs treat this
                    # item as usable evidence.
                    item["relevant"] = True
                    verified.append(item)
        # If embedding service was unavailable (no usable vectors),
        # return the filtered list unchanged — don't treat infrastructure
        # failure as "irrelevant evidence".
        if not verified and not embedding_available:
            logger.warning(
                "web_rag stage=6 run=%s query=%.80s infra_fail_keep=%d",
                run,
                short_query,
                len(filtered),
            )
            return filtered
        return verified

    return filtered
