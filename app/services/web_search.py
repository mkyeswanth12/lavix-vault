"""SearXNG-backed web evidence with bounded, SSRF-aware enrichment.

Page fetching is disabled by default.  Search snippets are still useful evidence,
and disabling arbitrary outbound fetches is the safe baseline for a vault service.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import socket
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
import redis

from agent_runtime.events import qlog as _qlog
from app.config import settings

_cache = redis.from_url(
    settings.redis_url,
    decode_responses=True,
    socket_connect_timeout=0.5,
    socket_timeout=0.5,
)

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_PORTS = {80, 443}
_ALLOWED_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}
_MAX_PAGES_TO_EXTRACT = 2


class UnsafeWebTarget(ValueError):
    """Raised when a URL may reach a non-public network target."""


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_reserved,
            address.is_unspecified,
        )
    )


async def _resolve_public_addresses(hostname: str, port: int) -> tuple[str, ...]:
    if not hostname:
        raise UnsafeWebTarget("URL has no hostname")

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None

    if literal is not None:
        if not _is_public_address(str(literal)):
            raise UnsafeWebTarget("non-public IP address")
        return (str(literal),)

    loop = asyncio.get_running_loop()
    records = await loop.run_in_executor(
        None,
        lambda: socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM),
    )
    addresses = tuple(sorted({record[4][0] for record in records}))
    if not addresses or any(not _is_public_address(address) for address in addresses):
        raise UnsafeWebTarget("hostname resolves to a non-public address")
    return addresses


async def validate_public_url(url: str) -> str:
    """Validate scheme, port and every currently resolved address."""

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeWebTarget("unsupported URL scheme")
    if parsed.username or parsed.password:
        raise UnsafeWebTarget("URL credentials are not allowed")

    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise UnsafeWebTarget("invalid URL port") from exc
    if port not in _ALLOWED_PORTS:
        raise UnsafeWebTarget("URL port is not allowed")

    await _resolve_public_addresses(parsed.hostname or "", port)
    return url


# Fallback per-leg budget when a caller passes no explicit num_results.
# The only production caller (the capability gateway) always passes the
# admin search_depth tier value explicitly; this mirrors the conservative
# tier so a direct call behaves like the default deployment.
_DEFAULT_WEB_RESULTS = 5


async def _fetch_searxng(
    query: str,
    num_results: int | None = None,
    *,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    limit = max(1, min(num_results or _DEFAULT_WEB_RESULTS, 12))
    timeout = min(float(settings.web_search_timeout), float(settings.web_search_hard_timeout))
    run = run_id or "-"
    short_query = (query or "")[:80]

    # Engine names must exist and be enabled in searxng/settings.yml, and the
    # engine MODULE must exist upstream (engine: bing_news, not "bing news" —
    # spaced names fail registration and silently fall back to defaults).
    # DuckDuckGo web and Startpage are CAPTCHA-dead from this egress IP and
    # DDG ships no news module, so the fan-out uses backends verified live
    # with per-result engine attribution: bing first for result-mix
    # stability, then news / community diversity. Wikipedia is disabled in
    # settings (403-walled and silent-empty from this egress); bing still
    # surfaces wikipedia.org URLs, so reference coverage survives.
    _ENGINE_PRIORITY = {"bing": 0, "bing news": 1, "mwmbl": 2}
    engines = "bing,bing news,mwmbl"

    started = datetime.now(UTC)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            params: dict[str, Any] = {
                "q": query.strip()[:512],
                "format": "json",
                "pageno": 1,
                "engines": engines,
            }
            response = await client.get(
                f"{settings.searxng_url.rstrip('/')}/search",
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.warning(
            "web_rag stage=2 run=%s query=%s outcome=error detail=%s",
            run,
            _qlog(short_query),
            type(exc).__name__,
        )
        return []
    elapsed_ms = round((datetime.now(UTC) - started).total_seconds() * 1000)

    unresponsive = payload.get("unresponsive_engines") or []
    # Collect and validate results
    raw_results: list[dict[str, Any]] = []
    ssrf_dropped = 0
    for item in payload.get("results", [])[: max(limit * 4, limit)]:
        url = str(item.get("url") or "")
        try:
            await validate_public_url(url)
        except (UnsafeWebTarget, OSError, socket.gaierror):
            ssrf_dropped += 1
            continue
        raw_results.append(
            {
                "title": str(item.get("title") or "")[:500],
                "url": url,
                # Web results only (this module serves the web search
                # path; vault retrieval never touches it). Raised 4000 ->
                # 6000 so the search_depth "pro" tier's 6000-char excerpt
                # budget can materialize; lower tiers excerpt at their own
                # smaller budgets downstream.
                "content": str(item.get("content") or item.get("snippet") or "")[:6000],
                "score": float(item.get("score") or 0.0),
                "engine": str(item.get("engine") or ""),
                "retrieved_at": datetime.now(UTC).isoformat(),
                "untrusted": True,
            }
        )

    # Sort by engine priority (duckduckgo first), then by score
    raw_results.sort(
        key=lambda r: (_ENGINE_PRIORITY.get(r.get("engine", ""), 99), -r.get("score", 0.0))
    )

    # Per-engine outcome accounting from the aggregated response.  SearXNG
    # fans out server-side, so per-engine latency is unavailable here;
    # total latency plus per-engine counts and the unresponsive list give
    # the same operational signal without N sequential requests.
    per_engine: dict[str, int] = {}
    for item in raw_results:
        name = str(item.get("engine") or "unknown")
        per_engine[name] = per_engine.get(name, 0) + 1
    logger.warning(
        "web_rag stage=2 run=%s query=%s engines=%s unresponsive=%s latency_ms=%d results=%d ssrf_dropped=%d",
        run,
        _qlog(short_query),
        ",".join(f"{name}:{count}" for name, count in sorted(per_engine.items())) or "-",
        ",".join(str(entry[-1] if isinstance(entry, (list, tuple)) and entry else entry) for entry in unresponsive[:4]) or "-",
        elapsed_ms,
        len(raw_results),
        ssrf_dropped,
    )

    # The engine field is kept through the return path so downstream stage
    # logs can attribute results; consumers read known keys only.
    # Engine-diverse cut (not a flat [:limit] slice): the sort above groups
    # one engine first, so a flat slice buries every other engine's items
    # before rerank/verify ever see them (bing filler crowding out bing-news
    # gems). Round-robin keeps each engine's best reachable; order within
    # an engine is preserved. (A twin helper lives in app/agent/gateway.py
    # as defense-in-depth; this one cannot be shared — gateway imports
    # this module, so the reverse import would cycle.)
    results = _diverse_engine_pick(raw_results, limit)
    return results


def _diverse_engine_pick(
    items: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    """Round-robin across search engines up to limit, order-stable."""
    try:
        count = max(0, int(limit if limit is not None else 0))
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


async def _extract_page(url: str) -> str | None:
    """Fetch one validated page with redirect and byte limits.

    DNS is revalidated for every hop.  This feature remains disabled by default;
    deployments enabling it should also enforce an outbound network policy.
    """

    current = url
    timeout = httpx.Timeout(connect=3.0, read=4.0, write=3.0, pool=3.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for _ in range(settings.web_page_max_redirects + 1):
            await validate_public_url(current)
            async with client.stream(
                "GET",
                current,
                headers={"User-Agent": "LavixVault/0.3 (+local RAG fetcher)"},
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        return None
                    current = urljoin(current, location)
                    continue

                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if content_type not in _ALLOWED_CONTENT_TYPES:
                    return None
                declared = response.headers.get("content-length")
                if declared and int(declared) > settings.web_page_max_bytes:
                    return None

                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > settings.web_page_max_bytes:
                        return None

            text = body.decode(response.encoding or "utf-8", errors="replace")
            if content_type == "text/plain":
                return text.strip()[:12_000] or None
            try:
                import trafilatura

                extracted = trafilatura.extract(
                    text,
                    include_comments=False,
                    include_links=False,
                    include_tables=True,
                    favor_precision=True,
                )
            except (ImportError, ValueError):
                return None
            return extracted.strip()[:12_000] if extracted and extracted.strip() else None
    return None


async def _enrich_with_page_content(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not settings.web_page_fetch_enabled or not results:
        return results

    candidates = results[:_MAX_PAGES_TO_EXTRACT]
    fetched = await asyncio.gather(
        *(_extract_page(item["url"]) for item in candidates),
        return_exceptions=True,
    )
    for item, content in zip(candidates, fetched, strict=True):
        if isinstance(content, str) and len(content) > len(item.get("content", "")):
            item["content"] = content[:3000]
            item["content_extracted"] = True
    return results


async def search_web(
    query: str,
    num_results: int | None = None,
    *,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return bounded, explicitly untrusted web evidence."""

    normalized = " ".join((query or "").split())
    if not normalized:
        return []

    cache_key = (
        "web:"
        + hashlib.sha256(
            f"{normalized.lower()}|{num_results or _DEFAULT_WEB_RESULTS}".encode()
        ).hexdigest()
    )
    try:
        cached = await asyncio.to_thread(_cache.get, cache_key)
        if cached:
            parsed = json.loads(cached)
            logger.warning(
                "web_rag stage=2 run=%s query=%s cache=hit results=%d",
                run_id or "-",
                _qlog(normalized[:80]),
                len(parsed) if isinstance(parsed, list) else 0,
            )
            logger.warning(
                "web_rag stage=5 run=%s query=%s cache=hit raw=%d kept=%d trunc=none",
                run_id or "-",
                _qlog(normalized[:80]),
                len(parsed) if isinstance(parsed, list) else 0,
                len(parsed) if isinstance(parsed, list) else 0,
            )
            return parsed
    except Exception:
        cached = None

    logger.warning(
        "web_rag stage=5 run=%s query=%s cache=miss",
        run_id or "-",
        _qlog(normalized[:80]),
    )

    results = await _enrich_with_page_content(
        await _fetch_searxng(normalized, num_results, run_id=run_id)
    )
    if results:
        try:
            await asyncio.to_thread(
                _cache.setex,
                cache_key,
                settings.web_search_cache_ttl_seconds,
                json.dumps(results, separators=(",", ":")),
            )
        except Exception:
            pass
    return results
