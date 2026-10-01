"""Runtime construction for the memory service.

PostgreSQL is the sole authority for consent, lifecycle, and recall.
An optional Neo4j projection (GRAPH_MEMORY_NEO4J_ENABLED) adds
graph-traversed recall context only; when it is absent or down, memory
keeps working via PostgreSQL alone.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from .neo4j_projection import Neo4jProjection, Neo4jUnavailable
from .service import GraphMemoryService, default_memory_embedder

logger = logging.getLogger(__name__)

_BUILT_SERVICES: list[GraphMemoryService] = []


def _read_secret_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _password_from_auth_value(raw: str) -> str:
    """Split the "neo4j/<password>" flat; tolerate a bare password."""
    text = str(raw or "").strip()
    if "/" in text:
        return text.split("/", 1)[1].strip()
    return text


def build_neo4j_projection() -> Neo4jProjection | None:
    """Build the optional projection; None means PostgreSQL-only mode.

    Settings resolve env -> config.yml -> default, so the default
    docker-compose.yaml flow works through NEO4J_PASSWORD while isolated
    stacks keep working through env. Flag default OFF. Neo4j down, missing
    secret, or constraint failure all resolve to None — memory keeps
    working via Postgres.
    """
    from app.config import settings

    if not settings.graph_memory_neo4j_enabled:
        return None
    password = _password_from_auth_value(
        settings.graph_memory_neo4j_password
        or _read_secret_file(settings.graph_memory_neo4j_password_file)
    )
    if not password:
        logger.warning("Neo4j projection disabled: no password file")
        return None
    projection = Neo4jProjection(
        uri=settings.graph_memory_neo4j_uri,
        user=settings.graph_memory_neo4j_user,
        password=password,
        database=settings.graph_memory_neo4j_database,
        timeout_seconds=settings.graph_memory_neo4j_timeout_seconds,
    )
    # Neo4j is often still booting when dependents start; retry briefly
    # before falling back to PostgreSQL-only mode (the worker maintenance
    # loop re-attempts later, so a slow database never disables the
    # projection permanently).
    for attempt in range(3):
        try:
            projection.ensure_constraints()
            return projection
        except Neo4jUnavailable as exc:
            logger.warning(
                "Neo4j constraints attempt %s failed: %s", attempt + 1, exc
            )
            if attempt < 2:
                import time as _time

                _time.sleep(5)
    logger.warning("Neo4j projection disabled: constraints unreachable")
    projection.close()
    return None


@lru_cache(maxsize=1)
def graph_memory_runtime_service() -> GraphMemoryService:
    """Build one process-local PostgreSQL-backed service."""

    service = GraphMemoryService(
        projection=build_neo4j_projection(), embed_fn=default_memory_embedder
    )
    _BUILT_SERVICES.append(service)
    return service


def recheck_neo4j_projection(service: GraphMemoryService) -> None:
    """Re-attempt the projection build when running PostgreSQL-only.

    Called from worker maintenance: a database that was down at boot (or
    restarted later) is picked up without a process restart. No-op unless
    the flag is on and no projection is attached.
    """
    if service.projection is not None:
        return
    from app.config import settings

    if not settings.graph_memory_neo4j_enabled:
        return
    rebuilt = build_neo4j_projection()
    if rebuilt is not None:
        logger.warning("Neo4j projection attached late")
        service.projection = rebuilt


def initialize_graph_projection() -> GraphMemoryService:
    """Return the process-local service (kept for worker startup compat)."""

    return graph_memory_runtime_service()


def close_graph_projection() -> None:
    """Forget the cached process-local service, closing any driver."""

    while _BUILT_SERVICES:
        service = _BUILT_SERVICES.pop()
        try:
            if service.projection is not None:
                service.projection.close()
        except Exception:
            logger.debug("Neo4j projection close failed", exc_info=True)
    graph_memory_runtime_service.cache_clear()


__all__ = [
    "build_neo4j_projection",
    "close_graph_projection",
    "graph_memory_runtime_service",
    "initialize_graph_projection",
    "recheck_neo4j_projection",
]
