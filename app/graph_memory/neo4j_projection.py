"""Neo4j projection of relationship memory: rebuildable, PG-canonical.

PostgreSQL (graph_memory_*) owns consent, lifecycle, expiry, and provenance.
This module is a thin, parameterized projection only:

- Nodes: (:User {user_id})-[:HAS_MEMORY]->(:Memory) plus
  (:Memory)-[:ABOUT]->(:Entity) for entity/relationship items.
- Idempotent MERGE keyed by a composite ``key`` (``user_id:fingerprint``);
  uniqueness constraints on User.user_id and Memory.key (+ Entity.key).
- EVERY query carries the tenant user id; Cypher is parameterized only.
- Only ACTIVE items are projected (pending/rejected/expired/secret-bearing
  inputs raise Neo4jProjectionRefused before any driver call).
- Timeouts are enforced with a future, not driver options: any stall or
  connection error surfaces as Neo4jUnavailable so callers fail open to
  PostgreSQL recall and the worker retries with backoff.

The driver is imported lazily so flag-OFF processes never require the
``neo4j`` package; tests inject a recording fake via ``driver=``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.security.redact import contains_credentials

from .validation import _contains_sensitive_data_any

logger = logging.getLogger(__name__)

#: Item kinds that sprout (:Entity) nodes + ABOUT edges.
_ENTITY_KINDS = frozenset({"entity", "relationship"})

#: Extra graph facts a single recall may absorb (token-budget cap).
GRAPH_RECALL_MAX_FACTS = 4

#: Question-topic TTL (Phase 5): topics are ephemeral recall aids.
QUESTION_TOPIC_TTL_DAYS = 30
QUESTION_TOPIC_MAX = 5


class Neo4jUnavailable(RuntimeError):
    """Neo4j is down, slow, or erroring: callers must fail open to Postgres."""


class Neo4jProjectionRefused(ValueError):
    """Boundary refusal: non-projectable input never reaches the driver."""


@dataclass(frozen=True, slots=True)
class ProjectableItem:
    user_id: int
    tenant_uuid: str
    fingerprint: str
    kind: str
    subject: str
    predicate: str
    object_value: str
    status: str


@dataclass(frozen=True, slots=True)
class RelatedFact:
    fingerprint: str
    subject: str
    predicate: str
    object_value: str


def _memory_key(user_id: int, fingerprint: str) -> str:
    return f"{int(user_id)}:{fingerprint}"


def _entity_key(user_id: int, name: str) -> str:
    return f"{int(user_id)}:{name.casefold()}"


def _check_projectable(item: ProjectableItem) -> None:
    if str(item.status or "").casefold() != "active":
        raise Neo4jProjectionRefused(f"status_not_active:{item.status}")
    if int(item.user_id) <= 0 or not item.fingerprint or not item.tenant_uuid:
        raise Neo4jProjectionRefused("missing_tenant_identity")
    haystack = " ".join(
        (item.subject, item.predicate, item.object_value)
    )
    if contains_credentials(haystack):
        raise Neo4jProjectionRefused("secret_bearing_text")
    # Defense in depth: the full validator lexicon (health, financial,
    # sensitive) applies at the boundary too, across every view.
    if _contains_sensitive_data_any(
        item.subject, item.predicate, item.object_value, haystack
    ):
        raise Neo4jProjectionRefused("sensitive_bearing_text")


class Neo4jProjection:
    """Tenant-scoped Neo4j projection with fail-open timeouts."""

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        timeout_seconds: float = 3.0,
        driver: Any | None = None,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database or "neo4j"
        self._timeout_seconds = max(0.01, float(timeout_seconds))
        self._driver = driver
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="neo4j-proj")

    # -- lifecycle ------------------------------------------------------

    def _connect(self) -> Any:
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                connection_timeout=self._timeout_seconds,
            )
        return self._driver

    def close(self) -> None:
        try:
            if self._driver is not None and hasattr(self._driver, "close"):
                self._driver.close()
        except Exception:
            logger.debug("Neo4j driver close failed", exc_info=True)
        finally:
            self._driver = None
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, fn: Callable[[], Any]) -> Any:
        future: Future[Any] = self._executor.submit(fn)
        try:
            return future.result(timeout=self._timeout_seconds)
        except FuturesTimeout as exc:
            raise Neo4jUnavailable("neo4j_timeout") from exc
        except Neo4jUnavailable:
            raise
        except Exception as exc:
            raise Neo4jUnavailable(f"neo4j_error:{type(exc).__name__}") from exc

    def _session(self) -> Any:
        return self._connect().session(database=self._database)

    @staticmethod
    def _close_session(session: Any) -> None:
        try:
            session.close()
        except Exception:
            logger.debug("Neo4j session close failed", exc_info=True)

    def _execute(self, cypher: str, params: dict[str, Any]) -> list[Any]:
        def _call() -> list[Any]:
            session = self._session()
            try:
                return list(session.run(cypher, params))
            finally:
                self._close_session(session)

        return self._run(_call)

    # -- schema ----------------------------------------------------------

    def ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT user_user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
            "CREATE CONSTRAINT memory_key IF NOT EXISTS FOR (m:Memory) REQUIRE m.key IS UNIQUE",
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
        ]
        for statement in statements:
            self._execute(statement, {})

    # -- writes ----------------------------------------------------------

    def upsert_item(self, item: ProjectableItem) -> None:
        _check_projectable(item)
        user_id = int(item.user_id)
        key = _memory_key(user_id, item.fingerprint)
        params: dict[str, Any] = {
            "user_id": user_id,
            "tenant_uuid": str(item.tenant_uuid),
            "key": key,
            "fingerprint": item.fingerprint,
            "kind": str(item.kind),
            "subject": item.subject,
            "predicate": item.predicate,
            "object_value": item.object_value,
        }
        cypher = """
        MERGE (u:User {user_id: $user_id})
        MERGE (m:Memory {key: $key})
        ON CREATE SET m.user_id = $user_id,
                      m.tenant_uuid = $tenant_uuid,
                      m.fingerprint = $fingerprint,
                      m.kind = $kind,
                      m.subject = $subject,
                      m.predicate = $predicate,
                      m.object_value = $object_value
        ON MATCH SET m.user_id = $user_id,
                     m.tenant_uuid = $tenant_uuid,
                     m.kind = $kind,
                     m.subject = $subject,
                     m.predicate = $predicate,
                     m.object_value = $object_value
        MERGE (u)-[:HAS_MEMORY]->(m)
        """
        self._execute(cypher, params)
        if str(item.kind or "").casefold() in _ENTITY_KINDS:
            self._upsert_entity_edges(user_id, key, item.subject, item.object_value)

    def _upsert_entity_edges(
        self, user_id: int, memory_key: str, subject: str, object_value: str
    ) -> None:
        names = [str(subject or "")[:200].strip(), str(object_value or "")[:200].strip()]
        for name in dict.fromkeys(n for n in names if n):
            self._execute(
                """
                MATCH (m:Memory {key: $key, user_id: $user_id})
                MERGE (e:Entity {key: $entity_key})
                ON CREATE SET e.user_id = $user_id, e.name = $name
                MERGE (m)-[:ABOUT]->(e)
                """,
                {
                    "key": memory_key,
                    "user_id": user_id,
                    "entity_key": _entity_key(user_id, name),
                    "name": name,
                },
            )

    def delete_item(self, *, tenant_uuid: str, user_id: int, fingerprint: str) -> None:
        self._execute(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory {key: $key})
            WHERE m.tenant_uuid = $tenant_uuid
            DETACH DELETE m
            """,
            {
                "user_id": int(user_id),
                "tenant_uuid": str(tenant_uuid),
                "key": _memory_key(int(user_id), fingerprint),
                "fingerprint": fingerprint,
            },
        )
        self._delete_orphan_entities(int(user_id))

    def delete_tenant(self, *, tenant_uuid: str, user_id: int) -> None:
        self._execute(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)
            WHERE m.tenant_uuid = $tenant_uuid
            DETACH DELETE m
            """,
            {"user_id": int(user_id), "tenant_uuid": str(tenant_uuid)},
        )
        self._execute(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_TOPIC]->(t:Topic)
            WHERE t.tenant_uuid = $tenant_uuid
            DETACH DELETE t
            """,
            {"user_id": int(user_id), "tenant_uuid": str(tenant_uuid)},
        )
        self._delete_orphan_entities(int(user_id))
        self._execute(
            """
            MATCH (u:User {user_id: $user_id})
            WHERE NOT (u)-[:HAS_MEMORY]->(:Memory)
            DETACH DELETE u
            """,
            {"user_id": int(user_id)},
        )

    def _delete_orphan_entities(self, user_id: int) -> None:
        self._execute(
            """
            MATCH (e:Entity {user_id: $user_id})
            WHERE NOT (e)<-[:ABOUT]-(:Memory)
            DETACH DELETE e
            """,
            {"user_id": int(user_id)},
        )

    # -- question topics (Phase 5: flag-gated, TTL, synthesis-only) --------

    @staticmethod
    def _topic_key(user_id: int, name: str) -> str:
        return f"{int(user_id)}:{str(name or '').casefold().strip()}"

    def store_question_topics(
        self,
        *,
        user_id: int,
        tenant_uuid: str,
        topics: list[str] | tuple[str, ...],
    ) -> None:
        """MERGE up to N topic nodes with a refreshed 30-day TTL."""
        names = []
        for raw in (topics or [])[:QUESTION_TOPIC_MAX]:
            name = " ".join(str(raw or "").split()).strip()[:100].strip()
            if len(name) < 2 or contains_credentials(name):
                continue
            if name.casefold() not in {n.casefold() for n in names}:
                names.append(name)
        if not names:
            return
        expires = datetime.now(UTC) + timedelta(days=QUESTION_TOPIC_TTL_DAYS)
        for name in names:
            self._execute(
                """
                MERGE (u:User {user_id: $user_id})
                MERGE (t:Topic {key: $key})
                ON CREATE SET t.user_id = $user_id,
                              t.tenant_uuid = $tenant_uuid,
                              t.name = $name,
                              t.expires_at = $expires_at
                ON MATCH SET t.user_id = $user_id,
                             t.tenant_uuid = $tenant_uuid,
                             t.name = $name,
                             t.expires_at = $expires_at
                MERGE (u)-[:HAS_TOPIC]->(t)
                """,
                {
                    "user_id": int(user_id),
                    "tenant_uuid": str(tenant_uuid),
                    "key": self._topic_key(int(user_id), name),
                    "name": name,
                    "expires_at": expires,
                },
            )

    def read_live_topics(
        self, *, user_id: int, tenant_uuid: str, terms: list[str], limit: int = 3
    ) -> list[str]:
        """Live (unexpired) topic names overlapping the query terms."""
        clean = [t.casefold() for t in (terms or []) if str(t or "").strip()]
        if not clean:
            return []
        rows = self._execute(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_TOPIC]->(t:Topic)
            WHERE t.tenant_uuid = $tenant_uuid
              AND t.expires_at > $now
              AND any(term IN $terms WHERE toLower(t.name) CONTAINS term)
            RETURN t.name AS name
            LIMIT $limit
            """,
            {
                "user_id": int(user_id),
                "tenant_uuid": str(tenant_uuid),
                "terms": clean,
                "now": datetime.now(UTC),
                "limit": max(1, min(int(limit or 1), 10)),
            },
        )
        names: list[str] = []
        for row in rows or []:
            get = row.get if hasattr(row, "get") else (lambda k, r=row: r[k])
            try:
                name = str(get("name") or "").strip()
            except (KeyError, TypeError, AttributeError):
                continue
            if name and name not in names:
                names.append(name)
        return names

    def list_topics(self, *, user_id: int, tenant_uuid: str) -> list[dict[str, Any]]:
        """All live topics for user controls (view)."""
        rows = self._execute(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_TOPIC]->(t:Topic)
            WHERE t.tenant_uuid = $tenant_uuid AND t.expires_at > $now
            RETURN t.name AS name, t.expires_at AS expires_at
            ORDER BY t.name
            """,
            {
                "user_id": int(user_id),
                "tenant_uuid": str(tenant_uuid),
                "now": datetime.now(UTC),
            },
        )
        out: list[dict[str, Any]] = []
        for row in rows or []:
            get = row.get if hasattr(row, "get") else (lambda k, r=row: r[k])
            try:
                expires = get("expires_at")
                out.append(
                    {
                        "name": str(get("name") or ""),
                        "expires_at": expires.isoformat()
                        if hasattr(expires, "isoformat")
                        else (str(expires) if expires is not None else None),
                    }
                )
            except (KeyError, TypeError, AttributeError):
                continue
        return out

    def delete_topic(self, *, user_id: int, name: str) -> None:
        self._execute(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_TOPIC]->(t:Topic {key: $key})
            DETACH DELETE t
            """,
            {"user_id": int(user_id), "key": self._topic_key(int(user_id), name)},
        )

    def delete_expired_topics(self) -> None:
        """Global TTL garbage collection (no user data read)."""
        self._execute(
            """
            MATCH (t:Topic)
            WHERE t.expires_at <= $now
            DETACH DELETE t
            """,
            {"now": datetime.now(UTC)},
        )

    # -- reads -----------------------------------------------------------

    def read_related(
        self, *, user_id: int, tenant_uuid: str, terms: list[str], limit: int = 4
    ) -> list[RelatedFact]:
        clean = [t.casefold() for t in (terms or []) if str(t or "").strip()]
        if not clean:
            return []
        rows = self._execute(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)
            WHERE m.tenant_uuid = $tenant_uuid
              AND any(t IN $terms WHERE toLower(m.subject) CONTAINS t
                                    OR toLower(m.predicate) CONTAINS t
                                    OR toLower(m.object_value) CONTAINS t)
            RETURN m.fingerprint AS fingerprint, m.subject AS subject,
                   m.predicate AS predicate, m.object_value AS object_value
            UNION
            MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)-[:ABOUT]->(e:Entity)<-[:ABOUT]-(rel:Memory)
            WHERE m.tenant_uuid = $tenant_uuid AND rel.tenant_uuid = $tenant_uuid
              AND any(t IN $terms WHERE toLower(m.subject) CONTAINS t
                                    OR toLower(m.predicate) CONTAINS t
                                    OR toLower(m.object_value) CONTAINS t)
            RETURN rel.fingerprint AS fingerprint, rel.subject AS subject,
                   rel.predicate AS predicate, rel.object_value AS object_value
            LIMIT $limit
            """,
            {
                "user_id": int(user_id),
                "tenant_uuid": str(tenant_uuid),
                "terms": clean,
                "limit": max(1, min(int(limit or 1), 25)),
            },
        )
        facts: list[RelatedFact] = []
        for row in rows or []:
            get = row.get if hasattr(row, "get") else (lambda k, r=row: r[k])
            try:
                facts.append(
                    RelatedFact(
                        fingerprint=str(get("fingerprint")),
                        subject=str(get("subject") or ""),
                        predicate=str(get("predicate") or ""),
                        object_value=str(get("object_value") or ""),
                    )
                )
            except (KeyError, TypeError, AttributeError):
                continue
        return facts

    def fingerprint_set(self, *, user_id: int, tenant_uuid: str) -> set[str]:
        rows = self._execute(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)
            WHERE m.tenant_uuid = $tenant_uuid
            RETURN m.fingerprint AS fingerprint
            """,
            {"user_id": int(user_id), "tenant_uuid": str(tenant_uuid)},
        )
        out: set[str] = set()
        for row in rows or []:
            get = row.get if hasattr(row, "get") else (lambda k, r=row: r[k])
            try:
                out.add(str(get("fingerprint")))
            except (KeyError, TypeError, AttributeError):
                continue
        return out

    def fetch_node(
        self, *, user_id: int, tenant_uuid: str, fingerprint: str
    ) -> dict[str, Any] | None:
        rows = self._execute(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory {key: $key})
            WHERE m.tenant_uuid = $tenant_uuid
            RETURN m.subject AS subject, m.predicate AS predicate,
                   m.object_value AS object_value
            """,
            {
                "user_id": int(user_id),
                "tenant_uuid": str(tenant_uuid),
                "key": _memory_key(int(user_id), fingerprint),
            },
        )
        if not rows:
            return None
        row = rows[0]
        get = row.get if hasattr(row, "get") else (lambda k: row[k])
        return {
            "subject": str(get("subject") or ""),
            "predicate": str(get("predicate") or ""),
            "object_value": str(get("object_value") or ""),
        }


__all__ = [
    "GRAPH_RECALL_MAX_FACTS",
    "QUESTION_TOPIC_MAX",
    "QUESTION_TOPIC_TTL_DAYS",
    "Neo4jProjection",
    "Neo4jProjectionRefused",
    "Neo4jUnavailable",
    "ProjectableItem",
    "RelatedFact",
]
