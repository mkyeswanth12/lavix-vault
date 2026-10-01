"""User-facing AI utility routes backed by PostgreSQL/pgvector."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from app.agent.retrieval import VaultRetriever
from app.auth import require_ai_permission
from app.config import settings
from app.database import get_db
from app.ingestion.routing import route_file

from .schemas import SemanticSearchRequest

logger = logging.getLogger(__name__)
router = APIRouter()
vault_retriever = VaultRetriever()

CurrentUser = Annotated[dict[str, Any], Depends(require_ai_permission)]

_ACTIVE_INGESTION_STATES = frozenset(
    {"queued", "decrypting", "converting", "parsing", "chunking", "embedding", "publishing"}
)


@router.post("/search")
async def semantic_search(request: SemanticSearchRequest, user: CurrentUser) -> dict[str, Any]:
    """Search the caller's current, consented pgvector document revisions."""

    query = " ".join(request.query.split())
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank")
    limit = max(1, min(int(request.max_results or 10), 20))
    # Retrieval ranks chunks. Ask for enough candidates to return `limit`
    # distinct file suggestions even when one document owns several top chunks.
    candidate_limit = min(60, max(limit * 4, limit))
    try:
        results = await vault_retriever.search(
            user_id=int(user["id"]),
            query=query,
            file_ids=None,
            top_k=candidate_limit,
            deep_search=False,
        )
    except Exception:
        logger.warning("Semantic search unavailable", exc_info=True)
        raise HTTPException(status_code=503, detail="Semantic search unavailable") from None

    formatted: list[dict[str, Any]] = []
    seen_file_ids: set[int] = set()
    for item in results:
        file_id = int(item["file_id"])
        if file_id in seen_file_ids:
            continue
        seen_file_ids.add(file_id)
        score = max(0.0, min(1.0, float(item.get("score") or 0.0)))
        formatted.append(
            {
                "file_id": file_id,
                "filename": item["filename"],
                "mime_type": item["mime_type"],
                "tags": list(item.get("tags") or []),
                "summary": item.get("summary"),
                "doc_type": item.get("doc_type"),
                "match_percentage": round(score * 100),
            }
        )
        if len(formatted) >= limit:
            break
    total_files = 0
    try:
        with get_db() as conn:
            total_files = conn.execute(
                "SELECT COUNT(*) AS cnt FROM files WHERE user_id = %s AND is_deleted = FALSE",
                [int(user["id"])],
            ).fetchone()["cnt"] or 0
    except Exception:
        logger.warning("Unable to count total files", exc_info=True)
    return {"query": query, "results": formatted, "count": len(formatted), "total_files": total_files}


@router.get("/tags")
def get_all_tags(user: CurrentUser) -> dict[str, Any]:
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT f.quick_tags
            FROM files AS f
            WHERE f.user_id = %s
              AND f.is_deleted = FALSE
              AND f.user_granted_ai_access = TRUE
              AND f.current_revision IS NOT NULL
              AND f.quick_tags IS NOT NULL
              AND COALESCE(to_jsonb(f)->>'intelligence_status', 'pending')
                    IN ('model', 'fallback')
            """,
            (user["id"],),
        )
        counts: dict[str, int] = {}
        for row in cursor.fetchall():
            for raw in row["quick_tags"] or []:
                tag = str(raw).lower().strip()
                if tag:
                    counts[tag] = counts.get(tag, 0) + 1
    tags = [
        {"tag": tag, "count": count}
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {"tags": tags, "total_unique_tags": len(tags)}


@router.get("/stats")
def get_ai_stats(user: CurrentUser) -> dict[str, Any]:
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT original_filename, mime_type, user_granted_ai_access,
                   current_revision, COALESCE(ai_status, 'not_granted') AS ai_status,
                   quick_tags,
                   COALESCE(to_jsonb(files)->>'intelligence_status', 'pending')
                       AS intelligence_status
            FROM files
            WHERE user_id = %s AND is_deleted = FALSE
            """,
            (user["id"],),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        cursor.execute(
            "SELECT COUNT(*) AS count FROM chat_messages WHERE user_id = %s AND role = 'assistant'",
            (user["id"],),
        )
        interactions = int(cursor.fetchone()["count"])

    total = 0
    supported = 0
    indexable = 0
    unactionable = 0
    awaiting_grant = 0
    needs_attention = 0
    media_files = 0
    tagged = 0
    searchable = 0
    ai_ready_files = 0
    active = 0
    queued = 0
    failed = 0
    cancelled = 0
    password_required = 0
    model_summaries = 0
    fallback_summaries = 0
    for row in rows:
        media_type = str(row.get("mime_type") or "")
        is_media = media_type.casefold().startswith(("audio/", "video/"))
        if is_media:
            media_files += 1
        else:
            # Retain the historical ``total`` meaning for existing clients while
            # publishing the exact parser-supported count separately.
            total += 1
        state = str(row.get("ai_status") or "not_granted")
        current = row.get("current_revision") is not None
        consent = bool(row.get("user_granted_ai_access"))
        intelligence_status = str(row.get("intelligence_status") or "pending")
        # Beta-dev parity: ai_ready means embeddings exist, period — a
        # published revision counts with or without consent. (Revoke nulls
        # the revision, so revokes still drop the count.)
        if current:
            ai_ready_files += 1
        if route_file(str(row.get("original_filename") or ""), media_type).supported:
            supported += 1
            if not consent:
                # Grantable but ungranted: ACTIVATE's target population.
                # Cancelled rows are already inside ``indexable`` (granted),
                # so they must not be counted here — no double count.
                awaiting_grant += 1
            if consent:
                # Honest denominator for progress UIs: consented AND
                # parser-supported AND still progressable by the system.
                # Unconsented vault files are intentionally dark and must
                # not look like a stalled backlog; terminal states that need
                # user action (password supply / retry-or-remove) surface
                # via needs_attention instead of pinning progress below 100%.
                if state in ("password_required", "failed"):
                    needs_attention += 1
                else:
                    indexable += 1
            # Permanently un-actionable: supported yet unable to ever become
            # searchable (terminal-bad with no usable prior revision).
            # Failed/cancelled/password-required rows that still carry a
            # searchable current revision are NOT counted here — they stay
            # in the actionable population via ``searchable``.
            if not (consent and current) and state in ("failed", "cancelled", "password_required"):
                unactionable += 1

        if state in _ACTIVE_INGESTION_STATES:
            active += 1
        if state == "queued":
            queued += 1
        if consent and current:
            searchable += 1
            if intelligence_status == "model":
                model_summaries += 1
            elif intelligence_status == "fallback":
                fallback_summaries += 1
            if intelligence_status in {"model", "fallback"} and row.get("quick_tags"):
                tagged += 1
        elif state == "failed":
            failed += 1
        elif state == "cancelled":
            cancelled += 1
        elif state == "password_required":
            password_required += 1

    # All-files total for the badge fraction: every non-deleted file,
    # media and unsupported formats included. Unlike ``total`` (which
    # preserves its historical non-media meaning), this is the raw
    # population the box fraction displays against.
    all_files_total = len(rows)
    actionable = max(searchable, supported - unactionable)
    # Eligible base: everything that could still become searchable —
    # indexable (granted, incl. cancelled-retry) plus grantable-but-
    # ungranted files. Failed/password-locked rows are in neither and
    # surface via needs_attention instead.
    eligible = indexable + awaiting_grant
    return {
        "total": total,
        "all_files_total": all_files_total,
        "supported": supported,
        "actionable": actionable,
        "unactionable": unactionable,
        "eligible": eligible,
        "awaiting_grant": awaiting_grant,
        "indexable": indexable,
        "needs_attention": needs_attention,
        "active": active,
        "queued": queued,
        "searchable": searchable,
        "tagged": tagged,
        "ai_ready": ai_ready_files,
        "failed": failed,
        "cancelled": cancelled,
        "password_required": password_required,
        "model_summaries": model_summaries,
        "fallback_summaries": fallback_summaries,
        "media_files": media_files,
        "ready_percentage": round(
            (ai_ready_files / max(eligible, ai_ready_files) * 100) if (eligible or ai_ready_files) else 0,
            2,
        ),
        "interactions": {"total": interactions, "message": "Durable assistant messages"},
        "features": {
            "pgvector_enabled": True,
            "embedding_api": settings.embedding_api_url,
        },
    }
