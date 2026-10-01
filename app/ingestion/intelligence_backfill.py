"""Backfill summaries, controlled types, and tags without re-embedding documents."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.database import get_db
from app.graph_memory.inference_priority import (
    BackgroundInferenceDeferred,
    background_inference_allowed,
)
from app.services.model_config import ModelConfigurationRepository

from .intelligence import (
    ConfiguredDocumentIntelligenceService,
    DocumentIntelligenceService,
    IntelligenceSettings,
    normalize_tags,
)


@dataclass(frozen=True, slots=True)
class _DocumentView:
    source_name: str
    source_sha256: str
    media_type: str
    parser_fingerprint: str


@dataclass(frozen=True, slots=True)
class _ProvenanceView:
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class _ChunkView:
    text: str
    provenance: tuple[_ProvenanceView, ...] = ()


_RASTER_SQL = """(
    LOWER(COALESCE(f.mime_type, '')) LIKE 'image/%%'
    OR LOWER(f.original_filename) ~ '\\.(bmp|gif|jpe?g|png|tiff?|webp)$'
)"""


def _candidates(
    user_id: int | None,
    limit: int | None,
    *,
    missing_tags_only: bool = False,
) -> list[dict[str, Any]]:
    where = ["f.is_deleted = FALSE", "f.current_revision IS NOT NULL"]
    params: list[Any] = []
    if missing_tags_only:
        where.extend(
            (
                f"NOT {_RASTER_SQL}",
                """
                NOT EXISTS (
                    SELECT 1
                    FROM unnest(COALESCE(f.quick_tags, ARRAY[]::TEXT[])) AS tag(value)
                    WHERE BTRIM(tag.value) <> ''
                )
                """,
                """
                EXISTS (
                    SELECT 1
                    FROM document_chunks AS indexed_chunk
                    WHERE indexed_chunk.file_id = f.id
                      AND indexed_chunk.user_id = f.user_id
                      AND indexed_chunk.revision = f.current_revision
                )
                """,
            )
        )
    if user_id is not None:
        where.append("f.user_id = %s")
        params.append(user_id)
    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT %s"
        params.append(limit)
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            f"""
            SELECT f.id AS file_id, f.user_id, f.current_revision AS revision,
                   f.original_filename AS source_name, f.mime_type,
                   f.quick_summary, f.quick_tags, f.doc_type, f.intelligence_status,
                   dr.source_sha256, dr.parser_fingerprint,
                   COALESCE((
                       SELECT jsonb_agg(
                           jsonb_build_object(
                               'text', dc.content,
                               'page_numbers', jsonb_path_query_array(
                                   dc.provenance,
                                   '$[*].page_number'
                               )
                           )
                           ORDER BY dc.ordinal
                       )
                       FROM document_chunks AS dc
                       WHERE dc.file_id = f.id
                         AND dc.user_id = f.user_id
                         AND dc.revision = f.current_revision
                   ), '[]'::jsonb) AS chunks
            FROM files AS f
            JOIN document_revisions AS dr
              ON dr.file_id = f.id
             AND dr.user_id = f.user_id
             AND dr.revision = f.current_revision
             AND dr.status = 'current'
            WHERE {" AND ".join(where)}
            ORDER BY f.id
            {limit_sql}
            """,
            tuple(params),
        )
        return [dict(row) for row in cursor.fetchall()]


def _raster_reindex_candidates(
    user_id: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """List missing-tag raster revisions that cannot be repaired without pixels."""

    where = [
        "f.is_deleted = FALSE",
        "f.current_revision IS NOT NULL",
        _RASTER_SQL,
        """
        NOT EXISTS (
            SELECT 1
            FROM unnest(COALESCE(f.quick_tags, ARRAY[]::TEXT[])) AS tag(value)
            WHERE BTRIM(tag.value) <> ''
        )
        """,
        """
        EXISTS (
            SELECT 1
            FROM document_chunks AS indexed_chunk
            WHERE indexed_chunk.file_id = f.id
              AND indexed_chunk.user_id = f.user_id
              AND indexed_chunk.revision = f.current_revision
        )
        """,
    ]
    params: list[Any] = []
    if user_id is not None:
        where.append("f.user_id = %s")
        params.append(user_id)
    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT %s"
        params.append(limit)
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            f"""
            SELECT f.id AS file_id, f.user_id, f.current_revision AS revision,
                   f.original_filename AS source_name, f.mime_type
            FROM files AS f
            JOIN document_revisions AS dr
              ON dr.file_id = f.id
             AND dr.user_id = f.user_id
             AND dr.revision = f.current_revision
             AND dr.status = 'current'
            WHERE {" AND ".join(where)}
            ORDER BY f.id
            {limit_sql}
            """,
            tuple(params),
        )
        return [dict(row) for row in cursor.fetchall()]


def _resolve_intelligence_role() -> tuple[bool, str | None]:
    """Use the same admin-owned role resolution as the ingestion worker."""

    with get_db() as connection:
        configuration = ModelConfigurationRepository(connection).get()
    return configuration.intelligence.enabled, configuration.intelligence.model


def _chunk_views(value: Any) -> tuple[_ChunkView, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    chunks: list[_ChunkView] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        pages: list[int] = []
        raw_pages = item.get("page_numbers")
        if isinstance(raw_pages, Sequence) and not isinstance(raw_pages, (str, bytes)):
            for raw_page in raw_pages:
                if isinstance(raw_page, bool):
                    continue
                try:
                    page = int(raw_page)
                except (TypeError, ValueError):
                    continue
                if page >= 1 and page not in pages:
                    pages.append(page)
        chunks.append(
            _ChunkView(
                text=text,
                provenance=tuple(_ProvenanceView(page_number=page) for page in pages),
            )
        )
    return tuple(chunks)


def _persist(row: dict[str, Any], intelligence: Any) -> bool:
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            UPDATE files
            SET quick_summary = %s,
                quick_tags = %s,
                doc_type = %s,
                intelligence_status = %s
            WHERE id = %s
              AND user_id = %s
              AND current_revision = %s
              AND is_deleted = FALSE
            """,
            (
                intelligence.summary,
                list(intelligence.tags),
                intelligence.doc_type,
                intelligence.status,
                row["file_id"],
                row["user_id"],
                row["revision"],
            ),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return False
        cursor.execute(
            """
            UPDATE document_revisions
            SET metadata = jsonb_set(
                COALESCE(metadata, '{}'::jsonb),
                '{intelligence}',
                %s::jsonb,
                TRUE
            )
            WHERE file_id = %s
              AND user_id = %s
              AND revision = %s
              AND status = 'current'
            """,
            (
                json.dumps(intelligence.metadata(), ensure_ascii=False),
                row["file_id"],
                row["user_id"],
                row["revision"],
            ),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return False
        connection.commit()
        return True


def _persist_tags(
    row: dict[str, Any],
    tags: tuple[str, ...],
    *,
    require_missing: bool = False,
) -> bool:
    missing_fence = """
              AND NOT EXISTS (
                  SELECT 1
                  FROM unnest(COALESCE(quick_tags, ARRAY[]::TEXT[])) AS tag(value)
                  WHERE BTRIM(tag.value) <> ''
              )
    """ if require_missing else ""
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            f"""
            UPDATE files
            SET quick_tags = %s
            WHERE id = %s
              AND user_id = %s
              AND current_revision = %s
              AND is_deleted = FALSE
              {missing_fence}
            """,
            (list(tags), row["file_id"], row["user_id"], row["revision"]),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return False
        cursor.execute(
            """
            UPDATE document_revisions
            SET metadata = jsonb_set(
                COALESCE(metadata, '{}'::jsonb),
                '{intelligence}',
                (
                    CASE
                        WHEN jsonb_typeof(COALESCE(metadata, '{}'::jsonb)->'intelligence') = 'object'
                        THEN COALESCE(metadata, '{}'::jsonb)->'intelligence'
                        ELSE '{}'::jsonb
                    END
                ) || jsonb_build_object('tags', %s::jsonb),
                TRUE
            )
            WHERE file_id = %s
              AND user_id = %s
              AND revision = %s
              AND status = 'current'
            """,
            (json.dumps(list(tags)), row["file_id"], row["user_id"], row["revision"]),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return False
        connection.commit()
        return True


async def run(
    *,
    apply: bool,
    user_id: int | None,
    limit: int | None,
    normalize_tags_only: bool = False,
    missing_tags_only: bool = False,
    priority_gate: Callable[[], Awaitable[bool]] | None = None,
    role_resolver: Callable[[], tuple[bool, str | None]] | None = None,
) -> int:
    if normalize_tags_only and missing_tags_only:
        raise ValueError("tag repair modes are mutually exclusive")
    intelligence_settings = IntelligenceSettings.from_environment()
    if role_resolver is not None:
        service = ConfiguredDocumentIntelligenceService(
            role_resolver,
            intelligence_settings,
            priority_gate=priority_gate,
        )
    else:
        service = (
            DocumentIntelligenceService(intelligence_settings, priority_gate=priority_gate)
            if priority_gate is not None
            else DocumentIntelligenceService(intelligence_settings)
        )
    rows = (
        await asyncio.to_thread(
            _candidates,
            user_id,
            limit,
            missing_tags_only=True,
        )
        if missing_tags_only
        else await asyncio.to_thread(_candidates, user_id, limit)
    )
    raster_rows = (
        await asyncio.to_thread(_raster_reindex_candidates, user_id, limit)
        if missing_tags_only
        else []
    )
    completed = 0
    changed_count = 0
    skipped_raster = 0
    deferred_count = 0
    by_file_type: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    for row in raster_rows:
        print(
            json.dumps(
                {
                    "file_id": row["file_id"],
                    "revision": row["revision"],
                    "filename": row["source_name"],
                    "status": "targeted_reindex_required",
                    "reason": "raster_requires_source_pixels",
                    "persisted": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for row in rows:
        chunks = _chunk_views(row.get("chunks"))
        if normalize_tags_only:
            before = tuple(str(tag) for tag in row.get("quick_tags") or [])
            # This is policy cleanup, not a weak replacement for model-backed
            # intelligence. A representative text sample is incomplete for
            # large documents and absent for image/VLM tags; using it as a
            # deletion or synthesis authority can remove good metadata or
            # manufacture repeated OCR fragments. Missing tags therefore stay
            # missing until a normal intelligence repair or reindex.
            tags = normalize_tags(
                before,
                source_name=str(row.get("source_name") or ""),
                preserve_existing=True,
            )
            changed = tags != before
            persisted = await asyncio.to_thread(_persist_tags, row, tags) if apply and changed else False
            print(
                json.dumps(
                    {
                        "file_id": row["file_id"],
                        "revision": row["revision"],
                        "filename": row["source_name"],
                        "before": list(before),
                        "tags": list(tags),
                        "changed": changed,
                        "persisted": persisted,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            changed_count += int(changed)
            completed += int(not apply or not changed or persisted)
            continue
        document = _DocumentView(
            source_name=str(row["source_name"]),
            source_sha256=str(row["source_sha256"]),
            media_type=str(row.get("mime_type") or "application/octet-stream"),
            parser_fingerprint=str(row.get("parser_fingerprint") or ""),
        )
        if document.media_type.lower().startswith("image/"):
            # Metadata-only repair cannot see source pixels. Preserve existing
            # VLM intelligence instead of replacing it from stale OCR; raster
            # files that genuinely need repair must be explicitly re-indexed.
            skipped_raster += 1
            print(
                json.dumps(
                    {
                        "file_id": row["file_id"],
                        "revision": row["revision"],
                        "filename": row["source_name"],
                        "status": "skipped",
                        "reason": "raster_requires_targeted_reindex",
                        "persisted": False,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            completed += 1
            continue
        try:
            intelligence = await service.analyze(document, chunks)
        except BackgroundInferenceDeferred:
            deferred_count += 1
            print(
                json.dumps(
                    {
                        "file_id": row["file_id"],
                        "revision": row["revision"],
                        "filename": row["source_name"],
                        "status": "deferred",
                        "reason": "foreground_inference_unavailable",
                        "persisted": False,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            # A closed gate affects every model-backed candidate. Stop after
            # the first deferral so an unavailable scheduler cannot turn this
            # bounded repair command into one timeout per row.
            break
        if missing_tags_only:
            tags = intelligence.tags
            changed = bool(tags)
            persisted = (
                await asyncio.to_thread(
                    _persist_tags,
                    row,
                    tags,
                    require_missing=True,
                )
                if apply and changed
                else False
            )
            if intelligence.fallback_reason:
                fallback_reasons[intelligence.fallback_reason] += 1
            print(
                json.dumps(
                    {
                        "file_id": row["file_id"],
                        "revision": row["revision"],
                        "filename": row["source_name"],
                        "tags": list(tags),
                        "analysis_status": intelligence.status,
                        "fallback_reason": intelligence.fallback_reason,
                        "changed": changed,
                        "persisted": persisted,
                        "preserved": [
                            "quick_summary",
                            "doc_type",
                            "intelligence_status",
                            "current_revision",
                            "document_chunks",
                            "embeddings",
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            changed_count += int(changed)
            completed += int(not apply or not changed or persisted)
            continue
        changed = any(
            (
                str(row.get("quick_summary") or "") != intelligence.summary,
                tuple(str(tag) for tag in row.get("quick_tags") or []) != intelligence.tags,
                str(row.get("doc_type") or "") != intelligence.doc_type,
                str(row.get("intelligence_status") or "pending") != intelligence.status,
            )
        )
        persisted = await asyncio.to_thread(_persist, row, intelligence) if apply and changed else False
        if changed:
            by_file_type[intelligence.doc_type] += 1
            if intelligence.fallback_reason:
                fallback_reasons[intelligence.fallback_reason] += 1
        print(
            json.dumps(
                {
                    "file_id": row["file_id"],
                    "revision": row["revision"],
                    "filename": row["source_name"],
                    "doc_type": intelligence.doc_type,
                    "tags": list(intelligence.tags),
                    "status": intelligence.status,
                    "fallback_reason": intelligence.fallback_reason,
                    "changed": changed,
                    "persisted": persisted,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        changed_count += int(changed)
        completed += int(not apply or not changed or persisted)
    mode = (
        "normalize-tags"
        if normalize_tags_only
        else "missing-tags"
        if missing_tags_only
        else "intelligence"
    )
    summary = {
        "mode": f"{mode}-{'apply' if apply else 'dry-run'}",
        "candidates": len(rows),
        "completed": completed,
        "changed": changed_count,
        "skipped_raster": skipped_raster,
        "deferred": deferred_count,
        "by_file_type": dict(sorted(by_file_type.items())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
    }
    if missing_tags_only:
        summary["targeted_reindex_rasters"] = len(raster_rows)
    print(
        json.dumps(summary)
    )
    return 0 if completed == len(rows) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="persist results; default is dry-run")
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--limit", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--normalize-tags-only",
        action="store_true",
        help="clean existing tags by policy without deleting valid values or calling a model",
    )
    modes.add_argument(
        "--missing-tags-only",
        action="store_true",
        help=(
            "repair only grounded tags on current indexed non-raster revisions; "
            "preserve summaries, types, revisions, chunks, and embeddings"
        ),
    )
    args = parser.parse_args()
    if args.user_id is not None and args.user_id < 1:
        parser.error("--user-id must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    raise SystemExit(
        asyncio.run(
            run(
                apply=args.apply,
                user_id=args.user_id,
                limit=args.limit,
                normalize_tags_only=args.normalize_tags_only,
                missing_tags_only=args.missing_tags_only,
                priority_gate=background_inference_allowed,
                role_resolver=_resolve_intelligence_role,
            )
        )
    )


if __name__ == "__main__":
    main()
