"""Encrypted file management and durable ingestion lifecycle endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from minio import Minio
from pydantic import BaseModel, Field

from app.auth import get_current_user, require_ai_permission
from app.config import settings
from app.database import get_db
from app.ingestion.commands import IngestionCommands, IngestionStatus
from app.ingestion.errors import UnsupportedFormatError
from app.ingestion.routing import route_file
from app.storage import StoredObjectIntegrityError, StoredObjectRef, StoredObjectUnavailable
from app.storage.access import VaultObjectAccess
from app.storage.upload import UploadRejected, object_keys_for_file, sanitize_filename

logger = logging.getLogger(__name__)
router = APIRouter()
CurrentUser = Annotated[dict, Depends(get_current_user)]

minio_client = Minio(
    settings.s3_endpoint.removeprefix("https://").removeprefix("http://"),
    access_key=settings.s3_access_key,
    secret_key=settings.s3_secret_key,
    secure=settings.s3_secure,
)
object_access = VaultObjectAccess(minio_client)
ingestion_commands = IngestionCommands(get_db)

_ACTIVE_STATES = frozenset(
    {"queued", "decrypting", "converting", "parsing", "chunking", "embedding", "publishing"}
)
_ACTIVE_DOCUMENT_MEDIA_TYPES = frozenset(
    {
        "application/xhtml+xml",
        "application/xml",
        "image/svg+xml",
        "text/html",
        "text/xml",
    }
)


class MoveFileRequest(BaseModel):
    folder_id: int | None = Field(default=None, gt=0)


class RenameFileRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)


class BulkFileIdsRequest(BaseModel):
    file_ids: list[int] = Field(min_length=1, max_length=100)


def _dedupe_file_ids(file_ids: list[int]) -> list[int]:
    seen: list[int] = []
    for file_id in file_ids:
        if file_id not in seen:
            seen.append(file_id)
    return seen


def _object_ref(row: dict[str, Any]) -> StoredObjectRef:
    return StoredObjectRef(
        bucket=str(row.get("s3_bucket_name") or settings.s3_bucket_name),
        payload_path=str(row["s3_path"]),
        wrapped_key_path=str(row["s3_key_path"]),
        sha256=str(row.get("sha256_hash") or "").strip(),
    )


def _compatibility_state(state: str) -> str:
    if state in _ACTIVE_STATES:
        return "processing"
    if state == "unsupported":
        return "not_supported"
    return state


def _safe_served_media_type(value: Any, filename: Any = None) -> str:
    media_type = str(value or "application/octet-stream")
    base_type = media_type.split(";", 1)[0].strip().lower()
    if base_type in _ACTIVE_DOCUMENT_MEDIA_TYPES:
        return "text/plain; charset=utf-8"
    if base_type == "application/octet-stream" and str(filename or "").lower().endswith(".pdf"):
        return "application/pdf"
    return media_type


def _has_generated_intelligence(row: Any) -> bool:
    return str(row.get("intelligence_status") or "pending") in {"model", "fallback"}


def _status_payload(status: IngestionStatus) -> dict[str, Any]:
    compatibility = _compatibility_state(status.state)
    searchable = status.consent_granted and status.current_revision is not None
    return {
        "file_id": status.file_id,
        "consent_granted": status.consent_granted,
        "ai_access_granted": status.consent_granted,
        "desired_revision": status.desired_revision,
        "current_revision": status.current_revision,
        "state": status.state,
        "status": compatibility,
        "job_id": status.job_id,
        "chunk_count": status.chunk_count,
        "chunks_indexed": status.chunk_count,
        "embeddings_ready": searchable,
        "error_code": status.error_code,
        "error_detail": status.error_detail,
    }


def _idempotency_key(value: str | None, *, operation: str) -> str:
    if value is None:
        return f"{operation}:{uuid4()}"
    normalized = value.strip()
    full_key = f"{operation}:{normalized}"
    if not normalized or len(full_key) > 160:
        raise HTTPException(status_code=422, detail="Idempotency-Key must contain 1-160 characters")
    return full_key


def _advisory_key(value: str) -> int:
    raw = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(raw, "big", signed=True)


def _require_folder(cursor: Any, user_id: int, folder_id: int | None) -> None:
    if folder_id is None:
        return
    cursor.execute(
        "SELECT id FROM folders WHERE id = %s AND user_id = %s AND is_deleted = FALSE",
        (folder_id, user_id),
    )
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Folder not found")


@router.post("/grant-ai-access/bulk")
async def bulk_grant_ai_access(body: BulkFileIdsRequest, user: CurrentUser) -> dict[str, Any]:
    """Grant AI access for a selection of files, best-effort per file.

    Item shape mirrors ``enable-all-embeddings``. Unsupported files land
    in ``failed`` (415 semantics per file) instead of aborting the batch.
    Retries are safe: re-granting an active revision returns
    ``created: False`` without queuing duplicate jobs.
    """
    require_ai_permission(user)
    items: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for file_id in _dedupe_file_ids(body.file_ids):
        try:
            result = await asyncio.to_thread(
                ingestion_commands.grant_and_enqueue,
                int(user["id"]),
                file_id,
                idempotency_key=f"grant-bulk:{uuid4().hex}:{file_id}",
            )
        except FileNotFoundError:
            failed.append({"file_id": file_id, "error": "File not found"})
            continue
        except UnsupportedFormatError as exc:
            failed.append({"file_id": file_id, "error": str(exc)})
            continue
        except ValueError as exc:
            failed.append({"file_id": file_id, "error": str(exc)})
            continue
        items.append(
            {
                "file_id": result.file_id,
                "revision": result.revision,
                "job_id": result.job_id,
                "state": result.state.value,
                "created": result.created,
            }
        )
    return {
        "message": f"AI access granted; {len(items)} ingestion job(s) queued.",
        "items": items,
        "failed": failed,
        "created_count": sum(1 for item in items if item["created"]),
        "count": len(items),
    }


@router.post("/grant-ai-access/{file_id}")
async def grant_ai_access(
    file_id: int,
    user: CurrentUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> dict[str, Any]:
    require_ai_permission(user)
    try:
        result = await asyncio.to_thread(
            ingestion_commands.grant_and_enqueue,
            int(user["id"]),
            file_id,
            idempotency_key=_idempotency_key(idempotency_key, operation=f"grant:{file_id}"),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found") from None
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    return {
        "message": "AI access granted; durable ingestion is queued."
        if result.created
        else "AI access is already active for this revision.",
        "file_id": result.file_id,
        "revision": result.revision,
        "job_id": result.job_id,
        "state": result.state.value,
        "status": _compatibility_state(result.state.value),
        "created": result.created,
    }


@router.post("/revoke-ai-access/bulk")
async def bulk_revoke_ai_access(body: BulkFileIdsRequest, user: CurrentUser) -> dict[str, Any]:
    """Revoke AI access for a selection of files, best-effort per file.

    Each file goes through the same atomic ``revoke_and_remove`` as the
    single route (consent flip + derived-data deletion in one commit), so
    revoked content is never RAG-visible. Failures are reported per file
    in ``failed`` — the batch never aborts early and never fails silent.
    Retries are safe: re-revoking an already-revoked file succeeds with
    zero removals (state idempotency, no key needed).
    """
    items: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for file_id in _dedupe_file_ids(body.file_ids):
        try:
            result = await asyncio.to_thread(
                ingestion_commands.revoke_and_remove,
                int(user["id"]),
                file_id,
            )
        except FileNotFoundError:
            failed.append({"file_id": file_id, "error": "File not found"})
            continue
        except Exception:
            logger.exception("bulk revoke failed for file %s", file_id)
            failed.append({"file_id": file_id, "error": "Revoke failed"})
            continue
        items.append(
            {
                "file_id": file_id,
                "status": "locked",
                "cancelled_jobs": result.cancelled_job_count,
                "removed_revisions": result.removed_revision_count,
                "removed_chunks": result.removed_chunk_count,
            }
        )
    return {
        "message": f"AI access revoked for {len(items)} file(s).",
        "items": items,
        "failed": failed,
        "revoked_count": len(items),
        "failed_count": len(failed),
    }


@router.post("/revoke-ai-access/{file_id}")
async def revoke_ai_access(file_id: int, user: CurrentUser) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(
            ingestion_commands.revoke_and_remove,
            int(user["id"]),
            file_id,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found") from None
    return {
        "message": "AI access revoked and derived index data removed.",
        "file_id": file_id,
        "status": "locked",
        "cancelled_jobs": result.cancelled_job_count,
        "removed_revisions": result.removed_revision_count,
        "removed_chunks": result.removed_chunk_count,
    }


@router.post("/disable-all-embeddings", deprecated=True)
async def disable_all_embeddings(user: CurrentUser) -> dict[str, Any]:
    """Deprecated compatibility alias for non-destructive cancellation.

    Older cached WebUI builds called this route from their ``Cancel indexing``
    button.  It must therefore never revoke consent or remove published data.
    Derived-data deletion remains available only through the explicitly named
    per-file ``revoke-ai-access`` route.
    """

    result = await asyncio.to_thread(ingestion_commands.cancel_indexing, int(user["id"]))
    return {
        "message": (
            "Deprecated endpoint: unfinished indexing cancelled; published index data was preserved."
        ),
        "deprecated": True,
        "replacement": "/api/files/cancel-indexing",
        "affected_files": result.file_count,
        "cancelled_jobs": result.cancelled_job_count,
        "ready_files_preserved": result.ready_file_count,
        "consent_preserved": True,
        "index_data_preserved": True,
    }


@router.post("/cancel-indexing")
async def cancel_indexing(user: CurrentUser) -> dict[str, Any]:
    """Stop unfinished ingestion while retaining consent and published vectors."""

    result = await asyncio.to_thread(ingestion_commands.cancel_indexing, int(user["id"]))
    return {
        "message": "Unfinished indexing cancelled; published index data was preserved.",
        "affected_files": result.file_count,
        "cancelled_jobs": result.cancelled_job_count,
        "ready_files_preserved": result.ready_file_count,
        "consent_preserved": True,
        "index_data_preserved": True,
    }


@router.post("/cancel-indexing/{file_id}")
async def cancel_file_indexing(file_id: int, user: CurrentUser) -> dict[str, Any]:
    """Stop one unfinished ingestion job without removing its published index."""

    try:
        result = await asyncio.to_thread(
            ingestion_commands.cancel_file_indexing,
            int(user["id"]),
            file_id,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found") from None
    return {
        "message": "Unfinished indexing cancelled; published index data was preserved.",
        "file_id": file_id,
        "affected_files": result.file_count,
        "cancelled_jobs": result.cancelled_job_count,
        "ready_files_preserved": result.ready_file_count,
        "consent_preserved": True,
        "index_data_preserved": True,
    }


@router.post("/enable-all-embeddings")
async def enable_all_embeddings(
    user: CurrentUser,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> dict[str, Any]:
    require_ai_permission(user)
    try:
        result = await asyncio.to_thread(
            ingestion_commands.grant_and_enqueue_all,
            int(user["id"]),
            idempotency_key=_idempotency_key(idempotency_key, operation="grant-all"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return {
        "message": f"Deep Search enabled; {result.created_count} ingestion job(s) queued.",
        "items": [
            {
                "file_id": item.file_id,
                "revision": item.revision,
                "job_id": item.job_id,
                "state": item.state.value,
                "created": item.created,
            }
            for item in result.items
        ],
        "created_count": result.created_count,
        "count": result.created_count,
        "skipped_unsupported": result.skipped_unsupported,
    }


@router.get("/ai-status/{file_id}")
async def get_ai_status(file_id: int, user: CurrentUser) -> dict[str, Any]:
    status = await asyncio.to_thread(ingestion_commands.status, int(user["id"]), file_id)
    if status is None:
        raise HTTPException(status_code=404, detail="File not found")
    return _status_payload(status)


async def _remove_object_refs(refs: list[StoredObjectRef]) -> int:
    semaphore = asyncio.Semaphore(8)

    async def remove_one(ref: StoredObjectRef) -> int:
        async with semaphore:
            return len(await object_access.remove(ref))

    return sum(await asyncio.gather(*(remove_one(ref) for ref in refs)))


@router.delete("/hard-delete/{file_id}")
async def hard_delete_file(file_id: int, user: CurrentUser) -> dict[str, Any]:
    if not user.get("perm_delete", False):
        raise HTTPException(
            status_code=403,
            detail="Delete permission denied. New accounts do not include file deletion "
            "by default; ask an administrator to grant Delete permission "
            "(Admin \u2192 Users \u2192 Permissions).",
        )
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            DELETE FROM files
            WHERE id = %s AND user_id = %s
            RETURNING s3_bucket_name, s3_path, s3_key_path, sha256_hash
            """,
            (file_id, int(user["id"])),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")
    failed = await object_access.remove(_object_ref(row))
    if failed:
        logger.error("Permanent file deletion left %d encrypted object(s) for cleanup", len(failed))
    return {
        "message": "File permanently deleted.",
        "file_id": file_id,
        "warning": "This action cannot be undone",
        "storage_cleanup_complete": not failed,
    }


@router.delete("/trash/empty")
async def empty_trash(user: CurrentUser) -> dict[str, Any]:
    if not user.get("perm_delete", False):
        raise HTTPException(
            status_code=403,
            detail="Delete permission denied. New accounts do not include file deletion "
            "by default; ask an administrator to grant Delete permission "
            "(Admin \u2192 Users \u2192 Permissions).",
        )
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            DELETE FROM files
            WHERE user_id = %s AND is_deleted = TRUE
            RETURNING s3_bucket_name, s3_path, s3_key_path, sha256_hash
            """,
            (int(user["id"]),),
        )
        rows = cursor.fetchall()
    refs = [_object_ref(row) for row in rows]
    failed_count = await _remove_object_refs(refs) if refs else 0
    if failed_count:
        logger.error("Trash deletion left %d encrypted object(s) for cleanup", failed_count)
    return {
        "message": "Trash emptied." if rows else "Trash is already empty.",
        "deleted_count": len(rows),
        "storage_cleanup_complete": failed_count == 0,
    }


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/create-session/{file_id}")
def create_decryption_session(
    file_id: int,
    request: Request,
    user: CurrentUser,
) -> dict[str, Any]:
    if not user.get("perm_download", True):
        raise HTTPException(status_code=403, detail="Download permission denied")
    session_token = str(uuid4())
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM decryption_sessions WHERE expires_at <= NOW()")
        cursor.execute(
            """
            INSERT INTO decryption_sessions (session_token, file_id, user_id, purpose, client_ip)
            SELECT %s, id, user_id, %s, %s
            FROM files
            WHERE id = %s AND user_id = %s AND is_deleted = FALSE
            RETURNING expires_at
            """,
            (session_token, "preview", _client_ip(request), file_id, int(user["id"])),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")
    expires_at = row["expires_at"]
    remaining = max(0, int((expires_at - datetime.now(UTC)).total_seconds()))
    return {
        "session_token": session_token,
        "expires_in_seconds": remaining,
        "preview_url": f"/api/files/preview/{session_token}",
        "message": "Temporary decryption session created.",
    }


async def _materialize_response(row: dict[str, Any]) -> Path:
    try:
        return await object_access.materialize(_object_ref(row))
    except StoredObjectUnavailable:
        raise HTTPException(status_code=404, detail="Encrypted file data is unavailable") from None
    except StoredObjectIntegrityError:
        logger.exception("Stored object integrity validation failed")
        raise HTTPException(status_code=500, detail="File integrity validation failed") from None


@router.get("/download/{file_id}")
async def download_file(
    file_id: int,
    background_tasks: BackgroundTasks,
    user: CurrentUser,
) -> FileResponse:
    if not user.get("perm_download", True):
        raise HTTPException(status_code=403, detail="Download permission denied")
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT COALESCE(NULLIF(display_name, ''), original_filename) AS filename,
                   mime_type, s3_bucket_name, s3_path, s3_key_path, sha256_hash
            FROM files
            WHERE id = %s AND user_id = %s AND is_deleted = FALSE
            """,
            (file_id, int(user["id"])),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")
    path = await _materialize_response(row)
    background_tasks.add_task(object_access.cleanup, path)
    return FileResponse(
        path,
        filename=str(row["filename"]),
        media_type=_safe_served_media_type(row["mime_type"], row["filename"]),
        background=background_tasks,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/preview/{token}")
async def preview_file(
    token: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> FileResponse:
    try:
        canonical_token = str(UUID(token))
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid or expired preview session") from None
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT COALESCE(NULLIF(f.display_name, ''), f.original_filename) AS filename,
                   f.mime_type, f.s3_bucket_name, f.s3_path, f.s3_key_path, f.sha256_hash,
                   s.client_ip
            FROM decryption_sessions AS s
            JOIN files AS f ON f.id = s.file_id AND f.user_id = s.user_id
            JOIN users AS u ON u.id = s.user_id AND u.is_active = TRUE AND u.perm_download = TRUE
            WHERE s.session_token = %s
              AND s.purpose = 'preview'
              AND s.expires_at > NOW()
              AND f.is_deleted = FALSE
            """,
            (canonical_token,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Invalid or expired preview session")
    if settings.validate_preview_ip and row["client_ip"] and row["client_ip"] != _client_ip(request):
        raise HTTPException(status_code=403, detail="Preview session client changed")
    path = await _materialize_response(row)
    background_tasks.add_task(object_access.cleanup, path)
    return FileResponse(
        path,
        filename=str(row["filename"]),
        media_type=_safe_served_media_type(row["mime_type"], row["filename"]),
        content_disposition_type="inline",
        background=background_tasks,
        headers={
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": (
                "sandbox; default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
            ),
            "Cross-Origin-Resource-Policy": "same-origin",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "SAMEORIGIN",
        },
    )


@router.get("/list")
def list_files(
    user: CurrentUser,
    show_deleted: Annotated[bool, Query()] = False,
) -> dict[str, Any]:
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, original_filename,
                   COALESCE(NULLIF(display_name, ''), original_filename) AS filename,
                   file_size_bytes, mime_type, uploaded_at, quick_tags, quick_summary,
                   doc_type, intelligence_status,
                   user_granted_ai_access, ai_status,
                   desired_revision, current_revision, ai_error_code, ai_error_detail,
                   is_deleted, deleted_at, folder_id,
                   rev.fallback_reason AS intelligence_fallback_reason
            FROM files
            LEFT JOIN LATERAL (
                SELECT dr.metadata->'intelligence'->>'fallback_reason' AS fallback_reason
                FROM document_revisions AS dr
                WHERE dr.file_id = files.id
                  AND dr.user_id = files.user_id
                  AND dr.status = 'current'
                LIMIT 1
            ) AS rev ON TRUE
            WHERE user_id = %s AND is_deleted = %s
            ORDER BY uploaded_at DESC
            """,
            (int(user["id"]), show_deleted),
        )
        rows = cursor.fetchall()
        cursor.execute(
            """
            SELECT storage_quota_bytes, storage_used_bytes
            FROM users WHERE id = %s
            """,
            (int(user["id"]),),
        )
        owner = cursor.fetchone()

    files: list[dict[str, Any]] = []
    for row in rows:
        route = route_file(str(row["original_filename"]), str(row["mime_type"] or ""))
        state = str(row["ai_status"] or "not_granted")
        if not route.supported and state == "not_granted":
            state = "unsupported"
        searchable = (
            bool(row["user_granted_ai_access"])
            and row["current_revision"] is not None
            and not bool(row["is_deleted"])
        )
        intelligence_ready = _has_generated_intelligence(row)
        files.append(
            {
                "id": row["id"],
                "filename": row["filename"],
                "original_filename": row["original_filename"],
                "size_bytes": row["file_size_bytes"],
                "mime_type": row["mime_type"],
                "uploaded_at": row["uploaded_at"].isoformat(),
                "parser_supported": bool(route.supported),
                "tags": (row["quick_tags"] or []) if intelligence_ready else [],
                "summary": row["quick_summary"] if intelligence_ready else None,
                "doc_type": row["doc_type"] if intelligence_ready else None,
                "intelligence_status": row["intelligence_status"],
                "intelligence_fallback_reason": row.get("intelligence_fallback_reason"),
                "ai_status": _compatibility_state(state),
                "ingestion_state": state,
                "ai_searchable": searchable,
                "ai_access_granted": bool(row["user_granted_ai_access"]),
                "desired_revision": row["desired_revision"],
                "current_revision": row["current_revision"],
                "ai_error_code": row["ai_error_code"],
                "ai_error_detail": row["ai_error_detail"],
                "is_deleted": row["is_deleted"],
                "deleted_at": row["deleted_at"].isoformat() if row["deleted_at"] else None,
                "folder_id": row["folder_id"],
            }
        )
    ready_count = sum(item["ai_searchable"] for item in files)
    return {
        "files": files,
        "total_files": len(files),
        "ai_ready_files": ready_count,
        "total_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "storage_used_bytes": int(owner["storage_used_bytes"] if owner else 0),
        "storage_quota_bytes": int(owner["storage_quota_bytes"] if owner else 0),
        "version": settings.app_version,
    }


@router.patch("/move/{file_id}")
def move_file(file_id: int, body: MoveFileRequest, user: CurrentUser) -> dict[str, Any]:
    if not user.get("perm_folders", True):
        raise HTTPException(status_code=403, detail="Folder move permission denied")
    with get_db() as connection:
        cursor = connection.cursor()
        _require_folder(cursor, int(user["id"]), body.folder_id)
        cursor.execute(
            """
            SELECT COALESCE(NULLIF(display_name, ''), original_filename) AS filename
            FROM files WHERE id = %s AND user_id = %s AND is_deleted = FALSE FOR UPDATE
            """,
            (file_id, int(user["id"])),
        )
        source = cursor.fetchone()
        if source is None:
            raise HTTPException(status_code=404, detail="File not found")
        cursor.execute(
            """
            SELECT id FROM files
            WHERE user_id = %s AND id <> %s AND is_deleted = FALSE
              AND folder_id IS NOT DISTINCT FROM %s
              AND COALESCE(NULLIF(display_name, ''), original_filename) = %s
            LIMIT 1
            """,
            (int(user["id"]), file_id, body.folder_id, source["filename"]),
        )
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="A file with this name already exists there")
        cursor.execute(
            "UPDATE files SET folder_id = %s WHERE id = %s AND user_id = %s",
            (body.folder_id, file_id, int(user["id"])),
        )
    return {"message": "File moved", "file_id": file_id, "folder_id": body.folder_id}


@router.patch("/rename/{file_id}")
def rename_file(file_id: int, body: RenameFileRequest, user: CurrentUser) -> dict[str, Any]:
    """Rename a file's display name (BUG-008; documented in docs/api.md).

    Display-only, mirroring the documented contract: the stored object
    name, original filename, MIME type, revision identity, consent, and
    index are untouched — parser routing keeps using the stored
    original filename and detected MIME type.
    """
    if not user.get("perm_rename", True):
        raise HTTPException(status_code=403, detail="Rename permission denied")
    try:
        display_name = sanitize_filename(body.filename)
    except UploadRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT folder_id FROM files
            WHERE id = %s AND user_id = %s AND is_deleted = FALSE FOR UPDATE
            """,
            (file_id, int(user["id"])),
        )
        source = cursor.fetchone()
        if source is None:
            raise HTTPException(status_code=404, detail="File not found")
        cursor.execute(
            """
            SELECT id FROM files
            WHERE user_id = %s AND id <> %s AND is_deleted = FALSE
              AND folder_id IS NOT DISTINCT FROM %s
              AND COALESCE(NULLIF(display_name, ''), original_filename) = %s
            LIMIT 1
            """,
            (int(user["id"]), file_id, source["folder_id"], display_name),
        )
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="A file with this name already exists there")
        cursor.execute(
            "UPDATE files SET display_name = %s WHERE id = %s AND user_id = %s",
            (display_name, file_id, int(user["id"])),
        )
    return {"message": "File renamed", "file_id": file_id, "filename": display_name}


@router.post("/copy/{file_id}")
async def copy_file(file_id: int, body: MoveFileRequest, user: CurrentUser) -> dict[str, Any]:
    if not user.get("perm_share", True):
        raise HTTPException(status_code=403, detail="Copy/share permission denied")
    user_id = int(user["id"])
    with get_db() as connection:
        cursor = connection.cursor()
        _require_folder(cursor, user_id, body.folder_id)
        cursor.execute(
            """
            SELECT original_filename, display_name, mime_type, file_size_bytes,
                   sha256_hash, s3_bucket_name, s3_path, s3_key_path
            FROM files WHERE id = %s AND user_id = %s AND is_deleted = FALSE
            """,
            (file_id, user_id),
        )
        source = cursor.fetchone()
    if source is None:
        raise HTTPException(status_code=404, detail="File not found")

    file_uuid = uuid4()
    keys = object_keys_for_file(user_id, file_uuid)
    destination = StoredObjectRef(
        bucket=settings.s3_bucket_name,
        payload_path=keys.payload,
        wrapped_key_path=keys.wrapped_key,
        sha256=str(source["sha256_hash"]).strip(),
    )
    await object_access.copy(_object_ref(source), destination)
    try:
        with get_db() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s, %s)",
                (user_id, _advisory_key(f"copy:{body.folder_id}:{source['display_name']}")),
            )
            cursor.execute(
                "SELECT storage_quota_bytes, storage_used_bytes FROM users WHERE id = %s FOR UPDATE",
                (user_id,),
            )
            owner = cursor.fetchone()
            if owner is None:
                raise HTTPException(status_code=404, detail="User not found")
            _require_folder(cursor, user_id, body.folder_id)
            cursor.execute(
                "SELECT id FROM files WHERE id = %s AND user_id = %s AND is_deleted = FALSE",
                (file_id, user_id),
            )
            if cursor.fetchone() is None:
                raise HTTPException(status_code=409, detail="Source file changed during copy")
            cursor.execute(
                """
                SELECT id FROM files
                WHERE user_id = %s AND is_deleted = FALSE
                  AND folder_id IS NOT DISTINCT FROM %s
                  AND COALESCE(NULLIF(display_name, ''), original_filename) = %s
                LIMIT 1
                """,
                (user_id, body.folder_id, source["display_name"] or source["original_filename"]),
            )
            if cursor.fetchone():
                raise HTTPException(status_code=409, detail="A file with this name already exists there")
            projected = int(owner["storage_used_bytes"]) + int(source["file_size_bytes"])
            if projected > int(owner["storage_quota_bytes"]):
                raise HTTPException(status_code=413, detail="Copy exceeds the remaining storage quota")
            cursor.execute(
                """
                INSERT INTO files (
                    uuid, user_id, original_filename, display_name, mime_type,
                    file_size_bytes, encrypted_filename, rsa_key_filename,
                    sha256_hash, s3_bucket_name, s3_path, s3_key_path,
                    folder_id, ai_status
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, 'payload.enc', 'payload.key.rsa4096',
                    %s, %s, %s, %s, %s, 'not_granted'
                )
                RETURNING id
                """,
                (
                    file_uuid,
                    user_id,
                    source["original_filename"],
                    source["display_name"],
                    source["mime_type"],
                    source["file_size_bytes"],
                    str(source["sha256_hash"]).strip(),
                    destination.bucket,
                    destination.payload_path,
                    destination.wrapped_key_path,
                    body.folder_id,
                ),
            )
            new_id = int(cursor.fetchone()["id"])
    except BaseException:
        await object_access.remove(destination)
        raise
    return {"file_id": new_id, "file_uuid": str(file_uuid), "message": "File copied"}


@router.delete("/delete/{file_id}")
async def delete_file(file_id: int, user: CurrentUser) -> dict[str, Any]:
    if not user.get("perm_delete", False):
        raise HTTPException(
            status_code=403,
            detail="Delete permission denied. New accounts do not include file deletion "
            "by default; ask an administrator to grant Delete permission "
            "(Admin \u2192 Users \u2192 Permissions).",
        )
    try:
        result = await asyncio.to_thread(ingestion_commands.trash_file, int(user["id"]), file_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found") from None
    return {
        "status": "moved_to_trash",
        "cancelled_jobs": result.cancelled_job_count,
        "index_data_preserved": True,
    }


@router.post("/restore/{file_id}")
def restore_file(file_id: int, user: CurrentUser) -> dict[str, str]:
    user_id = int(user["id"])
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT file_size_bytes, folder_id,
                   COALESCE(NULLIF(display_name, ''), original_filename) AS filename
            FROM files WHERE id = %s AND user_id = %s AND is_deleted = TRUE FOR UPDATE
            """,
            (file_id, user_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Trashed file not found")
        cursor.execute(
            "SELECT storage_quota_bytes, storage_used_bytes FROM users WHERE id = %s FOR UPDATE",
            (user_id,),
        )
        owner = cursor.fetchone()
        if int(owner["storage_used_bytes"]) + int(row["file_size_bytes"]) > int(owner["storage_quota_bytes"]):
            raise HTTPException(status_code=413, detail="Restore exceeds the remaining storage quota")
        folder_id = row["folder_id"]
        if folder_id is not None:
            cursor.execute(
                "SELECT id FROM folders WHERE id = %s AND user_id = %s AND is_deleted = FALSE",
                (folder_id, user_id),
            )
            if cursor.fetchone() is None:
                folder_id = None
        cursor.execute(
            """
            SELECT id FROM files
            WHERE user_id = %s AND is_deleted = FALSE
              AND folder_id IS NOT DISTINCT FROM %s
              AND COALESCE(NULLIF(display_name, ''), original_filename) = %s
            LIMIT 1
            """,
            (user_id, folder_id, row["filename"]),
        )
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="A file with this name already exists there")
        cursor.execute(
            """
            UPDATE files
            SET is_deleted = FALSE,
                deleted_at = NULL,
                folder_id = %s,
                ai_status = CASE
                    WHEN user_granted_ai_access = FALSE THEN 'not_granted'
                    WHEN current_revision IS NOT NULL THEN 'ready'
                    WHEN ai_status IN (
                        'queued', 'decrypting', 'converting', 'parsing',
                        'chunking', 'embedding', 'publishing'
                    ) THEN 'cancelled'
                    ELSE ai_status
                END,
                ai_error_code = CASE
                    WHEN current_revision IS NOT NULL THEN NULL
                    WHEN ai_status IN (
                        'queued', 'decrypting', 'converting', 'parsing',
                        'chunking', 'embedding', 'publishing'
                    ) THEN 'cancelled_while_trashed'
                    ELSE ai_error_code
                END,
                ai_error_detail = CASE
                    WHEN current_revision IS NOT NULL OR ai_status IN (
                        'queued', 'decrypting', 'converting', 'parsing',
                        'chunking', 'embedding', 'publishing'
                    ) THEN NULL
                    ELSE ai_error_detail
                END
            WHERE id = %s AND user_id = %s
            """,
            (folder_id, file_id, user_id),
        )
    return {"status": "restored"}
