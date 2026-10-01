"""Streaming, quota-fenced encrypted file upload endpoint."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from minio import Minio

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.routers.folders import _folder_name
from app.storage.service import UploadConflict, VaultStorageService
from app.storage.upload import QuotaExceeded, UploadRejected, UploadTooLarge

logger = logging.getLogger(__name__)
router = APIRouter()

minio_client = Minio(
    settings.s3_endpoint.removeprefix("https://").removeprefix("http://"),
    access_key=settings.s3_access_key,
    secret_key=settings.s3_secret_key,
    secure=settings.s3_secure,
)
storage_service = VaultStorageService(minio_client)


def _get_or_create_folder_id(user_id: int, name: str, user: dict[str, Any]) -> int:
    """Get-or-create a top-level folder by name. Never duplicates.

    Reuses the folders router's normalization and conflict rules: an
    active same-name folder wins, a soft-deleted one is restored, else a
    row is inserted. Raises 403 without folder permission, mirroring
    manual creation.
    """
    if not user.get("perm_folders", True):
        raise HTTPException(status_code=403, detail="Folder creation permission denied")
    clean = _folder_name(name)
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id
            FROM folders
            WHERE user_id = %s AND name = %s AND parent_id IS NULL
              AND is_deleted = FALSE
            LIMIT 1
            """,
            (user_id, clean),
        )
        row = cursor.fetchone()
        if row is not None:
            return int(row["id"])
        cursor.execute(
            """
            UPDATE folders
            SET is_deleted = FALSE, deleted_at = NULL
            WHERE user_id = %s AND name = %s AND parent_id IS NULL
              AND is_deleted = TRUE
            RETURNING id
            """,
            (user_id, clean),
        )
        restored = cursor.fetchone()
        if restored is not None:
            return int(restored["id"])
        cursor.execute(
            """
            INSERT INTO folders (user_id, name, parent_id)
            VALUES (%s, %s, NULL)
            RETURNING id
            """,
            (user_id, clean),
        )
        created = cursor.fetchone()
        return int(created["id"])


@router.post("/upload")
async def upload_file(
    file: Annotated[UploadFile, File()],
    user: Annotated[dict, Depends(get_current_user)],
    replace: Annotated[bool, Form()] = False,
    folder_id: Annotated[int | None, Form()] = None,
    folder_name: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    if not user.get("perm_upload", True):
        raise HTTPException(status_code=403, detail="Upload permission denied")
    if folder_id is None and folder_name and folder_name.strip():
        # Server-side auto-provision (chat paste flow): get-or-create a
        # top-level folder by name so callers never manage folder ids.
        # Same permission bar as manual folder creation.
        folder_id = _get_or_create_folder_id(int(user["id"]), folder_name, user)
    try:
        stored = await storage_service.upload(
            file,
            user_id=int(user["id"]),
            filename=file.filename or "",
            declared_mime_type=file.content_type,
            folder_id=folder_id,
            replace=replace,
        )
    except UploadConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "existing_id": exc.existing_id,
                "filename": file.filename,
            },
        ) from None
    except UploadTooLarge:
        raise HTTPException(status_code=413, detail="File exceeds the configured size limit") from None
    except QuotaExceeded:
        raise HTTPException(status_code=413, detail="File exceeds the remaining storage quota") from None
    except UploadRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except Exception as exc:
        logger.exception("Encrypted upload failed with %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Encrypted upload failed") from None
    finally:
        await file.close()

    return {
        "file_id": stored.file_id,
        "file_uuid": str(stored.file_uuid),
        "filename": stored.filename,
        "tags": list(stored.tags),
        "summary": None,
        "intelligence_status": "pending",
        "size_bytes": stored.size_bytes,
        "mime_type": stored.mime_type,
        "message": "File uploaded and secured.",
    }
