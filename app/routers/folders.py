"""Tenant-scoped folder hierarchy and lifecycle endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import get_current_user
from app.database import get_db

router = APIRouter()
CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]

_ACTIVE_INGESTION_STATES = (
    "queued",
    "decrypting",
    "converting",
    "parsing",
    "chunking",
    "embedding",
    "publishing",
)


class FolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    parent_id: int | None = Field(default=None, gt=0)


class FolderRename(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class FolderMove(BaseModel):
    parent_id: int | None = Field(default=None, gt=0)


def _folder_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Folder name cannot be blank")
    return name


def _require_permission(user: dict[str, Any], permission: str, detail: str) -> None:
    if not user.get(permission, True):
        raise HTTPException(status_code=403, detail=detail)


def _require_active_folder(cursor: Any, folder_id: int, user_id: int) -> dict[str, Any]:
    cursor.execute(
        """
        SELECT id, name, parent_id
        FROM folders
        WHERE id = %s AND user_id = %s AND is_deleted = FALSE
        FOR UPDATE
        """,
        (folder_id, user_id),
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Folder not found")
    return row


def _name_conflicts(
    cursor: Any,
    *,
    user_id: int,
    name: str,
    parent_id: int | None,
    exclude_id: int | None = None,
) -> bool:
    exclude_clause = ""
    params: tuple[Any, ...] = (user_id, name, parent_id)
    if exclude_id is not None:
        exclude_clause = "AND id <> %s"
        params += (exclude_id,)
    cursor.execute(
        f"""
        SELECT id
        FROM folders
        WHERE user_id = %s
          AND name = %s
          AND parent_id IS NOT DISTINCT FROM %s
          AND is_deleted = FALSE
          {exclude_clause}
        LIMIT 1
        """,
        params,
    )
    return cursor.fetchone() is not None


@router.get("/folders")
def list_folders(user: CurrentUser) -> list[dict[str, Any]]:
    """Return active folders with recursive file counts and byte totals."""

    user_id = int(user["id"])
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            WITH RECURSIVE folder_tree AS (
                SELECT id, id AS root_id
                FROM folders
                WHERE user_id = %s AND is_deleted = FALSE
                UNION ALL
                SELECT child.id, tree.root_id
                FROM folders AS child
                JOIN folder_tree AS tree ON child.parent_id = tree.id
                WHERE child.user_id = %s AND child.is_deleted = FALSE
            )
            SELECT folder.id, folder.name, folder.parent_id, folder.created_at,
                   COUNT(file.id) AS file_count,
                   COALESCE(SUM(file.file_size_bytes), 0) AS total_size_bytes
            FROM folders AS folder
            LEFT JOIN folder_tree AS tree ON tree.root_id = folder.id
            LEFT JOIN files AS file
              ON file.folder_id = tree.id
             AND file.user_id = %s
             AND file.is_deleted = FALSE
            WHERE folder.user_id = %s AND folder.is_deleted = FALSE
            GROUP BY folder.id, folder.name, folder.parent_id, folder.created_at
            ORDER BY folder.name ASC
            """,
            (user_id, user_id, user_id, user_id),
        )
        rows = cursor.fetchall()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "parent_id": row["parent_id"],
            "file_count": row["file_count"],
            "total_size_bytes": row["total_size_bytes"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


@router.get("/folders/trash")
def list_trash_folders(user: CurrentUser) -> list[dict[str, Any]]:
    user_id = int(user["id"])
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, name, parent_id, deleted_at, created_at
            FROM folders
            WHERE user_id = %s AND is_deleted = TRUE
            ORDER BY deleted_at DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "parent_id": row["parent_id"],
            "deleted_at": row["deleted_at"].isoformat() if row["deleted_at"] else None,
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


@router.post("/folders")
def create_folder(body: FolderCreate, user: CurrentUser) -> dict[str, Any]:
    _require_permission(user, "perm_folders", "Folder creation permission denied")
    user_id = int(user["id"])
    name = _folder_name(body.name)

    with get_db() as connection:
        cursor = connection.cursor()
        if body.parent_id is not None:
            _require_active_folder(cursor, body.parent_id, user_id)
        if _name_conflicts(cursor, user_id=user_id, name=name, parent_id=body.parent_id):
            raise HTTPException(status_code=409, detail=f"Folder '{name}' already exists here")

        cursor.execute(
            """
            SELECT id
            FROM folders
            WHERE user_id = %s
              AND name = %s
              AND parent_id IS NOT DISTINCT FROM %s
              AND is_deleted = TRUE
            ORDER BY deleted_at DESC NULLS LAST
            LIMIT 1
            FOR UPDATE
            """,
            (user_id, name, body.parent_id),
        )
        deleted = cursor.fetchone()
        if deleted is None:
            cursor.execute(
                """
                INSERT INTO folders (user_id, name, parent_id)
                VALUES (%s, %s, %s)
                RETURNING id, name, parent_id, created_at
                """,
                (user_id, name, body.parent_id),
            )
        else:
            cursor.execute(
                """
                UPDATE folders
                SET is_deleted = FALSE, deleted_at = NULL
                WHERE id = %s AND user_id = %s
                RETURNING id, name, parent_id, created_at
                """,
                (deleted["id"], user_id),
            )
        row = cursor.fetchone()

    return {
        "id": row["id"],
        "name": row["name"],
        "parent_id": row["parent_id"],
        "created_at": row["created_at"].isoformat(),
    }


@router.put("/folders/{folder_id}")
def rename_folder(folder_id: int, body: FolderRename, user: CurrentUser) -> dict[str, Any]:
    _require_permission(user, "perm_rename", "Rename permission denied")
    user_id = int(user["id"])
    name = _folder_name(body.name)

    with get_db() as connection:
        cursor = connection.cursor()
        folder = _require_active_folder(cursor, folder_id, user_id)
        if _name_conflicts(
            cursor,
            user_id=user_id,
            name=name,
            parent_id=folder["parent_id"],
            exclude_id=folder_id,
        ):
            raise HTTPException(status_code=409, detail=f"Folder '{name}' already exists here")
        cursor.execute(
            """
            UPDATE folders SET name = %s
            WHERE id = %s AND user_id = %s AND is_deleted = FALSE
            RETURNING id, name, parent_id
            """,
            (name, folder_id, user_id),
        )
        row = cursor.fetchone()

    return {"id": row["id"], "name": row["name"], "parent_id": row["parent_id"]}


@router.patch("/folders/{folder_id}/move")
def move_folder(folder_id: int, body: FolderMove, user: CurrentUser) -> dict[str, Any]:
    _require_permission(user, "perm_folders", "Folder move permission denied")
    user_id = int(user["id"])

    with get_db() as connection:
        cursor = connection.cursor()
        folder = _require_active_folder(cursor, folder_id, user_id)
        if body.parent_id == folder_id:
            raise HTTPException(status_code=400, detail="Cannot move a folder into itself")
        if body.parent_id is not None:
            _require_active_folder(cursor, body.parent_id, user_id)
            cursor.execute(
                """
                WITH RECURSIVE descendants AS (
                    SELECT id FROM folders WHERE parent_id = %s AND user_id = %s
                    UNION ALL
                    SELECT child.id
                    FROM folders AS child
                    JOIN descendants AS tree ON child.parent_id = tree.id
                    WHERE child.user_id = %s
                )
                SELECT 1 FROM descendants WHERE id = %s LIMIT 1
                """,
                (folder_id, user_id, user_id, body.parent_id),
            )
            if cursor.fetchone() is not None:
                raise HTTPException(status_code=400, detail="Cannot move a folder into its own subfolder")
        if _name_conflicts(
            cursor,
            user_id=user_id,
            name=str(folder["name"]),
            parent_id=body.parent_id,
            exclude_id=folder_id,
        ):
            raise HTTPException(status_code=409, detail="A folder with this name already exists there")
        cursor.execute(
            """
            UPDATE folders SET parent_id = %s
            WHERE id = %s AND user_id = %s AND is_deleted = FALSE
            RETURNING id, name, parent_id
            """,
            (body.parent_id, folder_id, user_id),
        )
        row = cursor.fetchone()

    return {"id": row["id"], "name": row["name"], "parent_id": row["parent_id"]}


@router.delete("/folders/{folder_id}")
def delete_folder(folder_id: int, user: CurrentUser) -> dict[str, Any]:
    """Trash a tree, cancelling work while preserving published index data."""

    _require_permission(user, "perm_folders", "Folder delete permission denied")
    user_id = int(user["id"])
    with get_db() as connection:
        cursor = connection.cursor()
        _require_active_folder(cursor, folder_id, user_id)
        cursor.execute(
            """
            WITH RECURSIVE descendants AS (
                SELECT id
                FROM folders
                WHERE id = %s AND user_id = %s AND is_deleted = FALSE
                UNION ALL
                SELECT child.id
                FROM folders AS child
                JOIN descendants AS tree ON child.parent_id = tree.id
                WHERE child.user_id = %s AND child.is_deleted = FALSE
            )
            SELECT id FROM descendants
            """,
            (folder_id, user_id, user_id),
        )
        folder_ids = [int(row["id"]) for row in cursor.fetchall()]
        cursor.execute(
            "SELECT id FROM folders WHERE user_id = %s AND id = ANY(%s) FOR UPDATE",
            (user_id, folder_ids),
        )
        cursor.fetchall()
        cursor.execute(
            "SELECT id FROM files WHERE user_id = %s AND folder_id = ANY(%s) FOR UPDATE",
            (user_id, folder_ids),
        )
        file_ids = [int(row["id"]) for row in cursor.fetchall()]

        cancelled_jobs = 0
        if file_ids:
            cursor.execute(
                """
                UPDATE ingestion_jobs
                SET state = 'cancelled',
                    cancel_requested_at = COALESCE(cancel_requested_at, NOW()),
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    error_code = 'folder_trashed',
                    error_detail = NULL,
                    finished_at = COALESCE(finished_at, NOW()),
                    updated_at = NOW()
                WHERE user_id = %s
                  AND file_id = ANY(%s)
                  AND state = ANY(%s)
                """,
                (user_id, file_ids, list(_ACTIVE_INGESTION_STATES)),
            )
            cancelled_jobs = cursor.rowcount
            cursor.execute(
                """
                UPDATE files
                SET ai_status = CASE
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
                        ) THEN 'folder_trashed'
                        ELSE ai_error_code
                    END,
                    ai_error_detail = CASE
                        WHEN current_revision IS NOT NULL OR ai_status IN (
                            'queued', 'decrypting', 'converting', 'parsing',
                            'chunking', 'embedding', 'publishing'
                        ) THEN NULL
                        ELSE ai_error_detail
                    END,
                    is_deleted = TRUE,
                    deleted_at = COALESCE(deleted_at, NOW())
                WHERE user_id = %s AND id = ANY(%s)
                """,
                (user_id, file_ids),
            )

        cursor.execute(
            """
            UPDATE folders
            SET is_deleted = TRUE, deleted_at = COALESCE(deleted_at, NOW())
            WHERE user_id = %s AND id = ANY(%s)
            """,
            (user_id, folder_ids),
        )

    return {
        "message": "Folder moved to trash",
        "folder_ids": folder_ids,
        "file_count": len(file_ids),
        "cancelled_jobs": cancelled_jobs,
        "removed_revisions": 0,
        "index_data_preserved": True,
    }


@router.post("/folders/{folder_id}/restore")
def restore_folder(folder_id: int, user: CurrentUser) -> dict[str, Any]:
    """Restore one folder; its files and deleted descendants remain in trash."""

    _require_permission(user, "perm_folders", "Folder restore permission denied")
    user_id = int(user["id"])
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, name, parent_id
            FROM folders
            WHERE id = %s AND user_id = %s AND is_deleted = TRUE
            FOR UPDATE
            """,
            (folder_id, user_id),
        )
        folder = cursor.fetchone()
        if folder is None:
            raise HTTPException(status_code=404, detail="Trashed folder not found")

        parent_id = folder["parent_id"]
        if parent_id is not None:
            cursor.execute(
                "SELECT id FROM folders WHERE id = %s AND user_id = %s AND is_deleted = FALSE",
                (parent_id, user_id),
            )
            if cursor.fetchone() is None:
                parent_id = None
        if _name_conflicts(
            cursor,
            user_id=user_id,
            name=str(folder["name"]),
            parent_id=parent_id,
            exclude_id=folder_id,
        ):
            raise HTTPException(
                status_code=409,
                detail=f"A folder named '{folder['name']}' already exists at the restore location",
            )
        cursor.execute(
            """
            UPDATE folders
            SET is_deleted = FALSE, deleted_at = NULL, parent_id = %s
            WHERE id = %s AND user_id = %s
            RETURNING id, name, parent_id, created_at
            """,
            (parent_id, folder_id, user_id),
        )
        restored = cursor.fetchone()

    return {
        "id": restored["id"],
        "name": restored["name"],
        "parent_id": restored["parent_id"],
        "created_at": restored["created_at"].isoformat(),
    }


@router.delete("/folders/{folder_id}/hard")
def hard_delete_folder(folder_id: int, user: CurrentUser) -> dict[str, str]:
    """Remove a trashed folder tree; file rows remain in trash with no folder."""

    _require_permission(user, "perm_folders", "Folder delete permission denied")
    user_id = int(user["id"])
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            DELETE FROM folders
            WHERE id = %s AND user_id = %s AND is_deleted = TRUE
            RETURNING id
            """,
            (folder_id, user_id),
        )
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Trashed folder not found")
    return {"message": "Folder permanently deleted"}
