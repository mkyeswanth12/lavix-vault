"""Small, explicit PostgreSQL-backed user memory surface."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Annotated, Any, Protocol

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.auth import require_ai_permission
from app.database import get_db
from app.graph_memory.repository import (
    GraphMemoryConflictError,
    GraphMemoryDisabledError,
    GraphMemoryNotFoundError,
    GraphMemoryValidationError,
)
from app.graph_memory.service import GraphMemoryService

router = APIRouter()

MAX_MEMORY_ITEMS = 20
MAX_MEMORY_TEXT_LENGTH = 500
MAX_RUNTIME_MEMORY_ITEMS = 6
MAX_RUNTIME_MEMORY_CHARS = 1_800

CurrentUser = Annotated[dict[str, Any], Depends(require_ai_permission)]


def get_graph_memory_service() -> GraphMemoryService:
    return GraphMemoryService()


MemoryService = Annotated[GraphMemoryService, Depends(get_graph_memory_service)]


class MemoryEnabledRequest(BaseModel):
    enabled: bool


class MemoryItemRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_MEMORY_TEXT_LENGTH)

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return " ".join(value.split())
        return value


def _public_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "text": str(row["memory_text"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@router.get("/memory")
def get_memory(user: CurrentUser) -> dict[str, Any]:
    """Return the caller's opt-in state and explicitly saved items."""

    user_id = int(user["id"])
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT memory_enabled FROM users WHERE id = %s", (user_id,))
        state = cursor.fetchone()
        if not state:
            raise HTTPException(status_code=404, detail="User not found")
        cursor.execute(
            """
            SELECT id, memory_text, created_at, updated_at
            FROM user_memories
            WHERE user_id = %s
            ORDER BY updated_at DESC, id DESC
            LIMIT %s
            """,
            (user_id, MAX_MEMORY_ITEMS),
        )
        items = [_public_item(row) for row in cursor.fetchall()]
    return {
        "enabled": bool(state["memory_enabled"]),
        "items": items,
        "max_items": MAX_MEMORY_ITEMS,
        "max_text_length": MAX_MEMORY_TEXT_LENGTH,
    }


@router.put("/memory")
def set_memory_enabled(
    body: MemoryEnabledRequest,
    user: CurrentUser,
    service: MemoryService,
) -> dict[str, bool]:
    """Compatibility route backed by the authoritative graph consent write."""

    try:
        status = service.update_settings(
            int(user["id"]),
            enabled=body.enabled,
            # The compatibility route has no retention control. ``None``
            # preserves the current choice atomically (or uses the default for
            # a tenant created by this request).
            retention_days=None,
            expected_revision=None,
        )
    except GraphMemoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (GraphMemoryConflictError, GraphMemoryDisabledError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GraphMemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"enabled": bool(status.enabled)}


@router.post("/memory", status_code=201)
def add_memory(body: MemoryItemRequest, user: CurrentUser) -> dict[str, Any]:
    """Add one user-authored item; never infer or extract memories from chat."""

    user_id = int(user["id"])
    with get_db() as connection:
        cursor = connection.cursor()
        # Serialize count checks for this user so concurrent requests cannot
        # exceed the fixed cap.
        cursor.execute(
            """
            SELECT users.memory_enabled,
                   tenant.enabled AS graph_memory_enabled
            FROM users
            LEFT JOIN graph_memory_tenants AS tenant ON tenant.user_id = users.id
            WHERE users.id = %s
            FOR UPDATE OF users
            """,
            (user_id,),
        )
        state = cursor.fetchone()
        if not state:
            raise HTTPException(status_code=404, detail="User not found")
        if not state["memory_enabled"] or not state["graph_memory_enabled"]:
            raise HTTPException(status_code=409, detail="Enable memory before adding an item")
        cursor.execute("SELECT COUNT(*) AS count FROM user_memories WHERE user_id = %s", (user_id,))
        if int(cursor.fetchone()["count"]) >= MAX_MEMORY_ITEMS:
            raise HTTPException(
                status_code=409,
                detail=f"Memory is limited to {MAX_MEMORY_ITEMS} items",
            )
        cursor.execute(
            """
            INSERT INTO user_memories (user_id, memory_text)
            VALUES (%s, %s)
            ON CONFLICT (user_id, memory_text) DO NOTHING
            RETURNING id, memory_text, created_at, updated_at
            """,
            (user_id, body.text),
        )
        item = cursor.fetchone()
        if not item:
            raise HTTPException(status_code=409, detail="Memory item already exists")
    return {"item": _public_item(item)}


@router.delete("/memory")
def clear_memory(user: CurrentUser) -> dict[str, int]:
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM user_memories WHERE user_id = %s", (int(user["id"]),))
        deleted = max(0, int(cursor.rowcount))
    return {"deleted_count": deleted}


@router.delete("/memory/{memory_id}")
def delete_memory_item(memory_id: int, user: CurrentUser) -> dict[str, int]:
    if memory_id <= 0:
        raise HTTPException(status_code=404, detail="Memory item not found")
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "DELETE FROM user_memories WHERE id = %s AND user_id = %s",
            (memory_id, int(user["id"])),
        )
        if cursor.rowcount != 1:
            raise HTTPException(status_code=404, detail="Memory item not found")
    return {"deleted_id": memory_id}


class RuntimeMemoryReader(Protocol):
    async def for_runtime(self, user_id: int) -> Sequence[str]: ...


class PostgresRuntimeMemoryReader:
    """Read a small set after the authenticated user-level opt-in check."""

    async def for_runtime(self, user_id: int) -> tuple[str, ...]:
        return await asyncio.to_thread(self._for_runtime, user_id)

    @staticmethod
    def _for_runtime(user_id: int) -> tuple[str, ...]:
        with get_db() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT m.memory_text
                FROM user_memories AS m
                JOIN users AS u ON u.id = m.user_id
                JOIN graph_memory_tenants AS tenant ON tenant.user_id = m.user_id
                WHERE m.user_id = %s
                  AND u.memory_enabled = TRUE
                  AND tenant.enabled = TRUE
                ORDER BY m.updated_at DESC, m.id DESC
                LIMIT %s
                """,
                (user_id, MAX_RUNTIME_MEMORY_ITEMS),
            )
            items: list[str] = []
            remaining = MAX_RUNTIME_MEMORY_CHARS
            for row in cursor.fetchall():
                value = str(row["memory_text"]).strip()
                if not value or len(value) > remaining:
                    continue
                items.append(value)
                remaining -= len(value)
            return tuple(items)
