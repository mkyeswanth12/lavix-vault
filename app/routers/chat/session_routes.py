"""Durable PostgreSQL chat CRUD and history compatibility routes."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_ai_permission
from app.database import get_db

from .public_contract import sanitize_public_message
from .schemas import ChatCreateRequest, ChatPatchRequest, ChatScopeRequest
from .scope_store import (
    SCOPE_FOLDER_KEY,
    get_chat_folders,
    get_chat_scope,
    scope_from_messages,
    scope_ids_from_messages,
    set_chat_scope,
)

logger = logging.getLogger(__name__)
router = APIRouter()
CurrentUser = Annotated[dict, Depends(require_ai_permission)]

_MAX_PATCH_MESSAGES = 1_000
_PUBLIC_EXTRA_FIELDS = (
    "followups",
    "images",
    "response_type",
    "timestamp",
    "unverified",
)


def _row_dict(row: Any, columns: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    return dict(zip(columns, row, strict=False))


def _decode_json(value: Any, expected_type: type, default: Any) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return default
    return value if isinstance(value, expected_type) else default


def _normalize_public_message(value: Any) -> dict[str, Any] | None:
    return sanitize_public_message(value)


def _safe_legacy_messages(value: Any) -> list[dict[str, Any]]:
    decoded = _decode_json(value, list, [])
    messages = (
        message
        for item in decoded[-_MAX_PATCH_MESSAGES:]
        if (message := _normalize_public_message(item)) is not None
    )
    return list(messages)


def _durable_message(row: Any) -> dict[str, Any] | None:
    columns = (
        "role",
        "content",
        "content_json",
        "provider",
        "model",
        "tool_name",
        "tool_call_id",
        "sources",
        "prompt_tokens",
        "completion_tokens",
        "created_at",
    )
    data = _row_dict(row, columns)
    created_at = data.get("created_at")
    if created_at is not None:
        created_at = created_at.isoformat()
    extra = _decode_json(data.get("content_json"), dict, {})
    public = _normalize_public_message(
        {
            **extra,
            "role": data.get("role"),
            "content": data.get("content"),
            "provider": data.get("provider"),
            "model": data.get("model"),
            "tool_name": data.get("tool_name"),
            "tool_call_id": data.get("tool_call_id"),
            "sources": _decode_json(data.get("sources"), list, []),
            "usage": {
                "prompt_tokens": data.get("prompt_tokens"),
                "completion_tokens": data.get("completion_tokens"),
            },
            "created_at": created_at,
        }
    )
    if public is None:
        return None
    public.setdefault("sources", [])
    return public


def _load_messages(cursor: Any, *, chat_id: str, user_id: int, legacy: Any) -> list[dict[str, Any]]:
    cursor.execute(
        """
        SELECT role, content, content_json, provider, model, tool_name,
               tool_call_id, sources, prompt_tokens, completion_tokens,
               created_at
        FROM chat_messages
        WHERE chat_id = %s AND user_id = %s
        ORDER BY sequence_number ASC
        """,
        (chat_id, user_id),
    )
    messages = [message for row in cursor.fetchall() if (message := _durable_message(row))]
    return messages if messages else _safe_legacy_messages(legacy)


def _chat_id_from_session(session_id: str) -> str | None:
    candidates = [session_id]
    if "_pgchat_" in session_id:
        candidates.insert(0, session_id.rsplit("_pgchat_", 1)[-1])
    for candidate in candidates:
        try:
            return str(UUID(candidate))
        except (TypeError, ValueError, AttributeError):
            continue
    return None


def _find_history_chat(cursor: Any, *, session_id: str | None, user_id: int) -> dict[str, Any] | None:
    if session_id is not None:
        chat_id = _chat_id_from_session(session_id)
        if chat_id is None:
            return None
        cursor.execute(
            """
            SELECT id, messages
            FROM chats
            WHERE id = %s AND user_id = %s AND archived = FALSE
            """,
            (chat_id, user_id),
        )
    else:
        cursor.execute(
            """
            SELECT id, messages
            FROM chats
            WHERE user_id = %s AND archived = FALSE
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
    row = cursor.fetchone()
    return _row_dict(row, ("id", "messages")) if row else None


def _clear_history(cursor: Any, *, session_id: str | None, user_id: int) -> bool:
    chat = _find_history_chat(cursor, session_id=session_id, user_id=user_id)
    if chat is None:
        return False
    chat_id = str(chat["id"])
    cursor.execute(
        "DELETE FROM chat_messages WHERE chat_id = %s AND user_id = %s",
        (chat_id, user_id),
    )
    cursor.execute(
        """
        UPDATE chats
        SET messages = '[]'::jsonb, updated_at = CURRENT_TIMESTAMP
        WHERE id = %s AND user_id = %s
        """,
        (chat_id, user_id),
    )
    return True


def _patch_messages(cursor: Any, *, chat_id: str, user_id: int, values: list[dict]) -> list[dict[str, Any]]:
    if len(values) > _MAX_PATCH_MESSAGES:
        raise HTTPException(status_code=422, detail="Too many chat messages")

    messages: list[dict[str, Any]] = []
    for value in values:
        message = _normalize_public_message(value)
        if message is None:
            raise HTTPException(status_code=422, detail="Invalid chat message")
        messages.append(message)

    cursor.execute(
        "DELETE FROM chat_messages WHERE chat_id = %s AND user_id = %s",
        (chat_id, user_id),
    )
    for sequence_number, message in enumerate(messages):
        sources = message.get("sources", [])
        usage = message.get("usage") if isinstance(message.get("usage"), Mapping) else {}
        content_json = {field: message[field] for field in _PUBLIC_EXTRA_FIELDS if field in message}
        cursor.execute(
            """
            INSERT INTO chat_messages (
                chat_id, user_id, sequence_number, role, content, content_json,
                provider, model, tool_name, tool_call_id, sources,
                prompt_tokens, completion_tokens
            )
            VALUES (
                %s, %s, %s, %s, %s, %s::jsonb,
                %s, %s, %s, %s, %s::jsonb, %s, %s
            )
            """,
            (
                chat_id,
                user_id,
                sequence_number,
                message["role"],
                message["content"],
                json.dumps(content_json, separators=(",", ":")),
                message.get("provider"),
                message.get("model"),
                message.get("tool_name"),
                message.get("tool_call_id"),
                json.dumps(sources, separators=(",", ":")),
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            ),
        )
    return messages


@router.get("/chats")
async def list_chats(user: CurrentUser):
    """List the user's durable, non-archived chats."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, title, updated_at, pinned, created_at
                FROM chats
                WHERE user_id = %s AND archived = FALSE
                ORDER BY pinned DESC, updated_at DESC
                LIMIT 100
                """,
                (user["id"],),
            )
            rows = cur.fetchall()
        return {"chats": [dict(row) for row in rows]}
    except Exception as exc:
        logger.exception("list_chats failed")
        raise HTTPException(status_code=500, detail="Failed to list chats") from exc


@router.post("/chats")
async def create_chat(body: ChatCreateRequest, user: CurrentUser):
    """Create a durable chat."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO chats (user_id, title) VALUES (%s, %s)
                RETURNING id, title, created_at, updated_at
                """,
                (user["id"], body.title or "New Chat"),
            )
            row = cur.fetchone()
        return dict(row)
    except Exception as exc:
        logger.exception("create_chat failed")
        raise HTTPException(status_code=500, detail="Failed to create chat") from exc


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str, user: CurrentUser):
    """Return one chat with v2 messages, falling back to safe legacy JSONB."""
    normalized_id = _chat_id_from_session(chat_id)
    if normalized_id is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, title, messages, created_at, updated_at, pinned
                FROM chats WHERE id = %s AND user_id = %s
                """,
                (normalized_id, user["id"]),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Chat not found")
            chat = _row_dict(
                row,
                ("id", "title", "messages", "created_at", "updated_at", "pinned"),
            )
            chat["messages"] = _load_messages(
                cur,
                chat_id=normalized_id,
                user_id=user["id"],
                legacy=chat.get("messages"),
            )
            # Session file scope rides in the legacy messages column (see
            # scope_store); read it from the raw row before projection.
            raw_messages = row.get("messages") if isinstance(row, dict) else None
            chat["scoped_file_ids"] = scope_from_messages(raw_messages) or []
            chat["scoped_folder_ids"] = (
                scope_ids_from_messages(raw_messages, SCOPE_FOLDER_KEY) or []
            )
        return chat
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("get_chat failed")
        raise HTTPException(status_code=500, detail="Failed to get chat") from exc


@router.patch("/chats/{chat_id}")
async def patch_chat(
    chat_id: str,
    body: ChatPatchRequest,
    user: CurrentUser,
):
    """Update chat metadata or replace its durable message sequence."""
    normalized_id = _chat_id_from_session(chat_id)
    if normalized_id is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, title FROM chats WHERE id = %s AND user_id = %s FOR UPDATE",
                (normalized_id, user["id"]),
            )
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Chat not found")
            current = _row_dict(existing, ("id", "title"))

            updates: list[str] = []
            params: list[Any] = []
            replacement: list[dict[str, Any]] | None = None
            if body.messages is not None:
                replacement = _patch_messages(
                    cur,
                    chat_id=normalized_id,
                    user_id=user["id"],
                    values=body.messages,
                )
                updates.append("messages = '[]'::jsonb")

            if body.title is not None:
                updates.append("title = %s")
                params.append(body.title)
            elif current.get("title") == "New Chat" and replacement:
                first_user_message = next(
                    (message["content"] for message in replacement if message["role"] == "user"),
                    None,
                )
                if first_user_message:
                    updates.append("title = %s")
                    params.append(first_user_message[:50])
            if body.pinned is not None:
                updates.append("pinned = %s")
                params.append(body.pinned)

            if not updates:
                return {"id": normalized_id}

            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.extend((normalized_id, user["id"]))
            cur.execute(
                f"UPDATE chats SET {', '.join(updates)} "
                "WHERE id = %s AND user_id = %s RETURNING id, title, updated_at",
                params,
            )
            row = cur.fetchone()
        return dict(row)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("patch_chat failed")
        raise HTTPException(status_code=500, detail="Failed to update chat") from exc


@router.post("/chats/{chat_id}/scope")
async def set_chat_file_scope(chat_id: str, body: ChatScopeRequest, user: CurrentUser):
    """Replace the chat's session file scope (explicit list overwrites)."""
    normalized_id = _chat_id_from_session(chat_id)
    if normalized_id is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM chats WHERE id = %s AND user_id = %s",
                (normalized_id, user["id"]),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Chat not found")
        set_chat_scope(
            get_db,
            user_id=user["id"],
            chat_id=normalized_id,
            file_ids=body.file_ids,
            folder_ids=body.folder_ids,
        )
        return {
            "id": normalized_id,
            "scoped_file_ids": get_chat_scope(
                get_db, user_id=user["id"], chat_id=normalized_id
            ) or [],
            "scoped_folder_ids": get_chat_folders(
                get_db, user_id=user["id"], chat_id=normalized_id
            ) or [],
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc) or "Invalid scope") from exc
    except Exception as exc:
        logger.exception("set_chat_file_scope failed")
        raise HTTPException(status_code=500, detail="Failed to update scope") from exc


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, user: CurrentUser):
    """Delete a chat; its durable messages cascade with it."""
    normalized_id = _chat_id_from_session(chat_id)
    if normalized_id is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM chats WHERE id = %s AND user_id = %s",
                (normalized_id, user["id"]),
            )
        return {"status": "deleted"}
    except Exception as exc:
        logger.exception("delete_chat failed")
        raise HTTPException(status_code=500, detail="Failed to delete chat") from exc


@router.get("/chat/sessions")
async def get_chat_sessions(user: CurrentUser):
    """Compatibility view of durable chats using the legacy sessions shape."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, title, updated_at
                FROM chats
                WHERE user_id = %s AND archived = FALSE
                ORDER BY updated_at DESC
                LIMIT 100
                """,
                (user["id"],),
            )
            sessions = [dict(row) for row in cur.fetchall()]
        return {"sessions": sessions}
    except Exception as exc:
        logger.exception("get_chat_sessions failed")
        raise HTTPException(status_code=500, detail="Failed to list chat sessions") from exc


@router.get("/chat/history")
async def get_chat_history(
    user: CurrentUser,
    session_id: str | None = None,
):
    """Return durable history for a chat ID or old ``_pgchat_`` session ID."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            chat = _find_history_chat(cur, session_id=session_id, user_id=user["id"])
            if chat is None:
                return {"history": []}
            history = _load_messages(
                cur,
                chat_id=str(chat["id"]),
                user_id=user["id"],
                legacy=chat.get("messages"),
            )
        return {"history": history}
    except Exception as exc:
        logger.exception("get_chat_history failed")
        raise HTTPException(status_code=500, detail="Failed to load chat history") from exc


@router.delete("/chat/history")
@router.delete("/chat/history/{session_id}")
async def delete_chat_history(
    user: CurrentUser,
    session_id: str | None = None,
):
    """Clear only this chat's durable and legacy message history."""
    try:
        with get_db() as conn:
            cleared = _clear_history(conn.cursor(), session_id=session_id, user_id=user["id"])
        return {
            "status": "success",
            "message": "Session deleted",
            "history_cleared": cleared,
        }
    except Exception as exc:
        logger.exception("delete_chat_history failed")
        raise HTTPException(status_code=500, detail="Failed to delete chat history") from exc
