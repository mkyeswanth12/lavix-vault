"""Session file-scope persistence — the single chokepoint for scope reads/writes.

Scope lives in ``chats.messages`` as ``{"scoped_file_ids": [...],
"scoped_folder_ids": [...]}``. Folder IDs are live: they re-expand to
current member files on every read path, so uploads join automatically.

Why this column: it is the documented legacy message store, permanently
``'[]'`` on the v2 path — every live reader (``_prepare`` history,
``_load_messages``, ``_safe_legacy_messages``, ``_legacy_history``)
tolerates a JSON object there and falls back to "no messages", so a scope
object is invisible to all of them. No migration, no backfill.

Rules enforced here (callers must not touch the column directly):
- Atomic single-key statements (``jsonb_set`` / ``-`` in SQL). Two
  scope-setting requests racing each other resolve to last-write-wins per
  key in Postgres; application-level read-modify-write is forbidden.
- History is never destroyed: writes only apply when ``messages`` is
  ``'[]'`` or already an object. A legacy non-empty message list is left
  untouched (scope silently unwritable there — pre-v2 chats behave as
  scopeless, i.e. exactly like before this feature).
- Reads never raise on shape problems: missing chat, missing key, wrong
  types, and empty lists all mean "scopeless" (``None``).

If scope ever moves to real columns, only this file changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID

SCOPE_KEY = "scoped_file_ids"
SCOPE_FOLDER_KEY = "scoped_folder_ids"
MAX_SCOPE_IDS = 50
MAX_SCOPE_FOLDERS = 20


def _coerce_ids(value: Any, *, limit: int = MAX_SCOPE_IDS) -> list[int] | None:
    """Return validated IDs or None when the stored value is unusable."""
    if not isinstance(value, list) or not value:
        return None
    cleaned: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            number = int(item)
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        cleaned.append(number)
    deduplicated = list(dict.fromkeys(cleaned))
    if not deduplicated:
        return None
    return deduplicated[:limit]


def _clean_ids(values: Sequence[int], label: str, *, limit: int) -> list[int]:
    """Strictly validate caller-supplied IDs; raise on anything dubious."""
    deduplicated: list[int] = []
    for value in list(values):
        if isinstance(value, bool):
            raise ValueError(f"invalid {label} selection")
        if isinstance(value, int):
            number = value
        elif isinstance(value, str) and value.strip().isdigit():
            number = int(value.strip())
        else:
            raise ValueError(f"invalid {label} selection")
        if number <= 0:
            raise ValueError(f"invalid {label} selection")
        if number not in deduplicated:
            deduplicated.append(number)
    if len(deduplicated) > limit:
        raise ValueError(f"invalid {label} selection")
    return deduplicated


def _normalize_chat_id(chat_id: str | None) -> str | None:
    if not chat_id:
        return None
    try:
        return str(UUID(str(chat_id)))
    except (ValueError, AttributeError, TypeError):
        return None


def scope_ids_from_messages(messages: Any, key: str) -> list[int] | None:
    """Extract one scope list from an already-fetched ``chats.messages`` value."""
    if not isinstance(messages, dict):
        return None
    limit = MAX_SCOPE_FOLDERS if key == SCOPE_FOLDER_KEY else MAX_SCOPE_IDS
    return _coerce_ids(messages.get(key), limit=limit)


def scope_from_messages(messages: Any) -> list[int] | None:
    """Extract file scope from an already-fetched ``chats.messages`` value.

    Same parsing as :func:`get_chat_scope` without the round-trip, for
    call sites that already hold the row.
    """
    return scope_ids_from_messages(messages, SCOPE_KEY)


def _read_messages(
    connection_factory: Callable[[], Any], *, user_id: int, normalized: str
) -> Any:
    with connection_factory() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT messages FROM chats WHERE id = %s AND user_id = %s",
            (normalized, user_id),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return row.get("messages") if isinstance(row, dict) else row[0]


def get_chat_scope(
    connection_factory: Callable[[], Any],
    *,
    user_id: int,
    chat_id: str | None,
) -> list[int] | None:
    """Read the session file scope. Never raises on shape problems."""
    normalized = _normalize_chat_id(chat_id)
    if normalized is None:
        return None
    return scope_from_messages(_read_messages(connection_factory, user_id=user_id, normalized=normalized))


def get_chat_folders(
    connection_factory: Callable[[], Any],
    *,
    user_id: int,
    chat_id: str | None,
) -> list[int] | None:
    """Read the session folder scope. Never raises on shape problems."""
    normalized = _normalize_chat_id(chat_id)
    if normalized is None:
        return None
    return scope_ids_from_messages(
        _read_messages(connection_factory, user_id=user_id, normalized=normalized),
        SCOPE_FOLDER_KEY,
    )


def _write_scope_key(
    cursor: Any,
    *,
    key: str,
    ids: list[int],
    normalized: str,
    user_id: int,
) -> None:
    scope_path = "{" + key + "}"
    if ids:
        cursor.execute(
            """
            UPDATE chats
            SET messages = CASE
                WHEN messages = '[]'::jsonb
                  OR jsonb_typeof(messages) = 'object'
                THEN jsonb_set(
                    CASE
                        WHEN messages = '[]'::jsonb THEN '{}'::jsonb
                        ELSE messages
                    END,
                    %s::text[],
                    to_jsonb(%s::INTEGER[]),
                    true
                )
                ELSE messages
            END
            WHERE id = %s AND user_id = %s
            """,
            (scope_path, ids, normalized, user_id),
        )
    else:
        cursor.execute(
            """
            UPDATE chats
            SET messages = CASE
                WHEN jsonb_typeof(messages) = 'object'
                THEN (messages - %s)
                ELSE messages
            END
            WHERE id = %s AND user_id = %s
            """,
            (key, normalized, user_id),
        )


def set_chat_scope(
    connection_factory: Callable[[], Any],
    *,
    user_id: int,
    chat_id: str | None,
    file_ids: Sequence[int] | None = None,
    folder_ids: Sequence[int] | None = None,
) -> None:
    """Persist session scope keys. ``None`` leaves a key alone.

    Non-empty lists overwrite that key; an empty list clears it. Callers
    with a fully null scope must not call this at all (null means "leave
    the stored scope alone"). Each key writes in its own atomic statement
    touching only that key, so concurrent writers resolve per-key.
    """
    normalized = _normalize_chat_id(chat_id)
    if normalized is None:
        if chat_id:
            raise ValueError("invalid chat id")
        return
    cleaned_files = _clean_ids(file_ids, "file", limit=MAX_SCOPE_IDS) if file_ids is not None else None
    cleaned_folders = (
        _clean_ids(folder_ids, "folder", limit=MAX_SCOPE_FOLDERS)
        if folder_ids is not None
        else None
    )
    if cleaned_files is None and cleaned_folders is None:
        return
    with connection_factory() as connection:
        cursor = connection.cursor()
        if cleaned_files is not None:
            _write_scope_key(
                cursor, key=SCOPE_KEY, ids=cleaned_files,
                normalized=normalized, user_id=user_id,
            )
        if cleaned_folders is not None:
            _write_scope_key(
                cursor, key=SCOPE_FOLDER_KEY, ids=cleaned_folders,
                normalized=normalized, user_id=user_id,
            )
        connection.commit()
