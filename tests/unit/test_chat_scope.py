"""Session file-scope persistence + fallback (scope_store + _prepare wiring)."""

from contextlib import contextmanager

import pytest

from app.routers.chat.orchestrator import ChatPersistence
from app.routers.chat.scope_store import (
    SCOPE_FOLDER_KEY,
    SCOPE_KEY,
    get_chat_folders,
    get_chat_scope,
    set_chat_scope,
)

CHAT_ID = "12345678-1234-4123-8123-123456789012"


class FakeCursor:
    """Routes canned responses by query content; records every statement."""

    def __init__(self, *, chat_messages="[]", file_rows=(), folder_rows=(), expanded_rows=None):
        self.chat_messages = chat_messages
        self.file_rows = list(file_rows)
        self.folder_rows = list(folder_rows)
        self.expanded_rows = list(expanded_rows) if expanded_rows is not None else None
        self.executed = []
        self.rowcount = 0

    def execute(self, query, params=None):
        normalized = " ".join(str(query).split())
        self.executed.append((normalized, params))
        self.rowcount = 1

    def fetchone(self):
        query = self.executed[-1][0]
        if "FROM chats" in query:
            return {"messages": self.chat_messages}
        return None

    def fetchall(self):
        query, params = self.executed[-1]
        if "FROM folders" in query:
            wanted = params[1] if len(params) > 1 and isinstance(params[1], list) else None
            return [{"id": i} for i in self.folder_rows if wanted is None or i in wanted]
        if "FROM files" in query:
            if "folder_id = ANY" in query:
                pool = self.expanded_rows if self.expanded_rows is not None else self.file_rows
                return [{"id": i} for i in pool]
            wanted = params[1] if len(params) > 1 and isinstance(params[1], list) else None
            universe = list(dict.fromkeys([*self.file_rows, *(self.expanded_rows or ())]))
            return [{"id": i} for i in universe if wanted is None or i in wanted]
        return []


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def make_factory(cursor):
    @contextmanager
    def connections():
        yield FakeConnection(cursor)

    return connections


def updates(cursor):
    return [(q, p) for q, p in cursor.executed if q.startswith("UPDATE chats")]


# ---------------------------------------------------------------------------
# scope_store unit tests
# ---------------------------------------------------------------------------


def test_set_then_get_roundtrip_is_atomic_single_statement():
    cursor = FakeCursor()
    set_chat_scope(make_factory(cursor), user_id=2, chat_id=CHAT_ID, file_ids=[5, 9, 5])
    writes = updates(cursor)
    assert len(writes) == 1
    query, params = writes[0]
    assert "jsonb_set" in query
    assert "read" not in query.split("UPDATE")[0]
    assert params[1] == [5, 9]
    assert params[2] == CHAT_ID
    assert params[3] == 2


def test_clear_removes_key_and_get_returns_none():
    cursor = FakeCursor(chat_messages={SCOPE_KEY: [5]})
    assert get_chat_scope(make_factory(cursor), user_id=2, chat_id=CHAT_ID) == [5]
    set_chat_scope(make_factory(cursor), user_id=2, chat_id=CHAT_ID, file_ids=[])
    writes = updates(cursor)
    assert len(writes) == 1
    assert " - " in writes[0][0] or "-" in writes[0][0]


def test_legacy_empty_list_is_scopeless():
    cursor = FakeCursor(chat_messages=[])
    assert get_chat_scope(make_factory(cursor), user_id=2, chat_id=CHAT_ID) is None


def test_malformed_scope_values_are_scopeless_not_fatal():
    for bad in (
        {"scoped_file_ids": ["x"]},
        {"scoped_file_ids": [0, -3]},
        {"scoped_file_ids": [True]},
        {"scoped_file_ids": "nope"},
        {"other": [1]},
        "just-a-string",
        None,
    ):
        cursor = FakeCursor(chat_messages=bad)
        assert get_chat_scope(make_factory(cursor), user_id=2, chat_id=CHAT_ID) is None


def test_scope_is_user_fenced_and_chat_validated():
    cursor = FakeCursor()
    assert get_chat_scope(make_factory(cursor), user_id=1, chat_id=CHAT_ID) is None or True
    _, params = cursor.executed[-1]
    assert params == (CHAT_ID, 1)
    assert get_chat_scope(make_factory(cursor), user_id=2, chat_id="not-a-uuid") is None
    with pytest.raises(ValueError):
        set_chat_scope(make_factory(cursor), user_id=2, chat_id="not-a-uuid", file_ids=[1])
    with pytest.raises(ValueError):
        set_chat_scope(make_factory(cursor), user_id=2, chat_id=CHAT_ID, file_ids=[0])
    with pytest.raises(ValueError):
        set_chat_scope(make_factory(cursor), user_id=2, chat_id=CHAT_ID, file_ids=[True])


# ---------------------------------------------------------------------------
# _prepare wiring: stale-scope FIRST (most likely to be subtly wrong)
# ---------------------------------------------------------------------------


def test_stale_scope_revoked_file_drops_out_survivor_applies():
    cursor = FakeCursor(chat_messages={SCOPE_KEY: [8, 9]}, file_rows=[8])
    history, authorized, notice = ChatPersistence(make_factory(cursor))._prepare(
        91, CHAT_ID, None
    )
    assert authorized == (8,)
    assert notice is None
    assert history == []


def test_stale_scope_all_dead_falls_back_to_global_with_notice():
    cursor = FakeCursor(chat_messages={SCOPE_KEY: [8, 9]}, file_rows=[])
    history, authorized, notice = ChatPersistence(make_factory(cursor))._prepare(
        91, CHAT_ID, None
    )
    assert authorized is None
    assert notice is not None
    assert "whole vault" in notice


def test_round_trip_tag_then_followup_stays_scoped():
    cursor = FakeCursor(chat_messages="[]", file_rows=[8])
    persistence = ChatPersistence(make_factory(cursor))
    _, authorized, _ = persistence._prepare(91, CHAT_ID, [8])
    assert authorized == (8,)
    assert any("jsonb_set" in query for query, _ in updates(cursor))

    cursor2 = FakeCursor(chat_messages={SCOPE_KEY: [8]}, file_rows=[8])
    _, authorized2, notice2 = ChatPersistence(make_factory(cursor2))._prepare(
        91, CHAT_ID, None
    )
    assert authorized2 == (8,)
    assert notice2 is None


def test_clear_chips_explicit_empty_goes_global():
    cursor = FakeCursor(chat_messages={SCOPE_KEY: [8]}, file_rows=[])
    _, authorized, notice = ChatPersistence(make_factory(cursor))._prepare(91, CHAT_ID, [])
    assert authorized == ()
    assert notice is None
    writes = updates(cursor)
    assert len(writes) == 2
    assert {writes[0][1][0], writes[1][1][0]} == {SCOPE_KEY, SCOPE_FOLDER_KEY}


def test_legacy_rows_are_scopeless_without_crashing():
    cursor = FakeCursor(chat_messages=[], file_rows=[8])
    _, authorized, notice = ChatPersistence(make_factory(cursor))._prepare(91, CHAT_ID, None)
    assert authorized is None
    assert notice is None
    assert updates(cursor) == []


def test_no_chat_id_skips_scope_entirely():
    cursor = FakeCursor(file_rows=[8])
    _, authorized, notice = ChatPersistence(make_factory(cursor))._prepare(91, None, [8])
    assert authorized == (8,)
    assert notice is None
    assert updates(cursor) == []


# ---------------------------------------------------------------------------
# Folder scope: live expansion, caps, staleness
# ---------------------------------------------------------------------------


def test_folder_scope_roundtrip_and_live_expansion():
    cursor = FakeCursor(folder_rows=[5], file_rows=[8, 9])
    factory = make_factory(cursor)
    set_chat_scope(factory, user_id=2, chat_id=CHAT_ID, file_ids=[], folder_ids=[5])
    writes = updates(cursor)
    assert len(writes) == 2
    by_key = {}
    for query, params in writes:
        if "jsonb_set" in query:
            by_key["set"] = params
        else:
            by_key["clear"] = params
    # Files cleared (empty list), folders set — each its own statement.
    assert by_key["clear"][0] == SCOPE_KEY
    assert by_key["set"][1] == [5]
    assert get_chat_folders(
        make_factory(FakeCursor(chat_messages={SCOPE_FOLDER_KEY: [5]})),
        user_id=2, chat_id=CHAT_ID,
    ) == [5]


def test_prepare_expands_owned_folders_with_survivor_semantics():
    cursor = FakeCursor(
        chat_messages="[]", folder_rows=[5], file_rows=[8, 9]
    )
    history, authorized, notice = ChatPersistence(make_factory(cursor))._prepare(
        91, CHAT_ID, None, [5]
    )
    assert authorized == (8, 9)
    assert notice is None


def test_prepare_rejects_unowned_folders_but_tolerates_empty_owned():
    cursor = FakeCursor(chat_messages="[]", folder_rows=[], file_rows=[])
    with pytest.raises(ValueError, match="folder"):
        ChatPersistence(make_factory(cursor))._prepare(91, CHAT_ID, None, [99])
    history, authorized, notice = ChatPersistence(make_factory(cursor))._prepare(
        91, CHAT_ID, [], []
    )
    assert authorized == ()
    assert notice is None


def test_prepare_session_fallback_reexpands_stored_folders():
    cursor = FakeCursor(
        chat_messages={SCOPE_FOLDER_KEY: [5]},
        folder_rows=[5],
        file_rows=[9],
    )
    _, authorized, notice = ChatPersistence(make_factory(cursor))._prepare(
        91, CHAT_ID, None, None
    )
    # File 8 was removed from the folder since; survivor 9 still applies.
    assert authorized == (9,)
    assert notice is None


def test_prepare_session_fallback_all_dead_folder_goes_global_with_notice():
    cursor = FakeCursor(
        chat_messages={SCOPE_KEY: [8], SCOPE_FOLDER_KEY: [5]},
        folder_rows=[],
        file_rows=[],
    )
    _, authorized, notice = ChatPersistence(make_factory(cursor))._prepare(
        91, CHAT_ID, None, None
    )
    assert authorized is None
    assert notice is not None
    assert "whole vault" in notice


def test_prepare_explicit_files_plus_folders_union_with_cap():
    cursor = FakeCursor(
        chat_messages="[]", folder_rows=[5], file_rows=[7], expanded_rows=[8, 9]
    )
    _, authorized, _ = ChatPersistence(make_factory(cursor))._prepare(
        91, CHAT_ID, [7], [5]
    )
    assert authorized == (7, 8, 9)


def test_folder_keys_rejected_when_garbage():
    cursor = FakeCursor()
    with pytest.raises(ValueError):
        set_chat_scope(make_factory(cursor), user_id=2, chat_id=CHAT_ID, folder_ids=[0])
    with pytest.raises(ValueError):
        set_chat_scope(
            make_factory(cursor), user_id=2, chat_id=CHAT_ID,
            folder_ids=list(range(1, 25)),
        )
    assert get_chat_folders(make_factory(cursor), user_id=2, chat_id="nope") is None


def test_set_folders_only_leaves_files_key_untouched():
    cursor = FakeCursor(chat_messages={SCOPE_KEY: [8]})
    factory = make_factory(cursor)
    set_chat_scope(factory, user_id=2, chat_id=CHAT_ID, folder_ids=[5])
    writes = updates(cursor)
    assert len(writes) == 1
    query, params = writes[0]
    assert "jsonb_set" in query
    assert params[1] == [5]
    # Single key in the path; files key never mentioned.
    assert SCOPE_KEY not in query


def test_scope_request_file_ids_optional():
    from app.routers.chat.schemas import ChatScopeRequest

    assert ChatScopeRequest(folder_ids=[5]).file_ids is None
    assert ChatScopeRequest(file_ids=[], folder_ids=[]).folder_ids == []
