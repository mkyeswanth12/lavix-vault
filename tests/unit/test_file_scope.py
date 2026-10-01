"""Phase 1 file scope: additive tags, @-mention resolution, no-fallback refusal.

RED-first: mentions module, UnresolvedMention, scope union, and the
attachment-sufficiency short-circuit do not exist yet.
"""

import asyncio
from contextlib import contextmanager

import pytest

from app.routers.chat.mentions import extract_at_tokens
from app.routers.chat.orchestrator import (
    ChatCoordinator,
    ChatPersistence,
    UnresolvedMention,
)
from app.routers.chat.schemas import ChatRequest
from app.routers.chat.scope_store import SCOPE_KEY

CHAT_ID = "12345678-1234-4123-8123-123456789012"


class FakeCursor:
    """Canned files/folders/chat rows plus filename lookup for @-mentions."""

    def __init__(self, *, chat_messages="[]", file_rows=(), folder_rows=(), name_rows=None, rewrite_rows=None):
        self.chat_messages = chat_messages
        self.file_rows = list(file_rows)
        self.folder_rows = list(folder_rows)
        self.name_rows = dict(name_rows or {})
        self.rewrite_rows = list(rewrite_rows) if rewrite_rows is not None else None
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
        if "FROM chat_messages" in query and "content_json" in query:
            # Production query is ORDER BY sequence_number DESC; callers
            # reverse back to chronological. Emulate that ordering.
            return list(reversed(self.rewrite_rows or []))
        if "original_filename" in query and "quick_summary" in query:
            # Attachment-text query (aliased COALESCE columns).
            return [
                {
                    "original_filename": "doc3.png",
                    "quick_summary": "Git push to example dot com",
                    "quick_tags": ["git", "push"],
                }
            ]
        if "original_filename" in query:
            wanted = params[1] if len(params) > 1 and isinstance(params[1], list) else []
            rows = []
            for name in wanted:
                for file_id in self.name_rows.get(str(name).lower(), ()):
                    rows.append({"id": file_id, "original_filename": name})
            return rows
        if "FROM folders" in query:
            wanted = params[1] if len(params) > 1 and isinstance(params[1], list) else None
            return [{"id": i} for i in self.folder_rows if wanted is None or i in wanted]
        if "FROM files" in query:
            wanted = params[1] if len(params) > 1 and isinstance(params[1], list) else None
            return [
                {
                    "id": i,
                    "ai_status": "indexed",
                    "current_revision": 1,
                    "recently_granted": True,
                }
                for i in self.file_rows
                if wanted is None or i in wanted
            ]
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
# @-mention token extraction (pure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("summarize @report.pdf please", ["report.pdf"]),
        ("@image.png is it pushed?", ["image.png"]),
        ("look at @doc-v2, ok?", ["doc-v2"]),
        ("compare @a.pdf and @b.pdf", ["a.pdf", "b.pdf"]),
        ("mail me at a@b.com now", []),
        ("no mentions here", []),
        ("lone @ sign", []),
        ("@trailing.", ["trailing"]),
    ],
)
def test_extract_at_tokens_shapes(text, expected):
    assert extract_at_tokens(text) == expected


# ---------------------------------------------------------------------------
# Additive session scope
# ---------------------------------------------------------------------------


def test_second_tag_adds_to_session_scope_instead_of_replacing():
    cursor = FakeCursor(chat_messages={SCOPE_KEY: [8]}, file_rows=[8, 9])
    _, authorized, notice = ChatPersistence(make_factory(cursor))._prepare(
        91, CHAT_ID, [9]
    )
    assert authorized == (8, 9)
    assert notice is None
    writes = updates(cursor)
    assert writes, "union must persist back to session scope"
    assert any("jsonb_set" in query for query, _ in writes)


def test_first_tag_becomes_session_scope():
    cursor = FakeCursor(chat_messages="[]", file_rows=[8])
    _, authorized, _ = ChatPersistence(make_factory(cursor))._prepare(91, CHAT_ID, [8])
    assert authorized == (8,)
    assert any("jsonb_set" in query for query, _ in updates(cursor))


def test_two_explicit_tags_scope_exactly_those_files():
    cursor = FakeCursor(chat_messages="[]", file_rows=[8, 9])
    _, authorized, _ = ChatPersistence(make_factory(cursor))._prepare(
        91, CHAT_ID, [8, 9]
    )
    assert authorized == (8, 9)


def test_other_users_file_id_is_never_searchable():
    cursor = FakeCursor(chat_messages="[]", file_rows=[])
    with pytest.raises(ValueError):
        ChatPersistence(make_factory(cursor))._prepare(91, CHAT_ID, [7])


# ---------------------------------------------------------------------------
# @-mention resolution through prepare()
# ---------------------------------------------------------------------------


def test_unresolved_mention_raises_before_any_retrieval():
    cursor = FakeCursor(chat_messages="[]", file_rows=[8], name_rows={"other.pdf": (8,)})
    persistence = ChatPersistence(make_factory(cursor))
    with pytest.raises(UnresolvedMention) as excinfo:
        asyncio.run(
            persistence.prepare(
                user_id=91,
                chat_id=CHAT_ID,
                requested_file_ids=None,
                message="summarize @ghost-file.pdf",
            )
        )
    assert "ghost-file.pdf" in str(excinfo.value)
    # No scope write and no retrieval: the only chats touch is the scope read.
    assert updates(cursor) == []


def test_resolved_mention_name_scopes_to_that_file():
    cursor = FakeCursor(chat_messages="[]", file_rows=[7], name_rows={"report.pdf": (7,)})
    persistence = ChatPersistence(make_factory(cursor))
    history, authorized, notice, _rewrite = asyncio.run(
        persistence.prepare(
            user_id=91,
            chat_id=CHAT_ID,
            requested_file_ids=None,
            message="@report.pdf summarize this",
        )
    )
    assert authorized == (7,)
    assert notice is None
    assert any("jsonb_set" in query for query, _ in updates(cursor))


def test_resolved_mention_adds_to_explicit_tags():
    cursor = FakeCursor(
        chat_messages="[]", file_rows=[7, 9], name_rows={"report.pdf": (7,)}
    )
    persistence = ChatPersistence(make_factory(cursor))
    _, authorized, _, _rewrite = asyncio.run(
        persistence.prepare(
            user_id=91,
            chat_id=CHAT_ID,
            requested_file_ids=[9],
            message="@report.pdf and the other one",
        )
    )
    assert authorized == (7, 9)  # _authorize_files orders by id


# ---------------------------------------------------------------------------
# stream() wiring: unresolved @ -> file-not-found answer, no fallback
# ---------------------------------------------------------------------------


class _MentionRaisingPersistence:
    def __init__(self):
        self.messages = []

    async def prepare(self, **kwargs):
        raise UnresolvedMention(["ghost-file.pdf"])

    async def add_message(self, **kwargs):
        self.messages.append(kwargs)


class _QuietRuntime:
    def __init__(self):
        self.calls = []

    async def stream(self, request, capability):
        self.calls.append((request, capability))
        yield {"type": "done"}


def _coordinator(persistence):
    from app.routers.chat.orchestrator import CapabilitySigner, RunEvidenceStore

    signer = CapabilitySigner("m" * 32)
    evidence = RunEvidenceStore()
    return ChatCoordinator(
        signer=signer,
        runtime=_QuietRuntime(),
        persistence=persistence,
        evidence_store=evidence,
    )


def test_stream_unresolved_mention_yields_file_not_found_without_runtime():
    persistence = _MentionRaisingPersistence()
    coordinator = _coordinator(persistence)
    request = ChatRequest(
        message="summarize @ghost-file.pdf",
        chat_id="b9d87b40-50e0-4bf4-84eb-86528fd770c9",
        web_search_enabled=False,
    )

    async def collect():
        return [event async for event in coordinator.stream(request, user_id=91)]

    events = asyncio.run(collect())
    tokens = [e for e in events if e.get("type") == "token"]
    assert tokens, "unresolved @ must produce a visible answer"
    combined = " ".join(str(e.get("content") or "") for e in tokens)
    assert "ghost-file.pdf" in combined
    assert "not found" in combined.casefold()
    assert [e for e in events if e.get("type") == "sources"]
    # The runtime was never consulted: no retrieval, no web, no synthesis.
    assert coordinator._runtime.calls == []
    assert persistence.messages, "user message must still persist"


# ---------------------------------------------------------------------------
# Chip truthfulness: whole-vault fallback notice kills the chip
# ---------------------------------------------------------------------------


class _FallbackPersistence:
    def __init__(self):
        self.messages = []

    async def prepare(self, **kwargs):
        return (
            [],
            None,
            "Previously tagged files or folders are no longer available; "
            "searching the whole vault.",
            ([], {}, False),
        )

    async def add_message(self, **kwargs):
        self.messages.append(kwargs)
        return None


class _FinalRuntime:
    def __init__(self, signer, evidence):
        self.signer = signer
        self.evidence = evidence

    async def stream(self, request, capability):
        from agent_runtime.events import AUTHORITATIVE_STREAM_PROVENANCE

        yield {"type": "status", "step": "planning"}
        yield {
            "type": "final",
            "answer": "Answer",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        }
        yield {"type": "usage", "prompt_tokens": 1, "completion_tokens": 1}
        yield {"type": "done"}


def test_scope_fallback_notice_carries_chip_kill_flag():
    from app.routers.chat.orchestrator import CapabilitySigner, RunEvidenceStore

    signer = CapabilitySigner("n" * 32)
    evidence = RunEvidenceStore()
    coordinator = ChatCoordinator(
        signer=signer,
        runtime=_FinalRuntime(signer, evidence),
        persistence=_FallbackPersistence(),
        evidence_store=evidence,
    )
    request = ChatRequest(
        message="summarize the changes",
        chat_id="b9d87b40-50e0-4bf4-84eb-86528fd770c9",
        web_search_enabled=False,
    )

    async def collect():
        return [event async for event in coordinator.stream(request, user_id=91)]

    events = asyncio.run(collect())
    notices = [e for e in events if e.get("type") == "notice"]
    assert notices, "stale scope must surface a notice"
    assert all(n.get("scope_fallback") is True for n in notices)


# ---------------------------------------------------------------------------
# Phase 4: user turns are scrubbed in the R1 rewrite window only
# ---------------------------------------------------------------------------


def test_rewrite_window_scrubs_user_pasted_secrets():
    cursor = FakeCursor(
        rewrite_rows=[
            {"role": "user", "content": "my password is hunter2 please help", "content_json": None},
            {"role": "assistant", "content": "I can help with that.", "content_json": None},
        ]
    )
    history, entities, pending = ChatPersistence(make_factory(cursor))._prepare_rewrite_state(
        91, CHAT_ID
    )
    user_texts = [m["content"] for m in history if m["role"] == "user"]
    assert user_texts, "user turn must survive redaction"
    assert not any("hunter2" in text for text in user_texts)
    assert any("REDACTED" in text for text in user_texts)
    # Structural: the rewrite fetch never writes stored rows.
    assert not any(q.startswith("UPDATE") for q, _ in cursor.executed)


def test_rewrite_window_keeps_non_secret_user_text_verbatim():
    cursor = FakeCursor(
        rewrite_rows=[
            {"role": "user", "content": "my name is XW", "content_json": None},
            {"role": "assistant", "content": "Hello XW.", "content_json": None},
        ]
    )
    history, _, _ = ChatPersistence(make_factory(cursor))._prepare_rewrite_state(
        91, CHAT_ID
    )
    assert [m["content"] for m in history if m["role"] == "user"] == ["my name is XW"]


def test_attachment_text_includes_summary_and_tags_via_aliases():
    # FIX 5 root cause: the COALESCE columns need AS aliases, or the row
    # dict carries "coalesce" keys and summary/tags silently vanish.
    persistence = ChatPersistence(make_factory(FakeCursor()))
    text, count = persistence._attachment_text(91, [7])
    assert count == 1
    assert "Git push to example dot com" in text
    assert "git" in text and "push" in text
