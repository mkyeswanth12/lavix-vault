"""Failed-turn isolation: pair-aware history + orphan backfill enqueue."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from app.routers.chat.orchestrator import (
    ChatPersistence,
    PostgresGraphExtractionScheduler,
    _paired_recent,
)


def _row(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


# --- _paired_recent pure behavior -------------------------------------------


def test_orphan_user_excluded_but_pairs_kept() -> None:
    rows = [
        _row("user", "butter chicken spices?"),
        _row("assistant", "spice blends answer"),
        _row("user", "Bheem dog memories?"),  # failed turn: no assistant follows
    ]
    assert _paired_recent(rows, limit=3) == rows[:2]


def test_normal_paired_turns_unchanged() -> None:
    rows = [_row("user", "q1"), _row("assistant", "a1"), _row("user", "q2"), _row("assistant", "a2")]
    assert _paired_recent(rows, limit=4) == rows


def test_two_consecutive_failures_excluded() -> None:
    rows = [
        _row("user", "q1"),
        _row("assistant", "a1"),
        _row("user", "failed one?"),
        _row("user", "failed two?"),
    ]
    assert _paired_recent(rows, limit=4) == rows[:2]


def test_lone_leading_assistant_dropped_after_window_trim() -> None:
    rows = [
        _row("user", "q1"),
        _row("assistant", "a1"),
        _row("user", "q2"),
        _row("assistant", "a2"),
    ]
    # Last 3 would start mid-pair; the lone assistant is dropped.
    assert _paired_recent(rows, limit=3) == rows[2:]


def test_newest_user_always_dropped_and_empty_safe() -> None:
    assert _paired_recent([_row("user", "q")], limit=3) == []
    assert _paired_recent([], limit=3) == []
    assert _paired_recent([_row("assistant", "a")], limit=3) == []


def test_rewrite_window_limit_respected() -> None:
    rows = [_row("user", f"q{i}") for i in range(3)] + [_row("assistant", "a")]
    # Only the last user/assistant adjacency pairs; limit caps output.
    assert _paired_recent(rows, limit=6) == [_row("user", "q2"), _row("assistant", "a")]
    assert _paired_recent(rows, limit=1) == []


# --- ChatPersistence._prepare regression -------------------------------------


class _FakeCursor:
    def __init__(self, chat_row: Any, message_rows: list[dict[str, Any]]) -> None:
        self._chat_row = chat_row
        self._message_rows = message_rows
        self.executed: list[str] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append(" ".join(str(sql).split()))

    def fetchone(self) -> Any:
        last = self.executed[-1]
        if "FROM chats" in last:
            return self._chat_row
        return None

    def fetchall(self) -> Any:
        last = self.executed[-1]
        if "FROM chat_messages" in last:
            # Production query is ORDER BY sequence_number DESC; callers
            # reverse back to chronological. Emulate that ordering.
            return list(reversed(self._message_rows))
        return []


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _prepare_with(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    chat_row = {"messages": []}
    factory_calls: list[_FakeConn] = []

    def factory() -> _FakeConn:
        conn = _FakeConn(_FakeCursor(chat_row, messages))
        factory_calls.append(conn)
        return conn

    persistence = ChatPersistence(connection_factory=factory)  # type: ignore[arg-type]
    history, authorized, _ = persistence._prepare(
        41, str(uuid4()), None, None
    )
    assert authorized is None
    return history


def test_prepare_drops_failed_turn_from_synthesis_window() -> None:
    history = _prepare_with(
        [
            {"role": "user", "content": "butter chicken spices?"},
            {"role": "assistant", "content": "spice blends answer"},
            {"role": "user", "content": "Bheem dog memories?"},
        ]
    )
    assert history == [
        {"role": "user", "content": "butter chicken spices?"},
        {"role": "assistant", "content": "spice blends answer"},
    ]


def test_prepare_keeps_healthy_history() -> None:
    history = _prepare_with(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
    )
    assert history == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


# --- orphan backfill ----------------------------------------------------------


class _BackfillService:
    def __init__(self, existing: set[UUID]) -> None:
        self.existing = set(existing)
        self.enqueued: list[tuple[int, UUID, UUID]] = []

    def extraction_job_exists(self, user_id: int, message_id: UUID) -> bool:
        return message_id in self.existing

    def enqueue_extraction(self, user_id: int, chat_id: UUID, message_id: UUID) -> UUID | None:
        self.enqueued.append((user_id, chat_id, message_id))
        self.existing.add(message_id)
        return uuid4()


def _enable_role(monkeypatch: Any) -> None:
    from app.services import model_config

    monkeypatch.setattr(
        model_config.ModelConfigurationRepository,
        "get",
        lambda self: SimpleNamespace(
            memory_extraction=SimpleNamespace(enabled=True, model="m")
        ),
    )


def test_backfill_enqueues_orphans_once_and_bounds_to_five(monkeypatch: Any) -> None:
    _enable_role(monkeypatch)
    chat_id = uuid4()
    current = uuid4()
    prior = [uuid4() for _ in range(7)]  # prior[6] newest … prior[0] oldest
    window = list(reversed(prior))[:5]  # what LIMIT 5 returns: newest five
    service = _BackfillService(existing={window[2]})  # one already filed

    class _BoundedCursor:
        def __init__(self) -> None:
            self._sql = ""
            self._limit = 5

        def execute(self, sql: str, params: Any = None) -> None:
            self._sql = " ".join(str(sql).split())
            self._limit = int(params[-1])

        def fetchall(self) -> list[dict[str, UUID]]:
            assert "ORDER BY sequence_number DESC" in self._sql
            assert "LIMIT" in self._sql
            return [{"id": mid} for mid in window[: self._limit]]

    class _BoundedConn:
        def cursor(self) -> _BoundedCursor:
            return _BoundedCursor()

        def __enter__(self) -> _BoundedConn:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    sched = PostgresGraphExtractionScheduler(
        connection_factory=lambda: _BoundedConn(),  # type: ignore[return-value]
        service=service,  # type: ignore[arg-type]
    )
    assert sched.backfill_orphans(41, chat_id, current) == 4
    assert {mid for _, _, mid in service.enqueued} == set(window) - {window[2]}
    assert all(call[1] == chat_id for call in service.enqueued)
    # Idempotent: second run files nothing new.
    assert sched.backfill_orphans(41, chat_id, current) == 0


def test_backfill_skips_when_role_disabled(monkeypatch: Any) -> None:
    from app.services import model_config

    monkeypatch.setattr(
        model_config.ModelConfigurationRepository,
        "get",
        lambda self: SimpleNamespace(
            memory_extraction=SimpleNamespace(enabled=False, model="m")
        ),
    )
    service = _BackfillService(existing=set())

    class _EmptyCursor:
        def execute(self, sql: str, params: Any = None) -> None:
            pass

        def fetchall(self) -> list[dict[str, UUID]]:
            return []

    class _EmptyConn:
        def cursor(self) -> _EmptyCursor:
            return _EmptyCursor()

        def __enter__(self) -> _EmptyConn:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    sched = PostgresGraphExtractionScheduler(
        connection_factory=lambda: _EmptyConn(),  # type: ignore[return-value]
        service=service,  # type: ignore[arg-type]
    )
    assert sched.backfill_orphans(41, uuid4(), uuid4()) == 0
    assert service.enqueued == []
