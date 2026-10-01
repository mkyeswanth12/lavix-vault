"""Live-schema proof: save path + job-type constraint + replay dedup.

Runs ONLY against an explicit disposable PostgreSQL (env-gated, skipped
otherwise). Proves on a real migrated schema what unit fakes cannot:
  1. fresh `migrate up` reaches head including 0022;
  2. every JOB_TYPES value INSERTs cleanly (the summarize outage stays fixed);
  3. saving the same statement twice leaves exactly one live item row
     (recovery replay cannot duplicate);
  4. a save produces a queued summarize job.

Enable with an empty disposable DB, e.g.:
    LAVIX_MEMORY_TEST_PG_DISPOSABLE=1 \\
    LAVIX_MEMORY_TEST_PG_DSN="dbname=memtest user=postgres host=127.0.0.1 port=55433" \\
    pytest tests/integration/test_graph_memory_save_path_pg.py
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from urllib.parse import urlparse
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration

_DSN_ENV = "LAVIX_MEMORY_TEST_PG_DSN"
_DISPOSABLE_ENV = "LAVIX_MEMORY_TEST_PG_DISPOSABLE"


def _dsn() -> str:
    if os.environ.get(_DISPOSABLE_ENV) != "1":
        pytest.skip(f"set {_DISPOSABLE_ENV}=1 and {_DSN_ENV} to an empty disposable database")
    dsn = os.environ.get(_DSN_ENV, "").strip()
    if not dsn:
        pytest.skip(f"set {_DISPOSABLE_ENV}=1 and {_DSN_ENV} to an empty disposable database")
    if "://" in dsn:
        host = urlparse(dsn).hostname or ""
    else:
        import re

        match = re.search(r"(?:^|\s)host=([^\s]+)", dsn)
        host = (match.group(1) if match else "localhost").strip("'\"")
    if host not in ("localhost", "127.0.0.1", "::1"):
        pytest.fail(f"{_DSN_ENV} must target a loopback disposable database, got host={host!r}")
    return dsn


def _connect(dsn: str):  # type: ignore[no-untyped-def]
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(dsn, row_factory=dict_row)


@contextmanager
def _committing(connection):  # type: ignore[no-untyped-def]
    """get_db-shaped connection scope: commit on clean exit, else rollback."""
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def test_fresh_migrate_up_reaches_head_with_summarize_allowed() -> None:
    from app.db.migrations import DEFAULT_MIGRATIONS_DIR, apply_migrations, discover_migrations

    conn = _connect(_dsn())
    try:
        report = apply_migrations(conn, DEFAULT_MIGRATIONS_DIR)
    finally:
        conn.close()
    head = max(item.version for item in discover_migrations())
    assert report.current_version == head
    assert report.head_version == head


def test_save_path_on_live_schema_with_replay_dedup() -> None:
    from app.db.migrations import DEFAULT_MIGRATIONS_DIR, apply_migrations
    from app.graph_memory.models import MemoryCandidate
    from app.graph_memory.repository import JOB_TYPES, PostgresGraphMemoryRepository
    from app.graph_memory.validation import validate_candidate

    conn = _connect(_dsn())
    try:
        apply_migrations(conn, DEFAULT_MIGRATIONS_DIR)
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) "
                "VALUES ('savetest', 'savetest@example.com', 'x') "
                "ON CONFLICT (username) DO NOTHING RETURNING id"
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute("SELECT id FROM users WHERE username = 'savetest'")
                row = cursor.fetchone()
            user_id = int(row["id"])
            # Idempotent re-runs: start the job/item slate clean.
            cursor.execute("DELETE FROM graph_memory_jobs WHERE user_id = %s", (user_id,))
            cursor.execute("DELETE FROM graph_memory_items WHERE user_id = %s", (user_id,))
            cursor.execute(
                "INSERT INTO graph_memory_tenants (user_id, enabled) "
                "VALUES (%s, TRUE) ON CONFLICT (user_id) DO NOTHING",
                (user_id,),
            )
            cursor.execute(
                "SELECT tenant_uuid, generation FROM graph_memory_tenants WHERE user_id = %s",
                (user_id,),
            )
            tenant_row = cursor.fetchone()
            tenant_uuid, generation = tenant_row["tenant_uuid"], tenant_row["generation"]
            chat_id, message_id = uuid4(), uuid4()
            cursor.execute("INSERT INTO chats (id, user_id) VALUES (%s, %s)", (chat_id, user_id))
            content = "My name is Jordan and I live in Hyderabad"
            cursor.execute(
                "INSERT INTO chat_messages (id, chat_id, user_id, sequence_number, role, content)"
                " VALUES (%s, %s, %s, 1, 'user', %s)",
                (message_id, chat_id, user_id, content),
            )
            # Every canonical job type must INSERT cleanly.
            for job_type in sorted(JOB_TYPES):
                cursor.execute(
                    "INSERT INTO graph_memory_jobs (user_id, tenant_uuid, generation, job_type)"
                    " VALUES (%s, %s, %s, %s)",
                    (user_id, tenant_uuid, generation, job_type),
                )
        conn.commit()

        repo = PostgresGraphMemoryRepository(connection_factory=lambda: _committing(conn))
        candidate = MemoryCandidate(
            kind="fact",
            subject="I",
            predicate="name",
            object_value="Jordan",
            confidence=0.95,
            source_excerpt=content,
            source_role="user",
            explicit_user_assertion=True,
        )
        decision = validate_candidate(candidate)
        assert decision.accepted

        kwargs = dict(
            expected_generation=int(generation),
            source_chat_id=chat_id,
            source_message_id=message_id,
            candidate=candidate,
            decision=decision,
        )
        first = repo.upsert_candidate(user_id, **kwargs)  # type: ignore[arg-type]
        second = repo.upsert_candidate(user_id, **kwargs)  # type: ignore[arg-type]
        assert first.item.id == second.item.id

        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) AS cnt FROM graph_memory_items "
                "WHERE user_id = %s AND status IN ('pending', 'active')",
                (user_id,),
            )
            live_rows = int(cursor.fetchone()["cnt"])
            cursor.execute(
                "SELECT count(*) AS cnt FROM graph_memory_jobs "
                "WHERE user_id = %s AND job_type = 'summarize' AND state IN ('queued', 'running')",
                (user_id,),
            )
            summarize_jobs = int(cursor.fetchone()["cnt"])
        conn.commit()
    finally:
        conn.close()

    assert live_rows == 1
    assert summarize_jobs == 1
