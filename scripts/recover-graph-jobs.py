#!/usr/bin/env python3
"""Recover graph_memory_jobs stuck after the summarize-constraint outage.

Every memory save used to enqueue a job_type='summarize' row while the DB
CHECK only allowed extract/project/expire/purge, so the whole save
transaction rolled back and the extract job retried to 'failed'. After
migration 0022 this script re-queues exactly those jobs.

Scope is explicit: --user-id <id> or --all is required. Only jobs with
state='failed' killed by this outage are touched (error_code
'graph_job_failed'). Saves rolled back fully, so replay re-extracts from
the source message and the (user_id, normalized_fingerprint) unique index
dedups — no duplicates.

Usage:
    python scripts/recover-graph-jobs.py --user-id 1            # dry run
    python scripts/recover-graph-jobs.py --user-id 1 --apply    # mutate
    python scripts/recover-graph-jobs.py --all --apply
    python scripts/recover-graph-jobs.py --user-id 1 --print-sql
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

#Saver: only these failed jobs are ever reset. Anything else (complete,
#cancelled, running, other error codes) is left alone.
RECOVERABLE_STATE = "failed"
RECOVERABLE_ERROR_CODE = "graph_job_failed"

_LIST_SQL = """
SELECT id, user_id, job_type, state, attempts, max_attempts,
       error_code, source_message_id, updated_at
FROM graph_memory_jobs
WHERE state = 'failed' AND error_code = 'graph_job_failed'
  {user_scope}
ORDER BY updated_at DESC
"""

_RESET_SQL = """
UPDATE graph_memory_jobs
SET state = 'queued',
    attempts = 0,
    error_code = NULL,
    error_detail = NULL,
    lease_owner = NULL,
    lease_expires_at = NULL,
    available_at = NOW(),
    started_at = NULL,
    finished_at = NULL,
    updated_at = NOW()
WHERE state = 'failed' AND error_code = 'graph_job_failed'
  {user_scope}
"""


def is_recoverable(row: dict[str, Any]) -> bool:
    """Pure scope predicate, unit-tested without a database."""
    return (
        str(row.get("state") or "") == RECOVERABLE_STATE
        and str(row.get("error_code") or "") == RECOVERABLE_ERROR_CODE
    )


def build_sql(user_id: int | None) -> tuple[str, str, tuple[Any, ...]]:
    scope = "" if user_id is None else "AND user_id = %s"
    params: tuple[Any, ...] = () if user_id is None else (user_id,)
    return (
        _LIST_SQL.format(user_scope=scope),
        _RESET_SQL.format(user_scope=scope),
        params,
    )


def _connect(dsn: str):  # type: ignore[no-untyped-def]
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(dsn, row_factory=dict_row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--user-id", type=int, default=None)
    scope.add_argument("--all", action="store_true")
    parser.add_argument("--apply", action="store_true", help="mutate; default is dry run")
    parser.add_argument("--print-sql", action="store_true", help="print statements and exit")
    parser.add_argument(
        "--dsn",
        default=os.environ.get(
            "LAVIX_RECOVER_DSN", "dbname=vault user=vault host=localhost password=vault"
        ),
    )
    args = parser.parse_args(argv)

    list_sql, reset_sql, params = build_sql(args.user_id)
    if args.print_sql:
        print(list_sql)
        print(reset_sql)
        return 0

    conn = _connect(args.dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(list_sql, params)
            candidates = [row for row in (cursor.fetchall() or []) if is_recoverable(row)]
    finally:
        conn.close()
    print(f"recoverable_jobs={len(candidates)}")
    for row in candidates:
        print(
            f"  {row['id']} user={row['user_id']} type={row['job_type']} "
            f"attempts={row['attempts']}/{row['max_attempts']} "
            f"source_message={row['source_message_id']} updated={row['updated_at']}"
        )
    if not args.apply:
        print("dry-run: no changes made (pass --apply to reset)")
        return 0
    if not candidates:
        print("nothing to reset")
        return 0
    conn = _connect(args.dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(reset_sql, params)
            reset = cursor.rowcount
        conn.commit()
    finally:
        conn.close()
    print(f"reset_jobs={reset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
