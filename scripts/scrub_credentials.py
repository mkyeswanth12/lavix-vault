#!/usr/bin/env python3
"""Scrub stored credentials from chat rows and vision chunks.

Dry-run by default (SELECT only, prints per-table counts, zero writes).
--apply performs redaction; --all --apply additionally requires either
--yes-really or the typed phrase "SCRUB SECRETS". Scope requires at least
one of --user-id, --chat-id, --file-id, or --all.

Only counts are ever printed — never row content, values, or filenames.
Chunk embeddings are nulled on redact (vectors encode the raw text);
re-ingest affected files to restore vector search.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIRM_PHRASE = "SCRUB SECRETS"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--chat-id", default=None)
    parser.add_argument("--file-id", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes-really", action="store_true")
    parser.add_argument(
        "--confirm-phrase",
        default="",
        help='Typed confirmation required with --all --apply (phrase: "SCRUB SECRETS").',
    )
    args = parser.parse_args(argv)
    if not (args.user_id or args.chat_id or args.file_id or args.all):
        parser.error("scope required: --user-id, --chat-id, --file-id, or --all")
    if args.all and args.apply and not args.yes_really:
        if args.confirm_phrase != CONFIRM_PHRASE:
            parser.error(
                '--all --apply needs --yes-really or --confirm-phrase "SCRUB SECRETS"'
            )
    return args


def _scope_filter(scope: str, table: str, params: list):
    clauses = []
    if scope.get("chat_id") and table in ("chat_messages",):
        clauses.append("chat_id = %s")
        params.append(scope["chat_id"])
    if scope.get("file_id") and table in ("document_chunks", "files"):
        key = "file_id" if table == "document_chunks" else "id"
        clauses.append(f"{key} = %s")
        params.append(scope["file_id"])
    if scope.get("user_id"):
        clauses.append("user_id = %s")
        params.append(scope["user_id"])
    if scope.get("all"):
        clauses.append("TRUE")
    return (" AND ".join(clauses)) if clauses else "FALSE"


def find_candidates(connection_factory, scope):
    """Read-only scan. Returns {table: [ids]}. Never prints row content."""
    from app.security.redact import contains_credentials

    found: dict[str, list] = {"chat_messages": [], "document_chunks": [], "files": []}
    targets = {
        "chat_messages": ("id", "content"),
        "document_chunks": ("chunk_id", "content"),
        "files": ("id", "quick_summary"),
    }
    with connection_factory() as connection:
        cursor = connection.cursor()
        for table, (id_col, text_col) in targets.items():
            params: list = []
            where = _scope_filter(scope, table, params)
            cursor.execute(
                f"SELECT {id_col}, {text_col} FROM {table} WHERE {where}",
                tuple(params),
            )
            for row in cursor.fetchall() or []:
                values = list(row.values()) if isinstance(row, dict) else list(row)
                if len(values) >= 2 and contains_credentials(values[1] or ""):
                    found[table].append(values[0])
    return found


def apply_scrub(connection_factory, found):
    """Redact matched rows; null chunk embeddings. Returns counts only."""
    from app.security.redact import redact_credentials

    counts = {"chat_messages": 0, "document_chunks": 0, "files": 0}
    with connection_factory() as connection:
        cursor = connection.cursor()
        for message_id in found["chat_messages"]:
            cursor.execute(
                "SELECT content FROM chat_messages WHERE id = %s", (message_id,)
            )
            row = cursor.fetchone()
            text = (row.get("content") if isinstance(row, dict) else row[0]) or ""
            cursor.execute(
                "UPDATE chat_messages SET content = %s WHERE id = %s",
                (redact_credentials(text), message_id),
            )
            counts["chat_messages"] += 1
        for chunk_id in found["document_chunks"]:
            cursor.execute(
                "SELECT content FROM document_chunks WHERE chunk_id = %s", (chunk_id,)
            )
            row = cursor.fetchone()
            text = (row.get("content") if isinstance(row, dict) else row[0]) or ""
            cursor.execute(
                "UPDATE document_chunks SET content = %s, embedding_text = %s,"
                " embedding = NULL WHERE chunk_id = %s",
                (redact_credentials(text), redact_credentials(text), chunk_id),
            )
            counts["document_chunks"] += 1
        for file_id in found["files"]:
            cursor.execute(
                "SELECT quick_summary FROM files WHERE id = %s", (file_id,)
            )
            row = cursor.fetchone()
            summary = (row.get("quick_summary") if isinstance(row, dict) else row[0]) or ""
            cursor.execute(
                "UPDATE files SET quick_summary = %s WHERE id = %s",
                (redact_credentials(summary), file_id),
            )
            counts["files"] += 1
        connection.commit()
    return counts


def main(argv=None, connection_factory=None) -> int:
    args = _parse_args(argv)
    if connection_factory is None:
        from app.database import get_db

        connection_factory = get_db
    scope = {
        "user_id": args.user_id,
        "chat_id": args.chat_id,
        "file_id": args.file_id,
        "all": args.all,
    }
    found = find_candidates(connection_factory, scope)
    total = sum(len(ids) for ids in found.values())
    for table in ("chat_messages", "document_chunks", "files"):
        print(f"{table}: {len(found[table])} matching rows")
    if not args.apply:
        print(f"dry-run: no writes performed ({total} rows would be redacted)")
        return 0
    counts = apply_scrub(connection_factory, found)
    print(
        "applied: chat_messages={chat_messages} document_chunks={document_chunks} "
        "files={files} (chunk embeddings nulled; re-ingest to restore search)".format(
            **counts
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
