"""Operator CLI for Lavix Vault (no auto-promotion, no public endpoint).

Usage (inside the running stack only — these need database access)::

    docker compose exec api python -m app.cli promote-admin <username>
    docker compose run --rm api python -m app.cli reembed --model <name> --dimensions <n> [--dry-run]

There is deliberately no INITIAL_ADMIN_* environment variable: the first
administrator is created by promoting exactly one registered account
through this command, which requires shell access to the deployment.
Note `run --rm` (not `exec`) for reembed: it must work even when the API
itself would refuse to boot on the new dimensions.
"""

from __future__ import annotations

import argparse
import sys

from app.database import get_db


def promote_admin(username: str) -> int:
    """Set is_admin for one registered, active user. Returns exit status."""
    name = username.strip()
    if not name:
        print("error: username must not be empty", file=sys.stderr)
        return 2
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, username, is_admin, is_active FROM users WHERE username = %s",
            (name,),
        )
        row = cursor.fetchone()
        if row is None:
            print(f"error: no such user: {name!r} (register first)", file=sys.stderr)
            return 1
        user_id = row["id"] if isinstance(row, dict) else row[0]
        active = row["is_active"] if isinstance(row, dict) else row[3]
        already = row["is_admin"] if isinstance(row, dict) else row[2]
        if not active:
            print(f"error: user {name!r} is deactivated", file=sys.stderr)
            return 1
        if already:
            print(f"ok: user {name!r} (id {user_id}) is already an admin")
            return 0
        cursor.execute("UPDATE users SET is_admin = TRUE WHERE id = %s", (user_id,))
    print(f"ok: promoted user {name!r} (id {user_id}) to admin")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lavix Vault operator commands")
    sub = parser.add_subparsers(dest="command", required=True)
    promote = sub.add_parser("promote-admin", help="promote one registered user to admin")
    promote.add_argument("username", help="exact username to promote")
    reembed = sub.add_parser("reembed", help="migrate stored embeddings to a new model")
    reembed.add_argument("--model", required=True, help="target Ollama embedding model")
    reembed.add_argument("--dimensions", required=True, type=int, help="target vector dimensions")
    reembed.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = parser.parse_args(argv)
    if args.command == "promote-admin":
        return promote_admin(args.username)
    if args.command == "reembed":
        from app.db.reembed import main as reembed_main

        return reembed_main(
            ["--model", args.model, "--dimensions", str(args.dimensions)]
            + (["--dry-run"] if args.dry_run else [])
        )
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
