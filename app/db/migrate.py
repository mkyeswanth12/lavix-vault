"""CLI for applying and verifying Lavix database migrations.

Usage::

    python -m app.db.migrate up
    python -m app.db.migrate status
    python -m app.db.migrate verify
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .migrations import DEFAULT_MIGRATIONS_DIR, MigrationError, apply_migrations
from .readiness import check_schema_head


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply or verify Lavix SQL migrations")
    parser.add_argument("command", nargs="?", choices=("up", "status", "verify"), default="up")
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=DEFAULT_MIGRATIONS_DIR,
        help="Directory containing NNNN_description.sql files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # Importing app.database loads deployment configuration, so keep it outside
    # module import time. This lets discovery/readiness code remain easy to test.
    from app.database import get_connection, put_connection

    connection = None
    try:
        connection = get_connection()
        if args.command == "up":
            report = apply_migrations(connection, args.migrations_dir)
            versions = ",".join(str(version) for version in report.applied_versions) or "none"
            print(
                f"schema_version={report.current_version} "
                f"head_version={report.head_version} applied={versions}"
            )
            # Fresh-install dimension sizing (empty tables only; populated
            # databases are refused here with a reembed pointer, never
            # silently altered).
            from app.config import settings
            from app.db.embedding_ddl import ensure_fresh_dimensions

            sizing = ensure_fresh_dimensions(
                connection,
                dimensions=settings.embedding_dimension,
                model=settings.embedding_model_name,
            )
            print(f"embedding_columns: {sizing}")
            return 0

        status = check_schema_head(connection, args.migrations_dir)
        pending = ",".join(str(version) for version in status.pending_versions) or "none"
        print(
            f"ready={str(status.ready).lower()} reason={status.reason} "
            f"schema_version={status.current_version} head_version={status.head_version} "
            f"pending={pending}"
        )
        if args.command == "verify" and not status.ready:
            return 1
        return 0
    except MigrationError as exc:
        print(f"migration_error={exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # Configuration errors (e.g. an out-of-range EMBEDDING_DIMENSIONS)
        # carry no credentials; print them plainly for the operator.
        print(f"migration_error={exc}", file=sys.stderr)
        return 2
    except Exception:
        # Do not echo arbitrary database exceptions: they can contain hosts,
        # usernames, or connection strings and this command is used by probes.
        print("migration_error=database_unavailable", file=sys.stderr)
        return 3
    finally:
        put_connection(connection)


if __name__ == "__main__":
    raise SystemExit(main())
