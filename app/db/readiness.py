"""Non-mutating schema-head checks for readiness probes and deployments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .migrations import (
    DEFAULT_MIGRATIONS_DIR,
    MIGRATION_TABLE,
    MigrationChecksumError,
    MigrationDiscoveryError,
    MigrationHistoryError,
    discover_migrations,
    load_history,
    validate_history,
)


@dataclass(frozen=True, slots=True)
class SchemaHeadStatus:
    ready: bool
    reason: str
    current_version: int
    head_version: int
    pending_versions: tuple[int, ...] = ()


class SchemaNotReadyError(RuntimeError):
    """Raised when a caller requires, rather than probes, schema readiness."""

    def __init__(self, status: SchemaHeadStatus):
        self.status = status
        super().__init__(
            f"Database schema is not ready ({status.reason}); "
            f"current={status.current_version}, head={status.head_version}"
        )


def check_schema_head(
    connection: Any,
    directory: Path | str = DEFAULT_MIGRATIONS_DIR,
) -> SchemaHeadStatus:
    """Return schema readiness without creating tables or applying migrations.

    Unexpected database errors are intentionally collapsed to a stable reason so
    connection details and credentials cannot leak through a health response.
    """

    try:
        migrations = discover_migrations(directory)
    except MigrationDiscoveryError:
        return SchemaHeadStatus(False, "local_migrations_invalid", 0, 0)

    head_version = migrations[-1].version
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT to_regclass(%s) AS table_name", (MIGRATION_TABLE,))
        table_row = cursor.fetchone()
        if table_row is None:
            return SchemaHeadStatus(False, "migration_table_missing", 0, head_version)

        if isinstance(table_row, dict):
            table_name = table_row.get("table_name")
        elif isinstance(table_row, (tuple, list)):
            table_name = table_row[0]
        else:
            table_name = getattr(table_row, "table_name", None)

        if table_name is None:
            return SchemaHeadStatus(False, "migration_table_missing", 0, head_version)

        history = load_history(cursor)
        current_version = history[-1].version if history else 0
        try:
            validate_history(migrations, history)
        except MigrationChecksumError:
            return SchemaHeadStatus(False, "checksum_mismatch", current_version, head_version)
        except MigrationHistoryError:
            return SchemaHeadStatus(False, "history_mismatch", current_version, head_version)

        applied_versions = {migration.version for migration in history}
        pending = tuple(
            migration.version for migration in migrations if migration.version not in applied_versions
        )
        if pending:
            return SchemaHeadStatus(
                False,
                "pending_migrations",
                current_version,
                head_version,
                pending,
            )

        return SchemaHeadStatus(True, "ready", current_version, head_version)
    except Exception:
        return SchemaHeadStatus(False, "database_unavailable", 0, head_version)
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()


def require_schema_head(
    connection: Any,
    directory: Path | str = DEFAULT_MIGRATIONS_DIR,
) -> SchemaHeadStatus:
    status = check_schema_head(connection, directory)
    if not status.ready:
        raise SchemaNotReadyError(status)
    return status
