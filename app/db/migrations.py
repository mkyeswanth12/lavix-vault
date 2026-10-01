"""Small, checksummed PostgreSQL migration runner.

Migrations are plain SQL files named ``NNNN_description.sql``. The runner
applies all pending files in one transaction while holding a PostgreSQL
transaction-scoped advisory lock. Applied checksums are immutable: editing an
already-applied migration is treated as a deployment error.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MIGRATION_TABLE = "public.schema_migrations"
MIGRATION_LOCK_ID = 4_839_202_607_912_024  # Stable Lavix migration lock key.

_FILENAME_RE = re.compile(r"^(?P<version>[0-9]{4,})_(?P<name>[a-z0-9][a-z0-9_]*)\.sql$")

_CREATE_MIGRATION_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} (
    version BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    checksum CHAR(64) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT schema_migrations_checksum_format
        CHECK (checksum ~ '^[0-9a-f]{{64}}$')
)
"""

_SELECT_HISTORY_SQL = f"""
SELECT version, name, checksum, applied_at
FROM {MIGRATION_TABLE}
ORDER BY version
"""


class MigrationError(RuntimeError):
    """Base class for migration failures safe to show in deployment logs."""


class MigrationDiscoveryError(MigrationError):
    """The local migration set is absent, malformed, or ambiguous."""


class MigrationChecksumError(MigrationError):
    """An applied migration no longer matches its local SQL file."""


class MigrationHistoryError(MigrationError):
    """Database history cannot be reconciled with the local migration set."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    checksum: str
    sql: str
    path: Path


@dataclass(frozen=True, slots=True)
class AppliedMigration:
    version: int
    name: str
    checksum: str
    applied_at: Any = None


@dataclass(frozen=True, slots=True)
class MigrationReport:
    head_version: int
    previous_version: int
    current_version: int
    applied: tuple[Migration, ...]

    @property
    def applied_versions(self) -> tuple[int, ...]:
        return tuple(migration.version for migration in self.applied)


def _checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def discover_migrations(directory: Path | str = DEFAULT_MIGRATIONS_DIR) -> tuple[Migration, ...]:
    """Load and validate the complete local SQL migration set."""

    migration_dir = Path(directory)
    if not migration_dir.is_dir():
        raise MigrationDiscoveryError(f"Migration directory does not exist: {migration_dir}")

    migrations: list[Migration] = []
    seen_versions: dict[int, Path] = {}
    for path in sorted(migration_dir.glob("*.sql")):
        match = _FILENAME_RE.fullmatch(path.name)
        if not match:
            raise MigrationDiscoveryError(
                f"Invalid migration filename {path.name!r}; expected NNNN_description.sql"
            )

        version = int(match.group("version"))
        if version in seen_versions:
            raise MigrationDiscoveryError(
                f"Duplicate migration version {version}: {seen_versions[version].name} and {path.name}"
            )

        content = path.read_bytes()
        if not content.strip():
            raise MigrationDiscoveryError(f"Migration is empty: {path.name}")

        seen_versions[version] = path
        migrations.append(
            Migration(
                version=version,
                name=match.group("name"),
                checksum=_checksum(content),
                sql=content.decode("utf-8"),
                path=path,
            )
        )

    if not migrations:
        raise MigrationDiscoveryError(f"No SQL migrations found in {migration_dir}")

    migrations.sort(key=lambda migration: migration.version)
    return tuple(migrations)


def _row_value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        return row[index]
    return getattr(row, key)


def rows_to_history(rows: Sequence[Any]) -> tuple[AppliedMigration, ...]:
    return tuple(
        AppliedMigration(
            version=int(_row_value(row, "version", 0)),
            name=str(_row_value(row, "name", 1)),
            checksum=str(_row_value(row, "checksum", 2)).strip(),
            applied_at=_row_value(row, "applied_at", 3),
        )
        for row in rows
    )


def validate_history(
    migrations: Sequence[Migration],
    history: Sequence[AppliedMigration],
) -> None:
    """Verify that database history is an unmodified prefix of local files."""

    local_by_version = {migration.version: migration for migration in migrations}
    applied_versions: list[int] = []

    for applied in history:
        local = local_by_version.get(applied.version)
        if local is None:
            raise MigrationHistoryError(f"Database contains unknown migration version {applied.version}")
        if applied.name != local.name:
            raise MigrationHistoryError(
                f"Migration {applied.version} name differs: database={applied.name!r}, local={local.name!r}"
            )
        if applied.checksum != local.checksum:
            raise MigrationChecksumError(
                f"Migration {applied.version}_{local.name}.sql checksum differs from database history"
            )
        applied_versions.append(applied.version)

    expected_prefix = [migration.version for migration in migrations[: len(history)]]
    if applied_versions != expected_prefix:
        raise MigrationHistoryError(
            "Applied migration history is not a contiguous prefix of local migrations"
        )


def load_history(cursor: Any) -> tuple[AppliedMigration, ...]:
    cursor.execute(_SELECT_HISTORY_SQL)
    return rows_to_history(cursor.fetchall())


def apply_migrations(
    connection: Any,
    directory: Path | str = DEFAULT_MIGRATIONS_DIR,
) -> MigrationReport:
    """Apply pending migrations atomically and return the resulting version."""

    migrations = discover_migrations(directory)
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
        cursor.execute(_CREATE_MIGRATION_TABLE_SQL)
        history = load_history(cursor)
        validate_history(migrations, history)

        previous_version = history[-1].version if history else 0
        applied_versions = {migration.version for migration in history}
        pending = tuple(migration for migration in migrations if migration.version not in applied_versions)

        for migration in pending:
            cursor.execute(migration.sql)
            cursor.execute(
                f"""
                INSERT INTO {MIGRATION_TABLE} (version, name, checksum)
                VALUES (%s, %s, %s)
                """,
                (migration.version, migration.name, migration.checksum),
            )

        connection.commit()
        head_version = migrations[-1].version
        return MigrationReport(
            head_version=head_version,
            previous_version=previous_version,
            current_version=head_version,
            applied=pending,
        )
    except Exception:
        connection.rollback()
        raise
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()
