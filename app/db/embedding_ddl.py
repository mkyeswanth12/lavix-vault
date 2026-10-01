"""Fresh-database embedding dimension sizing (migrator wrapper, not SQL).

Plain SQL migrations cannot read deployment configuration, so dimension
sizing lives here in Python and runs at the end of `migrate up`. Rules:

- Configured dimensions already match the columns: do nothing.
- Tables hold rows: REFUSE with a reembed pointer (never silently strand
  existing vectors behind a new dimension).
- Tables are empty (fresh install): ALTER the columns to the configured
  type/dimensions and rebuild the HNSW indexes. Zero data risk by
  construction — there is no data yet.
"""

from __future__ import annotations

from typing import Any

from app.db.migrations import MigrationError

from .embedding_store import (
    EMBEDDING_TABLES,
    column_info,
    column_type_for_dimensions,
    index_ops_for_type,
    table_row_count,
)

_INDEX_NAMES = {
    # name -> WHERE predicate, mirroring 0001/0014 exactly.
    "document_chunks": ("idx_document_chunks_embedding_hnsw", "WHERE embedding IS NOT NULL"),
    "graph_memory_items": ("idx_graph_memory_items_embedding", ""),
}


def ensure_fresh_dimensions(connection: Any, *, dimensions: int, model: str) -> str:
    """Size empty embedding columns to `dimensions`. Returns a report line."""
    column_type = column_type_for_dimensions(dimensions)
    ops = index_ops_for_type(column_type)
    reports: list[str] = []
    for table in EMBEDDING_TABLES:
        info = column_info(connection, table)
        if info is None:
            reports.append(f"{table}: no embedding column yet (migrations pending?)")
            continue
        current_type, current_dims = info
        if current_dims == dimensions and current_type == column_type:
            reports.append(f"{table}: already {current_type}({current_dims})")
            continue
        rows = table_row_count(connection, table)
        if rows:
            raise MigrationError(
                f"{table} holds {rows} rows of {current_type}({current_dims}) vectors "
                f"but EMBEDDING_DIMENSIONS={dimensions}: either revert "
                f"EMBEDDING_DIMENSIONS to {current_dims}, or migrate the data with "
                f"docker compose run --rm api python -m app.cli reembed "
                f"--model {model} --dimensions {dimensions} "
                f"(stop the ingestion workers first)"
            )
        cursor = connection.cursor()
        try:
            index_name, predicate = _INDEX_NAMES[table]
            cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            cursor.execute(
                f"ALTER TABLE {table} ALTER COLUMN embedding "
                f"TYPE {column_type}({dimensions}) USING embedding::{column_type}"
            )
            cursor.execute(
                f"CREATE INDEX {index_name} ON {table} "
                f"USING hnsw (embedding {ops}) {predicate}".rstrip()
            )
        finally:
            close = getattr(cursor, "close", None)
            if close is not None:
                close()
        reports.append(f"{table}: resized empty column to {column_type}({dimensions})")
    connection.commit()
    return "; ".join(reports)
