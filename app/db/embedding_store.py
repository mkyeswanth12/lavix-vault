"""Embedding storage helpers: pgvector type selection and column inspection.

Index limits (pgvector 0.8.x, verified against the pinned image docs):
HNSW/IVFFlat indexes support `vector` up to 2000 dimensions and `halfvec`
up to 4000 dimensions. Anything larger is refused with a clear message —
never silently left unindexed.
"""

from __future__ import annotations

from typing import Any

VECTOR_INDEX_LIMIT = 2000
HALFVEC_INDEX_LIMIT = 4000

EMBEDDING_TABLES = ("document_chunks", "graph_memory_items")

_INDEX_OPS = {"vector": "vector_cosine_ops", "halfvec": "halfvec_cosine_ops"}


def column_type_for_dimensions(dimensions: int) -> str:
    """Return the pgvector column type for a dimension count."""
    if not 1 <= dimensions <= HALFVEC_INDEX_LIMIT:
        raise ValueError(
            f"embedding dimensions must be between 1 and {HALFVEC_INDEX_LIMIT} "
            f"(2000 or less uses vector+HNSW, 2001-{HALFVEC_INDEX_LIMIT} uses "
            f"halfvec+HNSW), got {dimensions}"
        )
    if dimensions <= VECTOR_INDEX_LIMIT:
        return "vector"
    return "halfvec"


def index_ops_for_type(column_type: str) -> str:
    """Return the HNSW opclass for a pgvector column type."""
    try:
        return _INDEX_OPS[column_type]
    except KeyError:
        raise ValueError(f"unsupported embedding column type: {column_type!r}") from None


def column_info(connection: Any, table: str) -> tuple[str, int] | None:
    """Return (column_type, dimensions) for a table's embedding column.

    Returns None when the table or column does not exist (e.g. a fresh
    database before migrations run).
    """
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT format_type(atttypid, atttypmod) AS fmt
            FROM pg_attribute
            WHERE attrelid = %s::regclass AND attname = 'embedding' AND NOT attisdropped
            """,
            (table,),
        )
        row = cursor.fetchone()
    except Exception:
        return None
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()
    if row is None:
        return None
    formatted = row["fmt"] if isinstance(row, dict) else row[0]
    text = str(formatted or "").strip()
    if text.startswith("halfvec"):
        column_type = "halfvec"
    elif text.startswith("vector"):
        column_type = "vector"
    else:
        return None
    try:
        dimensions = int(text.split("(", 1)[1].rstrip(")"))
    except (IndexError, ValueError):
        return None
    return column_type, dimensions


def table_row_count(connection: Any, table: str) -> int:
    """Return the row count of a table, or 0 when it does not exist."""
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) AS cnt FROM {table}")
        row = cursor.fetchone()
    except Exception:
        return 0
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()
    if isinstance(row, dict):
        return int(row.get("cnt", 0) or 0)
    if row is None:
        return 0
    return int(row[0] or 0)


def resolve_vector_cast(connection: Any, table: str) -> str:
    """Return the ::cast suffix ("vector" or "halfvec") for a table.

    Read fresh on every call (one catalog lookup): always correct across a
    reembed switch with no process restart. Falls back to "vector" when the
    column cannot be inspected, preserving historical behavior.
    """
    try:
        info = column_info(connection, table)
    except Exception:
        return "vector"
    if info is None:
        return "vector"
    column_type, _dimensions = info
    return column_type if column_type in _INDEX_OPS else "vector"
