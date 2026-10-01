"""Embedding model migration (`python -m app.cli reembed`).

Moves every stored vector to a new model/dimensions without downtime and
without ever mixing models:

- Search keeps serving the OLD vectors until the atomic switch.
- Backfill is resumable: interrupting mid-run leaves old vectors live and
  the next run continues where it stopped (state in embedding_config).
- Counts are verified BEFORE the switch, and old vectors are dropped only
  AFTER the switch is re-verified.
- Covers document_chunks AND graph_memory_items.

Run it from a one-off container (the API with the new dimensions would
refuse to boot before the switch — that is the fail-closed check working)::

    docker compose stop ingestion-worker graph-memory-worker
    docker compose run --rm api python -m app.cli reembed \\
        --model nomic-embed-text --dimensions 768
    docker compose up -d ingestion-worker graph-memory-worker api
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from app.db.embedding_store import (
    EMBEDDING_TABLES,
    column_info,
    column_type_for_dimensions,
    index_ops_for_type,
    table_row_count,
)

_BATCH_SIZE = 50
_REEMBED_LOCK_KEY = 4_839_202_607_912_025  # Stable Lavix reembed lock key.

_TABLE_SOURCES = {
    # table -> (id column, text expression for re-embedding)
    "document_chunks": ("chunk_id", "embedding_text"),
    "graph_memory_items": (
        "id",
        "subject || ' ' || predicate || ' ' || object_value",
    ),
}

_INDEX_NAMES = {
    "document_chunks": ("idx_document_chunks_embedding_hnsw", "WHERE embedding IS NOT NULL"),
    "graph_memory_items": ("idx_graph_memory_items_embedding", ""),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate stored embeddings to a new model")
    parser.add_argument("--model", required=True, help="target Ollama embedding model")
    parser.add_argument("--dimensions", required=True, type=int, help="target vector dimensions")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    return parser


def _probe_model_dimensions(model: str, url: str) -> int:
    """Embed one string with the target model; return the vector length."""
    import httpx

    try:
        response = httpx.post(
            url, json={"model": model, "input": "Lavix reembed probe"}, timeout=120
        )
        response.raise_for_status()
        return len(response.json()["data"][0]["embedding"])
    except Exception as exc:
        print(f"error: target model {model!r} is unreachable: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1) from exc


def _read_config(connection: Any) -> dict[str, Any] | None:
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT model, dimensions, fingerprint, status, target_model, "
            "target_dimensions, backfilled_chunks FROM embedding_config "
            "WHERE singleton_id = 1"
        )
        row = cursor.fetchone()
    except Exception:
        connection.rollback()
        return None
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()
    if row is None:
        return None
    if isinstance(row, dict):
        return dict(row)
    keys = ("model", "dimensions", "fingerprint", "status", "target_model",
            "target_dimensions", "backfilled_chunks")
    return dict(zip(keys, row, strict=False))


def _execute(connection: Any, sql: str, params: tuple = (), *, fetch: bool = False) -> list[Any]:
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params)
        if fetch:
            return list(cursor.fetchall())
        return []
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model = args.model.strip()
    dimensions = args.dimensions
    if not model:
        print("error: --model must not be empty", file=sys.stderr)
        return 2
    try:
        column_type = column_type_for_dimensions(dimensions)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ops = index_ops_for_type(column_type)

    from app.config import settings
    from app.database import get_db

    url = settings.embedding_api_url
    probed = _probe_model_dimensions(model, url)
    if probed != dimensions:
        print(
            f"error: model {model!r} outputs {probed} dims, but --dimensions "
            f"is {dimensions}: re-run with --dimensions {probed}",
            file=sys.stderr,
        )
        return 2

    with get_db() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", (_REEMBED_LOCK_KEY,))
            row = cursor.fetchone()
            locked = row["pg_try_advisory_lock"] if isinstance(row, dict) else row[0]
        finally:
            close = getattr(cursor, "close", None)
            if close is not None:
                close()
        if not locked:
            print("error: another reembed run holds the lock", file=sys.stderr)
            return 1

        recorded = _read_config(connection)
        current: dict[str, tuple[str, int] | None] = {
            table: column_info(connection, table) for table in EMBEDDING_TABLES
        }
        counts = {table: table_row_count(connection, table) for table in EMBEDDING_TABLES}

        if args.dry_run:
            print(f"reembed plan: model {model!r}, dimensions {dimensions} ({column_type})")
            for table in EMBEDDING_TABLES:
                info = current[table]
                have = "missing" if info is None else f"{info[0]}({info[1]})"
                print(f"  {table}: {counts[table]} rows, current {have}")
                print(f"    -> add embedding_new {column_type}({dimensions}), "
                      f"backfill, verify, switch, rebuild {_INDEX_NAMES[table][0]}")
            print("dry-run: no changes made")
            return 0

        if recorded is not None and recorded.get("status") == "reembedding":
            if (recorded.get("target_model"), int(recorded.get("target_dimensions") or 0)) != (
                model, dimensions,
            ):
                print(
                    "error: an interrupted reembed targets "
                    f"{recorded.get('target_model')} "
                    f"({recorded.get('target_dimensions')} dims); finish it first "
                    "or reset embedding_config.status to 'active'",
                    file=sys.stderr,
                )
                return 1
            print(f"resuming interrupted reembed of {model} ({dimensions} dims)")
        else:
            from app.ingestion.embedding import EmbeddingSettings

            fingerprint = EmbeddingSettings(url=url, model=model, dimension=dimensions).fingerprint
            _execute(
                connection,
                "INSERT INTO embedding_config "
                "(singleton_id, model, dimensions, fingerprint, status, "
                "target_model, target_dimensions, backfilled_chunks) "
                "VALUES (1, %s, %s, %s, 'reembedding', %s, %s, 0) "
                "ON CONFLICT (singleton_id) DO UPDATE SET status='reembedding', "
                "target_model=EXCLUDED.target_model, "
                "target_dimensions=EXCLUDED.target_dimensions, "
                "backfilled_chunks=0, updated_at=NOW()",
                (
                    recorded["model"] if recorded else model,
                    int(recorded["dimensions"]) if recorded else dimensions,
                    recorded["fingerprint"] if recorded else fingerprint,
                    model, dimensions,
                ),
            )
            connection.commit()
            print(f"started reembed to {model} ({dimensions} dims, {column_type})")

        from app.ingestion.embedding import EmbeddingSettings, OllamaEmbeddingClient

        client_settings = EmbeddingSettings(
            url=url, model=model, dimension=dimensions, batch_size=16, concurrency=1
        )
        backfilled_total = 0
        for table in EMBEDDING_TABLES:
            id_column, text_expr = _TABLE_SOURCES[table]
            _execute(
                connection,
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "
                f"embedding_new {column_type}({dimensions})",
            )
            connection.commit()
            while True:
                rows = _execute(
                    connection,
                    f"SELECT {id_column} AS rid, {text_expr} AS text FROM {table} "
                    f"WHERE embedding IS NOT NULL AND embedding_new IS NULL "
                    f"LIMIT {_BATCH_SIZE}",
                    fetch=True,
                )
                if not rows:
                    break
                ids = [row["rid"] if isinstance(row, dict) else row[0] for row in rows]
                texts = [row["text"] if isinstance(row, dict) else row[1] for row in rows]
                prepared = asyncio.run(
                    OllamaEmbeddingClient(settings=client_settings).embed(texts)
                )
                for rid, vector in zip(ids, prepared.vectors, strict=True):
                    literal = "[" + ",".join(repr(float(value)) for value in vector) + "]"
                    _execute(
                        connection,
                        f"UPDATE {table} SET embedding_new = %s::{column_type} "
                        f"WHERE {id_column} = %s",
                        (literal, rid),
                    )
                backfilled_total += len(ids)
                _execute(
                    connection,
                    "UPDATE embedding_config SET backfilled_chunks = "
                    "backfilled_chunks + %s, updated_at = NOW() WHERE singleton_id = 1",
                    (len(ids),),
                )
                connection.commit()
                print(f"  {table}: backfilled {backfilled_total} chunks...")
        print(f"backfill complete: {backfilled_total} vectors from {model}")

        # Verify BEFORE the switch: every old vector must have a new one.
        for table in EMBEDDING_TABLES:
            old = _execute(
                connection, f"SELECT COUNT(*) AS cnt FROM {table} WHERE embedding IS NOT NULL",
                fetch=True,
            )
            new = _execute(
                connection, f"SELECT COUNT(*) AS cnt FROM {table} WHERE embedding_new IS NOT NULL",
                fetch=True,
            )
            old_count = old[0]["cnt"] if isinstance(old[0], dict) else old[0][0]
            new_count = new[0]["cnt"] if isinstance(new[0], dict) else new[0][0]
            print(f"  verify {table}: old={old_count} new={new_count}")
            if int(new_count) != int(old_count):
                print(
                    f"error: {table} count mismatch (old {old_count}, new {new_count}); "
                    "old vectors stay live, re-run to resume",
                    file=sys.stderr,
                )
                return 1
            sample = _execute(
                connection, f"SELECT embedding_new FROM {table} "
                f"WHERE embedding_new IS NOT NULL LIMIT 5",
                fetch=True,
            )
            for item in sample:
                value = item["embedding_new"] if isinstance(item, dict) else item[0]
                text = value if isinstance(value, str) else str(value)
                dims = text.count(",") + 1 if text.startswith("[") else -1
                if dims != dimensions:
                    print(
                        f"error: {table} sample has {dims} dims, want {dimensions}",
                        file=sys.stderr,
                    )
                    return 1
        print("verification passed: counts match, sample dimensions correct")

        # Atomic switch: renames + index rebuild + metadata update in one txn.
        cursor = connection.cursor()
        try:
            for table in EMBEDDING_TABLES:
                index_name, predicate = _INDEX_NAMES[table]
                suffix = f" {predicate}".rstrip()
                cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
                cursor.execute(f"ALTER TABLE {table} RENAME COLUMN embedding TO embedding_old")
                cursor.execute(f"ALTER TABLE {table} RENAME COLUMN embedding_new TO embedding")
                cursor.execute(
                    f"CREATE INDEX {index_name} ON {table} "
                    f"USING hnsw (embedding {ops}){suffix}"
                )
            cursor.execute(
                "UPDATE embedding_config SET model=%s, dimensions=%s, "
                "fingerprint=%s, status='active', target_model=NULL, "
                "target_dimensions=NULL, verified_at=NOW(), updated_at=NOW() "
                "WHERE singleton_id=1",
                (
                    model, dimensions,
                    EmbeddingSettings(url=url, model=model, dimension=dimensions).fingerprint,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            print("error: switch failed and was rolled back; old vectors stay live",
                  file=sys.stderr)
            raise
        finally:
            close = getattr(cursor, "close", None)
            if close is not None:
                close()
        print("switch complete: new vectors are live")

        # Re-verify on the live column, then drop the old vectors.
        for table in EMBEDDING_TABLES:
            live = _execute(
                connection, f"SELECT COUNT(*) AS cnt FROM {table} WHERE embedding IS NOT NULL",
                fetch=True,
            )
            live_count = live[0]["cnt"] if isinstance(live[0], dict) else live[0][0]
            print(f"  live {table}: {live_count} vectors")
        for table in EMBEDDING_TABLES:
            _execute(connection, f"ALTER TABLE {table} DROP COLUMN IF EXISTS embedding_old")
        connection.commit()
        print("old vectors dropped")
        print("done: restart the api and ingestion workers to pick up the new model")
        return 0
