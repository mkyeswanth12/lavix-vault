"""Startup embedding-dimension self-check (api + ingestion worker).

Embeds one test string with the configured model and compares the vector
length against EMBEDDING_DIMENSIONS and the database column dimension.
Mismatches fail closed with an actionable message pointing at reembed;
an unreachable Ollama only warns (preflight covers availability, and the
API must stay up for non-AI administration).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_REEMBED_HINT = (
    "use an {expected}-dim model, or run: "
    "docker compose run --rm api python -m app.cli reembed "
    "--model {model} --dimensions {expected}"
)

_TEST_STRING = "Lavix embedding self-check"


def _reembed_message(model: str, produced: int, expected: int) -> str:
    return (
        f"model {model} outputs {produced} dims, database expects {expected}: "
        + _REEMBED_HINT.format(model=model, produced=produced, expected=expected)
    )


def _probe_dimensions(model: str, url: str) -> int | None:
    """Embed the test string; return the vector length, or None when Ollama
    is unreachable (warn-only path)."""
    try:
        import httpx

        response = httpx.post(
            url, json={"model": model, "input": _TEST_STRING}, timeout=60
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        return len(vector)
    except Exception as exc:
        logger.warning("embedding self-check probe unavailable (%s); skipping", type(exc).__name__)
        return None


def _read_embedding_config(connection: Any) -> dict[str, Any] | None:
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT model, dimensions, status FROM embedding_config WHERE singleton_id = 1")
        row = cursor.fetchone()
    except Exception:
        return None
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()
    if row is None:
        return None
    if isinstance(row, dict):
        return {"model": row["model"], "dimensions": int(row["dimensions"]), "status": row["status"]}
    return {"model": row[0], "dimensions": int(row[1]), "status": row[2]}


def check_embedding_deployment() -> None:
    """Fail closed on dimension/model drift; warn-only when Ollama is down."""
    from app.config import settings
    from app.database import get_db
    from app.db.embedding_store import EMBEDDING_TABLES, column_info, table_row_count

    model = settings.embedding_model_name
    expected = settings.embedding_dimension  # validates the 1..4000 range
    with get_db() as connection:
        recorded = _read_embedding_config(connection)
        column_dims: dict[str, int] = {}
        for table in EMBEDDING_TABLES:
            info = column_info(connection, table)
            if info is not None:
                column_dims[table] = info[1]
        vectors_present = any(
            table_row_count(connection, table) > 0 for table in EMBEDDING_TABLES
        )
        if recorded is None and not vectors_present:
            # Fresh install: record what this database was built for.
            from app.ingestion.embedding import EmbeddingSettings

            fingerprint = EmbeddingSettings(
                url=settings.embedding_api_url, model=model, dimension=expected
            ).fingerprint
            cursor = connection.cursor()
            try:
                cursor.execute(
                    "INSERT INTO embedding_config "
                    "(singleton_id, model, dimensions, fingerprint, status) "
                    "VALUES (1, %s, %s, %s, 'active') "
                    "ON CONFLICT (singleton_id) DO NOTHING",
                    (model, expected, fingerprint),
                )
            finally:
                close = getattr(cursor, "close", None)
                if close is not None:
                    close()
        elif recorded is not None and vectors_present:
            if recorded["model"] != model or recorded["dimensions"] != expected:
                raise RuntimeError(
                    f"database holds {recorded['model']} vectors "
                    f"({recorded['dimensions']} dims) but the deployment configures "
                    f"{model} ({expected} dims): refusing to silently mix models. "
                    + _REEMBED_HINT.format(
                        model=model, produced=recorded["dimensions"], expected=expected
                    )
                )

    produced = _probe_dimensions(model, settings.embedding_api_url)
    if produced is None:
        return
    if produced != expected:
        raise RuntimeError(_reembed_message(model, produced, expected))
    for table, dims in column_dims.items():
        if dims != produced:
            raise RuntimeError(
                f"model {model} outputs {produced} dims, but {table} holds "
                f"{dims}-dim vectors: "
                + _REEMBED_HINT.format(model=model, produced=produced, expected=dims)
            )
    logger.info("embedding self-check ok: %s outputs %d dims", model, produced)
