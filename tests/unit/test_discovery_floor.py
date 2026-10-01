"""Discovery floor: unscoped sub-threshold retrieval returns nothing."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any

from app.agent.retrieval import (
    VaultRetriever,
    _apply_discovery_floor,
    _effective_score,
)
from app.ingestion.embedding import PreparedEmbeddings


def _row(chunk: str, file_id: int, score: float | None, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "chunk_id": chunk,
        "file_id": file_id,
        "revision": 1,
        "ordinal": 0,
        "content": f"content of {chunk}",
        "display_name": f"file-{file_id}.pdf",
        "mime_type": "application/pdf",
        "quick_summary": "",
        "quick_tags": [],
        "doc_type": "",
        "intelligence_status": "",
        "section_path": [],
        "provenance": [],
    }
    if score is not None:
        row["rerank_score"] = score
    row.update(extra)
    return row


def test_effective_score_prefers_best_available() -> None:
    assert _effective_score({"rerank_score": 0.2, "score": 0.9}) == 0.9
    assert _effective_score({"score": 0.5}) == 0.5
    assert _effective_score({}) is None


def test_floor_drops_everything_below() -> None:
    rows = [_row("a", 1, 0.22), _row("b", 2, 0.12)]
    assert _apply_discovery_floor(rows, floor=0.35) == []


def test_floor_keeps_above_and_literal_pins() -> None:
    good = _row("good", 1, 0.55)
    pin = _row("pin", 2, 0.10, literal_match=True)
    junk = _row("junk", 3, 0.22)
    assert _apply_discovery_floor([junk, good, pin], floor=0.35) == [good, pin]


class _Embedding:
    async def embed(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        return PreparedEmbeddings(vectors=((0.1, 0.2, 0.3),), model="t", dimension=3, fingerprint="f")


class _Reranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    async def rerank(self, query, rows, *, top_k):  # type: ignore[no-untyped-def]
        for row, score in zip(rows, self.scores, strict=False):
            row["rerank_score"] = score
        return rows[:top_k]


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def execute(self, query, params):  # type: ignore[no-untyped-def]
        self._query = query

    def fetchall(self):  # type: ignore[no-untyped-def]
        if "WITH words" in self._query:
            return []
        return [dict(row) for row in self._rows]

    def fetchone(self):  # type: ignore[no-untyped-def]
        return None  # no saved admin config -> settings defaults


class _Conn:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def cursor(self):  # type: ignore[no-untyped-def]
        return self._cursor

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _search(rows: list[dict[str, Any]], scores: list[float], file_ids: Any) -> list[dict[str, Any]]:
    @contextmanager
    def connections():
        yield _Conn(_Cursor(rows))

    retriever = VaultRetriever(
        connections,  # type: ignore[arg-type]
        embedding_client=_Embedding(),  # type: ignore[arg-type]
        reranker=_Reranker(scores),  # type: ignore[arg-type]
    )
    return asyncio.run(
        retriever.search(user_id=41, query="find me", file_ids=file_ids, top_k=8, deep_search=False)
    )


def test_unscoped_all_below_floor_returns_nothing() -> None:
    rows = [_row("a", 1, None), _row("b", 2, None), _row("c", 1, None)]
    assert _search(rows, [0.22, 0.22, 0.12], None) == []


def test_unscoped_keeps_genuine_match_only() -> None:
    rows = [_row("junk", 1, None), _row("real", 2, None)]
    result = _search(rows, [0.22, 0.667], None)
    assert [item["chunk_id"] for item in result] == ["real"]


def test_scoped_retrieval_keeps_best_effort_below_floor() -> None:
    # @-tagged runs are unaffected: the old keep-all fallback still applies.
    rows = [_row("a", 7, None), _row("b", 7, None)]
    result = _search(rows, [0.12, 0.10], [7])
    assert [item["chunk_id"] for item in result] == ["a", "b"]
