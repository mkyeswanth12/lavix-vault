"""Regression tests for BUG-003: one scoped file must not starve another.

Fairness re-attaches each missing explicitly-scoped file's best hybrid
chunk (floor-gated). Global thresholds, ordering, consent and exclusion
semantics are unchanged.
"""
import asyncio
from contextlib import contextmanager

from app.agent.retrieval import VaultRetriever, _ensure_scoped_file_representation
from app.ingestion.embedding import PreparedEmbeddings


def row(chunk_id, file_id, score, **over):
    base = {
        "chunk_id": chunk_id,
        "file_id": file_id,
        "revision": 1,
        "ordinal": 0,
        "content": f"content of {chunk_id}",
        "display_name": f"file-{file_id}.pdf",
        "mime_type": "application/pdf",
        "literal_match": False,
        "score": score,
    }
    base.update(over)
    return base


class FakeEmbeddingClient:
    async def embed(self, texts, **kwargs):
        return PreparedEmbeddings(vectors=((0.1, 0.2, 0.3),), model="t", dimension=3, fingerprint="f")


class StarvingReranker:
    """Reranks file 8 above the floor and file 7 below it."""

    async def rerank(self, query, rows, *, top_k):
        out = []
        for r in rows:
            item = dict(r)
            item["rerank_score"] = 0.9 if r["file_id"] == 8 else 0.05
            out.append(item)
        out.sort(key=lambda r: r["rerank_score"], reverse=True)
        return out[:top_k]


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, query, params):
        pass

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def search(rows, file_ids, top_k=4, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("RERANK_MIN_SCORE", "0.15")
    cursor = FakeCursor(rows)

    @contextmanager
    def connections():
        yield FakeConnection(cursor)

    retriever = VaultRetriever(
        connections,
        embedding_client=FakeEmbeddingClient(),
        reranker=StarvingReranker(),
        reranking_enabled=lambda: True,
    )
    return asyncio.run(
        retriever.search(user_id=41, query="find me", file_ids=file_ids, top_k=top_k, deep_search=False)
    )


def test_missing_scoped_file_is_reattached():
    rows = [
        row("a1", 8, 0.9),
        row("a2", 8, 0.8),
        row("b1", 7, 0.5),
        row("b2", 7, 0.4),
    ]
    result = search(rows, [7, 8], top_k=2)
    files = {r["file_id"] for r in result}
    assert files == {7, 8}
    assert any(r["chunk_id"] == "b1" for r in result)


def test_single_file_scope_unchanged():
    rows = [row("a1", 8, 0.9), row("b1", 7, 0.5)]
    result = search(rows, [8], top_k=2)
    assert {r["file_id"] for r in result} == {8}


def test_irrelevant_file_stays_out():
    rows = [row("a1", 8, 0.9), row("b1", 7, 0.05)]
    result = search(rows, [7, 8], top_k=2)
    assert {r["file_id"] for r in result} == {8}


def test_helper_never_exceeds_bounded_growth():
    ranked = [row("a1", 8, 0.9)]
    pool = [row("a1", 8, 0.9), row("b1", 7, 0.5)]
    out = _ensure_scoped_file_representation(ranked, pool, [7, 8], top_k=1, min_score=0.15)
    assert len(out) == 2
    assert [r["chunk_id"] for r in out] == ["a1", "b1"]
