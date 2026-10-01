import asyncio
from contextlib import contextmanager

import pytest

from app.agent.retrieval import (
    HYBRID_SEARCH_SQL,
    InfinityReranker,
    VaultRetriever,
    _literal_query,
    _vector_literal,
)
from app.ingestion.embedding import PreparedEmbeddings


class FakeEmbeddingClient:
    async def embed(self, texts, **kwargs):
        assert texts == ["find me"]
        return PreparedEmbeddings(
            vectors=((0.1, 0.2, 0.3),),
            model="test",
            dimension=3,
            fingerprint="emb_test",
        )


class FakeReranker:
    async def rerank(self, query, rows, *, top_k):
        assert query == "find me"
        rows[0]["rerank_score"] = 0.97
        return rows[:top_k]


class ScoreReranker:
    def __init__(self, scores):
        self.scores = scores

    async def rerank(self, query, rows, *, top_k):
        assert query == "find me"
        for row, score in zip(rows, self.scores, strict=True):
            row["rerank_score"] = score
        return rows[:top_k]


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = None
        self.params = None
        self.calls = []

    def execute(self, query, params):
        self.calls.append((query, params))
        # search() now issues a second statement (rare-term pin lookup);
        # keep legacy single-statement assertions pointed at the main query.
        if "vector_candidates" in query:
            self.query = query
            self.params = params

    def fetchall(self):
        if self.calls and "WITH words" in self.calls[-1][0]:
            return []
        return self.rows


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def test_hybrid_retrieval_is_tenant_revision_and_consent_fenced():
    cursor = FakeCursor(
        [
            {
                "chunk_id": "chk_1",
                "file_id": 7,
                "revision": 3,
                "ordinal": 0,
                "content": "answer",
                "display_name": "report.pdf",
                "mime_type": "application/pdf",
                "quick_summary": "Quarterly launch report",
                "quick_tags": ["launch", "finance"],
                "doc_type": "report",
                "intelligence_status": "model",
                "section_path": ["Results"],
                "provenance": [{"page": 2}],
                "score": 0.8,
            }
        ]
    )

    @contextmanager
    def connections():
        yield FakeConnection(cursor)

    retriever = VaultRetriever(
        connections,
        embedding_client=FakeEmbeddingClient(),
        reranker=FakeReranker(),
    )
    result = asyncio.run(
        retriever.search(
            user_id=41,
            query="find me",
            file_ids=[7],
            top_k=4,
            deep_search=False,
        )
    )

    assert "dc.user_id = %s" in HYBRID_SEARCH_SQL
    assert "f.user_id = %s" in HYBRID_SEARCH_SQL
    assert "f.user_granted_ai_access = TRUE" in HYBRID_SEARCH_SQL
    assert "f.ai_status = 'ready'" not in HYBRID_SEARCH_SQL
    assert "f.current_revision = dc.revision" in HYBRID_SEARCH_SQL
    assert "dr.status = 'current'" in HYBRID_SEARCH_SQL
    assert "f.quick_summary, f.quick_tags" in HYBRID_SEARCH_SQL
    assert "to_jsonb(f)->>'doc_type' AS doc_type" in HYBRID_SEARCH_SQL
    assert "to_jsonb(f)->>'intelligence_status' AS intelligence_status" in HYBRID_SEARCH_SQL
    assert cursor.params[:5] == (41, 41, 41, [7], [7])
    assert result[0]["file_id"] == 7
    assert result[0]["score"] == 0.97
    assert result[0]["match_percentage"] == 97
    assert result[0]["reranked"] is True
    assert result[0]["provenance"] == [{"page": 2}]
    assert result[0]["summary"] == "Quarterly launch report"
    assert result[0]["tags"] == ["launch", "finance"]
    assert result[0]["doc_type"] == "report"


def test_vector_literal_rejects_empty_and_non_finite_values():
    assert _vector_literal([0.125, -1]) == "[0.125,-1]"
    with pytest.raises(ValueError, match="non-finite"):
        _vector_literal([])
    with pytest.raises(ValueError, match="non-finite"):
        _vector_literal([float("nan")])


def test_literal_query_only_pins_identifiers_and_phrases():
    assert _literal_query("HTML-4096") == "html-4096"
    assert _literal_query("  exact phrase  ") == "exact phrase"
    assert _literal_query("warranty") is None
    assert _literal_query("AI") is None


def test_exact_literal_survives_rerank_threshold(monkeypatch):
    monkeypatch.setenv("RERANK_MIN_SCORE", "0.01")
    cursor = FakeCursor(
        [
            {
                "chunk_id": "exact",
                "file_id": 7,
                "revision": 1,
                "ordinal": 0,
                "content": "The retrieval marker is HTML-4096.",
                "display_name": "format-proof.html",
                "mime_type": "text/html",
                "literal_match": True,
                "score": 0.08,
            },
            {
                "chunk_id": "semantic",
                "file_id": 8,
                "revision": 1,
                "ordinal": 0,
                "content": "A semantically similar but different document.",
                "display_name": "other.txt",
                "mime_type": "text/plain",
                "literal_match": False,
                "score": 0.81,
            },
        ]
    )

    @contextmanager
    def connections():
        yield FakeConnection(cursor)

    class CopyingReranker:
        async def rerank(self, query, rows, *, top_k):
            assert query == "find me"
            ranked = []
            for row, score in zip(rows, (0.001, 0.91), strict=True):
                item = dict(row)
                item["rerank_score"] = score
                ranked.append(item)
            return ranked[:top_k]

    retriever = VaultRetriever(
        connections,
        embedding_client=FakeEmbeddingClient(),
        reranker=CopyingReranker(),
        reranking_enabled=lambda: True,
    )
    result = asyncio.run(
        retriever.search(
            user_id=41,
            query="find me",
            file_ids=None,
            top_k=2,
            deep_search=False,
        )
    )

    assert [item["file_id"] for item in result] == [7, 8]
    assert result[0]["reranked"] is False
    assert "literal_candidates AS MATERIALIZED" in HYBRID_SEARCH_SQL
    assert "ORDER BY literal_match DESC" in HYBRID_SEARCH_SQL


def test_disabled_reranking_uses_hybrid_order(monkeypatch):
    monkeypatch.setenv("ENABLE_RERANKING", "false")
    cursor = FakeCursor(
        [
            {
                "chunk_id": "chk_1",
                "file_id": 7,
                "revision": 1,
                "ordinal": 0,
                "content": "hybrid result",
                "display_name": "report.txt",
                "mime_type": "text/plain",
                "quick_summary": "File: report.txt",
                "quick_tags": ["report", "txt"],
                "score": 0.8,
            }
        ]
    )

    @contextmanager
    def connections():
        yield FakeConnection(cursor)

    class UnexpectedReranker:
        async def rerank(self, *_args, **_kwargs):
            pytest.fail("disabled reranking must not call Infinity")

    retriever = VaultRetriever(
        connections,
        embedding_client=FakeEmbeddingClient(),
        reranker=UnexpectedReranker(),
    )
    result = asyncio.run(
        retriever.search(
            user_id=41,
            query="find me",
            file_ids=None,
            top_k=4,
            deep_search=False,
        )
    )

    assert result[0]["file_id"] == 7
    assert result[0]["reranked"] is False
    assert result[0]["summary"] is None
    assert result[0]["tags"] == []
    assert result[0]["doc_type"] is None


def test_database_model_configuration_controls_reranking_per_search():
    cursor = FakeCursor(
        [
            {
                "chunk_id": "chk_1",
                "file_id": 7,
                "revision": 1,
                "ordinal": 0,
                "content": "hybrid result",
                "display_name": "report.txt",
                "mime_type": "text/plain",
                "score": 0.8,
            }
        ]
    )

    @contextmanager
    def connections():
        yield FakeConnection(cursor)

    class UnexpectedReranker:
        async def rerank(self, *_args, **_kwargs):
            pytest.fail("the current admin setting must disable Infinity")

    retriever = VaultRetriever(
        connections,
        embedding_client=FakeEmbeddingClient(),
        reranker=UnexpectedReranker(),
        reranking_enabled=lambda: False,
    )
    result = asyncio.run(
        retriever.search(
            user_id=41,
            query="find me",
            file_ids=None,
            top_k=4,
            deep_search=False,
        )
    )

    assert result[0]["reranked"] is False


# NOTE (BUG-003 fairness): an explicitly scoped file keeps its best hybrid
# chunk even when its rerank score falls below the floor, so one strong file
# can no longer starve a second requested file. The [7, 8] case below changed
# from [7] to [7, 8] deliberately; the hybrid floor still bars junk.
@pytest.mark.parametrize(
    ("file_ids", "scores", "expected_ids"),
    [
        (None, (0.87, 0.001), [7]),
        ([7, 8], (0.87, 0.001), [7, 8]),
        ([7, 8], (0.0009, 0.0004), [7, 8]),
    ],
)
def test_rerank_floor_preserves_low_scored_selected_file_summaries(
    monkeypatch,
    file_ids,
    scores,
    expected_ids,
):
    monkeypatch.setenv("RERANK_MIN_SCORE", "0.01")
    cursor = FakeCursor(
        [
            {
                "chunk_id": f"chk_{file_id}",
                "file_id": file_id,
                "revision": 1,
                "ordinal": 0,
                "content": content,
                "display_name": f"{file_id}.txt",
                "mime_type": "text/plain",
                "score": score,
            }
            for file_id, content, score in (
                (7, "the exact answer", 0.8),
                (8, "unrelated material", 0.4),
            )
        ]
    )

    @contextmanager
    def connections():
        yield FakeConnection(cursor)

    retriever = VaultRetriever(
        connections,
        embedding_client=FakeEmbeddingClient(),
        reranker=ScoreReranker(scores),
    )
    result = asyncio.run(
        retriever.search(
            user_id=41,
            query="find me",
            file_ids=file_ids,
            top_k=4,
            deep_search=False,
        )
    )

    assert [item["file_id"] for item in result] == expected_ids


def test_single_candidate_is_still_scored_by_infinity(monkeypatch):
    class FakeResponse:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"results": [{"index": 0, "relevance_score": 0.75}]}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, _url, *, json):
            assert json["documents"] == ["one candidate"]
            return FakeResponse()

    monkeypatch.setattr(
        "app.agent.retrieval.httpx.AsyncClient",
        lambda **_kwargs: FakeClient(),
    )
    rows = asyncio.run(
        InfinityReranker("http://reranker:7997").rerank(
            "find me",
            [{"content": "one candidate"}],
            top_k=1,
        )
    )

    assert rows[0]["rerank_score"] == 0.75


def test_rare_query_words_filters_and_caps():
    from app.agent.retrieval import _rare_query_words

    assert _rare_query_words("did we have RAJESH MAHESHWARI invoice?") == [
        "did",
        "have",
        "rajesh",
        "maheshwari",
        "invoice",
    ]
    assert _rare_query_words("a I x") == []
    assert _rare_query_words("spam spam spam eggs") == ["spam", "eggs"]
    assert len(_rare_query_words(" ".join(f"word{i}" for i in range(30)))) == 10


def test_rare_words_extraction_short_circuits_without_distinctive_words():
    # Rarity now resolves inside HYBRID_SEARCH_SQL (rare_file_counts CTE);
    # the Python side only extracts candidate words.
    from app.agent.retrieval import _rare_query_words

    assert _rare_query_words("a I x") == []


def test_rare_pin_ctes_exist_with_bounded_params():
    assert "rare_file_counts AS MATERIALIZED" in HYBRID_SEARCH_SQL
    assert "pin_candidates AS MATERIALIZED" in HYBRID_SEARCH_SQL
    assert "BETWEEN 1 AND 5" in HYBRID_SEARCH_SQL
    assert "SELECT * FROM pin_candidates" in HYBRID_SEARCH_SQL
    pin_block = HYBRID_SEARCH_SQL.split("pin_candidates AS MATERIALIZED")[1].split("scores AS (")[0]
    assert "TRUE AS literal_match" in pin_block
    assert "0.50::REAL AS summary_tag_score" in pin_block


def test_rare_pins_prepend_with_real_scores(monkeypatch):
    from app.ingestion.embedding import PreparedEmbeddings

    class AnyQueryEmbeddings:
        async def embed(self, texts, **kwargs):
            assert len(texts) == 1
            return PreparedEmbeddings(
                vectors=((0.1, 0.2, 0.3),),
                model="test",
                dimension=3,
                fingerprint="emb_test",
            )

    rows = [
        {
            "chunk_id": "chk_1",
            "file_id": 7,
            "revision": 1,
            "ordinal": 0,
            "content": "unrelated",
            "display_name": "other.txt",
            "mime_type": "text/plain",
            "quick_summary": None,
            "quick_tags": [],
            "doc_type": None,
            "intelligence_status": "model",
            "score": 0.30,
            "vector_score": 0.30,
            "lexical_score": 0.0,
            "literal_match": False,
            "summary_tag_score": 0.0,
        },
        {
            "chunk_id": "chk_9",
            "file_id": 218,
            "revision": 2,
            "ordinal": 0,
            "content": "Mr. RAJESH MAHESHWARI",
            "display_name": "1715773827.webp",
            "mime_type": "image/webp",
            "quick_summary": "tax invoice RAJESH MAHESHWARI",
            "quick_tags": ["warranty"],
            "doc_type": None,
            "intelligence_status": "model",
            "score": 0.14,
            "vector_score": 0.0,
            "lexical_score": 0.1,
            "literal_match": True,
            "summary_tag_score": 0.5,
        },
    ]

    class Cursor:
        def execute(self, query, params):
            pass

        def fetchall(self):
            return rows

    class Connection:
        def cursor(self):
            return Cursor()

    @contextmanager
    def connections():
        yield Connection()

    class NoRerank:
        async def rerank(self, query, rows, *, top_k):
            return rows[:top_k]

    retriever = VaultRetriever(
        connections,
        embedding_client=AnyQueryEmbeddings(),
        reranker=NoRerank(),
    )
    monkeypatch.setattr(retriever, "_reranking_enabled", lambda: False)
    result = asyncio.run(
        retriever.search(
            user_id=41,
            query="maheshwari",
            file_ids=None,
            top_k=2,
            deep_search=False,
        )
    )
    assert [item["file_id"] for item in result] == [218]
    assert result[0]["score"] > 0
    # The 0.30 unscoped non-pin row is dropped by the discovery floor
    # (settings.discovery_min_score); the literal pin survives regardless.


def test_summary_representative_is_deterministic_best_vector_chunk():
    assert "ORDER BY eligible.file_id, vc.vector_score DESC NULLS LAST" in HYBRID_SEARCH_SQL
    assert "0.44 * scores.summary_tag_score" in HYBRID_SEARCH_SQL


def test_rerank_input_is_bounded_for_interactive_latency(monkeypatch):
    import json as jsonlib

    import httpx

    import app.agent.retrieval as retrieval_module
    from app.agent.retrieval import InfinityReranker

    seen = {}

    def handler(request):
        seen["body"] = jsonlib.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"results": []})

    real_client = httpx.AsyncClient

    class PatchedClient(real_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(retrieval_module.httpx, "AsyncClient", PatchedClient)
    reranker = InfinityReranker()
    rows = [
        {"content": "x" * 2000, "chunk_id": f"chk_{i}", "file_id": 1, "revision": 1}
        for i in range(40)
    ]
    result = asyncio.run(reranker.rerank("find me", rows, top_k=10))
    assert result == rows[:10]
    assert len(seen["body"]["documents"]) == 40
    assert all(len(doc) <= 600 for doc in seen["body"]["documents"])


def test_search_caps_rerank_candidates():
    from app.agent.retrieval import VaultRetriever
    from app.ingestion.embedding import PreparedEmbeddings

    rows = [
        {
            "chunk_id": f"chk_{i}",
            "file_id": 7,
            "revision": 1,
            "ordinal": i,
            "content": "answer",
            "display_name": "report.txt",
            "mime_type": "text/plain",
            "quick_summary": None,
            "quick_tags": [],
            "doc_type": None,
            "intelligence_status": "model",
            "score": 0.5,
            "vector_score": 0.5,
            "lexical_score": 0.0,
            "literal_match": False,
            "summary_tag_score": 0.0,
        }
        for i in range(40)
    ]

    class Cursor:
        def execute(self, query, params):
            pass

        def fetchall(self):
            return rows

    class Connection:
        def cursor(self):
            return Cursor()

    @contextmanager
    def connections():
        yield Connection()

    class CountingReranker:
        def __init__(self):
            self.seen = None

        async def rerank(self, query, rows, *, top_k):
            self.seen = len(rows)
            return rows[:top_k]

    class AnyQueryEmbeddings:
        async def embed(self, texts, **kwargs):
            return PreparedEmbeddings(
                vectors=((0.1, 0.2, 0.3),),
                model="test",
                dimension=3,
                fingerprint="emb_test",
            )

    reranker = CountingReranker()
    retriever = VaultRetriever(
        connections,
        embedding_client=AnyQueryEmbeddings(),
        reranker=reranker,
    )
    asyncio.run(
        retriever.search(
            user_id=41,
            query="find me here",
            file_ids=None,
            top_k=10,
            deep_search=False,
        )
    )
    assert reranker.seen == 24


def _ranked_rows(n_files=3, per_file=4, base_score=0.9):
    rows = []
    ordinal = 0
    for file_id in range(7, 7 + n_files):
        for chunk in range(per_file):
            rows.append(
                {
                    "chunk_id": f"chk_{file_id}_{chunk}",
                    "file_id": file_id,
                    "revision": 1,
                    "ordinal": ordinal,
                    "content": f"content {file_id} {chunk}",
                    "display_name": f"file-{file_id}.pdf",
                    "mime_type": "application/pdf",
                    "quick_summary": "",
                    "quick_tags": [],
                    "doc_type": "",
                    "intelligence_status": "model",
                    "section_path": [],
                    "provenance": [],
                    "score": base_score - ordinal * 0.01,
                }
            )
            ordinal += 1
    return rows


def _search_with(rows, *, file_ids, top_k, file_scope=None, deep_search=False):
    cursor = FakeCursor(rows)

    @contextmanager
    def connections():
        yield FakeConnection(cursor)

    class PassthroughReranker:
        async def rerank(self, query, candidates, *, top_k):
            for row in candidates:
                row["rerank_score"] = 0.5
            return candidates[:top_k]

    retriever = VaultRetriever(
        connections,
        embedding_client=FakeEmbeddingClient(),
        reranker=PassthroughReranker(),
    )
    if file_scope is not None:
        retriever._configured_file_scope = lambda: file_scope
    return asyncio.run(
        retriever.search(
            user_id=41,
            query="find me",
            file_ids=file_ids,
            top_k=top_k,
            deep_search=deep_search,
        )
    )


def _file_ids(result):
    return [item["file_id"] for item in result]


def test_file_scope_caps_distinct_files_in_discovery():
    result = _search_with(_ranked_rows(3, 4), file_ids=None, top_k=20, file_scope=2)

    assert sorted(set(_file_ids(result))) == [7, 8]
    # Rank order preserved; surviving files keep all ranked chunks.
    assert _file_ids(result) == [7, 7, 7, 7, 8, 8, 8, 8]


def test_file_scope_default_five_keeps_current_width():
    result = _search_with(_ranked_rows(3, 4), file_ids=None, top_k=20)

    assert sorted(set(_file_ids(result))) == [7, 8, 9]


def test_file_scope_never_truncates_explicit_scope():
    result = _search_with(_ranked_rows(3, 4), file_ids=[7, 8, 9], top_k=20, file_scope=1)

    assert sorted(set(_file_ids(result))) == [7, 8, 9]


def test_top_k_scales_rerank_headroom():
    seen = []

    class RecordingReranker:
        async def rerank(self, query, candidates, *, top_k):
            seen.append(len(candidates))
            for row in candidates:
                row["rerank_score"] = 0.5
            return candidates[:top_k]

    cursor = FakeCursor(_ranked_rows(8, 8))

    @contextmanager
    def connections():
        yield FakeConnection(cursor)

    retriever = VaultRetriever(
        connections,
        embedding_client=FakeEmbeddingClient(),
        reranker=RecordingReranker(),
    )
    retriever._configured_file_scope = lambda: 100
    asyncio.run(
        retriever.search(user_id=41, query="find me", file_ids=None, top_k=5, deep_search=False)
    )
    asyncio.run(
        retriever.search(user_id=41, query="find me", file_ids=None, top_k=50, deep_search=False)
    )

    assert seen[0] == 24
    assert seen[1] == 50


def test_file_scope_and_top_k_are_independent():
    narrow = _search_with(_ranked_rows(4, 4), file_ids=None, top_k=20, file_scope=2)
    wide = _search_with(_ranked_rows(4, 4), file_ids=None, top_k=50, file_scope=2)

    assert sorted(set(_file_ids(narrow))) == [7, 8]
    assert sorted(set(_file_ids(wide))) == [7, 8]
    # Same surviving files in the same order regardless of top_k.
    assert [item["chunk_id"] for item in narrow] == [item["chunk_id"] for item in wide][: len(narrow)]
