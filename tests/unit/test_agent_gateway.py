from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.capability import CapabilitySigner
from app.agent.evidence import RunEvidenceStore
from app.agent.gateway import create_agent_gateway_router


class FakeRetriever:
    def __init__(self):
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "kind": "vault",
                "file_id": 7,
                "revision": 2,
                "chunk_id": "chunk-1",
                "filename": "report.pdf",
                "content": "evidence",
            }
        ]


def _app(retriever, web_search, graph_recall=None):
    signer = CapabilitySigner("g" * 32, clock=lambda: 1_000)
    store = RunEvidenceStore()

    class FakeReranker:
        async def rerank(self, query, rows, *, top_k):
            return list(rows)

    app = FastAPI()
    kwargs = {
        "signer": signer,
        "retriever": retriever,
        "evidence_store": store,
        "web_search": web_search,
        "reranker": FakeReranker(),
    }
    if graph_recall is not None:
        kwargs["graph_recall"] = graph_recall
    app.include_router(create_agent_gateway_router(**kwargs))
    return app, signer, store


def _token(signer, run_id, *, files=(7, 8), web=True, deep=True):
    return signer.mint(
        run_id=run_id,
        user_id=44,
        file_ids=files,
        web_search_enabled=web,
        deep_search=deep,
    )


def test_vault_gateway_enforces_token_subject_and_file_intersection():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return []

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    headers = {
        "X-Lavix-Capability": _token(signer, run_id),
        "X-Lavix-Run-ID": run_id,
    }
    response = TestClient(app).post(
        "/search-vault",
        headers=headers,
        json={"query": "quarterly result", "file_ids": [8, 999], "top_k": 5, "deep_search": True},
    )

    assert response.status_code == 200
    assert response.json()["evidence"][0]["id"] == "V1"
    assert retriever.calls == [
        {
            "user_id": 44,
            "query": "quarterly result",
            "file_ids": [8],
            "top_k": 5,
            "deep_search": True,
        }
    ]


def test_vault_gateway_never_expands_an_empty_scope():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return []

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/search-vault",
        headers={
            "X-Lavix-Capability": _token(signer, run_id, files=()),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "anything"},
    )
    assert response.json() == {"ok": True, "evidence": [], "count": 0}
    assert retriever.calls == []


def test_gateway_rejects_wrong_run_and_disallowed_web_search():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return [{"title": "Result", "url": "https://example.test", "content": "text"}]

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    client = TestClient(app)
    wrong_run = client.post(
        "/search-vault",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": str(uuid4()),
        },
        json={"query": "anything"},
    )
    denied_web = client.post(
        "/search-web",
        headers={
            "X-Lavix-Capability": _token(signer, run_id, web=False),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "today"},
    )
    assert wrong_run.status_code == 403
    assert denied_web.status_code == 403


def test_web_gateway_records_bounded_untrusted_evidence():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        assert (query, limit) == ("release notes", 2)
        return [
            {"title": "One", "url": "https://one.test", "content": "A", "score": 0.5},
            {"title": "Two", "url": "https://two.test", "content": "B", "score": 0.4},
        ]

    app, signer, store = _app(retriever, web_search)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/search-web",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "release notes", "max_results": 2},
    )
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["evidence"]] == ["W1", "W2"]
    assert [item["match_percentage"] for item in response.json()["evidence"]] == [100, 80]
    assert len(__import__("asyncio").run(store.get(run_id))) == 2


def test_graph_gateway_derives_tenant_from_capability_and_never_records_evidence():
    retriever = FakeRetriever()
    calls = []

    async def web_search(query, limit, **kwargs):
        return []

    async def graph_recall(user_id, query, limit):
        calls.append((user_id, query, limit))
        return [
            {
                "id": "private-internal-id",
                "kind": "preference",
                "subject": "I",
                "predicate": "prefer",
                "object_value": "concise answers",
                "confidence": 0.94,
                "source_message_id": "must-not-escape",
            }
        ]

    app, signer, store = _app(retriever, web_search, graph_recall)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/recall-graph",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": " How should you answer me? ", "max_results": 8},
    )

    assert response.status_code == 200
    assert calls == [(44, "How should you answer me?", 8)]
    assert response.json() == {
        "ok": True,
        "memories": [
            {
                "kind": "preference",
                "subject": "I",
                "predicate": "prefer",
                "object_value": "concise answers",
                "confidence": 0.94,
            }
        ],
        "count": 1,
        "untrusted": True,
    }
    assert __import__("asyncio").run(store.get(run_id)) == []


def test_graph_gateway_rejects_user_id_and_cypher_fields():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return []

    async def graph_recall(user_id, query, limit):
        raise AssertionError("validation should reject before recall")

    app, signer, _ = _app(retriever, web_search, graph_recall)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/recall-graph",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "anything", "user_id": 99, "cypher": "MATCH (n) RETURN n"},
    )

    assert response.status_code == 422


def test_graph_gateway_fails_open_to_empty_optional_memory():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return []

    async def graph_recall(user_id, query, limit):
        raise RuntimeError("private projection unavailable")

    app, signer, store = _app(retriever, web_search, graph_recall)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/recall-graph",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "What do you remember about me?", "max_results": 6},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "memories": [],
        "count": 0,
        "untrusted": True,
    }
    assert __import__("asyncio").run(store.get(run_id)) == []


def test_web_gateway_logs_per_item_relevance_at_stage4(caplog):
    import logging

    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return [
            {"title": "Cricket schedule", "url": "https://a.test", "content": "cricket schedule tour dates", "score": 0.9},
            {"title": "Cake", "url": "https://b.test", "content": "flour sugar eggs", "score": 0.1},
        ]

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    with caplog.at_level(logging.WARNING, logger="app.agent.gateway"):
        response = TestClient(app).post(
            "/search-web",
            headers={
                "X-Lavix-Capability": _token(signer, run_id),
                "X-Lavix-Run-ID": run_id,
            },
            json={"query": "cricket schedule", "max_results": 2},
        )
    assert response.status_code == 200
    stage4 = [r.message for r in caplog.records if "web_rag stage=4" in r.message]
    assert any("matches=2" in m and "relevant=True" in m for m in stage4), stage4
    assert any("matches=0" in m and "relevant=False" in m for m in stage4), stage4


def test_web_gateway_logs_domain_drops_with_rule_and_run(caplog):
    import logging

    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return [
            {"title": "Good", "url": "https://example.test/a", "content": "cricket schedule", "score": 0.9},
            {"title": "Share", "url": "https://twitter.com/intent/tweet?x=1", "content": "cricket schedule", "score": 0.8},
        ]

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    with caplog.at_level(logging.WARNING, logger="app.agent.gateway"):
        response = TestClient(app).post(
            "/search-web",
            headers={
                "X-Lavix-Capability": _token(signer, run_id),
                "X-Lavix-Run-ID": run_id,
            },
            json={"query": "cricket schedule", "max_results": 2},
        )
    assert response.status_code == 200
    stage3 = [r.message for r in caplog.records if "web_rag stage=3" in r.message]
    assert len(stage3) == 1, stage3
    assert "drop=domain" in stage3[0] and "twitter.com" in stage3[0], stage3
    assert "rule=" in stage3[0] and run_id in stage3[0], stage3


class FakeScoringReranker:
    """Reorders + scores like Infinity; records calls for assertions."""

    def __init__(self, order=(1, 0), scores=(0.9, 0.1)):
        self.calls = []
        self.order = order
        self.scores = scores

    async def rerank(self, query, rows, *, top_k):
        self.calls.append((query, len(rows), top_k))
        ranked = []
        for position, source in enumerate(self.order[: len(rows)]):
            item = dict(rows[source])
            item["rerank_score"] = self.scores[position]
            ranked.append(item)
        return ranked


class FakeFailingReranker:
    async def rerank(self, query, rows, *, top_k):
        raise ConnectionError("reranker down")


def _web_app_with_reranker(reranker):
    retriever = FakeRetriever()
    signer = CapabilitySigner("g" * 32, clock=lambda: 1_000)
    store = RunEvidenceStore()

    async def web_search(query, limit, **kwargs):
        return [
            {"title": "Alpha", "url": "https://a.test/x", "content": "alpha alpha cricket schedule", "score": 9.0},
            {"title": "Beta", "url": "https://b.test/y", "content": "beta beta cricket schedule", "score": 1.0},
        ]

    app = FastAPI()
    app.include_router(
        create_agent_gateway_router(
            signer=signer,
            retriever=retriever,
            evidence_store=store,
            web_search=web_search,
            reranker=reranker,
        )
    )
    return app, signer, store


def test_web_rerank_reorders_and_replaces_scores(caplog):
    import logging

    reranker = FakeScoringReranker()
    app, signer, _ = _web_app_with_reranker(reranker)
    run_id = str(uuid4())
    with caplog.at_level(logging.WARNING, logger="app.agent.gateway"):
        response = TestClient(app).post(
            "/search-web",
            headers={
                "X-Lavix-Capability": _token(signer, run_id),
                "X-Lavix-Run-ID": run_id,
            },
            json={"query": "cricket schedule", "max_results": 2},
        )
    assert response.status_code == 200
    assert reranker.calls and reranker.calls[0][1] == 2
    body = response.json()
    assert [item["url"] for item in body["evidence"]] == ["https://b.test/y", "https://a.test/x"]
    scores = [item["score"] for item in body["evidence"]]
    assert scores == [0.9, 0.1]
    assert all("search_score" in item for item in body["evidence"])
    rerank_logs = [r.message for r in caplog.records if "rerank=ok" in r.message]
    assert len(rerank_logs) == 1, rerank_logs


def test_web_rerank_fails_open_to_searxng_order(caplog):
    import logging

    app, signer, _ = _web_app_with_reranker(FakeFailingReranker())
    run_id = str(uuid4())
    with caplog.at_level(logging.WARNING, logger="app.agent.gateway"):
        response = TestClient(app).post(
            "/search-web",
            headers={
                "X-Lavix-Capability": _token(signer, run_id),
                "X-Lavix-Run-ID": run_id,
            },
            json={"query": "cricket schedule", "max_results": 2},
        )
    assert response.status_code == 200
    body = response.json()
    assert [item["url"] for item in body["evidence"]] == ["https://a.test/x", "https://b.test/y"]
    rerank_logs = [r.message for r in caplog.records if "rerank=unavailable" in r.message]
    assert len(rerank_logs) == 1, rerank_logs


def test_web_gate_single_entity_query_passes_on_one_match():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return [
            {"title": "iPhone specs", "url": "https://a.test", "content": "iphone battery and camera details", "score": 0.9},
            {"title": "Cake", "url": "https://b.test", "content": "flour sugar eggs", "score": 0.1},
        ]

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/search-web",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "iPhone", "max_results": 2},
    )
    assert response.status_code == 200
    evidence = response.json()["evidence"]
    by_url = {item["url"]: item for item in evidence}
    assert by_url["https://a.test"]["relevant"] is True
    assert by_url["https://b.test"]["relevant"] is False


def test_web_gate_multi_word_query_still_needs_two_matches():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return [
            {"title": "Cricket only", "url": "https://a.test", "content": "cricket cricket cricket", "score": 0.9},
        ]

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/search-web",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "cricket schedule", "max_results": 1},
    )
    assert response.status_code == 200
    assert response.json()["evidence"][0]["relevant"] is False


def test_evidence_store_tiebreak_prefers_relevant_true(caplog):
    import asyncio
    import logging

    store = RunEvidenceStore()
    run_id = str(uuid4())

    async def scenario():
        first = await store.record(run_id, "web", [
            {"title": "Cricinfo", "url": "https://cric.test/table", "content": "generic", "score": 0.1, "relevant": False},
        ])
        second = await store.record(run_id, "web", [
            {"title": "Cricinfo", "url": "https://cric.test/table", "content": "wtc points table", "score": 0.9, "relevant": True},
        ])
        return first, second, await store.get(run_id)

    with caplog.at_level(logging.WARNING, logger="app.agent.evidence"):
        first, second, stored = asyncio.run(scenario())
    assert first[0]["relevant"] is False
    # Later relevant=True upgrades the stored copy (same card id).
    assert second[0]["relevant"] is True
    assert second[0]["id"] == first[0]["id"]
    assert stored[0]["relevant"] is True
    assert any("evidence_tiebreak" in r.message for r in caplog.records)


def test_evidence_store_tiebreak_ignores_unflagged_vault_items():
    import asyncio

    store = RunEvidenceStore()
    run_id = str(uuid4())

    async def scenario():
        first = await store.record(run_id, "vault", [
            {"file_id": 7, "revision": 2, "chunk_id": "c1", "content": "old"},
        ])
        second = await store.record(run_id, "vault", [
            {"file_id": 7, "revision": 2, "chunk_id": "c1", "content": "new"},
        ])
        return first, second

    first, second = asyncio.run(scenario())
    # No relevant flags anywhere: first-wins, exactly as before.
    assert second[0]["content"] == "old"
    assert second[0]["id"] == first[0]["id"]


def test_plural_stem_handles_ies_both_directions():
    from app.agent.gateway import _distinct_substantive_matches, _plural_stem

    assert _plural_stem("countries") == "country"
    assert _plural_stem("dies") == "dies"
    assert _distinct_substantive_matches(["countries"], "the member country voted") == 1
    assert _distinct_substantive_matches(["country"], "member countries meet") == 1
    # Unrelated words still miss.
    assert _distinct_substantive_matches(["countries"], "flour sugar eggs") == 0


def test_web_gateway_selection_is_engine_diverse_not_first_n():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return [
            {"title": "Bing filler one", "url": "https://bing1.test/", "content": "filler", "engine": "bing"},
            {"title": "Bing filler two", "url": "https://bing2.test/", "content": "filler", "engine": "bing"},
            {"title": "Bing filler three", "url": "https://bing3.test/", "content": "filler", "engine": "bing"},
            {"title": "News gem one", "url": "https://news1.test/", "content": "topical", "engine": "bing news"},
            {"title": "News gem two", "url": "https://news2.test/", "content": "topical", "engine": "bing news"},
            {"title": "Mwmbl gem", "url": "https://mwmbl.test/", "content": "topical", "engine": "mwmbl"},
        ]

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/search-web",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "conflict coverage", "max_results": 3},
    )
    assert response.status_code == 200
    assert [item["url"] for item in response.json()["evidence"]] == [
        "https://bing1.test/",
        "https://news1.test/",
        "https://mwmbl.test/",
    ]


def test_web_gateway_selection_honors_limit_and_engine_order():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        assert limit == 5
        return [
            {"title": f"Bing {index}", "url": f"https://bing{index}.test/", "content": "x", "engine": "bing"}
            for index in range(4)
        ] + [
            {"title": "News one", "url": "https://news1.test/", "content": "x", "engine": "bing news"},
            {"title": "News two", "url": "https://news2.test/", "content": "x", "engine": "bing news"},
        ]

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/search-web",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "conflict coverage", "max_results": 5},
    )
    assert response.status_code == 200
    assert [item["url"] for item in response.json()["evidence"]] == [
        "https://bing0.test/",
        "https://news1.test/",
        "https://bing1.test/",
        "https://news2.test/",
        "https://bing2.test/",
    ]


def test_web_gateway_selection_keeps_unattributed_items():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return [
            {"title": "No engine", "url": "https://plain.test/", "content": "x"},
            {"title": "Bing one", "url": "https://bing1.test/", "content": "x", "engine": "bing"},
        ]

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/search-web",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "conflict coverage", "max_results": 2},
    )
    assert response.status_code == 200
    assert [item["url"] for item in response.json()["evidence"]] == [
        "https://plain.test/",
        "https://bing1.test/",
    ]


def test_web_search_query_scrubbed_before_vendor_fanout():
    retriever = FakeRetriever()
    seen = {}

    async def web_search(query, limit, **kwargs):
        seen["query"] = query
        return [
            {"title": "One", "url": "https://one.test/", "content": "A", "score": 0.5},
        ]

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    response = TestClient(app).post(
        "/search-web",
        headers={
            "X-Lavix-Capability": _token(signer, run_id),
            "X-Lavix-Run-ID": run_id,
        },
        json={"query": "my password is hunter2 Paris", "max_results": 2},
    )
    assert response.status_code == 200
    assert seen["query"] == "my password is [REDACTED] Paris"
    assert "hunter2" not in seen["query"]


def test_vault_top_k_defaults_to_configured_twenty_and_caps_at_500():
    retriever = FakeRetriever()

    async def web_search(query, limit, **kwargs):
        return []

    app, signer, _ = _app(retriever, web_search)
    run_id = str(uuid4())
    headers = {
        "X-Lavix-Capability": _token(signer, run_id),
        "X-Lavix-Run-ID": run_id,
    }
    client = TestClient(app)

    defaulted = client.post("/search-vault", headers=headers, json={"query": "q"})
    assert defaulted.status_code == 200
    assert retriever.calls[-1]["top_k"] == 20

    capped = client.post(
        "/search-vault", headers=headers, json={"query": "q", "top_k": 500}
    )
    assert capped.status_code == 200
    assert retriever.calls[-1]["top_k"] == 500

    rejected = client.post(
        "/search-vault", headers=headers, json={"query": "q", "top_k": 501}
    )
    assert rejected.status_code == 422

    zeroed = client.post(
        "/search-vault", headers=headers, json={"query": "q", "top_k": 0}
    )
    assert zeroed.status_code == 422
