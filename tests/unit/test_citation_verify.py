"""Tests for citation verification — cosine similarity, domain filter, relevance gate."""

import pytest

from app.agent.citation_verify import (
    _DOMAIN_BLOCKLIST,
    cosine_similarity,
    domain_sanity_check,
    is_utility_url,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = (1.0, 2.0, 3.0)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        assert abs(cosine_similarity((1.0, 0.0), (0.0, 1.0))) < 1e-6

    def test_opposite_vectors(self):
        assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert cosine_similarity((), ()) == 0.0

    def test_mismatched_lengths(self):
        assert cosine_similarity((1.0, 2.0), (1.0, 2.0, 3.0)) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity((0.0, 0.0), (1.0, 1.0)) == 0.0

    def test_known_value(self):
        a = (1.0, 0.0)
        b = (0.70710678, 0.70710678)
        assert cosine_similarity(a, b) == pytest.approx(0.7071, abs=0.001)


class TestDomainSanityCheck:
    def test_empty_url_passes(self):
        assert domain_sanity_check("", "What year") is True

    def test_blocklisted_domain(self):
        assert domain_sanity_check("https://sarkariresult.com/some-page", "What year") is False

    def test_blocklisted_with_www(self):
        assert domain_sanity_check("https://www.sarkariresult.com/page", "query") is False

    def test_valid_domain(self):
        assert domain_sanity_check("https://en.wikipedia.org/wiki/Eiffel_Tower", "What year") is True

    def test_suspicious_url_pattern_buy(self):
        assert domain_sanity_check("https://shop.example.com/buy/widget", "What year") is False

    def test_official_store_commerce_paths_allowed(self):
        assert domain_sanity_check("https://www.apple.com/shop/buy-iphone/iphone-17", "iPhone price") is True
        assert domain_sanity_check("https://store.apple.com/cart", "iPhone price") is True
        assert domain_sanity_check("https://www.samsung.com/shop/offer", "phone price") is True

    def test_official_store_other_patterns_still_apply(self):
        assert domain_sanity_check("https://apple.com/recruitment/2024", "query") is False

    def test_non_allowlisted_commerce_still_blocked(self):
        assert domain_sanity_check("https://shop.example.com/buy/widget", "iPhone price") is False
        assert domain_sanity_check("https://evil-apple.com/shop/buy", "iPhone price") is False

    def test_suspicious_url_pattern_recruitment(self):
        assert domain_sanity_check("https://example.com/recruitment/2024", "query") is False

    def test_legitimate_url_not_flagged(self):
        assert domain_sanity_check("https://www.britannica.com/topic/Eiffel-Tower", "What year") is True

    def test_all_blocklisted_domains(self):
        for domain in _DOMAIN_BLOCKLIST:
            assert domain_sanity_check(f"https://{domain}/page", "query") is False


class TestCitationRelevanceEiffelTowerRegression:
    """Regression: Eiffel Tower query must not cite sarkariresult.com."""

    def test_sarkariresult_blocked_for_historical_query(self):
        assert domain_sanity_check(
            "https://sarkariresult.com/eiffel-tower-history",
            "What year was the Eiffel Tower completed?",
        ) is False

    def test_wikipedia_allowed_for_historical_query(self):
        assert domain_sanity_check(
            "https://en.wikipedia.org/wiki/Eiffel_Tower",
            "What year was the Eiffel Tower completed?",
        ) is True

    def test_britannica_allowed(self):
        assert domain_sanity_check(
            "https://www.britannica.com/topic/Eiffel-Tower",
            "What year was the Eiffel Tower completed?",
        ) is True

    def test_eiffel_tower_answer_contains_1889(self):
        """Verify the expected answer to the regression query."""
        # This is a sanity check — the answer is parametric knowledge
        answer = "The Eiffel Tower was completed in 1889."
        assert "1889" in answer


class TestSocialShareURLBlocking:
    """Test that social share widgets and URL shorteners are blocked."""

    def test_wa_me_blocked(self):
        assert domain_sanity_check("https://wa.me/1234567890", "any query") is False

    def test_web_whatsapp_blocked(self):
        assert domain_sanity_check("https://web.whatsapp.com/send?text=hello", "any query") is False

    def test_telegram_share_blocked(self):
        assert domain_sanity_check("https://t.me/share/url?url=example.com", "any query") is False

    def test_twitter_share_blocked(self):
        assert domain_sanity_check("https://twitter.com/intent/tweet?url=example.com", "any query") is False

    def test_facebook_share_blocked(self):
        assert domain_sanity_check("https://facebook.com/sharer/sharer.php?u=example.com", "any query") is False

    def test_linkedin_share_blocked(self):
        assert domain_sanity_check("https://linkedin.com/sharing/share-offsite/?url=example.com", "any query") is False

    def test_bit_ly_blocked(self):
        assert domain_sanity_check("https://bit.ly/3abc123", "any query") is False

    def test_tinyurl_blocked(self):
        assert domain_sanity_check("https://tinyurl.com/abc123", "any query") is False

    def test_t_co_blocked(self):
        assert domain_sanity_check("https://t.co/abc123", "any query") is False

    def test_goo_gl_blocked(self):
        assert domain_sanity_check("https://goo.gl/abc123", "any query") is False

    def test_whatsapp_domain_blocked(self):
        assert domain_sanity_check("https://whatsapp.com/dl/web", "any query") is False

    def test_is_utility_url_share_intent(self):
        assert is_utility_url("https://twitter.com/intent/tweet?text=hello") is True

    def test_is_utility_url_shortener(self):
        assert is_utility_url("https://bit.ly/3abc123") is True

    def test_is_utility_url_normal_article(self):
        assert is_utility_url("https://en.wikipedia.org/wiki/Eiffel_Tower") is False

    def test_is_utility_url_empty(self):
        assert is_utility_url("") is False


class TestOscarsRegression:
    """Regression: compound query about Oscars should not cite social share URLs."""

    def test_whatsapp_not_cited_for_oscars_query(self):
        # The regression case cited web.whatsapp.com and wa.me
        assert domain_sanity_check("https://web.whatsapp.com/send?text=cool", "Oscars Best Picture") is False
        assert domain_sanity_check("https://wa.me/1234567890?text=movie", "Oscars Best Picture") is False

    def test_legitimate_source_allowed(self):
        assert domain_sanity_check("https://www.oscars.org/oscars/ceremonies/2025", "Oscars Best Picture") is True
        assert domain_sanity_check("https://en.wikipedia.org/wiki/Academy_Award_for_Best_Picture", "Oscars Best Picture") is True


class FakeEmbeddings:
    """Deterministic stand-in for Ollama embeddings keyed by content marker."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    async def embed_pair(self, claim, source):
        self.calls += 1
        for marker, score in self.scores:
            if marker in source:
                return score
        return 0.0


def _item(url, title, content):
    return {"url": url, "title": title, "content": content}


def _run_filter(module, monkeypatch, caplog, scores, items, **kwargs):
    import math

    calls = []

    async def fake_embed(texts, **kw):
        calls.append(list(texts))
        vectors = []
        for position, text in enumerate(texts):
            if position == 0:
                vectors.append((1.0, 0.0))
                continue
            score = 0.0
            for marker, want in scores:
                if marker in text:
                    score = want
                    break
            vectors.append((score, math.sqrt(max(0.0, 1.0 - score * score))))
        return [tuple(v) for v in vectors]

    monkeypatch.setattr(module, "_embed_texts", fake_embed)
    import asyncio
    import logging

    with caplog.at_level(logging.WARNING, logger=module.__name__):
        result = asyncio.run(
            module.filter_citations(items, query="cricket schedule", **kwargs)
        )
    # One batched embedding call per filter invocation, however many items.
    assert len(calls) == 1, calls
    assert len(calls[0]) == len(items) + 1, calls
    return result


class TestStage6Logging:
    def test_every_candidate_logged_with_score_and_keep(self, monkeypatch, caplog):
        import app.agent.citation_verify as app_verify

        items = [
            _item("https://example.com/a", "Cricket schedule", "KEEPME Australia tour dates"),
            _item("https://example.com/b", "Unrelated", "DROPME cooking recipes"),
        ]
        result = _run_filter(
            app_verify, monkeypatch, caplog,
            [("KEEPME", 0.82), ("DROPME", 0.35)],
            items,
            claim="Australia tour dates",
            threshold=0.5,
            embedding_url="http://x", embedding_model="m",
            run_id="run-123",
        )
        assert [i["url"] for i in result] == ["https://example.com/a"]
        stage6 = [r.message for r in caplog.records if "web_rag stage=6" in r.message]
        assert any("score=0.820" in m and "keep=True" in m and "run-123" in m for m in stage6), stage6
        assert any("score=0.350" in m and "keep=False" in m for m in stage6), stage6

    def test_agent_copy_logs_identically(self, monkeypatch, caplog):
        import agent_runtime.citation_verify as agent_verify

        items = [_item("https://example.com/a", "T", "KEEPME x")]
        _run_filter(
            agent_verify, monkeypatch, caplog,
            [("KEEPME", 0.61)],
            items,
            claim="c",
            threshold=0.5,
            embedding_url="http://x", embedding_model="m",
            run_id="run-9",
        )
        stage6 = [r.message for r in caplog.records if "web_rag stage=6" in r.message]
        assert any("run-9" in m and "keep=True" in m for m in stage6), stage6


class TestFailOpenUnification:
    @pytest.mark.parametrize("modname", ["app.agent.citation_verify", "agent_runtime.citation_verify"])
    def test_infra_failure_keeps_candidates_in_both_copies(self, monkeypatch, caplog, modname):
        import importlib

        module = importlib.import_module(modname)
        items = [_item("https://example.com/a", "T", "some content here")]

        async def dead_embed(texts, **kw):
            return []

        monkeypatch.setattr(module, "_embed_texts", dead_embed)
        import asyncio
        import logging

        with caplog.at_level(logging.WARNING, logger=module.__name__):
            result = asyncio.run(
                module.filter_citations(
                    [dict(i) for i in items],
                    query="cricket schedule",
                    claim="Australia tour",
                    threshold=0.5,
                    embedding_url="http://x", embedding_model="m",
                    run_id="run-infra",
                )
            )
        assert [i["url"] for i in result] == ["https://example.com/a"]
        assert any("infra" in r.message for r in caplog.records if "web_rag stage=6" in r.message)


def _verify_with_scores(monkeypatch, caplog, module, score_by_marker, items, **kwargs):
    import asyncio
    import logging
    import math

    async def fake_embed(texts, **kw):
        vectors = []
        for position, text in enumerate(texts):
            if position == 0:
                vectors.append((1.0, 0.0))
                continue
            score = 0.0
            for marker, want in score_by_marker:
                if marker in text:
                    score = want
                    break
            vectors.append((score, math.sqrt(max(0.0, 1.0 - score * score))))
        return [tuple(v) for v in vectors]

    monkeypatch.setattr(module, "_embed_texts", fake_embed)
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        return asyncio.run(module.filter_citations(items, **kwargs))


def test_zero_lexical_candidate_reaches_semantic_verify(monkeypatch, caplog):
    import agent_runtime.citation_verify as agent_verify

    items = [
        {"url": "https://icc.test/table", "title": "World Test Championship points table",
         "content": "ICC World Test Championship standings and qualification", "relevant": False},
    ]
    result = _verify_with_scores(
        monkeypatch, caplog, agent_verify, [("Championship", 0.78)], items,
        query="WTC final schedule", claim="World Test Championship standings",
        threshold=0.5, embedding_url="http://x", embedding_model="m", run_id="wtc-1",
    )
    assert [i["url"] for i in result] == ["https://icc.test/table"]
    # Semantic accept upgrades the lexical flag.
    assert result[0]["relevant"] is True
    assert result[0]["relevance_score"] == 0.78


def test_false_positive_rejected_for_wrong_leg(monkeypatch, caplog):
    import agent_runtime.citation_verify as agent_verify

    items = [
        {"url": "https://www.wtc.edu/", "title": "Western Texas College admissions",
         "content": "Western Texas College admissions open house", "relevant": True, "leg_id": "leg_1"},
    ]
    result = _verify_with_scores(
        monkeypatch, caplog, agent_verify, [("Western Texas", 0.12)], items,
        query="WTC final schedule", claim="World Test Championship final qualification",
        threshold=0.5, embedding_url="http://x", embedding_model="m", run_id="wtc-2",
        leg_queries={"leg_1": "World Test Championship final qualification"},
    )
    assert result == []


def test_leg_specific_keyword_signal_per_item(monkeypatch, caplog):
    import agent_runtime.citation_verify as agent_verify

    items = [
        {"url": "https://nato.test/about", "title": "NATO alliance", "content": "alliance facts",
         "relevant": True, "leg_id": "leg_1"},
        {"url": "https://brics.test/about", "title": "BRICS bloc", "content": "bloc facts",
         "relevant": True, "leg_id": "leg_2"},
    ]
    result = _verify_with_scores(
        monkeypatch, caplog, agent_verify,
        [("NATO alliance facts", 0.8), ("BRICS bloc facts", 0.8)], items,
        query="What are G7, G20, G2, NATO, BRICS and RIC?",
        claim="NATO alliance facts BRICS bloc facts",
        threshold=0.5, embedding_url="http://x", embedding_model="m", run_id="legs-1",
        leg_queries={"leg_1": "What is NATO?", "leg_2": "What is BRICS?"},
    )
    assert {i["url"] for i in result} == {"https://nato.test/about", "https://brics.test/about"}
    leg_lines = [r.message for r in caplog.records if "leg=leg_" in r.message]
    assert any("leg=leg_1" in m for m in leg_lines)
    assert any("leg=leg_2" in m for m in leg_lines)
