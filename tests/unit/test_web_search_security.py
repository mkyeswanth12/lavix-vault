from __future__ import annotations

import socket

import pytest

from app.services.web_search import UnsafeWebTarget, validate_public_url


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/admin",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/",
        "http://user:password@example.com/",
        "http://example.com:8080/",
    ],
)
async def test_validate_public_url_rejects_unsafe_targets(url: str) -> None:
    with pytest.raises((UnsafeWebTarget, OSError, socket.gaierror)):
        await validate_public_url(url)


async def test_validate_public_url_accepts_public_https() -> None:
    assert await validate_public_url("https://example.com/path") == "https://example.com/path"


class TestStage2EngineAccounting:
    def test_requested_engines_exist_in_settings(self):
        import yaml

        with open("searxng/settings.yml", encoding="utf-8") as handle:
            settings = yaml.safe_load(handle)
        enabled = {
            engine["name"]
            for engine in settings.get("engines", [])
            if not engine.get("disabled", False)
        }
        assert {"bing", "bing news", "mwmbl"} <= enabled
        assert "wikipedia" not in enabled

    def test_fetch_logs_per_engine_outcomes_and_keeps_engine(self, monkeypatch, caplog):
        import asyncio
        import logging

        import app.services.web_search as module

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "results": [
                        {"title": "A", "url": "https://a.test/x", "content": "a", "score": 5.0, "engine": "mwmbl"},
                        {"title": "B", "url": "https://b.test/y", "content": "b", "score": 1.0, "engine": "bing"},
                    ],
                    "unresponsive_engines": [["timeout", "startpage"]],
                }

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, params=None):
                assert params.get("engines", "") == "bing,bing news,mwmbl"
                return FakeResponse()

        async def no_validate(url):
            return url

        monkeypatch.setattr(module.httpx, "AsyncClient", FakeClient)
        monkeypatch.setattr(module, "validate_public_url", no_validate)
        with caplog.at_level(logging.WARNING, logger="app.services.web_search"):
            results = asyncio.run(module._fetch_searxng("cricket schedule", 5, run_id="run-e2e"))
        assert [r["engine"] for r in results] == ["bing", "mwmbl"]
        lines = [r.message for r in caplog.records if "web_rag stage=2" in r.message]
        assert any("bing:1" in m and "mwmbl:1" in m and "startpage" in m for m in lines), lines


def test_fetch_cut_is_engine_diverse_not_first_n(monkeypatch, caplog):
    import asyncio
    import logging

    import app.services.web_search as module

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"title": "B1", "url": "https://b1.test/", "content": "x", "score": 9.0, "engine": "bing"},
                    {"title": "B2", "url": "https://b2.test/", "content": "x", "score": 8.0, "engine": "bing"},
                    {"title": "B3", "url": "https://b3.test/", "content": "x", "score": 7.0, "engine": "bing"},
                    {"title": "N1", "url": "https://n1.test/", "content": "x", "score": 6.0, "engine": "bing news"},
                    {"title": "N2", "url": "https://n2.test/", "content": "x", "score": 5.0, "engine": "bing news"},
                    {"title": "M1", "url": "https://m1.test/", "content": "x", "score": 4.0, "engine": "mwmbl"},
                ],
                "unresponsive_engines": [],
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None):
            return FakeResponse()

    async def no_validate(url):
        return url

    monkeypatch.setattr(module.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(module, "validate_public_url", no_validate)
    with caplog.at_level(logging.WARNING, logger="app.services.web_search"):
        results = asyncio.run(module._fetch_searxng("conflict coverage", 3, run_id="run-diverse"))
    assert [r["url"] for r in results] == [
        "https://b1.test/",
        "https://n1.test/",
        "https://m1.test/",
    ]


def test_diverse_engine_pick_edge_cases():
    from app.services.web_search import _diverse_engine_pick

    assert _diverse_engine_pick([], 3) == []
    assert _diverse_engine_pick([{"url": "https://a.test/"}], 0) == []
    assert _diverse_engine_pick([{"url": "https://a.test/"}], None) == []
    items = [
        {"url": "https://b1.test/", "engine": "bing"},
        {"url": "https://plain.test/"},
        {"url": "https://b2.test/", "engine": "bing"},
    ]
    assert [i["url"] for i in _diverse_engine_pick(items, 2)] == [
        "https://b1.test/",
        "https://plain.test/",
    ]
    assert len(_diverse_engine_pick(items, 99)) == 3
