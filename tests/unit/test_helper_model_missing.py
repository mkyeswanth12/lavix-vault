"""Helper-model-missing: rewrite/plan/expansion degrade gracefully, never 500.

Simulates Ollama answering 404 (model not installed) for the agent's pinned
helper default. Each helper must skip its step, log a warning naming the
model, and let the chat continue on the fallback path.
"""

from __future__ import annotations

import asyncio
import logging

import agent_runtime.cuga_adapter as adapter_module


class _MissingModelResponse:
    status_code = 404

    def raise_for_status(self):
        import httpx

        raise httpx.HTTPStatusError(
            "404 model not found",
            request=None,
            response=self,
        )

    def json(self):
        return {"error": "model 'llama3.2:3b' not found"}


class _MissingModelClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, *args, **kwargs):
        return _MissingModelResponse()


def _patch_httpx(monkeypatch):
    import httpx as httpx_mod

    monkeypatch.setattr(httpx_mod, "AsyncClient", _MissingModelClient)


def test_planner_missing_model_falls_back_with_named_warning(monkeypatch, caplog) -> None:
    _patch_httpx(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="agent_runtime.cuga_adapter"):
        plan, reason = asyncio.run(
            adapter_module._plan_search_queries(
                "Tell me about G7 geopolitics and trade",
                ollama_base_url="http://ollama:11434",
                model="llama3.2:3b",
                topic="geopolitics",
            )
        )
    assert plan is None
    assert reason.startswith("transport:")
    assert any(
        "llama3.2:3b" in record.message for record in caplog.records
    ), "fallback warning must name the missing model"


def _install_fake_query_transform():
    """Stub only the (uninstalled) CUGA query_transform import; the real
    _expand_search_queries filter and fallback path still execute."""
    import sys
    import types

    fake_qt = types.ModuleType("query_transform")

    async def fake_expand_query(kind, text, generator, n=3, timeout_s=8.0):
        await generator.generate(text)
        return None

    fake_qt.expand_query = fake_expand_query
    fake_cuga = types.ModuleType("cuga")
    fake_backend = types.ModuleType("backend")
    fake_knowledge = types.ModuleType("knowledge")
    fake_backend.knowledge = fake_knowledge
    fake_knowledge.query_transform = fake_qt
    fake_cuga.backend = fake_backend
    saved = {
        name: sys.modules.get(name)
        for name in ("cuga", "cuga.backend", "cuga.backend.knowledge",
                     "cuga.backend.knowledge.query_transform")
    }
    sys.modules["cuga"] = fake_cuga
    sys.modules["cuga.backend"] = fake_backend
    sys.modules["cuga.backend.knowledge"] = fake_knowledge
    sys.modules["cuga.backend.knowledge.query_transform"] = fake_qt
    return saved


def _restore_query_transform(saved):
    import sys

    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def test_expansion_missing_model_returns_empty_with_named_warning(monkeypatch, caplog) -> None:
    _patch_httpx(monkeypatch)
    saved = _install_fake_query_transform()
    try:
        with caplog.at_level(logging.WARNING, logger="agent_runtime.cuga_adapter"):
            variants = asyncio.run(
                adapter_module._expand_search_queries(
                    "cricket schedule rehearsal details",
                    ollama_base_url="http://ollama:11434",
                    model="llama3.2:3b",
                )
            )
    finally:
        _restore_query_transform(saved)
    assert variants == []
    assert any(
        "llama3.2:3b" in record.message for record in caplog.records
    ), "fallback warning must name the missing model"


def test_rewrite_missing_model_falls_back_without_raising(monkeypatch, caplog) -> None:
    _patch_httpx(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="agent_runtime.cuga_adapter"):
        rewritten, info = asyncio.run(
            adapter_module._rewrite_web_query(
                "who scored most runs",
                ollama_base_url="http://ollama:11434",
                model="llama3.2:3b",
                history_messages=[],
            )
        )
    assert rewritten is None
    assert info.get("reason", "").startswith("transport:")
    assert any(
        "llama3.2:3b" in record.message for record in caplog.records
    ), "fallback warning must name the missing model"
