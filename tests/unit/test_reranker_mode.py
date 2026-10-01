"""Reranker mode wiring: parsing, aliases, and saved-admin-wins proofs."""

from __future__ import annotations

import pytest

from app.agent.retrieval import VaultRetriever
from app.config import Settings, reload_config
from app.services.model_config import _resolve_row


def _saved_row(**overrides):
    row = {
        "revision": 3,
        "chat_enabled": True,
        "chat_default_model": "saved-chat:latest",
        "chat_allowed_models": ["saved-chat:latest", "env-chat:latest"],
        "vision_enabled": True,
        "vision_model": "saved-vision:latest",
        "intelligence_enabled": False,
        "intelligence_model": None,
        "memory_extraction_enabled": False,
        "memory_extraction_model": None,
        "reranker_enabled": False,
    }
    row.update(overrides)
    return row


def test_rerank_mode_defaults_to_local(monkeypatch) -> None:
    monkeypatch.delenv("RERANKER_MODE", raising=False)
    reload_config()
    try:
        assert Settings().rerank_mode == "local"
    finally:
        reload_config()


def test_rerank_mode_rejects_unknown_values(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_MODE", "sometimes")
    reload_config()
    try:
        with pytest.raises(ValueError, match="RERANKER_MODE"):
            _ = Settings().rerank_mode
    finally:
        reload_config()


def test_rerank_mode_external_requires_a_valid_url(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_MODE", "external")
    reload_config()
    try:
        monkeypatch.delenv("RERANKER_URL", raising=False)
        monkeypatch.delenv("RERANK_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="RERANKER_URL"):
            _ = Settings().rerank_mode
        monkeypatch.setenv("RERANKER_URL", "not-a-url")
        with pytest.raises(ValueError, match="RERANKER_URL"):
            _ = Settings().rerank_mode
        monkeypatch.setenv("RERANKER_URL", "http://reranker.local:8000")
        assert Settings().rerank_mode == "external"
    finally:
        reload_config()


def test_rerank_model_and_url_prefer_canonical_names(monkeypatch) -> None:
    monkeypatch.setenv("RERANK_MODEL", "legacy-model:latest")
    monkeypatch.setenv("RERANKER_MODEL", "canonical-model:latest")
    monkeypatch.setenv("RERANK_BASE_URL", "http://legacy:7997")
    monkeypatch.setenv("RERANKER_URL", "http://canonical:7997")
    reload_config()
    try:
        assert Settings().rerank_model == "canonical-model:latest"
        assert Settings().rerank_base_url == "http://canonical:7997"
    finally:
        reload_config()


def test_legacy_rerank_names_still_work_as_aliases(monkeypatch) -> None:
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    monkeypatch.delenv("RERANKER_URL", raising=False)
    monkeypatch.setenv("RERANK_MODEL", "legacy-model:latest")
    monkeypatch.setenv("RERANK_BASE_URL", "http://legacy:7997")
    reload_config()
    try:
        assert Settings().rerank_model == "legacy-model:latest"
        assert Settings().rerank_base_url == "http://legacy:7997"
    finally:
        reload_config()


def test_saved_chat_and_vision_models_win_over_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "env-chat:latest")
    monkeypatch.setenv("VISION_MODEL", "env-vision:latest")
    reload_config()
    try:
        resolved = _resolve_row(_saved_row())
        assert resolved.source == "database"
        assert resolved.chat.default_model == "saved-chat:latest"
        assert resolved.vision.model == "saved-vision:latest"
    finally:
        reload_config()


def test_saved_reranker_toggle_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_MODE", "local")
    monkeypatch.setenv("ENABLE_RERANKING", "true")
    reload_config()
    try:
        assert _resolve_row(_saved_row()).reranker_enabled is False
    finally:
        reload_config()


class _FailingConnections:
    def __call__(self):
        raise RuntimeError("no database")


class _NullConnection:
    def __enter__(self):
        return object()

    def __exit__(self, *args):
        return False


def test_mode_off_disables_reranking_without_a_database(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_MODE", "off")
    reload_config()
    try:
        retriever = VaultRetriever(connection_factory=_FailingConnections())
        assert retriever._configured_reranking_enabled() is False
    finally:
        reload_config()


def test_mode_off_wins_over_an_enabled_admin_toggle(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RERANKER_MODE", "off")
    reload_config()
    try:
        import app.agent.retrieval as retrieval_module

        monkeypatch.setattr(
            retrieval_module.ModelConfigurationRepository,
            "get",
            lambda self: _resolve_row(_saved_row(reranker_enabled=True)),
        )
        retriever = VaultRetriever(connection_factory=lambda: _NullConnection())
        assert retriever._configured_reranking_enabled() is False
    finally:
        reload_config()


def test_saved_toggle_disables_reranking_while_env_enables(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_MODE", "local")
    monkeypatch.setenv("ENABLE_RERANKING", "true")
    reload_config()
    try:
        import app.agent.retrieval as retrieval_module

        monkeypatch.setattr(
            retrieval_module.ModelConfigurationRepository,
            "get",
            lambda self: _resolve_row(_saved_row(reranker_enabled=False)),
        )
        retriever = VaultRetriever(connection_factory=lambda: _NullConnection())
        assert retriever._configured_reranking_enabled() is False
    finally:
        reload_config()


def test_unreadable_configuration_falls_back_to_env_default(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_MODE", "local")
    monkeypatch.setenv("ENABLE_RERANKING", "true")
    reload_config()
    try:
        retriever = VaultRetriever(connection_factory=_FailingConnections())
        assert retriever._configured_reranking_enabled() is True
    finally:
        reload_config()
