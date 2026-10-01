"""Embedding presets and dimension rules."""

from __future__ import annotations

import pytest

from app.config import Settings, reload_config
from app.db.embedding_store import column_type_for_dimensions, index_ops_for_type
from app.ingestion.embedding_presets import prepare_texts, preset_for


def test_known_families_get_their_prefixes() -> None:
    nomic = preset_for("nomic-embed-text")
    assert nomic.query_prefix == "search_query: "
    assert nomic.document_prefix == "search_document: "

    e5 = preset_for("intfloat/e5-large-v2")
    assert e5.query_prefix == "query: "
    assert e5.document_prefix == "passage: "

    plain = preset_for("snowflake-arctic-embed2:cpu")
    assert plain.query_prefix == ""
    assert plain.document_prefix == ""


def test_unknown_models_fall_back_to_no_prefix(caplog) -> None:
    preset = preset_for("some-future-model-9000")
    assert preset.query_prefix == ""
    assert preset.document_prefix == ""
    assert "no embedding preset" in caplog.text


def test_prepare_texts_applies_kind_prefix_and_truncates() -> None:
    assert prepare_texts("nomic-embed-text", ["hello"], kind="query") == [
        "search_query: hello"
    ]
    assert prepare_texts("nomic-embed-text", ["hello"], kind="document") == [
        "search_document: hello"
    ]
    long_text = "x" * 5000
    prepared = prepare_texts("mxbai-embed-large", [long_text], kind="document")[0]
    assert len(prepared) == 1800


def test_column_type_boundaries() -> None:
    assert column_type_for_dimensions(1) == "vector"
    assert column_type_for_dimensions(1024) == "vector"
    assert column_type_for_dimensions(2000) == "vector"
    assert column_type_for_dimensions(2001) == "halfvec"
    assert column_type_for_dimensions(4000) == "halfvec"
    for bad in (0, -8, 4001, 16000):
        with pytest.raises(ValueError, match="between 1 and 4000"):
            column_type_for_dimensions(bad)
    assert index_ops_for_type("vector") == "vector_cosine_ops"
    assert index_ops_for_type("halfvec") == "halfvec_cosine_ops"
    with pytest.raises(ValueError, match="unsupported"):
        index_ops_for_type("bit")


def test_embedding_settings_prefer_canonical_names(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "legacy-embed")
    monkeypatch.setenv("EMBEDDING_MODEL", "canonical-embed")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "111")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "222")
    reload_config()
    try:
        assert Settings().embedding_model_name == "canonical-embed"
        assert Settings().embedding_dimension == 222
    finally:
        reload_config()


def test_legacy_embedding_names_still_work(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "legacy-embed")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
    reload_config()
    try:
        assert Settings().embedding_model_name == "legacy-embed"
        assert Settings().embedding_dimension == 768
    finally:
        reload_config()


def test_embedding_dimensions_refuse_unindexable_sizes(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "5000")
    reload_config()
    try:
        with pytest.raises(ValueError, match="between 1 and 4000"):
            _ = Settings().embedding_dimension
    finally:
        reload_config()


def test_reembed_cli_rejects_bad_dimensions_without_a_database(capsys) -> None:
    from app.db.reembed import main as reembed_main

    assert reembed_main(["--model", "nomic-embed-text", "--dimensions", "5000"]) == 2
    assert "between 1 and 4000" in capsys.readouterr().err
    assert reembed_main(["--model", "nomic-embed-text", "--dimensions", "0"]) == 2
