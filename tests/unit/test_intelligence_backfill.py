from __future__ import annotations

import json
from typing import Any

import pytest

from app.graph_memory.inference_priority import BackgroundInferenceDeferred
from app.ingestion import intelligence_backfill
from app.ingestion.intelligence import DocumentIntelligence, deterministic_intelligence


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.rowcount = 1

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.calls.append((query, params))

    def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _Cursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _DatabaseContext:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def __enter__(self) -> _Connection:
        return self.connection

    def __exit__(self, *_args: Any) -> bool:
        return False


def test_candidates_load_parser_fingerprint_and_ordered_chunk_provenance(monkeypatch) -> None:
    cursor = _Cursor()
    connection = _Connection(cursor)
    monkeypatch.setattr(
        intelligence_backfill,
        "get_db",
        lambda: _DatabaseContext(connection),
    )

    assert intelligence_backfill._candidates(user_id=7, limit=3) == []

    query, params = cursor.calls[0]
    assert "dr.parser_fingerprint" in query
    assert "jsonb_path_query_array" in query
    assert "'$[*].page_number'" in query
    assert "ORDER BY dc.ordinal" in query
    assert "dc.revision = f.current_revision" in query
    assert "dr.status = 'current'" in query
    assert params == (7, 3)


def test_missing_tag_candidates_are_current_indexed_non_raster_revisions(monkeypatch) -> None:
    cursor = _Cursor()
    connection = _Connection(cursor)
    monkeypatch.setattr(
        intelligence_backfill,
        "get_db",
        lambda: _DatabaseContext(connection),
    )

    assert (
        intelligence_backfill._candidates(
            user_id=9,
            limit=12,
            missing_tags_only=True,
        )
        == []
    )

    query, params = cursor.calls[0]
    normalized = " ".join(query.split()).casefold()
    assert "like 'image/%%'" in normalized
    assert "dr.status = 'current'" in normalized
    assert "not ( lower(coalesce(f.mime_type, '')) like 'image/%%'" in normalized
    assert "unnest(coalesce(f.quick_tags, array[]::text[]))" in normalized
    assert "btrim(tag.value) <> ''" in normalized
    assert "exists ( select 1 from document_chunks as indexed_chunk" in normalized
    assert "indexed_chunk.revision = f.current_revision" in normalized
    assert params == (9, 12)


def test_missing_tag_rasters_are_reported_as_targeted_reindex_candidates(monkeypatch) -> None:
    cursor = _Cursor()
    connection = _Connection(cursor)
    monkeypatch.setattr(
        intelligence_backfill,
        "get_db",
        lambda: _DatabaseContext(connection),
    )

    assert intelligence_backfill._raster_reindex_candidates(user_id=4, limit=6) == []

    query, params = cursor.calls[0]
    normalized = " ".join(query.split()).casefold()
    assert "like 'image/%%'" in normalized
    assert "lower(coalesce(f.mime_type, '')) like 'image/%%'" in normalized
    assert "lower(f.original_filename) ~" in normalized
    assert "unnest(coalesce(f.quick_tags, array[]::text[]))" in normalized
    assert "indexed_chunk.revision = f.current_revision" in normalized
    assert params == (4, 6)


def test_chunk_views_keep_only_valid_unique_page_numbers() -> None:
    chunks = intelligence_backfill._chunk_views(
        [
            {
                "text": " first page ",
                "page_numbers": [1, "1", None, 0],
            },
            {"text": "second page", "page_numbers": [2]},
            {"text": "  ", "page_numbers": [3]},
            "invalid",
        ]
    )

    assert [chunk.text for chunk in chunks] == ["first page", "second page"]
    assert [[item.page_number for item in chunk.provenance] for chunk in chunks] == [[1], [2]]


@pytest.mark.asyncio
async def test_scan_aware_dry_run_uses_revision_fingerprint_and_pages(
    monkeypatch,
    capsys,
) -> None:
    row = {
        "file_id": 209,
        "user_id": 2,
        "revision": 7,
        "source_name": "AGC_2025.pdf",
        "source_sha256": "a" * 64,
        "mime_type": "application/pdf",
        "parser_fingerprint": (
            "opendataloader:2.4.7:result=empty+docling:2.112.0:targeted:local+"
            "ocr:tesseract-cli:tesseract_5.3.0:lang=eng,kan:psm=6"
        ),
        "quick_tags": [],
        "chunks": [
            {
                "text": "ಕರ್ನಾಟಕ ಕಂದಾಯ ಭೂಮಿ ದಾಖಲೆ ಕರ್ನಾಟಕ ಕಂದಾಯ ಭೂಮಿ ದಾಖಲೆ revenue stamp",
                "page_numbers": [1],
            },
            {
                "text": "ತಾಲ್ಲೂಕು ಗ್ರಾಮ ಭೂಮಿ ದಾಖಲೆ ತಾಲ್ಲೂಕು ಗ್ರಾಮ ಭೂಮಿ ದಾಖಲೆ government record",
                "page_numbers": [2],
            },
        ],
    }
    captured: dict[str, Any] = {}

    class _Service:
        def __init__(self, _settings: Any) -> None:
            pass

        async def analyze(self, document: Any, chunks: Any) -> DocumentIntelligence:
            captured["document"] = document
            captured["chunks"] = chunks
            return deterministic_intelligence(document, chunks)

    monkeypatch.setattr(intelligence_backfill, "_candidates", lambda _user_id, _limit: [row])
    monkeypatch.setattr(intelligence_backfill, "DocumentIntelligenceService", _Service)
    monkeypatch.setattr(
        intelligence_backfill,
        "_persist",
        lambda *_args: pytest.fail("a dry-run must not persist metadata"),
    )

    result = await intelligence_backfill.run(apply=False, user_id=None, limit=None)

    assert result == 0
    assert captured["document"].parser_fingerprint == row["parser_fingerprint"]
    assert [provenance.page_number for chunk in captured["chunks"] for provenance in chunk.provenance] == [
        1,
        2,
    ]
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["status"] == "fallback"
    assert records[0]["tags"] == []
    assert records[0]["persisted"] is False
    assert records[-1] == {
        "mode": "intelligence-dry-run",
        "candidates": 1,
        "completed": 1,
        "changed": 1,
        "skipped_raster": 0,
        "deferred": 0,
        "by_file_type": {"other": 1},
        "fallback_reasons": {"deterministic": 1},
    }


@pytest.mark.asyncio
async def test_backfill_stops_cleanly_on_priority_deferral_without_writing(
    monkeypatch,
    capsys,
) -> None:
    rows = [
        {
            "file_id": file_id,
            "user_id": 2,
            "revision": 7,
            "source_name": f"report-{file_id}.txt",
            "source_sha256": f"{file_id}" * 64,
            "mime_type": "text/plain",
            "parser_fingerprint": "text:v1",
            "quick_summary": "",
            "quick_tags": [],
            "doc_type": "other",
            "intelligence_status": "pending",
            "chunks": [{"text": "A grounded report about cloud security.", "page_numbers": [1]}],
        }
        for file_id in (1, 2)
    ]
    calls = 0

    class _Service:
        def __init__(self, _settings: Any) -> None:
            pass

        async def analyze(self, _document: Any, _chunks: Any) -> DocumentIntelligence:
            nonlocal calls
            calls += 1
            raise BackgroundInferenceDeferred()

    monkeypatch.setattr(intelligence_backfill, "_candidates", lambda _user_id, _limit: rows)
    monkeypatch.setattr(intelligence_backfill, "DocumentIntelligenceService", _Service)
    monkeypatch.setattr(
        intelligence_backfill,
        "_persist",
        lambda *_args: pytest.fail("deferred backfill must not write metadata"),
    )

    result = await intelligence_backfill.run(apply=True, user_id=2, limit=None)

    assert result == 1
    assert calls == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0] == {
        "file_id": 1,
        "revision": 7,
        "filename": "report-1.txt",
        "status": "deferred",
        "reason": "foreground_inference_unavailable",
        "persisted": False,
    }
    assert records[-1]["candidates"] == 2
    assert records[-1]["completed"] == 0
    assert records[-1]["deferred"] == 1


def test_persist_keeps_both_current_revision_fences(monkeypatch) -> None:
    cursor = _Cursor()
    connection = _Connection(cursor)
    monkeypatch.setattr(
        intelligence_backfill,
        "get_db",
        lambda: _DatabaseContext(connection),
    )
    row = {"file_id": 209, "user_id": 2, "revision": 7}
    intelligence = DocumentIntelligence(
        doc_type="other",
        summary="Evidence-based summary.",
        tags=("kannada script",),
        status="fallback",
    )

    assert intelligence_backfill._persist(row, intelligence) is True

    file_query, file_params = cursor.calls[0]
    revision_query, revision_params = cursor.calls[1]
    assert "current_revision = %s" in file_query
    assert file_params == (
        "Evidence-based summary.",
        ["kannada script"],
        "other",
        "fallback",
        209,
        2,
        7,
    )
    assert file_params[-3:] == (209, 2, 7)
    assert "revision = %s" in revision_query
    assert "status = 'current'" in revision_query
    assert revision_params[-3:] == (209, 2, 7)
    sql = " ".join(query for query, _params in cursor.calls).casefold()
    assert "document_chunks" not in sql
    assert "embedding" not in sql
    assert "insert " not in sql
    assert "delete " not in sql
    assert connection.commits == 1
    assert connection.rollbacks == 0


def test_missing_tag_persist_is_revision_fenced_and_changes_only_tag_metadata(monkeypatch) -> None:
    cursor = _Cursor()
    connection = _Connection(cursor)
    monkeypatch.setattr(
        intelligence_backfill,
        "get_db",
        lambda: _DatabaseContext(connection),
    )
    row = {"file_id": 209, "user_id": 2, "revision": 7}

    assert intelligence_backfill._persist_tags(
        row,
        ("cloud security", "access controls"),
        require_missing=True,
    ) is True

    file_query, file_params = cursor.calls[0]
    revision_query, revision_params = cursor.calls[1]
    normalized_file = " ".join(file_query.split()).casefold()
    normalized_revision = " ".join(revision_query.split()).casefold()
    assert "set quick_tags = %s" in normalized_file
    assert "current_revision = %s" in normalized_file
    assert "unnest(coalesce(quick_tags, array[]::text[]))" in normalized_file
    assert "set current_revision" not in normalized_file
    assert file_params == (["cloud security", "access controls"], 209, 2, 7)
    assert "status = 'current'" in normalized_revision
    assert "jsonb_build_object('tags', %s::jsonb)" in normalized_revision
    assert revision_params[-3:] == (209, 2, 7)
    sql = " ".join(query for query, _params in cursor.calls).casefold()
    assert "quick_summary" not in sql
    assert "doc_type" not in sql
    assert "intelligence_status" not in sql
    assert "document_chunks" not in sql
    assert "embedding" not in sql
    assert "insert " not in sql
    assert "delete " not in sql
    assert connection.commits == 1
    assert connection.rollbacks == 0


@pytest.mark.asyncio
async def test_tag_only_repair_normalizes_existing_values_without_sampling_deletions(
    monkeypatch,
    capsys,
) -> None:
    row = {
        "file_id": 8,
        "user_id": 2,
        "revision": 4,
        "source_name": "policy.pdf",
        "mime_type": "application/pdf",
        "quick_summary": "The cloud security policy defines production access controls.",
        "quick_tags": ["cloud-policy", "cloud_security", "2025", "blue background"],
        "chunks": [
            {
                "text": (
                    "The cloud security policy defines production access controls. "
                    "The report was published in 2025 with a blue background."
                ),
                "page_numbers": [1],
            }
        ],
    }
    persisted: list[tuple[int, tuple[str, ...]]] = []
    monkeypatch.setattr(intelligence_backfill, "_candidates", lambda _user_id, _limit: [row])
    monkeypatch.setattr(
        intelligence_backfill,
        "_persist_tags",
        lambda item, tags: persisted.append((item["file_id"], tags)) or True,
    )

    result = await intelligence_backfill.run(
        apply=True,
        user_id=2,
        limit=None,
        normalize_tags_only=True,
    )

    assert result == 0
    assert persisted == [(8, ("cloud policy", "cloud security"))]
    record = json.loads(capsys.readouterr().out.splitlines()[0])
    assert record["changed"] is True
    assert record["tags"] == ["cloud policy", "cloud security"]


@pytest.mark.asyncio
async def test_tag_only_repair_leaves_empty_tags_for_model_backed_repair(monkeypatch, capsys) -> None:
    row = {
        "file_id": 10,
        "user_id": 2,
        "revision": 4,
        "source_name": "security-notes.txt",
        "mime_type": "text/plain",
        "quick_summary": "Network security controls protect cloud workloads.",
        "quick_tags": [],
        "parser_fingerprint": "text:v1",
        "chunks": [
            {
                "text": (
                    "Network security controls protect cloud workloads. "
                    "Network security controls require access reviews."
                ),
                "page_numbers": [1],
            }
        ],
    }
    persisted = []
    monkeypatch.setattr(intelligence_backfill, "_candidates", lambda _user_id, _limit: [row])
    monkeypatch.setattr(
        intelligence_backfill,
        "_persist_tags",
        lambda item, tags: persisted.append((item["file_id"], tags)) or True,
    )

    result = await intelligence_backfill.run(
        apply=True,
        user_id=2,
        limit=None,
        normalize_tags_only=True,
    )

    assert result == 0
    assert persisted == []
    record = json.loads(capsys.readouterr().out.splitlines()[0])
    assert record["tags"] == []
    assert record["changed"] is False
    assert record["persisted"] is False


@pytest.mark.asyncio
async def test_tag_only_repair_cleans_existing_tags_without_replacing_good_values(
    monkeypatch,
    capsys,
) -> None:
    row = {
        "file_id": 11,
        "user_id": 2,
        "revision": 4,
        "source_name": "security-notes.txt",
        "mime_type": "text/plain",
        "quick_summary": "Network security controls protect cloud workloads.",
        "quick_tags": ["cloud security", "the", "tool"],
        "parser_fingerprint": "text:v1",
        "chunks": [
            {
                "text": (
                    "Cloud security controls protect cloud workloads. "
                    "Network security controls require access reviews. "
                    "Network security controls are audited."
                ),
                "page_numbers": [1],
            }
        ],
    }
    persisted: list[tuple[int, tuple[str, ...]]] = []
    monkeypatch.setattr(intelligence_backfill, "_candidates", lambda _user_id, _limit: [row])
    monkeypatch.setattr(
        intelligence_backfill,
        "_persist_tags",
        lambda item, tags: persisted.append((item["file_id"], tags)) or True,
    )

    result = await intelligence_backfill.run(
        apply=True,
        user_id=2,
        limit=None,
        normalize_tags_only=True,
    )

    assert result == 0
    assert persisted == [(11, ("cloud security",))]
    record = json.loads(capsys.readouterr().out.splitlines()[0])
    assert record["tags"] == ["cloud security"]


@pytest.mark.asyncio
async def test_missing_tag_repair_proposes_only_tags_and_uses_missing_write_fence(
    monkeypatch,
    capsys,
) -> None:
    row = {
        "file_id": 12,
        "user_id": 2,
        "revision": 4,
        "source_name": "cloud-policy.txt",
        "source_sha256": "c" * 64,
        "mime_type": "text/plain",
        "quick_summary": "The policy defines cloud workload access controls.",
        "quick_tags": [],
        "doc_type": "policy",
        "intelligence_status": "fallback",
        "parser_fingerprint": "text:v1",
        "chunks": [
            {
                "text": (
                    "Cloud security controls protect production workloads. "
                    "Cloud security controls require access reviews."
                ),
                "page_numbers": [1],
            }
        ],
    }
    observed: dict[str, Any] = {}

    class _Service:
        def __init__(self, _settings: Any, *, priority_gate: Any = None) -> None:
            observed["priority_gate"] = priority_gate

        async def analyze(self, document: Any, chunks: Any) -> DocumentIntelligence:
            observed["document"] = document
            observed["chunks"] = chunks
            return DocumentIntelligence(
                doc_type="report",
                summary="A replacement summary that must not be written.",
                tags=("cloud security", "access controls"),
                status="model",
                model="local-test",
            )

    async def priority_gate() -> bool:
        return True

    persisted: list[tuple[int, tuple[str, ...], bool]] = []
    monkeypatch.setattr(
        intelligence_backfill,
        "_candidates",
        lambda _user_id, _limit, *, missing_tags_only=False: [row]
        if missing_tags_only
        else [],
    )
    monkeypatch.setattr(intelligence_backfill, "_raster_reindex_candidates", lambda *_args: [])
    monkeypatch.setattr(intelligence_backfill, "DocumentIntelligenceService", _Service)

    def persist_tags(item: dict[str, Any], tags: tuple[str, ...], *, require_missing: bool) -> bool:
        persisted.append((item["file_id"], tags, require_missing))
        return True

    monkeypatch.setattr(intelligence_backfill, "_persist_tags", persist_tags)

    result = await intelligence_backfill.run(
        apply=True,
        user_id=2,
        limit=10,
        missing_tags_only=True,
        priority_gate=priority_gate,
    )

    assert result == 0
    assert observed["priority_gate"] is priority_gate
    assert observed["document"].source_name == "cloud-policy.txt"
    assert persisted == [(12, ("cloud security", "access controls"), True)]
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["tags"] == ["cloud security", "access controls"]
    assert records[0]["preserved"] == [
        "quick_summary",
        "doc_type",
        "intelligence_status",
        "current_revision",
        "document_chunks",
        "embeddings",
    ]
    assert records[-1]["mode"] == "missing-tags-apply"
    assert records[-1]["targeted_reindex_rasters"] == 0


@pytest.mark.asyncio
async def test_missing_tag_repair_does_not_write_when_no_grounded_tag_is_available(
    monkeypatch,
    capsys,
) -> None:
    row = {
        "file_id": 13,
        "user_id": 2,
        "revision": 5,
        "source_name": "fragmented.pdf",
        "source_sha256": "d" * 64,
        "mime_type": "application/pdf",
        "quick_summary": "Only fragmented text was available.",
        "quick_tags": [],
        "doc_type": "other",
        "intelligence_status": "fallback",
        "parser_fingerprint": "pdf:text:v1",
        "chunks": [{"text": "M a n age m ent syste m s.", "page_numbers": [1]}],
    }

    class _Service:
        def __init__(self, _settings: Any) -> None:
            pass

        async def analyze(self, _document: Any, _chunks: Any) -> DocumentIntelligence:
            return DocumentIntelligence(
                doc_type="other",
                summary="Only fragmented text was available.",
                tags=(),
                status="fallback",
                fallback_reason="deterministic",
            )

    monkeypatch.setattr(
        intelligence_backfill,
        "_candidates",
        lambda _user_id, _limit, *, missing_tags_only=False: [row]
        if missing_tags_only
        else [],
    )
    monkeypatch.setattr(intelligence_backfill, "_raster_reindex_candidates", lambda *_args: [])
    monkeypatch.setattr(intelligence_backfill, "DocumentIntelligenceService", _Service)
    monkeypatch.setattr(
        intelligence_backfill,
        "_persist_tags",
        lambda *_args, **_kwargs: pytest.fail("empty tag proposals must not be written"),
    )

    result = await intelligence_backfill.run(
        apply=True,
        user_id=None,
        limit=None,
        missing_tags_only=True,
    )

    assert result == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["changed"] is False
    assert records[0]["persisted"] is False
    assert records[-1]["fallback_reasons"] == {"deterministic": 1}


@pytest.mark.asyncio
async def test_missing_tag_repair_reports_rasters_without_model_or_write(monkeypatch, capsys) -> None:
    raster = {
        "file_id": 14,
        "user_id": 2,
        "revision": 3,
        "source_name": "receipt.png",
        "mime_type": "image/png",
    }
    monkeypatch.setattr(
        intelligence_backfill,
        "_candidates",
        lambda _user_id, _limit, *, missing_tags_only=False: [],
    )
    monkeypatch.setattr(
        intelligence_backfill,
        "_raster_reindex_candidates",
        lambda *_args: [raster],
    )
    monkeypatch.setattr(
        intelligence_backfill,
        "_persist_tags",
        lambda *_args, **_kwargs: pytest.fail("raster metadata repair must not write"),
    )

    result = await intelligence_backfill.run(
        apply=True,
        user_id=2,
        limit=None,
        missing_tags_only=True,
    )

    assert result == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["status"] == "targeted_reindex_required"
    assert records[0]["reason"] == "raster_requires_source_pixels"
    assert records[-1]["candidates"] == 0
    assert records[-1]["targeted_reindex_rasters"] == 1


@pytest.mark.asyncio
async def test_full_metadata_backfill_preserves_raster_intelligence(monkeypatch, capsys) -> None:
    row = {
        "file_id": 9,
        "user_id": 2,
        "revision": 5,
        "source_name": "invoice.png",
        "source_sha256": "b" * 64,
        "mime_type": "image/png",
        "quick_summary": "A VLM-grounded invoice summary.",
        "quick_tags": ["spray paint"],
        "doc_type": "invoice",
        "intelligence_status": "model",
        "parser_fingerprint": "ollama-vision:v8:test",
        "chunks": [{"text": "Invoice for spray paint.", "page_numbers": [1]}],
    }

    class _Service:
        def __init__(self, _settings: Any) -> None:
            pass

        async def analyze(self, _document: Any, _chunks: Any) -> DocumentIntelligence:
            pytest.fail("raster metadata-only backfill must not call the text model")

    monkeypatch.setattr(intelligence_backfill, "_candidates", lambda _user_id, _limit: [row])
    monkeypatch.setattr(intelligence_backfill, "DocumentIntelligenceService", _Service)
    monkeypatch.setattr(
        intelligence_backfill,
        "_persist",
        lambda *_args: pytest.fail("raster metadata must not be overwritten"),
    )

    result = await intelligence_backfill.run(apply=True, user_id=2, limit=None)

    assert result == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["reason"] == "raster_requires_targeted_reindex"
    assert records[-1]["skipped_raster"] == 1
    assert records[-1]["changed"] == 0
