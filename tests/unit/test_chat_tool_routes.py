from contextlib import contextmanager

import pytest
from fastapi import HTTPException

from app.routers.chat import tool_routes
from app.routers.chat.schemas import SemanticSearchRequest


class FakeRetriever:
    def __init__(self):
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "file_id": 4,
                "filename": "invoice.pdf",
                "mime_type": "application/pdf",
                "summary": "April invoice",
                "tags": ["invoice", "april"],
                "doc_type": "invoice",
                "score": 1.4,
                "content": "best matching invoice chunk",
                "chunk_id": "c1",
                "revision": 2,
                "provenance": [{"page": 1}],
                "reranked": True,
            },
            {
                "file_id": 4,
                "filename": "invoice.pdf",
                "mime_type": "application/pdf",
                "summary": "April invoice",
                "tags": ["invoice"],
                "doc_type": "invoice",
                "score": 0.8,
                "content": "duplicate file chunk",
                "chunk_id": "c2",
                "revision": 2,
                "provenance": [{"page": 2}],
                "reranked": True,
            },
            {
                "file_id": 9,
                "filename": "brochure.docx",
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "summary": "Product brochure",
                "tags": ["product"],
                "doc_type": "brochure",
                "score": 0.424,
                "content": "matching brochure chunk",
                "chunk_id": "c3",
                "revision": 1,
                "provenance": [],
                "reranked": False,
            },
        ]


@pytest.mark.asyncio
async def test_ai_search_returns_distinct_ranked_file_suggestions(monkeypatch):
    retriever = FakeRetriever()
    monkeypatch.setattr(tool_routes, "vault_retriever", retriever)

    response = await tool_routes.semantic_search(
        SemanticSearchRequest(query=" April invoice ", max_results=5),
        {"id": 77, "perm_ai": True},
    )

    assert retriever.calls == [
        {
            "user_id": 77,
            "query": "April invoice",
            "file_ids": None,
            "top_k": 20,
            "deep_search": False,
        }
    ]
    assert response["count"] == 2
    assert [item["file_id"] for item in response["results"]] == [4, 9]
    assert response["results"][0]["match_percentage"] == 100
    assert response["results"][0]["summary"] == "April invoice"
    assert response["results"][0]["tags"] == ["invoice", "april"]
    assert response["results"][0]["doc_type"] == "invoice"
    assert response["results"][1]["match_percentage"] == 42
    assert "relevance_score" not in response["results"][0]
    assert "matched_content" not in response["results"][0]
    assert "chunk_id" not in response["results"][0]


@pytest.mark.asyncio
async def test_ai_search_rejects_whitespace_query():
    with pytest.raises(HTTPException) as exc:
        await tool_routes.semantic_search(
            SemanticSearchRequest(query="   ", max_results=5),
            {"id": 77, "perm_ai": True},
        )
    assert exc.value.status_code == 422


def test_tags_read_current_generated_intelligence_independent_of_job_state(monkeypatch):
    class Cursor:
        query = ""

        def execute(self, query, params):
            self.query = query
            assert params == (77,)

        @staticmethod
        def fetchall():
            return [{"quick_tags": ["invoice", "april"]}, {"quick_tags": ["invoice"]}]

    class Connection:
        def __init__(self, cursor):
            self._cursor = cursor

        def cursor(self):
            return self._cursor

    cursor = Cursor()

    @contextmanager
    def database():
        yield Connection(cursor)

    monkeypatch.setattr(tool_routes, "get_db", database)
    response = tool_routes.get_all_tags({"id": 77, "perm_ai": True})

    normalized_sql = " ".join(cursor.query.lower().split())
    assert "f.ai_status = 'ready'" not in normalized_sql
    assert "f.user_granted_ai_access = true" in normalized_sql
    assert "f.current_revision is not null" in normalized_sql
    assert "to_jsonb(f)->>'intelligence_status'" in normalized_sql
    assert "in ('model', 'fallback')" in normalized_sql
    assert response == {
        "tags": [{"tag": "invoice", "count": 2}, {"tag": "april", "count": 1}],
        "total_unique_tags": 2,
    }


def test_stats_count_published_revisions_without_double_counting_failed_replacements(monkeypatch):
    class Cursor:
        def __init__(self):
            self.queries = []

        def execute(self, query, params):
            self.queries.append(query)
            assert params == (77,)

        def fetchone(self):
            return {"count": 4}

        def fetchall(self):
            return [
                {
                    "original_filename": "ready.pdf",
                    "mime_type": "application/pdf",
                    "user_granted_ai_access": True,
                    "current_revision": 1,
                    "ai_status": "ready",
                    "quick_tags": ["policy"],
                    "intelligence_status": "model",
                },
                {
                    "original_filename": "replacement.docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "user_granted_ai_access": True,
                    "current_revision": 2,
                    "ai_status": "failed",
                    "quick_tags": ["risk"],
                    "intelligence_status": "fallback",
                },
                {
                    "original_filename": "broken.pdf",
                    "mime_type": "application/pdf",
                    "user_granted_ai_access": True,
                    "current_revision": None,
                    "ai_status": "failed",
                    "quick_tags": [],
                    "intelligence_status": "pending",
                },
            ]

    class Connection:
        def __init__(self, cursor):
            self._cursor = cursor

        def cursor(self):
            return self._cursor

    cursor = Cursor()

    @contextmanager
    def database():
        yield Connection(cursor)

    monkeypatch.setattr(tool_routes, "get_db", database)
    response = tool_routes.get_ai_stats({"id": 77, "perm_ai": True})

    normalized_sql = " ".join(cursor.queries[0].lower().split())
    assert "original_filename, mime_type, user_granted_ai_access" in normalized_sql
    assert "where user_id = %s and is_deleted = false" in normalized_sql
    assert response["ai_ready"] == 2
    assert response["searchable"] == 2
    assert response["supported"] == 3
    # Terminal failed files surface via needs_attention instead of pinning
    # the progress denominator below 100%.
    assert response["indexable"] == 1
    assert response["needs_attention"] == 2
    assert response["tagged"] == 2
    assert response["model_summaries"] == 1
    assert response["fallback_summaries"] == 1
    assert response["failed"] == 1
    # Permanently failed file without a usable prior revision leaves the
    # actionable denominator; the failed-with-revision file stays counted.
    assert response["unactionable"] == 1
    assert response["actionable"] == 2
    # Beta-dev parity: the ready file plus the failed-with-revision file
    # both hold published revisions; only the broken file is missing.
    assert response["ai_ready"] == 2
    assert response["eligible"] == 1
    assert response["awaiting_grant"] == 0
    assert response["ready_percentage"] == 100.0
    assert response["all_files_total"] == 3


def test_stats_expose_queue_cancel_password_and_exact_supported_counts(monkeypatch):
    class Cursor:
        query_count = 0

        def execute(self, _query, _params):
            self.query_count += 1

        def fetchall(self):
            return [
                {
                    "original_filename": "queued.pdf",
                    "mime_type": "application/pdf",
                    "user_granted_ai_access": True,
                    "current_revision": None,
                    "ai_status": "queued",
                    "quick_tags": [],
                    "intelligence_status": "pending",
                },
                {
                    "original_filename": "cancelled.pdf",
                    "mime_type": "application/pdf",
                    "user_granted_ai_access": True,
                    "current_revision": None,
                    "ai_status": "cancelled",
                    "quick_tags": [],
                    "intelligence_status": "pending",
                },
                {
                    "original_filename": "locked.pdf",
                    "mime_type": "application/pdf",
                    "user_granted_ai_access": True,
                    "current_revision": None,
                    "ai_status": "password_required",
                    "quick_tags": [],
                    "intelligence_status": "pending",
                },
                {
                    "original_filename": "movie.mp4",
                    "mime_type": "video/mp4",
                    "user_granted_ai_access": False,
                    "current_revision": None,
                    "ai_status": "not_granted",
                    "quick_tags": [],
                    "intelligence_status": "pending",
                },
            ]

        def fetchone(self):
            return {"count": 0}

    class Connection:
        def __init__(self, cursor):
            self._cursor = cursor

        def cursor(self):
            return self._cursor

    cursor = Cursor()

    @contextmanager
    def database():
        yield Connection(cursor)

    monkeypatch.setattr(tool_routes, "get_db", database)
    response = tool_routes.get_ai_stats({"id": 77, "perm_ai": True})

    assert response["supported"] == 3
    assert response["active"] == 1
    assert response["queued"] == 1
    assert response["cancelled"] == 1
    assert response["password_required"] == 1
    assert response["media_files"] == 1
    # Locked (password_required) file is terminal without user action:
    # excluded from the progress denominator, counted for attention.
    assert response["indexable"] == 2
    assert response["needs_attention"] == 1
    # Cancelled + password-required files leave the actionable denominator:
    # only the queued file can still become searchable.
    assert response["unactionable"] == 2
    assert response["actionable"] == 1
    assert response["ready_percentage"] == 0.0


def _stats_with_rows(monkeypatch, rows):
    from contextlib import contextmanager

    class Cursor:
        def execute(self, _query, _params):
            pass

        def fetchall(self):
            return rows

        def fetchone(self):
            return {"count": 0}

    class Connection:
        def __init__(self, cursor):
            self._cursor = cursor

        def cursor(self):
            return self._cursor

    cursor = Cursor()

    @contextmanager
    def database():
        yield Connection(cursor)

    monkeypatch.setattr(tool_routes, "get_db", database)
    return tool_routes.get_ai_stats({"id": 77, "perm_ai": True})


def _ready_row(name):
    return {
        "original_filename": name,
        "mime_type": "application/pdf",
        "user_granted_ai_access": True,
        "current_revision": 1,
        "ai_status": "ready",
        "quick_tags": ["t"],
        "intelligence_status": "model",
    }


def test_stats_ninety_nine_searchable_one_failed_reaches_hundred_percent(monkeypatch):
    rows = [_ready_row(f"doc-{index:03d}.pdf") for index in range(99)]
    rows.append(
        {
            "original_filename": "broken.pdf",
            "mime_type": "application/pdf",
            "user_granted_ai_access": True,
            "current_revision": None,
            "ai_status": "failed",
            "quick_tags": [],
            "intelligence_status": "pending",
        }
    )
    response = _stats_with_rows(monkeypatch, rows)
    assert response["supported"] == 100
    assert response["searchable"] == 99
    assert response["failed"] == 1
    assert response["unactionable"] == 1
    assert response["actionable"] == 99
    assert response["all_files_total"] == 100
    # Beta-dev parity: 99 revisions exist; the failed file holds none.
    assert response["ai_ready"] == 99
    assert response["eligible"] == 99
    assert response["ready_percentage"] == 100.0
    assert response["needs_attention"] == 1


def test_stats_zero_actionable_is_zero_percent_without_error(monkeypatch):
    response = _stats_with_rows(
        monkeypatch,
        [
            {
                "original_filename": "broken.pdf",
                "mime_type": "application/pdf",
                "user_granted_ai_access": True,
                "current_revision": None,
                "ai_status": "failed",
                "quick_tags": [],
                "intelligence_status": "pending",
            }
        ],
    )
    assert response["supported"] == 1
    assert response["searchable"] == 0
    assert response["actionable"] == 0
    assert response["ready_percentage"] == 0


def test_stats_empty_vault_is_zero_percent_without_error(monkeypatch):
    response = _stats_with_rows(monkeypatch, [])
    assert response["actionable"] == 0
    assert response["ready_percentage"] == 0
    assert response["all_files_total"] == 0


def test_stats_all_failed_is_zero_percent_not_hundred(monkeypatch):
    """Every file failed: nothing searchable reads 0%, never 100%."""
    response = _stats_with_rows(
        monkeypatch,
        [
            {
                "original_filename": "broken-a.pdf",
                "mime_type": "application/pdf",
                "user_granted_ai_access": True,
                "current_revision": None,
                "ai_status": "failed",
                "quick_tags": [],
                "intelligence_status": "pending",
            },
            {
                "original_filename": "broken-b.pdf",
                "mime_type": "application/pdf",
                "user_granted_ai_access": True,
                "current_revision": None,
                "ai_status": "failed",
                "quick_tags": [],
                "intelligence_status": "pending",
            },
        ],
    )
    assert response["all_files_total"] == 2
    assert response["searchable"] == 0
    assert response["failed"] == 2
    assert response["ready_percentage"] == 0


def test_stats_unsupported_failed_row_never_shrinks_denominator(monkeypatch):
    """A failed row outside `supported` must not leak into `unactionable`."""
    response = _stats_with_rows(
        monkeypatch,
        [
            _ready_row("ok.pdf"),
            {
                "original_filename": "archive.zip",
                "mime_type": "application/zip",
                "user_granted_ai_access": True,
                "current_revision": None,
                "ai_status": "failed",
                "quick_tags": [],
                "intelligence_status": "pending",
            },
        ],
    )
    assert response["supported"] == 1
    assert response["unactionable"] == 0
    assert response["actionable"] == 1
    assert response["ai_ready"] == 1
    assert response["eligible"] == 1
    assert response["ready_percentage"] == 100.0


def test_stats_ready_percentage_uses_eligible_base(monkeypatch):
    """Eligible base: cancelled rows count via indexable, so the server
    percentage matches the frontend eligible-base formula."""
    response = _stats_with_rows(
        monkeypatch,
        [
            _ready_row("ok.pdf"),
            _ready_row("ok2.pdf"),
            {
                "original_filename": "stuck.pdf",
                "mime_type": "application/pdf",
                "user_granted_ai_access": True,
                "current_revision": None,
                "ai_status": "cancelled",
                "quick_tags": [],
                "intelligence_status": "pending",
            },
        ],
    )
    assert response["indexable"] == 3
    assert response["searchable"] == 2
    assert response["ready_percentage"] == round(2 / 3 * 100, 2)


def test_stats_ai_ready_ignores_consent_beta_parity(monkeypatch):
    """Beta-dev parity: a published revision counts as ai_ready with or
    without consent, while `searchable` keeps its consent gate."""
    response = _stats_with_rows(
        monkeypatch,
        [
            _ready_row("ok.pdf"),
            {
                "original_filename": "legacy.pdf",
                "mime_type": "application/pdf",
                "user_granted_ai_access": False,
                "current_revision": 2,
                "ai_status": "not_granted",
                "quick_tags": [],
                "intelligence_status": "pending",
            },
        ],
    )
    assert response["searchable"] == 1
    assert response["ai_ready"] == 2
    assert response["all_files_total"] == 2
    assert response["ready_percentage"] == 100.0


def _cancelled_row(name):
    return {
        "original_filename": name,
        "mime_type": "application/pdf",
        "user_granted_ai_access": True,
        "current_revision": None,
        "ai_status": "cancelled",
        "quick_tags": [],
        "intelligence_status": "pending",
    }


def _ungranted_row(name):
    return {
        "original_filename": name,
        "mime_type": "application/pdf",
        "user_granted_ai_access": False,
        "current_revision": None,
        "ai_status": "not_granted",
        "quick_tags": [],
        "intelligence_status": "pending",
    }


def test_stats_eligible_counts_backlog_once_without_double_count(monkeypatch):
    """eligible = indexable + grantable-but-ungranted. Cancelled rows live
    inside indexable already, so five cancelled files read eligible 7
    (not 12) with two searchable files present."""
    rows = [_ready_row("ok1.pdf"), _ready_row("ok2.pdf")]
    rows += [_cancelled_row(f"stuck-{index}.pdf") for index in range(5)]
    rows += [_ungranted_row("fresh1.pdf"), _ungranted_row("fresh2.pdf")]
    response = _stats_with_rows(monkeypatch, rows)
    assert response["indexable"] == 7
    assert response["awaiting_grant"] == 2
    assert response["eligible"] == 9
    assert response["searchable"] == 2
    assert response["ready_percentage"] == round(2 / 9 * 100, 2)


def test_stats_empty_eligible_is_zero_percent_without_error(monkeypatch):
    response = _stats_with_rows(monkeypatch, [])
    assert response["eligible"] == 0
    assert response["awaiting_grant"] == 0
    assert response["ready_percentage"] == 0
