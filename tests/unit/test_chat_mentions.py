"""Unit tests for @-mention extraction, file resolution, folder tolerance."""

from __future__ import annotations

from app.routers.chat.mentions import (
    extract_at_paths,
    extract_at_tokens,
    folder_named_tokens,
    resolve_mention_ids,
)


class _FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed.append((sql, params or ()))

    def fetchall(self) -> list[dict]:
        return self._rows


class _ScriptedCursor:
    """Answers the files query and the folders query with separate rows."""

    def __init__(self, *, files: list[dict], folders: list[dict]) -> None:
        self._files = files
        self._folders = folders
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed.append((sql, params or ()))

    def fetchall(self) -> list[dict]:
        sql = self.executed[-1][0]
        if "FROM folders" in sql:
            return self._folders
        return self._files


def test_extract_simple_token() -> None:
    assert extract_at_tokens("look at @photo.png please") == ["photo.png"]


def test_extract_path_mention_yields_final_segment() -> None:
    assert extract_at_tokens("@chat_uploads / pasted-image-20260923035417.png what is this?") == [
        "pasted-image-20260923035417.png"
    ]


def test_extract_multi_segment_path() -> None:
    assert extract_at_tokens("see @a / b / doc.pdf ok") == ["doc.pdf"]


def test_extract_trailing_slash_yields_folder_token() -> None:
    assert extract_at_tokens("open @chat_uploads/") == ["chat_uploads"]


def test_extract_multi_word_folder_trailing_slash() -> None:
    # A trailing slash marks a folder-only mention: the full spaced folder
    # name is the token, and text after the slash is query, not a filename.
    assert extract_at_tokens("@Project NSA/ good, what is this?") == ["Project NSA"]
    assert extract_at_tokens("open @Project NSA/") == ["Project NSA"]


def test_extract_multi_word_folder_without_slash_is_single_word() -> None:
    # Without a trailing slash, spaces still terminate the token (current
    # behavior preserved) — multi-word folders need the slash or the picker.
    assert extract_at_tokens("@Project NSA") == ["Project"]


def test_extract_multi_word_folder_with_file() -> None:
    # Folder path with a filename after the slash still resolves the file.
    assert extract_at_tokens("@Project NSA/photo.png") == ["photo.png"]
    assert extract_at_tokens("@Project NSA / photo.png") == ["photo.png"]


def test_extract_multi_word_does_not_break_plain_queries() -> None:
    # Spaces between words in a plain query must not be swallowed into a token.
    assert extract_at_tokens("@Project and @Health") == ["Project", "Health"]
    assert extract_at_tokens("what is the scope of @Project NSA") == ["Project"]


def test_extract_multi_word_filename_in_folder_path() -> None:
    # A folder+file path may carry a spaced filename, anchored at a real
    # file extension; the query tail after the extension is not swallowed.
    assert extract_at_tokens("@sf / SalesForce User Manual.pdf is Salesforce doc?") == [
        "SalesForce User Manual.pdf"
    ]
    assert extract_at_tokens(
        "@images / WhatsApp Image 2025-06-03 at 6.59.43 PM.jpeg what is CPU usage"
    ) == ["WhatsApp Image 2025-06-03 at 6.59.43 PM.jpeg"]
    assert extract_at_tokens("@sf / Report.PDF details") == ["Report.PDF"]


def test_extract_multi_word_filename_without_extension_falls_back() -> None:
    # No extension anchor: the old single-word tail applies, never a guess.
    assert extract_at_tokens("@sf / notes check the pdf version") == ["notes"]


def test_extract_at_paths_carries_spaced_filename() -> None:
    assert extract_at_paths("@sf / SalesForce User Manual.pdf is it?") == {
        "salesforce user manual.pdf": ["sf", "SalesForce User Manual.pdf"]
    }


def test_resolve_spaced_filename_exact_match() -> None:
    cursor = _ScriptedCursor(
        files=[{"id": 302, "original_filename": "SalesForce User Manual.pdf", "folder_id": 10}],
        folders=[],
    )
    tokens = extract_at_tokens("@sf / SalesForce User Manual.pdf is Salesforce doc?")

    resolved, missing = resolve_mention_ids(cursor, 2, tokens)

    assert resolved == [302]
    assert missing == []


def test_resolve_spaced_duplicate_filenames_disambiguate_by_path() -> None:
    cursor = _ScriptedCursor(
        files=[
            {"id": 1, "original_filename": "Sales Report.docx", "folder_id": 20},
            {"id": 2, "original_filename": "Sales Report.docx", "folder_id": 21},
        ],
        folders=[
            {"id": 20, "name": "q1", "parent_id": None},
            {"id": 21, "name": "q2", "parent_id": None},
        ],
    )
    text = "@q2 / Sales Report.docx figures?"
    tokens = extract_at_tokens(text)
    paths = extract_at_paths(text)

    assert tokens == ["Sales Report.docx"]
    resolved, missing = resolve_mention_ids(cursor, 2, tokens, folder_paths=paths)

    assert resolved == [2]
    assert missing == []


def test_extract_ignores_email_and_plain_prose() -> None:
    assert extract_at_tokens("mail me at a@b.com about @report final") == ["report"]
    assert extract_at_tokens("what is the URL of this vault?") == []


def test_extract_dedupes_in_order() -> None:
    assert extract_at_tokens("@a.png and @a.png then @b.png") == ["a.png", "b.png"]


def test_resolve_exact_single_match() -> None:
    cursor = _FakeCursor([{"id": 7, "original_filename": "Photo.PNG"}])

    resolved, missing = resolve_mention_ids(cursor, 1, ["photo.png"])

    assert resolved == [7]
    assert missing == []


def test_resolve_zero_or_ambiguous_is_unresolved() -> None:
    cursor = _FakeCursor([])

    assert resolve_mention_ids(cursor, 1, ["ghost.png"]) == ([], ["ghost.png"])

    dupes = _FakeCursor(
        [
            {"id": 7, "original_filename": "same.png"},
            {"id": 8, "original_filename": "same.png"},
        ]
    )

    assert resolve_mention_ids(dupes, 1, ["same.png"]) == ([], ["same.png"])


def test_folder_named_tokens_tolerated_case_insensitively() -> None:
    cursor = _FakeCursor([{"name": "chat_uploads"}, {"name": "Receipts"}])

    assert folder_named_tokens(cursor, 1, ["chat_uploads", "ghost"]) == {"chat_uploads"}
    assert folder_named_tokens(cursor, 1, ["RECEIPTS"]) == {"RECEIPTS"}
    assert folder_named_tokens(cursor, 1, []) == set()


def test_extract_reported_incident_message() -> None:
    text = "@chat_uploads / pasted-image-20260923035417.png what is the URL of this Lavix Vault?"
    assert extract_at_tokens(text) == ["pasted-image-20260923035417.png"]


def test_extract_at_paths_keeps_full_chain() -> None:
    assert extract_at_paths("@Books / 2 / 33 / 3 / ggg.png what is this?") == {
        "ggg.png": ["Books", "2", "33", "3", "ggg.png"]
    }
    assert extract_at_paths("@Project NSA/ good, what is this?") == {
        "project nsa": ["Project NSA"]
    }
    assert extract_at_paths("@photo.png and @photo.png") == {"photo.png": ["photo.png"]}


def _dup_files() -> list[dict]:
    return [
        {"id": 377, "original_filename": "ggg.png", "folder_id": 28},
        {"id": 303, "original_filename": "ggg.png", "folder_id": 29},
    ]


def _dup_folders() -> list[dict]:
    return [
        {"id": 5, "name": "Books", "parent_id": None},
        {"id": 26, "name": "2", "parent_id": 5},
        {"id": 27, "name": "33", "parent_id": 26},
        {"id": 28, "name": "3", "parent_id": 27},
        {"id": 9, "name": "images", "parent_id": None},
        {"id": 29, "name": "2", "parent_id": 9},
    ]


def test_resolve_path_disambiguates_duplicate_filenames() -> None:
    cursor = _ScriptedCursor(files=_dup_files(), folders=_dup_folders())
    tokens = extract_at_tokens("@Books / 2 / 33 / 3 / ggg.png what is this?")
    paths = extract_at_paths("@Books / 2 / 33 / 3 / ggg.png what is this?")

    resolved, missing = resolve_mention_ids(cursor, 2, tokens, folder_paths=paths)

    assert resolved == [377]
    assert missing == []


def test_resolve_path_picks_the_other_twin() -> None:
    cursor = _ScriptedCursor(files=_dup_files(), folders=_dup_folders())
    tokens = extract_at_tokens("@images / 2 / ggg.png what is this?")
    paths = extract_at_paths("@images / 2 / ggg.png what is this?")

    resolved, missing = resolve_mention_ids(cursor, 2, tokens, folder_paths=paths)

    assert resolved == [303]
    assert missing == []


def test_resolve_path_wrong_chain_still_refuses() -> None:
    cursor = _ScriptedCursor(files=_dup_files(), folders=_dup_folders())
    tokens = extract_at_tokens("@Books / WRONG / ggg.png what is this?")
    paths = extract_at_paths("@Books / WRONG / ggg.png what is this?")

    resolved, missing = resolve_mention_ids(cursor, 2, tokens, folder_paths=paths)

    assert resolved == []
    assert missing == ["ggg.png"]


def test_resolve_path_ambiguous_folder_level_refuses() -> None:
    cursor = _ScriptedCursor(
        files=[
            {"id": 1, "original_filename": "f.png", "folder_id": 10},
            {"id": 2, "original_filename": "f.png", "folder_id": 11},
        ],
        folders=[
            {"id": 10, "name": "x", "parent_id": None},
            {"id": 11, "name": "x", "parent_id": None},
        ],
    )

    resolved, missing = resolve_mention_ids(
        cursor, 1, ["f.png"], folder_paths={"f.png": ["x", "f.png"]}
    )

    assert resolved == []
    assert missing == ["f.png"]


def test_resolve_single_match_never_queries_folders() -> None:
    cursor = _ScriptedCursor(
        files=[{"id": 7, "original_filename": "Photo.PNG", "folder_id": 3}],
        folders=[],
    )

    resolved, missing = resolve_mention_ids(cursor, 1, ["photo.png"])

    assert resolved == [7]
    assert missing == []
    assert sum("FROM folders" in sql for sql, _ in cursor.executed) == 0
