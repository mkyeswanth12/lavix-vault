"""@-mention token extraction and filename resolution for chat scope.

A typed-but-unpicked ``@name`` must never fall back to whole-DB search: names
that resolve to exactly one of the user's files join the request scope, and
anything else aborts the turn with a file-not-found answer. Resolution is
SELECT-only and case-insensitive exact match on ``original_filename``.

Safer-option notes (recorded, deliberate):
- Tokens never span bare whitespace: multi-word filenames must be picked via
  the picker (which sends integer ids). This avoids guessing word boundaries.
  The single exception is ``folder / file`` paths written by the @ menu
  (``@chat_uploads / photo.png``): the path tail is accepted and only the
  final segment resolves as the filename. A trailing slash with no filename
  (``@folder/``) resolves nothing here — folder scope rides taggedFolders
  and folder-name tokens are tolerated at validation, never as file scope.
- A name matching zero files, or more than one file, is unresolved: answering
  from the wrong same-named file is worse than asking the user to tag it.
  The single exception is a fully typed folder path (``@Books / 2 / 33 / 3 /
  ggg.png``): when the filename alone is ambiguous, the folder chain
  disambiguates — an exact root-down match on every level plus exactly one
  same-named file in the terminal folder resolves. Any mismatch along the
  chain keeps the old refusal.
- ``@`` inside emails (``a@b.com``) is ignored: the ``@`` must start the
  string or follow whitespace.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

# @token: @ at start/after-whitespace, then a filename-ish run (no spaces),
# optionally followed by " / "-separated path segments (the @ menu writes
# "@folder / file" for files inside folders). A segment may contain spaces
# only when it is immediately followed by a "/" path separator — this lets
# multi-word folder names resolve ("@Project NSA/") without making plain
# queries with spaces between words ambiguous ("@Project and @Health").
_AT_SEGMENT = r"[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}"
_AT_SPACED_SEGMENT = r"[A-Za-z0-9_][A-Za-z0-9_.\- ]*[A-Za-z0-9_.\-](?=\s*/)"
# Path separator: "/" joins folder+file only when a filename char follows
# ("@folder / file", "@folder/file"). A "/" followed by a space or end
# ("@folder/ ") terminates the mention as folder-only — the spaced segment
# before it keeps its spaces ("@Project NSA/").
_AT_SEP = r"(?:\s+/\s*|/(?=\S))"
# Multi-word filenames are only ever read inside an explicit folder+file
# path, anchored at a real file extension ("@sf / SalesForce User Manual.pdf
# is ...?" yields the full filename, then the query). Bare single-word
# mentions keep the old no-spaces behavior, so plain queries stay
# unambiguous. The extension list mirrors the vault's supported types.
_AT_EXTS = (
    r"pdf|png|jpe?g|gif|webp|bmp|svg|tiff?|docx?|xlsx?|pptx?|"
    r"txt|md|markdown|csv|tsv|json|ya?ml|zip|tar|gz|"
    r"mp4|m4a|mp3|wav|ogg|webm|mov|avi|mkv|epub|rtf|odt|ods|odp|html?|xml|log"
)
_AT_FILE_SPACED = rf"[A-Za-z0-9_][A-Za-z0-9_.\- ]*?\.(?i:{_AT_EXTS})\b"
_AT_PATH = rf"(?:{_AT_SPACED_SEGMENT}|{_AT_SEGMENT})(?:{_AT_SEP}(?:{_AT_SPACED_SEGMENT}|{_AT_SEGMENT}))*"
_AT_PATH_LAZY = rf"(?:{_AT_SPACED_SEGMENT}|{_AT_SEGMENT})(?:{_AT_SEP}(?:{_AT_SPACED_SEGMENT}|{_AT_SEGMENT}))*?"
# The folder+file branch leads with a lazy path so a spaced filename wins
# over the plain single-word tail ("@sf / SalesForce User Manual.pdf").
_AT_TOKEN_RE = re.compile(
    rf"(?:^|\s)@({_AT_PATH_LAZY}{_AT_SEP}{_AT_FILE_SPACED}|{_AT_PATH})"
)


def extract_at_tokens(text: str) -> list[str]:
    """Return deduplicated @-mention tokens in order of appearance.

    Path mentions (``@folder / file.png``) yield the final segment only;
    a trailing slash with no filename yields the folder token itself.
    """
    tokens: list[str] = []
    for match in _AT_TOKEN_RE.finditer(str(text or "")):
        raw = match.group(1)
        segments = [segment.strip().rstrip(".") for segment in raw.split("/")]
        segments = [segment for segment in segments if segment]
        if not segments:
            continue
        # Last segment is the token: a filename ("@folder / file.png" ->
        # "file.png") or, for a folder-only mention, the folder name
        # ("@Project NSA/" — the trailing slash is not captured, so the
        # spaced folder name is the final segment). Folder tokens are
        # tolerated downstream via folder_named_tokens.
        token = segments[-1]
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def extract_at_paths(text: str) -> dict[str, list[str]]:
    """Map each @-mention's final-segment key to its full path segments.

    Same walk as extract_at_tokens, but keeps the whole folder chain so
    duplicate filenames can be disambiguated by their typed path. Keys are
    casefolded final segments; the first mention wins. Folder-only mentions
    (trailing slash) map to their single folder-name segment.
    """
    paths: dict[str, list[str]] = {}
    for match in _AT_TOKEN_RE.finditer(str(text or "")):
        raw = match.group(1)
        segments = [segment.strip().rstrip(".") for segment in raw.split("/")]
        segments = [segment for segment in segments if segment]
        if not segments:
            continue
        key = segments[-1].casefold()
        if key and key not in paths:
            paths[key] = segments
    return paths


def _resolve_path_file_id(
    cursor: Any,
    user_id: int,
    segments: Sequence[str],
    candidates: Sequence[tuple[int, int | None]],
) -> int | None:
    """Disambiguate same-named files by their typed folder chain.

    Walks ``segments[:-1]`` down the user's folder tree (exactly one folder
    per level, else None), then requires exactly one candidate in the
    terminal folder. Returns the file id, or None (caller refuses).
    SELECT-only.
    """
    if len(segments) < 2 or not candidates:
        return None
    cursor.execute(
        """
        SELECT id, name, parent_id FROM folders
        WHERE user_id = %s AND is_deleted = FALSE
        """,
        (user_id,),
    )
    children: dict[tuple[int | None, str], list[int]] = {}
    for row in cursor.fetchall() or []:
        try:
            folder_id = int(row["id"])
        except (TypeError, ValueError, KeyError):
            continue
        name = str(row.get("name") or "").casefold()
        if folder_id <= 0 or not name:
            continue
        parent = row.get("parent_id")
        try:
            parent = None if parent is None else int(parent)
        except (TypeError, ValueError):
            continue
        children.setdefault((parent, name), []).append(folder_id)
    current: list[int | None] = [None]
    for level in segments[:-1]:
        key = level.casefold()
        nxt: list[int] = []
        for parent_id in current:
            nxt.extend(children.get((parent_id, key), []))
        nxt = list(dict.fromkeys(nxt))
        if len(nxt) != 1:
            return None
        current = nxt
    terminal = current[0]
    if terminal is None:
        return None
    matches = [file_id for file_id, file_folder in candidates if file_folder == terminal]
    if len(matches) != 1:
        return None
    return matches[0]


def resolve_mention_ids(
    cursor: Any,
    user_id: int,
    tokens: list[str],
    folder_paths: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[int], list[str]]:
    """Resolve mention tokens to file ids; return (ids, unresolved names).

    Exactly one case-insensitive filename match resolves; zero matches leave
    the name unresolved. Multiple matches resolve only via ``folder_paths``:
    the token's typed folder chain must match exactly one folder per level
    with exactly one same-named file in the terminal folder — otherwise the
    name stays unresolved.
    """
    if not tokens:
        return [], []
    lowered = [token.casefold() for token in tokens]
    cursor.execute(
        """
        SELECT id, original_filename, folder_id
        FROM files
        WHERE user_id = %s
          AND is_deleted = FALSE
          AND LOWER(original_filename) = ANY(%s)
        """,
        (user_id, lowered),
    )
    hits: dict[str, list[tuple[int, int | None]]] = {}
    for row in cursor.fetchall() or []:
        try:
            file_id = int(row["id"])
        except (TypeError, ValueError, KeyError):
            continue
        name = str(row.get("original_filename") or "").casefold()
        folder_id = row.get("folder_id")
        try:
            folder_id = None if folder_id is None else int(folder_id)
        except (TypeError, ValueError):
            folder_id = None
        if file_id > 0 and name:
            hits.setdefault(name, []).append((file_id, folder_id))
    resolved: list[int] = []
    unresolved: list[str] = []
    for token, key in zip(tokens, lowered, strict=True):
        candidates = hits.get(key, [])
        if len(candidates) == 1:
            if candidates[0][0] not in resolved:
                resolved.append(candidates[0][0])
        elif len(candidates) > 1:
            picked = None
            path = (folder_paths or {}).get(key)
            if path is not None and len(path) >= 2:
                picked = _resolve_path_file_id(cursor, user_id, list(path), candidates)
            if picked is not None and picked not in resolved:
                resolved.append(picked)
            else:
                unresolved.append(token)
        else:
            unresolved.append(token)
    return resolved, unresolved


def folder_named_tokens(
    cursor: Any, user_id: int, tokens: list[str]
) -> set[str]:
    """Return the subset of tokens matching the user's folder names.

    Folder scope rides taggedFolders (ids), so a folder-name @-token in text
    is display-only and must not fail validation. SELECT-only, exact
    case-insensitive match on non-deleted folders. Unknown tokens are never
    tolerated here — they stay unresolved upstream.
    """
    if not tokens:
        return set()
    cursor.execute(
        """
        SELECT name FROM folders
        WHERE user_id = %s AND is_deleted = FALSE
        """,
        (user_id,),
    )
    owned = {str(row.get("name") or "").casefold() for row in cursor.fetchall() or []}
    owned.discard("")
    return {token for token in tokens if token.casefold() in owned}
