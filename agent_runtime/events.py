"""Safe projection of CUGA/LangGraph state updates onto the runtime wire."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

AUTHORITATIVE_STREAM_PROVENANCE = "lavix.authoritative-synthesis.v1"


def qlog(text: object) -> str:
    """Length + hash for log lines: user text never lands in logs.

    Emitted at the call site for every logger argument derived from user
    queries, rewrites, or answers. Shape (length + stable short hash) keeps
    run correlation and volume debugging without the content.
    """
    body = str(text or "")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    return f"len={len(body)} sha={digest}"

_STATUS_BY_NODE = {
    "prepare": "planning",
    "call_model": "generating",
    "sandbox": "executing_tools",
    "CugaLiteSubgraph": "planning",
}

_FOLLOWUPS_OPEN = re.compile(r"<\s*LAVIX_FOLLOWUPS\s*>", re.IGNORECASE)
_FOLLOWUPS_CLOSE = re.compile(r"<\s*/\s*LAVIX_FOLLOWUPS\s*>", re.IGNORECASE)
_FOLLOWUPS_TAG = re.compile(
    r"<\s*/?\s*LAVIX_FOLLOWUPS\b[^>]*>[\s\S]*$", re.IGNORECASE
)
_FOLLOWUPS_MARKDOWN_HEADING = re.compile(
    r"^\*{0,2}\s*followups?(?:\s+questions?)?\s*:?\*{0,2}$", re.IGNORECASE
)
_FOLLOWUPS_NOTE_LINE = re.compile(r"^note:\s*followups?\b", re.IGNORECASE)
_LIST_ITEM_LINE = re.compile(r"^([-*•]\s+|\d+[.)]\s+)")
_PUBLIC_WRAPPER_TAG_NAMES = ("lavix_answer",)
_PUBLIC_WRAPPER_TAG = re.compile(
    r"<\s*/?\s*lavix_answer(?![a-z0-9_-])[^>]*>",
    re.IGNORECASE,
)
_PRIVATE_STATIC_TAG_NAMES = (
    "untrusted_response_style_preference",
    "untrusted_user_preferences",
    "analysis",
    "thinking",
    "think",
    "reasoning",
    "private",
    "internal",
    "system",
    "developer",
    "tool_call",
    "tool_result",
    "tool_output",
    "metadata",
)
_PRIVATE_LAVIX_TAG_ROOTS = (
    "lavix_runtime",
    "lavix_evidence",
    "lavix_tool",
    "lavix_execution",
)
_PRIVATE_TAG_PREFIX_ROOTS = (
    *_PRIVATE_STATIC_TAG_NAMES,
    *_PRIVATE_LAVIX_TAG_ROOTS,
    *_PUBLIC_WRAPPER_TAG_NAMES,
)
_PRIVATE_TAG_NAME = (
    r"(?:"
    + "|".join(re.escape(name) for name in _PRIVATE_STATIC_TAG_NAMES)
    + r"|(?:"
    + "|".join(re.escape(root) for root in _PRIVATE_LAVIX_TAG_ROOTS)
    + r")[a-z0-9_-]*)"
)
_PRIVATE_TAG_NAME_BOUNDARY = r"(?![a-z0-9_-])"
_INTERNAL_TAG_BLOCK = re.compile(
    rf"<\s*(?P<tag>{_PRIVATE_TAG_NAME}){_PRIVATE_TAG_NAME_BOUNDARY}[^>]*>.*?"
    rf"<\s*/\s*(?P=tag){_PRIVATE_TAG_NAME_BOUNDARY}[^>]*>",
    re.IGNORECASE | re.DOTALL,
)
_INTERNAL_TAG_OPEN = re.compile(
    rf"<\s*{_PRIVATE_TAG_NAME}{_PRIVATE_TAG_NAME_BOUNDARY}[^>]*>",
    re.IGNORECASE,
)
_INTERNAL_TAG_CLOSE = re.compile(
    rf"<\s*/\s*{_PRIVATE_TAG_NAME}{_PRIVATE_TAG_NAME_BOUNDARY}[^>]*>",
    re.IGNORECASE,
)
_TRUSTED_DIRECTIVE = re.compile(
    r"^[ \t]*\[\s*LAVIX\s+trusted\b[^\n]*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
_TRUSTED_MARKER = re.compile(r"\[\s*LAVIX\s+trusted\b", re.IGNORECASE)
_FENCE_MARKER = "```"
_STRONG_OPERATIONAL_LABEL = (
    r"(?:execution[ \t]+output|generated[ \t]+python|"
    r"tool[ \t]+(?:call|result|output)|user[ \t]+request|"
    r"request[ \t]+options?|personalization[ \t]+context"
    r"(?:[ \t]+\(not[ \t]+evidence\))?|logs?|"
    r"log[ \t]+output|stdout|stderr|analysis|reasoning|thoughts?)"
)
_OPERATIONAL_LABEL = rf"(?:{_STRONG_OPERATIONAL_LABEL}|evidence|sources?)"
_MARKDOWN_DECORATION = r"(?:[*_~`][ \t]*)"
_EXECUTION_MARKER = re.compile(
    rf"(?<!\w)(?:[ \t]*(?:\#+|[-+>]|{_MARKDOWN_DECORATION}))*[ \t]*"
    rf"{_STRONG_OPERATIONAL_LABEL}[ \t]*(?:[*_~`][ \t]*)*[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_EVIDENCE_MARKER = re.compile(
    rf"(?:^|\n|(?<=[.!?])\s+)[ \t]*"
    rf"(?:\#+[ \t]+|[-+>][ \t]+|{_MARKDOWN_DECORATION})*"
    rf"(?:evidence|sources?)[ \t]*(?:[*_~`][ \t]*)*[ \t]*:",
    re.IGNORECASE,
)
_OPERATIONAL_HEADING = re.compile(
    rf"(?:^|\n)[ \t]*(?:\#{{1,6}}[ \t]+|[-+>][ \t]+)?"
    rf"(?:[*_~`]{{1,3}}[ \t]*)?{_OPERATIONAL_LABEL}"
    rf"(?:[ \t]*[*_~`]{{1,3}})?[ \t]*:?[ \t]*(?=\n|$)",
    re.IGNORECASE,
)
_RAW_ARTIFACT = re.compile(
    r"(?:result\s*=\s*await\s+(?:search_vault|search_web|recall_graph|calculate|"
    r"current_datetime)|print\s*\(\s*result\s*\)|"
    r"(?:request_options?|tool_call|tool_result|generated_python)\s*[:=]|"
    r"[\"'](?:evidence|provenance|block_id|tool_call|tool_result)[\"']\s*:)",
    re.IGNORECASE | re.MULTILINE,
)
_PUBLIC_CITATION = re.compile(
    r"[ \t]*\[\s*[VW][1-9][0-9]*(?:\s*[,;]\s*[VW][1-9][0-9]*)*\s*\]",
    re.IGNORECASE,
)
# Model-written source attributions ("According to sportingnews.com, ...",
# "according to Flipkart, ..."). Only fires on domain/URL-shaped tokens or
# capitalized brand/org phrases; generic prose ("According to the report",
# "According to experts") is untouched. Backstop for the prompt ban —
# citations render from metadata, never from model prose.
_ATTRIBUTED_SOURCE = re.compile(
    r"[ \t]*\b(?i:according to)\s+(?:https?://)?(?:"
    r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}[^\s.,;!?]*(?:\s*,)?"
    r"|[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,2}\s*,"
    r")",
)
_ATTRIBUTION_TRIGGER = re.compile(r"\b(?i:according to)\b")
_ATTRIBUTION_FIRST_WORD = re.compile(r"\s*([A-Za-z][\w'.-]*)([\s,]|$)")


def _attribution_hold_index(value: str) -> int | None:
    """Start index from which an attribution-shaped tail must be held.

    A partially streamed "According to X" is indistinguishable from one
    the final projection will strip: emitting it first and stripping later
    trips the stream/final mismatch guard (fail-closed, answer suppressed).
    Hold from the trigger while the attribution could still complete into
    a stripped shape; release once the first word proves generic
    (complete, lowercase-initial, dotless — "the report", "experts").
    Over-holding is safe (delayed to the final flush); emitting a later
    stripped span is not.
    """

    hold: int | None = None
    # Partial trigger still forming ("acc", "accord", "according t"): it
    # may complete into a stripped attribution, and emitting the fragment
    # first would make the later strip a shrinking projection (fatal).
    # Length >= 4 keeps this narrow: only "accord*"/"according*" fragments
    # are prefixes of the trigger.
    partial = re.search(r"(\bacc[a-z]*(?:\s+t[a-z]*)?)$", value, re.IGNORECASE)
    if partial and len(partial.group(1)) >= 4:
        frag = partial.group(1).casefold()
        if "according to".startswith(frag):
            return partial.start(1)
    for trigger in _ATTRIBUTION_TRIGGER.finditer(value):
        rest = value[trigger.end() :]
        first = _ATTRIBUTION_FIRST_WORD.match(rest)
        if first is None:
            hold = trigger.start()  # bare trigger or mid-word: still forming
            continue
        word, terminator = first.group(1), first.group(2)
        if (
            terminator
            and terminator != ""
            and word[:1].islower()
            and "." not in word
            and not rest[len(first.group(0)) :].lstrip().startswith(".")
        ):
            continue  # provably generic prose: let it stream
        hold = trigger.start()
    return hold
_ANSWER_METADATA_JSON = re.compile(
    r'[ \t]*\{\s*"answer_id"\s*:\s*"[^"]*"\s*(?:,\s*"[^"]*"\s*:\s*(?:"[^"]*"|\d+(?:\.\d+)?)\s*)*\s*\}',
    re.IGNORECASE,
)
_ANSWER_METADATA_JSON_NULL = re.compile(
    r'[ \t]*\{\s*"answer_id"\s*:\s*null\s*(?:,\s*"[^"]*"\s*:\s*(?:"[^"]*"|null|\d+(?:\.\d+)?)\s*)*\s*\}',
    re.IGNORECASE,
)
_METADATA_HEADER_JSON = re.compile(
    r"\n?[ \t]*`?Metadata`?[ \t]*\n[ \t]*\{[^{}\n]*\}",
    re.IGNORECASE,
)
_ANSWER_ID_LITERAL = re.compile(
    r"[ \t]*answer_id\s*=\s*\d+[ \t]*",
    re.IGNORECASE,
)
_ANSWER_ID_MARKER = re.compile(
    r"(?:^|\n)[ \t]*answer_id(?:[ \t]*[:=][^\n]*)?(?=\n|$)",
    re.IGNORECASE,
)
_NO_CITATION_NEEDED = re.compile(
    r"\bno_citation_needed\b",
    re.IGNORECASE,
)
_CURRENT_QUESTION_PREFIX = re.compile(
    r"\[Current question\s*[—–-]\s*answer this only\]\s*\n*",
)
_LOG_MARKER = re.compile(
    r"(?:^|\n|(?<=[.!?])\s+)\s*\[\s*(?:debug|info|warning|error|trace)\s*\]",
    re.IGNORECASE,
)
_TIMESTAMPED_LOG = re.compile(
    r"(?:^|\n)[ \t]*(?:\[?[12][0-9]{3}-[01][0-9]-[0-3][0-9]"
    r"(?:[T ][0-2][0-9]:[0-5][0-9]:[0-6][0-9](?:[.,][0-9]{1,9})?"
    r"(?:Z|[+-][0-2][0-9]:?[0-5][0-9])?)?\]?[ \t]+)"
    r"\[?[ \t]*(?:debug|info|warning|error|trace)[ \t]*\]?\b",
    re.IGNORECASE,
)
_TIMESTAMP_DATE = re.compile(r"^[12][0-9]{3}-[01][0-9]-[0-3][0-9]")
_TIMESTAMP_TIME_PREFIX = re.compile(
    r"^[0-2]?[0-9]?(?::[0-5]?[0-9]?(?::[0-6]?[0-9]?"
    r"(?:[.,][0-9]{0,9})?(?:Z|[+-][0-2]?[0-9]?:?[0-5]?[0-9]?)?)?)?$",
    re.IGNORECASE,
)
_TIMESTAMP_TIME = re.compile(
    r"^[0-2][0-9]:[0-5][0-9]:[0-6][0-9](?:[.,][0-9]{1,9})?"
    r"(?:Z|[+-][0-2][0-9]:?[0-5][0-9])?$",
    re.IGNORECASE,
)
_LOG_LEVELS = ("debug", "info", "warning", "error", "trace")
_PLANNING_MARKER = re.compile(
    r"(?:^|\n|(?<=[.!?])\s+)(?:let me|i(?:'ll| will| need to)|we(?:'ll| will| need to))\s+"
    r"(?:search|query|call|invoke|use|inspect|retrieve)\b|"
    r"(?:^|\n|(?<=[.!?])\s+)(?:calling|invoking|running|searching|querying)\s+"
    r"(?:the\s+)?(?:tool|vault|web|database|search)\b",
    re.IGNORECASE,
)
_PRIVATE_PREFIXES = (
    "<lavix_followups>",
    "< lavix_followups",
    "</lavix_followups>",
    "<lavix_answer",
    "< lavix_answer",
    "</lavix_answer",
    "<analysis>",
    "<thinking>",
    "<think>",
    "<reasoning>",
    "<tool_output>",
    "<tool_result>",
    "<tool_call>",
    "<metadata>",
    "<metadata",
    "</metadata>",
    "< metadata",
    "</ metadata",
    "<private>",
    "<internal>",
    "<lavix_runtime",
    "< lavix_runtime",
    "<lavix_evidence",
    "< lavix_evidence",
    "<lavix_tool",
    "< lavix_tool",
    "<lavix_execution",
    "< lavix_execution",
    "<untrusted_response_style_preference",
    "< untrusted_response_style_preference",
    "<untrusted_user_preferences",
    "< untrusted_user_preferences",
    "[lavix trusted",
    "evidence:",
    "sources:",
    "source:",
    "execution output:",
    "generated python:",
    "tool call:",
    "tool result:",
    "tool output:",
    "logs:",
    "log output:",
    "stdout:",
    "stderr:",
    "analysis:",
    "reasoning:",
    "thoughts:",
    "user request:",
    "request option:",
    "request options:",
    '"evidence":',
    '"provenance":',
    '"block_id":',
    "result = await search_",
    "result = await recall_graph",
    "let me search",
    "let me query",
    "let me call",
    "i'll search",
    "i will search",
    "i need to call",
    "i need to use",
    "calling the tool",
    "invoking the tool",
)
_STRONG_PRIVATE_PREFIXES = tuple(
    prefix
    for prefix in _PRIVATE_PREFIXES
    if prefix.startswith(("<", "[", '"', "'"))
    or prefix.startswith(
        (
            "execution ",
            "generated ",
            "tool ",
            "user request",
            "request option",
            "log output",
            "stdout",
            "stderr",
            "analysis",
            "reasoning",
            "thoughts",
            "result =",
            "let me ",
            "i'll ",
            "i will ",
            "i need ",
            "calling ",
            "invoking ",
        )
    )
)


def _strip_private_fence(language: str, body: str, span: str) -> str:
    language = language.casefold()
    body = body.casefold()
    private_json = language in {"json", "jsonl"} and any(
        marker in body for marker in ('"evidence"', '"provenance"', '"block_id"')
    )
    private_code = _RAW_ARTIFACT.search(body) is not None
    return "" if private_json or private_code else span


_FENCE_LANGUAGE = re.compile(r"[a-z0-9_+-]*", re.IGNORECASE)


# Fence languages that are provably public code/text. An unterminated fence
# in one of these is rendered (fail-open) instead of dropped, so a truncated
# generation shows its code ending abruptly rather than vanishing silently.
# Absent from this set: json/jsonl (evidence-shaped payloads stay fail-closed
# via _strip_private_fence) and anything unrecognized (old cut behavior).
_PUBLIC_FENCE_LANGUAGES = frozenset(
    {
        "bash", "sh", "shell", "zsh", "powershell", "ps1", "bat", "cmd",
        "python", "py", "javascript", "js", "typescript", "ts", "ruby", "rb",
        "go", "rust", "java", "c", "cpp", "h", "csharp", "php", "perl",
        "r", "lua", "sql", "yaml", "yml", "toml", "ini", "cfg", "conf",
        "dockerfile", "html", "css", "xml", "markdown", "md", "text", "txt",
        "log", "diff", "csv",
    }
)


def _is_public_unclosed_fence(value: str, fence: int) -> bool:
    """Decide whether a trailing unmatched fence is displayable code."""
    after = value[fence + len(_FENCE_MARKER):]
    newline = after.find("\n")
    if newline < 0:
        return False
    language = _FENCE_LANGUAGE.match(after[:newline]).group(0).casefold()
    if language not in _PUBLIC_FENCE_LANGUAGES:
        return False
    body = after[newline + 1 :]
    if not body.strip():
        return False
    if _RAW_ARTIFACT.search(body) is not None:
        return False
    return _unfinished_private_tag_start(body) is None


def _strip_private_fences(value: str) -> str:
    """Remove private tool-output code fences in linear time.

    This replaces a regex whose ambiguous language/line split backtracked
    cubically on unterminated fences, which froze the runtime while streaming.
    Unterminated fences are left untouched, exactly like the old non-match.
    """

    if _FENCE_MARKER not in value:
        return value
    parts: list[str] = []
    cursor = 0
    length = len(value)
    while cursor < length:
        start = value.find(_FENCE_MARKER, cursor)
        if start < 0:
            parts.append(value[cursor:])
            break
        newline = value.find("\n", start + 3)
        first_close = value.find(_FENCE_MARKER, start + 3)
        if newline < 0 or (first_close >= 0 and first_close < newline):
            # Inline fence pair on one line (```lang```): empty body.
            if first_close < 0:
                parts.append(value[cursor:])
                break
            info = value[start + 3 : first_close]
            body = ""
            end = first_close + 3
        else:
            close = value.find(_FENCE_MARKER, newline + 1)
            if close < 0:
                parts.append(value[cursor:])
                break
            info = value[start + 3 : newline]
            body = value[newline + 1 : close]
            end = close + 3
        span = value[start:end]
        parts.append(value[cursor:start])
        language = _FENCE_LANGUAGE.match(info).group(0) if info else ""
        parts.append(_strip_private_fence(language, body, span))
        cursor = end
    return "".join(parts)


def _private_boundary(value: str) -> int:
    """Return the first complete private/operational construct."""

    boundary = len(value)
    for pattern in (
        _FOLLOWUPS_OPEN,
        _INTERNAL_TAG_OPEN,
        _INTERNAL_TAG_CLOSE,
        _TRUSTED_MARKER,
        _ANSWER_ID_MARKER,
        _EXECUTION_MARKER,
        _EVIDENCE_MARKER,
        _OPERATIONAL_HEADING,
        _RAW_ARTIFACT,
        _LOG_MARKER,
        _TIMESTAMPED_LOG,
        _PLANNING_MARKER,
    ):
        match = pattern.search(value)
        if match is not None:
            boundary = min(boundary, match.start())
    return boundary


def _log_level_prefix_is_ambiguous(value: str) -> bool:
    candidate = value.lstrip()
    if candidate.startswith("["):
        candidate = candidate[1:].lstrip()
    candidate = candidate.rstrip("] \t").casefold()
    return not candidate or any(level.startswith(candidate) for level in _LOG_LEVELS)


def _timestamped_log_prefix_is_ambiguous(value: str) -> bool:
    """Hold only a prefix that can still become one timestamped log line."""

    candidate = value.lstrip()
    if candidate.startswith("["):
        candidate = candidate[1:]
    date_template = "0000-00-00"
    if len(candidate) < len(date_template):
        for index, character in enumerate(candidate):
            expected = date_template[index]
            if (expected == "0" and not character.isdecimal()) or (expected != "0" and character != expected):
                return False
        return bool(candidate)
    if _TIMESTAMP_DATE.match(candidate) is None:
        return False

    remainder = candidate[len(date_template) :]
    if not remainder:
        return True
    if remainder.startswith("]"):
        after_bracket = remainder[1:]
        return bool(after_bracket[:1].isspace()) and _log_level_prefix_is_ambiguous(after_bracket)

    if remainder.startswith("T"):
        time_and_level = remainder[1:]
    elif remainder.startswith(" "):
        time_and_level = remainder.lstrip()
        if not time_and_level or not time_and_level[0].isdecimal():
            return _log_level_prefix_is_ambiguous(time_and_level)
    else:
        return False

    if not time_and_level:
        return True
    parts = time_and_level.split(maxsplit=1)
    time_token = parts[0].rstrip("]")
    if len(parts) == 1:
        return _TIMESTAMP_TIME_PREFIX.fullmatch(time_token) is not None
    return _TIMESTAMP_TIME.fullmatch(time_token) is not None and _log_level_prefix_is_ambiguous(parts[1])


def _answer_id_prefix_is_ambiguous(value: str) -> bool:
    """Hold a possible metadata key only when it begins the current line."""

    candidate = value.lstrip(" \t").casefold()
    return bool(candidate) and "answer_id".startswith(candidate)


def _prefix_has_private_boundary(value: str, index: int, prefix: str) -> bool:
    if prefix.startswith(("<", "[", '"', "'")):
        return True
    return index == 0 or not value[index - 1].isalnum()


def _unfinished_private_tag_start(value: str) -> int | None:
    """Locate a trailing tag fragment that can still be private protocol."""

    # A completed ``>`` lets the normal private-boundary projection take over.
    # Only the tail after the last one can contain an opening/closing tag whose
    # classification is not yet stable.
    search_from = value.rfind(">") + 1
    while True:
        tag_start = value.find("<", search_from)
        if tag_start < 0:
            return None

        cursor = tag_start + 1
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor == len(value):
            return tag_start
        if value[cursor] == "/":
            cursor += 1
            while cursor < len(value) and value[cursor].isspace():
                cursor += 1
            if cursor == len(value):
                return tag_start

        name_start = cursor
        while cursor < len(value) and (
            (value[cursor].isascii() and value[cursor].isalnum()) or value[cursor] == "_"
        ):
            cursor += 1
        candidate = value[name_start:cursor].casefold()
        remainder = value[cursor:]

        # Before a full name is present, hold only an exact prefix with no
        # intervening punctuation. Once a protected name/root is complete,
        # everything through its eventual ``>`` is private attributes/content.
        if (
            candidate
            and not remainder
            and any(name.startswith(candidate) for name in _PRIVATE_TAG_PREFIX_ROOTS)
        ):
            return tag_start
        remainder_starts_word = bool(remainder) and (remainder[0].isalnum() or remainder[0] == "_")
        if candidate in _PRIVATE_STATIC_TAG_NAMES and not remainder_starts_word:
            return tag_start
        if any(candidate.startswith(root) for root in _PRIVATE_LAVIX_TAG_ROOTS) and not remainder_starts_word:
            return tag_start
        if candidate in _PUBLIC_WRAPPER_TAG_NAMES and not remainder_starts_word:
            return tag_start

        search_from = tag_start + 1


def _stable_raw_prefix_length(value: str, *, include_ambiguous: bool) -> int:
    """Hold unfinished syntax that could still become private protocol."""

    cut = len(value)

    # Hold every incomplete private opening or closing tag from its first ``<``.
    # This applies during final projection too, so malformed protocol is dropped
    # at EOF instead of becoming a public suffix that streaming cannot retract.
    unfinished_tag = _unfinished_private_tag_start(value)
    if unfinished_tag is not None:
        cut = min(cut, unfinished_tag)

    fence = value.rfind("```")
    if fence >= 0 and value.count("```") % 2:
        # Fail-open only for provably public code fences; everything else
        # keeps the old cut so protocol can never leak through truncation.
        if not _is_public_unclosed_fence(value, fence):
            cut = min(cut, fence)

    citation = value.rfind("[")
    if citation >= 0 and "]" not in value[citation:]:
        suffix = value[citation:]
        if re.fullmatch(
            r"\[\s*(?:[VW](?:[1-9][0-9]*)?(?:\s*[,;]\s*[VW](?:[1-9][0-9]*)?)*)?\s*",
            suffix,
            re.IGNORECASE,
        ):
            cut = min(cut, citation)

    if include_ambiguous:
        # A timestamped log line is private from its first character, not only
        # once its later INFO/ERROR token arrives. Hold an ISO-date line until
        # it either proves to be ordinary prose or the completed projection
        # can discard it. This avoids an irreversible partial timestamp leak.
        line_start = value.rfind("\n") + 1
        if _timestamped_log_prefix_is_ambiguous(value[line_start:]):
            cut = min(cut, line_start)
        if _answer_id_prefix_is_ambiguous(value[line_start:]):
            cut = min(cut, line_start)
        # A bare trailing "Metadata" (no colon yet), optionally preceded by
        # fence/markdown ticks (the model sometimes emits "`Metadata"), is a
        # metadata marker under construction: hold the whole tail line, or it
        # streams publicly and the final "Metadata:" strip makes
        # stream/final diverge (runtime_protocol_error, answer suppressed).
        # Only the tail line is held: a lone line-start tick is held with it
        # because emitting "`" before knowing the next chunk is "Metadata"
        # is equally fatal (the projector can never retract). Mid-line
        # ticks (inline code) are unaffected.
        tail_word = value[line_start:].strip()
        tail_core = re.sub(r"^[`*_~#>+-]+", "", tail_word).strip()
        if (
            tail_core
            and ":" not in tail_core
            and " " not in tail_core
            and "metadata".startswith(tail_core.casefold())
        ):
            # Partial marker under construction ("Meta", "`Metadat").
            cut = min(cut, line_start)
        elif re.match(r"(?i)metadata\s*:", tail_core):
            # Complete footer line ("Metadata: ...", "`Metadata: ..."):
            # the final projection strips it, so streaming must hold it.
            cut = min(cut, line_start)
        else:
            # A complete Metadata-only line earlier in the buffer ("Metadata"
            # with no colon, followed by a JSON block): the final
            # projection strips the header plus block, so streaming must
            # hold from that line or stream/final diverge.
            for header in re.finditer(r"(?im)^[ \t]*`?metadata`?[ \t]*$", value):
                cut = min(cut, header.start())
                break
        if re.fullmatch(r"[`*_~#>+-]+", tail_word):
            # Lone line-start tick: emitting it before knowing the next
            # chunk is "Metadata" is equally fatal (no retraction).
            cut = min(cut, line_start)

        # A model-written attribution ("According to X") that the final
        # projection strips must never stream first: the fail-closed
        # projector cannot retract it and the mismatch guard would
        # suppress the whole answer.
        attribution_hold = _attribution_hold_index(value)
        if attribution_hold is not None:
            cut = min(cut, attribution_hold)

    folded = value.casefold()
    prefixes = _PRIVATE_PREFIXES if include_ambiguous else _STRONG_PRIVATE_PREFIXES
    longest = max(len(prefix) for prefix in prefixes) + 32
    for index in range(max(0, len(value) - longest), len(value)):
        suffix = folded[index:]
        if not suffix:
            continue
        normalized_suffix = re.sub(r"^[ \t*_~`#>+\-]+", "", suffix)
        normalized_suffix = re.sub(r"[*_~`]+(?=[ \t]*:)", "", normalized_suffix)
        normalized_suffix = re.sub(r"[*_~`]+$", "", normalized_suffix)
        normalized_suffix = re.sub(r"[ \t]+", " ", normalized_suffix).lstrip()
        if not normalized_suffix:
            if index == 0 or value[index - 1] in " \t\r\n.!?":
                cut = min(cut, index)
            continue
        for prefix in prefixes:
            normalized_prefix = re.sub(r"[ \t]+", " ", prefix)
            if normalized_prefix.startswith(normalized_suffix) and _prefix_has_private_boundary(
                value, index, prefix
            ):
                cut = min(cut, index)
                break
    return cut


def _strip_markdown_followup_block(text: str) -> str:
    """Remove a trailing markdown followup section leaked as prose.

    Some models disobey the <LAVIX_FOLLOWUPS>-only contract and write
    ``Followup questions:`` / ``Followups:`` sections (plus a mimicked
    ``Note: Followups could not be verified...`` line) into the answer
    body. The tag stripper cannot see those, and the trailing-question
    extractor stops at the heading/Note lines, so the whole block would
    persist and duplicate the suggestion chips. Anchored to a standalone
    trailing heading whose remaining lines are all blank, questions, list
    items, or the followup Note — mid-text discussion of followups and
    genuine atom caveats (``Note: August, 25 could not be verified...``)
    never match and are preserved.
    """
    if not text:
        return text
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if not _FOLLOWUPS_MARKDOWN_HEADING.match(line.strip()):
            continue
        end = index
        in_note = 0
        for tail_index in range(index + 1, len(lines)):
            stripped = lines[tail_index].strip()
            if not stripped:
                in_note = 0
                continue
            if in_note:
                # Wrapped continuation of the followup Note (capped: a real
                # paragraph after the block must survive).
                in_note += 1
                if in_note > 2:
                    break
                end = tail_index
                continue
            if stripped.endswith("?"):
                end = tail_index
                continue
            if _LIST_ITEM_LINE.match(stripped):
                end = tail_index
                continue
            if _FOLLOWUPS_NOTE_LINE.match(stripped):
                in_note = 1
                end = tail_index
                continue
            # Anything else (including genuine atom caveats) ends the block:
            # drop the heading through the last consumed line, keep the rest.
            break
        if end > index:
            del lines[index : end + 1]
            return "\n".join(lines).rstrip()
    # Lone trailing followup Note without a heading (same mimicry).
    stripped_lines = text.rstrip().split("\n")
    if stripped_lines and _FOLLOWUPS_NOTE_LINE.match(stripped_lines[-1].strip()):
        return "\n".join(stripped_lines[:-1]).rstrip()
    return text


def project_answer_text(value: str) -> str:
    """Apply the sole display projection used by streaming and persistence."""

    raw = str(value or "")
    visible = raw[: _stable_raw_prefix_length(raw, include_ambiguous=False)]
    visible = _PUBLIC_WRAPPER_TAG.sub("", visible)
    visible = _INTERNAL_TAG_BLOCK.sub("", visible)
    visible = _TRUSTED_DIRECTIVE.sub("", visible)
    visible = _CURRENT_QUESTION_PREFIX.sub("", visible)
    visible = _strip_private_fences(visible)

    visible = visible[: _private_boundary(visible)]
    cleaned = "\n".join(line.rstrip() for line in visible.splitlines())
    cleaned = _PUBLIC_CITATION.sub("", cleaned)
    cleaned = _ATTRIBUTED_SOURCE.sub("", cleaned)
    cleaned = _ANSWER_METADATA_JSON.sub("", cleaned)
    cleaned = _ANSWER_METADATA_JSON_NULL.sub("", cleaned)
    cleaned = _METADATA_HEADER_JSON.sub("", cleaned)
    cleaned = _ANSWER_ID_LITERAL.sub("", cleaned)
    cleaned = re.sub(r"[ \t]*answer_id\s*[:=]\s*\d{8}_\d{3,}\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = _ANSWER_ID_MARKER.sub("", cleaned)
    cleaned = re.sub(r"\n?[ \t]*`?Metadata:.*", "", cleaned, flags=re.IGNORECASE)
    # Model-emitted <metadata>...</metadata> blocks (with answer_id JSON
    # inside): strip the whole block so final agrees with the streaming
    # holdback above. Second pattern drops a truncated trailing opener.
    cleaned = re.sub(r"<\s*/?\s*metadata\b[^>]*>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?[ \t]*<\s*/?\s*metadata\b[^>]*$", "", cleaned, flags=re.IGNORECASE)
    # Bare trailing metadata key with no colon yet (truncated marker): drop
    # it so final agrees with the streaming holdback above.
    cleaned = re.sub(r"\n[ \t]*metadata[ \t]*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?[ \t]*run-scope:.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<lavix_answer_id>\d*</lavix_answer_id>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<answer_id>\s*</answer_id>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?[ \t]*Lavix Runtime Answer ID:.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = _NO_CITATION_NEEDED.sub("", cleaned)
    cleaned = re.sub(r"[ \t]*answer_id\s*=\s*(?=\s*$)", "", cleaned, flags=re.IGNORECASE)
    # Strip <LAVIX_FOLLOWUPS> tag and everything after it — prevents
    # followup questions from leaking into the displayed answer during
    # streaming (the tag is extracted separately by split_answer_metadata).
    cleaned = _FOLLOWUPS_TAG.sub("", cleaned)
    # Strip trailing markdown followup sections (model-protocol leakage the
    # tag stripper cannot see); suggestion chips arrive separately.
    cleaned = _strip_markdown_followup_block(cleaned)
    # Strip trailing followups JSON or key-only versions
    cleaned = re.sub(r'[ \t]*[,]?\s*"followups"\s*:\s*\[[\s\S]*?\]\s*[},]?\s*$', "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[ \t]*[,]?\s*followups\s*:\s*\[[\s\S]*?\]\s*[},]?\s*$', "", cleaned, flags=re.IGNORECASE)
    # Strip bare trailing date-like IDs (20230615_001)
    cleaned = re.sub(r"\n[ \t]*\d{8}_\d{3,}[ \t]*\n*$", "", cleaned)
    # Strip trailing dateline echoes: isolated ISO-date/datetime lines the
    # model appends after the answer ("2026-09-05", "2026-09-05 14:30:00").
    # The lookbehind requires real content before the line, and only a
    # date-only line matches, so genuine in-answer dates ("the event was
    # held on 2026-09-05") and sole-date answers are untouched — the same
    # distinction the Metadata/run_id stripping above makes.
    cleaned = re.sub(
        r"(?<=\S)\s*\n[ \t]*\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?[ \t]*\n*$",
        "",
        cleaned,
    )
    # Strip trailing lines that are only whitespace, answer_id markers,
    # or long bare numeric IDs that the LLM emits as metadata. Short numbers
    # (totals, years, counts) are legitimate answer content and stay.
    cleaned = re.sub(r"(?:\n[ \t]*\d{6,}[ \t]*)+\n*$", "", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_unchanged_public_text(value: str) -> bool:
    """Return whether text is already public and contains no held protocol prefix."""

    raw = str(value or "")
    if not raw or raw != raw.strip():
        return False
    return project_answer_text(raw) == raw and _stable_raw_prefix_length(raw, include_ambiguous=True) == len(
        raw
    )


def _strip_echoed_followups(text: str, questions: list[str]) -> str:
    """Remove trailing lines that echo followup questions from the answer body."""
    if not text or not questions:
        return text
    normalized = {(q.strip().rstrip("?").lower()) for q in questions if q.strip()}
    if not normalized:
        return text
    lines = text.split("\n")
    trimmed = len(lines)
    while trimmed > 0:
        candidate = lines[trimmed - 1].strip().rstrip("?").lower()
        if candidate and candidate in normalized:
            trimmed -= 1
        else:
            break
    if trimmed == len(lines):
        return text
    result = "\n".join(lines[:trimmed]).rstrip()
    return result if result else text


def _extract_trailing_question_followups(text: str) -> list[str]:
    """Extract trailing question-ending lines as followup candidates.

    When the 3B model emits followup questions as plain prose instead of
    inside the <LAVIX_FOLLOWUPS> tag, this heuristic picks up lines
    ending with '?' at the tail of the answer.
    """
    raw = str(text or "").strip()
    if not raw:
        return []
    lines = raw.splitlines()
    tail: list[str] = []
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            break
        if stripped.endswith("?") and len(stripped) <= 120:
            # Avoid extracting obvious inline questions that are part of
            # the answer body (e.g. "The question is: what year was it?").
            # Require the line to start with a capital letter or question word
            # and not contain sentence-continuation markers.
            if stripped[0].isupper() or stripped.lower().startswith(("what", "how", "when", "where", "why", "who", "which", "is ", "are ", "did ", "do ", "can ", "could ", "will ")):
                tail.append(stripped)
        else:
            break
    tail.reverse()
    # Deduplicate while preserving order
    seen: set[str] = set()
    questions: list[str] = []
    for q in tail:
        key = q.strip().rstrip("?").lower()
        if key not in seen:
            seen.add(key)
            questions.append(q)
    return questions[:4]


def split_answer_metadata(value: str) -> tuple[str, list[str]]:
    """Project an answer and extract only bounded same-call suggestion strings."""

    raw = str(value or "").strip()
    opening = _FOLLOWUPS_OPEN.search(raw)
    if opening is None:
        close = _FOLLOWUPS_CLOSE.search(raw)
        visible = raw[: close.start()] if close is not None else raw
        # Fallback: the 3B model sometimes emits followup questions as
        # plain prose instead of inside the <LAVIX_FOLLOWUPS> tag.  Extract
        # trailing question-ending lines as followup candidates so the
        # suggestion buttons still appear and the questions are stripped
        # from the answer body.
        followups = _extract_trailing_question_followups(visible)
        if followups:
            # Strip the trailing question lines from the visible answer.
            # Guard: never strip so aggressively that the entire answer
            # disappears — that happens with single-line greeting replies
            # like "Hello! How can I help you today?" which match themselves
            # as trailing followup questions.
            lines = visible.splitlines()
            trim = len(lines)
            normalized = {q.strip().rstrip("?").lower() for q in followups}
            while trim > 1:
                candidate = lines[trim - 1].strip().rstrip("?").lower()
                if candidate and candidate in normalized:
                    trim -= 1
                else:
                    break
            visible = "\n".join(lines[:trim]).rstrip()
        return project_answer_text(visible), followups

    visible = project_answer_text(raw[: opening.start()])
    closing = _FOLLOWUPS_CLOSE.search(raw, opening.end())
    if closing is None:
        return visible, []
    try:
        payload = json.loads(raw[opening.end() : closing.start()].strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return visible, []
    raw_questions = payload.get("followups") if isinstance(payload, Mapping) else None
    if not isinstance(raw_questions, list):
        return visible, []
    questions: list[str] = []
    for raw_question in raw_questions[:10]:
        if not isinstance(raw_question, str):
            continue
        if not is_unchanged_public_text(raw_question):
            continue
        question = " ".join(raw_question.split())[:500]
        if question and question not in questions:
            questions.append(question)
    if questions:
        visible = _strip_echoed_followups(visible, questions)
    return visible, questions


_NUMERIC_ATOM_HOLD = re.compile(
    r"[\u20b9$\u20ac\u00a3\u00a5]?\d[\d,]*(?:\.\d+)?%?"
    r"|\b(?:19|20)\d{2}-\d{2}-\d{2}\b"
    r"|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,3}(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)
_NUMERIC_TRAIL = re.compile(
    r"[\u20b9$\u20ac\u00a3\u00a5]?\d[\d,./-]*(?:\.\d*)?%?$"
    r"|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)[a-z]*$",
    re.IGNORECASE,
)


class NumericAtomHold:
    """Hold numeric/date atoms out of streamed deltas (BUG-009).

    Mirrors CredentialStreamScrubber: clean text streams immediately;
    from the first complete numeric/date atom onward output is held until
    finalize(). A trailing fragment that could extend into an atom is held
    too, so a number is never split across deltas. Invariant: callers must
    either release held text through the projector (verified path) or drop
    it when the final becomes a refusal (unverified path).
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: object) -> str:
        self._buffer += str(delta or "")
        cut = len(self._buffer)
        for match in _NUMERIC_ATOM_HOLD.finditer(self._buffer):
            if match.start() < cut:
                cut = match.start()
        trail = _NUMERIC_TRAIL.search(self._buffer[:cut])
        if trail is not None and trail.start() < cut:
            cut = trail.start()
        emit, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return emit

    def finalize(self) -> str:
        held, self._buffer = self._buffer, ""
        return held

    def drop(self) -> None:
        self._buffer = ""


class AnswerDeltaProjector:
    """Incrementally emit only prefixes stable under the public projection."""

    def __init__(self) -> None:
        self._raw = ""
        self._emitted = ""

    @property
    def public_text(self) -> str:
        return project_answer_text(self._raw)

    def feed(self, chunk: str, *, final: bool = False) -> str:
        self._raw += str(chunk or "")
        # ``final`` intentionally does not relax the projection. An unfinished
        # private marker is discarded rather than appended as a final remainder.
        raw = self._raw
        if not final:
            raw = raw[: _stable_raw_prefix_length(raw, include_ambiguous=True)]
        public = project_answer_text(raw)
        if not public.startswith(self._emitted):
            # Projection is deliberately fail-closed: an already emitted
            # prefix is never rewritten or replaced by internal state.
            return ""
        delta = public[len(self._emitted) :]
        self._emitted = public
        return delta


def _updates_from_event(event: Any) -> Mapping[str, Any] | None:
    # LangGraph with subgraphs=True: (namespace_tuple, updates_dict)
    if isinstance(event, tuple) and len(event) == 2 and isinstance(event[1], Mapping):
        return event[1]
    if isinstance(event, Mapping):
        return event
    return None


def _authoritative_final_answer(node_name: Any, node_state: Any) -> str | None:
    """Accept only terminal state shapes emitted by pinned CUGA graph nodes.

    The compiled Lite graph's successful terminal update is a top-level
    ``call_model`` state with ``execution_complete=True`` and ``script=None``.
    Its sandbox error path can also contain ``final_answer``, so recursive key
    discovery (or accepting sandbox state) would turn tool output into a public
    answer. The pinned standalone Lite graph has no ``FinalAnswerAgent`` node,
    so accepting that parent-graph shape would only widen this trust boundary.
    """

    if not isinstance(node_state, Mapping):
        return None
    raw_answer = node_state.get("final_answer")
    if not isinstance(raw_answer, str):
        return None
    if node_name != "call_model":
        return None
    if (
        node_state.get("execution_complete") is not True
        or "script" not in node_state
        or node_state.get("script") is not None
    ):
        return None
    return raw_answer


def _has_direct_error(node_name: Any, node_state: Any) -> bool:
    """Recognize only node-owned error fields, never nested tool payloads."""

    if node_name not in {"prepare", "call_model", "sandbox"}:
        return False
    if not isinstance(node_state, Mapping):
        return False
    return any(
        isinstance(node_state.get(key), str) and bool(str(node_state[key]).strip())
        for key in ("error", "error_message")
    )


class StateUpdateNormalizer:
    """Extracts safe progress/final events and suppresses all agent internals."""

    def __init__(self) -> None:
        self._final_emitted = False
        self._terminal_observed = False
        self._seen_statuses: set[str] = set()
        self._error_emitted = False

    @property
    def has_final_answer(self) -> bool:
        return self._final_emitted

    @property
    def has_terminal_state(self) -> bool:
        """Whether the pinned direct ``call_model`` terminal shape arrived."""

        return self._terminal_observed

    def feed(self, event: Any) -> list[dict[str, Any]]:
        updates = _updates_from_event(event)
        if not updates:
            return []

        output: list[dict[str, Any]] = []
        for node_name, node_state in updates.items():
            status = _STATUS_BY_NODE.get(str(node_name))
            if status and status not in self._seen_statuses:
                self._seen_statuses.add(status)
                output.append({"type": "status", "step": status})

            raw_answer = _authoritative_final_answer(node_name, node_state)
            if raw_answer is not None:
                self._terminal_observed = True
            if raw_answer is not None and not self._final_emitted:
                answer, followups = split_answer_metadata(raw_answer)
                if answer:
                    self._final_emitted = True
                    final_event: dict[str, Any] = {"type": "final", "answer": answer}
                    if followups:
                        final_event["followups"] = followups
                    output.append(final_event)

            if not self._error_emitted and _has_direct_error(node_name, node_state):
                self._error_emitted = True
                output.append(
                    {
                        "type": "error",
                        "code": "agent_execution_failed",
                        "message": "Agent execution failed",
                    }
                )
        return output
