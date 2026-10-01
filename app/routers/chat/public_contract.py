"""Single public projection boundary for chat answers, suggestions, and sources.

The agent runtime and retrieval stack intentionally retain richer execution
state.  Nothing in this module accepts that state as public by default: callers
must project it through these helpers before streaming or persisting it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from agent_runtime.events import (
    is_unchanged_public_text,
    project_answer_text,
    split_answer_metadata,
)

MAX_FOLLOWUP_CHARS = 64
MAX_FOLLOWUP_WORDS = 10

_CITATION_ID = re.compile(r"^[VW][1-9][0-9]*$")
_FOLLOWUP_PLACEHOLDER = re.compile(
    r"^(?:(?:another|a|the|relevant|useful|suggested|possible)\s+)*"
    r"(?:(?:next|follow[ -]?up)\s+)?question(?:\s+\d+)?$",
    re.IGNORECASE,
)
_GENERIC_FOLLOWUP = re.compile(
    r"^(?:what (?:happened|changed)(?: immediately)? (?:before|after|next|as a result)(?: this)?|"
    r"when did that happen\??|"
    r"what caused this\??|"
    r"tell me more about .+|"
    r"what (?:is|are) the (?:most )?(?:practical )?next steps?|"
    r"what (?:trade[ -]?offs|context) (?:should we consider first|could change this conclusion)|"
    r"which evidence (?:most )?directly supports this answer)\??$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\w'-]+", re.UNICODE)
_CAPITALIZED_DEICTIC = re.compile(r"\b(?:Here|There|This|That|These|Those)\b")
_TOPIC_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "again",
        "all",
        "also",
        "an",
        "and",
        "answer",
        "are",
        "before",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "here",
        "how",
        "in",
        "is",
        "it",
        "most",
        "of",
        "on",
        "or",
        "say",
        "should",
        "source",
        "sources",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
    }
)
_TRACKING_QUERY_KEYS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid"})


def _compact_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip(" \"'\u201c\u201d")
    return text[:limit]


def sanitize_answer(answer: str) -> tuple[str, list[str]]:
    """Return display-safe answer text and raw same-call follow-up candidates."""

    visible, followups = split_answer_metadata(answer)
    return project_answer_text(visible), followups


def normalized_match_percentage(source: Mapping[str, Any]) -> int:
    """Normalize both modern percentages and legacy fractional/raw scores."""

    raw = source.get("match_percentage")
    from_score = raw is None
    if from_score:
        raw = source.get("score")
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(value):
        return 0
    if from_score and 0.0 <= value <= 1.0:
        value *= 100.0
    return round(max(0.0, min(100.0, value)))


def _public_url(value: Any) -> str | None:
    url = _compact_text(value, limit=2_048)
    if not url:
        return None
    parsed = urlsplit(url)
    return url if parsed.scheme.casefold() in {"http", "https"} and parsed.netloc else None


def _canonical_web_url(value: Any) -> str | None:
    """Return a stable public page URL while dropping fragments/tracking noise."""

    url = _public_url(value)
    if url is None:
        return None
    parsed = urlsplit(url)
    try:
        host = (parsed.hostname or "").casefold().encode("idna").decode("ascii")
        port = parsed.port
    except (UnicodeError, ValueError):
        return None
    if not host:
        return None
    scheme = parsed.scheme.casefold()
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if port is None or default_port else f"{host}:{port}"
    if parsed.username or parsed.password:
        # Public source cards never need embedded URL credentials.
        return None
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.casefold() not in _TRACKING_QUERY_KEYS and not key.casefold().startswith("utm_")
        ],
        doseq=True,
    )
    return urlunsplit((scheme, netloc, parsed.path or "/", query, ""))


MAX_WEB_CARDS = 3


def project_public_sources(
    sources: Sequence[Any],
    report: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Reduce evidence to the lean source-card contract.

    When ``report`` is provided, every dropped item appends
    ``{"id": ..., "reason": ...}`` with one of: ``invalid-id`` (ID-mismatch),
    ``kind-mismatch``, ``gate`` (gateway relevance flag), ``verify``
    (citation verification), ``grounding`` (post-answer grounding gap),
    ``dedupe``, ``cap-3``, ``cap-40`` (evidence-store bound). Callers use it
    for stage-8 discard observability; the default ``None`` preserves the
    old signature exactly.
    """

    def _drop(raw: Any, reason: str) -> None:
        if report is None:
            return
        try:
            raw_id = str((raw.get("id") if isinstance(raw, Mapping) else "") or "?")
        except Exception:
            raw_id = "?"
        report.append({"id": raw_id, "reason": reason})

    projected: list[dict[str, Any]] = []
    positions: dict[tuple[str, Any], int] = {}
    for raw in sources:
        if not isinstance(raw, Mapping):
            _drop(raw, "invalid-id")
            continue
        citation_id = str(raw.get("id") or "").strip().upper()
        if not _CITATION_ID.fullmatch(citation_id):
            _drop(raw, "invalid-id")
            continue
        kind = "vault" if citation_id.startswith("V") else "web"
        explicit_kind = str(raw.get("kind") or "").strip().casefold()
        if explicit_kind in {"vault", "web"} and explicit_kind != kind:
            _drop(raw, "kind-mismatch")
            continue

        item: dict[str, Any] = {
            "id": citation_id,
            "kind": kind,
            "match_percentage": normalized_match_percentage(raw),
        }
        if kind == "vault":
            try:
                file_id = int(raw.get("file_id"))
            except (TypeError, ValueError):
                file_id = 0
            if file_id > 0:
                item["file_id"] = file_id
        filename = _compact_text(raw.get("filename"), limit=500)
        title = _compact_text(raw.get("title"), limit=500)
        mime_type = _compact_text(raw.get("mime_type"), limit=255)
        url = _canonical_web_url(raw.get("url"))
        if filename:
            item["filename"] = filename
        if title:
            item["title"] = title
        if mime_type:
            item["mime_type"] = mime_type
        if url:
            item["url"] = url
        # Carry the gateway relevance flag through so the drop-block below
        # can actually fire. Items without the flag keep legacy behavior.
        if isinstance(raw.get("relevant"), bool):
            item["relevant"] = raw["relevant"]
        if kind == "vault" and item.get("file_id"):
            key: tuple[str, Any] = (kind, item["file_id"])
        elif kind == "web" and url:
            key = (kind, url)
        else:
            key = (kind, citation_id)
        position = positions.get(key)
        if position is None:
            positions[key] = len(projected)
            projected.append(item)
        elif item["match_percentage"] > projected[position]["match_percentage"]:
            # A source group keeps its earliest rank while showing its strongest hit.
            _drop(raw, "dedupe")
            projected[position] = item
        else:
            _drop(raw, "dedupe")

    # Web cards: drop results the gateway flagged irrelevant, then cap at the
    # three strongest. Vault cards keep their file-level grouping untouched.
    web_cards = [item for item in projected if item["kind"] == "web" and not item.pop("relevant", True)]
    if web_cards:
        dropped = {id(item) for item in web_cards}
        for item in web_cards:
            _drop({"id": item.get("id")}, "gate")
        projected = [item for item in projected if id(item) not in dropped]
    capped: list[dict[str, Any]] = []
    web_seen = 0
    for item in projected:
        if item["kind"] == "web":
            web_seen += 1
            if web_seen > MAX_WEB_CARDS:
                _drop({"id": item.get("id")}, "cap-3")
                continue
        capped.append(item)
    return capped


def _normalized_words(value: str) -> list[str]:
    return [word.casefold() for word in _WORD.findall(value)]


def _topic_words(value: str) -> set[str]:
    return {
        word
        for word in _normalized_words(value)
        if len(word) >= 2 and word not in _TOPIC_STOPWORDS and not word.isdecimal()
    }


def _topics_overlap(left: set[str], right: set[str]) -> bool:
    exact = left & right
    if exact:
        return True
    prefix_hits = sum(
        1
        for first in left
        for second in right
        if min(len(first), len(second)) >= 2 and (first.startswith(second) or second.startswith(first))
    )
    # A single shared/prefix word lets tangential questions through on a 3B
    # model. Long candidates must clear at least two topic signals.
    if len(left) <= 2:
        return prefix_hits >= 1
    return (len(exact) + prefix_hits) >= 2


def _contains_substantial_echo(question: str, original: str) -> bool:
    question_words = _normalized_words(question)
    original_words = _normalized_words(original)
    if len(original_words) < 4:
        return " ".join(original_words) == " ".join(question_words)
    question_set = set(question_words)
    overlap = sum(1 for word in original_words if word in question_set)
    return overlap / len(original_words) >= 0.75


def _contains_source_title(question: str, sources: Sequence[Mapping[str, Any]]) -> bool:
    question_folded = " ".join(_normalized_words(question))
    for source in sources:
        label = str(source.get("title") or source.get("filename") or "")
        words = _normalized_words(label)
        if not words:
            continue
        label_folded = " ".join(words)
        if len(label_folded) >= 8 and label_folded in question_folded:
            return True
        if len(words) >= 4 and any(
            " ".join(words[index : index + 4]) in question_folded for index in range(len(words) - 3)
        ):
            return True
    return False


def _valid_followup(
    value: Any,
    *,
    message: str,
    answer: str,
    sources: Sequence[Mapping[str, Any]],
) -> str | None:
    if not isinstance(value, str):
        return None
    # Suggestions are public text, not a second execution channel. Reject the
    # entire candidate when the canonical projector would remove or truncate
    # any part; never present a sanitized rewrite of operational/private text.
    if not is_unchanged_public_text(value):
        return None
    question = " ".join(value.split()).strip(" \"'\u201c\u201d")
    if not question:
        return None
    if not question.endswith("?"):
        question = f"{question.rstrip('.!')}?"
    words = _normalized_words(question)
    normalized = " ".join(words)
    question_topics = _topic_words(question)
    context_topics = _topic_words(f"{message} {answer}")
    if (
        len(question) > MAX_FOLLOWUP_CHARS
        or not words
        or len(words) > MAX_FOLLOWUP_WORDS
        or _FOLLOWUP_PLACEHOLDER.fullmatch(normalized)
        or _GENERIC_FOLLOWUP.fullmatch(question)
        or _CAPITALIZED_DEICTIC.search(question)
        or "lavix" in normalized
        or "followups" in normalized
        or _contains_substantial_echo(question, message)
        or _contains_source_title(question, sources)
        or not question_topics
        or not context_topics
        or not _topics_overlap(question_topics, context_topics)
    ):
        return None
    return question



def resolve_followups(
    candidates: Sequence[Any],
    *,
    message: str,
    sources: Sequence[Mapping[str, Any]],
    answer: str = "",
) -> list[str]:
    """Validate zero-to-two same-call suggestions without fabricating any."""

    questions: list[str] = []
    normalized_seen: set[str] = set()
    for candidate in candidates:
        question = _valid_followup(candidate, message=message, answer=answer, sources=sources)
        key = " ".join(_normalized_words(question or ""))
        if not question or key in normalized_seen:
            continue
        questions.append(question)
        normalized_seen.add(key)
        if len(questions) == 2:
            return questions
    # Deliberately no fallback fabrication: an empty suggestion row beats an
    # irrelevant "Tell me more about <random web title>".
    return questions


def sanitize_public_message(value: Any) -> dict[str, Any] | None:
    """Project a durable/legacy message without exposing private chat state."""

    if not isinstance(value, Mapping):
        return None
    role = value.get("role")
    content = value.get("content")
    if role not in {"user", "assistant"} or not isinstance(content, str):
        return None

    embedded_followups: list[str] = []
    if role == "assistant":
        content, embedded_followups = sanitize_answer(content)
    else:
        content = content.strip()
    if not content:
        return None

    raw_sources = value.get("sources")
    has_source_list = isinstance(raw_sources, list)
    sources = project_public_sources(raw_sources if has_source_list else [])
    message: dict[str, Any] = {"role": role, "content": content}
    if sources or has_source_list:
        message["sources"] = sources

    raw_followups = value.get("followups")
    explicit_followups = raw_followups if isinstance(raw_followups, list) else []
    if explicit_followups:
        # Already resolved at stream time with full context. Re-validating
        # here without the user message would drop good candidates and
        # diverge persisted history from what the user actually saw.
        followups: list[str] = []
        seen: set[str] = set()
        for item in explicit_followups[:2]:
            question = str(item or "").strip()
            key = question.casefold()
            if question and key not in seen:
                followups.append(question)
                seen.add(key)
    else:
        followups = resolve_followups(
            embedded_followups,
            message="",
            answer=content,
            sources=sources,
        )
    if followups:
        message["followups"] = followups

    for field in ("images", "response_type", "timestamp", "created_at"):
        field_value = value.get(field)
        if field_value is not None:
            message[field] = field_value
    # Structured explanatory flag only: absent/False stays out of the wire
    # contract so verified answers carry no extra payload.
    if value.get("unverified") is True:
        message["unverified"] = True
    web_auto_note = value.get("web_auto_note")
    if isinstance(web_auto_note, str) and web_auto_note.strip():
        message["web_auto_note"] = web_auto_note.strip()[:200]
    fallback_note = value.get("fallback_note")
    if isinstance(fallback_note, str) and fallback_note.strip():
        message["fallback_note"] = fallback_note.strip()[:200]
    for field in ("provider", "model"):
        field_value = value.get(field)
        if isinstance(field_value, str) and field_value:
            message[field] = field_value[:200]
    usage = value.get("usage")
    if isinstance(usage, Mapping):
        public_usage: dict[str, int] = {}
        for field in ("prompt_tokens", "completion_tokens"):
            try:
                raw = usage.get(field)
                if raw is not None:
                    public_usage[field] = max(0, int(raw))
            except (TypeError, ValueError):
                continue
        if public_usage:
            message["usage"] = public_usage
    return message
