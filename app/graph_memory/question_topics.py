"""Deterministic topic extraction from the user's own questions.

Phase 5 (flag-gated, OFF by default): question topics become (:Topic)
nodes for recall boosting and follow-up help as synthesis context only.
Stored text is data, never instructions: topics never enter R1, the
planner, or web queries (they only ride the recall output channel).

Filtering is stricter than memory validation: any secret, health,
financial, sensitive-trait, or other-person signal drops every topic for
that question. Nothing imported here weakens validation rules.
"""

from __future__ import annotations

import re

from app.security.redact import contains_credentials

from .validation import (
    _FINANCIAL_DATA,
    _HEALTH_DATA,
    _SENSITIVE_PERSONAL_TRAIT,
    _THIRD_PARTY_REFERENCE,
)

_TOPIC_QUOTED = re.compile(r'"([^"]{2,100})"')
_TOPIC_CAPS_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_TOPIC_ACRONYM = re.compile(r"\b([A-Z]{2,})\b")
_TOPIC_TECH_WORD = re.compile(r"\b([a-z][a-z0-9\-]{4,39})\b")

_TECH_STOPS = frozenset(
    {
        "what", "whats", "which", "when", "where", "there", "their", "these",
        "those", "about", "would", "could", "should", "does", "with", "from",
        "that", "this", "have", "under", "while", "after", "before",
        "pushed", "called", "known", "asked",
    }
)

# Other people: a capitalized name with a biographical predicate is someone
# else's fact ("does Alexei work…"). Missing a topic only weakens the
# recall boost; storing one would leak third-party data. Drop everything.
_BIO_PREDICATE = re.compile(
    r"\b(?:work|works|live|lives|born|earn|earns|married|divorced|diagnosed)\b",
    re.IGNORECASE,
)
_SINGLE_NAME = re.compile(r"\b([A-Z][a-z]{2,})\b")

# Health cues beyond the validation patterns (which target stored facts,
# not question phrasing like "back pain").
_HEALTH_CUE = re.compile(
    r"\b(?:pain|ache|aching|hurt|hurts|fever|cough|injury|injured|disease|"
    r"illness|doctor|hospital|surgery|symptom|treatment|pill|dose)\b",
    re.IGNORECASE,
)

_MAX_TOPICS = 5
_MAX_TOPIC_CHARS = 100


def extract_question_topics(text: str) -> tuple[str, ...]:
    """Return up to 5 deterministic topics, or () when excluded."""
    cleaned = " ".join(str(text or "").split()).strip()
    if len(cleaned) < 8:
        return ()
    if contains_credentials(cleaned):
        return ()
    lowered = cleaned.casefold()
    if (
        _HEALTH_DATA.search(cleaned) is not None
        or _FINANCIAL_DATA.search(cleaned) is not None
        or _SENSITIVE_PERSONAL_TRAIT.search(cleaned) is not None
        or _THIRD_PARTY_REFERENCE.search(cleaned) is not None
    ):
        return ()
    # Named people are someone else's facts unless first-person anchored.
    if _BIO_PREDICATE.search(cleaned) is not None and _SINGLE_NAME.search(cleaned) is not None:
        return ()
    if _HEALTH_CUE.search(cleaned) is not None:
        return ()
    topics: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        name = " ".join(str(value or "").split()).strip().rstrip(".,!;:?")[:_MAX_TOPIC_CHARS].strip()
        if len(name) < 2:
            return
        key = name.casefold()
        if key in seen or key in _TECH_STOPS:
            return
        seen.add(key)
        topics.append(name)

    for match in _TOPIC_QUOTED.finditer(cleaned):
        _add(match.group(1))
    for match in _TOPIC_CAPS_PHRASE.finditer(cleaned):
        _add(match.group(1))
    for match in _TOPIC_ACRONYM.finditer(cleaned):
        if match.group(1) != "OK":
            _add(match.group(1))
    if not topics:
        # Lowercase tech words are a fallback only: precise signals above
        # win outright, so generic verbs never dilute the topic set.
        for match in _TOPIC_TECH_WORD.finditer(lowered):
            _add(match.group(1))
    return tuple(topics[:_MAX_TOPICS])


__all__ = ["extract_question_topics"]
