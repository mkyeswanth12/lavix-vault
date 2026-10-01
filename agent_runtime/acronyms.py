"""Deterministic short-form / acronym resolution for Web-RAG queries.

Isolated dictionary + context-aware resolver. Never imported by search
logic directly — callers are the rewrite pre-LLM hint, the post-LLM
repair, and the entity-extract fallback in ``cuga_adapter``.

Design (per requirements):
- Small high-confidence seed map (NOT a huge hard-coded list). New
  mappings are added as data (one ``ShortForm`` entry) without touching
  the resolver logic.
- Country / country-code entries live in the same structure.
- Ambiguous codes expand only on positive context quorum; otherwise the
  original term is preserved untouched.
- The original short form is always preserved alongside the expansion
  (``Expansion (CODE)``) so SearXNG sees both.
- No LLM calls here. The existing rewrite LLM call consumes the hints;
  deterministic repair covers 3B-model misses.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_SHORT_FORM_RE = re.compile(r"\b([A-Z]{2,})\b")

# Chat filler that matches the ALL-CAPS shape but is never an entity.
# Same convention as the entity-extract acronym anchor pass.
_SKIP_TOKENS = frozenset({"OK"})

# "What does X mean / stand for" justifies showing the default expansion:
# the user is asking about the term itself.
_DEFINITION_TRIGGERS = frozenset(
    {
        "mean",
        "means",
        "meaning",
        "stand",
        "stands",
        "full",
        "form",
        "abbreviation",
        "abbreviations",
        "acronym",
        "acronyms",
        "expand",
        "standsfor",
    }
)


@dataclass(frozen=True)
class ShortForm:
    """One resolvable short form.

    ``default`` is the expansion applied when quorum passes.
    ``topics`` are whole-word selectors that count as context quorum.
    ``require_quorum``: True for ambiguous codes (expand ONLY on positive
    quorum); False for high-confidence codes (expand unless a conflicting
    alternate's topics are present).
    ``alternates`` lists other known meanings (extension hook — never
    auto-applied; their topics act as conflict signals).
    ``org_quorum``: an adjacent proper-noun organization ("CEO of
    Microsoft") also counts as quorum.
    """

    default: str
    topics: frozenset = frozenset()
    require_quorum: bool = False
    alternates: tuple = ()
    org_quorum: bool = False


def _topics(*words: str) -> frozenset:
    return frozenset(word.casefold() for word in words)


SHORT_FORMS: dict[str, ShortForm] = {
    # --- Requested seed terms ---
    "UPI": ShortForm(
        default="Unified Payments Interface",
        topics=_topics("payment", "payments", "india", "bank", "transfer", "phonepe", "gpay"),
    ),
    "CEC": ShortForm(
        default="Chief Election Commissioner",
        topics=_topics("election", "elections", "india", "eci", "commission", "commissioner", "voter", "poll", "polls"),
    ),
    "ECI": ShortForm(
        default="Election Commission of India",
        topics=_topics("election", "elections", "india", "cec", "commission", "voter", "poll", "polls"),
    ),
    "RBI": ShortForm(
        default="Reserve Bank of India",
        topics=_topics("bank", "banks", "repo", "rate", "rates", "india", "monetary", "inflation"),
    ),
    "CEO": ShortForm(
        default="Chief Executive Officer",
        topics=_topics("company", "companies", "corporate", "business", "officer", "executive", "startup", "firm", "board"),
        require_quorum=True,
        org_quorum=True,
    ),
    # --- High-confidence country / country-code entries ---
    "AUS": ShortForm(
        default="Australia",
        topics=_topics("cricket", "sydney", "melbourne", "ashes", "match", "australia"),
    ),
    "IND": ShortForm(
        default="India",
        topics=_topics("cricket", "india", "delhi", "mumbai", "gdp", "bcci"),
    ),
    "USA": ShortForm(
        default="United States",
        topics=_topics("america", "washington", "states", "dollar"),
    ),
    "UK": ShortForm(
        default="United Kingdom",
        topics=_topics("britain", "london", "parliament", "kingdom"),
    ),
    "UAE": ShortForm(
        default="United Arab Emirates",
        topics=_topics("dubai", "abu", "dhabi", "emirates", "gulf"),
    ),
}

# Tokens that must never be treated as resolvable short forms even if a
# future map entry collides with chat filler.
_RESERVED_BARE = frozenset({"OK"})


@dataclass
class ShortFormResolution:
    """Outcome of resolving one query string."""

    detected: list[str] = field(default_factory=list)
    expansions: dict[str, str] = field(default_factory=dict)
    resolved_query: str = ""


def _whole_words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z]+", str(text or "").casefold()))


def _map_words(known_entities: Any) -> set[str]:
    words: set[str] = set()
    if isinstance(known_entities, Mapping):
        for key, value in known_entities.items():
            words.update(_whole_words(key))
            words.update(_whole_words(value))
    elif isinstance(known_entities, (list, tuple, set)):
        for item in known_entities:
            words.update(_whole_words(item))
    return words


_LEAD_VERBS = frozenset(
    {
        "tell",
        "show",
        "give",
        "find",
        "what",
        "who",
        "when",
        "where",
        "which",
        "how",
        "explain",
        "describe",
        "list",
    }
)


def _has_org_context(text: str) -> bool:
    """True when a Title-case proper noun sits past the first token."""
    body = str(text or "")
    for match in re.finditer(r"\b([A-Z][a-z]{2,})\b", body):
        if match.start() == 0:
            continue
        prefix = body[: match.start()].strip()
        if not prefix:
            continue
        if match.group(1).casefold() in _LEAD_VERBS:
            continue
        return True
    return False


def resolve_short_forms(
    text: str,
    *,
    history_text: str = "",
    known_entities: Any = None,
    topic_words: Any = None,
) -> ShortFormResolution:
    """Resolve known short forms in ``text`` with context quorum.

    Never raises on odd input; unknown codes pass through untouched.
    Already-expanded codes (expansion words present in the text itself)
    are left alone (idempotent).
    """
    original = " ".join(str(text or "").split())
    result = ShortFormResolution(detected=[], expansions={}, resolved_query=original)
    if not original:
        return result
    context_words = _whole_words(original) | _whole_words(history_text) | _map_words(known_entities)
    if topic_words:
        if isinstance(topic_words, Mapping):
            context_words.update(_map_words(topic_words))
        elif isinstance(topic_words, (list, tuple, set)):
            for item in topic_words:
                context_words.update(_whole_words(item))
        else:
            context_words.update(_whole_words(topic_words))
    definition_asked = bool(context_words & _DEFINITION_TRIGGERS)
    text_words = _whole_words(original)
    org_context = _has_org_context(original)

    for match in _SHORT_FORM_RE.finditer(original):
        code = match.group(1)
        if code in _RESERVED_BARE or code in _SKIP_TOKENS:
            continue
        entry = SHORT_FORMS.get(code)
        if entry is None:
            continue
        if code in result.detected:
            continue
        result.detected.append(code)
        expansion = entry.default
        # Idempotent: expansion already present next to the code.
        if _whole_words(expansion) <= text_words:
            continue
        quorum = False
        if definition_asked:
            quorum = True
        elif context_words & set(entry.topics):
            quorum = True
        elif _whole_words(expansion) & (context_words - {code.casefold()}):
            quorum = True
        elif entry.org_quorum and org_context:
            quorum = True
        elif not entry.require_quorum:
            quorum = True
        # Conflicting alternate topics withdraw a default expansion.
        if quorum:
            for alternate in entry.alternates:
                if _whole_words(alternate) & context_words:
                    quorum = False
                    break
        if not quorum:
            continue
        result.expansions[code] = expansion
    if result.expansions:
        resolved = original
        for code, expansion in result.expansions.items():
            resolved = re.sub(
                r"\b" + re.escape(code) + r"\b",
                f"{expansion} ({code})",
                resolved,
            )
        result.resolved_query = " ".join(resolved.split())
    return result


def acronym_hint_block(resolution: ShortFormResolution) -> str:
    """Deterministic hint lines for the rewrite LLM user block."""
    lines = []
    for code in resolution.detected:
        entry = SHORT_FORMS.get(code)
        if entry is None:
            continue
        applied = resolution.expansions.get(code)
        if applied is not None:
            lines.append(f"- {code} -> {applied} (supported by context; prefer this expansion)")
        elif entry.alternates:
            lines.append(
                f"- {code} is ambiguous ({entry.default} vs "
                + ", ".join(entry.alternates)
                + "); expand only if the context clearly supports one, else keep the original"
            )
        else:
            lines.append(
                f"- {code} may mean {entry.default}; expand only if the context supports it, else keep the original"
            )
    return "\n".join(lines)
