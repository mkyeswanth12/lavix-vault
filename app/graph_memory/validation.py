"""Deterministic safety checks around model-produced memory candidates."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import MemoryCandidate, MemoryStatus

AUTO_ACTIVATE_CONFIDENCE = 0.88
MIN_REVIEW_CONFIDENCE = 0.55

_QUESTION_PREFIX = re.compile(
    r"^(?:who|what|where|when|why|how|can|could|would|should|do|does|did|is|are|am)\b",
    re.IGNORECASE,
)
_UNCERTAIN = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|i\s+think|i\s+guess|not\s+sure|might|could\s+be)\b",
    re.IGNORECASE,
)
_SELF_REFERENCE = re.compile(
    r"\b(?:i|i'm|i've|me|my|mine|myself)\b",
    re.IGNORECASE,
)
_SELF_SUBJECTS = frozenset({"i", "me", "myself", "user", "the user"})
_IMPERATIVE_LEAD = re.compile(
    r"^(?:ask|tell|be|call|explain|show|give|make|help|remind|try|use|"
    r"write|create|generate|summarize|speak|talk|act|pretend|do not|don't|please)\b",
    re.IGNORECASE,
)
_THIRD_PARTY_REFERENCE = re.compile(
    r"\bmy\s+(?:friend|colleague|coworker|co-worker|boss|manager|employee|client|"
    r"customer|neighbou?r|doctor|teacher|student|brother|sister|mother|father|"
    r"wife|husband|spouse|partner|son|daughter|child|relative|roommate)\b",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT = re.compile(
    r"\b(?:password|passcode|pin|api[ _-]?(?:key|token)|access[ _-]?token|"
    r"auth[ _-]?token|refresh[ _-]?token|token|"
    r"secret|private[ _-]?key|bearer|passphrase)\b\s*(?:is|=|:)?\s*[A-Za-z0-9_./+@=-]{4,}",
    re.IGNORECASE,
)
# Valueless labels: no secret value needs to be present. Bare words only
# match as the whole message or before a colon (prose like "the secret
# to good dosa" stays storable); password-for and ssh/git credential
# lines match anywhere because they never carry a storable fact.
_SECRET_LABEL = re.compile(
    r"(?:^|[\s:;(\[])(?:password|passcode)(?:\s+for\b[^\n]{0,120})?\s*:?\s*$"
    r"|(?:^|[\s:;(\[])(?:api[ _-]?(?:key|token)|private[ _-]?key|bearer|access[ _-]?token|"
    r"auth[ _-]?token|refresh[ _-]?token|secret|passphrase|pin)\s*:"
    r"|\bssh(?:[ _-]?(?:key|fingerprint|passphrase))?\b"
    r"|\bgit[ _-]?credential\b"
    r"|^(?:token|secret|api[ _-]?(?:key|token))\s*$",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_GOVERNMENT_ID = re.compile(
    r"\b(?:\d{3}-\d{2}-\d{4}|\d{4}[ -]?\d{4}[ -]?\d{4})\b"
)
_LONG_DIGIT_SEQUENCE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\s().-]?){10,15}(?!\w)")
_HEALTH_DATA = re.compile(
    r"\b(?:cancer|diabet(?:es|ic)|hiv|aids|hepatitis|epilepsy|asthma|autis(?:m|tic)|"
    r"depression|anxiety disorder|bipolar|schizophrenia|ptsd|pregnan(?:t|cy)|"
    r"disab(?:ility|led)|diagnos(?:is|ed|e)|medical (?:condition|history|record)|"
    r"health (?:condition|status|record)|mental health|blood type|allerg(?:y|ies|ic)|"
    r"prescription|medication|therapy|therapist|pain|back pain|symptom|symptoms|"
    r"illness|ill|injury|injured|disease|surgery|doctor|hospital|treatment|"
    r"pill|dose|dosage|fever|cough|sleep (?:problem|trouble|apnea|insomnia)|"
    r"trouble sleeping|difficulty sleeping|sleeping (?:problem|trouble|difficulty)|"
    r"(?:can't|cannot|unable to|no|lack of) sleep|sleep depriva(?:tion|ted)|"
    r"insomnia|ache|aching|hurt|hurts|sick)\b",
    re.IGNORECASE,
)
_FINANCIAL_DATA = re.compile(
    r"\b(?:salary|income|net worth|bank balance|account balance|bank account|"
    r"accounts?|credit score|"
    r"loan balance|mortgage balance|tax return|financial assets?|wealth|wealthy|debt)\b|"
    r"\b(?:i|user)\s+(?:earn|earns|make|makes|take home)\s+"
    r"(?:[$€£¥₹]|\d)|(?:[$€£¥₹]\s*\d)|"
    r"\b\d[\d,.]*\s*(?:rupees?|dollars?|euros?|pounds?|yen)\b",
    re.IGNORECASE,
)
_SENSITIVE_PERSONAL_TRAIT = re.compile(
    r"\b(?:sexual orientation|gender identity|religion|religious beliefs?|caste|"
    r"ethnicity|racial identity|political affiliation|political party membership|"
    r"trade union membership|criminal record|biometric data)\b|"
    r"\b(?:i am|i'm|user is)\s+(?:gay|lesbian|bisexual|transgender|hindu|muslim|"
    r"christian|sikh|jain|buddhist)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    accepted: bool
    status: MemoryStatus | None
    reason: str | None
    fingerprint: str | None
    identity_key: tuple[str, str, str] | None = None


@dataclass(frozen=True, slots=True)
class IdentityFact:
    """Deterministic identity fact parsed from the source text itself.

    kind/predicate are canonical (not model-chosen) so corrections match
    by key; value is the captured span. Returned only when the text is an
    explicit user statement free of secrets, uncertainty, and third-party
    or health/financial content.
    """

    kind: str
    predicate: str
    value: str


_IDENTITY_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("fact", "name", re.compile(r"\bmy name is (?:now )?([A-Za-z][\w\-']{0,39})\b", re.IGNORECASE)),
    ("fact", "name", re.compile(r"\bmy name's ([A-Za-z][\w\-']{0,39})\b", re.IGNORECASE)),
    ("fact", "name", re.compile(r"\bcall me ([A-Za-z][\w\-']{0,39})\b", re.IGNORECASE)),
    ("fact", "language", re.compile(r"\bmy (?:preferred )?language is ([A-Za-z ]{2,30})\b", re.IGNORECASE)),
    ("fact", "language", re.compile(r"\bspeak to me in ([A-Za-z ]{2,30})\b", re.IGNORECASE)),
    ("fact", "units", re.compile(r"\bi prefer (celsius|fahrenheit|metric|imperial)(?: units)?\b", re.IGNORECASE)),
    ("fact", "timezone", re.compile(r"\bmy timezone is ([A-Za-z_\/+-]{2,40})\b", re.IGNORECASE)),
)
_IDENTITY_DENY = re.compile(
    r"\b(?:salary|income|wealth|health|diagnos|disease|syndrome|disorder|"
    r"allergy|pregnan|friend|brother|sister|mother|father|wife|husband)\b",
    re.IGNORECASE,
)
_IDENTITY_VALUE_SHAPE = re.compile(r"^[A-Za-z][\w\-' ]{0,63}$")


def match_identity_fact(text: str) -> IdentityFact | None:
    """Match explicit identity facts without any model call.

    Returns None for anything uncertain, sensitive, third-party, or
    health/financial — those keep the model-confidence bands. Values are
    shape-checked; the caller still runs secret validation on the value.
    """
    raw = str(text or "").strip()
    if not raw or _UNCERTAIN.search(raw):
        return None
    if _IDENTITY_DENY.search(raw):
        return None
    for kind, predicate, pattern in _IDENTITY_PATTERNS:
        match = pattern.search(raw)
        if match is None:
            continue
        value = " ".join(str(match.group(1) or "").split())
        if not value or not _IDENTITY_VALUE_SHAPE.match(value):
            continue
        if _contains_sensitive_data(value) or _contains_sensitive_data(raw):
            return None
        return IdentityFact(kind=kind, predicate=predicate, value=value)
    return None


def normalize_component(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def candidate_fingerprint(candidate: MemoryCandidate) -> str:
    kind = candidate.kind.value if hasattr(candidate.kind, "value") else str(candidate.kind)
    canonical = "\x1f".join(
        (
            kind,
            normalize_component(candidate.subject),
            normalize_component(candidate.subject),
            normalize_component(candidate.predicate),
            normalize_component(candidate.object_value),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _contains_sensitive_data(value: str) -> bool:
    return any(
        pattern.search(value)
        for pattern in (
            _SECRET_ASSIGNMENT,
            _SECRET_LABEL,
            _EMAIL,
            _GOVERNMENT_ID,
            _LONG_DIGIT_SEQUENCE,
            _PHONE,
            _HEALTH_DATA,
            _FINANCIAL_DATA,
            _SENSITIVE_PERSONAL_TRAIT,
        )
    )


def _normalize_text(value: str) -> str:
    """Fold case/whitespace/punctuation for the second screening pass."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _contains_sensitive_data_any(
    subject: str, predicate: str, object_value: str, excerpt: str
) -> bool:
    """Screen every view: excerpt, each field, joined triple, normalized join.

    Split-evasion (value only in the excerpt, label only in a field) fails
    closed: any single view tripping rejects the candidate.
    """
    joined = " ".join((subject, predicate, object_value))
    views = (
        excerpt,
        subject,
        predicate,
        object_value,
        joined,
        _normalize_text(joined),
        _normalize_text(excerpt),
    )
    return any(_contains_sensitive_data(view) for view in views)


def validate_candidate(candidate: MemoryCandidate) -> CandidateDecision:
    """Accept only grounded, personal, non-sensitive user assertions.

    This validator is intentionally independent from prompts. A model marking a
    candidate as safe cannot bypass deterministic source and content checks.
    """

    combined = " ".join(
        (candidate.subject, candidate.predicate, candidate.object_value, candidate.source_excerpt)
    )
    excerpt = candidate.source_excerpt.strip()
    if candidate.contains_sensitive_data or _contains_sensitive_data_any(
        candidate.subject, candidate.predicate, candidate.object_value, combined
    ):
        return CandidateDecision(False, None, "sensitive_data", None)
    if _QUESTION_PREFIX.search(excerpt):
        return CandidateDecision(False, None, "question", None)
    if _UNCERTAIN.search(excerpt):
        return CandidateDecision(False, None, "uncertain_statement", None)
    # Personal authority must be visible in the exact user excerpt. Model-made
    # subject fields and safety booleans are untrusted hints, not the boundary.
    # Deterministic identity facts ("my name is mk", "call me Sam") are
    # recognized from the excerpt itself and take precedence over the model's
    # explicit_user_assertion boolean (small models mis-set it): when the
    # model's candidate agrees on the value, instruction-shape text still
    # counts as a declaration — the service canonicalizes and auto-activates
    # these; everything else keeps the model-confidence bands.
    identity_key: tuple[str, str, str] | None = None
    hit = match_identity_fact(excerpt)
    agreed = hit is not None and normalize_component(
        candidate.object_value
    ) == normalize_component(hit.value)
    if agreed:
        identity_key = (hit.kind, "i", hit.predicate)
    if identity_key is None and not candidate.explicit_user_assertion:
        return CandidateDecision(False, None, "not_explicit_user_assertion", None)
    if identity_key is None and _IMPERATIVE_LEAD.search(excerpt):
        # Instructions are not facts about the user: an imperative lead
        # ("ask me …", "be concise") asserting an "I <verb>" shape is a
        # request to the assistant, never a memory. Names inside such
        # commands are captured by the deterministic identity path
        # instead; this gate only drops the model-shaped fact.
        return CandidateDecision(False, None, "instruction_not_fact", None)
    if not _SELF_REFERENCE.search(excerpt):
        return CandidateDecision(False, None, "not_personal", None)
    if normalize_component(candidate.subject) not in _SELF_SUBJECTS:
        return CandidateDecision(False, None, "third_party_assertion", None)
    if _THIRD_PARTY_REFERENCE.search(excerpt):
        return CandidateDecision(False, None, "third_party_assertion", None)
    if candidate.confidence < MIN_REVIEW_CONFIDENCE:
        return CandidateDecision(False, None, "confidence_too_low", None)

    status = (
        MemoryStatus.ACTIVE
        if candidate.confidence >= AUTO_ACTIVATE_CONFIDENCE
        else MemoryStatus.PENDING
    )
    return CandidateDecision(
        True, status, None, candidate_fingerprint(candidate), identity_key
    )
