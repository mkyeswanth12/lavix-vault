"""Relationship-memory lifecycle service shared by HTTP and worker adapters."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID

from .models import (
    DEFAULT_RETENTION_DAYS,
    MAX_RECALL_RESULTS,
    AboutMeProfile,
    ClearMemoryResponse,
    GraphMemoryItem,
    GraphMemoryItemPage,
    GraphMemoryStatusResponse,
    MemoryCandidate,
    MemoryKind,
    MemoryStatus,
    RecallRecord,
)
from .neo4j_projection import GRAPH_RECALL_MAX_FACTS, Neo4jProjection, Neo4jUnavailable
from .repository import (
    GraphMemoryNotFoundError,
    GraphMemoryValidationError,
    ItemMutation,
    PostgresGraphMemoryRepository,
    TenantRecord,
)
from .validation import (
    candidate_fingerprint,
    match_identity_fact,
    normalize_component,
    validate_candidate,
)

logger = logging.getLogger(__name__)


def question_memory_enabled() -> bool:
    """Phase 5 flag: OFF by default, isolated-stack only until rollout."""
    from app.config import settings

    return bool(settings.graph_question_memory_enabled)


def default_memory_embedder(text: str) -> list[float] | None:
    """Sync embedding for one memory text; None fail-open.

    Small standalone client (not the ingestion priority client): memory
    volume is tiny and every caller already fails open to keyword recall.
    Uses the configured embedding model with the document-kind preset, so
    memory vectors stay comparable with chunk vectors from the same model.
    """
    from app.config import settings
    from app.ingestion.embedding_presets import prepare_texts

    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return None
    try:
        import httpx

        shaped = prepare_texts(settings.embedding_model_name, [cleaned], kind="document")[0]
        response = httpx.post(
            settings.embedding_api_url,
            json={"model": settings.embedding_model_name, "input": shaped},
            timeout=30,
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        values = [float(value) for value in vector]
    except Exception:
        logger.warning("memory embedding failed open", exc_info=True)
        return None
    if len(values) != settings.embedding_dimension:
        logger.warning("memory embedding dimension mismatch: %s", len(values))
        return None
    return values


_EMPTY_ABOUT_ME_SUMMARY = "No relationship memories yet."
_MAX_PROFILE_ITEMS = 12
_MAX_PROFILE_WORDS = 70
_MAX_PROFILE_SENTENCES = 4

#: Predicates that carry the person's own verb for setup/tooling talk.
#: Rendered with the stored verb intact ("use Linux", never "work with
#: Linux"). Noun predicates (tools/tool/tech_stack/stack) have no verb
#: and keep the neutral "work with" framing.
_SETUP_VERBS = {
    "use": ("use", ""),
    "uses": ("use", ""),
    "using": ("use", ""),
    "work": ("work", "with "),
    "works": ("work", "with "),
    "working": ("work", "with "),
    "work_with": ("work", "with "),
    "works_with": ("work", "with "),
    "build": ("build", ""),
    "builds": ("build", ""),
    "tools": ("work", "with "),
    "tool": ("work", "with "),
    "tech_stack": ("work", "with "),
    "stack": ("work", "with "),
}

#: Lifestyle predicates mapped to the verb stem rendered in prose
#: ("love dogs", never rephrased). Hobby rows have no verb of their own
#: and take "enjoy".
_LIFESTYLE_VERBS = {
    "enjoy": "enjoy",
    "enjoys": "enjoy",
    "like": "like",
    "likes": "like",
    "love": "love",
    "loves": "love",
    "hobby": "enjoy",
    "hobbies": "enjoy",
}

#: Predicates whose verb is kept verbatim in the tastes sentence.
_MAKE_VERBS = {
    "make": "make",
    "makes": "make",
    "cook": "cook",
    "cooks": "cook",
    "bake": "bake",
    "bakes": "bake",
}

#: Objects naming hardware condense to one short clause instead of
#: listing every spec.
_HARDWARE_RE = re.compile(
    r"\d\s*(?:GB|TB|MB|GHz|MHz)\b|(?:ram|ssd|cpu)\b", re.IGNORECASE
)
_HARDWARE_SPEC_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:GB|TB|MB|GHz|MHz)\b", re.IGNORECASE
)
_TIME_RANGE_RE = re.compile(
    r"\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\s*(?:to|-)\s*\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?"
)


def _hardware_clause(obj: str) -> str | None:
    """Short clause for a multi-spec hardware object, else None.

    Only objects that actually enumerate specs (commas, "including",
    several quantities) condense — a single "32gb of ram" or "PC with
    32gb ram" stays a plain possession, and spec-less parts keep their
    stored verb. Returns e.g. "on a 32 GB machine".
    """
    if _HARDWARE_SPEC_RE.search(obj) is None:
        return None
    if "," not in obj and " including " not in obj.casefold():
        return None
    match = _HARDWARE_SPEC_RE.search(obj)
    assert match is not None
    return f"on a {match.group(0).strip()} machine"


def _is_schedule(predicate: str, obj: str) -> bool:
    """A row stating working hours (predicate names work/schedule/hours)."""
    lowered = predicate.casefold()
    if "work" not in lowered and "schedul" not in lowered and "hour" not in lowered and "shift" not in lowered:
        return False
    return _TIME_RANGE_RE.search(obj) is not None


#: Pet nouns (beyond _PET_NOUNS) for recognizing specific-pet content in
#: past-tense and possession rows.
_PET_WORDS = frozenset(
    {
        "pet", "pets", "dog", "dogs", "cat", "cats", "animal", "animals",
        "puppy", "puppies", "kitten", "kittens", "retriever", "labrador",
        "shepherd", "poodle", "beagle", "bulldog", "persian", "siamese",
        "parrot", "parakeet", "hamster", "rabbit", "breed",
    }
)

#: Sentence leads that vary structure without adding claims. Every word
#: is glue or too short to check, so validator-safe by construction.
_VARIETY_LEADS = ("Outside of that, ", "In your time off, ", "Day to day, ")
_YOU_STARTS = ("you", "your", "you're", "you've")

_NAME_PREDICATES = frozenset({"name", "full_name", "first_name", "last_name", "called"})
_FROM_PREDICATES = frozenset({"from", "hometown", "birthplace", "born_in"})
_LOCATION_PREDICATES = frozenset({"live_in", "lives_in", "located_in", "based_in", "city", "country"})
_PET_NAME_PREDICATES = frozenset(
    {"has_pet", "owns_pet", "own_pet", "pet", "has_dog", "owns_dog", "has_cat", "owns_cat"}
)
_PET_NOUNS = frozenset({"pet", "dog", "cat", "animal", "puppy", "kitten"})
_BREED_PREDICATES = frozenset(
    {"pet_breed", "dog_breed", "cat_breed", "breed", "pet_type", "pet_species", "species"}
)
_PREFERENCE_VERBS = (
    ("prefers", "prefer"),
    ("prefer", "prefer"),
    ("likes", "like"),
    ("like", "like"),
    ("loves", "love"),
    ("love", "love"),
    ("enjoys", "enjoy"),
    ("enjoy", "enjoy"),
    ("dislikes", "dislike"),
    ("dislike", "dislike"),
    ("hates", "hate"),
    ("hate", "hate"),
)
_YOU_CONJUGATION = {
    "is": "are",
    "was": "were",
    "has": "have",
    "does": "do",
    "goes": "go",
    "asks": "ask",
    "works": "work",
    "plays": "play",
    "watches": "watch",
    "reads": "read",
    "writes": "write",
    "cooks": "cook",
    "manages": "manage",
    "uses": "use",
    "builds": "build",
    "breaks": "break",
}

_ROLE_PREDICATES = frozenset({"occupation", "job", "profession", "role", "title"})
_STACK_PREDICATES = frozenset(
    {
        "use",
        "uses",
        "using",
        "work",
        "works",
        "working",
        "work_with",
        "works_with",
        "build",
        "builds",
        "tools",
        "tool",
        "tech_stack",
        "stack",
    }
)
_INTEREST_PREDICATES = frozenset(
    {
        "interested",
        "interested_in",
        "learning",
        "learn",
        "learns",
        "exploring",
        "explore",
        "explores",
        "studying",
        "study",
        "studies",
        "curious",
        "curious_about",
    }
)
_LIFESTYLE_PREDICATES = frozenset(
    {"hobby", "hobbies", "enjoy", "enjoys", "like", "likes", "love", "loves"}
)


def _clean_object(value: object) -> str:
    return str(value or "").strip().rstrip(".")


def _render_name(obj: str) -> str:
    return f"Your name is {obj}."


def _render_favorite(predicate: str, obj: str) -> str | None:
    if predicate == "favorite":
        return f"Your favorite is {obj}."
    if predicate.startswith("favorite_"):
        topic = predicate[len("favorite_") :].replace("_", " ").strip()
        if topic:
            return f"Your favorite {topic} is {obj}."
    return None


def _render_preference(predicate: str, obj: str) -> str | None:
    for form, verb in _PREFERENCE_VERBS:
        if predicate == form:
            return f"You {verb} {obj}."
    return None


def _render_location(predicate: str, obj: str) -> str | None:
    if predicate in _FROM_PREDICATES:
        return f"You are from {obj}."
    if predicate in _LOCATION_PREDICATES:
        return f"You live in {obj}."
    return None


_ARTICLE_LEADERS = frozenset(
    {"a", "an", "the", "my", "your", "his", "her", "its", "our", "their",
     "this", "that", "these", "those", "some", "any", "no", "each", "every"}
)


def _with_indefinite_article(obj: str) -> str:
    """Prepend a/an to a bare countable noun phrase; pass through otherwise."""
    if not obj or "," in obj:
        return obj
    tokens = obj.split()
    if len(tokens) == 1:
        return obj
    first = tokens[0].lower()
    if first in _ARTICLE_LEADERS or not first[:1].isalpha():
        return obj
    article = "an" if first[:1] in "aeiou" else "a"
    return f"{article} {obj}"


def _render_possession(predicate: str, obj: str) -> str | None:
    """Render has_X / owns_X rows. Name-like objects get a named sentence."""
    noun = ""
    if predicate.startswith(("has_", "owns_", "own_")):
        noun = predicate.split("_", 1)[1].replace("_", " ").strip()
    elif predicate in ("has", "have", "owns", "own"):
        return f"You have {_with_indefinite_article(obj)}."
    if not noun:
        return None
    if "," in obj or len(obj.split()) > 4:
        return f"You have {obj}."
    article = "an" if noun[:1].lower() in "aeiou" else "a"
    return f"You have {article} {noun} named {obj}."


def _render_generic(predicate: str, obj: str) -> str:
    words = predicate.replace("_", " ").split()
    verb = _YOU_CONJUGATION.get(words[0], words[0]) if words else "noted"
    tail = " ".join([verb, *words[1:], obj]).strip()
    return f"You {tail}."


def _render_breed(obj: str) -> str:
    if "," in obj or len(obj.split()) > 4:
        return f"Your pet is {obj}."
    article = "an" if obj[:1].lower() in "aeiou" else "a"
    return f"Your pet is {article} {obj}."


def _render_single(predicate: str, obj: str) -> str:
    for renderer in (
        lambda: _render_favorite(predicate, obj),
        lambda: _render_preference(predicate, obj),
        lambda: _render_location(predicate, obj),
        lambda: _render_possession(predicate, obj),
    ):
        rendered = renderer()
        if rendered is not None:
            return rendered
    if predicate in _BREED_PREDICATES or "breed" in predicate:
        return _render_breed(obj)
    if predicate in _NAME_PREDICATES:
        return _render_name(obj)
    if predicate in _ROLE_PREDICATES:
        article = "an" if obj[:1].lower() in "aeiou" else "a"
        return f"You're {article} {obj}."
    return _render_generic(predicate, obj)


def _split_list_object(obj: str) -> list[str]:
    """Split a stored list ("a, b, and c") without breaking inner phrases."""
    items: list[str] = []
    for chunk in obj.split(","):
        chunk = chunk.strip()
        if chunk.lower().startswith("and "):
            chunk = chunk[4:].strip()
        if chunk:
            items.append(chunk)
    return items


def _oxford_join(items: Sequence[str]) -> str:
    items = [part for part in (str(item).strip() for item in items) if part]
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _role_article(role: str) -> str:
    return "an" if role[:1].lower() in "aeiou" else "a"


def _pet_phrase(predicate: str, obj: str) -> str:
    """Noun phrase for an unmerged pet row ("a pet named Bheem")."""
    sentence = _render_possession(predicate, obj)
    if sentence is not None and sentence.startswith("You have ") and sentence.endswith("."):
        return sentence[len("You have ") : -1]
    return _with_indefinite_article(obj)


def _breed_phrase(obj: str) -> str:
    # Clauses ("Tommy is a pug") and lists pass through; only bare breed
    # nouns take an article.
    if "," in obj or " is " in obj.casefold() or len(obj.split()) > 3:
        return obj
    article = "an" if obj[:1].lower() in "aeiou" else "a"
    return f"{article} {obj}"


def _pet_name_rows(items: Sequence[GraphMemoryItem]) -> list[GraphMemoryItem]:
    rows = []
    for item in items:
        predicate = item.predicate.strip().lower()
        if predicate in _PET_NAME_PREDICATES:
            rows.append(item)
            continue
        if predicate.startswith(("has_", "owns_")):
            noun = predicate.split("_", 1)[1]
            if noun in _PET_NOUNS:
                rows.append(item)
    return rows


def _breed_descriptor(breed_obj: str, pet_name: str) -> str | None:
    """Extract the breed/type descriptor, or None when the link is uncertain."""
    text = breed_obj.strip()
    lowered = text.casefold()
    if pet_name.casefold() not in lowered:
        # No name in the breed row: merge only when nothing else is named.
        capitalized = [token.strip(".,") for token in text.split() if token[:1].isupper()]
        if capitalized:
            return None
        descriptor = text
    else:
        match = re.search(r"\bis\s+(?:an?\s+)?(.+)$", text, flags=re.IGNORECASE)
        if not match:
            return None
        descriptor = match.group(1).strip().rstrip(".")
    if not descriptor:
        return None
    if descriptor[:1].isalpha() and not descriptor.lower().startswith(("a ", "an ", "the ")):
        article = "an" if descriptor[:1].lower() in "aeiou" else "a"
        descriptor = f"{article} {descriptor}"
    return descriptor


def _render_pet_merge(pet_item: GraphMemoryItem, breed_item: GraphMemoryItem) -> str | None:
    pet_name = _clean_object(pet_item.object_value)
    breed_obj = _clean_object(breed_item.object_value)
    if not pet_name or not breed_obj:
        return None
    descriptor = _breed_descriptor(breed_obj, pet_name)
    if descriptor is None:
        return None
    return f"You have {descriptor} named {pet_name}."


def deterministic_about_me(
    items: Sequence[GraphMemoryItem],
    *,
    graph_revision: int,
) -> AboutMeProfile:
    """Produce the authoritative grounded summary from active memories."""

    if not items:
        return AboutMeProfile(
            status="empty",
            summary=_EMPTY_ABOUT_ME_SUMMARY,
        )

    selected = tuple(items[:_MAX_PROFILE_ITEMS])
    pet_rows = _pet_name_rows(selected)
    breed_rows = [
        item
        for item in selected
        if item.predicate.strip().lower() in _BREED_PREDICATES or "breed" in item.predicate.strip().lower()
    ]
    consumed: set[int] = set()
    # Merge a pet name with its breed/type only when the link is certain:
    # exactly one pet, exactly one breed row, and a clean descriptor.
    # Any ambiguity keeps them as separate grouped sentences — a wrong
    # merge that invents a relationship is worse than two plain ones.
    pet_sentence: str | None = None
    if len(pet_rows) == 1 and len(breed_rows) == 1:
        pet_item, breed_item = pet_rows[0], breed_rows[0]
        if id(pet_item) != id(breed_item):
            sentence = _render_pet_merge(pet_item, breed_item)
            if sentence is not None:
                pet_sentence = sentence
                consumed.add(id(pet_item))
                consumed.add(id(breed_item))

    def _dedup(values: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                ordered.append(value)
        return ordered

    def _note_seen(obj: str) -> bool:
        """True when this object text was already rendered (near-dup drop)."""
        key = obj.casefold()
        if key in seen_objects:
            return True
        seen_objects.add(key)
        return False

    names: list[str] = []
    origins: list[str] = []
    places: list[str] = []
    roles: list[str] = []
    setup_verbs: dict[str, list[str]] = {}
    setup_raw: list[str] = []
    schedules: list[str] = []
    schedule_weekdays: list[str] = []
    possessions: list[str] = []
    past: list[str] = []
    hardware_clauses: list[str] = []
    pet_parts: list[str] = []
    breed_parts: list[str] = []
    enjoy: list[tuple[str, str]] = []
    interested: list[str] = []
    prefer: list[str] = []
    avoid: list[str] = []
    make_groups: dict[str, list[str]] = {}
    favorites: list[str] = []
    seen_objects: set[str] = set()
    for item in selected:
        if id(item) in consumed:
            continue
        predicate = item.predicate.strip().lower()
        obj = _clean_object(item.object_value)
        if not obj:
            consumed.add(id(item))
            continue
        if predicate in _NAME_PREDICATES:
            names.append(obj)
        elif predicate in _FROM_PREDICATES:
            origins.append(obj)
        elif predicate in _LOCATION_PREDICATES:
            places.append(obj)
        elif predicate in _ROLE_PREDICATES:
            roles.append(obj)
        elif _is_schedule(predicate, obj):
            # Working-hours rows get their own sentence ("You work
            # weekdays 10:30 AM to 6 PM"), never folded into tooling.
            if "weekday" in predicate:
                schedule_weekdays.append(obj)
            elif "weekend" in predicate:
                schedule_weekdays.append(obj)
            else:
                schedules.append(obj)
            seen_objects.add(obj.casefold())
        elif predicate in _SETUP_VERBS:
            verb, _ = _SETUP_VERBS[predicate]
            for part in _split_list_object(obj):
                clause = _hardware_clause(part)
                if clause is not None:
                    if clause not in hardware_clauses:
                        hardware_clauses.append(clause)
                elif not _note_seen(part):
                    setup_verbs.setdefault(verb, []).append(part)
                    setup_raw.append(part)
        elif predicate in ("has", "have", "owns", "own"):
            clause = _hardware_clause(obj)
            if clause is not None:
                if clause not in hardware_clauses:
                    hardware_clauses.append(clause)
            elif not _note_seen(obj):
                possessions.append(_with_indefinite_article(obj))
        elif predicate == "had":
            # Past possession ("had a dog named Bheem"): render in past
            # tense — present tense would invent a living pet.
            if not _note_seen(obj):
                past.append(_with_indefinite_article(obj))
        elif predicate in ("prefer", "prefers"):
            for part in _split_list_object(obj):
                if not _note_seen(part):
                    prefer.append(part)
        elif predicate in ("dislike", "dislikes", "hate", "hates"):
            for part in _split_list_object(obj):
                if not _note_seen(part):
                    avoid.append(part)
        elif predicate in _INTEREST_PREDICATES:
            for part in _split_list_object(obj):
                if not _note_seen(part):
                    interested.append(part)
        elif predicate in _LIFESTYLE_PREDICATES:
            verb = _LIFESTYLE_VERBS.get(predicate, "enjoy")
            for part in _split_list_object(obj):
                if not _note_seen(part):
                    enjoy.append((verb, part))
        elif predicate in _MAKE_VERBS:
            for part in _split_list_object(obj):
                if not _note_seen(part):
                    make_groups.setdefault(_MAKE_VERBS[predicate], []).append(part)
        elif predicate == "favorite":
            favorites.append(f"Your favorite is {obj}.")
        elif predicate.startswith("favorite_"):
            topic = predicate[len("favorite_") :].replace("_", " ").strip()
            favorites.append(f"Your favorite {topic} is {obj}." if topic else f"You note {obj}.")
        elif predicate in _BREED_PREDICATES or "breed" in predicate:
            if not _note_seen(obj):
                breed_parts.append(_breed_phrase(obj))
        elif item in pet_rows:
            if not _note_seen(obj):
                pet_parts.append(_pet_phrase(predicate, obj))
        else:
            # Low-signal or unclassifiable rows stay visible in the
            # Relationship memory list; the portrait drops them rather
            # than reading out the database.
            continue
        consumed.add(id(item))

    # A generic pet affinity ("love dogs") is dropped when a specific pet
    # renders, and folded into one clause with it ("You love dogs and had
    # a Golden Retriever named Bheem").
    generic_pet: tuple[str, str] | None = None
    for index, (verb, candidate) in enumerate(list(enjoy)):
        lowered = candidate.casefold().strip()
        if lowered in _PET_WORDS or lowered.rstrip("s") in _PET_WORDS:
            generic_pet = (verb, candidate)
            del enjoy[index]
            break
    pet_specific: str | None = None
    breed_in_affinity = False
    if pet_sentence:
        pet_specific = pet_sentence
    elif pet_parts:
        pet_specific = f"You have {_oxford_join(_dedup(pet_parts))}."
    elif breed_parts:
        pet_specific = f"Your pet is {_oxford_join(_dedup(breed_parts))}."
        breed_in_affinity = True
    else:
        for past_obj in past:
            if any(word in past_obj.casefold() for word in _PET_WORDS):
                pet_specific = f"You had {past_obj}."
                past[:] = [entry for entry in past if entry != past_obj]
                break
    affinity_core: str | None = None
    if generic_pet is not None and pet_specific is not None:
        generic_verb, generic_obj = generic_pet
        tail = pet_specific
        if tail.startswith("You "):
            tail = tail[len("You ") :]
            tail = tail[0].lower() + tail[1:]
        affinity_core = f"{generic_verb} {generic_obj} and {tail.rstrip('.')}"
    elif pet_specific is not None:
        affinity_core = None  # rendered verbatim below (already a sentence)

    sentences: list[str] = []
    kinds: list[str] = []
    # Identity: name plus origin/place in one idea.
    if names or origins or places:
        head = f"You're {_dedup(names)[0]}" if names else "You're"
        tails: list[str] = []
        if origins:
            tails.append(f"from {_oxford_join(_dedup(origins))}")
        if places:
            tails.append(f"living in {_oxford_join(_dedup([place for place in places if place.casefold() not in {origin.casefold() for origin in origins}]))}")
        if names and not tails:
            sentences.append(head + ".")
        elif tails:
            sentences.append(head + " " + " and ".join(tails) + ".")
        kinds.append("identity")
    # How they work: stored verbs kept intact, merged with "and".
    _SETUP_PREP = {"use": "", "build": "", "work": "with "}
    work_clauses: list[str] = []
    for verb in ("use", "build", "work"):
        objs = _dedup(setup_verbs.get(verb, []))
        if not objs:
            continue
        prep = _SETUP_PREP[verb]
        work_clauses.append(f"{verb} {prep}{_oxford_join(objs)}".rstrip())
    if hardware_clauses and not roles:
        first, *rest = _dedup(hardware_clauses)
        clause = first
        for extra in rest:
            clause += " and " + extra[len("on a ") :] if extra.startswith("on a ") else " and " + extra
        if work_clauses:
            # Hardware rides the work sentence ("You use Linux on a 32 GB
            # machine") instead of listing specs on its own.
            work_clauses[-1] = f"{work_clauses[-1]} {clause}"
        else:
            work_clauses.append(f"have {clause[len('on a '):]}" if clause.startswith("on a ") else f"have {clause}")
    if roles or work_clauses:
        if roles:
            role_part = _oxford_join(
                [f"{_role_article(role)} {role}" for role in _dedup(roles)]
            )
            sentence = f"You work as {role_part}"
            setup_objs = _dedup(setup_raw)
            if hardware_clauses:
                first, *rest = _dedup(hardware_clauses)
                clause = first
                for extra in rest:
                    clause += " and " + extra[len("on a ") :] if extra.startswith("on a ") else " and " + extra
                if setup_objs:
                    sentence += f" with {_oxford_join(setup_objs)} {clause}"
                else:
                    sentence += f" {clause}"
            elif setup_objs:
                sentence += f" with {_oxford_join(setup_objs)}"
            sentences.append(sentence + ".")
        else:
            sentences.append(f"You {' and '.join(work_clauses)}.")
        kinds.append("work")
    # Schedule: working hours get their own sentence.
    schedule_all = _dedup(schedule_weekdays + schedules)
    if schedule_all:
        if schedule_weekdays:
            sentences.append(
                f"You work weekdays {_oxford_join(_dedup(schedule_weekdays))}."
                if not schedules
                else f"You work weekdays {_oxford_join(_dedup(schedule_weekdays))} and {_oxford_join(_dedup(schedules))}."
            )
        else:
            sentences.append(f"You work {_oxford_join(schedule_all)}.")
        kinds.append("schedule")
    if possessions:
        sentences.append(f"You have {_oxford_join(_dedup(possessions))}.")
        kinds.append("possessions")
    if past:
        sentences.append(f"You had {_oxford_join(_dedup(past))}.")
        kinds.append("past")
    # Close relationships: affinity or pet sentence, never folded into
    # anything else.
    if affinity_core is not None:
        if sentences:
            sentences.append(f"In your time off, you {affinity_core}.")
        else:
            sentences.append(f"You {affinity_core}.")
        kinds.append("pet")
    elif pet_specific is not None:
        sentences.append(pet_specific)
        kinds.append("pet")
    if breed_parts and not breed_in_affinity:
        sentences.append(f"Your pet is {_oxford_join(_dedup(breed_parts))}.")
        kinds.append("pet")
    # Tastes: remaining enjoy / interested / prefer / avoid / make groups
    # folded into one idea with verbs intact.
    taste_clauses: list[str] = []
    enjoy_verbs: dict[str, list[str]] = {}
    for verb, obj in enjoy:
        enjoy_verbs.setdefault(verb, []).append(obj)
    for verb, objs in enjoy_verbs.items():
        taste_clauses.append(f"{verb} {_oxford_join(_dedup(objs))}")
    if interested:
        taste_clauses.append(f"are interested in {_oxford_join(_dedup(interested))}")
    for verb, objs in make_groups.items():
        taste_clauses.append(f"{verb} {_oxford_join(_dedup(objs))}")
    if prefer:
        taste_clauses.append(f"prefer {_oxford_join(_dedup(prefer))}")
    if avoid:
        taste_clauses.append(f"avoid {_oxford_join(_dedup(avoid))}")
    if taste_clauses:
        lead = "Outside of that, you " if sentences else "You "
        sentences.append(f"{lead}{' and '.join(taste_clauses)}.")
        kinds.append("tastes")
    for favorite in _dedup(favorites):
        if len(sentences) >= _MAX_PROFILE_SENTENCES:
            break
        sentences.append(favorite)
        kinds.append("favorites")
    sentences = sentences[:_MAX_PROFILE_SENTENCES]
    kinds = kinds[:_MAX_PROFILE_SENTENCES]
    # Variety: no more than two consecutive "You" sentences. Rewrite the
    # third in a run with a claim-free lead (glue/short words only),
    # preferring a lead not already used above.
    used_leads = {
        lead for sentence in sentences for lead in _VARIETY_LEADS if sentence.startswith(lead)
    }
    fixed: list[str] = []
    fixed_kinds: list[str] = []
    run = 0
    for sentence, kind in zip(sentences, kinds, strict=True):
        first = sentence.split(" ", 1)[0].rstrip(",")
        if first in ("You", "Your", "You're", "You've"):
            run += 1
        else:
            run = 0
        if run >= 3:
            lead = next(
                (candidate for candidate in _VARIETY_LEADS if candidate not in used_leads),
                _VARIETY_LEADS[0],
            )
            used_leads.add(lead)
            rest = sentence[len(first) :].lstrip()
            rest = (first.lower() + " " + rest) if rest else first.lower()
            sentence = f"{lead}{rest}"
            run = 0
        fixed.append(sentence)
        fixed_kinds.append(kind)
    sentences, kinds = fixed, fixed_kinds
    # Word budget: drop lowest-priority sentences first, never below two.
    _KIND_PRIORITY = ("favorites", "tastes", "possessions", "past", "schedule", "work", "pet", "identity")
    while len(" ".join(sentences).split()) > _MAX_PROFILE_WORDS and len(sentences) > 2:
        drop_at = -1
        for priority in _KIND_PRIORITY:
            candidates = [index for index, kind in enumerate(kinds) if kind == priority]
            if candidates:
                drop_at = candidates[-1]
                break
        if drop_at < 0:
            break
        del sentences[drop_at]
        del kinds[drop_at]
    if not sentences:
        return AboutMeProfile(
            status="empty",
            summary=_EMPTY_ABOUT_ME_SUMMARY,
        )
    return AboutMeProfile(
        status="ready",
        summary=" ".join(sentences),
    )


class GraphMemoryService:
    def __init__(
        self,
        repository: PostgresGraphMemoryRepository | None = None,
        projection: Neo4jProjection | None = None,
        embed_fn: Callable[[str], list[float] | None] | None = None,
    ) -> None:
        self.repository = repository or PostgresGraphMemoryRepository()
        # Optional Neo4j projection (None = PostgreSQL-only mode). Wired by
        # app/graph_memory/runtime.py when GRAPH_MEMORY_NEO4J_ENABLED=true.
        self.projection = projection
        # Semantic-recall embedder (None = keyword-only recall). Runtime
        # wires the default snowflake embedder; tests inject fakes.
        self.embed_fn = embed_fn

    def get_status(self, user_id: int) -> GraphMemoryStatusResponse:
        tenant = self.repository.get_tenant(user_id)
        if tenant is None:
            about_me = deterministic_about_me((), graph_revision=0)
            return GraphMemoryStatusResponse(
                enabled=False,
                retention_days=DEFAULT_RETENTION_DAYS,
                revision=0,
                generation=0,
                purge_state="ready",
                last_learned_at=None,
                next_expiry_at=None,
                counts={"active": 0, "pending": 0, "expired": 0},
                about_me=about_me,
            )

        aggregate = self.repository.aggregate(user_id)

        summary_row = self.repository.get_summary(user_id)
        active_items = self.repository.active_items_for_profile(user_id)
        if not active_items:
            # Nothing to summarize: the deterministic empty state is
            # truthful. A stale stored row (all items expired or deleted)
            # must not linger as the profile.
            about_me = deterministic_about_me((), graph_revision=tenant.graph_revision)
        else:
            summary_stale = (
                summary_row is None
                or summary_row["graph_revision"] != tenant.graph_revision
                or summary_row["is_fallback"]
            )
            if summary_stale:
                # Stale-while-revalidate: the graph_revision bump committed
                # with the write, so the stored summary is behind. Trigger
                # the worker (best-effort, single-flight) and serve what is
                # stored until it regenerates. This also heals a skipped
                # summarize enqueue on the next read.
                self.repository.enqueue_summarize_best_effort(
                    user_id, tenant.tenant_uuid, tenant.generation
                )
            if summary_row is not None:
                about_me = AboutMeProfile(
                    status="empty" if summary_row["summary"] == _EMPTY_ABOUT_ME_SUMMARY else "ready",
                    summary=summary_row["summary"],
                )
            else:
                about_me = deterministic_about_me(active_items, graph_revision=tenant.graph_revision)

        return GraphMemoryStatusResponse(
            enabled=tenant.enabled,
            retention_days=tenant.retention_days,
            revision=tenant.revision,
            generation=tenant.generation,
            purge_state=tenant.purge_state,
            last_learned_at=tenant.last_learned_at,
            next_expiry_at=aggregate.pop("next_expiry_at"),
            counts=aggregate,
            about_me=about_me,
        )

    def update_settings(
        self,
        user_id: int,
        *,
        enabled: bool,
        retention_days: int | None,
        expected_revision: int | None,
    ) -> GraphMemoryStatusResponse:
        self.repository.update_settings(
            user_id,
            enabled=enabled,
            retention_days=retention_days,
            expected_revision=expected_revision,
        )
        return self.get_status(user_id)

    def list_items(
        self,
        user_id: int,
        *,
        status: MemoryStatus | None,
        kind: MemoryKind | None,
        limit: int,
        offset: int,
    ) -> GraphMemoryItemPage:
        page = self.repository.list_items(
            user_id,
            status=status,
            kind=kind.value if kind else None,
            limit=limit,
            offset=offset,
        )
        return GraphMemoryItemPage(
            items=list(page.items),
            total=page.total,
            limit=limit,
            offset=offset,
        )

    def edit_item(
        self,
        user_id: int,
        memory_id: UUID,
        *,
        subject: str,
        predicate: str,
        object_value: str,
        expected_revision: int,
    ) -> ItemMutation:
        current = self.repository.get_item(user_id, memory_id)
        candidate = MemoryCandidate(
            kind=current.item.kind,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            confidence=1,
            source_excerpt=f"{subject} {predicate} {object_value}",
            explicit_user_assertion=True,
        )
        decision = validate_candidate(candidate)
        if not decision.accepted or decision.fingerprint is None:
            raise GraphMemoryValidationError(decision.reason or "invalid_memory_item")
        return self.repository.edit_item(
            user_id,
            memory_id,
            subject=candidate.subject,
            predicate=candidate.predicate,
            object_value=candidate.object_value,
            fingerprint=decision.fingerprint,
            expected_revision=expected_revision,
        )

    def approve_item(self, user_id: int, memory_id: UUID) -> ItemMutation:
        return self.repository.approve_item(user_id, memory_id)

    def renew_item(self, user_id: int, memory_id: UUID) -> ItemMutation:
        return self.repository.renew_item(user_id, memory_id)

    def delete_item(self, user_id: int, memory_id: UUID) -> ItemMutation:
        return self.repository.delete_item(user_id, memory_id)

    def clear_relationship_memory(self, user_id: int) -> ClearMemoryResponse:
        result = self.repository.clear(user_id, include_saved_preferences=False)
        return ClearMemoryResponse(
            graph_deleted_count=result.graph_deleted_count,
            generation=result.tenant.generation,
            purge_state=result.tenant.purge_state,
        )

    def clear_personal_memory(self, user_id: int) -> ClearMemoryResponse:
        result = self.repository.clear(user_id, include_saved_preferences=True)
        return ClearMemoryResponse(
            graph_deleted_count=result.graph_deleted_count,
            saved_preferences_deleted_count=result.saved_preferences_deleted_count,
            generation=result.tenant.generation,
            purge_state=result.tenant.purge_state,
        )

    def enqueue_extraction(self, user_id: int, chat_id: UUID, user_message_id: UUID) -> UUID | None:
        return self.repository.enqueue_extraction(user_id, chat_id, user_message_id)

    def extraction_job_exists(self, user_id: int, source_message_id: UUID) -> bool:
        return self.repository.extraction_job_exists(user_id, source_message_id)

    def accept_candidate(
        self,
        user_id: int,
        *,
        expected_generation: int,
        source_chat_id: UUID,
        source_message_id: UUID,
        candidate: MemoryCandidate,
        lease_job_id: UUID | None = None,
        lease_owner: str | None = None,
    ) -> ItemMutation:
        decision = validate_candidate(candidate)
        if not decision.accepted:
            raise GraphMemoryValidationError(decision.reason or "candidate_rejected")
        if decision.identity_key is not None:
            # Deterministic identity fact: canonicalize to the allowlist
            # shape (stable predicate for corrections) and force ACTIVE.
            # Secrets were already rejected above; agreement on the value
            # was checked in validation. Re-validate the canonical form
            # rather than trusting the rewrite.
            kind, subject, predicate = decision.identity_key
            hit = match_identity_fact(candidate.source_excerpt)
            if hit is None or normalize_component(
                candidate.object_value
            ) != normalize_component(hit.value):
                raise GraphMemoryValidationError("identity_mismatch")
            candidate = candidate.model_copy(
                update={
                    "kind": MemoryKind(hit.kind),
                    "subject": "I",
                    "predicate": predicate,
                    "object_value": hit.value,
                    "confidence": 1.0,
                }
            )
            decision = validate_candidate(candidate)
            if not decision.accepted:
                raise GraphMemoryValidationError(decision.reason or "candidate_rejected")
        mutation = self.repository.upsert_candidate(
            user_id,
            expected_generation=expected_generation,
            source_chat_id=source_chat_id,
            source_message_id=source_message_id,
            candidate=candidate,
            decision=decision,
            lease_job_id=lease_job_id,
            lease_owner=lease_owner,
        )
        if decision.identity_key is not None:
            # Corrections supersede: "now Sam" expires live "mk" rows with
            # the same key. Runs after a successful upsert so a failed
            # insert never orphans existing rows.
            kind, subject, predicate = decision.identity_key
            self.repository.expire_superseded(
                user_id,
                kind=kind,
                subject=subject,
                predicate=predicate,
                keep_fingerprint=decision.fingerprint or "",
            )
        return mutation

    def recall(self, user_id: int, query: str, *, limit: int = MAX_RECALL_RESULTS) -> tuple[RecallRecord, ...]:
        """PostgreSQL-authoritative recall: keyword-ranked active items.

        Deterministic stopgap until pgvector-backed semantic recall lands:
        distinct query terms (len>=3, small stops) matched case-insensitively
        against subject/predicate/object; ties break by confidence. Only
        live active rows are ever returned, capped at `limit` for the
        synthesis token budget. Empty/stop-only queries return ().
        """

        terms = {
            term
            for term in re.findall(
                r"[A-Za-z][A-Za-z'\-]*", str(query or "").casefold()
            )
            if len(term) >= 3
            and term
            not in {
                "what", "who", "when", "where", "how", "which", "that",
                "this", "the", "and", "for", "are", "was", "were", "with",
                "from", "you", "your", "about", "into", "have", "has",
                "did", "there", "their", "do", "does",
            }
        }
        if not terms:
            return ()
        scored: list[tuple[int, float, int, GraphMemoryItem]] = []
        for position, item in enumerate(
            self.repository.active_items_for_profile(user_id, limit=100)
        ):
            # Belt-and-braces: only live active rows ever leave recall,
            # even if a repository implementation returns mixed states.
            if str(getattr(item.status, "value", item.status) or "").casefold() != "active":
                continue
            haystack = " ".join(
                str(getattr(item, key, "") or "")
                for key in ("subject", "predicate", "object_value")
            ).casefold()
            matched = sum(1 for term in terms if term in haystack)
            if matched:
                scored.append(
                    (matched, float(item.confidence or 0.0), position, item)
                )
        scored.sort(key=lambda entry: (-entry[0], -entry[1], entry[2]))
        records: list[RecallRecord] = []
        for _, _, _, item in scored[: max(1, int(limit or 0))]:
            records.append(
                RecallRecord(
                    id=item.id,
                    kind=item.kind,
                    subject=item.subject,
                    predicate=item.predicate,
                    object_value=item.object_value,
                    confidence=float(item.confidence or 0.0),
                    expires_at=item.expires_at,
                )
            )
        budget = max(0, int(limit or 0) - len(records))
        if budget > 0:
            seen_ids = {record.id for record in records}
            records.extend(
                self._vector_supplement(user_id, str(query or ""), seen_ids, budget)
            )
            budget = max(0, int(limit or 0) - len(records))
        if self.projection is not None and budget > 0 and terms:
            records.extend(
                self._graph_supplement(
                    user_id, terms, {record.id for record in records}, budget
                )
            )
        return tuple(records)

    def _vector_supplement(
        self,
        user_id: int,
        query: str,
        seen: set[UUID],
        budget: int,
    ) -> tuple[RecallRecord, ...]:
        """Top up recall with cosine-similar rows (synthesis context only).

        Paraphrases keyword recall misses ("address me" vs "call me") land
        here. Every hit is a live ACTIVE PostgreSQL row; the embedder or
        the vector column failing just yields nothing. Never feeds
        rewriting, planning, or web queries.
        """
        if self.embed_fn is None:
            return ()
        search = getattr(self.repository, "search_similar", None)
        if not callable(search):
            return ()
        try:
            vector = self.embed_fn(query)
        except Exception:
            logger.warning("memory query embedding failed open", exc_info=True)
            return ()
        if not vector:
            return ()
        try:
            rows = search(user_id, vector, limit=budget, exclude_ids=sorted(seen))
        except Exception:
            logger.warning("memory vector search failed open", exc_info=True)
            return ()
        out: list[RecallRecord] = []
        for row in rows or ():
            if row.id in seen or len(out) >= budget:
                continue
            seen.add(row.id)
            out.append(
                RecallRecord(
                    id=row.id,
                    kind=row.kind,
                    subject=row.subject,
                    predicate=row.predicate,
                    object_value=row.object_value,
                    confidence=float(row.confidence or 0.0),
                    expires_at=row.expires_at,
                )
            )
        return tuple(out)

    def _embed_row(self, user_id: int, memory_id: UUID, projectable: Any) -> None:
        """Embed one ACTIVE row fail-open (NULL rows stay keyword-only)."""
        if self.embed_fn is None:
            return
        store = getattr(self.repository, "update_embedding", None)
        if not callable(store):
            return
        try:
            vector = self.embed_fn(
                " ".join(
                    (
                        str(projectable.subject or ""),
                        str(projectable.predicate or ""),
                        str(projectable.object_value or ""),
                    )
                )
            )
        except Exception:
            logger.warning("memory row embedding failed open", exc_info=True)
            return
        if not vector:
            return
        try:
            store(user_id, memory_id, vector)
        except Exception:
            logger.warning("memory embedding store failed open", exc_info=True)

    def _graph_supplement(
        self,
        user_id: int,
        terms: set[str],
        seen: set[UUID],
        budget: int,
    ) -> tuple[RecallRecord, ...]:
        """Top up recall with graph-traversed facts (synthesis context only).

        Neo4j contributes traversal (multi-hop related facts keyword recall
        misses); every fact re-resolves to a live ACTIVE PostgreSQL row, so
        Postgres stays authoritative over what the model sees. Any failure
        fails open to the PostgreSQL results above. Never called for
        rewriting, planning, or web queries — recall output only enters the
        synthesis message assembly.
        """
        assert self.projection is not None
        try:
            tenant = self.repository.get_tenant(user_id)
        except Exception:
            logger.warning("Neo4j recall tenant lookup failed open", exc_info=True)
            return ()
        if tenant is None or not bool(getattr(tenant, "enabled", False)):
            return ()
        expanded = set(terms)
        if question_memory_enabled():
            # Phase 5 boost: live question topics overlapping the query add
            # their words to the graph traversal only. The same
            # fail-open, PG re-resolution, and budget rules apply; topics
            # never leave the recall output channel.
            try:
                live = self.projection.read_live_topics(
                    user_id=user_id,
                    tenant_uuid=str(tenant.tenant_uuid),
                    terms=sorted(terms),
                    limit=3,
                )
            except Exception:
                logger.warning(
                    "Neo4j topic boost failed open for user %s", user_id
                )
                live = []
            for topic in live:
                expanded.update(str(topic or "").casefold().split())
            if len(expanded) > 20:
                expanded = set(sorted(expanded)[:20])
        try:
            facts = self.projection.read_related(
                user_id=user_id,
                tenant_uuid=str(tenant.tenant_uuid),
                terms=sorted(expanded),
                limit=min(max(1, budget), GRAPH_RECALL_MAX_FACTS),
            )
        except Exception:
            logger.warning(
                "Neo4j recall failed open for user %s", user_id, exc_info=True
            )
            return ()
        if not facts:
            return ()
        try:
            rows = self.repository.fetch_active_by_fingerprints(
                user_id, [fact.fingerprint for fact in facts]
            )
        except Exception:
            logger.warning(
                "Neo4j recall re-resolution failed open for user %s", user_id
            )
            return ()
        out: list[RecallRecord] = []
        for row in rows:
            if row.id in seen or len(out) >= budget:
                continue
            seen.add(row.id)
            out.append(
                RecallRecord(
                    id=row.id,
                    kind=row.kind,
                    subject=row.subject,
                    predicate=row.predicate,
                    object_value=row.object_value,
                    confidence=float(row.confidence or 0.0),
                    expires_at=row.expires_at,
                )
            )
        return tuple(out)

    def project_item(self, mutation: ItemMutation, *, deleted: bool = False) -> TenantRecord:
        current = self.repository.get_tenant(mutation.tenant.user_id)
        if (
            current is None
            or current.generation != mutation.tenant.generation
            or current.tenant_uuid != mutation.tenant.tenant_uuid
            or (not deleted and current.purge_state != "ready")
        ):
            raise GraphMemoryValidationError("memory_generation_changed")
        user_id = mutation.tenant.user_id
        generation = mutation.tenant.generation
        tenant_uuid = str(mutation.tenant.tenant_uuid)
        if deleted:
            fingerprint = getattr(mutation.item, "fingerprint", None) or None
            if fingerprint is None:
                projectable = self.repository.get_projectable_item(
                    user_id, mutation.item.id
                )
                fingerprint = projectable.fingerprint if projectable is not None else None
            try:
                self.repository.delete_item(user_id, mutation.item.id)
            except GraphMemoryNotFoundError:
                # Already gone from PostgreSQL; the Neo4j cleanup below
                # still applies when the fingerprint was captured.
                pass
            if self.projection is not None and fingerprint:
                # Propagates on driver failure: the retry re-resolves the
                # payload fingerprint and retries only the Neo4j delete —
                # idempotent by fingerprint key.
                self.projection.delete_item(
                    tenant_uuid=tenant_uuid,
                    user_id=user_id,
                    fingerprint=fingerprint,
                )
            return self.repository.mark_item_projection_complete(
                user_id, generation, mutation.item.id, mutation.job_id, deleted=True
            )
        projectable = self.repository.get_projectable_item(user_id, mutation.item.id)
        if projectable is None:
            # Expired or vanished between queue and execution: job done.
            return self.repository.mark_item_projection_complete(
                user_id,
                generation,
                mutation.item.id,
                mutation.job_id,
                deleted=False,
                projected=False,
            )
        if str(projectable.status or "").casefold() != "active":
            # Pending/rejected rows are PG-correct as-is ('pending'); the job
            # completes without ever touching Neo4j.
            return self.repository.mark_item_projection_complete(
                user_id,
                generation,
                mutation.item.id,
                mutation.job_id,
                deleted=False,
                projected=False,
            )
        if self.projection is None:
            # PostgreSQL-only mode: the job completes, the row stays
            # 'pending' — nothing may claim 'projected' without a driver.
            # Loud on purpose: silent pending rows mean the projector never
            # attached (see runtime.build_neo4j_projection + recheck).
            logger.warning(
                "projection job completed PostgreSQL-only for user %s (no Neo4j projector)",
                user_id,
            )
            self._embed_row(user_id, mutation.item.id, projectable)
            return self.repository.mark_item_projection_complete(
                user_id,
                generation,
                mutation.item.id,
                mutation.job_id,
                deleted=False,
                projected=False,
            )
        from app.security.redact import contains_credentials

        text = " ".join(
            (projectable.subject, projectable.predicate, projectable.object_value)
        )
        if contains_credentials(text):
            # Backstop (validation already rejects these): never project
            # secret-bearing text; deterministic skip, no retry loop.
            self.repository.mark_item_projection_skipped(
                user_id, generation, mutation.item.id, "secret_suspect"
            )
            return self.repository.mark_item_projection_complete(
                user_id,
                generation,
                mutation.item.id,
                mutation.job_id,
                deleted=False,
                projected=False,
            )
        # Driver failure propagates: the worker requeues with backoff and
        # marks 'failed' only at terminal attempts. 'projected' is written
        # solely by mark_item_projection_complete(projected=True) below.
        self.projection.upsert_item(projectable)
        self._embed_row(user_id, mutation.item.id, projectable)
        return self.repository.mark_item_projection_complete(
            user_id,
            generation,
            mutation.item.id,
            mutation.job_id,
            deleted=False,
            projected=True,
        )

    def purge_projection(self, user_id: int, generation: int, job_id: UUID | None) -> TenantRecord:
        self.repository.delete_items_by_tenant(user_id, generation)
        if self.projection is not None:
            tenant = self.repository.get_tenant(user_id)
            if tenant is None:
                logger.warning(
                    "Neo4j tenant purge skipped: no tenant row for user %s", user_id
                )
            elif int(tenant.generation) != int(generation):
                # Stale purge job: a newer clear/relearn cycle owns the
                # tenant_uuid now — deleting would wipe live nodes.
                logger.warning(
                    "Neo4j tenant purge fenced: job generation %s != current %s",
                    generation,
                    tenant.generation,
                )
            else:
                self.projection.delete_tenant(
                    tenant_uuid=str(tenant.tenant_uuid), user_id=user_id
                )
        return self.repository.mark_purge_complete(user_id, generation, job_id)

    def expire_due(self, *, limit: int = 500) -> int:
        """Delete due canonical rows; canonical state is the only authority."""

        expired = self.repository.expire_due(limit=limit)
        if self.projection is not None:
            for item in expired:
                if not item.fingerprint or item.tenant_uuid is None:
                    continue
                try:
                    self.projection.delete_item(
                        tenant_uuid=str(item.tenant_uuid),
                        user_id=item.user_id,
                        fingerprint=item.fingerprint,
                    )
                except Neo4jUnavailable:
                    # Fail-open: expiry already committed in Postgres; the
                    # drift report surfaces the orphan for healing.
                    logger.warning(
                        "Neo4j expiry cleanup deferred for user %s", item.user_id
                    )
            try:
                self.projection.delete_expired_topics()
            except Neo4jUnavailable:
                logger.warning("Neo4j topic TTL cleanup deferred")
        return len(expired)

    def record_question_topics(self, user_id: int, text: str) -> tuple[str, ...]:
        """Store deterministic question topics (flag + consent gated).

        Never raises: topic learning must not break message extraction.
        Topics boost recall term expansion only — they never enter R1,
        the planner, or web queries (recall output is synthesis context).
        """
        from .question_topics import extract_question_topics

        if not question_memory_enabled():
            return ()
        if self.projection is None:
            return ()
        try:
            tenant = self.repository.get_tenant(user_id)
        except Exception:
            logger.warning("question topics tenant lookup failed open", exc_info=True)
            return ()
        if tenant is None or not bool(getattr(tenant, "enabled", False)):
            return ()
        topics = extract_question_topics(text)
        if not topics:
            return ()
        try:
            self.projection.store_question_topics(
                user_id=user_id,
                tenant_uuid=str(tenant.tenant_uuid),
                topics=list(topics),
            )
        except Neo4jUnavailable:
            logger.warning(
                "question topics store failed open for user %s", user_id
            )
            return ()
        except Exception:
            logger.warning(
                "question topics store failed open for user %s",
                user_id,
                exc_info=True,
            )
            return ()
        return topics

    def list_question_topics(self, user_id: int) -> list[dict[str, Any]]:
        """User control: view live question topics (flag-gated)."""
        if not question_memory_enabled() or self.projection is None:
            return []
        try:
            tenant = self.repository.get_tenant(user_id)
        except Exception:
            return []
        if tenant is None:
            return []
        try:
            return self.projection.list_topics(
                user_id=user_id, tenant_uuid=str(tenant.tenant_uuid)
            )
        except Neo4jUnavailable:
            return []

    def delete_question_topic(self, user_id: int, name: str) -> bool:
        """User control: delete one topic; False when disabled or missing."""
        if not question_memory_enabled() or self.projection is None:
            return False
        cleaned = " ".join(str(name or "").split()).strip()
        if not cleaned:
            return False
        try:
            self.projection.delete_topic(user_id=user_id, name=cleaned)
        except Neo4jUnavailable:
            return False
        return True


__all__ = [
    "GraphMemoryService",
    "candidate_fingerprint",
    "deterministic_about_me",
]
