from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.graph_memory.models import (
    GraphMemoryItem,
    MemoryCandidate,
    MemoryKind,
    MemoryStatus,
    RecallRecord,
)
from app.graph_memory.repository import GraphMemoryValidationError, ItemMutation, TenantRecord
from app.graph_memory.service import GraphMemoryService, deterministic_about_me
from app.graph_memory.validation import candidate_fingerprint, validate_candidate
from app.graph_memory.worker import GraphMemoryJob, GraphMemoryWorker


def candidate(**overrides) -> MemoryCandidate:
    values = {
        "kind": "relationship",
        "subject": "I",
        "predicate": "has_pet",
        "object_value": "Bheem, a Labrador",
        "confidence": 0.95,
        "source_excerpt": "My dog Bheem is a Labrador.",
        "explicit_user_assertion": True,
    }
    values.update(overrides)
    return MemoryCandidate(**values)


def item(**overrides) -> GraphMemoryItem:
    now = datetime(2026, 7, 16, tzinfo=UTC)
    values = {
        "id": uuid4(),
        "kind": "relationship",
        "subject": "I",
        "predicate": "has_pet",
        "object_value": "Bheem, a Labrador",
        "confidence": 0.95,
        "status": "active",
        "source_chat_id": uuid4(),
        "source_message_id": uuid4(),
        "source_excerpt": "My dog Bheem is a Labrador.",
        "source_created_at": now,
        "last_confirmed_at": now,
        "expires_at": now + timedelta(days=90),
        "revision": 1,
        "projection_state": "projected",
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return GraphMemoryItem(**values)


def tenant(**overrides) -> TenantRecord:
    values = {
        "user_id": 41,
        "tenant_uuid": uuid4(),
        "enabled": True,
        "retention_days": 90,
        "generation": 3,
        "revision": 2,
        "graph_revision": 7,
        "purge_state": "ready",
        "last_learned_at": None,
    }
    values.update(overrides)
    return TenantRecord(**values)


def test_safe_automatic_candidate_decision_and_stable_fingerprint() -> None:
    active = validate_candidate(candidate())
    assert active.accepted is True
    assert active.status == "active"
    assert len(active.fingerprint or "") == 64

    review = validate_candidate(candidate(confidence=0.7))
    assert review.accepted is True
    assert review.status == "pending"

    normalized = candidate(
        subject=" i ", predicate="HAS__PET", object_value="  BHEEM,   A LABRADOR "
    )
    assert candidate_fingerprint(normalized) == candidate_fingerprint(candidate())

    preference = validate_candidate(
        candidate(
            kind="preference",
            predicate="prefers",
            object_value="concise answers",
            source_excerpt="I prefer concise answers.",
        )
    )
    assert preference.accepted is True
    assert preference.status == "active"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"explicit_user_assertion": False}, "not_explicit_user_assertion"),
        ({"source_excerpt": "What kind of dog is Bheem?"}, "question"),
        ({"source_excerpt": "Maybe my dog is a Labrador."}, "uncertain_statement"),
        (
            {
                "object_value": "password is secret123",
                "source_excerpt": "My password is secret123.",
            },
            "sensitive_data",
        ),
        (
            {
                "subject": "Japan",
                "predicate": "capital",
                "object_value": "Tokyo",
                "source_excerpt": "Japan's capital is Tokyo.",
            },
            "not_personal",
        ),
        (
            {
                "subject": "I",
                "predicate": "friend_lives_in",
                "object_value": "Delhi",
                "source_excerpt": "My friend Alice lives in Delhi.",
            },
            "third_party_assertion",
        ),
        (
            {
                "subject": "Alice",
                "predicate": "has_condition",
                "object_value": "cancer",
                "source_excerpt": "My friend Alice has cancer.",
                "contains_sensitive_data": False,
            },
            "sensitive_data",
        ),
        (
            {
                "predicate": "has_condition",
                "object_value": "cancer",
                "source_excerpt": "I have cancer.",
                "contains_sensitive_data": False,
            },
            "sensitive_data",
        ),
        (
            {
                "predicate": "annual_income",
                "object_value": "₹2,000,000",
                "source_excerpt": "I earn ₹2,000,000 per year.",
                "contains_sensitive_data": False,
            },
            "sensitive_data",
        ),
        (
            {
                "predicate": "religion",
                "object_value": "Hindu",
                "source_excerpt": "I am Hindu.",
                "contains_sensitive_data": False,
            },
            "sensitive_data",
        ),
        ({"confidence": 0.2}, "confidence_too_low"),
    ],
)
def test_candidate_validator_rejects_unsafe_or_non_personal_content(changes, reason) -> None:
    decision = validate_candidate(candidate(**changes))
    assert decision.accepted is False
    assert decision.status is None
    assert decision.reason == reason
    assert decision.fingerprint is None


def test_deterministic_about_me_has_per_sentence_backing_and_groups() -> None:
    pet = item()
    preference = item(
        id=uuid4(),
        kind="preference",
        predicate="prefers",
        object_value="concise answers",
    )

    profile = deterministic_about_me([pet, preference], graph_revision=9)

    assert profile.status == "ready"
    assert "Bheem" in profile.summary
    assert "concise answers" in profile.summary


def _named_item(name: str, **overrides) -> GraphMemoryItem:
    values = {"predicate": "name", "object_value": name, "source_excerpt": f"My name is {name}."}
    values.update(overrides)
    return item(id=uuid4(), **values)


def test_about_me_two_row_case_renders_natural_english() -> None:
    pet = item(
        id=uuid4(),
        predicate="has_pet",
        object_value="Bheem",
        source_excerpt="I have pet called Bheem, and it is a golden retriever.",
    )
    name = _named_item("Maximus Decimus Meridius")

    profile = deterministic_about_me([pet, name], graph_revision=9)

    assert profile.status == "ready"
    assert "I has" not in profile.summary
    assert "I name" not in profile.summary
    assert "You're Maximus Decimus Meridius." in profile.summary
    assert "You have a pet named Bheem." in profile.summary


def test_about_me_name_only() -> None:
    profile = deterministic_about_me([_named_item("Sam")], graph_revision=1)

    assert profile.summary == "You're Sam."


def test_about_me_preference_only() -> None:
    preference = item(
        id=uuid4(),
        kind="preference",
        predicate="prefers",
        object_value="concise answers",
    )

    profile = deterministic_about_me([preference], graph_revision=1)

    assert profile.summary == "You prefer concise answers."


def test_about_me_three_plus_facts_flow() -> None:
    rows = [
        _named_item("Sam"),
        item(id=uuid4(), predicate="favorite_color", object_value="blue"),
        item(id=uuid4(), predicate="collect", object_value="vintage stamps"),
    ]

    profile = deterministic_about_me(rows, graph_revision=1)

    # Grouped prose: identity + favorite in two sentences; the vague
    # "collect" row is dropped from the portrait (still listed in UI).
    assert profile.summary == (
        "You're Sam. Your favorite color is blue."
    )
    assert "vintage stamps" not in profile.summary


def test_about_me_pet_name_and_breed_merge() -> None:
    pet = item(id=uuid4(), predicate="has_pet", object_value="Bheem")
    breed = item(id=uuid4(), predicate="pet_breed", object_value="golden retriever")

    profile = deterministic_about_me([pet, breed], graph_revision=1)

    assert profile.summary == "You have a golden retriever named Bheem."


def test_about_me_ambiguous_pet_link_stays_separate() -> None:
    bheem = item(id=uuid4(), predicate="has_pet", object_value="Bheem")
    tommy = item(id=uuid4(), predicate="has_dog", object_value="Tommy")
    breed = item(id=uuid4(), predicate="pet_breed", object_value="golden retriever")

    profile = deterministic_about_me([bheem, tommy, breed], graph_revision=1)

    assert "retriever named" not in profile.summary
    assert "You have a pet named Bheem and a dog named Tommy." in profile.summary
    assert "Your pet is a golden retriever." in profile.summary


def test_about_me_breed_naming_another_pet_stays_separate() -> None:
    pet = item(id=uuid4(), predicate="has_pet", object_value="Bheem")
    breed = item(id=uuid4(), predicate="pet_breed", object_value="Tommy is a pug")

    profile = deterministic_about_me([pet, breed], graph_revision=1)

    assert "pug named Bheem" not in profile.summary
    assert "You have a pet named Bheem." in profile.summary
    assert "Your pet is Tommy is a pug." in profile.summary


def test_about_me_past_possession_renders_past_tense() -> None:
    profile = deterministic_about_me(
        [item(id=uuid4(), predicate="had", object_value="a Golden Retriever named Bheem")],
        graph_revision=1,
    )

    assert profile.summary == "You had a Golden Retriever named Bheem."


def test_about_me_zero_facts_unchanged() -> None:
    profile = deterministic_about_me([], graph_revision=0)

    assert profile.status == "empty"
    assert profile.summary == "No relationship memories yet."


def test_about_me_live_rows_second_person_with_articles() -> None:
    pc = item(
        id=uuid4(),
        predicate="have",
        object_value="PC with 32gb ram and 8c AMD Radeon 7 CPU",
    )
    dog = item(id=uuid4(), predicate="has", object_value="dog called Bheem")

    profile = deterministic_about_me([pc, dog], graph_revision=1)

    assert profile.summary == (
        "You have a PC with 32gb ram and 8c AMD Radeon 7 CPU and a dog called Bheem."
    )


def test_about_me_article_passthrough_cases() -> None:
    def row(predicate: str, obj: str) -> GraphMemoryItem:
        return item(id=uuid4(), predicate=predicate, object_value=obj)

    assert deterministic_about_me(
        [row("has", "Bheem, a Labrador")], graph_revision=1
    ).summary == "You have Bheem, a Labrador."
    assert deterministic_about_me(
        [row("has", "32gb of ram")], graph_revision=1
    ).summary == "You have 32gb of ram."
    assert deterministic_about_me(
        [row("has", "the flu")], graph_revision=1
    ).summary == "You have the flu."
    assert deterministic_about_me(
        [row("has", "Bheem")], graph_revision=1
    ).summary == "You have Bheem."


def _profile_row(predicate: str, obj: str, **overrides) -> GraphMemoryItem:
    values = {"predicate": predicate, "object_value": obj, "source_excerpt": obj}
    values.update(overrides)
    return item(id=uuid4(), **values)


def test_about_me_paragraph_composer_full_profile() -> None:
    rows = [
        _profile_row("name", "Jordan"),
        _profile_row("occupation", "tech-focused engineer"),
        _profile_row("enjoys", "building and breaking things until they finally behave"),
        _profile_row(
            "work",
            "Linux, Docker, Kubernetes, AI/RAG pipelines, databases, "
            "and monitoring infrastructure",
        ),
        _profile_row(
            "interested_in",
            "business, banking, finance, and how companies and markets actually work",
        ),
        _profile_row("hobby", "gaming"),
        _profile_row("hobby", "cooking"),
        _profile_row("asks", "increasingly chaotic questions at 3 AM"),
        _profile_row("favorite_color", "blue"),
        _profile_row("has_pet", "Bheem"),
        _profile_row("collect", "vintage stamps"),
    ]

    profile = deterministic_about_me(rows, graph_revision=1)

    assert profile.status == "ready"
    assert profile.summary == (
        "You're Jordan. "
        "You work as a tech-focused engineer with Linux, Docker, Kubernetes, "
        "AI/RAG pipelines, databases, and monitoring infrastructure. "
        "In your time off, you have a pet named Bheem. "
        "Outside of that, you enjoy building and breaking things until they "
        "finally behave, gaming, and cooking and are interested in business, "
        "banking, finance, and how companies and markets actually work."
    )
    # Portrait condenses: low-signal rows never become sentences.
    assert "chaotic questions" not in profile.summary
    assert "vintage stamps" not in profile.summary


def test_about_me_paragraph_composer_no_ungrounded_color() -> None:
    rows = [
        _profile_row("name", "Jordan"),
        _profile_row("work", "Linux, Docker"),
        _profile_row("hobby", "gaming"),
    ]

    summary = deterministic_about_me(rows, graph_revision=1).summary.lower()

    for color in ("heavily", "seriously", "somehow", "extremely"):
        assert color not in summary


def test_about_me_identity_without_role_keeps_stable_name() -> None:
    rows = [
        _profile_row("name", "Sam"),
        _profile_row("work", "Linux"),
    ]

    profile = deterministic_about_me(rows, graph_revision=1)

    assert profile.summary == "You're Sam. You work with Linux."


def test_about_me_role_without_name() -> None:
    profile = deterministic_about_me(
        [_profile_row("occupation", "engineer")], graph_revision=1
    )

    assert profile.summary == "You work as an engineer."


def test_status_builds_about_me_directly_from_current_active_items() -> None:
    remembered = item()
    repository = SimpleNamespace(
        get_tenant=lambda _user_id: tenant(),
        aggregate=lambda _user_id: {
            "active": 1,
            "pending": 0,
            "expired": 0,
            "next_expiry_at": remembered.expires_at,
        },
        active_items_for_profile=lambda _user_id: (remembered,),
        # No persisted summary: get_status must build About Me from active items.
        get_summary=lambda _user_id: None,
        enqueue_summarize_best_effort=lambda *args: True,
    )

    status = GraphMemoryService(repository=repository).get_status(41)  # type: ignore[arg-type]

    assert status.about_me.status == "ready"
    assert "Bheem" in status.about_me.summary


def test_new_user_status_is_disabled_without_creating_memory_state() -> None:
    repository = SimpleNamespace(get_tenant=lambda _user_id: None)

    status = GraphMemoryService(repository=repository).get_status(41)  # type: ignore[arg-type]

    assert status.enabled is False
    assert status.retention_days == 90
    assert status.revision == 0
    assert status.about_me.status == "empty"


class RecallRepository:
    def __init__(self, *, tenant_record: TenantRecord, authorized: list[GraphMemoryItem]) -> None:
        self.tenant_record = tenant_record
        self.authorized = authorized
        self.requested_ids: list[UUID] = []

    def get_tenant(self, user_id: int):
        assert user_id == self.tenant_record.user_id
        return self.tenant_record

    def authorized_recall_items(self, user_id: int, memory_ids: list[UUID]):
        assert user_id == self.tenant_record.user_id
        self.requested_ids = memory_ids
        by_id = {value.id: value for value in self.authorized}
        return tuple(by_id[value] for value in memory_ids if value in by_id)


class RecallProjection:
    def __init__(self, records: list[RecallRecord]) -> None:
        self.records = records
        self.calls = []

    def recall(self, tenant_uuid, query, *, limit):
        self.calls.append((tenant_uuid, query, limit))
        return self.records


def test_recall_is_postgres_authoritative_keyword_ranked() -> None:
    """Neo4j projection excised; recall ranks live active PG rows only."""

    remembered = item(
        kind="fact",
        subject="I",
        predicate="name",
        object_value="mk",
        confidence=0.95,
        status="active",
    )
    service = GraphMemoryService(
        repository=SimpleNamespace(
            active_items_for_profile=lambda user_id, *, limit=100: (remembered,)
        )
    )  # type: ignore[arg-type]

    assert [r.object_value for r in service.recall(41, "what is my name?")] == ["mk"]
    assert service.recall(41, "anything unrelated at all") == ()
    assert service.recall(41, "") == ()


def test_safe_candidate_boundary_never_calls_repository_for_rejected_candidate() -> None:
    repository = SimpleNamespace()
    service = GraphMemoryService(repository=repository)  # type: ignore[arg-type]
    with pytest.raises(GraphMemoryValidationError, match="sensitive_data"):
        service.accept_candidate(
            41,
            expected_generation=1,
            source_chat_id=uuid4(),
            source_message_id=uuid4(),
            candidate=candidate(
                object_value="API key: abcdef123456",
                source_excerpt="My API key is abcdef123456.",
            ),
        )


class ProjectionRepository:
    def __init__(self, current: TenantRecord) -> None:
        self.current = current
        self.marked = False

    def get_tenant(self, _user_id):
        return self.current

    def mark_item_projection_complete(self, *_args, **_kwargs):
        self.marked = True
        return self.current


def test_projection_generation_fence_prevents_post_clear_recreation() -> None:
    stale_tenant = tenant(generation=3)
    repository = ProjectionRepository(tenant(generation=4, tenant_uuid=stale_tenant.tenant_uuid))
    service = GraphMemoryService(repository=repository)  # type: ignore[arg-type]
    mutation = ItemMutation(stale_tenant, item(), uuid4())

    with pytest.raises(GraphMemoryValidationError, match="generation"):
        service.project_item(mutation)

    assert repository.marked is False


# FIX 1 golden set: instructions are not facts about the user.
def test_instruction_shaped_candidates_are_dropped() -> None:
    cases = [
        ("ask me questions", "I", "enjoy", "asking questions"),
        ("explain like I'm five", "I", "prefer", "simple explanations"),
        ("be concise", "I", "prefer", "concise answers"),
    ]
    for excerpt, subject, predicate, obj in cases:
        decision = validate_candidate(
            candidate(
                subject=subject,
                predicate=predicate,
                object_value=obj,
                confidence=0.9,
                source_excerpt=excerpt,
                explicit_user_assertion=True,
            )
        )
        assert decision.accepted is False, excerpt
        assert decision.reason == "instruction_not_fact", excerpt


def test_genuine_first_person_facts_still_accepted() -> None:
    for excerpt, obj in [("I love sushi", "sushi"), ("my name is mk", "mk")]:
        decision = validate_candidate(
            candidate(
                subject="I",
                predicate="loves" if "sushi" in obj else "name",
                object_value=obj,
                confidence=0.7,
                source_excerpt=excerpt,
                explicit_user_assertion=True,
            )
        )
        assert decision.accepted is True, excerpt


# FIX 2: valueless secret labels.
def test_valueless_secret_labels_rejected() -> None:
    from app.graph_memory.validation import _contains_sensitive_data

    for text in [
        "Password for http://example.test:1:",
        "password:",
        "api key",
        "secret",
        "token",
        "ssh key fingerprint",
        "my git credential",
    ]:
        assert _contains_sensitive_data(text) is True, text


def test_legit_prose_with_secret_adjacent_words_passes() -> None:
    from app.graph_memory.validation import _contains_sensitive_data

    assert _contains_sensitive_data("the secret to good dosa is patience") is False


def test_rejected_secret_never_logged_nor_stored(caplog) -> None:
    import logging

    from app.graph_memory.repository import GraphMemoryValidationError
    from app.graph_memory.service import GraphMemoryService

    marker = "sk-marker-not-a-real-secret-xyz"

    class FakeRepo:
        def __init__(self):
            self.inserts = []

        def upsert_candidate(self, *args, **kwargs):
            self.inserts.append(kwargs)
            raise AssertionError("must not persist secrets")

    repo = FakeRepo()
    service = GraphMemoryService(repository=repo)  # type: ignore[arg-type]
    with caplog.at_level(logging.INFO, logger="app.graph_memory.worker"):
        try:
            service.accept_candidate(
                41,
                expected_generation=3,
                source_chat_id=uuid4(),
                source_message_id=uuid4(),
                candidate=candidate(
                    subject="I",
                    predicate="uses",
                    object_value=f"token {marker}",
                    confidence=0.9,
                    source_excerpt=f"my token is {marker}",
                    explicit_user_assertion=True,
                ),
            )
        except GraphMemoryValidationError:
            pass
    assert repo.inserts == []
    assert marker not in caplog.text


# FIX 3: deterministic identity allowlist.
def test_identity_allowlist_matches_and_denies() -> None:
    from app.graph_memory.validation import match_identity_fact

    hit = match_identity_fact("my name is mk")
    assert hit is not None and hit.value == "mk" and hit.predicate == "name"
    hit = match_identity_fact("call me Sam")
    assert hit is not None and hit.value == "Sam"
    hit = match_identity_fact("speak to me in Tamil")
    assert hit is not None and hit.predicate == "language"
    hit = match_identity_fact("I prefer Celsius units")
    assert hit is not None and hit.predicate == "units"
    for text in [
        "my salary is high",
        "my sister likes cricket",
        "I might be Sam",
        "my password is hunter2",
        "hello there",
        "",
    ]:
        assert match_identity_fact(text) is None, text


def test_allowlisted_fact_activates_and_supersedes() -> None:
    from app.graph_memory.service import GraphMemoryService

    expired: list = []
    inserted: list = []

    class FakeRepo:
        def expire_superseded(self, user_id, *, kind, subject, predicate, keep_fingerprint):
            expired.append((kind, subject, predicate))
            return 1

        def upsert_candidate(self, *args, **kwargs):
            inserted.append(kwargs)
            return SimpleNamespace(item=SimpleNamespace(id=uuid4()))

    service = GraphMemoryService(repository=FakeRepo())  # type: ignore[arg-type]
    service.accept_candidate(
        41,
        expected_generation=3,
        source_chat_id=uuid4(),
        source_message_id=uuid4(),
        candidate=candidate(
            kind="fact",
            subject="I",
            predicate="name",
            object_value="mk",
            confidence=0.6,
            source_excerpt="my name is mk",
            explicit_user_assertion=True,
        ),
    )
    assert expired == [("fact", "i", "name")]
    assert inserted, "allowlisted identity facts must persist as ACTIVE"
    assert inserted[0]["decision"].status.value == "active"


def test_recall_returns_active_bounded_and_ignores_pending() -> None:
    from app.graph_memory.service import GraphMemoryService

    active = item(
        kind="fact",
        subject="I",
        predicate="name",
        object_value="mk",
        confidence=0.95,
        status="active",
    )
    pending = item(
        kind="fact",
        subject="I",
        predicate="name",
        object_value="Intruder",
        confidence=0.6,
        status="pending",
    )

    class FakeRepo:
        def active_items_for_profile(self, user_id, *, limit=100):
            assert limit <= 100
            return (active, pending)

    service = GraphMemoryService(repository=FakeRepo())  # type: ignore[arg-type]
    found = service.recall(41, "what is my name?")
    objects = [record.object_value for record in found]
    assert "mk" in objects
    assert "Intruder" not in objects


def test_service_recall_acceptance_mk_then_sam() -> None:
    from app.graph_memory.service import GraphMemoryService

    rows: dict = {}

    class FakeRepo:
        def active_items_for_profile(self, user_id, *, limit=100):
            return tuple(rows.values())

        def expire_superseded(self, user_id, *, kind, subject, predicate, keep_fingerprint):
            doomed = [
                key
                for key, row in rows.items()
                if row.predicate == predicate and row.fingerprint != keep_fingerprint
            ]
            for key in doomed:
                del rows[key]
            return len(doomed)

        def upsert_candidate(self, *args, **kwargs):
            cand = kwargs["candidate"]
            decision = kwargs["decision"]
            kind = cand.kind if isinstance(cand.kind, str) else cand.kind.value
            rows[cand.object_value] = SimpleNamespace(
                id=uuid4(),
                kind=kind,
                subject=cand.subject,
                predicate=cand.predicate,
                object_value=cand.object_value,
                confidence=1.0,
                status="active",
                fingerprint=decision.fingerprint,
                expires_at=datetime(2027, 1, 1, tzinfo=UTC),
            )
            return SimpleNamespace(item=rows[cand.object_value])

    service = GraphMemoryService(repository=FakeRepo())  # type: ignore[arg-type]

    def accept(text, value):
        service.accept_candidate(
            41,
            expected_generation=3,
            source_chat_id=uuid4(),
            source_message_id=uuid4(),
            candidate=candidate(
                kind="fact",
                subject="I",
                predicate="name",
                object_value=value,
                confidence=0.6,
                source_excerpt=text,
                explicit_user_assertion=True,
            ),
        )

    accept("my name is mk", "mk")
    assert [r.object_value for r in service.recall(41, "what is my name?")] == ["mk"]
    accept("my name is now Sam", "Sam")
    assert [r.object_value for r in service.recall(41, "what is my name?")] == ["Sam"]


# ---------------------------------------------------------------------------
# Phase 4: vector recall tops up keyword misses (synthesis context only)
# ---------------------------------------------------------------------------


def _recall_row(object_value="KING", predicate="is called"):
    return SimpleNamespace(
        id=uuid4(),
        kind=MemoryKind.FACT,
        subject="I",
        predicate=predicate,
        object_value=object_value,
        confidence=0.95,
        status=MemoryStatus.ACTIVE,
        expires_at=datetime(2027, 1, 1, tzinfo=UTC),
    )


def _vector_service(**overrides):
    import app.graph_memory.service as service_module

    kwargs = {
        "repository": SimpleNamespace(
            active_items_for_profile=lambda user_id, *, limit=100: (),
            search_similar=lambda user_id, vector, *, limit, exclude_ids: (),
        ),
        "embed_fn": lambda text: [1.0] * 1024,
    }
    kwargs.update(overrides)
    return service_module.GraphMemoryService(**kwargs)


def test_vector_recall_finds_paraphrase_keyword_misses():
    king = _recall_row()
    service = _vector_service(
        repository=SimpleNamespace(
            active_items_for_profile=lambda user_id, *, limit=100: (king,),
            search_similar=lambda user_id, vector, *, limit, exclude_ids: (king,),
        )
    )
    found = service.recall(41, "how should you address me?")
    assert [r.object_value for r in found] == ["KING"]


def test_vector_recall_without_embedder_is_keyword_only():
    service = _vector_service(embed_fn=None)
    assert service.recall(41, "how should you address me?") == ()


def test_vector_recall_survives_embedder_and_search_failures():
    def boom(text):
        raise ConnectionError("ollama down")

    service = _vector_service(embed_fn=boom)
    assert service.recall(41, "how should you address me?") == ()
    remembered = _recall_row(object_value="mk", predicate="name")
    service2 = _vector_service(
        repository=SimpleNamespace(
            active_items_for_profile=lambda user_id, *, limit=100: (remembered,),
            search_similar=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db")),
        ),
        embed_fn=boom,
    )
    assert [r.object_value for r in service2.recall(41, "what is my name?")] == ["mk"]


def test_vector_recall_respects_total_budget():
    rows = tuple(_recall_row(object_value=f"v{i}") for i in range(6))
    service = _vector_service(
        repository=SimpleNamespace(
            active_items_for_profile=lambda user_id, *, limit=100: (),
            search_similar=lambda user_id, vector, *, limit, exclude_ids: rows,
        )
    )
    found = service.recall(41, "how should you address me?", limit=3)
    assert len(found) == 3


# ---------------------------------------------------------------------------
# FIX 1: validation gaps — health/financial/credential lexicon + split views
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "excerpt",
    [
        "my back pain is worse today",
        "I have chronic pain in my knee",
        "my symptoms started yesterday",
        "I was diagnosed with flu",
        "I take medication every morning",
        "seeing my doctor about this illness",
        "my therapy appointment is Tuesday",
        "I injured my ankle running",
        "my mental health has been rough",
        "I have trouble sleeping most nights",
    ],
)
def test_health_statements_reject(excerpt):
    decision = validate_candidate(
        candidate(
            subject="I",
            predicate="suffers from",
            object_value="back pain",
            confidence=1.0,
            source_excerpt=excerpt,
            explicit_user_assertion=True,
        )
    )
    assert decision.accepted is False, excerpt


def test_health_rejects_when_only_excerpt_carries_signal():
    # Split form: object is innocuous, excerpt holds the health content.
    decision = validate_candidate(
        candidate(
            subject="I",
            predicate="have",
            object_value="most nights",
            confidence=1.0,
            source_excerpt="I have trouble sleeping most nights",
            explicit_user_assertion=True,
        )
    )
    assert decision.accepted is False


@pytest.mark.parametrize(
    ("excerpt", "obj"),
    [
        ("my api token is mock-token-1", "mock-token-1"),
        ("my token: mock-token-2", "mock-token-2"),
        ("token mock-token-3", "mock-token-3"),
        ("auth token = mock-token-4", "mock-token-4"),
        ("bearer mock-token-5", "mock-token-5"),
        ("my private key is mock-key-6", "mock-key-6"),
        ("the secret is mock-secret-7", "mock-secret-7"),
        ("passphrase mock-phrase-8", "mock-phrase-8"),
        ("my pin is 4820", "4820"),
        ("my api token is mock-token-9", "mock"),
        ("my api token is mock-token-10", "api token"),
    ],
)
def test_credential_phrasings_reject_including_splits(excerpt, obj):
    decision = validate_candidate(
        candidate(
            subject="I",
            predicate="possesses",
            object_value=obj,
            confidence=1.0,
            source_excerpt=excerpt,
            explicit_user_assertion=True,
        )
    )
    assert decision.accepted is False, excerpt


@pytest.mark.parametrize(
    "excerpt",
    [
        "my salary was increased",
        "I am in debt",
        "my income is monthly",
        "looking at my bank account",
    ],
)
def test_financial_statements_reject(excerpt):
    decision = validate_candidate(
        candidate(
            subject="I",
            predicate="has",
            object_value="money trouble",
            confidence=1.0,
            source_excerpt=excerpt,
            explicit_user_assertion=True,
        )
    )
    assert decision.accepted is False, excerpt


@pytest.mark.parametrize(
    ("excerpt", "obj"),
    [
        ("my name is mk", "mk"),
        ("I use Kubernetes daily", "Kubernetes"),
        ("my favorite color is blue", "blue"),
        ("I live in Berlin", "Berlin"),
    ],
)
def test_legit_facts_still_accept(excerpt, obj):
    decision = validate_candidate(
        candidate(
            subject="I",
            predicate="likes",
            object_value=obj,
            confidence=0.95,
            source_excerpt=excerpt,
            explicit_user_assertion=True,
        )
    )
    assert decision.accepted is True, excerpt


def test_third_party_still_rejected_not_merely_pending():
    decision = validate_candidate(
        candidate(
            subject="Alexei",
            predicate="works at",
            object_value="Initech",
            confidence=1.0,
            source_excerpt="my brother Alexei works at Initech",
            explicit_user_assertion=True,
        )
    )
    assert decision.accepted is False


def test_projection_boundary_refuses_sensitive_rows():
    from app.graph_memory.neo4j_projection import (
        Neo4jProjection,
        Neo4jProjectionRefused,
        ProjectableItem,
    )

    class RecordingDriver:
        def session(self, **kwargs):
            raise AssertionError("driver must never be reached")

        def close(self):
            pass

    proj = Neo4jProjection(
        uri="bolt://x:7687", user="n", password="p", driver=RecordingDriver()
    )
    with pytest.raises(Neo4jProjectionRefused):
        proj.upsert_item(
            ProjectableItem(
                user_id=41,
                tenant_uuid="11111111-1111-4111-8111-111111111111",
                fingerprint="b" * 64,
                kind="fact",
                subject="I",
                predicate="suffers from",
                object_value="back pain",
                status="active",
            )
        )

def test_status_uses_cached_summary_when_current() -> None:
    repo = MagicMock()
    repo.get_tenant.return_value = tenant(graph_revision=5)
    repo.get_summary.return_value = {
        "summary": "You are caching tests.",
        "source_memory_ids": [],
        "graph_revision": 5,
        "generated_at": datetime.now(),
        "model": "gemma",
        "is_fallback": False,
    }
    repo.aggregate.return_value = {"next_expiry_at": None, "active": 1, "pending": 0, "expired": 0}
    
    svc = GraphMemoryService(repository=repo)
    status = svc.get_status(41)
    
    assert status.about_me.summary == "You are caching tests."
    assert status.about_me.status == "ready"

def test_status_uses_stale_summary_while_revalidating() -> None:
    repo = MagicMock()
    repo.get_tenant.return_value = tenant(graph_revision=6)
    repo.get_summary.return_value = {
        "summary": "You are stale but served.",
        "source_memory_ids": [],
        "graph_revision": 5,
        "generated_at": datetime.now(),
        "model": "gemma",
        "is_fallback": False,
    }
    repo.aggregate.return_value = {"next_expiry_at": None, "active": 1, "pending": 0, "expired": 0}
    
    svc = GraphMemoryService(repository=repo)
    status = svc.get_status(41)
    
    assert status.about_me.summary == "You are stale but served."
    assert status.about_me.status == "ready"

def test_status_falls_back_to_deterministic_when_empty() -> None:
    repo = MagicMock()
    repo.get_tenant.return_value = tenant(graph_revision=6)
    repo.get_summary.return_value = None
    repo.active_items_for_profile.return_value = [item(predicate="like", object_value="testing")]
    repo.aggregate.return_value = {"next_expiry_at": None, "active": 1, "pending": 0, "expired": 0}
    
    svc = GraphMemoryService(repository=repo)
    status = svc.get_status(41)
    
    assert status.about_me.summary == "You like testing."
    assert status.about_me.status == "ready"

@patch("app.graph_memory.worker.OllamaAboutMeSynthesizer")
@pytest.mark.asyncio
async def test_background_summarize_job_persists_synthesis_and_clears_queue(MockSynthesizer) -> None:
    mock_synth_instance = MagicMock()
    async def mock_synthesize(*args, **kwargs):
        return SimpleNamespace(
            summary="You're living in Hyderabad.",
            source_memory_ids=[uuid4()],
            model="llama3.1",
            is_fallback=False,
        )
    mock_synth_instance.synthesize = mock_synthesize
    MockSynthesizer.return_value = mock_synth_instance
    
    svc = MagicMock()
    svc.repository.get_tenant.return_value = tenant(graph_revision=5)
    svc.repository.get_summary.return_value = None
    svc.repository.active_items_for_profile.return_value = [
        item(predicate="live_in", object_value="Hyderabad")
    ]
    
    jobs = MagicMock()
    jobs.current_memory_model.return_value = (True, "llama3.1")
    
    worker = GraphMemoryWorker(service=svc, jobs=jobs)
    job = GraphMemoryJob(
        id=uuid4(),
        user_id=41,
        tenant_uuid=uuid4(),
        generation=3,
        job_type="summarize",
        source_chat_id=None,
        source_message_id=None,
        payload={},
        attempts=1,
        max_attempts=3,
        lease_owner="test",
        lease_expires_at=datetime.now()
    )
    
    lease_lost = asyncio.Event()
    await worker._summarize(job, lease_lost)
    
    svc.repository.store_summary.assert_called_once()
    call_args = svc.repository.store_summary.call_args[1]
    assert call_args["summary"] == "You're living in Hyderabad."
    assert call_args["is_fallback"] is False

@patch("app.graph_memory.worker.OllamaAboutMeSynthesizer")
@pytest.mark.asyncio
async def test_summarize_job_fails_open_to_deterministic_fallback(MockSynthesizer) -> None:
    mock_synth_instance = MagicMock()
    async def failing_synthesize(*args, **kwargs):
        raise ValueError("LLM unavailable")
    mock_synth_instance.synthesize = failing_synthesize
    MockSynthesizer.return_value = mock_synth_instance
    
    svc = MagicMock()
    svc.repository.get_tenant.return_value = tenant(graph_revision=5)
    svc.repository.get_summary.return_value = None
    svc.repository.active_items_for_profile.return_value = [item(predicate="like", object_value="fallbacks")]
    
    jobs = MagicMock()
    jobs.current_memory_model.return_value = (True, "llama3.1")
    
    worker = GraphMemoryWorker(service=svc, jobs=jobs)
    job = GraphMemoryJob(
        id=uuid4(),
        user_id=41,
        tenant_uuid=uuid4(),
        generation=3,
        job_type="summarize",
        source_chat_id=None,
        source_message_id=None,
        payload={},
        attempts=1,
        max_attempts=3,
        lease_owner="test",
        lease_expires_at=datetime.now()
    )
    
    lease_lost = asyncio.Event()
    await worker._summarize(job, lease_lost)
    
    svc.repository.store_summary.assert_called_once()
    call_args = svc.repository.store_summary.call_args[1]
    assert call_args["summary"] == "You like fallbacks."
    assert call_args["is_fallback"] is True
