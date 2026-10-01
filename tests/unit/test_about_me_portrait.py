"""Natural-portrait About Me: validator, ranking, fallback, regen-skip."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.graph_memory.models import GraphMemoryItem
from app.graph_memory.service import deterministic_about_me
from app.graph_memory.synthesizer import (
    MAX_SYNTHESIS_INPUT_ITEMS,
    rank_synthesis_items,
    render_synthesis_input,
    validate_portrait,
)
from app.graph_memory.worker import GraphMemoryJob, GraphMemoryWorker


def mem(
    predicate: str,
    obj: str,
    *,
    kind: str = "fact",
    confidence: float = 0.9,
    subject: str = "I",
) -> GraphMemoryItem:
    now = datetime(2026, 7, 16)
    return GraphMemoryItem(
        id=uuid4(),
        kind=kind,  # type: ignore[arg-type]
        subject=subject,
        predicate=predicate,
        object_value=obj,
        confidence=confidence,
        status="active",  # type: ignore[arg-type]
        source_chat_id=uuid4(),
        source_message_id=uuid4(),
        source_excerpt=f"{subject} {predicate} {obj}",
        source_created_at=now,
        last_confirmed_at=now,
        expires_at=now,
        revision=1,
        projection_state="projected",  # type: ignore[arg-type]
        created_at=now,
        updated_at=now,
    )


def base_items() -> list[GraphMemoryItem]:
    return [
        mem("name", "Jordan", confidence=0.95),
        mem("live_in", "Hyderabad", confidence=0.9),
        mem("use", "Linux", confidence=0.85),
        mem("hobby", "long bike trips", confidence=0.8),
    ]


def test_good_portrait_accepted() -> None:
    summary = (
        "You're Jordan, living in Hyderabad. "
        "You work with Linux. "
        "Outside of that, you enjoy long bike trips."
    )
    assert validate_portrait(summary, base_items()) == summary


def test_fabricated_place_rejected() -> None:
    with pytest.raises(ValueError, match="invents entity|ungrounded word"):
        validate_portrait("You're Jordan, living in Bangalore.", base_items())


def test_fabricated_number_rejected() -> None:
    items = [mem("has", "PC with 32 GB RAM")]
    with pytest.raises(ValueError, match="invents number"):
        validate_portrait("You work with a 64 GB machine.", items)


def test_matching_number_accepted() -> None:
    items = [mem("has", "PC with 32 GB RAM")]
    summary = "You work with a PC with 32 GB RAM."
    assert validate_portrait(summary, items) == summary


def test_list_style_rejected() -> None:
    items = [mem("like", "dogs"), mem("love", "bikes"), mem("enjoy", "hikes")]
    with pytest.raises(ValueError, match="fact list"):
        validate_portrait("You like dogs. You love bikes. You enjoy hikes.", items)


def test_interest_cliche_rejected() -> None:
    with pytest.raises(ValueError, match="fact list"):
        validate_portrait("You have an interest in dogs.", [mem("like", "dogs")])


def test_bullet_layout_rejected() -> None:
    with pytest.raises(ValueError, match="bullet"):
        validate_portrait("- dogs", [mem("like", "dogs")])


def test_label_layout_rejected() -> None:
    with pytest.raises(ValueError, match="labels"):
        validate_portrait("Name: Jordan", [mem("name", "Jordan")])


def test_empty_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_portrait("   ", base_items())


def test_word_budget_rejected() -> None:
    with pytest.raises(ValueError, match="word budget"):
        validate_portrait("word " * 91, base_items())


def test_sentence_cap_rejected() -> None:
    summary = (
        "You're Jordan. "
        "Outside of that, you enjoy dogs. "
        "You live in Hyderabad. "
        "Meanwhile you work with Linux. "
        "You prefer metric units."
    )
    with pytest.raises(ValueError, match="too many sentences"):
        validate_portrait(summary, base_items())


def test_ranking_stability_first_then_confidence() -> None:
    noisy = mem("collect", "stamps", confidence=0.99)
    pref = mem("like", "dogs", kind="preference", confidence=0.6)
    identity = mem("name", "Sam", confidence=0.7)
    relation = mem("knows", "Anu", kind="relationship", confidence=0.6)
    entity = mem("has_dog", "Bheem", kind="entity", confidence=0.61)
    place = mem("live_in", "Hyd", confidence=0.9)

    ranked = rank_synthesis_items([noisy, pref, identity, relation, entity, place])

    assert [item.predicate for item in ranked] == [
        "name",
        "knows",
        "has_dog",
        "like",
        "live_in",
        "collect",
    ]


def test_ranking_confidence_tiebreak_and_stable_order() -> None:
    low = mem("collect", "a", confidence=0.8)
    high = mem("collect", "b", confidence=0.9)
    assert rank_synthesis_items([low, high])[0].object_value == "b"
    first = mem("collect", "x", confidence=0.8)
    second = mem("collect", "y", confidence=0.8)
    assert rank_synthesis_items([first, second])[0].object_value == "x"


def test_render_cap_drops_lowest_priority() -> None:
    rows = [mem("collect", f"thing-{index:02d}", confidence=0.5 + index / 100) for index in range(13)]
    keeper = mem("name", "Sam", confidence=0.5)
    rows.append(keeper)

    text, supplied = render_synthesis_input(rows)

    assert len(text.splitlines()) == MAX_SYNTHESIS_INPUT_ITEMS
    assert keeper.id in supplied
    dropped = min(rows, key=lambda row: row.confidence)
    assert dropped.id not in supplied


def test_deterministic_single_memory() -> None:
    profile = deterministic_about_me([mem("name", "Sam")], graph_revision=1)
    assert profile.summary == "You're Sam."


def test_deterministic_many_memories_condense() -> None:
    rows = [
        mem("name", "Jordan", confidence=0.99),
        mem("live_in", "Hyderabad", confidence=0.95),
        mem("occupation", "engineer", confidence=0.9),
        mem("use", "Linux", confidence=0.9),
        mem("use", "Docker", confidence=0.88),
        mem("hobby", "gaming", confidence=0.8),
        mem("hobby", "cooking", confidence=0.8),
        mem("like", "dogs", confidence=0.8),
        mem("prefer", "metric units", kind="preference", confidence=0.85),
        mem("has_pet", "Bheem", kind="entity", confidence=0.9),
        mem("favorite_color", "blue", confidence=0.7),
        mem("collect", "stamps", confidence=0.6),
    ]
    summary = deterministic_about_me(rows, graph_revision=3).summary
    assert len([part for part in summary.replace("?", ".").split(". ") if part.strip()]) <= 4
    assert "Jordan" in summary
    assert "stamps" not in summary


def _summarize_job() -> GraphMemoryJob:
    return GraphMemoryJob(
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
        lease_expires_at=datetime.now(),
    )


def _tenant_record(graph_revision: int) -> MagicMock:
    record = MagicMock()
    record.graph_revision = graph_revision
    return record


@patch("app.graph_memory.worker.OllamaAboutMeSynthesizer")
@pytest.mark.asyncio
async def test_summarize_skips_regeneration_when_current(MockSynthesizer: MagicMock) -> None:
    async def _must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("LLM must not be called for a current summary")

    MockSynthesizer.return_value.synthesize = _must_not_run
    svc = MagicMock()
    svc.repository.get_tenant.return_value = _tenant_record(5)
    svc.repository.get_summary.return_value = {
        "summary": "You're current.",
        "source_memory_ids": [],
        "graph_revision": 5,
        "generated_at": datetime.now(),
        "model": "llama3.1",
        "is_fallback": False,
    }
    svc.repository.active_items_for_profile.return_value = [mem("name", "Sam")]
    jobs = MagicMock()
    jobs.current_memory_model.return_value = (True, "llama3.1")

    await GraphMemoryWorker(service=svc, jobs=jobs)._summarize(_summarize_job(), asyncio.Event())

    svc.repository.store_summary.assert_not_called()


@patch("app.graph_memory.worker.OllamaAboutMeSynthesizer")
@pytest.mark.asyncio
async def test_fallback_stores_traceable_source_ids(MockSynthesizer: MagicMock) -> None:
    async def _boom(*args: object, **kwargs: object) -> object:
        raise ValueError("LLM unavailable")

    MockSynthesizer.return_value.synthesize = _boom
    first, second = mem("name", "Sam"), mem("live_in", "Hyderabad")
    svc = MagicMock()
    svc.repository.get_tenant.return_value = _tenant_record(5)
    svc.repository.get_summary.return_value = None
    svc.repository.active_items_for_profile.return_value = [first, second]
    jobs = MagicMock()
    jobs.current_memory_model.return_value = (True, "llama3.1")

    await GraphMemoryWorker(service=svc, jobs=jobs)._summarize(_summarize_job(), asyncio.Event())

    call = svc.repository.store_summary.call_args[1]
    assert call["is_fallback"] is True
    assert call["source_memory_ids"] == [first.id, second.id]


@patch("app.graph_memory.worker.OllamaAboutMeSynthesizer")
@pytest.mark.asyncio
async def test_portrait_rejection_falls_back_after_two_tries(MockSynthesizer: MagicMock) -> None:
    from types import SimpleNamespace

    calls = {"count": 0}

    async def _fabricate(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return SimpleNamespace(summary="You're Jordan, living in Bangalore.", source_memory_ids=[])

    MockSynthesizer.return_value.synthesize = _fabricate
    svc = MagicMock()
    svc.repository.get_tenant.return_value = _tenant_record(5)
    svc.repository.get_summary.return_value = None
    svc.repository.active_items_for_profile.return_value = [
        mem("name", "Jordan"),
        mem("live_in", "Hyderabad"),
    ]
    jobs = MagicMock()
    jobs.current_memory_model.return_value = (True, "llama3.1")

    await GraphMemoryWorker(service=svc, jobs=jobs)._summarize(_summarize_job(), asyncio.Event())

    assert calls["count"] == 2
    call = svc.repository.store_summary.call_args[1]
    assert call["is_fallback"] is True
    assert "Bangalore" not in call["summary"]


def test_fallback_source_ids_are_real_uuids() -> None:
    first = mem("name", "Sam")
    assert isinstance(first.id, UUID)


@patch("app.graph_memory.worker.OllamaAboutMeSynthesizer")
@pytest.mark.asyncio
async def test_portrait_model_override_is_used_when_configured(
    MockSynthesizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.graph_memory.worker import GraphMemoryWorker

    monkeypatch.setenv("ABOUT_ME_PORTRAIT_MODEL", "huge-model:q8")
    seen: dict[str, str] = {}

    async def _capture(model: str, items):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        seen["model"] = model
        return SimpleNamespace(
            summary="You work with Linux.", source_memory_ids=[items[0].id]
        )

    MockSynthesizer.return_value.synthesize = _capture
    svc = MagicMock()
    svc.repository.get_tenant.return_value = _tenant_record(9)
    svc.repository.get_summary.return_value = None
    svc.repository.active_items_for_profile.return_value = [
        mem("use", "Linux", confidence=0.9)
    ]
    jobs = MagicMock()
    jobs.current_memory_model.return_value = (True, "small-model")

    await GraphMemoryWorker(service=svc, jobs=jobs)._summarize(
        _summarize_job(), asyncio.Event()
    )

    assert seen["model"] == "huge-model:q8"
    assert svc.repository.store_summary.call_args[1]["model"] == "huge-model:q8"


@patch("app.graph_memory.worker.OllamaAboutMeSynthesizer")
@pytest.mark.asyncio
async def test_portrait_model_defaults_to_extraction_model(
    MockSynthesizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.graph_memory.worker import GraphMemoryWorker

    monkeypatch.delenv("ABOUT_ME_PORTRAIT_MODEL", raising=False)
    seen: dict[str, str] = {}

    async def _capture(model: str, items):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        seen["model"] = model
        return SimpleNamespace(
            summary="You work with Linux.", source_memory_ids=[items[0].id]
        )

    MockSynthesizer.return_value.synthesize = _capture
    svc = MagicMock()
    svc.repository.get_tenant.return_value = _tenant_record(9)
    svc.repository.get_summary.return_value = None
    svc.repository.active_items_for_profile.return_value = [
        mem("use", "Linux", confidence=0.9)
    ]
    jobs = MagicMock()
    jobs.current_memory_model.return_value = (True, "small-model")

    await GraphMemoryWorker(service=svc, jobs=jobs)._summarize(
        _summarize_job(), asyncio.Event()
    )

    assert seen["model"] == "small-model"

def test_tech_savvy_professional_rejected() -> None:
    items = base_items()
    with pytest.raises(ValueError, match="trait"):
        validate_portrait(
            "You're a tech-savvy professional with a powerful computer.",
            items,
        )


def test_value_language_rejected() -> None:
    items = base_items()
    with pytest.raises(ValueError, match="motive|trait"):
        validate_portrait("You're someone who values learning.", items)


def test_looking_to_expand_rejected() -> None:
    items = base_items()
    with pytest.raises(ValueError, match="motive"):
        validate_portrait(
            "You work with Linux, looking to expand your skills.", items
        )


def test_grounded_use_of_blocklisted_word_allowed() -> None:
    items = [mem("occupation", "medical professional")]
    summary = "You work as a medical professional."
    assert validate_portrait(summary, items) == summary


def test_concrete_facts_accepted() -> None:
    items = [
        mem("name", "Jordan", confidence=0.95),
        mem("live_in", "Hyderabad", confidence=0.9),
        mem("use", "Linux", confidence=0.85),
        mem("has", "32 GB RAM", confidence=0.9),
        mem("love", "dogs", confidence=0.8),
    ]
    summary = (
        "You're Jordan, living in Hyderabad. "
        "You work with Linux on a 32 GB machine. "
        "Outside of that, you love dogs."
    )
    assert validate_portrait(summary, items) == summary


def test_vague_portrait_rejected() -> None:
    items = base_items()
    with pytest.raises(ValueError, match="top-ranked facts|no source memory|ungrounded word"):
        validate_portrait("You're someone with varied interests.", items)


def test_unmapped_clause_rejected() -> None:
    items = [
        mem("use", "Linux", confidence=0.9),
        mem("love", "dogs", confidence=0.85),
        mem("hobby", "long bike trips", confidence=0.8),
    ]
    with pytest.raises(ValueError, match="no source memory|ungrounded word"):
        validate_portrait(
            "You work with Linux and love dogs. You do it.", items
        )


def test_fallback_never_fabricates() -> None:
    from app.graph_memory.service import deterministic_about_me
    from app.graph_memory.synthesizer import _reject_unsupported_inference, _source_texts

    rows = [
        mem("name", "Jordan", confidence=0.99),
        mem("live_in", "Hyderabad", confidence=0.95),
        mem("occupation", "engineer", confidence=0.9),
        mem("use", "Linux", confidence=0.9),
        mem("has", "32 GB RAM, an 8-core CPU", confidence=0.9),
        mem("hobby", "gaming", confidence=0.8),
        mem("love", "dogs", confidence=0.8),
        mem("has_pet", "Bheem", kind="entity", confidence=0.9),
        mem("prefer", "spicy food", kind="preference", confidence=0.8),
    ]
    profile = deterministic_about_me(rows, graph_revision=3)
    assert profile.status == "ready"
    # No invented traits/motives: the grouped composer only reuses
    # stored strings.
    _reject_unsupported_inference(profile.summary, _source_texts(rows))


def test_question_derived_memory_cannot_reach_portrait_input() -> None:
    """Turn-B chain: the question half never reaches extractor, excerpt,
    or portrait input."""
    from app.graph_memory.models import MemoryCandidate
    from app.graph_memory.synthesizer import render_synthesis_input
    from app.graph_memory.worker import attribute_span, split_declarative_spans

    message = (
        "I normally work from 10:30 AM to 6 PM on weekdays. "
        "How could I organize my evenings around work and personal projects?"
    )
    spans = split_declarative_spans(message)
    fed = "\n".join(spans)
    assert "?" not in fed
    # A model that can only quote the fed text cannot smuggle the
    # question into the stored row: attribution pins the span.
    candidate = MemoryCandidate(
        kind="fact",
        subject="I",
        predicate="work",
        object_value="10:30 AM to 6 PM on weekdays",
        confidence=0.9,
        source_excerpt=fed,
        source_role="user",
        explicit_user_assertion=True,
    )
    excerpt = attribute_span(candidate, spans, fed)
    assert "?" not in excerpt
    assert "organize" not in excerpt
    text, supplied = render_synthesis_input(
        [
            mem(
                "work",
                "10:30 AM to 6 PM on weekdays",
                confidence=0.9,
            )
        ]
    )
    assert "?" not in text
    assert len(supplied) == 1


def test_strip_code_fences() -> None:
    from app.graph_memory.synthesizer import _strip_code_fences

    assert _strip_code_fences('{"a": 1}') == '{"a": 1}'
    assert (
        _strip_code_fences('```json\n{"a": 1}\n```')
        == '{"a": 1}'
    )
    assert _strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_code_fences('  ```json\n{"a": 1}\n```  ') == '{"a": 1}'
