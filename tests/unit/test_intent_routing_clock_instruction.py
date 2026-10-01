"""Regression tests for BUG-006 (clock intent) and BUG-010 (instructions)."""
import pytest

from agent_runtime.cuga_adapter import (
    _EVIDENCE_EXEMPT,
    _EVIDENCE_LABEL,
    _EVIDENCE_REFUSE,
    _evidence_requirement,
    _has_explicit_datetime_intent,
    _is_instruction_following_request,
)


def req(query, **over):
    base = {"identity_background": False, "needs_web": False,
            "has_explicit_scope": False, "is_personal": False}
    base.update(over)
    return _evidence_requirement(query, **base)


@pytest.mark.parametrize("text", [
    "what is the time",
    "what is the time right now",
    "tell me the time",
    "What time is it?",
    "What's the current time?",
    "Current time?",
    "Time now?",
])
def test_clock_variants_all_exempt(text):
    assert _has_explicit_datetime_intent(text) is True
    assert req(text) == _EVIDENCE_EXEMPT
    # Even when recency wording sets needs_web, clock intent wins.
    assert req(text, needs_web=True) == _EVIDENCE_EXEMPT


def test_clock_prose_stays_factual():
    assert _has_explicit_datetime_intent("current time complexity") is False
    assert req("current time complexity") == _EVIDENCE_LABEL


@pytest.mark.parametrize("text", [
    "Reply with exactly the word SWITCHED and nothing else",
    "Reply exactly: Hello",
    "Rewrite this sentence",
    "Summarise this supplied text",
    "Give me three words",
])
def test_instructions_exempt_without_entities(text):
    assert _is_instruction_following_request(text) is True
    assert req(text) == _EVIDENCE_EXEMPT


@pytest.mark.parametrize("text", [
    "Summarise the iPhone 17 launch event",
    "Reply with exactly what the Prime Minister said",
    "What is the time right now in Paris?",
])
def test_entity_instructions_stay_evidence_bound(text):
    assert req(text) != _EVIDENCE_EXEMPT


def test_scoped_instruction_uses_scope_not_exemption():
    assert req("Rewrite this sentence", has_explicit_scope=True) == _EVIDENCE_REFUSE
