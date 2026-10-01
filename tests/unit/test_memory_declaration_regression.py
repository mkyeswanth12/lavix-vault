"""Regression tests for BUG-001 (silent memory drop) and BUG-002 (path-dependent declarations).

Locks in:
- deterministic identity facts outrank the model's explicit_user_assertion boolean;
- widened declaration leads ("please call me", "my name's");
- safety gates (secrets/questions/uncertainty/consent bands) unchanged.
"""
from __future__ import annotations

import pytest

from agent_runtime.cuga_adapter import (
    _is_personal_declaration,
    _personal_declaration_acknowledgment,
)
from app.graph_memory.models import MemoryCandidate
from app.graph_memory.validation import match_identity_fact, validate_candidate


def identity_candidate(**overrides):
    values = {
        "kind": "fact",
        "subject": "I",
        "predicate": "name",
        "object_value": "QAT",
        "confidence": 0.95,
        "source_excerpt": "My name is QAT",
        "explicit_user_assertion": True,
    }
    values.update(overrides)
    return MemoryCandidate(**values)


def test_identity_rescue_outranks_false_boolean():
    """BUG-001: 3B sets explicit_user_assertion=False on a valid declaration."""
    decision = validate_candidate(identity_candidate(explicit_user_assertion=False))
    assert decision.accepted is True
    assert decision.status == "active"
    assert decision.identity_key == ("fact", "i", "name")


def test_identity_rescue_requires_value_agreement():
    """A disagreeing model value must not be laundered into an identity fact."""
    decision = validate_candidate(
        identity_candidate(object_value="Someone Else", explicit_user_assertion=False)
    )
    assert decision.accepted is False
    assert decision.reason == "not_explicit_user_assertion"


def test_identity_match_is_deterministic_without_model():
    assert match_identity_fact("My name is QAT").value == "QAT"
    assert match_identity_fact("Please call me QAT").value == "QAT"
    assert match_identity_fact("My name's QAT").value == "QAT"
    assert match_identity_fact("What is my name?") is None
    assert match_identity_fact("Maybe my name is QAT") is None


def test_safety_gates_still_fire_first():
    assert (
        validate_candidate(
            identity_candidate(
                source_excerpt="My password is hunter2-hunter2",
                object_value="hunter2-hunter2",
            )
        ).reason
        == "sensitive_data"
    )
    assert (
        validate_candidate(
            identity_candidate(source_excerpt="What is my name?")
        ).reason
        == "question"
    )
    assert (
        validate_candidate(
            identity_candidate(source_excerpt="Maybe my name is QAT")
        ).reason
        == "uncertain_statement"
    )


def test_confidence_bands_unchanged_for_non_identity():
    low = validate_candidate(
        identity_candidate(
            kind="preference",
            predicate="prefers",
            object_value="dark mode",
            source_excerpt="I prefer dark mode.",
            confidence=0.7,
        )
    )
    assert low.accepted is True and low.status == "pending"
    too_low = validate_candidate(
        identity_candidate(
            kind="preference",
            predicate="prefers",
            object_value="dark mode",
            source_excerpt="I prefer dark mode.",
            confidence=0.2,
        )
    )
    assert too_low.accepted is False and too_low.reason == "confidence_too_low"


@pytest.mark.parametrize(
    "text",
    [
        "Please call me QAT",
        "please call me QAT. I work on Lavix QA testing.",
        "My name's QAT",
        "My name is QAT",
        "Call me QAT",
    ],
)
def test_declaration_leads_cover_polite_forms(text):
    """BUG-002: every equivalent declaration shape classifies identically."""
    assert _is_personal_declaration(text) is True
    ack = _personal_declaration_acknowledgment(text, memory_opted_in=True)
    assert ack is not None and "QAT" in ack
    assert "try to save" in ack
    assert "I'll remember you as" not in ack


def test_declaration_ack_never_overpromises():
    """BUG-001: the ack must not claim persistence the worker may not deliver."""
    ack = _personal_declaration_acknowledgment("My name is QAT", memory_opted_in=True)
    assert "try to save" in ack
    off = _personal_declaration_acknowledgment("My name is QAT", memory_opted_in=False)
    assert "memory is currently off" in off
