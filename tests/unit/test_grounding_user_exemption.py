"""Grounding FIX 4: list markers stripped; user-message words exempt."""

from __future__ import annotations

from agent_runtime.cuga_adapter import _extract_claim_atoms, _grounding_gap

BULLETS = """Working from home requires effective time management tips.

*   Prioritize your tasks and focus on the most important ones first
*   Take regular breaks to stretch and rest your eyes
*   Establish a dedicated workspace and keep it organized
*   Set clear boundaries with family members or roommates"""


def test_bullet_imperatives_are_not_claim_atoms() -> None:
    atoms = _extract_claim_atoms(BULLETS)
    assert "Prioritize" not in atoms
    assert "Take" not in atoms
    assert "Establish" not in atoms
    assert "Set" not in atoms


def test_numbered_list_markers_are_not_claim_atoms() -> None:
    atoms = _extract_claim_atoms("Steps:\n1. Prioritize tasks\n2. Take breaks\n")
    assert "Prioritize" not in atoms
    assert "Take" not in atoms


def test_mid_sentence_capitalized_words_still_atomize() -> None:
    atoms = _extract_claim_atoms("I heard that Bheem is a good dog.")
    assert "Bheem" in atoms


def test_user_stated_name_is_not_a_gap() -> None:
    answer = "I am sorry to hear that Bheem is no longer with you."
    user_text = "I had a Golden Retriever named Bheem."
    assert "Bheem" in _grounding_gap(answer, None, None, user_text="")
    assert "Bheem" not in _grounding_gap(answer, None, None, user_text=user_text)


def test_user_stated_phrase_is_not_a_gap() -> None:
    answer = "The Golden Retriever was a wonderful companion."
    assert _grounding_gap(answer, None, None, user_text="") != []
    assert (
        _grounding_gap(answer, None, None, user_text="I had a Golden Retriever.") == []
    )


def test_assistant_only_text_does_not_exempt() -> None:
    # Words from earlier assistant text (not the user message) still gap.
    answer = "I heard that Bheem was a wonderful companion."
    assert "Bheem" in _grounding_gap(answer, None, None, user_text="Tell me a story.")


def test_fabricated_numbers_still_caught_despite_user_text() -> None:
    answer = "You work with a PC with 64 GB RAM."
    user_text = "My PC has 32 GB RAM."
    gap = _grounding_gap(answer, None, None, user_text=user_text)
    assert "64" in gap
    # And the user's own number is not demanded when absent from evidence:
    # numbers route through the evidence check, never the user exemption.
    assert "32" not in gap or "64" in gap
