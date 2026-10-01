"""Deterministic fallback polish: verbs, schedule, hardware, affinity, variety."""

from __future__ import annotations

from app.graph_memory.service import deterministic_about_me
from tests.unit.test_about_me_portrait import mem


def _live_five() -> list:
    return [
        mem("has", "32 GB RAM, an 8-core CPU, and 4 TB storage including a 1 TB SSD", confidence=0.9),
        mem("use", "Linux", confidence=1.0),
        mem("work from 10:30 AM to 6 PM on weekdays", "10:30 AM to 6 PM", confidence=0.9),
        mem("love", "dogs", confidence=1.0),
        mem("had", "a Golden Retriever named Bheem", confidence=1.0),
    ]


def _live_pending() -> list:
    return [
        mem("enjoy", "long bike trips", confidence=0.8),
        mem("love", "dogs", confidence=0.7, kind="preference"),
        mem("prefer", "spicy food", confidence=0.8, kind="preference"),
        mem("make", "butter chicken at home", confidence=0.7),
    ]


def _starts(sentences: list[str]) -> list[str]:
    return [sentence.split(" ", 1)[0].rstrip(",") for sentence in sentences]


def test_five_active_exact_text() -> None:
    summary = deterministic_about_me(_live_five(), graph_revision=1).summary
    assert summary == (
        "You use Linux on a 32 GB machine. "
        "You work weekdays 10:30 AM to 6 PM. "
        "In your time off, you love dogs and had a Golden Retriever named Bheem."
    )


def test_nine_active_exact_text() -> None:
    summary = deterministic_about_me(_live_five() + _live_pending(), graph_revision=1).summary
    assert summary == (
        "You use Linux on a 32 GB machine. "
        "You work weekdays 10:30 AM to 6 PM. "
        "In your time off, you love dogs and had a Golden Retriever named Bheem. "
        "Outside of that, you enjoy long bike trips and make butter chicken "
        "at home and prefer spicy food."
    )


def test_stored_verb_used_not_rephrased() -> None:
    summary = deterministic_about_me(
        [mem("use", "Linux"), mem("build", "side projects")], graph_revision=1
    ).summary
    assert "You use Linux and build side projects." in summary
    assert "work with" not in summary


def test_schedule_gets_own_sentence() -> None:
    summary = deterministic_about_me(
        [mem("work from 10:30 AM to 6 PM on weekdays", "10:30 AM to 6 PM")],
        graph_revision=1,
    ).summary
    assert summary == "You work weekdays 10:30 AM to 6 PM."


def test_single_spec_hardware_stays_full() -> None:
    summary = deterministic_about_me(
        [mem("has", "32gb of ram")], graph_revision=1
    ).summary
    assert summary == "You have 32gb of ram."


def test_multi_spec_hardware_condenses() -> None:
    summary = deterministic_about_me(
        [mem("use", "Linux"), mem("has", "32 GB RAM, 1 TB SSD")],
        graph_revision=1,
    ).summary
    assert "32 GB machine" in summary
    assert "SSD" not in summary
    assert "RAM," not in summary


def test_affinity_dedupes_general_fact() -> None:
    summary = deterministic_about_me(
        [mem("love", "dogs"), mem("had", "a Golden Retriever named Bheem")],
        graph_revision=1,
    ).summary
    assert summary == "You love dogs and had a Golden Retriever named Bheem."
    assert summary.count("dog") == 1


def test_no_three_consecutive_you_sentences() -> None:
    for rows in (_live_five(), _live_five() + _live_pending()):
        sentences = [
            part.strip()
            for part in deterministic_about_me(rows, graph_revision=1).summary.split(". ")
            if part.strip()
        ]
        run = 0
        for sentence in sentences:
            first = sentence.split(" ", 1)[0].rstrip(",")
            run = run + 1 if first in ("You", "Your", "You're", "You've") else 0
            assert run < 3, sentences


def test_word_budget() -> None:
    rows = _live_five() + _live_pending() + [
        mem("occupation", "systems engineer"),
        mem("live_in", "Hyderabad"),
        mem("hobby", "gaming"),
        mem("hobby", "cooking"),
        mem("favorite_color", "blue"),
    ]
    summary = deterministic_about_me(rows, graph_revision=1).summary
    assert len(summary.split()) <= 70
    assert len([part for part in summary.split(". ") if part.strip()]) <= 4
