"""Topic/context transition handling for the Web-RAG leg (turn classifier).

Pure-function coverage for `_classify_turn_context` and its consumers.
Histories are SimpleNamespace (role/content) triples with the current
query last, mirroring `_followup_search_query` conventions.
"""

from types import SimpleNamespace as NS

from agent_runtime.cuga_adapter import (
    _build_planner_context,
    _classify_turn_context,
    _compose_clarification_question,
    _filter_context_messages,
    _followup_search_query,
    _has_own_entity,
    _inherit_followup_entity,
)


def _msg(role, content):
    return NS(role=role, content=content)


def test_proper_new_topic_outranks_history():
    history = [
        _msg("user", "Tell me about famous shipwrecks near France"),
        _msg("assistant", "France has several well-charted wreck sites off its coast."),
        _msg("user", "Where did the Titanic sink?"),
    ]
    verdict = _classify_turn_context("Where did the Titanic sink?", history, None)
    assert verdict is not None
    assert verdict.classification == "NEW_TOPIC"
    assert verdict.active_topic == "Titanic"


def test_competing_entities_are_ambiguous():
    history = [
        _msg("user", "Tell me about India and France"),
        _msg("assistant", "India and France are both large economies."),
        _msg("user", "What is its capital?"),
    ]
    verdict = _classify_turn_context("What is its capital?", history, None)
    assert verdict is not None
    assert verdict.classification == "AMBIGUOUS"
    assert {"India", "France"} <= set(verdict.candidates)


def test_gendered_pronoun_rejects_lowercase_nouns():
    # Casual lowercase history: no capitalized candidate exists, so the
    # masculine pronoun binds nothing — lowercase nouns ("ship") are
    # rejected, never promoted. (With sentence-initial capitals such as
    # "Tell me ...", the capitalized token counts as the single
    # candidate instead; ship still never binds.)
    history = [
        _msg("user", "saw the old ship yesterday"),
        _msg("assistant", "nice, what did you think of it"),
        _msg("user", "He is taller"),
    ]
    verdict = _classify_turn_context("He is taller", history, None)
    assert verdict is not None
    assert verdict.classification == "AMBIGUOUS"
    assert verdict.candidates == ()


def test_demonstrative_new_subject_short_circuits_legacy():
    history = [
        _msg("user", "Tell me about France"),
        _msg("assistant", "France is a country in Europe."),
        _msg("user", "Tell me about this ship"),
    ]
    resolved, method = _followup_search_query("Tell me about this ship", history)
    assert method == "demonstrative"
    assert resolved == "this ship"
    # Mid-sentence demonstratives capture the approved trailing window
    # (verb-inclusive): deterministic, still retrieves, pinned as-is.
    history_mid = [
        _msg("user", "Tell me about France"),
        _msg("assistant", "France is a country in Europe."),
        _msg("user", "Where did this ship sink?"),
    ]
    resolved_mid, method_mid = _followup_search_query(
        "Where did this ship sink?", history_mid
    )
    assert method_mid == "demonstrative"
    assert resolved_mid == "this ship sink"


def test_deterministic_clarification_names_candidates():
    question = _compose_clarification_question(
        "What is its capital?", ["India", "France"]
    )
    assert question == "Who do you mean \u2014 India or France?"


def test_filter_keeps_topic_turns_and_current():
    messages = [
        _msg("user", "Tell me about France"),
        _msg("assistant", "France is a country in Europe."),
        _msg("user", "Tell me about the Titanic"),
        _msg("assistant", "The Titanic sank in 1912."),
        _msg("user", "What is the capital?"),
    ]
    kept = _filter_context_messages(messages, ("France",))
    assert len(kept) == 3
    assert kept[-1] is messages[-1]
    assert "Titanic" not in " ".join(
        str(getattr(message, "content", "")) for message in kept
    )


def test_planner_topic_is_demonstrative_phrase():
    history = [
        _msg("user", "Tell me about France"),
        _msg("assistant", "France is a country in Europe."),
        _msg("user", "Tell me about this ship"),
    ]
    verdict = _classify_turn_context("Tell me about this ship", history, None)
    assert verdict is not None and verdict.reason == "demonstrative-new"
    topic, _entities = _build_planner_context(
        "Tell me about this ship", messages=history, turn_context=verdict
    )
    assert topic == "this ship"


def test_single_turn_never_classifies():
    assert _classify_turn_context("hello", [_msg("user", "hello")], None) is None
    assert _classify_turn_context("hello", [], None) is None


def test_gendered_pronoun_names_no_entity_and_inherits_nothing():
    ship_history = [
        _msg("user", "Tell me about the old ship"),
        _msg("assistant", "The ship sank in 1912 after hitting an iceberg."),
        _msg("user", "He is taller"),
    ]
    assert _has_own_entity("He is taller") is False
    assert _inherit_followup_entity("He is taller", ship_history) is None
    assert _has_own_entity("Which is cheaper?") is False


def _kohli_history():
    return [
        _msg("user", "Tell me about Virat Kohli."),
        _msg("assistant", "Kohli is a cricketer."),
    ]


def test_kohli_pronoun_resolves_followup():
    history = [
        *_kohli_history(),
        _msg("user", "How many centuries has he scored?"),
    ]
    verdict = _classify_turn_context("How many centuries has he scored?", history, None)
    assert verdict is not None
    assert verdict.classification == "FOLLOW_UP"
    assert verdict.active_topic == "Kohli"


def test_kohli_ship_pronoun_is_ambiguous():
    history = [
        *_kohli_history(),
        _msg("user", "Tell me about the Titanic."),
        _msg("assistant", "The Titanic was British."),
        _msg("user", "How many centuries has he scored?"),
    ]
    verdict = _classify_turn_context("How many centuries has he scored?", history, None)
    assert verdict is not None
    assert verdict.classification == "AMBIGUOUS"
    assert len(verdict.candidates) >= 2
    assert "british" in [str(c).casefold() for c in verdict.candidates]


def test_ship_it_resolves_followup():
    history = [
        *_kohli_history(),
        _msg("user", "Tell me about the old ship."),
        _msg("assistant", "old ship."),
        _msg("user", "What caused it?"),
    ]
    verdict = _classify_turn_context("What caused it?", history, None)
    assert verdict is not None
    assert verdict.classification == "FOLLOW_UP"
    assert verdict.active_topic == "ship"


def test_why_never_becomes_topic():
    history = [
        _msg("user", "Tell me about the Titanic"),
        _msg("assistant", "The Titanic was a large ship that hit an iceberg."),
        _msg("user", "Why did the ship sink?"),
    ]
    verdict = _classify_turn_context("Why did the ship sink?", history, None)
    assert verdict is not None
    assert "why" not in verdict.active_topic.casefold()
    assert "why" not in [str(c).casefold() for c in verdict.candidates]
    assert "ship" in [str(e).casefold() for e in verdict.active_entities]


def test_why_never_becomes_topic_docker():
    history = [
        _msg("user", "What is Docker?"),
        _msg("assistant", "Docker is a platform."),
        _msg("user", "Why is it popular?"),
    ]
    verdict = _classify_turn_context("Why is it popular?", history, None)
    assert verdict is not None
    assert "why" not in verdict.active_topic.casefold()
    assert "why" not in [str(c).casefold() for c in verdict.candidates]


def test_tell_never_becomes_entity_france_is_topic():
    history = [
        _msg("user", "Tell me about India."),
        _msg("assistant", "India is in South Asia."),
        _msg("user", "Tell me about France."),
    ]
    verdict = _classify_turn_context("Tell me about France.", history, None)
    assert verdict is not None
    assert verdict.classification == "NEW_TOPIC"
    assert verdict.active_topic == "France"
    assert "tell" not in [str(c).casefold() for c in verdict.candidates]


def test_used_never_becomes_topic():
    history = [
        _msg("user", "What is Docker?"),
        _msg("assistant", "Docker is a platform."),
        _msg("user", "How is biryani traditionally prepared?"),
        _msg("assistant", "Layer rice and meat."),
        _msg("user", "What spices are commonly used?"),
    ]
    verdict = _classify_turn_context("What spices are commonly used?", history, None)
    assert verdict is not None
    assert verdict.active_topic == "spices"
    # Entities stay intact for retrieval shaping; only the head is fixed.
    assert "used" in [str(e).casefold() for e in verdict.active_entities]


def test_india_france_capital_is_ambiguous():
    history = [
        _msg("user", "Tell me about India."),
        _msg("assistant", "India is large."),
        _msg("user", "Tell me about France."),
        _msg("assistant", "France and Spain are neighbours."),
        _msg("user", "What is its capital?"),
    ]
    verdict = _classify_turn_context("What is its capital?", history, None)
    assert verdict is not None
    assert verdict.classification == "AMBIGUOUS"
    assert len(verdict.candidates) >= 2
    question = _compose_clarification_question("What is its capital?", verdict.candidates)
    assert question == "Who do you mean \u2014 France or Spain?"
