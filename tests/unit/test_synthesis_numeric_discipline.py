"""Regression tests for BUG-005 (vault/web numeric roles) and BUG-009 (atom hold).

- The synthesis prompt binds numbers to their SOURCE block.
- NumericAtomHold keeps digits out of streamed deltas until finalize().
- _strict_number_refusal escalates only load-bearing specifics.
"""
from agent_runtime.cuga_adapter import (
    AUTHORITATIVE_SYNTHESIS_PROMPT,
    _is_date_atom,
    _seeks_number_or_date,
    _strict_number_refusal,
)
from agent_runtime.events import NumericAtomHold


def test_prompt_binds_numbers_to_source_blocks():
    assert "Numbers stay with their SOURCE block" in AUTHORITATIVE_SYNTHESIS_PROMPT
    assert "never restated as tax" in AUTHORITATIVE_SYNTHESIS_PROMPT
    assert "note the conflict" in AUTHORITATIVE_SYNTHESIS_PROMPT


def test_hold_keeps_atoms_out_of_deltas():
    hold = NumericAtomHold()
    assert hold.feed("Narendra Modi took oath on ") == "Narendra Modi took oath on "
    assert hold.feed("June 9, 2026 in Delhi.") == ""
    assert hold.finalize() == "June 9, 2026 in Delhi."


def test_hold_never_splits_partial_number():
    hold = NumericAtomHold()
    assert hold.feed("Total 97") == "Total "
    assert hold.feed(",799 confirmed.") == ""
    assert hold.finalize() == "97,799 confirmed."


def test_hold_passes_clean_text_untouched():
    hold = NumericAtomHold()
    assert hold.feed("Hello! How can I help you today?") == "Hello! How can I help you today?"
    assert hold.finalize() == ""


def test_strict_requires_number_gap():
    assert _strict_number_refusal([], scoped=True, question="q") is False
    assert _strict_number_refusal(["Modi"], scoped=True, question="q") is False


def test_strict_fires_for_scoped_and_seeking_and_dates():
    assert _strict_number_refusal(["7799"], scoped=True, question="summarize") is True
    assert _strict_number_refusal(["7799"], scoped=False, question="how much tax?") is True
    assert _strict_number_refusal(["2026-06-09"], scoped=False, question="who is PM?") is True


def test_strict_spares_peripheral_quantities():
    assert _strict_number_refusal(["three"], scoped=False, question="who is PM?") is False


def test_seeking_and_date_shapes():
    assert _seeks_number_or_date("When did he take oath?") is True
    assert _seeks_number_or_date("What is the total?") is True
    assert _seeks_number_or_date("Who is the PM?") is False
    assert _is_date_atom("2026-06-09") is True
    assert _is_date_atom("June 9, 2026") is True
    assert _is_date_atom("7799") is False


def test_name_numbers_do_not_pair():
    from agent_runtime.cuga_adapter import _answer_number_pairs

    pairs = _answer_number_pairs("Qatar National Vision 2030 aims for growth.")
    assert all("2030" not in number for _, number in pairs)
    pairs = _answer_number_pairs("Invoice CJB1-2154218 totals 7799.")
    assert all("2154218" not in number for _, number in pairs)
    assert any("7799" in number for _, number in pairs)


def test_small_values_still_pair():
    from agent_runtime.cuga_adapter import _answer_number_pairs

    pairs = _answer_number_pairs("Team A scored 95 points.")
    assert ("team a", "95") in pairs


def test_name_number_mismatch_no_longer_refuses():
    from agent_runtime.cuga_adapter import _pairing_mismatches

    draft = "Qatar National Vision 2030 aims for 4 percent growth."
    evidence = ["Qatar National Vision 2030 report: economic diversification, 2 pillars."]
    assert _pairing_mismatches(draft, evidence) == []


def test_true_contradiction_still_refuses():
    from agent_runtime.cuga_adapter import _pairing_mismatches

    draft = "Team A scored 120 points."
    evidence = ["Team A scored 95 points in the final."]
    assert _pairing_mismatches(draft, evidence) != []
