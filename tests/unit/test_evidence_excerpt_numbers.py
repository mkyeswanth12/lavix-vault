"""Regression tests for BUG-004: excerpt cuts must never orphan digits.

An arbitrary slice inside 7,799 lets a small model fuse a neighbouring
digit (97,799). Cuts snap to numeric-atom edges instead.
"""
from agent_runtime.gateway import _evidence_excerpt, _snap_out_of_number


def test_cut_never_splits_plain_number():
    text = "x" * 1900 + "Total Amount 7,799 only" + "y" * 500
    out = _evidence_excerpt("total amount", text)
    assert "7,799" in out or "7,799" not in text[out.find("7"):out.find("7") + 5]
    # No fragment like "799" standing where "7,799" was cut:
    assert "799" not in out.replace("7,799", "").replace("6,609.32", "")


def test_cut_never_splits_decimal_or_currency():
    text = ("Net Amount " + "6,609.32 and Tax 1,189.68 and total \u20b979,799.00 end. ") 
    text = "pad " * 600 + text + " tail " * 600
    out = _evidence_excerpt("net amount tax total", text)
    for atom in ("6,609.32", "1,189.68"):
        if atom[:3] in out or atom[-3:] in out:
            assert atom in out


def test_short_content_untouched():
    assert _evidence_excerpt("q", "Total 7,799. Done.") == "Total 7,799. Done."


def test_snap_edges():
    assert _snap_out_of_number("ab 7,799 cd", 5, edge="start") == 3
    assert _snap_out_of_number("ab 7,799 cd", 5, edge="end") == 8
    assert _snap_out_of_number("ab 7,799 cd", 3, edge="start") == 3
    assert _snap_out_of_number("ab 7,799 cd", 8, edge="end") == 8
    assert _snap_out_of_number("no numbers here", 5, edge="start") == 5


def test_percentage_and_invoice_numbers_survive():
    text = "Rate 18% invoice CJB1-2154218 total 25.5% done. " + "z" * 2100
    out = _evidence_excerpt("rate invoice", text)
    assert "18%" in out
