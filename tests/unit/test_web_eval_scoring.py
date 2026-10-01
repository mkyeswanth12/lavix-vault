"""Unit tests for the eval harness scoring (pure functions, no live stack)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_web_eval", str(Path(__file__).resolve().parent.parent / "eval" / "run_web_eval.py")
)
assert _SPEC and _SPEC.loader
_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_module)


def test_score_retrieval_matches_domain():
    ok, domain = _module.score_retrieval(
        ["https://www.python.org/downloads/", "https://x.test/"], ["python.org"]
    )
    assert (ok, domain) == (True, "python.org")
    ok, _ = _module.score_retrieval(["https://x.test/"], ["python.org"])
    assert ok is False


def test_score_faithfulness_keywords():
    ok, problems = _module.score_faithfulness("Benzema won in 2022.", ["Benzema"], ["Mbapp"])
    assert ok and problems == []
    ok, problems = _module.score_faithfulness("Kylian won.", ["Benzema"], ["Kylian"])
    assert not ok and problems == ["missing:Benzema", "present:Kylian"]


def test_grade_bands():
    assert (_module.grade(95), _module.grade(80), _module.grade(65), _module.grade(40)) == (
        "A", "B", "C", "D",
    )
