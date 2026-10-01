"""Unit tests for the web relevance gate (distinct-word matching)."""

import pytest

from app.agent.gateway import (
    _distinct_substantive_matches,
    _is_generic_content,
    _is_promotional,
    _web_result_sort_key,
)


def test_repeated_single_word_cannot_satisfy_gate():
    # Regression: the duplicated entity string
    # "Best Picture Oscar Best Picture Oscar 2025" used to pass Best Buy
    # because "best" matched twice. Distinct counting closes that hole.
    query_words = ["best", "picture", "oscar", "best", "picture", "oscar", "2025"]
    hay = "best buy shop best tech deals".casefold()
    assert _distinct_substantive_matches(query_words, hay) == 1


def test_two_distinct_words_pass():
    assert _distinct_substantive_matches(["lehman", "brothers"], "lehman brothers bankruptcy filing") == 2


def test_singular_plural_collapse_to_one():
    assert _distinct_substantive_matches(["stock", "stocks"], "stock market prices") == 1
    assert _distinct_substantive_matches(["stocks"], "stock market prices") == 1


def test_empty_haystack_matches_nothing():
    assert _distinct_substantive_matches(["lehman", "brothers"], "") == 0


def test_short_words_match_literally():
    assert _distinct_substantive_matches(["uk", "pm"], "uk prime minister") == 1


@pytest.mark.parametrize(
    ("title", "url", "expected"),
    [
        ("Lehman plans to end bankruptcy", "https://www.prnewswire.com/news/x", True),
        ("Earnings call transcript", "https://globenewswire.com/news/y", True),
        ("Q3 results", "https://investor.example.com/press-release/q3", True),
        ("Visit our newsroom today", "https://example.com/newsroom/q3", True),
        ("Microsoft announces partnership with Adobe", "https://news.example.com/p", True),
        ("Two firms partners with rivals", "https://example.com/a", True),
        ("Stock market news today", "https://example.com/m", False),
        ("Lehman Brothers - Wikipedia", "https://en.wikipedia.org/wiki/Lehman_Brothers", False),
        ("Partnership explained: a history", "https://example.com/analysis", False),
    ],
)
def test_promotional_flags_only_wire_and_announcement_shapes(title, url, expected):
    assert _is_promotional(title, url) is expected


def test_result_sort_puts_relevant_nonpromo_first_and_keeps_stability():
    items = [
        {"id": "irrelevant-promo", "relevant": False, "promotional": True},
        {"id": "relevant-promo", "relevant": True, "promotional": True},
        {"id": "relevant-plain-b", "relevant": True},
        {"id": "irrelevant-plain", "relevant": False},
        {"id": "relevant-plain-a", "relevant": True},
        {"id": "legacy-no-flags"},
    ]
    ordered = sorted(items, key=_web_result_sort_key)
    assert [item["id"] for item in ordered] == [
        "relevant-plain-b",
        "relevant-plain-a",
        "legacy-no-flags",
        "relevant-promo",
        "irrelevant-plain",
        "irrelevant-promo",
    ]


@pytest.mark.parametrize(
    ("title", "content", "expected"),
    [
        # Dictionary-definition shape: reference title + defining body.
        (
            "BEST Definition & Meaning - Merriam-Webster",
            "The meaning of BEST is excelling all others. How to use best in a sentence.",
            True,
        ),
        (
            "BEST | English meaning - Cambridge Dictionary",
            "BEST definition: 1. of the highest quality.",
            True,
        ),
        # Two or more nav/footer chrome phrases: site chrome, not prose.
        (
            "Indian people - Wikipedia",
            "Indian people - Wikipedia. Jump to content. Main menu. move to sidebarhide.",
            True,
        ),
        # Single chrome phrase alone is not enough.
        (
            "Some Article",
            "Sign in to read more about this topic in depth and detail.",
            False,
        ),
        # Topical snippets never match: no topical words are listed.
        (
            "Lehman Brothers - Wikipedia",
            "Lehman Brothers was an American global financial services firm.",
            False,
        ),
        (
            "Stock Market Prices - Google Finance",
            "Real-time market quotes, international stock information.",
            False,
        ),
        ("", "", False),
    ],
)
def test_generic_content_flags_boilerplate_and_definitions_only(title, content, expected):
    assert _is_generic_content(title, content) is expected


def test_result_sort_demotes_generic_within_relevant_group():
    items = [
        {"id": "relevant-generic", "relevant": True, "generic": True},
        {"id": "relevant-plain", "relevant": True},
        {"id": "relevant-promo-generic", "relevant": True, "promotional": True, "generic": True},
    ]
    ordered = sorted(items, key=_web_result_sort_key)
    assert [item["id"] for item in ordered] == [
        "relevant-plain",
        "relevant-generic",
        "relevant-promo-generic",
    ]


def test_whats_does_not_match_whatsapp():
    # Regression ("whats the price of 512?"): stemmed "whats"->"what" plus
    # substring "whats" ⊂ "whatsapp" donated a free overlap. Token matching
    # plus stemmed-stop filtering leave only the real "512" overlap.
    from app.agent.gateway import _distinct_substantive_matches

    stop = frozenset({"what"})
    hay = "whatsapp - wikipedia 100 mb to 2 gb, and the maximum group size increased to 512 members"
    assert _distinct_substantive_matches(["whats", "price", "512"], hay, stopwords=stop) == 1
    assert _distinct_substantive_matches(["whats", "price"], hay, stopwords=stop) == 0


def test_stemmed_stopwords_contribute_nothing():
    from app.agent.gateway import _distinct_substantive_matches

    stop = frozenset({"about"})
    assert _distinct_substantive_matches(["abouts", "phones"], "about phones", stopwords=stop) == 1


def test_short_and_vowel_stems_match_literally_only():
    from app.agent.gateway import _distinct_substantive_matches

    assert _distinct_substantive_matches(["thats"], "that thing") == 0
    # Symmetric plural fallback is preserved: singular query finds plural hay.
    assert _distinct_substantive_matches(["iphone"], "iphones review") == 1
    assert _distinct_substantive_matches(["stocks"], "stock market prices") == 1
