"""Tests for compound query decomposition and reformulation."""

from agent_runtime.cuga_adapter import (
    _decompose_compound_query,
    _fix_question,
    _has_evidence_relevance,
    _reformulate_web_query,
    _shape_web_query,
)


class TestDecomposeCompoundQuery:
    def test_simple_query_not_decomposed(self):
        result = _decompose_compound_query("What is the capital of France?")
        assert result is None

    def test_nested_of_pattern(self):
        result = _decompose_compound_query(
            "What is the hometown of the director of the movie that won Best Picture?"
        )
        assert result is not None
        assert len(result) == 2
        # First sub-query resolves the inner dependency (movie that won)
        assert "movie" in result[0].lower()
        assert "won" in result[0].lower() or "best picture" in result[0].lower()
        # Second sub-query resolves the outer dependency (hometown of director)
        assert "hometown" in result[1].lower()

    def test_oscars_hometown_case(self):
        result = _decompose_compound_query(
            "What's the hometown of the director of the movie that won Best Picture at the most recent Oscars?"
        )
        assert result is not None
        assert len(result) == 2
        # Both should be valid question strings
        for q in result:
            assert q.endswith("?")
            assert q[0].isupper()

    def test_multiple_question_clauses(self):
        result = _decompose_compound_query(
            "What is the GDP of India and how does it compare to China?"
        )
        assert result is not None
        assert len(result) >= 2

    def test_nested_of_chain(self):
        result = _decompose_compound_query(
            "What is the CEO of the company that owns the brand that makes iPhone?"
        )
        assert result is not None
        assert len(result) >= 2

    def test_empty_query(self):
        assert _decompose_compound_query("") is None
        assert _decompose_compound_query("   ") is None


class TestFixQuestion:
    def test_adds_question_mark(self):
        assert _fix_question("What is the capital of France") == "What is the capital of France?"

    def test_capitalizes_first_letter(self):
        assert _fix_question("what is life?") == "What is life?"

    def test_collapses_whitespace(self):
        assert _fix_question("What  is   the    capital?") == "What is the capital?"

    def test_preserves_correct_format(self):
        assert _fix_question("What is the capital of France?") == "What is the capital of France?"


class TestReformulateWebQuery:
    def test_strips_filler_phrases(self):
        result = _reformulate_web_query("Could you please tell me about quantum computing?")
        assert "quantum computing" in result.lower()
        assert "could you" not in result.lower()
        assert "please" not in result.lower()
        assert "tell me about" not in result.lower()

    def test_strips_relative_clauses(self):
        result = _reformulate_web_query(
            "What is the hometown of the director that won Best Picture?"
        )
        assert "that won" not in result.lower()

    def test_keeps_core_entities(self):
        result = _reformulate_web_query(
            "Who is the CEO of Tesla who replaced Elon Musk?"
        )
        assert "tesla" in result.lower()

    def test_truncates_long_queries(self):
        long_query = " ".join(["word"] * 100)
        result = _reformulate_web_query(long_query)
        assert len(result) <= 240

    def test_empty_query(self):
        assert _reformulate_web_query("") == ""

    def test_simple_query_unchanged_or_simplified(self):
        result = _reformulate_web_query("capital of France")
        assert "capital" in result.lower()
        assert "france" in result.lower()


class TestShapeWebQuery:
    def test_extracts_last_question(self):
        result = _shape_web_query("Tell me about movies. What year was Titanic released?")
        assert "Titanic" in result

    def test_strips_filler(self):
        result = _shape_web_query("Could you please search for Python tutorials?")
        assert "python" in result.lower()
        assert "could you" not in result.lower()

    def test_max_chars_limit(self):
        long_text = "A" * 300
        result = _shape_web_query(long_text)
        assert len(result) <= 240

    def test_empty(self):
        assert _shape_web_query("") == ""


class TestHasEvidenceRelevance:
    """Test the pre-generation evidence relevance gate."""

    def test_empty_evidence_fails(self):
        assert _has_evidence_relevance("Oscars Best Picture", []) is False

    def test_oscar_evidence_passes(self):
        evidence = [
            {"title": "Ludwig Goransson wins Best Original Score at Oscars", "content": "Swedish composer won for Sinners at 98th Academy Awards."},
        ]
        assert _has_evidence_relevance(
            "What's the birthplace of the composer who won Best Original Score at the most recent Oscars?",
            evidence,
        ) is True

    def test_car_forum_evidence_fails(self):
        evidence = [
            {"title": "Mercedes W213 Discussion", "content": "Best score for the new E-Class? What do you think about the best package option?"},
            {"title": "BMW vs Mercedes comparison", "content": "The best driving score goes to BMW according to recent reviews."},
        ]
        # Car forums don't contain any query substantive words
        assert _has_evidence_relevance(
            "What's the birthplace of the composer who won Best Original Score at the most recent Oscars?",
            evidence,
        ) is False

    def test_single_word_match_fails(self):
        """Evidence with no matching query words should fail."""
        evidence = [
            {"title": "Car review scores", "content": "Best score in class for the new Mercedes."},
        ]
        assert _has_evidence_relevance(
            "What's the birthplace of the composer who won Best Original Score at the most recent Oscars?",
            evidence,
        ) is False

    def test_no_query_words_all_stop(self):
        """If all query words are stop words, should pass (nothing to anchor on)."""
        evidence = [{"title": "Some content", "content": "Some more content"}]
        assert _has_evidence_relevance("What is the?", evidence) is True


class TestReformulatePreservesAwardCategories:
    """Test that reformulation preserves award categories and event names."""

    def test_best_original_score_preserved(self):
        result = _reformulate_web_query(
            "What's the birthplace of the composer who won Best Original Score at the most recent Oscars?"
        )
        assert "Best Original Score" in result
        assert "Oscar" in result

    def test_best_picture_preserved(self):
        result = _reformulate_web_query(
            "What is the hometown of the director of the movie that won Best Picture at the most recent Oscars?"
        )
        assert "Best Picture" in result
        assert "Oscar" in result

    def test_best_director_preserved(self):
        result = _reformulate_web_query(
            "Who is the cinematographer of the film that won Best Director at the most recent Academy Awards?"
        )
        assert "Best Director" in result
        assert "Academy Award" in result

    def test_grammy_preserved(self):
        result = _reformulate_web_query(
            "What is the nationality of the singer who won Best Pop Vocal Album at the Grammys?"
        )
        assert "Grammy" in result
        assert "Best Pop Vocal Album" in result


class TestOscarsDecompositionRegression:
    """Regression: Oscars compound query decomposition."""

    def test_birthplace_of_composer_oscars(self):
        """The original regression case: birthplace of Oscar-winning composer."""
        result = _decompose_compound_query(
            "What's the birthplace of the composer who won Best Original Score at the most recent Oscars?"
        )
        assert result is not None
        assert len(result) == 2
        # Both should be valid question strings
        for q in result:
            assert q.endswith("?")
            assert q[0].isupper()
        # First sub-query should mention the award category
        assert "Best Original Score" in result[0]
        # Second sub-query should ask about birthplace
        assert "birthplace" in result[1].lower()

    def test_hometown_of_director_oscars(self):
        """Prior regression case still works."""
        result = _decompose_compound_query(
            "What's the hometown of the director of the movie that won Best Picture at the most recent Oscars?"
        )
        assert result is not None
        assert len(result) == 2
        assert "Best Picture" in result[0] or "Best Picture" in result[1]
