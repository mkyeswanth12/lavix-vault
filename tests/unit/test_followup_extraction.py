"""Tests for followup extraction — especially the plain-text fallback path."""

from agent_runtime.events import (
    _extract_trailing_question_followups,
    project_answer_text,
    split_answer_metadata,
)


class TestExtractTrailingQuestionFollowups:
    def test_empty_text(self):
        assert _extract_trailing_question_followups("") == []

    def test_no_questions(self):
        assert _extract_trailing_question_followups("The answer is 42.") == []

    def test_trailing_questions(self):
        text = "The Eiffel Tower was completed in 1889.\nWhat year was the Eiffel Tower completed?\nHow tall is it?"
        result = _extract_trailing_question_followups(text)
        assert len(result) == 2
        assert "What year" in result[0]
        assert "How tall" in result[1]

    def test_single_trailing_question(self):
        text = "The answer is 42.\nWhat is the meaning of life?"
        result = _extract_trailing_question_followups(text)
        assert len(result) == 1
        assert "meaning of life" in result[0]

    def test_inline_question_not_extracted(self):
        text = "The question is: what year was it completed? The answer is 1889."
        result = _extract_trailing_question_followups(text)
        # Inline question in the middle should NOT be extracted
        assert len(result) == 0

    def test_max_four_followups(self):
        lines = ["Answer."] + [f"Question {i}?" for i in range(10)]
        text = "\n".join(lines)
        result = _extract_trailing_question_followups(text)
        assert len(result) <= 4

    def test_deduplication(self):
        text = "Answer.\nWhat year?\nWhat year?"
        result = _extract_trailing_question_followups(text)
        assert len(result) == 1


class TestProjectAnswerTextStripsFollowupsTag:
    def test_strips_lavix_followups_tag(self):
        raw = "The answer is 42.\n\n<LAVIX_FOLLOWUPS>{\"followups\":[\"Q1\",\"Q2\"]}</LAVIX_FOLLOWUPS>"
        result = project_answer_text(raw)
        assert "LAVIX_FOLLOWUPS" not in result
        assert "followups" not in result.lower()
        assert "The answer is 42" in result

    def test_strips_tag_with_newline(self):
        raw = "Answer text.\n<LAVIX_FOLLOWUPS>{\"followups\":[]}</LAVIX_FOLLOWUPS>\n"
        result = project_answer_text(raw)
        assert "LAVIX_FOLLOWUPS" not in result

    def test_strips_malformed_tag(self):
        raw = "Answer.\n<LAVIX_FOLLOWUPS>some content"
        result = project_answer_text(raw)
        assert "LAVIX_FOLLOWUPS" not in result


class TestSplitAnswerMetadataFallback:
    def test_tag_present_extracts_followups(self):
        raw = 'The answer is 42.\n\n<LAVIX_FOLLOWUPS>{"followups":["What is life?","How does it work?"]}</LAVIX_FOLLOWUPS>'
        visible, followups = split_answer_metadata(raw)
        assert "LAVIX_FOLLOWUPS" not in visible
        assert len(followups) == 2
        assert "What is life?" in followups

    def test_no_tag_fallback_extracts_trailing_questions(self):
        raw = "The Eiffel Tower was completed in 1889.\nWhat year was the Eiffel Tower completed?\nHow tall is the Eiffel Tower?"
        visible, followups = split_answer_metadata(raw)
        assert len(followups) == 2
        # The trailing questions should be stripped from visible
        assert "What year" not in visible
        assert "How tall" not in visible
        assert "1889" in visible

    def test_no_tag_no_questions(self):
        raw = "The answer is simply 42."
        visible, followups = split_answer_metadata(raw)
        assert followups == []
        assert "42" in visible

    def test_only_closing_tag_legacy(self):
        raw = "Answer.</LAVIX_FOLLOWUPS>"
        visible, followups = split_answer_metadata(raw)
        assert followups == []
        assert "LAVIX_FOLLOWUPS" not in visible

    def test_regression_eiffel_tower_plain_text(self):
        """Regression: model emits followup questions as plain prose."""
        raw = (
            "The Eiffel Tower was completed in 1889.\n\n"
            "What year was the Eiffel Tower completed?\n"
            "How tall is the Eiffel Tower?\n"
            "Who designed the Eiffel Tower?"
        )
        visible, followups = split_answer_metadata(raw)
        # Should extract 3 followups from trailing question lines
        assert len(followups) == 3
        # Questions should be stripped from visible text
        assert "What year" not in visible
        assert "How tall" not in visible
        assert "Who designed" not in visible
        # The actual answer should remain
        assert "1889" in visible
