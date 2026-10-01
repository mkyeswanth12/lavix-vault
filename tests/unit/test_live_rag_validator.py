from scripts.validate_live_rag import _evaluate


def _result(answer: str) -> dict:
    return {
        "answer": answer,
        "errors": [],
        "sources": [{"id": "V1", "filename": "qna copy.docx", "match_percentage": 42}],
        "followups": [
            "Which evidence most directly supports this answer?",
            "What context could change this conclusion?",
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def test_live_validator_requires_grounded_answer_terms_without_internal_citations() -> None:
    question = {
        "question": "Which local model is configured?",
        "expected_source_terms": ["qna copy"],
        "expected_answer_terms": ["llama3.1:8b", "Ollama"],
    }

    assert _evaluate(question, _result("The Ollama command uses llama3.1:8b.")) == []


def test_live_validator_rejects_raw_tool_payloads_and_placeholders() -> None:
    question = {
        "question": "Which vision backend is configured?",
        "expected_source_terms": [],
        "expected_answer_terms": ["Vertex AI Vision"],
    }
    result = _result('```json\n{"evidence": [{"provenance": {"block_id": "x"}}]}\n``` [V1]')
    result["followups"] = ["Relevant next question?", "Another useful question?"]

    failures = _evaluate(question, result)

    assert "answer exposes raw tool or provenance data" in failures
    assert "expected answer term is absent: Vertex AI Vision" in failures
    assert "follow-up suggestions are duplicated or placeholders" in failures
    assert "answer exposes internal evidence labels" in failures


def test_live_validator_rejects_internal_sources_and_long_or_echoed_followups() -> None:
    question = {
        "question": "What is the relationship between Russia and Japan?",
        "expected_source_terms": [],
        "expected_answer_terms": [],
    }
    result = _result("They have a complex relationship.")
    result["sources"][0]["provenance"] = {"page": 1}
    result["followups"] = [
        "What is the relationship between Russia and Japan?",
        "Which additional historical, political, economic, and diplomatic details matter most now?",
    ]

    failures = _evaluate(question, result)

    assert "source exposes internal retrieval fields" in failures
    assert "follow-up suggestions are too long" in failures
