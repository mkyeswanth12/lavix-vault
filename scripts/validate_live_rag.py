#!/usr/bin/env python3
"""Run repeatable live RAG questions and write machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_PLACEHOLDER_FOLLOWUPS = {
    "relevant next question?",
    "another useful question?",
    "what would you like to know next?",
}
_TOOL_LEAK_MARKERS = (
    '"evidence":',
    '"provenance":',
    '"block_id":',
    "execution output:",
    "result = await search_",
    "lavix_followups",
    "lavix trusted run scope",
    "lavix trusted answer contract",
)
_PUBLIC_SOURCE_FIELDS = {
    "id",
    "kind",
    "file_id",
    "filename",
    "mime_type",
    "title",
    "url",
    "match_percentage",
}


def _headers() -> dict[str, str]:
    token = os.environ.get("LAVIX_ACCESS_TOKEN", "").strip()
    if not token:
        raise SystemExit("LAVIX_ACCESS_TOKEN is required")
    return {"Authorization": f"Bearer {token}"}


def _request_json(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, path, **kwargs)
    if not response.is_success:
        raise RuntimeError(f"{method} {path} returned HTTP {response.status_code}: {response.text[:500]}")
    return response.json()


def _chat(client: httpx.Client, body: dict[str, Any]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    started = time.monotonic()
    done = False
    with client.stream("POST", "/ai/chat", json=body, timeout=420) as response:
        if not response.is_success:
            response.read()
            raise RuntimeError(f"chat returned HTTP {response.status_code}: {response.text[:500]}")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                done = True
                break
            value = json.loads(raw)
            if isinstance(value, dict):
                events.append(value)
    if not done:
        raise RuntimeError("chat stream ended without [DONE]")

    answer = "".join(
        str(event.get("content") or "") for event in events if event.get("type") == "token"
    ).strip()
    source_events = [event for event in events if event.get("type") == "sources"]
    followup_events = [event for event in events if event.get("type") == "followups"]
    usage_events = [event for event in events if event.get("type") == "usage"]
    return {
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "answer": answer,
        "sources": list(source_events[-1].get("sources") or []) if source_events else [],
        "followups": list(followup_events[-1].get("questions") or []) if followup_events else [],
        "usage": usage_events[-1] if usage_events else None,
        "statuses": [
            {"step": event.get("step"), "detail": event.get("detail")}
            for event in events
            if event.get("type") == "status"
        ],
        "errors": [event for event in events if event.get("type") == "error"],
        "event_types": [str(event.get("type")) for event in events],
    }


def _matches(filename: str, patterns: list[str]) -> bool:
    folded = filename.casefold()
    return any(pattern.casefold() in folded for pattern in patterns)


def _evaluate(question: dict[str, Any], result: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if result["errors"]:
        failures.append("chat emitted an error")
    if not result["answer"]:
        failures.append("answer is empty")
    folded_answer = str(result["answer"]).casefold()
    if any(marker in folded_answer for marker in _TOOL_LEAK_MARKERS):
        failures.append("answer exposes raw tool or provenance data")
    if re.search(r"\[[VW]\d+\]", result["answer"], re.IGNORECASE) or re.search(
        r"^\s*Evidence\s*:", result["answer"], re.IGNORECASE | re.MULTILINE
    ):
        failures.append("answer exposes internal evidence labels")
    for term in question.get("expected_answer_terms") or []:
        if str(term).casefold() not in folded_answer:
            failures.append(f"expected answer term is absent: {term}")
    expected = [str(value) for value in question.get("expected_source_terms") or []]
    filenames = [str(source.get("filename") or source.get("title") or "") for source in result["sources"]]
    if expected and not any(_matches(filename, expected) for filename in filenames):
        failures.append("expected source filename is absent")
    for source in result["sources"]:
        if set(source) - _PUBLIC_SOURCE_FIELDS:
            failures.append("source exposes internal retrieval fields")
            break
        percentage = source.get("match_percentage")
        if not isinstance(percentage, int | float) or not 0 <= percentage <= 100:
            failures.append("source match_percentage is missing or invalid")
            break
    followups = [" ".join(str(value).split()) for value in result["followups"] if str(value).strip()]
    normalized_followups = {value.casefold() for value in followups}
    if len(followups) > 2:
        failures.append("at most two same-call follow-up suggestions are allowed")
    elif len(normalized_followups) != len(followups) or normalized_followups & _PLACEHOLDER_FOLLOWUPS:
        failures.append("follow-up suggestions are duplicated or placeholders")
    elif any(len(value) > 64 or len(value.split()) > 10 for value in followups):
        failures.append("follow-up suggestions are too long")
    else:
        original = " ".join(str(question.get("question") or "").casefold().split())
        labels = [" ".join(filename.casefold().split()) for filename in filenames if filename]
        if (original and any(original in value.casefold() for value in followups)) or any(
            label and any(label in value.casefold() for value in followups) for label in labels
        ):
            failures.append("follow-up suggestions echo the query or source title")
    usage = result.get("usage") or {}
    if not isinstance(usage.get("prompt_tokens"), int) or not isinstance(usage.get("completion_tokens"), int):
        failures.append("token usage is missing")
    if question.get("web_search") and not any(
        source.get("kind") == "web" or source.get("mime_type") == "web" for source in result["sources"]
    ):
        failures.append("web evidence is missing")
    return failures


def run(base_url: str, questions_path: Path, output_path: Path) -> int:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise SystemExit("questions file must contain a non-empty JSON array")

    with httpx.Client(
        base_url=base_url.rstrip("/"),
        headers=_headers(),
        timeout=60,
        follow_redirects=False,
    ) as client:
        files_payload = _request_json(client, "GET", "/files/list")
        ready_files = [
            value
            for value in files_payload.get("files", [])
            if value.get("ingestion_state") == "ready" and value.get("current_revision") is not None
        ]
        records: list[dict[str, Any]] = []
        for question in questions:
            question_text = " ".join(str(question.get("question") or "").split())
            if not question_text:
                raise SystemExit("every question needs non-empty text")
            patterns = [str(value) for value in question.get("expected_source_terms") or []]
            selected_ids = [
                int(value["id"])
                for value in ready_files
                if patterns and _matches(str(value.get("filename") or ""), patterns)
            ][:10]
            discovery = _request_json(
                client,
                "POST",
                "/ai/search",
                json={"query": question_text, "max_results": 5},
            )
            chat = _request_json(
                client,
                "POST",
                "/ai/chats",
                json={"title": f"Validation: {question.get('id', 'question')}"},
            )
            result = _chat(
                client,
                {
                    "message": question_text,
                    "chat_id": str(chat["id"]),
                    "file_ids": selected_ids or None,
                    "deep_search": not bool(question.get("web_search")),
                    "web_search_enabled": bool(question.get("web_search")),
                    "provider": "ollama",
                },
            )
            failures = _evaluate(question, result)
            records.append(
                {
                    "id": question.get("id"),
                    "question": question_text,
                    "expected_source_terms": patterns,
                    "expected_answer_terms": [
                        str(value) for value in question.get("expected_answer_terms") or []
                    ],
                    "selected_file_ids": selected_ids,
                    "suggestions": discovery.get("results", []),
                    **result,
                    "passed": not failures,
                    "failures": failures,
                }
            )

    payload = {
        "base_url": base_url,
        "ready_file_count": len(ready_files),
        "question_count": len(records),
        "passed": sum(record["passed"] for record in records),
        "failed": sum(not record["passed"] for record in records),
        "results": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("question_count", "passed", "failed")}))
    return 0 if payload["failed"] == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9999/api")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        status = run(args.base_url, args.questions, args.output)
    except (httpx.HTTPError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(status)


if __name__ == "__main__":
    main()
