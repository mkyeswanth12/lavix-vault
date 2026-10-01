#!/usr/bin/env python3
"""Run the live adversarial-regression bank and write machine-readable results.

Covers what unit tests cannot: end-to-end behavior through retrieval,
synthesis, and cards against live services and the real model. Verdicts
are PASS / FLAG / FAIL on purpose:

- FAIL (nonzero exit): transport or shape regressions — HTTP errors on
  must-answer controls, malformed SSE, marker leaks into answers,
  deterministic-path mismatches (calculator exact answers, guard
  fall-through). These are stable across runs.
- FLAG (exit 0, listed in JSON): content-quality signals that vary with
  live engines and model sampling — grounding gaps over card titles,
  card-count shortfalls, abstention wording. A human reads flags; the
  suite never goes red on model text.

Usage:
    LAVIX_ACCESS_TOKEN=... python3 scripts/eval_live_rag.py \
        --bank scripts/eval_question_bank.json --output /tmp/eval.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_MARKER_RE = re.compile(r"\[lavix[^\[\]]*?\]|</?lavix[^<>]*>", re.IGNORECASE)
_CAP_WORD_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_SKIP_WORDS = frozenset(
    "what who when where how which that this the most recent first last "
    "his her its there here it he she they we you as but and for with "
    "from an no yes not all also however meanwhile hello hi hey thanks "
    "sorry please current".split()
)


def _headers() -> dict[str, str]:
    token = os.environ.get("LAVIX_ACCESS_TOKEN", "").strip()
    if not token:
        raise SystemExit("LAVIX_ACCESS_TOKEN is required")
    return {"Authorization": f"Bearer {token}"}


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
    return {
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "answer": answer,
        "sources": list(source_events[-1].get("sources") or []) if source_events else [],
        "followups": list(followup_events[-1].get("questions") or []) if followup_events else [],
        "errors": [event for event in events if event.get("type") == "error"],
    }


def _grounding_flag(answer: str, sources: list[dict[str, Any]]) -> list[str]:
    """Approximate Issue-8 check over card titles (cards lack snippets).

    Mirrors agent_runtime ground-truth logic loosely: capitalized atoms in
    the answer must substring-match some card title. Informational only —
    card titles under-approximate shown evidence, so misses here are flags,
    never failures. Canonical implementation lives in cuga_adapter.py.
    """

    atoms: list[str] = []
    for match in _CAP_WORD_RE.finditer(answer or ""):
        phrase = match.group(1)
        if " " not in phrase and phrase.lower() in _SKIP_WORDS:
            continue
        atoms.append(phrase)
    hay = " ".join(str(item.get("title") or "") for item in sources).casefold()
    flagged: list[str] = []
    seen: set[str] = set()
    for atom in atoms:
        key = atom.casefold()
        if key in seen:
            continue
        seen.add(key)
        if key not in hay:
            flagged.append(atom)
    return flagged


def _evaluate(entry: dict[str, Any], result: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    flags: list[str] = []
    accept = entry.get("accept", ["answer"])
    got_answer = bool(result["answer"])
    got_error = bool(result["errors"])
    error_ok = any(str(accept_item).startswith("error:") for accept_item in accept)
    if got_error and not error_ok:
        failures.append(f"unexpected error card: {result['errors']}")
    if not got_answer and "answer" in accept and not (got_error and error_ok):
        failures.append("empty answer with no accepted error")
    if got_answer:
        if _MARKER_RE.search(result["answer"]):
            failures.append("answer exposes LAVIX-directive markers")
        for term in entry.get("must_contain") or []:
            if term not in result["answer"]:
                failures.append(f"answer missing required text: {term!r}")
        for term in entry.get("must_not_contain") or []:
            if term in result["answer"]:
                failures.append(f"answer contains forbidden text: {term!r}")
        if "sources_empty" in (entry.get("assert") or []):
            if result["sources"]:
                failures.append(f"expected zero source cards, got {len(result['sources'])}")
        if "sources_nonempty" in (entry.get("assert") or []):
            if not result["sources"]:
                flags.append("no source cards rendered")
        gap = _grounding_flag(result["answer"], result["sources"])
        if gap and result["sources"]:
            flags.append(f"answer atoms missing from card titles: {gap[:6]}")
    return ("FAIL" if failures else ("FLAG" if flags else "PASS")), failures + flags


def run(base_url: str, bank_path: Path, output_path: Path) -> int:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    hard_failures = 0
    with httpx.Client(base_url=base_url.rstrip("/"), headers=_headers(), timeout=30) as client:
        for entry in bank.get("questions", []):
            chat = client.post("/ai/chats", json={"title": f"Eval: {entry.get('id')}"}).json()
            try:
                result = _chat(
                    client,
                    {
                        "message": entry["question"],
                        "chat_id": str(chat["id"]),
                        "web_search_enabled": bool(entry.get("web_search", True)),
                        "provider": "ollama",
                    },
                )
            except (httpx.HTTPError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                records.append({"id": entry.get("id"), "verdict": "FAIL", "transport_error": str(exc)[:300]})
                hard_failures += 1
                continue
            verdict, notes = _evaluate(entry, result)
            if verdict == "FAIL":
                hard_failures += 1
            records.append(
                {
                    "id": entry.get("id"),
                    "question": entry["question"],
                    "verdict": verdict,
                    "notes": notes,
                    "answer_preview": result["answer"][:400],
                    "source_count": len(result["sources"]),
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
    payload = {
        "question_count": len(records),
        "passed": sum(1 for record in records if record["verdict"] == "PASS"),
        "flagged": sum(1 for record in records if record["verdict"] == "FLAG"),
        "failed": sum(1 for record in records if record["verdict"] == "FAIL"),
        "results": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("question_count", "passed", "flagged", "failed")}))
    return 0 if hard_failures == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9999/api")
    parser.add_argument("--bank", type=Path, default=Path(__file__).with_name("eval_question_bank.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        status = run(args.base_url, args.bank, args.output)
    except (httpx.HTTPError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"eval failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(status)


if __name__ == "__main__":
    main()
