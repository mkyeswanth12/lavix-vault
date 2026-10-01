"""Live web-search eval: retrieval hit-rate + answer faithfulness.

Usage:
    uv run python tests/eval/run_web_eval.py --token <JWT> [--base http://localhost:9999]

Scores fixed question sets with known answers against the running stack
and prints per-case results plus aggregate grades (A>=90, B>=75, C>=60).
Refusals count as failures (every case here is answerable); caveats are
reported but not penalized.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def score_retrieval(card_urls: list[str], expected_domains: list[str]) -> tuple[bool, str]:
    lowered = [url.casefold() for url in card_urls]
    for domain in expected_domains:
        if any(domain.casefold() in url for url in lowered):
            return True, domain
    return False, ""


def score_faithfulness(
    answer: str, must_contain: list[str], must_not_contain: list[str]
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    folded = answer.casefold()
    for required in must_contain:
        if required.casefold() not in folded:
            problems.append(f"missing:{required}")
    for banned in must_not_contain:
        if banned.casefold() in folded:
            problems.append(f"present:{banned}")
    return (not problems), problems


def login(base: str, username: str, password: str) -> str:
    body = json.dumps({"username": username, "password": password}).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/api/auth/login",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.load(res)["access_token"]


def chat(base: str, token: str, message: str, timeout: int = 300) -> dict:
    body = json.dumps(
        {"message": message, "provider": "ollama", "web_search_enabled": True}
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/api/ai/chat",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    events: list[dict] = []
    started = __import__("time").monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            buf = ""
            while True:
                chunk = res.read(4096)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", "replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            events.append(json.loads(data))
                        except ValueError:
                            pass
                else:
                    continue
                break
    except Exception as exc:  # noqa: BLE001 - eval harness reports transport as failure
        return {"transport_error": f"{type(exc).__name__}: {exc}"}
    answer = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    cards = next((e.get("sources", []) for e in events if e.get("type") == "sources"), [])
    error = next((e.get("code") for e in events if e.get("type") == "error"), None)
    return {
        "answer": answer,
        "card_urls": [str(c.get("url") or "") for c in cards if isinstance(c, dict)],
        "error": error,
        "caveat": "could not be verified" in answer,
        "latency_s": round(__import__("time").monotonic() - started, 1),
    }


def grade(pct: float) -> str:
    if pct >= 90:
        return "A"
    if pct >= 75:
        return "B"
    if pct >= 60:
        return "C"
    return "D"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--base", default="http://localhost:9999")
    parser.add_argument(
        "--login",
        default="",
        help="USER:PASS to re-authenticate when the token expires mid-run",
    )
    args = parser.parse_args()
    token = args.token
    credentials = args.login.split(":", 1) if ":" in args.login else None

    def ask(query: str) -> dict:
        nonlocal token
        result = chat(args.base, token, query)
        message = str(result.get("transport_error") or "")
        if "401" in message and credentials:
            token = login(args.base, credentials[0], credentials[1])
            result = chat(args.base, token, query)
        return result

    retrieval = json.loads((HERE / "web_retrieval_set.json").read_text())
    faithfulness = json.loads((HERE / "web_faithfulness_set.json").read_text())

    ret_pass = 0
    ret_refused = 0
    print("== retrieval (hit-rate@cards) ==")
    for case in retrieval:
        result = ask(case["query"])
        if result.get("transport_error"):
            print(f"  FAIL {case['id']} {case['query']!r} -> {result['transport_error']}")
            continue
        if result.get("error"):
            ret_refused += 1
            print(f"  REFUSED {case['id']} {case['query']!r} -> {result['error']} ({result['latency_s']}s)")
            continue
        ok, domain = score_retrieval(result["card_urls"], case["expected_domains"])
        ret_pass += ok
        print(f"  {'PASS' if ok else 'FAIL'} {case['id']} {case['query']!r} -> {domain or result['card_urls'][:2]} ({result['latency_s']}s)")

    faith_pass = 0
    faith_refused = 0
    print("== faithfulness (keywords, no fabrications) ==")
    for case in faithfulness:
        result = ask(case["query"])
        if result.get("transport_error"):
            print(f"  FAIL {case['id']} {case['query']!r} -> {result['transport_error']}")
            continue
        if result.get("error"):
            faith_refused += 1
            print(f"  REFUSED {case['id']} {case['query']!r} -> {result['error']} ({result['latency_s']}s)")
            continue
        ok, problems = score_faithfulness(result["answer"], case["must_contain"], case["must_not_contain"])
        faith_pass += ok
        flag = " (caveat)" if result["caveat"] else ""
        print(f"  {'PASS' if ok else 'FAIL'} {case['id']} {case['query']!r}{flag} -> {problems or 'clean'} ({result['latency_s']}s)")

    ret_pct = 100.0 * ret_pass / max(1, len(retrieval))
    faith_pct = 100.0 * faith_pass / max(1, len(faithfulness))
    print(f"retrieval: {ret_pass}/{len(retrieval)} correct, {ret_refused} honest-refusals = {ret_pct:.0f}% grade {grade(ret_pct)}")
    print(f"faithfulness: {faith_pass}/{len(faithfulness)} correct, {faith_refused} honest-refusals = {faith_pct:.0f}% grade {grade(faith_pct)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
