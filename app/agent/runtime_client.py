"""Strict NDJSON client for the private CUGA runtime API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agent_runtime.events import AUTHORITATIVE_STREAM_PROVENANCE, is_unchanged_public_text

_PUBLIC_STATUS_STEPS = frozenset(
    {
        "queued",
        "running",
        "planning",
        "generating",
        "executing_tools",
        "searching_vault",
        "reranking",
        "web_search",
    }
)
_PUBLIC_ERROR_MESSAGES = {
    "agent_busy": "Agent runtime is busy; try again shortly",
    "vault_retrieval_failed": "Authorized vault evidence could not be retrieved",
    "vault_no_evidence": "No matching evidence was found in the authorized files",
    "web_search_failed": "Current web evidence could not be retrieved",
    "web_no_evidence": "No matching evidence was found on the current web",
    "no_verifiable_evidence": "No verifiable evidence could be found for this question",
    "empty_agent_response": "Agent completed without an answer",
    "agent_timeout": "Agent run exceeded its time limit",
    "agent_execution_failed": "Agent execution failed",
    "runtime_stream_failed": "Runtime stream failed",
}


class AgentRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AgentRuntimeClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 450.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/internal/v1/stream"
        self._timeout = httpx.Timeout(connect=5.0, read=timeout_seconds, write=10.0, pool=5.0)
        self._transport = transport

    async def stream(
        self,
        request: dict[str, Any],
        capability_token: str,
    ) -> AsyncIterator[dict[str, Any]]:
        headers = {"X-Lavix-Capability": capability_token, "Accept": "application/x-ndjson"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                async with client.stream("POST", self._url, json=request, headers=headers) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        code = "runtime_rejected" if response.status_code < 500 else "runtime_unavailable"
                        raise AgentRuntimeError(code, "Agent runtime rejected the request")
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if len(line) > 256_000:
                            raise AgentRuntimeError("runtime_protocol_error", "Runtime event is too large")
                        yield self._event(line)
        except AgentRuntimeError:
            raise
        except httpx.HTTPError as exc:
            raise AgentRuntimeError("runtime_unavailable", "Agent runtime is unavailable") from exc

    @staticmethod
    def _event(line: str) -> dict[str, Any]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AgentRuntimeError("runtime_protocol_error", "Runtime emitted invalid data") from exc
        if not isinstance(value, dict) or value.get("type") not in {
            "status",
            "answer_delta",
            "final",
            "usage",
            "error",
            "clarification",
            "done",
        }:
            raise AgentRuntimeError("runtime_protocol_error", "Runtime emitted an unknown event")
        event_type = value["type"]
        if event_type == "status":
            step = str(value.get("step") or "")
            return {"type": "status", "step": step if step in _PUBLIC_STATUS_STEPS else "working"}
        if event_type == "answer_delta":
            delta = value.get("delta")
            if (
                not isinstance(delta, str)
                or not delta
                or len(delta) > 16_000
                or value.get("provenance") != AUTHORITATIVE_STREAM_PROVENANCE
            ):
                raise AgentRuntimeError("runtime_protocol_error", "Runtime emitted an invalid answer delta")
            return {
                "type": "answer_delta",
                "delta": delta,
                "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
            }
        if event_type == "final":
            answer = str(value.get("answer") or "").strip()
            if not answer or len(answer) > 100_000:
                raise AgentRuntimeError("runtime_protocol_error", "Runtime emitted an invalid answer")
            raw_followups = value.get("followups", [])
            if not isinstance(raw_followups, list) or any(
                not isinstance(question, str) or len(question) > 500 or not is_unchanged_public_text(question)
                for question in raw_followups
            ):
                raise AgentRuntimeError(
                    "runtime_protocol_error", "Runtime emitted invalid follow-up metadata"
                )
            followups = list(dict.fromkeys(raw_followups))[:10]
            event = {"type": "final", "answer": answer}
            provenance = value.get("provenance")
            if provenance != AUTHORITATIVE_STREAM_PROVENANCE:
                raise AgentRuntimeError(
                    "runtime_protocol_error", "Runtime emitted invalid answer provenance"
                )
            event["provenance"] = AUTHORITATIVE_STREAM_PROVENANCE
            # Structured explanatory flag: the answer is background
            # knowledge with no retrievable evidence. A boolean only —
            # never answer prose — so projection/stripping cannot mangle it.
            if value.get("unverified") is True:
                event["unverified"] = True
            # Web auto-check note (recency/follow-up override fired despite
            # the toggle). Short validated string, same philosophy.
            raw_note = value.get("web_auto_note")
            if isinstance(raw_note, str) and raw_note.strip():
                event["web_auto_note"] = raw_note.strip()[:200]
            if followups:
                event["followups"] = followups
            # Citation-membership produced by post-generation checks. Unlike
            # answer/followups (rejected when malformed), an unusable value
            # here must not fail the answer: fall back to unfiltered cards.
            raw_evidence_ids = value.get("evidence_ids")
            if isinstance(raw_evidence_ids, dict):
                evidence_ids: dict[str, list[str]] = {}
                for _kind in ("vault", "web"):
                    _ids = raw_evidence_ids.get(_kind)
                    if isinstance(_ids, list):
                        evidence_ids[_kind] = [
                            str(_item) for _item in _ids if isinstance(_item, str)
                        ][:100]
                if evidence_ids:
                    event["evidence_ids"] = evidence_ids
            # Per-ID drop reasons (verify/grounding) for the discard report.
            # Same fail-open philosophy: malformed entries are dropped, and
            # a malformed field never fails the answer.
            raw_discards = value.get("discard_reasons")
            if isinstance(raw_discards, list):
                discards: list[dict[str, str]] = []
                for _entry in raw_discards[:100]:
                    if not isinstance(_entry, dict):
                        continue
                    _did = _entry.get("id")
                    _why = _entry.get("reason")
                    if (
                        isinstance(_did, str)
                        and _did.strip()
                        and _why in ("verify", "grounding")
                    ):
                        discards.append({"id": _did.strip(), "reason": _why})
                if discards:
                    event["discard_reasons"] = discards
            # Entity carry-forward from the web-query rewriter. Same
            # fail-open philosophy: malformed entries are dropped, and a
            # malformed field never fails the answer.
            raw_entities = value.get("resolved_entities")
            if isinstance(raw_entities, dict):
                entities: dict[str, str] = {}
                for _key, _val in list(raw_entities.items())[:20]:
                    name, target = str(_key or "").strip(), str(_val or "").strip()
                    if name and target:
                        entities[name[:64]] = target[:200]
                if entities:
                    event["resolved_entities"] = entities
            return event
        if event_type == "usage":
            try:
                prompt_tokens = max(0, int(value.get("prompt_tokens") or 0))
                completion_tokens = max(0, int(value.get("completion_tokens") or 0))
            except (TypeError, ValueError) as exc:
                raise AgentRuntimeError(
                    "runtime_protocol_error", "Runtime emitted invalid token usage"
                ) from exc
            return {
                "type": "usage",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        if event_type == "error":
            raw_code = str(value.get("code") or "")
            code = raw_code if raw_code in _PUBLIC_ERROR_MESSAGES else "agent_execution_failed"
            error_event: dict[str, Any] = {
                "type": "error",
                "code": code,
                "message": _PUBLIC_ERROR_MESSAGES[code],
            }
            # Machine-readable refusal path (vault/web/vault+web/memory/none
            # + web-auto-* overrides). Allowlisted so junk cannot ride along;
            # without it the orchestrator falls back to message sniffing.
            raw_path = value.get("failing_path")
            if raw_path in (
                "memory",
                "vault",
                "web",
                "vault+web",
                "none",
                "web-auto-recency",
                "web-auto-followup",
            ):
                error_event["failing_path"] = raw_path
            return error_event
        if event_type == "clarification":
            # One-round web-clarification turn. Same strict bar as
            # follow-ups: public-projection-stable text only, so protocol
            # markers or citations smuggled into the question fail the run
            # instead of reaching the user.
            question = str(value.get("question") or "").strip()
            if not question or len(question) > 500 or not is_unchanged_public_text(question):
                raise AgentRuntimeError(
                    "runtime_protocol_error", "Runtime emitted an invalid clarification"
                )
            return {"type": "clarification", "question": question}
        return {"type": "done"}
