import asyncio

import pytest
from httpx import MockTransport, Request, Response

from agent_runtime.events import AUTHORITATIVE_STREAM_PROVENANCE
from app.agent.runtime_client import AgentRuntimeClient, AgentRuntimeError
from app.config import Settings


def test_api_agent_timing_defaults_are_aligned(monkeypatch):
    monkeypatch.delenv("AGENT_RUNTIME_READ_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("AGENT_CAPABILITY_TTL_SECONDS", raising=False)
    settings = Settings()
    client = AgentRuntimeClient("http://runtime.test")

    assert settings.agent_runtime_read_timeout_seconds == 450
    assert settings.agent_capability_ttl_seconds == 300
    assert settings.web_search_timeout == 12
    assert settings.web_search_hard_timeout == 12.0
    assert client._timeout.read == 450.0


def test_runtime_client_normalizes_ndjson_events():
    async def handler(request: Request) -> Response:
        assert request.url == "http://runtime.test/internal/v1/stream"
        assert request.headers["X-Lavix-Capability"] == "opaque-token"
        return Response(
            200,
            content=(
                b'{"type":"status","step":"planning","ignored":"state"}\n'
                b'{"type":"answer_delta","delta":"The ","provenance":"lavix.authoritative-synthesis.v1"}\n'
                b'{"type":"answer_delta","delta":"answer","provenance":"lavix.authoritative-synthesis.v1"}\n'
                b'{"type":"final","answer":"The answer","provenance":"lavix.authoritative-synthesis.v1","followups":["What changed?"]}\n'
                b'{"type":"usage","prompt_tokens":17,"completion_tokens":9}\n'
                b'{"type":"done"}\n'
            ),
        )

    async def collect():
        client = AgentRuntimeClient("http://runtime.test", transport=MockTransport(handler))
        return [event async for event in client.stream({"messages": []}, "opaque-token")]

    assert asyncio.run(collect()) == [
        {"type": "status", "step": "planning"},
        {
            "type": "answer_delta",
            "delta": "The ",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        },
        {
            "type": "answer_delta",
            "delta": "answer",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        },
        {
            "type": "final",
            "answer": "The answer",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
            "followups": ["What changed?"],
        },
        {"type": "usage", "prompt_tokens": 17, "completion_tokens": 9},
        {"type": "done"},
    ]


def test_runtime_client_rejects_http_and_protocol_failures():
    responses = [
        Response(503, json={"detail": "secret internal detail"}),
        Response(200, content=b'{"type":"unsafe_state"}\n'),
    ]

    async def handler(_: Request) -> Response:
        return responses.pop(0)

    transport = MockTransport(handler)

    async def collect():
        client = AgentRuntimeClient("http://runtime.test", transport=transport)
        return [event async for event in client.stream({"messages": []}, "opaque-token")]

    with pytest.raises(AgentRuntimeError) as unavailable:
        asyncio.run(collect())
    assert unavailable.value.code == "runtime_unavailable"

    with pytest.raises(AgentRuntimeError) as protocol:
        asyncio.run(collect())
    assert protocol.value.code == "runtime_protocol_error"


def test_runtime_client_never_forwards_private_status_or_error_text():
    assert AgentRuntimeClient._event('{"type":"status","step":"postgresql://user:secret@database"}') == {
        "type": "status",
        "step": "working",
    }
    assert AgentRuntimeClient._event(
        '{"type":"error","code":"agent_timeout","message":"secret stack trace"}'
    ) == {
        "type": "error",
        "code": "agent_timeout",
        "message": "Agent run exceeded its time limit",
    }
    assert AgentRuntimeClient._event(
        '{"type":"error","code":"private_failure","message":"password=secret"}'
    ) == {
        "type": "error",
        "code": "agent_execution_failed",
        "message": "Agent execution failed",
    }


@pytest.mark.parametrize(
    "line",
    [
        '{"type":"answer_delta","delta":""}',
        '{"type":"answer_delta","delta":7}',
        '{"type":"answer_delta","delta":"unsafe","provenance":"cuga.callback"}',
        '{"type":"final","answer":"safe"}',
        '{"type":"final","answer":"safe","provenance":"cuga.callback"}',
        '{"type":"final","answer":"safe","followups":"not-a-list"}',
        '{"type":"final","answer":"safe","followups":[7]}',
        '{"type":"final","answer":"safe","followups":[" Evidence: private?"]}',
        '{"type":"final","answer":"safe","followups":["**Tool output:** private?"]}',
    ],
)
def test_runtime_client_rejects_invalid_public_answer_events(line):
    with pytest.raises(AgentRuntimeError) as error:
        AgentRuntimeClient._event(line)

    assert error.value.code == "runtime_protocol_error"


def test_runtime_client_forwards_evidence_ids():
    assert AgentRuntimeClient._event(
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1",'
        '"evidence_ids":{"vault":["V1"],"web":[]}}'
    ) == {
        "type": "final",
        "answer": "Safe",
        "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        "evidence_ids": {"vault": ["V1"], "web": []},
    }


def test_runtime_client_forwards_unverified_flag_only_when_true():
    flagged = AgentRuntimeClient._event(
        '{"type":"final","answer":"Background overview","provenance":"lavix.authoritative-synthesis.v1",'
        '"unverified":true}'
    )
    assert flagged["unverified"] is True
    plain = AgentRuntimeClient._event(
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1"}'
    )
    assert "unverified" not in plain
    falsy = AgentRuntimeClient._event(
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1","unverified":false}'
    )
    assert "unverified" not in falsy


@pytest.mark.parametrize(
    "line",
    [
        # Absent field stays absent (legacy unfiltered-card behavior).
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1"}',
        # Malformed shapes fall back to unfiltered cards, never an error.
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1","evidence_ids":[1,2]}',
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1","evidence_ids":{"vault":"V1"}}',
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1","evidence_ids":{}}',
    ],
)
def test_runtime_client_omits_unusable_evidence_ids(line):
    assert "evidence_ids" not in AgentRuntimeClient._event(line)


def test_runtime_client_drops_non_string_evidence_id_entries():
    event = AgentRuntimeClient._event(
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1",'
        '"evidence_ids":{"web":["W1",7,{}]}}'
    )
    assert event["evidence_ids"] == {"web": ["W1"]}


def test_runtime_client_forwards_discard_reasons():
    event = AgentRuntimeClient._event(
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1",'
        '"evidence_ids":{"vault":[],"web":["W1"]},'
        '"discard_reasons":[{"id":"W2","reason":"verify"},{"id":"W3","reason":"grounding"}]}'
    )
    assert event["discard_reasons"] == [
        {"id": "W2", "reason": "verify"},
        {"id": "W3", "reason": "grounding"},
    ]


def test_runtime_client_forwards_resolved_entities():
    event = AgentRuntimeClient._event(
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1",'
        '"resolved_entities":{"his":"Don Bradman","the Ashes":"Ashes cricket series"}}'
    )
    assert event["resolved_entities"] == {"his": "Don Bradman", "the Ashes": "Ashes cricket series"}


@pytest.mark.parametrize(
    "line",
    [
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1"}',
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1","resolved_entities":[]}',
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1","resolved_entities":{}}',
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1","resolved_entities":{"":"x"}}',
    ],
)
def test_runtime_client_omits_unusable_resolved_entities(line):
    assert "resolved_entities" not in AgentRuntimeClient._event(line)


@pytest.mark.parametrize(
    "line",
    [
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1"}',
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1","discard_reasons":"nope"}',
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1","discard_reasons":[{"id":"W2","reason":"bogus"}]}',
        '{"type":"final","answer":"Safe","provenance":"lavix.authoritative-synthesis.v1","discard_reasons":[{"id":"","reason":"verify"}]}',
    ],
)
def test_runtime_client_omits_unusable_discard_reasons(line):
    assert "discard_reasons" not in AgentRuntimeClient._event(line)


def test_runtime_client_forwards_clarification_question():
    event = AgentRuntimeClient._event(
        '{"type":"clarification","question":"Just to confirm — by \'his\' do you mean Don Bradman?"}'
    )
    assert event == {
        "type": "clarification",
        "question": "Just to confirm — by 'his' do you mean Don Bradman?",
    }


@pytest.mark.parametrize(
    "line",
    [
        '{"type":"clarification"}',
        '{"type":"clarification","question":"   "}',
        '{"type":"clarification","question":"Could you clarify your question? [V1]"}',
    ],
)
def test_runtime_client_rejects_unusable_clarification(line):
    with pytest.raises(AgentRuntimeError):
        AgentRuntimeClient._event(line)


def test_runtime_client_forwards_refusal_path_and_note():
    assert AgentRuntimeClient._event(
        '{"type":"error","code":"no_verifiable_evidence","message":"anything",'
        '"failing_path":"web-auto-recency"}'
    ) == {
        "type": "error",
        "code": "no_verifiable_evidence",
        "message": "No verifiable evidence could be found for this question",
        "failing_path": "web-auto-recency",
    }
    # Unknown paths are dropped, never forwarded.
    assert AgentRuntimeClient._event(
        '{"type":"error","code":"no_verifiable_evidence","message":"x",'
        '"failing_path":"elsewhere"}'
    ) == {
        "type": "error",
        "code": "no_verifiable_evidence",
        "message": "No verifiable evidence could be found for this question",
    }


def test_runtime_client_forwards_web_auto_note_on_final():
    import json

    event = AgentRuntimeClient._event(
        json.dumps(
            {
                "type": "final",
                "answer": "ok",
                "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
                "web_auto_note": "Checked the web for this (auto-enabled: time-sensitive question)",
            }
        )
    )

    assert event["web_auto_note"].startswith("Checked the web")


def test_refusal_path_survives_client_to_card() -> None:
    import json

    from app.routers.chat.orchestrator import ChatCoordinator

    raw = json.dumps(
        {
            "type": "error",
            "code": "no_verifiable_evidence",
            "message": "Web was auto-checked for this time-sensitive question",
            "failing_path": "web-auto-recency",
        }
    )
    event = AgentRuntimeClient._event(raw)
    card = ChatCoordinator._public_error(
        event["code"], event["message"], event.get("failing_path")
    )

    assert "auto-checked" in card["message"] or "toggle" in card["hint"].casefold()
    assert "turn web on" in card["hint"].casefold()
