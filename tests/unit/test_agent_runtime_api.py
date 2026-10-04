import json
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent_runtime.config import RuntimeSettings
from agent_runtime.events import AUTHORITATIVE_STREAM_PROVENANCE
from agent_runtime.main import create_app

CAPABILITY = "opaque-capability-token-for-tests"


def test_importing_api_does_not_import_cuga_or_langchain_ollama():
    loaded = set(sys.modules)

    assert "cuga" not in loaded
    assert not any(name.startswith("cuga.") for name in loaded)
    assert "langchain_ollama" not in loaded


class FakeRuntime:
    is_initialized = False

    def __init__(self):
        self.calls = []
        self.closed = False

    async def invoke(self, request, capability):
        self.calls.append(("invoke", request, capability))
        return SimpleNamespace(
            run_id=str(request.run_id),
            model=request.model or "llama3b-instruct-q6kl-16k:latest",
            answer="complete answer",
        )

    async def stream_events(self, request, capability):
        self.calls.append(("stream", request, capability))
        yield {"type": "status", "step": "planning"}
        yield {
            "type": "answer_delta",
            "delta": "streamed ",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        }
        yield {
            "type": "answer_delta",
            "delta": "answer",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        }
        yield {
            "type": "final",
            "answer": "streamed answer",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        }
        yield {"type": "done"}

    async def close(self):
        self.closed = True


class FailingStreamRuntime(FakeRuntime):
    async def stream_events(self, request, capability):
        self.calls.append(("stream", request, capability))
        raise RuntimeError("private exception details")
        yield  # pragma: no cover - keeps this an async generator


def payload(**overrides):
    value = {
        "run_id": "00000000-0000-0000-0000-000000000101",
        "user_query": "hello",
        "messages": [{"role": "user", "content": "hello"}],
    }
    value.update(overrides)
    return value


def test_invoke_requires_capability_and_forwards_it_opaquely():
    runtime = FakeRuntime()
    app = create_app(settings=RuntimeSettings(), runtime=runtime)

    with TestClient(app) as client:
        missing = client.post("/internal/v1/invoke", json=payload())
        response = client.post(
            "/internal/v1/invoke",
            json=payload(),
            headers={"X-Lavix-Capability": CAPABILITY},
        )

    assert missing.status_code == 422
    assert response.status_code == 200
    assert response.json()["answer"] == "complete answer"
    assert runtime.calls[0][2] == CAPABILITY
    assert runtime.closed is True


def test_model_outside_allowlist_is_rejected_before_runtime_call():
    runtime = FakeRuntime()
    app = create_app(settings=RuntimeSettings(), runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/invoke",
            json=payload(model="openrouter/auto"),
            headers={"X-Lavix-Capability": CAPABILITY},
        )

    assert response.status_code == 422
    assert runtime.calls == []


def test_private_user_query_is_required_nonblank_and_bounded():
    runtime = FakeRuntime()
    app = create_app(settings=RuntimeSettings(), runtime=runtime)
    missing_query = payload()
    missing_query.pop("user_query")

    with TestClient(app) as client:
        missing = client.post(
            "/internal/v1/invoke",
            json=missing_query,
            headers={"X-Lavix-Capability": CAPABILITY},
        )
        blank = client.post(
            "/internal/v1/invoke",
            json=payload(user_query="   "),
            headers={"X-Lavix-Capability": CAPABILITY},
        )
        too_large = client.post(
            "/internal/v1/invoke",
            json=payload(user_query="q" * 20_001),
            headers={"X-Lavix-Capability": CAPABILITY},
        )

    assert (missing.status_code, blank.status_code, too_large.status_code) == (422, 422, 422)
    assert runtime.calls == []


def test_stream_is_safe_ndjson_with_no_sse_or_internal_state():
    runtime = FakeRuntime()
    app = create_app(settings=RuntimeSettings(), runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/stream",
            json=payload(),
            headers={"X-Lavix-Capability": CAPABILITY},
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert events == [
        {"type": "status", "step": "planning"},
        {
            "type": "answer_delta",
            "delta": "streamed ",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        },
        {
            "type": "answer_delta",
            "delta": "answer",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        },
        {
            "type": "final",
            "answer": "streamed answer",
            "provenance": AUTHORITATIVE_STREAM_PROVENANCE,
        },
        {"type": "done"},
    ]
    assert "data:" not in response.text


def test_health_does_not_initialize_cuga():
    runtime = FakeRuntime()
    app = create_app(settings=RuntimeSettings(), runtime=runtime)

    with TestClient(app) as client:
        response = client.get("/internal/v1/health/ready")

    assert app.version == "1.0.0"
    assert response.json()["cuga_initialized"] is False
    assert runtime.calls == []


def test_stream_failure_is_safe_and_has_exactly_one_done_event():
    runtime = FailingStreamRuntime()
    app = create_app(settings=RuntimeSettings(), runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/stream",
            json=payload(),
            headers={"X-Lavix-Capability": CAPABILITY},
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    assert events == [
        {
            "type": "error",
            "code": "runtime_stream_failed",
            "message": "Runtime stream failed",
        },
        {"type": "done"},
    ]
    assert "private exception" not in response.text
