from __future__ import annotations

import json
import sys
import time

import httpx
import pytest

# Polling helper under test (moved here when the disposable-stack e2e
# scaffolding was removed from the public tree). It waits for one durable
# ingestion job to reach a ready or typed terminal state.
INGESTION_TIMEOUT = 10.0
TERMINAL_FAILURE_STATES = {"failed", "cancelled", "unsupported", "password_required"}


def _assert_ok(response: httpx.Response) -> httpx.Response:
    assert response.is_success, f"{response.request.method} {response.request.url}: {response.text}"
    return response


def _wait_for_ingestion(api, file_id: int) -> dict:
    """Wait for one durable job to reach a ready or typed terminal state."""

    deadline = time.monotonic() + INGESTION_TIMEOUT
    last: dict = {}
    while time.monotonic() < deadline:
        response = api.get(f"/files/ai-status/{file_id}")
        if response.status_code == 503:
            retry_after = response.headers.get("Retry-After")
            try:
                retry_delay = float(retry_after) if retry_after is not None else 1.0
            except ValueError:
                retry_delay = 1.0
            if not 0 <= retry_delay < float("inf"):
                retry_delay = 1.0
            last = {
                "state": "service_unavailable",
                "http_status": 503,
                "retry_after": retry_after,
            }
            remaining = max(0.0, deadline - time.monotonic())
            if remaining:
                time.sleep(min(retry_delay, remaining))
            continue

        last = _assert_ok(response).json()
        if last["state"] == "ready" or last["state"] in TERMINAL_FAILURE_STATES:
            return last
        remaining = max(0.0, deadline - time.monotonic())
        if remaining:
            time.sleep(min(1.0, remaining))
    pytest.fail(f"ingestion timed out for file {file_id}: {json.dumps(last, default=str)}")


_SELF = sys.modules[__name__]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _Api:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def get(self, _path: str) -> httpx.Response:
        self.calls += 1
        return next(self.responses)


def _response(
    status_code: int,
    payload: dict,
    *,
    retry_after: str | None = None,
) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    return httpx.Response(
        status_code,
        json=payload,
        headers=headers,
        request=httpx.Request("GET", "http://webui/api/files/ai-status/42"),
    )


def _install_clock(monkeypatch: pytest.MonkeyPatch, timeout: float) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(_SELF, "time", clock)
    monkeypatch.setattr(_SELF, "INGESTION_TIMEOUT", timeout)
    return clock


def test_ingestion_poll_retries_503_using_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _install_clock(monkeypatch, 10)
    ready = {"state": "ready", "chunk_count": 1}
    api = _Api(
        [
            _response(503, {"detail": "Service temporarily unavailable"}, retry_after="2"),
            _response(200, {"state": "processing"}),
            _response(200, ready),
        ]
    )

    assert _wait_for_ingestion(api, 42) == ready
    assert api.calls == 3
    assert clock.sleeps == [2.0, 1.0]


def test_ingestion_poll_persistent_503_stops_at_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _install_clock(monkeypatch, 3)
    api = _Api(
        [
            _response(503, {"detail": "Service temporarily unavailable"}, retry_after="2"),
            _response(503, {"detail": "Service temporarily unavailable"}, retry_after="2"),
        ]
    )

    with pytest.raises(pytest.fail.Exception, match='"http_status": 503'):
        _wait_for_ingestion(api, 42)

    assert api.calls == 2
    assert clock.now == 3
    assert clock.sleeps == [2.0, 1.0]


def test_ingestion_poll_does_not_retry_unexpected_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch, 10)
    api = _Api([_response(500, {"error": "Internal server error"})])

    with pytest.raises(AssertionError, match="500"):
        _wait_for_ingestion(api, 42)

    assert api.calls == 1
    assert clock.sleeps == []


def test_ingestion_poll_returns_typed_terminal_state_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch, 10)
    failed = {"state": "failed", "error_code": "parser_failed"}
    api = _Api([_response(200, failed)])

    assert _wait_for_ingestion(api, 42) == failed
    assert api.calls == 1
    assert clock.sleeps == []
