"""Request capability scope propagated to asynchronous CUGA tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

EventSink = Callable[[dict], Awaitable[None]]


async def _discard_event(_: dict) -> None:
    return None


@dataclass(slots=True)
class RunScope:
    run_id: str
    capability_token: str = field(repr=False)
    requested_file_ids: tuple[int, ...] | None = None
    web_search_enabled: bool = True
    deep_search: bool = False
    chat_mode: str | None = None
    event_sink: EventSink = field(default=_discard_event, repr=False)
    prefetched_vault_result: dict[str, object] | None = field(default=None, repr=False)
    prefetched_web_result: dict[str, object] | None = field(default=None, repr=False)
    prefetched_graph_result: dict[str, object] | None = field(default=None, repr=False)
    local_tool_expectations: dict[str, dict[str, object]] = field(default_factory=dict, repr=False)
    verified_local_results: dict[str, dict[str, object]] = field(default_factory=dict, repr=False)
    local_tool_violation: bool = field(default=False, repr=False)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    async def emit(self, event: dict) -> None:
        await self.event_sink(event)

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += max(0, int(prompt_tokens))
        self.completion_tokens += max(0, int(completion_tokens))


_CURRENT_SCOPE: ContextVar[RunScope | None] = ContextVar("lavix_agent_run_scope", default=None)


class MissingRunScopeError(RuntimeError):
    """Raised if a tool is invoked outside a capability-bound agent run."""


def current_run_scope() -> RunScope:
    scope = _CURRENT_SCOPE.get()
    if scope is None:
        raise MissingRunScopeError("agent tool called without a run capability")
    return scope


@contextmanager
def bind_run_scope(scope: RunScope) -> Iterator[None]:
    token = _CURRENT_SCOPE.set(scope)
    try:
        yield
    finally:
        _CURRENT_SCOPE.reset(token)
