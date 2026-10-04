"""Private FastAPI surface for the isolated CUGA process."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from .config import ModelNotAllowedError, RuntimeSettings
from .cuga_adapter import (
    AgentExecutionError,
    CugaAdapter,
    InputTooLargeError,
    _RedactingLogFilter,
)
from .schemas import InvokeResponse, RunRequest

CAPABILITY_HEADER = "X-Lavix-Capability"
CapabilityToken = Annotated[
    str,
    Header(
        alias=CAPABILITY_HEADER,
        min_length=16,
        max_length=8_192,
        description="Opaque, short-lived capability minted by the Lavix API",
    ),
]


def _preflight(settings: RuntimeSettings, request: RunRequest) -> None:
    try:
        settings.resolve_model(request.model, request.options.allowed_models)
    except ModelNotAllowedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if sum(len(message.content) for message in request.messages) > settings.max_input_chars:
        raise HTTPException(status_code=413, detail="Conversation exceeds input limit")


def create_app(
    *,
    settings: RuntimeSettings | None = None,
    runtime: Any | None = None,
) -> FastAPI:
    runtime_settings = settings or RuntimeSettings.from_env()
    agent_runtime = runtime or CugaAdapter(runtime_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        close = getattr(agent_runtime, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    app = FastAPI(
        title="Lavix CUGA Runtime",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = agent_runtime
    app.state.settings = runtime_settings

    @app.get("/internal/v1/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/internal/v1/health/ready")
    async def ready() -> dict[str, Any]:
        # Readiness is intentionally import-free; CUGA initializes on first use.
        return {
            "status": "ready",
            "model": runtime_settings.default_model,
            "cuga_initialized": bool(getattr(agent_runtime, "is_initialized", False)),
        }

    @app.post("/internal/v1/invoke", response_model=InvokeResponse)
    async def invoke(request: RunRequest, capability_token: CapabilityToken) -> InvokeResponse:
        _preflight(runtime_settings, request)
        try:
            result = await agent_runtime.invoke(request, capability_token)
        except InputTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except AgentExecutionError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc
        return InvokeResponse(
            run_id=result.run_id,
            model=result.model,
            answer=result.answer,
        )

    @app.post("/internal/v1/stream")
    async def stream(request: RunRequest, capability_token: CapabilityToken) -> StreamingResponse:
        _preflight(runtime_settings, request)

        async def ndjson() -> AsyncIterator[bytes]:
            done_sent = False
            try:
                async with aclosing(agent_runtime.stream_events(request, capability_token)) as runtime_events:
                    async for event in runtime_events:
                        if done_sent:
                            continue
                        if event.get("type") == "done":
                            done_sent = True
                        # Compact one-object-per-line framing; never serialize CUGA state.
                        yield (json.dumps(event, separators=(",", ":")) + "\n").encode()
            except Exception:
                # Headers have already been sent, so failures stay inside the stream.
                if not done_sent:
                    yield b'{"type":"error","code":"runtime_stream_failed","message":"Runtime stream failed"}\n'
            if not done_sent:
                yield b'{"type":"done"}\n'

        return StreamingResponse(
            ndjson(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()

# Scrub credential values from every emitted log record (root logger, so
# uvicorn-configured handlers are covered too). The agent image ships
# without app/, hence the local filter mirroring app.security.redact.
logging.getLogger().addFilter(_RedactingLogFilter())
