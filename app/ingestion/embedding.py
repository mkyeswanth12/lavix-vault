"""Concrete OpenAI-compatible/Ollama embedding client with strict dimensions."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from app.graph_memory.inference_priority import BackgroundInferenceDeferred

from .embedding_presets import prepare_texts
from .errors import EmbeddingDimensionError, EmbeddingError, IngestionCancelled


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    url: str = "http://host.docker.internal:11434/v1/embeddings"
    model: str = "snowflake-arctic-embed2:cpu"
    dimension: int = 1024
    # 16-chunk batches: 8-chunk batches starve CPU Ollama on bulk re-ingest.
    batch_size: int = 16
    # A loaded 16-chunk batch needs ~60-120s on single-slot Ollama.
    timeout_seconds: float = 300.0
    max_attempts: int = 3
    max_response_bytes: int = 64 * 1024 * 1024
    # Single-slot Ollama serializes anyway; one in-flight batch per worker.
    concurrency: int = 1

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.batch_size < 1 or self.max_attempts < 1:
            raise ValueError("embedding limits must be positive")
        if self.concurrency < 1:
            raise ValueError("embedding concurrency must be positive")

    @classmethod
    def from_environment(cls) -> EmbeddingSettings:
        base = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
        model = os.environ.get("EMBEDDING_MODEL") or os.environ.get(
            "EMBEDDING_MODEL_NAME", "snowflake-arctic-embed2:cpu"
        )
        dimension = os.environ.get("EMBEDDING_DIMENSIONS") or os.environ.get(
            "EMBEDDING_DIMENSION", "1024"
        )
        return cls(
            url=os.environ.get("EMBEDDING_API_URL", f"{base}/v1/embeddings"),
            model=model,
            dimension=int(dimension),
            batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "16")),
            timeout_seconds=float(os.environ.get("EMBEDDING_TIMEOUT", "300")),
            max_attempts=int(os.environ.get("EMBEDDING_MAX_ATTEMPTS", "3")),
            concurrency=int(os.environ.get("EMBEDDING_CONCURRENCY", "1")),
        )

    @property
    def fingerprint(self) -> str:
        material = json.dumps(
            {
                "protocol": "openai-compatible-embeddings-v1",
                "model": self.model,
                "dimension": self.dimension,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"emb_{hashlib.sha256(material.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class PreparedEmbeddings:
    vectors: tuple[tuple[float, ...], ...]
    model: str
    dimension: int
    fingerprint: str


class OllamaEmbeddingClient:
    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        *,
        priority_gate: Callable[[], Awaitable[bool]] | None = None,
        priority_timeout_seconds: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings or EmbeddingSettings.from_environment()
        self._priority_gate = priority_gate
        self._priority_timeout_seconds = max(0.01, priority_timeout_seconds)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _shared_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(self.settings.timeout_seconds),
                        follow_redirects=False,
                        trust_env=False,
                        transport=self._transport,
                    )
        assert self._client is not None
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def _await_model_call(
        self,
        texts: list[str],
        cancel_event: asyncio.Event | None,
    ) -> Any:
        async def invoke() -> Any:
            result = self._post(texts)
            return await result if inspect.isawaitable(result) else result

        request_task = asyncio.create_task(invoke())
        cancel_task = (
            asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
        )
        try:
            if cancel_task is not None:
                done, _pending = await asyncio.wait(
                    (request_task, cancel_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_task in done:
                    raise IngestionCancelled()
            return await request_task
        finally:
            if cancel_task is not None:
                cancel_task.cancel()
            if not request_task.done():
                request_task.cancel()
            cleanup_tasks = [request_task]
            if cancel_task is not None:
                cleanup_tasks.append(cancel_task)
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    @staticmethod
    async def _retry_delay(seconds: float, cancel_event: asyncio.Event | None) -> None:
        if cancel_event is None:
            await asyncio.sleep(seconds)
            return
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=seconds)
        except TimeoutError:
            return
        raise IngestionCancelled()

    async def _require_priority(self) -> None:
        if self._priority_gate is None:
            return
        try:
            allowed = await asyncio.wait_for(
                self._priority_gate(),
                timeout=self._priority_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise BackgroundInferenceDeferred(
                "foreground inference priority gate is unavailable"
            ) from exc
        if not allowed:
            raise BackgroundInferenceDeferred("foreground inference capacity is unavailable")

    async def embed(
        self,
        texts: Sequence[str],
        *,
        cancel_event: asyncio.Event | None = None,
        kind: str = "document",
    ) -> PreparedEmbeddings:
        if cancel_event and cancel_event.is_set():
            raise IngestionCancelled()
        # Model presets shape inputs identically at index and query time
        # (prefixes, conservative truncation); kind selects the prefix.
        prepared_texts = prepare_texts(self.settings.model, list(texts), kind=kind)
        # Foreground-inference gate is checked once per embed call: an
        # in-flight model request cannot be preempted anyway, so per-batch
        # re-checks only added latency without freeing capacity.
        await self._require_priority()
        batches = [
            prepared_texts[start : start + self.settings.batch_size]
            for start in range(0, len(prepared_texts), self.settings.batch_size)
        ]
        semaphore = asyncio.Semaphore(self.settings.concurrency)

        async def _run(batch: list[str]) -> list[tuple[float, ...]]:
            if cancel_event and cancel_event.is_set():
                raise IngestionCancelled()
            async with semaphore:
                if cancel_event and cancel_event.is_set():
                    raise IngestionCancelled()
                return await self._embed_batch(batch, cancel_event=cancel_event)

        grouped = await asyncio.gather(*(_run(batch) for batch in batches))
        vectors: list[tuple[float, ...]] = [
            vector for group in grouped for vector in group
        ]
        if len(vectors) != len(texts):
            raise EmbeddingError("embedding service returned the wrong vector count")
        return PreparedEmbeddings(
            vectors=tuple(vectors),
            model=self.settings.model,
            dimension=self.settings.dimension,
            fingerprint=self.settings.fingerprint,
        )

    async def _embed_batch(
        self,
        texts: list[str],
        *,
        cancel_event: asyncio.Event | None,
    ) -> list[tuple[float, ...]]:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_attempts):
            if cancel_event and cancel_event.is_set():
                raise IngestionCancelled()
            try:
                payload = await self._await_model_call(texts, cancel_event)
                return self._vectors(payload, len(texts))
            except BackgroundInferenceDeferred:
                raise
            except IngestionCancelled:
                raise
            except EmbeddingDimensionError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.settings.max_attempts:
                    await self._retry_delay(
                        min(4.0, 0.5 * (2**attempt)),
                        cancel_event,
                    )
        raise EmbeddingError("embedding request failed") from last_error

    async def _post(self, texts: list[str]) -> Any:
        body = json.dumps(
            {"model": self.settings.model, "input": texts},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            client = await self._shared_client()
            async with client.stream(
                "POST",
                self.settings.url,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            ) as response:
                response.raise_for_status()
                payload = bytearray()
                async for chunk in response.aiter_bytes():
                    payload.extend(chunk)
                    if len(payload) > self.settings.max_response_bytes:
                        raise EmbeddingError(
                            "embedding response exceeds the configured limit"
                        )
                raw = bytes(payload)
        except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as exc:
            raise EmbeddingError("embedding endpoint is unavailable") from exc
        try:
            return json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise EmbeddingError("embedding endpoint returned invalid JSON") from exc

    def _vectors(self, payload: Any, expected: int) -> list[tuple[float, ...]]:
        raw_vectors: list[Any] = []
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            rows = payload["data"]
            if rows and any(isinstance(row, dict) and "index" in row for row in rows):
                if not all(isinstance(row, dict) and "index" in row for row in rows):
                    raise EmbeddingError("embedding response has inconsistent indices")
                try:
                    rows = sorted(rows, key=lambda row: int(row["index"]))
                    indices = [int(row["index"]) for row in rows]
                except (TypeError, ValueError) as exc:
                    raise EmbeddingError("embedding response has an invalid index") from exc
                if indices != list(range(expected)):
                    raise EmbeddingError("embedding response indices are not contiguous")
            raw_vectors = [row.get("embedding") if isinstance(row, dict) else None for row in rows]
        elif isinstance(payload, dict) and isinstance(payload.get("embeddings"), list):
            raw_vectors = payload["embeddings"]
        elif expected == 1 and isinstance(payload, dict) and isinstance(payload.get("embedding"), list):
            raw_vectors = [payload["embedding"]]
        if len(raw_vectors) != expected:
            raise EmbeddingError("embedding response count mismatch")

        vectors: list[tuple[float, ...]] = []
        for raw in raw_vectors:
            if not isinstance(raw, list) or len(raw) != self.settings.dimension:
                raise EmbeddingDimensionError(f"expected embedding dimension {self.settings.dimension}")
            try:
                vector = tuple(float(value) for value in raw)
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("embedding contains a non-numeric value") from exc
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("embedding contains a non-finite value")
            vectors.append(vector)
        return vectors
