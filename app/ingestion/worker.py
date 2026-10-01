"""Injectable durable ingestion worker and command-line entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
import os
import signal
import socket
import time
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol

from app.graph_memory.inference_priority import BackgroundInferenceDeferred
from app.security.redact import redact_credentials

from .chunking import CanonicalChunker
from .errors import (
    IngestionCancelled,
    IngestionError,
    PasswordRequiredError,
    UnsupportedFormatError,
    bounded_safe_error_detail,
)
from .models import (
    CanonicalChunk,
    CanonicalDocument,
    IngestionJob,
    JobState,
    PublicationResult,
)
from .parsers.base import ParseRequest
from .parsers.registry import ParserRegistry
from .repository import JobRepository
from .routing import ParserKind, route_file

logger = logging.getLogger(__name__)

# Bounded enrichment retry (BUG-007): deferred intelligence re-runs inline a
# fixed number of times, then terminates explicitly to the operator backfill.
# No infinite loop: attempts are counted, delays are fixed, and every path
# lands in a terminal outcome (done/deferred-exhausted/skipped).
_ENRICH_MAX_ATTEMPTS = 3
_ENRICH_RETRY_DELAY_SECONDS = 20.0


def redact_document_text(document: CanonicalDocument) -> CanonicalDocument:
    """Scrub credential values from parsed element texts before chunking.

    Covers chunks, embeddings, and model-derived summaries/tags in one
    move: everything downstream reads the redacted document. Filenames,
    fingerprints, and structure are untouched; only element text changes.
    """
    import dataclasses

    redacted = []
    changed = False
    for element in document.elements:
        text = element.text or ""
        cleaned = redact_credentials(text)
        if not cleaned.strip():
            # An element that was only a secret must not become empty
            # (the frozen model rejects empty text) nor keep the value.
            cleaned = "[REDACTED]"
        if cleaned != text:
            changed = True
        redacted.append(dataclasses.replace(element, text=cleaned))
    if not changed:
        return document
    return dataclasses.replace(document, elements=tuple(redacted))


class SourceProvider(Protocol):
    def open(
        self,
        job: IngestionJob,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> AbstractAsyncContextManager[ParseRequest]: ...


class IndexSink(Protocol):
    async def embed(
        self,
        job: IngestionJob,
        document: CanonicalDocument,
        chunks: Sequence[CanonicalChunk],
        *,
        cancel_event: asyncio.Event,
    ) -> Any: ...
    async def publish(
        self,
        job: IngestionJob,
        document: CanonicalDocument,
        chunks: Sequence[CanonicalChunk],
        prepared: Any,
        *,
        cancel_event: asyncio.Event,
    ) -> PublicationResult | None: ...
    async def enrich(
        self,
        job: IngestionJob,
        document: CanonicalDocument,
        chunks: Sequence[CanonicalChunk],
        *,
        cancel_event: asyncio.Event,
    ) -> str: ...


class WorkerRuntime(Protocol):
    async def run_forever(
        self,
        stop_event: asyncio.Event,
        *,
        idle_seconds: float = 1.0,
    ) -> None: ...


class WorkerHealthReporter(Protocol):
    async def run(self, stop_event: asyncio.Event) -> None: ...

    async def record_pdf_success(self) -> None: ...


class IngestionWorker:
    def __init__(
        self,
        *,
        repository: JobRepository,
        source_provider: SourceProvider,
        sink: IndexSink,
        parser_registry: ParserRegistry | None = None,
        chunker: CanonicalChunker | None = None,
        worker_id: str | None = None,
        lease_seconds: int = 90,
        heartbeat_seconds: float = 20.0,
        cancellation_poll_seconds: float = 0.5,
        retry_base_seconds: int = 5,
        health_reporter: WorkerHealthReporter | None = None,
    ) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        if heartbeat_seconds <= 0 or cancellation_poll_seconds <= 0:
            raise ValueError("worker polling intervals must be positive")
        if retry_base_seconds < 0:
            raise ValueError("retry_base_seconds cannot be negative")
        self.repository = repository
        self.source_provider = source_provider
        self.sink = sink
        self.parsers = parser_registry or ParserRegistry()
        self.chunker = chunker or CanonicalChunker()
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.cancellation_poll_seconds = cancellation_poll_seconds
        self.retry_base_seconds = retry_base_seconds
        self.health_reporter = health_reporter

    async def run_forever(
        self,
        stop_event: asyncio.Event,
        *,
        idle_seconds: float = 1.0,
    ) -> None:
        health_stop_event = asyncio.Event()
        health_task = (
            asyncio.create_task(self.health_reporter.run(health_stop_event))
            if self.health_reporter is not None
            else None
        )
        failure_streak = 0
        try:
            while not stop_event.is_set():
                try:
                    claimed = await self.run_once()
                    failure_streak = 0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # A transient database/network outage must not terminate the
                    # long-running worker. Any claimed job remains lease-fenced
                    # and can be recovered after its lease expires.
                    failure_streak += 1
                    retry_seconds = min(
                        30.0,
                        max(1.0, idle_seconds) * (2 ** min(failure_streak - 1, 5)),
                    )
                    logger.warning(
                        "Ingestion worker iteration failed (%s); retrying in %.1fs",
                        type(exc).__name__,
                        retry_seconds,
                    )
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=retry_seconds,
                        )
                    except TimeoutError:
                        pass
                    continue
                if claimed:
                    continue
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=idle_seconds)
                except TimeoutError:
                    pass
        finally:
            if health_task is not None:
                health_stop_event.set()
                health_task.cancel()
                await asyncio.gather(health_task, return_exceptions=True)

    async def run_once(self) -> bool:
        job = await asyncio.to_thread(self.repository.claim_next, self.worker_id, self.lease_seconds)
        if job is None:
            return False
        await self._process(job)
        return True

    async def _process(self, job: IngestionJob) -> None:
        state = JobState.DECRYPTING
        source_path = ""
        cancel_event = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(job, cancel_event))
        cancellation = asyncio.create_task(self._watch_cancellation(job, cancel_event))
        stage_times: dict[str, float] = {}
        wall_start = time.perf_counter()
        try:
            self._raise_if_cancelled(cancel_event)
            stage_start = time.perf_counter()
            async with self.source_provider.open(job, cancel_event=cancel_event) as request:
                stage_times["decrypt_s"] = round(time.perf_counter() - stage_start, 2)
                stage_start = time.perf_counter()
                source_path = str(request.path)
                self._raise_if_cancelled(cancel_event)
                route = route_file(request.source_name, request.media_type)
                if not route.supported:
                    raise UnsupportedFormatError(route.reason)

                if route.parser is ParserKind.LIBREOFFICE_DOCLING:
                    await self._transition(job, state, JobState.CONVERTING)
                    state = JobState.CONVERTING
                else:
                    await self._transition(job, state, JobState.PARSING)
                    state = JobState.PARSING

                document = await self.parsers.parse(route, request, cancel_event=cancel_event)
                stage_times["parse_s"] = round(time.perf_counter() - stage_start, 2)
                document = redact_document_text(document)
                if route.parser is ParserKind.OPEN_DATALOADER_PDF and self.health_reporter is not None:
                    await self.health_reporter.record_pdf_success()
                self._raise_if_cancelled(cancel_event)
                if state is JobState.CONVERTING:
                    await self._transition(
                        job,
                        state,
                        JobState.PARSING,
                        parser_fingerprint=document.parser_fingerprint,
                    )
                    state = JobState.PARSING

            # The source context owns decrypted plaintext. Leave it immediately
            # after parsing so chunking and model calls cannot extend its lifetime.
            await self._transition(
                job,
                state,
                JobState.CHUNKING,
                parser_fingerprint=document.parser_fingerprint,
            )
            state = JobState.CHUNKING
            stage_start = time.perf_counter()
            chunks = await asyncio.to_thread(self.chunker.chunk, document)
            stage_times["chunk_s"] = round(time.perf_counter() - stage_start, 2)
            stage_times["num_chunks"] = len(chunks)
            if not chunks:
                from .errors import EmptyDocumentError

                raise EmptyDocumentError("parser output produced no chunks")
            self._raise_if_cancelled(cancel_event)

            await self._transition(job, state, JobState.EMBEDDING)
            state = JobState.EMBEDDING
            stage_start = time.perf_counter()
            prepared = await self.sink.embed(job, document, chunks, cancel_event=cancel_event)
            stage_times["embed_s"] = round(time.perf_counter() - stage_start, 2)
            self._raise_if_cancelled(cancel_event)

            await self._transition(job, state, JobState.PUBLISHING)
            state = JobState.PUBLISHING
            stage_start = time.perf_counter()
            result = await self.sink.publish(job, document, chunks, prepared, cancel_event=cancel_event)
            stage_times["publish_s"] = round(time.perf_counter() - stage_start, 2)
            await self._enrich_best_effort(job, document, chunks, cancel_event, stage_times)
            stage_times["wall_s"] = round(time.perf_counter() - wall_start, 2)
            logger.info(
                "Ingestion stage timings job=%s file=%s stages=%s",
                job.job_id,
                job.source_name,
                stage_times,
            )
            if result is not None and result.job_completed:
                return
            self._raise_if_cancelled(cancel_event)
            changed = await asyncio.to_thread(
                self.repository.finish,
                job,
                self.worker_id,
                state,
                JobState.READY,
            )
            if not changed:
                raise IngestionCancelled("revision fence rejected ready publication")
        except BackgroundInferenceDeferred:
            logger.info(
                "Deferring ingestion job %s while foreground inference owns capacity",
                job.job_id,
            )
            await self._defer_best_effort(job, state)
        except IngestionCancelled as exc:
            await self._finish_best_effort(job, state, JobState.CANCELLED, exc.code, str(exc))
        except PasswordRequiredError as exc:
            await self._finish_best_effort(job, state, JobState.PASSWORD_REQUIRED, exc.code, str(exc))
        except UnsupportedFormatError as exc:
            await self._finish_best_effort(job, state, JobState.UNSUPPORTED, exc.code, str(exc))
        except IngestionError as exc:
            error_detail = bounded_safe_error_detail(
                exc.detail or str(exc),
                replacements=(job.source_name, source_path),
            )
            if exc.retryable and job.can_retry:
                logger.warning(
                    "Retrying ingestion job %s after %s: %s",
                    job.job_id,
                    exc.code,
                    error_detail,
                )
                await self._requeue_best_effort(job, state, exc.code, error_detail)
            else:
                logger.warning(
                    "Failing ingestion job %s after %s: %s",
                    job.job_id,
                    exc.code,
                    error_detail,
                )
                await self._finish_best_effort(
                    job,
                    state,
                    JobState.FAILED,
                    exc.code,
                    error_detail,
                )
        except Exception as exc:  # defensive worker boundary
            error_detail = (
                bounded_safe_error_detail(
                    str(exc),
                    replacements=(job.source_name, source_path),
                )
                or type(exc).__name__
            )
            logger.error(
                "Unhandled ingestion failure for job %s (%s): %s",
                job.job_id,
                type(exc).__name__,
                error_detail,
            )
            if job.can_retry:
                await self._requeue_best_effort(job, state, "internal_error", error_detail)
            else:
                await self._finish_best_effort(
                    job,
                    state,
                    JobState.FAILED,
                    "internal_error",
                    error_detail,
                )
        finally:
            cancel_event.set()
            for task in (heartbeat, cancellation):
                task.cancel()
            await asyncio.gather(heartbeat, cancellation, return_exceptions=True)

    async def _enrich_best_effort(
        self,
        job: IngestionJob,
        document: CanonicalDocument,
        chunks: Sequence[CanonicalChunk],
        cancel_event: asyncio.Event,
        stage_times: dict[str, float],
    ) -> None:
        """Model metadata upgrade after chunks are searchable. Never fails the job.

        Deferred enrichment (foreground inference owned the model) retries a
        bounded number of times, then terminates explicitly to the operator
        backfill instead of stranding the row in pending forever (BUG-007).
        """
        enrich = getattr(self.sink, "enrich", None)
        if enrich is None:
            return
        # Post-commit upgrade: the job's cancel event is poisoned by design
        # once publish clears the lease (should_cancel goes true), so enrich
        # runs under a fresh event. Shutdown mid-enrich is harmless: the
        # enrich UPDATE is revision-fenced and the backfill covers leftovers.
        stage_start = time.perf_counter()
        outcome = "skipped"
        try:
            for attempt in range(1, _ENRICH_MAX_ATTEMPTS + 1):
                try:
                    outcome = await enrich(job, document, chunks, cancel_event=asyncio.Event())
                except Exception as exc:
                    logger.warning(
                        "Intelligence enrich failed job=%s (%s); row stays pending for backfill",
                        job.job_id,
                        type(exc).__name__,
                    )
                    outcome = "deferred"
                if outcome != "deferred":
                    break
                if attempt < _ENRICH_MAX_ATTEMPTS:
                    logger.warning(
                        "Intelligence enrich deferred job=%s attempt=%d/%d; retrying",
                        job.job_id,
                        attempt,
                        _ENRICH_MAX_ATTEMPTS,
                    )
                    await asyncio.sleep(_ENRICH_RETRY_DELAY_SECONDS)
            if outcome == "deferred":
                logger.warning(
                    "Intelligence enrich exhausted job=%s after %d attempts; "
                    "row stays pending for operator backfill "
                    "(python -m app.ingestion.intelligence_backfill --apply)",
                    job.job_id,
                    _ENRICH_MAX_ATTEMPTS,
                )
        finally:
            stage_times["enrich_s"] = round(time.perf_counter() - stage_start, 2)

    async def _transition(
        self,
        job: IngestionJob,
        current: JobState,
        target: JobState,
        *,
        parser_fingerprint: str | None = None,
    ) -> None:
        changed = await asyncio.to_thread(
            self.repository.transition,
            job,
            self.worker_id,
            current,
            target,
            parser_fingerprint=parser_fingerprint,
        )
        if not changed:
            raise IngestionCancelled(
                f"lease, consent, cancellation, or revision fence rejected {target.value}"
            )

    async def _heartbeat(self, job: IngestionJob, cancel_event: asyncio.Event) -> None:
        while not cancel_event.is_set():
            await asyncio.sleep(self.heartbeat_seconds)
            if cancel_event.is_set():
                return
            healthy = await asyncio.to_thread(
                self.repository.heartbeat, job, self.worker_id, self.lease_seconds
            )
            if not healthy:
                cancel_event.set()
                return

    async def _watch_cancellation(self, job: IngestionJob, cancel_event: asyncio.Event) -> None:
        while not cancel_event.is_set():
            requested = await asyncio.to_thread(self.repository.should_cancel, job, self.worker_id)
            if requested:
                cancel_event.set()
                return
            await asyncio.sleep(self.cancellation_poll_seconds)

    @staticmethod
    def _raise_if_cancelled(cancel_event: asyncio.Event) -> None:
        if cancel_event.is_set():
            raise IngestionCancelled()

    async def _finish_best_effort(
        self,
        job: IngestionJob,
        current: JobState,
        target: JobState,
        code: str,
        detail: str,
    ) -> None:
        try:
            await asyncio.to_thread(
                self.repository.finish,
                job,
                self.worker_id,
                current,
                target,
                error_code=code,
                error_detail=detail,
            )
        except Exception:
            logger.exception("Failed to persist terminal state for job %s", job.job_id)

    async def _requeue_best_effort(
        self,
        job: IngestionJob,
        current: JobState,
        code: str,
        detail: str,
    ) -> None:
        delay = min(300, self.retry_base_seconds * (2 ** max(0, job.attempts - 1)))
        try:
            await asyncio.to_thread(
                self.repository.requeue,
                job,
                self.worker_id,
                current,
                delay_seconds=delay,
                error_code=code,
                error_detail=detail,
            )
        except Exception:
            logger.exception("Failed to requeue job %s", job.job_id)

    async def _defer_best_effort(self, job: IngestionJob, current: JobState) -> None:
        delay = max(1, min(60, self.retry_base_seconds))
        try:
            changed = await asyncio.to_thread(
                self.repository.defer,
                job,
                self.worker_id,
                current,
                delay_seconds=delay,
            )
            if not changed:
                logger.warning(
                    "Unable to defer ingestion job %s because its fence changed",
                    job.job_id,
                )
        except Exception:
            logger.exception("Failed to defer ingestion job %s", job.job_id)


def _load_factory(spec: str) -> WorkerRuntime:
    if ":" not in spec:
        raise ValueError("worker factory must use module:callable syntax")
    module_name, attribute = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    value = factory()
    if inspect.isawaitable(value):
        raise TypeError("worker factory must be synchronous")
    # ``python -m app.ingestion.worker`` executes this file as ``__main__``.
    # A factory imports the canonical module name, so strict class identity
    # would reject the same class loaded under those two module names.  The CLI
    # only requires this narrow runtime interface and validates it directly.
    if not callable(getattr(value, "run_forever", None)):
        raise TypeError("worker factory did not return a worker runtime")
    return value


async def _run_cli(factory_spec: str, idle_seconds: float) -> None:
    worker = _load_factory(factory_spec)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass
    await worker.run_forever(stop_event, idle_seconds=idle_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lavix durable ingestion worker")
    parser.add_argument(
        "--factory",
        default=os.environ.get("INGESTION_WORKER_FACTORY", "app.ingestion.bootstrap:create_worker"),
        help="module:callable returning a fully wired IngestionWorker",
    )
    parser.add_argument("--idle-seconds", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    from app.db.embedding_check import check_embedding_deployment

    # Same fail-closed dimension check as the API: never publish vectors
    # into a column built for a different model/dimensions.
    check_embedding_deployment()
    asyncio.run(_run_cli(args.factory, args.idle_seconds))


if __name__ == "__main__":
    main()
