"""Production wiring for ``python -m app.ingestion.worker``."""

from __future__ import annotations

import os
import socket

from app.graph_memory.inference_priority import background_inference_allowed
from app.services.model_config import ModelConfigurationRepository

from .chunking import CanonicalChunker, ChunkingPolicy
from .embedding import EmbeddingSettings, OllamaEmbeddingClient
from .health import OpenDataLoaderReadinessProbe, RedisWorkerHealthReporter
from .intelligence import ConfiguredDocumentIntelligenceService, IntelligenceSettings
from .parsers.registry import ParserRegistry, ParserSettings
from .publisher import PgVectorIndexSink
from .repository import PostgresJobRepository
from .source import VaultEncryptedSourceProvider, VaultSourceSettings
from .worker import IngestionWorker


def create_worker() -> IngestionWorker:
    """Build the durable worker; database and heavy parser imports remain lazy."""

    from app.database import get_db

    def resolve_role(name: str) -> tuple[bool, str | None]:
        with get_db() as connection:
            configuration = ModelConfigurationRepository(connection).get()
        role = getattr(configuration, name)
        return role.enabled, role.model

    chunker = CanonicalChunker(ChunkingPolicy())
    embedding_settings = EmbeddingSettings.from_environment()
    # Dimension-vs-column agreement is enforced by check_embedding_deployment
    # at process start (fail-closed with a reembed pointer); the sink below
    # additionally stamps every revision with the produced dimensions.
    embeddings = OllamaEmbeddingClient(
        embedding_settings,
        priority_gate=background_inference_allowed,
        priority_timeout_seconds=float(os.environ.get("INTELLIGENCE_PRIORITY_WAIT_SECONDS", "2")),
    )
    sink = PgVectorIndexSink(
        get_db,
        embeddings,
        chunker_fingerprint=chunker.policy.fingerprint,
        expected_dimension=embedding_settings.dimension,
        intelligence=ConfiguredDocumentIntelligenceService(
            lambda: resolve_role("intelligence"),
            IntelligenceSettings.from_environment(),
            priority_gate=background_inference_allowed,
        ),
    )
    worker_id = os.environ.get("INGESTION_WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"
    parser_settings = ParserSettings.from_environment()
    parser_registry = ParserRegistry(
        parser_settings,
        vision_role_resolver=lambda: resolve_role("vision"),
        vision_priority_gate=background_inference_allowed,
        vision_priority_timeout_seconds=float(
            os.environ.get("INTELLIGENCE_PRIORITY_WAIT_SECONDS", "2")
        ),
    )
    health_reporter = RedisWorkerHealthReporter(
        os.environ.get("REDIS_URL", "redis://redis:6379/0"),
        worker_id,
        OpenDataLoaderReadinessProbe(parser_registry, parser_settings.temp_root),
        heartbeat_seconds=float(os.environ.get("INGESTION_HEALTH_HEARTBEAT_SECONDS", "20")),
        probe_seconds=float(os.environ.get("INGESTION_HEALTH_PROBE_SECONDS", "900")),
        ttl_seconds=int(os.environ.get("INGESTION_HEALTH_TTL_SECONDS", "75")),
    )
    return IngestionWorker(
        repository=PostgresJobRepository(get_db),
        source_provider=VaultEncryptedSourceProvider(VaultSourceSettings.from_environment()),
        sink=sink,
        parser_registry=parser_registry,
        chunker=chunker,
        worker_id=worker_id,
        lease_seconds=int(os.environ.get("INGESTION_LEASE_SECONDS", "90")),
        heartbeat_seconds=float(os.environ.get("INGESTION_HEARTBEAT_SECONDS", "20")),
        cancellation_poll_seconds=float(os.environ.get("INGESTION_CANCELLATION_POLL_SECONDS", "0.5")),
        retry_base_seconds=int(os.environ.get("INGESTION_RETRY_BASE_SECONDS", "5")),
        health_reporter=health_reporter,
    )
