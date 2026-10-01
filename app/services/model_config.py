"""Revisioned system AI-model configuration with environment fallbacks.

PostgreSQL is authoritative after the first admin save.  Until then, the
singleton table is empty and the resolver reproduces the existing deployment
environment.  This keeps migrations non-disruptive while giving every caller a
single configuration contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from typing import Any

from app.config import settings

_SELECT_COLUMNS = """
    revision,
    chat_enabled,
    chat_default_model,
    chat_allowed_models,
    vision_enabled,
    vision_model,
    intelligence_enabled,
    intelligence_model,
    memory_extraction_enabled,
    memory_extraction_model,
    reranker_enabled,
    clock_timezone,
    model_max_num_ctx,
    fallback_chat_model,
    file_scope,
    top_k,
    search_depth,
    session_timeout_minutes,
    registration_enabled,
    updated_by,
    created_at,
    updated_at
"""

#: Fixed allowlist for the admin context ceiling. Free-text values risk
#: OOM kills (262144 proven fatal on this hardware); only these ship.
MODEL_MAX_NUM_CTX_OPTIONS: tuple[int, ...] = (16384, 32768, 65536, 131072, 262144)
DEFAULT_MODEL_MAX_NUM_CTX = 16384

#: Admin session-timeout choices (minutes). One value drives both the idle
#: lock and the access-token lifetime for newly issued sessions/tokens.
SESSION_TIMEOUT_OPTIONS: tuple[int, ...] = (15, 30, 60, 120, 240, 480, 1440)
DEFAULT_SESSION_TIMEOUT_MINUTES = 120

#: Admin web search depth tiers. The tier->value preset table lives in the
#: agent runtime (agent_runtime.cuga_adapter), the only consumer; this module
#: keeps the allowlist + default for the admin surface and DB row handling.
SEARCH_DEPTH_OPTIONS: tuple[str, ...] = ("conservative", "balanced", "deep", "pro")
DEFAULT_SEARCH_DEPTH = "conservative"

#: RAG retrieval scope bounds. file_scope caps distinct files in
#: discovery/unscoped retrieval (explicitly tagged files are always fully
#: represented); top_k caps candidate chunks from vector retrieval.
#: Defaults preserve historical effective behavior.
FILE_SCOPE_MIN = 1
FILE_SCOPE_MAX = 100
DEFAULT_FILE_SCOPE = 5
TOP_K_MIN = 1
TOP_K_MAX = 500
DEFAULT_TOP_K = 20


class ModelConfigurationConflict(RuntimeError):
    """Raised when an admin writes against a stale configuration revision."""

    def __init__(self, *, expected_revision: int, current_revision: int) -> None:
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        super().__init__(
            f"AI model configuration revision conflict: expected "
            f"{expected_revision}, current {current_revision}"
        )


@dataclass(frozen=True, slots=True)
class ChatModelConfiguration:
    enabled: bool
    default_model: str
    allowed_models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OptionalModelConfiguration:
    enabled: bool
    model: str | None


@dataclass(frozen=True, slots=True)
class SystemAIConfiguration:
    revision: int
    source: str
    chat: ChatModelConfiguration
    vision: OptionalModelConfiguration
    intelligence: OptionalModelConfiguration
    memory_extraction: OptionalModelConfiguration
    reranker_enabled: bool
    clock_timezone: str | None = None
    model_max_num_ctx: int = DEFAULT_MODEL_MAX_NUM_CTX
    fallback_chat_model: str | None = None
    file_scope: int = DEFAULT_FILE_SCOPE
    top_k: int = DEFAULT_TOP_K
    search_depth: str = DEFAULT_SEARCH_DEPTH
    session_timeout_minutes: int | None = None
    registration_enabled: bool | None = None
    updated_by: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SystemAIConfigurationWrite:
    chat: ChatModelConfiguration
    vision: OptionalModelConfiguration
    intelligence: OptionalModelConfiguration
    memory_extraction: OptionalModelConfiguration
    reranker_enabled: bool
    clock_timezone: str | None
    model_max_num_ctx: int = DEFAULT_MODEL_MAX_NUM_CTX
    fallback_chat_model: str | None = None
    file_scope: int = DEFAULT_FILE_SCOPE
    top_k: int = DEFAULT_TOP_K
    search_depth: str = DEFAULT_SEARCH_DEPTH
    session_timeout_minutes: int | None = None
    registration_enabled: bool | None = None


@dataclass(frozen=True, slots=True)
class AccountChatModelResolution:
    """One account's effective Chat model under current runtime availability."""

    preferred_model: str | None
    active_model: str | None
    available: bool
    fallback_used: bool = False


def _unique_models(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def deployment_chat_model_ceiling() -> tuple[str, ...] | None:
    """Return the immutable deployment ceiling, or None when unconstrained.

    Empty/unset ALLOWED_CHAT_MODELS leaves model choice to the admin via
    Settings; a non-empty value pins the deployment and admin saves
    outside it are rejected with 422.
    """

    pinned = settings.chat_model_ceiling
    return None if pinned is None else _unique_models(pinned)


def resolve_account_chat_model(
    config: SystemAIConfiguration,
    preferred_model: str | None,
    installed_models: Sequence[str],
) -> AccountChatModelResolution:
    """Resolve the same account model for Settings and real Chat execution.

    A preference is effective only while it remains both system-allowlisted and
    installed.  The configured system default remains visible when Ollama is
    unavailable so Settings can report the intended model as unavailable rather
    than silently inventing a different choice.  When neither preference nor
    default is usable, the admin-chosen fallback serves (flagged via
    fallback_used); a stale or uninstalled fallback behaves as unset.
    """

    if not config.chat.enabled:
        return AccountChatModelResolution(
            preferred_model=None,
            active_model=None,
            available=False,
        )

    allowed_models = frozenset(config.chat.allowed_models)
    installed_names = frozenset(
        model.strip() for model in installed_models if model and model.strip()
    )
    normalized_preference = (preferred_model or "").strip() or None
    effective_preference = (
        normalized_preference
        if normalized_preference in allowed_models and normalized_preference in installed_names
        else None
    )
    candidate = effective_preference or config.chat.default_model or None
    if candidate and candidate in installed_names:
        return AccountChatModelResolution(
            preferred_model=effective_preference,
            active_model=candidate,
            available=True,
        )
    fallback = (getattr(config, "fallback_chat_model", None) or "").strip() or None
    if (
        fallback
        and fallback in allowed_models
        and fallback in installed_names
    ):
        return AccountChatModelResolution(
            preferred_model=effective_preference,
            active_model=fallback,
            available=True,
            fallback_used=True,
        )
    return AccountChatModelResolution(
        preferred_model=effective_preference,
        active_model=candidate,
        available=False,
    )


def environment_model_configuration() -> SystemAIConfiguration:
    """Resolve the legacy environment into the unified configuration shape."""

    allowed_models = deployment_chat_model_ceiling()
    configured_default = settings.llm_model.strip()
    if allowed_models is None:
        allowed_models = _unique_models([configured_default])
    if configured_default not in allowed_models:
        configured_default = allowed_models[0] if allowed_models else configured_default

    vision_model = settings.vision_model.strip() or None
    intelligence_model = settings.intelligence_model.strip() or None
    # Memory extraction did not previously have an independent deployment
    # role.  It safely follows Document Intelligence until an admin saves an
    # explicit role configuration.
    return SystemAIConfiguration(
        revision=0,
        source="environment",
        chat=ChatModelConfiguration(
            enabled=bool(configured_default and allowed_models),
            default_model=configured_default,
            allowed_models=allowed_models,
        ),
        vision=OptionalModelConfiguration(enabled=bool(vision_model), model=vision_model),
        intelligence=OptionalModelConfiguration(
            enabled=bool(intelligence_model),
            model=intelligence_model,
        ),
        memory_extraction=OptionalModelConfiguration(
            enabled=bool(intelligence_model),
            model=intelligence_model,
        ),
        reranker_enabled=settings.enable_reranking,
        clock_timezone=None,
        file_scope=DEFAULT_FILE_SCOPE,
        top_k=DEFAULT_TOP_K,
        search_depth=DEFAULT_SEARCH_DEPTH,
    )


def _row_value(row: Mapping[str, Any], key: str, fallback: Any) -> Any:
    value = row.get(key)
    return fallback if value is None else value


def _clean_search_depth(value: Any) -> str:
    cleaned = str(value or "").strip().lower()
    return cleaned if cleaned in SEARCH_DEPTH_OPTIONS else DEFAULT_SEARCH_DEPTH


def _clean_session_timeout_minutes(value: Any) -> int | None:
    """Return a valid saved timeout, else None (fall back to deployment)."""

    if value is None:
        return None
    try:
        minutes = int(value)
    except (TypeError, ValueError):
        return None
    return minutes if minutes in SESSION_TIMEOUT_OPTIONS else None


def _clean_registration_enabled(value: Any) -> bool | None:
    """Return a saved boolean toggle, else None (fall back to deployment)."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _resolve_row(row: Mapping[str, Any]) -> SystemAIConfiguration:
    fallback = environment_model_configuration()
    allowed_models = _unique_models(
        _row_value(row, "chat_allowed_models", fallback.chat.allowed_models)
    )
    default_model = str(
        _row_value(row, "chat_default_model", fallback.chat.default_model)
    ).strip()
    if default_model not in allowed_models:
        default_model = allowed_models[0] if allowed_models else ""

    def optional_role(prefix: str, default: OptionalModelConfiguration) -> OptionalModelConfiguration:
        raw_model = _row_value(row, f"{prefix}_model", default.model)
        model = str(raw_model).strip() if raw_model else None
        return OptionalModelConfiguration(
            enabled=bool(_row_value(row, f"{prefix}_enabled", default.enabled)),
            model=model,
        )

    return SystemAIConfiguration(
        revision=int(row["revision"]),
        source="database",
        chat=ChatModelConfiguration(
            enabled=bool(_row_value(row, "chat_enabled", fallback.chat.enabled)),
            default_model=default_model,
            allowed_models=allowed_models,
        ),
        vision=optional_role("vision", fallback.vision),
        intelligence=optional_role("intelligence", fallback.intelligence),
        memory_extraction=optional_role("memory_extraction", fallback.memory_extraction),
        reranker_enabled=bool(
            _row_value(row, "reranker_enabled", fallback.reranker_enabled)
        ),
        clock_timezone=_clean_clock_timezone(row.get("clock_timezone")),
        model_max_num_ctx=int(
            _row_value(row, "model_max_num_ctx", DEFAULT_MODEL_MAX_NUM_CTX)
        ),
        fallback_chat_model=_clean_model_name(row.get("fallback_chat_model")),
        file_scope=int(_row_value(row, "file_scope", DEFAULT_FILE_SCOPE)),
        top_k=int(_row_value(row, "top_k", DEFAULT_TOP_K)),
        search_depth=_clean_search_depth(_row_value(row, "search_depth", DEFAULT_SEARCH_DEPTH)),
        session_timeout_minutes=_clean_session_timeout_minutes(row.get("session_timeout_minutes")),
        registration_enabled=_clean_registration_enabled(row.get("registration_enabled")),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _clean_clock_timezone(value: Any) -> str | None:
    """Normalize a stored clock override; blank means the deployment default."""

    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _clean_model_name(value: Any) -> str | None:
    """Normalize a stored optional model name; blank means unset."""

    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def resolve_clock_timezone(config: SystemAIConfiguration) -> str:
    """Return the effective IANA zone: Settings override, else deployment default."""

    return config.clock_timezone or settings.clock_timezone


def clock_timezone_source(config: SystemAIConfiguration) -> str:
    """Report where the effective clock zone comes from for Settings display."""

    return "database" if config.clock_timezone else "environment"


class ModelConfigurationRepository:
    """Read and atomically replace the singleton model configuration."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def get(self) -> SystemAIConfiguration:
        cursor = self._connection.cursor()
        cursor.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM system_ai_configuration
            WHERE singleton_id = 1
            """
        )
        row = cursor.fetchone()
        return _resolve_row(row) if row else environment_model_configuration()

    def replace(
        self,
        value: SystemAIConfigurationWrite,
        *,
        expected_revision: int,
        updated_by: int,
    ) -> SystemAIConfiguration:
        """Atomically replace all writable roles using optimistic concurrency."""

        cursor = self._connection.cursor()
        cursor.execute(
            f"""
            UPDATE system_ai_configuration
            SET revision = revision + 1,
                chat_enabled = %s,
                chat_default_model = %s,
                chat_allowed_models = %s,
                vision_enabled = %s,
                vision_model = %s,
                intelligence_enabled = %s,
                intelligence_model = %s,
                memory_extraction_enabled = %s,
                memory_extraction_model = %s,
                reranker_enabled = %s,
                clock_timezone = %s,
                model_max_num_ctx = %s,
                fallback_chat_model = %s,
                file_scope = %s,
                top_k = %s,
                search_depth = %s,
                session_timeout_minutes = %s,
                registration_enabled = %s,
                updated_by = %s,
                updated_at = NOW()
            WHERE singleton_id = 1 AND revision = %s
            RETURNING {_SELECT_COLUMNS}
            """,
            (
                value.chat.enabled,
                value.chat.default_model,
                list(value.chat.allowed_models),
                value.vision.enabled,
                value.vision.model,
                value.intelligence.enabled,
                value.intelligence.model,
                value.memory_extraction.enabled,
                value.memory_extraction.model,
                value.reranker_enabled,
                value.clock_timezone,
                value.model_max_num_ctx,
                value.fallback_chat_model,
                value.file_scope,
                value.top_k,
                value.search_depth,
                value.session_timeout_minutes,
                value.registration_enabled,
                updated_by,
                expected_revision,
            ),
        )
        row = cursor.fetchone()
        if row is None and expected_revision == 0:
            # The migration intentionally leaves the singleton absent.  A
            # first save may create revision 1, but ON CONFLICT keeps two
            # simultaneous first writers from both succeeding.
            cursor.execute(
                f"""
                INSERT INTO system_ai_configuration (
                    singleton_id,
                    revision,
                    chat_enabled,
                    chat_default_model,
                    chat_allowed_models,
                    vision_enabled,
                    vision_model,
                    intelligence_enabled,
                    intelligence_model,
                    memory_extraction_enabled,
                    memory_extraction_model,
                    reranker_enabled,
                    clock_timezone,
                    model_max_num_ctx,
                    fallback_chat_model,
                    file_scope,
                    top_k,
                    search_depth,
                    session_timeout_minutes,
                    registration_enabled,
                    updated_by
                )
                VALUES (1, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (singleton_id) DO NOTHING
                RETURNING {_SELECT_COLUMNS}
                """,
                (
                    value.chat.enabled,
                    value.chat.default_model,
                    list(value.chat.allowed_models),
                    value.vision.enabled,
                    value.vision.model,
                    value.intelligence.enabled,
                    value.intelligence.model,
                    value.memory_extraction.enabled,
                    value.memory_extraction.model,
                    value.reranker_enabled,
                    value.clock_timezone,
                    value.model_max_num_ctx,
                    value.fallback_chat_model,
                    value.file_scope,
                    value.top_k,
                    value.search_depth,
                    value.session_timeout_minutes,
                    value.registration_enabled,
                    updated_by,
                ),
            )
            row = cursor.fetchone()
        if row is None:
            cursor.execute(
                "SELECT revision FROM system_ai_configuration WHERE singleton_id = 1"
            )
            current = cursor.fetchone()
            current_revision = int(current["revision"]) if current else 0
            raise ModelConfigurationConflict(
                expected_revision=expected_revision,
                current_revision=current_revision,
            )

        # A removed model cannot remain an apparently valid account choice.
        # This update is part of the same transaction as the configuration
        # revision, so every subsequent reader observes one coherent state.
        cursor.execute(
            """
            UPDATE users
            SET preferred_chat_model = NULL
            WHERE preferred_chat_model IS NOT NULL
              AND NOT (preferred_chat_model = ANY(%s::text[]))
            """,
            (list(value.chat.allowed_models),),
        )
        return _resolve_row(row)


def get_system_ai_configuration(connection: Any) -> SystemAIConfiguration:
    """Convenience entry point for request paths and workers."""

    return ModelConfigurationRepository(connection).get()


#: TTL for saved admin-preference reads (seconds). The API may run several
#: workers/processes, so no write is visible everywhere instantly; a 5-second
#: ceiling bounds staleness without a database round trip per request. Writes
#: invalidate immediately via invalidate_preferences_cache().
PREFERENCES_CACHE_TTL_SECONDS = 5.0

_preferences_cache: dict[str, Any] = {"expires_at": 0.0, "timeout": None, "registration": None}


def invalidate_preferences_cache() -> None:
    """Drop cached admin preferences (call after a successful admin save)."""

    _preferences_cache["expires_at"] = 0.0
    _preferences_cache["timeout"] = None
    _preferences_cache["registration"] = None


def _read_saved_preferences() -> tuple[int | None, bool | None]:
    """Read raw saved preferences, honoring the TTL cache.

    Returns (session_timeout_minutes | None, registration_enabled | None)
    where None means "not saved, fall back to deployment configuration".
    Failures fail open to deployment configuration (logged by callers' paths).
    """

    now = monotonic()
    if now < float(_preferences_cache["expires_at"]):
        return (
            _preferences_cache["timeout"],
            _preferences_cache["registration"],
        )
    timeout: int | None = None
    registration: bool | None = None
    try:
        from app.database import get_db

        with get_db() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT session_timeout_minutes, registration_enabled
                FROM system_ai_configuration
                WHERE singleton_id = 1
                """
            )
            row = cursor.fetchone()
        if row:
            timeout = _clean_session_timeout_minutes(row.get("session_timeout_minutes"))
            registration = _clean_registration_enabled(row.get("registration_enabled"))
    except Exception:
        # Table may predate migration 0021 on old checkouts, or the database
        # may be briefly unreachable: deployment configuration stays in force.
        timeout, registration = None, None
    _preferences_cache["expires_at"] = now + PREFERENCES_CACHE_TTL_SECONDS
    _preferences_cache["timeout"] = timeout
    _preferences_cache["registration"] = registration
    return timeout, registration


def get_saved_session_timeout_minutes() -> int | None:
    """Saved admin session timeout, or None when the deployment default rules."""

    timeout, _ = _read_saved_preferences()
    return timeout


def get_saved_registration_enabled() -> bool | None:
    """Saved admin registration toggle, or None when the deployment default rules."""

    _, registration = _read_saved_preferences()
    return registration
