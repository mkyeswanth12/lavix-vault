"""Administrative user, policy, storage, and service diagnostics."""

import logging
from datetime import datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import redis
from fastapi import APIRouter, Depends, HTTPException
from minio import Minio
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from app.auth import get_current_user, get_password_hash, validate_bcrypt_password
from app.config import settings
from app.database import get_db
from app.ingestion.health import WORKER_HEALTH_KEY, decode_worker_health
from app.services.model_config import (
    DEFAULT_FILE_SCOPE,
    DEFAULT_MODEL_MAX_NUM_CTX,
    DEFAULT_SEARCH_DEPTH,
    DEFAULT_TOP_K,
    FILE_SCOPE_MAX,
    FILE_SCOPE_MIN,
    MODEL_MAX_NUM_CTX_OPTIONS,
    SEARCH_DEPTH_OPTIONS,
    SESSION_TIMEOUT_OPTIONS,
    TOP_K_MAX,
    TOP_K_MIN,
    ChatModelConfiguration,
    ModelConfigurationConflict,
    ModelConfigurationRepository,
    OptionalModelConfiguration,
    SystemAIConfiguration,
    SystemAIConfigurationWrite,
    clock_timezone_source,
    deployment_chat_model_ceiling,
    invalidate_preferences_cache,
    resolve_clock_timezone,
)
from app.services.model_service import get_model_service

router = APIRouter()
logger = logging.getLogger(__name__)
CurrentUser = Annotated[dict, Depends(get_current_user)]
BcryptPassword = Annotated[
    str,
    Field(min_length=1, max_length=72),
    AfterValidator(validate_bcrypt_password),
]


def _validate_policy_name(value: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("policy name must not be blank")
    return normalized


PolicyName = Annotated[
    str,
    Field(min_length=1, max_length=100),
    AfterValidator(_validate_policy_name),
]


class AdminRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


def require_admin(user: CurrentUser) -> dict:
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


CurrentAdmin = Annotated[dict, Depends(require_admin)]


# ── AI MODEL CONTROL PLANE ────────────────────────────────────────────────


def _normalize_model_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("model name must not be blank")
    return normalized


ModelName = Annotated[
    str,
    Field(min_length=1, max_length=200),
    AfterValidator(_normalize_model_name),
]


class AdminChatModelConfiguration(AdminRequest):
    enabled: bool
    default_model: ModelName
    allowed_models: list[ModelName] = Field(min_length=1, max_length=32)

    @field_validator("allowed_models")
    @classmethod
    def validate_unique_allowed_models(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("allowed_models must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_default_is_allowed(self) -> "AdminChatModelConfiguration":
        if self.default_model not in self.allowed_models:
            raise ValueError("default_model must be included in allowed_models")
        return self


class AdminOptionalModelConfiguration(AdminRequest):
    enabled: bool
    model: ModelName | None = None

    @model_validator(mode="after")
    def validate_enabled_role_has_model(self) -> "AdminOptionalModelConfiguration":
        if self.enabled and self.model is None:
            raise ValueError("an enabled model role requires a model")
        return self


class AdminRerankerConfiguration(AdminRequest):
    enabled: bool


def _normalize_clock_timezone(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        ZoneInfo(normalized)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"clock timezone must be a valid IANA timezone name, got {normalized!r}"
        ) from exc
    return normalized


ClockTimezone = Annotated[
    str | None,
    Field(default=None, max_length=64),
    AfterValidator(_normalize_clock_timezone),
]


class AdminClockConfiguration(AdminRequest):
    timezone: ClockTimezone = None


def _normalize_max_num_ctx(value: int | None) -> int | None:
    if value is None:
        return None
    if value not in MODEL_MAX_NUM_CTX_OPTIONS:
        raise ValueError(
            f"model_max_num_ctx must be one of {list(MODEL_MAX_NUM_CTX_OPTIONS)}, got {value!r}"
        )
    return value


# Admin-only context ceiling. Optional for backward compatibility:
# frontends predating it send nothing and keep the stored value.
MaxNumCtx = Annotated[int | None, Field(default=None), AfterValidator(_normalize_max_num_ctx)]


def _normalize_file_scope(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"file_scope must be an integer, got {value!r}")
    if not FILE_SCOPE_MIN <= value <= FILE_SCOPE_MAX:
        raise ValueError(
            f"file_scope must be between {FILE_SCOPE_MIN} and {FILE_SCOPE_MAX}, got {value!r}"
        )
    return value


# Admin-only cap on distinct files contributing to the RAG context in
# discovery/unscoped retrieval. Optional for backward compatibility.
FileScope = Annotated[int | None, Field(default=None), AfterValidator(_normalize_file_scope)]


def _normalize_top_k(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"top_k must be an integer, got {value!r}")
    if not TOP_K_MIN <= value <= TOP_K_MAX:
        raise ValueError(
            f"top_k must be between {TOP_K_MIN} and {TOP_K_MAX}, got {value!r}"
        )
    return value


# Admin-only vector-retrieval candidate count. Optional for backward
# compatibility. Downstream rerank/evidence/synthesis stages keep their
# own independent safety limits.
RetrievalTopK = Annotated[int | None, Field(default=None), AfterValidator(_normalize_top_k)]


def _normalize_search_depth(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if cleaned not in SEARCH_DEPTH_OPTIONS:
        raise ValueError(
            f"search_depth must be one of {SEARCH_DEPTH_OPTIONS}, got {value!r}"
        )
    return cleaned


# Admin-only web search depth preset. Optional for backward compatibility.
SearchDepthField = Annotated[str | None, Field(default=None), AfterValidator(_normalize_search_depth)]


class AdminAIModelsUpdate(AdminRequest):
    expected_revision: int = Field(ge=0)
    chat: AdminChatModelConfiguration
    vision: AdminOptionalModelConfiguration
    intelligence: AdminOptionalModelConfiguration
    memory_extraction: AdminOptionalModelConfiguration
    reranker: AdminRerankerConfiguration
    # Optional for backward compatibility: frontends predating the clock
    # override send no clock section; that means "no override", not an error.
    clock: AdminClockConfiguration = Field(default_factory=AdminClockConfiguration)
    model_max_num_ctx: MaxNumCtx = None
    file_scope: FileScope = None
    top_k: RetrievalTopK = None
    search_depth: SearchDepthField = None
    # Optional explicit fallback: served when neither preference nor default
    # is usable. Null means "no fallback — error honestly". Validated
    # against allowed+installed in _validate_admin_ai_models.
    fallback_chat_model: str | None = Field(default=None, max_length=200)

    @field_validator("fallback_chat_model", mode="before")
    @classmethod
    def normalize_fallback_chat_model(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class InstalledModelResponse(BaseModel):
    name: str
    size: int
    digest: str
    families: list[str]
    modalities: list[str]
    parameter_size: str
    quantization_level: str


class ChatModelConfigurationResponse(BaseModel):
    enabled: bool
    default_model: str
    allowed_models: list[str]
    fallback_chat_model: str | None = None
    available: bool
    optional: Literal[False]


class OptionalModelConfigurationResponse(BaseModel):
    enabled: bool
    model: str | None
    configured: bool
    available: bool
    optional: Literal[True]


class EmbeddingConfigurationResponse(BaseModel):
    model: str
    dimension: int
    configured: bool
    available: bool
    optional: Literal[False]
    read_only: Literal[True]


class RerankerConfigurationResponse(BaseModel):
    enabled: bool
    model: str
    configured: bool
    available: bool
    optional: Literal[True]
    read_only_model: Literal[True]


class ClockConfigurationResponse(BaseModel):
    timezone: str | None
    effective_timezone: str
    source: Literal["environment", "database"]


class AdminAIModelsResponse(BaseModel):
    revision: int
    source: Literal["environment", "database"]
    updated_by: int | None
    updated_at: datetime | None
    deployment_allowed_chat_models: list[str]
    installed_models: list[InstalledModelResponse]
    model_max_num_ctx: int = DEFAULT_MODEL_MAX_NUM_CTX
    model_max_num_ctx_options: list[int] = Field(
        default_factory=lambda: list(MODEL_MAX_NUM_CTX_OPTIONS)
    )
    file_scope: int = DEFAULT_FILE_SCOPE
    top_k: int = DEFAULT_TOP_K
    search_depth: str = DEFAULT_SEARCH_DEPTH
    search_depth_options: list[str] = Field(
        default_factory=lambda: list(SEARCH_DEPTH_OPTIONS)
    )
    chat: ChatModelConfigurationResponse
    vision: OptionalModelConfigurationResponse
    intelligence: OptionalModelConfigurationResponse
    memory_extraction: OptionalModelConfigurationResponse
    embedding: EmbeddingConfigurationResponse
    reranker: RerankerConfigurationResponse
    clock: ClockConfigurationResponse


def _installed_model_map(model_service: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    installed = [
        model
        for model in model_service.get_available_models()
        if isinstance(model, dict) and str(model.get("name") or "").strip()
    ]
    return installed, {str(model["name"]): model for model in installed}


def _role_payload(
    role: OptionalModelConfiguration,
    *,
    installed_names: set[str],
) -> dict[str, Any]:
    return {
        "enabled": role.enabled,
        "model": role.model,
        "configured": bool(role.model),
        "available": bool(role.model and role.model in installed_names),
        "optional": True,
    }


def _admin_ai_models_payload(
    config: SystemAIConfiguration,
    *,
    installed: list[dict[str, Any]],
    model_service: Any,
) -> dict[str, Any]:
    installed_names = {
        str(model["name"])
        for model in installed
        if str(model.get("name") or "").strip()
    }
    reranker_configured = bool(settings.rerank_base_url.strip() and settings.rerank_model.strip())
    health_probe = getattr(model_service, "is_service_available", None)
    reranker_available = bool(
        config.reranker_enabled
        and reranker_configured
        and callable(health_probe)
        and health_probe(f"{settings.rerank_base_url.rstrip('/')}/health")
    )
    public_models = [
        {
            "name": str(model["name"]),
            "size": int(model.get("size") or 0),
            "digest": str(model.get("digest") or ""),
            "families": list(model.get("families") or []),
            "modalities": list(model.get("modalities") or []),
            "parameter_size": str(model.get("parameter_size") or ""),
            "quantization_level": str(model.get("quantization_level") or ""),
        }
        for model in installed
    ]
    return {
        "revision": config.revision,
        "source": config.source,
        "updated_by": config.updated_by,
        "updated_at": config.updated_at,
        "deployment_allowed_chat_models": list(deployment_chat_model_ceiling() or []),
        "installed_models": public_models,
        "model_max_num_ctx": config.model_max_num_ctx,
        "model_max_num_ctx_options": list(MODEL_MAX_NUM_CTX_OPTIONS),
        "file_scope": config.file_scope,
        "top_k": config.top_k,
        "search_depth": config.search_depth,
        "search_depth_options": list(SEARCH_DEPTH_OPTIONS),
        "chat": {
            "enabled": config.chat.enabled,
            "default_model": config.chat.default_model,
            "allowed_models": list(config.chat.allowed_models),
            "fallback_chat_model": config.fallback_chat_model,
            "available": bool(
                config.chat.default_model
                and config.chat.default_model in installed_names
                and set(config.chat.allowed_models).issubset(installed_names)
            ),
            "optional": False,
        },
        "vision": _role_payload(config.vision, installed_names=installed_names),
        "intelligence": _role_payload(config.intelligence, installed_names=installed_names),
        "memory_extraction": _role_payload(
            config.memory_extraction,
            installed_names=installed_names,
        ),
        "embedding": {
            "model": settings.embedding_model_name.strip(),
            "dimension": settings.embedding_dimension,
            "configured": bool(
                settings.embedding_model_name.strip() and settings.embedding_api_url.strip()
            ),
            "available": settings.embedding_model_name.strip() in installed_names,
            "optional": False,
            "read_only": True,
            **_embedding_database_info(),
        },
        "reranker": {
            "enabled": config.reranker_enabled,
            "mode": settings.rerank_mode,
            "model": settings.rerank_model,
            "url": settings.rerank_base_url,
            "configured": reranker_configured,
            "available": reranker_available,
            "optional": True,
            "read_only_model": True,
        },
        "clock": {
            "timezone": config.clock_timezone,
            "effective_timezone": resolve_clock_timezone(config),
            "source": clock_timezone_source(config),
        },
    }


def _embedding_database_info() -> dict:
    """Database-side embedding truth for the admin UI (read-only).

    There is deliberately no PUT path for the embedding model: changing it
    requires `python -m app.cli reembed`, never a silent swap. When the
    configured model/dimensions differ from the database, the UI points at
    that command via change_requires_reembed.
    """
    from app.config import settings

    try:
        from app.database import get_db

        with get_db() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(
                    "SELECT model, dimensions FROM embedding_config WHERE singleton_id = 1"
                )
                row = cursor.fetchone()
            finally:
                close = getattr(cursor, "close", None)
                if close is not None:
                    close()
    except Exception:
        return {"database_model": None, "database_dimensions": None,
                "change_requires_reembed": False}
    if row is None:
        return {"database_model": None, "database_dimensions": None,
                "change_requires_reembed": False}
    db_model = row["model"] if isinstance(row, dict) else row[0]
    db_dims = int(row["dimensions"] if isinstance(row, dict) else row[1])
    mismatch = (
        str(db_model or "").strip() != settings.embedding_model_name.strip()
        or db_dims != settings.embedding_dimension
    )
    return {
        "database_model": db_model,
        "database_dimensions": db_dims,
        "change_requires_reembed": bool(mismatch),
    }


def _validate_admin_ai_models(
    body: AdminAIModelsUpdate,
    *,
    model_service: Any,
    installed_by_name: dict[str, dict[str, Any]],
) -> None:
    deployment_ceiling = deployment_chat_model_ceiling()
    if deployment_ceiling is not None:
        outside_ceiling = sorted(set(body.chat.allowed_models) - set(deployment_ceiling))
        if outside_ceiling:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Chat models exceed the deployment ALLOWED_CHAT_MODELS ceiling: "
                    + ", ".join(outside_ceiling)
                ),
            )

    selected_models = set(body.chat.allowed_models)
    selected_models.update(
        role.model
        for role in (body.vision, body.intelligence, body.memory_extraction)
        if role.enabled and role.model is not None
    )
    missing_models = sorted(selected_models - set(installed_by_name))
    if missing_models:
        if not installed_by_name:
            raise HTTPException(
                status_code=503,
                detail="Ollama model discovery is unavailable or no models are installed",
            )
        raise HTTPException(
            status_code=422,
            detail="Selected models are not installed in Ollama: " + ", ".join(missing_models),
        )

    if body.fallback_chat_model is not None:
        if body.fallback_chat_model not in set(body.chat.allowed_models):
            raise HTTPException(
                status_code=422,
                detail="Fallback model must be one of the allowed chat models",
            )
        if body.fallback_chat_model not in set(installed_by_name):
            raise HTTPException(
                status_code=422,
                detail="Fallback model is not installed in Ollama: " + body.fallback_chat_model,
            )

    if body.vision.enabled:
        capability_probe = getattr(model_service, "get_model_capabilities", None)
        if not callable(capability_probe):
            raise HTTPException(
                status_code=503,
                detail="Ollama vision-capability verification is unavailable",
            )
        capabilities = {str(value).lower() for value in capability_probe(body.vision.model)}
        if not capabilities:
            raise HTTPException(
                status_code=503,
                detail="Ollama could not verify the selected model's vision capability",
            )
        if "vision" not in capabilities:
            raise HTTPException(
                status_code=422,
                detail=f"Selected Vision model is not vision-capable: {body.vision.model}",
            )


@router.get("/ai-models", response_model=AdminAIModelsResponse)
def get_admin_ai_models(_admin: CurrentAdmin) -> dict[str, Any]:
    model_service = get_model_service()
    installed, _installed_by_name = _installed_model_map(model_service)
    with get_db() as connection:
        config = ModelConfigurationRepository(connection).get()
    return _admin_ai_models_payload(
        config,
        installed=installed,
        model_service=model_service,
    )


@router.put("/ai-models", response_model=AdminAIModelsResponse)
def update_admin_ai_models(
    body: AdminAIModelsUpdate,
    admin_user: CurrentAdmin,
) -> dict[str, Any]:
    model_service = get_model_service()
    installed, installed_by_name = _installed_model_map(model_service)
    _validate_admin_ai_models(
        body,
        model_service=model_service,
        installed_by_name=installed_by_name,
    )
    write = SystemAIConfigurationWrite(
        chat=ChatModelConfiguration(
            enabled=body.chat.enabled,
            default_model=body.chat.default_model,
            allowed_models=tuple(body.chat.allowed_models),
        ),
        vision=OptionalModelConfiguration(enabled=body.vision.enabled, model=body.vision.model),
        intelligence=OptionalModelConfiguration(
            enabled=body.intelligence.enabled,
            model=body.intelligence.model,
        ),
        memory_extraction=OptionalModelConfiguration(
            enabled=body.memory_extraction.enabled,
            model=body.memory_extraction.model,
        ),
        reranker_enabled=body.reranker.enabled,
        clock_timezone=body.clock.timezone,
        model_max_num_ctx=body.model_max_num_ctx,
        fallback_chat_model=(body.fallback_chat_model or "").strip() or None,
        file_scope=body.file_scope,
        top_k=body.top_k,
        search_depth=body.search_depth,
    )
    try:
        with get_db() as connection:
            repository = ModelConfigurationRepository(connection)
            # Admin Preferences live in the same row: an AI-model save must
            # carry the stored preference values through, never reset them.
            stored_for_prefs = repository.get()
            if (
                write.model_max_num_ctx is None
                or write.file_scope is None
                or write.top_k is None
            ):
                # Old frontends omit newer knobs: preserve each stored value
                # instead of resetting it.
                stored = repository.get()
                write = SystemAIConfigurationWrite(
                    chat=write.chat,
                    vision=write.vision,
                    intelligence=write.intelligence,
                    memory_extraction=write.memory_extraction,
                    reranker_enabled=write.reranker_enabled,
                    clock_timezone=write.clock_timezone,
                    model_max_num_ctx=(
                        write.model_max_num_ctx
                        if write.model_max_num_ctx is not None
                        else stored.model_max_num_ctx
                    ),
                    fallback_chat_model=(
                        write.fallback_chat_model
                        if "fallback_chat_model" in body.model_fields_set
                        else stored.fallback_chat_model
                    ),
                    file_scope=(
                        write.file_scope
                        if write.file_scope is not None
                        else stored.file_scope
                    ),
                    top_k=(
                        write.top_k
                        if write.top_k is not None
                        else stored.top_k
                    ),
                    search_depth=(
                        write.search_depth
                        if "search_depth" in body.model_fields_set
                        else stored.search_depth
                    ),
                    session_timeout_minutes=stored_for_prefs.session_timeout_minutes,
                    registration_enabled=stored_for_prefs.registration_enabled,
                )
            elif (
                "fallback_chat_model" not in body.model_fields_set
                or "search_depth" not in body.model_fields_set
            ):
                # Same backward-compatibility guard for the web search depth
                # tier: an old frontend that omits it must not reset a stored
                # value on an unrelated save.
                stored = repository.get()
                write = SystemAIConfigurationWrite(
                    chat=write.chat,
                    vision=write.vision,
                    intelligence=write.intelligence,
                    memory_extraction=write.memory_extraction,
                    reranker_enabled=write.reranker_enabled,
                    clock_timezone=write.clock_timezone,
                    model_max_num_ctx=write.model_max_num_ctx,
                    fallback_chat_model=stored.fallback_chat_model,
                    file_scope=write.file_scope,
                    top_k=write.top_k,
                    search_depth=(
                        write.search_depth
                        if "search_depth" in body.model_fields_set
                        else stored.search_depth
                    ),
                    session_timeout_minutes=stored_for_prefs.session_timeout_minutes,
                    registration_enabled=stored_for_prefs.registration_enabled,
                )
            config = repository.replace(
                write,
                expected_revision=body.expected_revision,
                updated_by=int(admin_user["id"]),
            )
    except ModelConfigurationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "model_configuration_conflict",
                "expected_revision": exc.expected_revision,
                "current_revision": exc.current_revision,
            },
        ) from None
    return _admin_ai_models_payload(
        config,
        installed=installed,
        model_service=model_service,
    )


# ── PREFERENCES ────────────────────────────────────────────────────────────
# Admin-only runtime preferences stored in the same singleton row as the AI
# model configuration. Saved values take effect immediately (no restart):
# /register reads the toggle per request, and new sessions/tokens use the
# saved timeout. NULL (never saved) falls back to deployment configuration.


class AdminPreferencesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_timeout_minutes: int
    session_timeout_saved: bool
    registration_enabled: bool
    registration_saved: bool
    expected_revision: int


class AdminPreferencesUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_timeout_minutes: int | None = None
    registration_enabled: StrictBool | None = None
    expected_revision: int


def _admin_preferences_payload(config: SystemAIConfiguration) -> dict[str, Any]:
    return {
        "session_timeout_minutes": settings.session_idle_timeout_minutes,
        "session_timeout_saved": config.session_timeout_minutes is not None,
        "registration_enabled": settings.registration_enabled,
        "registration_saved": config.registration_enabled is not None,
        "expected_revision": config.revision,
    }


@router.get("/preferences", response_model=AdminPreferencesResponse)
def get_admin_preferences(_admin: CurrentAdmin) -> dict[str, Any]:
    with get_db() as connection:
        config = ModelConfigurationRepository(connection).get()
    return _admin_preferences_payload(config)


@router.put("/preferences", response_model=AdminPreferencesResponse)
def update_admin_preferences(
    body: AdminPreferencesUpdate,
    admin_user: CurrentAdmin,
) -> dict[str, Any]:
    if body.session_timeout_minutes is None and body.registration_enabled is None:
        raise HTTPException(status_code=422, detail="Nothing to save")
    if (
        body.session_timeout_minutes is not None
        and body.session_timeout_minutes not in SESSION_TIMEOUT_OPTIONS
    ):
        raise HTTPException(
            status_code=422,
            detail=f"session_timeout_minutes must be one of {list(SESSION_TIMEOUT_OPTIONS)}",
        )
    try:
        with get_db() as connection:
            repository = ModelConfigurationRepository(connection)
            stored = repository.get()
            write = SystemAIConfigurationWrite(
                chat=stored.chat,
                vision=stored.vision,
                intelligence=stored.intelligence,
                memory_extraction=stored.memory_extraction,
                reranker_enabled=stored.reranker_enabled,
                clock_timezone=stored.clock_timezone,
                model_max_num_ctx=stored.model_max_num_ctx,
                fallback_chat_model=stored.fallback_chat_model,
                file_scope=stored.file_scope,
                top_k=stored.top_k,
                search_depth=stored.search_depth,
                session_timeout_minutes=(
                    body.session_timeout_minutes
                    if body.session_timeout_minutes is not None
                    else stored.session_timeout_minutes
                ),
                registration_enabled=(
                    body.registration_enabled
                    if body.registration_enabled is not None
                    else stored.registration_enabled
                ),
            )
            config = repository.replace(
                write,
                expected_revision=body.expected_revision,
                updated_by=int(admin_user["id"]),
            )
    except ModelConfigurationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "model_configuration_conflict",
                "expected_revision": exc.expected_revision,
                "current_revision": exc.current_revision,
            },
        ) from None
    invalidate_preferences_cache()
    with get_db() as connection:
        config = ModelConfigurationRepository(connection).get()
    return _admin_preferences_payload(config)


# ── USERS ──────────────────────────────────────────────────────────────────

PERM_FIELDS = [
    "perm_upload",
    "perm_download",
    "perm_delete",
    "perm_ai",
    "perm_share",
    "perm_folders",
    "perm_rename",
]


@router.get("/users")
def list_users(_admin: CurrentAdmin):
    """List all users with file counts, storage, and permissions"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT u.id, u.username, u.email, u.is_active, u.is_admin,
                   u.storage_quota_bytes, u.storage_used_bytes,
                   u.created_at, u.last_login,
                   u.perm_upload, u.perm_download, u.perm_delete,
                   u.perm_ai, u.perm_share, u.perm_folders, u.perm_rename,
                   COUNT(f.id) FILTER (WHERE f.is_deleted = FALSE) AS file_count
            FROM users u
            LEFT JOIN files f ON f.user_id = u.id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


class UserPatch(AdminRequest):
    is_active: bool | None = None
    storage_quota_gb: int | None = Field(default=None, ge=0, le=1_000_000)
    is_admin: bool | None = None


@router.patch("/users/{user_id}")
def update_user(user_id: int, body: UserPatch, _admin: CurrentAdmin):
    """Toggle is_active, adjust quota, set admin flag"""
    updates = []
    values = []
    if body.is_active is not None:
        updates.append("is_active = %s")
        values.append(body.is_active)
    if body.storage_quota_gb is not None:
        updates.append("storage_quota_bytes = %s")
        values.append(body.storage_quota_gb * 1024 * 1024 * 1024)
    if body.is_admin is not None:
        updates.append("is_admin = %s")
        values.append(body.is_admin)
    if not updates:
        return {"message": "Nothing to update"}
    values.append(user_id)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = %s RETURNING id",
            values,
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User updated"}


class ResetPasswordRequest(AdminRequest):
    new_password: BcryptPassword


@router.post("/users/{user_id}/reset-password")
def reset_user_password(user_id: int, body: ResetPasswordRequest, _admin: CurrentAdmin):
    """Admin resets any user's password"""
    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    new_hash = get_password_hash(body.new_password)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s RETURNING id",
            (new_hash, user_id),
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="User not found")
        cur.execute(
            "UPDATE refresh_tokens SET is_revoked = TRUE WHERE user_id = %s",
            (user_id,),
        )
    return {"message": "Password reset successfully"}


# ── STORAGE ────────────────────────────────────────────────────────────────


@router.get("/storage")
def storage_overview(_admin: CurrentAdmin):
    """Per-user storage summary"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT u.id AS user_id, u.username,
                   u.storage_used_bytes AS storage_used,
                   u.storage_quota_bytes AS storage_quota,
                   COUNT(f.id) FILTER (WHERE f.is_deleted = FALSE) AS file_count
            FROM users u
            LEFT JOIN files f ON f.user_id = u.id
            GROUP BY u.id
            ORDER BY u.storage_used_bytes DESC
        """)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── SYSTEM ─────────────────────────────────────────────────────────────────


@router.get("/system")
def system_info(_admin: CurrentAdmin):
    """Report the v2 service graph without exposing credentials or internals."""

    services: dict[str, str] = {}
    worker_health = None
    try:
        client = redis.from_url(
            settings.redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        worker_health = decode_worker_health(client.get(WORKER_HEALTH_KEY))
        client.close()
        services["redis"] = "ok"
    except Exception:
        services["redis"] = "error"

    database_stats: dict[str, int] = {}
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM users) AS users,
                    (SELECT COUNT(*) FROM document_chunks) AS vectors,
                    (SELECT COUNT(*) FROM ingestion_jobs WHERE state = 'queued') AS queued_jobs,
                    (SELECT COUNT(*) FROM ingestion_jobs WHERE state IN (
                        'decrypting', 'converting', 'parsing', 'chunking',
                        'embedding', 'publishing'
                    )) AS active_jobs
                """
            )
            row = cursor.fetchone()
            database_stats = {key: int(value or 0) for key, value in row.items()}
        services["postgres"] = "ok"
    except Exception:
        services["postgres"] = "error"

    try:
        minio = Minio(
            settings.s3_endpoint.removeprefix("https://").removeprefix("http://"),
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_secure,
        )
        services["minio"] = "ok" if minio.bucket_exists(settings.s3_bucket_name) else "error"
    except Exception:
        services["minio"] = "error"

    probes = {
        "ollama": f"{settings.ollama_base_url.rstrip('/')}/api/tags",
        "agent_runtime": (f"{settings.agent_runtime_url.rstrip('/')}/internal/v1/health/ready"),
        "reranker": f"{settings.rerank_base_url.rstrip('/')}/health",
        "searxng": f"{settings.searxng_url.rstrip('/')}/healthz",
    }
    with httpx.Client(timeout=3, follow_redirects=False) as client:
        for name, url in probes.items():
            try:
                response = client.get(url)
                services[name] = "ok" if response.status_code < 400 else "error"
            except Exception:
                services[name] = "error"

    # The API deliberately has no route to the document-processing network.
    # Redis carries expiring evidence from the worker's real PDF self-test.
    worker_live = worker_health is not None
    odl_usable = bool(worker_health and worker_health["status"] in {"ready", "busy"})
    services["ingestion_worker"] = "ok" if worker_live else "error"
    services["opendataloader"] = "ok" if odl_usable else "error"

    user_count = database_stats.get("users", 0)

    return {
        "version": settings.app_version,
        "user_count": user_count,
        "services": services,
        "index": {
            "vectors": database_stats.get("vectors", 0),
            "queued_jobs": database_stats.get("queued_jobs", 0),
            "active_jobs": database_stats.get("active_jobs", 0),
        },
        "readiness": {
            "ingestion_worker": {
                "status": "live" if worker_live else "offline",
                "observed_at": worker_health["observed_at"] if worker_health else None,
            },
            "opendataloader": {
                "status": worker_health["status"] if worker_health else "offline",
                "verified_at": worker_health["pdf_verified_at"] if worker_health else None,
                "detail": worker_health["detail"] if worker_health else None,
                "consecutive_failures": (worker_health["consecutive_failures"] if worker_health else 0),
                "verification_source": (worker_health["verification_source"] if worker_health else None),
                "topology": "private",
            },
            "user_count": user_count,
        },
    }


@router.post("/clear-embeddings")
def clear_embeddings(_admin: CurrentAdmin):
    """Delete all embeddings and reset every file's AI state."""
    with get_db() as conn:
        conn.execute("DELETE FROM document_chunks")
        conn.execute("DELETE FROM document_revisions WHERE status != 'building'")
        conn.execute("DELETE FROM ingestion_jobs")
        conn.execute("""
            UPDATE files
            SET user_granted_ai_access = FALSE,
                current_revision = NULL,
                desired_revision = 0,
                ai_status = 'not_granted',
                ai_error_code = NULL,
                ai_error_detail = NULL,
                quick_summary = NULL,
                quick_tags = NULL,
                doc_type = NULL,
                intelligence_status = 'pending'
        """)
        conn.commit()
    return {"ok": True}


# ── PERMISSIONS ─────────────────────────────────────────────────────────────


class PermissionsUpdate(AdminRequest):
    perm_upload: bool | None = None
    perm_download: bool | None = None
    perm_delete: bool | None = None
    perm_ai: bool | None = None
    perm_share: bool | None = None
    perm_folders: bool | None = None
    perm_rename: bool | None = None


@router.get("/users/{user_id}/permissions")
def get_user_permissions(user_id: int, _admin: CurrentAdmin):
    """Get permission flags for a specific user"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT {', '.join(PERM_FIELDS)} FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(row)


@router.patch("/users/{user_id}/permissions")
def set_user_permissions(user_id: int, body: PermissionsUpdate, _admin: CurrentAdmin):
    """Update one or more permission flags for a user"""
    updates, values = [], []
    for field in PERM_FIELDS:
        val = getattr(body, field, None)
        if val is not None:
            updates.append(f"{field} = %s")
            values.append(val)
    if not updates:
        return {"message": "Nothing to update"}
    values.append(user_id)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = %s RETURNING id",
            values,
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Permissions updated"}


# ── POLICY TEMPLATES ─────────────────────────────────────────────────────────


class PolicyCreate(AdminRequest):
    name: PolicyName
    perm_upload: bool = True
    perm_download: bool = True
    perm_delete: bool = False
    perm_ai: bool = True
    perm_share: bool = True
    perm_folders: bool = True
    perm_rename: bool = True


@router.get("/policies")
def list_policies(_admin: CurrentAdmin):
    """List all policy templates"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM user_policy_templates ORDER BY is_builtin DESC, name")
        rows = cur.fetchall()
    return [dict(r) for r in rows]


@router.post("/policies")
def create_policy(body: PolicyCreate, _admin: CurrentAdmin):
    """Create a custom policy template"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_policy_templates
              (name, perm_upload, perm_download, perm_delete, perm_ai, perm_share, perm_folders, perm_rename, is_builtin)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE)
            RETURNING id
        """,
            (
                body.name,
                body.perm_upload,
                body.perm_download,
                body.perm_delete,
                body.perm_ai,
                body.perm_share,
                body.perm_folders,
                body.perm_rename,
            ),
        )
        row = cur.fetchone()
        conn.commit()
    return {"id": row["id"], "message": "Policy created"}


class ApplyPolicyRequest(AdminRequest):
    policy_id: int = Field(gt=0)


@router.post("/users/{user_id}/apply-policy")
def apply_policy(user_id: int, body: ApplyPolicyRequest, _admin: CurrentAdmin):
    """Apply a policy template's permissions to a user"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(PERM_FIELDS)} FROM user_policy_templates WHERE id = %s", (body.policy_id,)
        )
        policy = cur.fetchone()
        if not policy:
            raise HTTPException(status_code=404, detail="Policy not found")
        vals = [policy[f] for f in PERM_FIELDS]
        vals.append(user_id)
        set_clause = ", ".join(f"{f} = %s" for f in PERM_FIELDS)
        cur.execute(f"UPDATE users SET {set_clause} WHERE id = %s RETURNING id", vals)
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Policy applied"}
