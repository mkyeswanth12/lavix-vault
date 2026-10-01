"""Configuration owned solely by the isolated agent runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

# Pinned helper default (README pull list): rewrite/planner steps always
# use this model, even when Admin Settings selects another chat model.
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"
DEFAULT_TOOL_GATEWAY_URL = "http://api:8080/api/internal/agent"


class ConfigurationError(ValueError):
    """Raised when runtime configuration is invalid."""


class ModelNotAllowedError(ValueError):
    """Raised when a request attempts to select a non-allowlisted model."""


def _validated_http_url(name: str, value: str) -> str:
    cleaned = value.strip().rstrip("/")
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigurationError(f"{name} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ConfigurationError(f"{name} must not contain credentials")
    return cleaned


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _unit_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = _positive_float(env, name, default)
    if value > 1.0:
        raise ConfigurationError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Small, explicit configuration surface for the CUGA sidecar.

    Deliberately absent: database, Redis, object-storage, and end-user auth
    credentials.  The runtime can reach vault data only through capability-
    authenticated tool gateway requests.
    """

    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    default_model: str = DEFAULT_MODEL
    allowed_models: frozenset[str] = frozenset({DEFAULT_MODEL})
    tool_gateway_url: str = DEFAULT_TOOL_GATEWAY_URL
    ollama_num_ctx: int = 16_384
    # Practical ceiling for auto-detected native context lengths. Models
    # advertise values (e.g. 262144 via rope scaling from a 16k base) far
    # beyond what the hardware serves well — KV cache grows with context.
    # Raise only with measured headroom. Env: LAVIX_AGENT_MAX_NUM_CTX.
    model_max_num_ctx: int = 32_768
    # 2048 reduces Ollama streaming/finalization mismatches (was 512).
    ollama_num_predict: int = 2048
    # Must cover SearXNG fan-out (~12s) + rerank + sequential verify (was 20).
    tool_timeout_seconds: int = 45
    queue_wait_timeout_seconds: int = 30
    # Admitted-run ceiling with margin above the tool budget (was 180).
    run_timeout_seconds: int = 420
    cuga_max_steps: int = 6
    max_input_chars: int = 20_000
    citation_verification_enabled: bool = True
    citation_relevance_threshold: float = 0.5
    embedding_model_name: str = "snowflake-arctic-embed2:cpu"
    strict_grounding: bool = False
    # Conversational web-query rewriter (web leg only; vault leg untouched).
    web_rewrite_enabled: bool = True
    # Dedicated rewriter model; None falls back to default_model at call time.
    web_rewrite_model: str | None = None
    web_rewrite_timeout_seconds: float = 20.0
    web_rewrite_confidence_min: float = 0.5
    web_rewrite_history_messages: int = 6
    web_rewrite_history_token_cap: int = 1500
    web_rewrite_assistant_trim_chars: int = 300
    # Search-query planner budget. Sized above measured CPU service time
    # (~6-11s) so the planner survives instead of always falling back.
    web_planner_timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ollama_base_url",
            _validated_http_url("ollama_base_url", self.ollama_base_url),
        )
        object.__setattr__(
            self,
            "tool_gateway_url",
            _validated_http_url("tool_gateway_url", self.tool_gateway_url),
        )
        if not self.default_model.strip():
            raise ConfigurationError("default_model must not be empty")
        cleaned_models = frozenset(model.strip() for model in self.allowed_models if model.strip())
        if self.default_model not in cleaned_models:
            raise ConfigurationError("default_model must be present in allowed_models")
        object.__setattr__(self, "allowed_models", cleaned_models)
        rewrite_model = (self.web_rewrite_model or "").strip() or self.default_model
        if rewrite_model not in cleaned_models:
            raise ConfigurationError("web_rewrite_model must be present in allowed_models")
        object.__setattr__(self, "web_rewrite_model", (self.web_rewrite_model or "").strip() or None)
        if not 0.0 < self.web_rewrite_confidence_min <= 1.0:
            raise ConfigurationError("web_rewrite_confidence_min must be between 0 and 1")
        for name in (
            "ollama_num_ctx",
            "ollama_num_predict",
            "tool_timeout_seconds",
            "queue_wait_timeout_seconds",
            "run_timeout_seconds",
            "cuga_max_steps",
            "max_input_chars",
            "web_rewrite_timeout_seconds",
            "web_planner_timeout_seconds",
            "web_rewrite_history_messages",
            "web_rewrite_history_token_cap",
            "web_rewrite_assistant_trim_chars",
        ):
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"{name} must be greater than zero")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RuntimeSettings:
        env = os.environ if environ is None else environ
        default_model = env.get("LAVIX_AGENT_MODEL", DEFAULT_MODEL).strip()
        allowlist_value = env.get("LAVIX_AGENT_ALLOWED_MODELS", default_model)
        allowed_models = frozenset(item.strip() for item in allowlist_value.split(",") if item.strip())
        return cls(
            ollama_base_url=env.get("LAVIX_AGENT_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            default_model=default_model,
            allowed_models=allowed_models,
            tool_gateway_url=env.get("LAVIX_AGENT_TOOL_GATEWAY_URL", DEFAULT_TOOL_GATEWAY_URL),
            ollama_num_ctx=_positive_int(env, "LAVIX_AGENT_OLLAMA_NUM_CTX", 16_384),
            model_max_num_ctx=_positive_int(env, "LAVIX_AGENT_MAX_NUM_CTX", 32_768),
            ollama_num_predict=_positive_int(env, "LAVIX_AGENT_OLLAMA_NUM_PREDICT", 2048),
            tool_timeout_seconds=_positive_int(env, "LAVIX_AGENT_TOOL_TIMEOUT_SECONDS", 45),
            queue_wait_timeout_seconds=_positive_int(env, "LAVIX_AGENT_QUEUE_WAIT_TIMEOUT_SECONDS", 30),
            run_timeout_seconds=_positive_int(env, "LAVIX_AGENT_RUN_TIMEOUT_SECONDS", 420),
            cuga_max_steps=_positive_int(env, "LAVIX_AGENT_CUGA_MAX_STEPS", 6),
            max_input_chars=_positive_int(env, "LAVIX_AGENT_MAX_INPUT_CHARS", 20_000),
            citation_verification_enabled=env.get("LAVIX_CITATION_VERIFICATION_ENABLED", "true").lower() in ("true", "1", "yes"),
            citation_relevance_threshold=float(env.get("LAVIX_CITATION_RELEVANCE_THRESHOLD", "0.5")),
            embedding_model_name=env.get("LAVIX_EMBEDDING_MODEL_NAME", "snowflake-arctic-embed2:cpu"),
            strict_grounding=env.get("LAVIX_AGENT_STRICT_GROUNDING", "false").lower() in ("true", "1", "yes"),
            web_rewrite_enabled=env.get("LAVIX_WEB_REWRITE_ENABLED", "true").lower() in ("true", "1", "yes"),
            web_rewrite_model=(env.get("LAVIX_WEB_REWRITE_MODEL", "").strip() or None),
            web_rewrite_timeout_seconds=_positive_float(env, "LAVIX_WEB_REWRITE_TIMEOUT_SECONDS", 20.0),
            web_planner_timeout_seconds=_positive_float(env, "LAVIX_WEB_PLANNER_TIMEOUT_SECONDS", 20.0),
            web_rewrite_confidence_min=_unit_float(env, "LAVIX_WEB_REWRITE_CONFIDENCE_MIN", 0.5),
            web_rewrite_history_messages=_positive_int(env, "LAVIX_WEB_REWRITE_HISTORY_MESSAGES", 6),
            web_rewrite_history_token_cap=_positive_int(env, "LAVIX_WEB_REWRITE_HISTORY_TOKEN_CAP", 1500),
            web_rewrite_assistant_trim_chars=_positive_int(env, "LAVIX_WEB_REWRITE_ASSISTANT_TRIM_CHARS", 300),
        )

    @property
    def resolved_rewrite_model(self) -> str:
        """Rewriter model: dedicated override or the default chat model."""
        return (self.web_rewrite_model or "").strip() or self.default_model

    def resolve_model(
        self,
        requested_model: str | None,
        allowed_override: Sequence[str] | None = None,
    ) -> str:
        """Resolve the run model against the effective allowlist.

        The API passes its DB-backed allowed list per request; when present
        it governs (so admin model changes apply without agent restarts).
        Otherwise the static env allowlist applies as before.
        """
        model = (requested_model or self.default_model).strip()
        if allowed_override:
            allowed = frozenset(
                str(item).strip() for item in allowed_override if str(item).strip()
            )
        else:
            allowed = self.allowed_models
        if model not in allowed:
            raise ModelNotAllowedError(f"Model '{model}' is not allowlisted")
        return model
