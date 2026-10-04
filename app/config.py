"""Environment-first application configuration.

Values resolve in order: environment variable -> ``app/config/config.yml``
(optional fallback for advanced knobs) -> built-in safe default. The default
``docker-compose.yaml`` flow sets every credential and URL through container
environment, so no ``config.yml`` and no ``app/config/secrets`` folder is
needed. Two settings keep one higher layer: the database-backed Admin
Preferences for session timeout and registration (see
``app.services.model_config``), which override everything when an admin has
saved them. Runtime secrets intentionally have no repository defaults
anywhere.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

_PROCESS_SECRET = secrets.token_urlsafe(48)
_TRUE_VALUES = frozenset({"1", "true", "yes"})
_NON_PRODUCTION_ENVIRONMENTS = frozenset({"dev", "development", "e2e", "test", "testing"})
_PLACEHOLDER_PREFIXES = (
    "change-me",
    "changeme",
    "default-",
    "example-",
    "replace-with",
)
_INSECURE_SECRET_VALUES = frozenset({"lavix-vault-secret-key-change-in-production-2024"})
_CONFIG_FILE_ENV = "LAVIX_CONFIG_FILE"
_MISSING = object()


def config_file_path() -> Path:
    """Resolve the active configuration file location."""

    override = os.environ.get(_CONFIG_FILE_ENV, "").strip()
    if override:
        return Path(override)
    container_mount = Path("/app/config/config.yml")
    if container_mount.is_file():
        return container_mount
    return Path(__file__).resolve().parent / "config" / "config.yml"


@lru_cache(maxsize=1)
def _raw_document() -> dict[str, Any]:
    """Parse config.yml once per process. Missing file yields an empty map."""

    path = config_file_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return data


@lru_cache(maxsize=1)
def _flat_settings() -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def _walk(prefix: str, node: dict[str, Any]) -> None:
        for key, value in node.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                _walk(dotted, value)
            else:
                flattened[dotted] = value

    _walk("", _raw_document())
    return flattened


def reload_config() -> None:
    """Drop cached configuration (used by tests and tooling)."""

    _raw_document.cache_clear()
    _flat_settings.cache_clear()


def _lookup(env_name: str, dotted_key: str) -> Any:
    if env_name in os.environ:
        return os.environ[env_name]
    return _flat_settings().get(dotted_key, _MISSING)


def _setting_str(dotted_key: str, env_name: str, default: str = "") -> str:
    value = _lookup(env_name, dotted_key)
    if value is _MISSING:
        return default
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value)


def _setting_str_first(dotted_key: str, env_names: tuple[str, ...], default: str = "") -> str:
    """First set env var wins, then config.yml, then the default.

    Lets new canonical names (RERANKER_MODEL) take precedence while old
    names (RERANK_MODEL) keep working as aliases for existing deployments.
    """
    for env_name in env_names:
        if env_name in os.environ:
            return _setting_str(dotted_key, env_name, default)
    return _setting_str(dotted_key, env_names[-1], default)


def _setting_int(dotted_key: str, env_name: str, default: int) -> int:
    value = _lookup(env_name, dotted_key)
    if value is _MISSING or (isinstance(value, str) and not value.strip()):
        return default
    return int(value)


def _setting_int_first(dotted_key: str, env_names: tuple[str, ...], default: int) -> int:
    """First set env var wins, then config.yml, then the default."""
    for env_name in env_names:
        if env_name in os.environ:
            return _setting_int(dotted_key, env_name, default)
    return _setting_int(dotted_key, env_names[-1], default)


def _setting_float(dotted_key: str, env_name: str, default: float) -> float:
    value = _lookup(env_name, dotted_key)
    if value is _MISSING or (isinstance(value, str) and not value.strip()):
        return default
    return float(value)


def _setting_bool(dotted_key: str, env_name: str, default: bool) -> bool:
    value = _lookup(env_name, dotted_key)
    if value is _MISSING:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE_VALUES


def _require_http_host_port(name: str, value: str) -> str:
    """Validate an http(s)://host:port URL, returning the stripped value."""
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError(f"{name} must be set to an http(s)://host:port URL")
    scheme, _, hostport = cleaned.partition("://")
    if scheme not in ("http", "https") or not hostport:
        raise ValueError(f"{name} must start with http:// or https:// (got {cleaned!r})")
    hostport = hostport.split("/", 1)[0]
    if ":" not in hostport:
        raise ValueError(f"{name} must include a port (got {cleaned!r})")
    port = hostport.rsplit(":", 1)[1]
    if not port.isdigit():
        raise ValueError(f"{name} port must be numeric (got {cleaned!r})")
    return cleaned


class Settings:
    """Read current settings from config.yml / environment on access."""

    @property
    def app_env(self) -> str:
        # Container deployments are secure by default. Development, test, and
        # E2E processes opt out explicitly through APP_ENV.
        return _setting_str("app.env", "APP_ENV", "production").strip().lower()

    @property
    def is_production(self) -> bool:
        return self.app_env not in _NON_PRODUCTION_ENVIRONMENTS

    def validate_config_file(self) -> None:
        """No-op when config.yml is absent (it is an optional fallback).

        The default docker-compose.yaml flow configures everything through
        environment variables. Kept as a hook so startup still calls one
        named validation step; malformed YAML still raises at read time.
        """

        return None

    def validate_runtime_secrets(self) -> None:
        """Reject missing, placeholder, or shared production signing secrets."""

        if not self.is_production:
            return

        configured = {
            # Raw lookups only: the _PROCESS_SECRET convenience fallback used
            # by the signing properties must NOT mask a missing production
            # secret here.
            "SECRET_KEY": _setting_str("security.secret_key", "SECRET_KEY").strip(),
            "AGENT_CAPABILITY_SECRET": _setting_str(
                "security.agent_capability_secret", "AGENT_CAPABILITY_SECRET"
            ).strip(),
        }
        invalid = [
            name
            for name, value in configured.items()
            if len(value.encode("utf-8")) < 32
            or value.lower().startswith(_PLACEHOLDER_PREFIXES)
            or value.lower() in _INSECURE_SECRET_VALUES
        ]
        if invalid:
            names = ", ".join(sorted(invalid))
            raise ValueError(f"Production requires explicit non-placeholder secrets: {names}")
        if configured["SECRET_KEY"] == configured["AGENT_CAPABILITY_SECRET"]:
            raise ValueError("SECRET_KEY and AGENT_CAPABILITY_SECRET must be independent")

    @property
    def app_name(self) -> str:
        return _setting_str("app.name", "APP_NAME", "Lavix Vault")

    @property
    def app_version(self) -> str:
        return _setting_str("app.version", "APP_VERSION", "1.0.0")

    @property
    def log_level(self) -> str:
        return _setting_str("app.log_level", "LOG_LEVEL", "INFO")

    @property
    def clock_timezone(self) -> str:
        value = _setting_str("app.timezone", "LAVIX_TIMEZONE", "Asia/Kolkata").strip()
        if not value:
            value = "Asia/Kolkata"
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"LAVIX_TIMEZONE/app.timezone must be a valid IANA timezone name, got {value!r}"
            ) from exc
        return value

    @property
    def secret_key(self) -> str:
        return _setting_str("security.secret_key", "SECRET_KEY") or _PROCESS_SECRET

    @property
    def db_host(self) -> str:
        return _setting_str("database.host", "DB_HOST", "postgres")

    @property
    def db_port(self) -> int:
        return _setting_int("database.port", "DB_PORT", 5432)

    @property
    def db_name(self) -> str:
        return _setting_str("database.name", "DB_NAME", "vault")

    @property
    def db_user(self) -> str:
        return _setting_str("database.user", "DB_USER", "vault")

    @property
    def db_pass(self) -> str:
        # docker-compose.yaml passes the same value as DB_PASS and
        # DB_PASSWORD (Postgres wants POSTGRES_PASSWORD); accept both.
        for env_name in ("DB_PASS", "DB_PASSWORD"):
            raw = os.environ.get(env_name, "")
            if raw.strip():
                return raw.strip()
        return _setting_str("database.password", "_LAVIX_DB_PASSWORD_UNSET_")

    @property
    def redis_url(self) -> str:
        return _setting_str("redis.url", "REDIS_URL", "redis://redis:6379/0")

    @property
    def s3_endpoint(self) -> str:
        endpoint = _setting_str("minio.endpoint", "S3_ENDPOINT", "minio:9000").strip()
        scheme = "https://" if self.s3_secure else "http://"
        return endpoint if "://" in endpoint else f"{scheme}{endpoint}"

    @property
    def s3_access_key(self) -> str:
        return _setting_str("minio.access_key", "S3_ACCESS_KEY")

    @property
    def s3_secret_key(self) -> str:
        return _setting_str("minio.secret_key", "S3_SECRET_KEY")

    @property
    def minio_root_user(self) -> str:
        return _setting_str("minio.root_user", "MINIO_ROOT_USER")

    @property
    def minio_root_password(self) -> str:
        return _setting_str("minio.root_password", "MINIO_ROOT_PASSWORD")

    @property
    def s3_bucket_name(self) -> str:
        return _setting_str("minio.bucket", "S3_BUCKET_NAME", "lavix-vault")

    @property
    def s3_secure(self) -> bool:
        return _setting_bool("minio.secure", "S3_SECURE", False)

    @property
    def ollama_base_url(self) -> str:
        return _setting_str(
            "models.ollama_base_url", "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
        )

    @property
    def llm_model(self) -> str:
        return _setting_str("models.llm_model", "LLM_MODEL", "llama3.2:3b")

    @property
    def allowed_chat_models(self) -> list[str]:
        raw = _lookup("ALLOWED_CHAT_MODELS", "models.allowed_chat_models")
        if raw is _MISSING:
            return [self.llm_model]
        if isinstance(raw, list):
            values = [str(item).strip() for item in raw if str(item).strip()]
        else:
            values = [part.strip() for part in str(raw).split(",") if part.strip()]
        return values or [self.llm_model]

    @property
    def chat_model_ceiling(self) -> list[str] | None:
        """Pinned deployment ceiling, or None when unconstrained.

        Empty/unset ALLOWED_CHAT_MODELS means the admin manages models via
        Settings; a non-empty value pins the deployment and admin saves
        outside it are rejected. (Allowed_chat_models above keeps the
        legacy [llm_model] fallback for pre-first-save defaults.)
        """
        raw = _lookup("ALLOWED_CHAT_MODELS", "models.allowed_chat_models")
        if raw is _MISSING:
            return None
        if isinstance(raw, list):
            values = [str(item).strip() for item in raw if str(item).strip()]
        else:
            values = [part.strip() for part in str(raw).split(",") if part.strip()]
        return values or None

    @property
    def vision_model(self) -> str:
        return _setting_str("models.vision_model", "VISION_MODEL", "qwen2.5vl:3b").strip()

    @property
    def intelligence_model(self) -> str:
        return _setting_str("models.intelligence_model", "INTELLIGENCE_MODEL", "qwen2.5vl:3b").strip()

    @property
    def agent_runtime_url(self) -> str:
        return _setting_str("agent_runtime.url", "AGENT_RUNTIME_URL", "http://agent-runtime:8090")

    @property
    def agent_runtime_read_timeout_seconds(self) -> int:
        return _setting_int("agent_runtime.read_timeout_seconds", "AGENT_RUNTIME_READ_TIMEOUT_SECONDS", 450)

    @property
    def agent_capability_secret(self) -> str:
        return _setting_str("security.agent_capability_secret", "AGENT_CAPABILITY_SECRET") or self.secret_key

    @property
    def agent_capability_ttl_seconds(self) -> int:
        return _setting_int("security.agent_capability_ttl_seconds", "AGENT_CAPABILITY_TTL_SECONDS", 300)

    @property
    def embedding_api_url(self) -> str:
        return _setting_str(
            "models.embedding_api_url",
            "EMBEDDING_API_URL",
            f"{self.ollama_base_url}/v1/embeddings",
        )

    @property
    def embedding_model_name(self) -> str:
        # Canonical EMBEDDING_MODEL first; legacy EMBEDDING_MODEL_NAME honored.
        # Initial default only: a model saved in Admin Settings always wins.
        return _setting_str_first(
            "models.embedding_model_name",
            ("EMBEDDING_MODEL", "EMBEDDING_MODEL_NAME"),
            "snowflake-arctic-embed2:cpu",
        ).strip()

    @property
    def embedding_dimension(self) -> int:
        # Canonical EMBEDDING_DIMENSIONS first; legacy EMBEDDING_DIMENSION
        # honored. pgvector HNSW indexes stop at 2000 dims (vector) and
        # 4000 dims (halfvec); anything larger is refused, never silently
        # left unindexed.
        value = _setting_int_first(
            "models.embedding_dimension",
            ("EMBEDDING_DIMENSIONS", "EMBEDDING_DIMENSION"),
            1024,
        )
        if not 1 <= value <= 4000:
            raise ValueError(
                f"EMBEDDING_DIMENSIONS must be between 1 and 4000 "
                f"(2000 or less uses vector+HNSW, 2001-4000 uses halfvec+HNSW), "
                f"got {value}"
            )
        return value

    @property
    def embedding_batch_size(self) -> int:
        return _setting_int("models.embedding_batch_size", "EMBEDDING_BATCH_SIZE", 8)

    @property
    def embedding_timeout(self) -> int:
        return _setting_int("models.embedding_timeout_seconds", "EMBEDDING_TIMEOUT", 120)

    @property
    def vision_timeout(self) -> int:
        return _setting_int("models.vision_timeout_seconds", "VISION_TIMEOUT", 300)

    @property
    def intelligence_timeout(self) -> int:
        return _setting_int("models.intelligence_timeout_seconds", "INTELLIGENCE_TIMEOUT", 300)

    @property
    def intelligence_priority_wait_seconds(self) -> int:
        return _setting_int(
            "models.intelligence_priority_wait_seconds",
            "INTELLIGENCE_PRIORITY_WAIT_SECONDS",
            2,
        )

    @property
    def rerank_mode(self) -> str:
        """Reranker wiring: local (compose service), external (RERANKER_URL),
        or off (skip reranking; retrieval degrades to hybrid order)."""
        mode = _setting_str("reranker.mode", "RERANKER_MODE", "local").strip().lower()
        if mode not in ("local", "external", "off"):
            raise ValueError(
                f'RERANKER_MODE must be "local", "external" or "off", got {mode!r}'
            )
        if mode == "external":
            if "RERANKER_URL" not in os.environ and "RERANK_BASE_URL" not in os.environ:
                raise ValueError("RERANKER_MODE=external needs RERANKER_URL to be set")
            _require_http_host_port("RERANKER_URL", self.rerank_base_url)
        return mode

    @property
    def rerank_base_url(self) -> str:
        # Canonical RERANKER_URL first; legacy RERANK_BASE_URL still honored.
        return _setting_str_first(
            "reranker.base_url", ("RERANKER_URL", "RERANK_BASE_URL"), "http://reranker:7997"
        )

    @property
    def rerank_model(self) -> str:
        # Canonical RERANKER_MODEL first; legacy RERANK_MODEL still honored.
        return _setting_str_first(
            "reranker.model", ("RERANKER_MODEL", "RERANK_MODEL"), "BAAI/bge-reranker-base"
        )

    @property
    def graph_memory_neo4j_enabled(self) -> bool:
        return _setting_bool(
            "graph_memory.neo4j_enabled", "GRAPH_MEMORY_NEO4J_ENABLED", False
        )

    @property
    def graph_memory_neo4j_uri(self) -> str:
        return _setting_str(
            "graph_memory.neo4j_uri", "GRAPH_MEMORY_NEO4J_URI", "bolt://neo4j:7687"
        )

    @property
    def graph_memory_neo4j_user(self) -> str:
        return _setting_str("graph_memory.neo4j_user", "GRAPH_MEMORY_NEO4J_USER", "neo4j")

    @property
    def graph_memory_neo4j_password_file(self) -> str:
        return _setting_str(
            "graph_memory.neo4j_password_file",
            "GRAPH_MEMORY_NEO4J_PASSWORD_FILE",
            "/app/config/secrets/neo4j_password",
        )

    @property
    def graph_memory_neo4j_password(self) -> str:
        """Neo4j password: NEO4J_PASSWORD env first (docker-compose.yaml),
        then the optional config.yml value, then the legacy secrets flat."""

        raw = os.environ.get("NEO4J_PASSWORD", "")
        if raw.strip():
            return raw.strip()
        configured = _setting_str("graph_memory.neo4j_password", "NEO4J_PASSWORD")
        if configured.strip():
            return configured.strip()
        return ""

    @property
    def graph_memory_neo4j_database(self) -> str:
        return _setting_str(
            "graph_memory.neo4j_database", "GRAPH_MEMORY_NEO4J_DATABASE", "neo4j"
        )

    @property
    def graph_memory_neo4j_timeout_seconds(self) -> float:
        return _setting_float(
            "graph_memory.neo4j_timeout_seconds",
            "GRAPH_MEMORY_NEO4J_TIMEOUT_SECONDS",
            3,
        )

    @property
    def graph_question_memory_enabled(self) -> bool:
        return _setting_bool(
            "graph_memory.question_memory_enabled",
            "GRAPH_QUESTION_MEMORY_ENABLED",
            False,
        )

    @property
    def about_me_portrait_model(self) -> str:
        """Model that writes the About Me portrait. Empty means the current
        memory-extraction role model (no behavior change by default); set it
        to select a larger model without code changes."""
        return _setting_str(
            "about_me.portrait_model", "ABOUT_ME_PORTRAIT_MODEL", ""
        ).strip()

    @property
    def interactive_rerank_timeout(self) -> int:
        return _setting_int("reranker.interactive_timeout_seconds", "INTERACTIVE_RERANK_TIMEOUT", 20)

    @property
    def rerank_max_input_chars(self) -> int:
        return _setting_int("reranker.max_input_chars", "RERANK_MAX_INPUT_CHARS", 900)

    @property
    def rerank_min_score(self) -> float:
        value = _setting_float("reranker.min_score", "RERANK_MIN_SCORE", 0.15)
        if not 0.0 <= value <= 1.0:
            raise ValueError("RERANK_MIN_SCORE must be between 0 and 1")
        return value

    @property
    def discovery_min_score(self) -> float:
        """Floor for unscoped (discovery) retrieval.

        Calibrated 2026-09-30 on the live index: irrelevant chunks cluster
        at rerank 0.119-0.22 (0.22 is the reranker no-signal floor) while
        genuine matches score 0.47+. Below this floor with no file scope,
        retrieval returns nothing instead of best-effort junk, and the
        caller answers via the unverified_explanatory path.
        """
        value = _setting_float("retriever.discovery_min_score", "DISCOVERY_MIN_SCORE", 0.35)
        if not 0.0 <= value <= 1.0:
            raise ValueError("DISCOVERY_MIN_SCORE must be between 0 and 1")
        return value

    @property
    def enable_reranking(self) -> bool:
        return _setting_bool("reranker.enabled", "ENABLE_RERANKING", True)

    @property
    def searxng_url(self) -> str:
        return _setting_str("search.searxng_url", "SEARXNG_URL", "http://searxng:8080")

    @property
    def web_search_timeout(self) -> int:
        return _setting_int("search.web_search_timeout_seconds", "WEB_SEARCH_TIMEOUT", 12)

    @property
    def web_search_hard_timeout(self) -> float:
        return _setting_float("search.web_search_hard_timeout_seconds", "WEB_SEARCH_HARD_TIMEOUT", 12.0)

    @property
    def web_search_cache_ttl_seconds(self) -> int:
        return _setting_int("search.web_search_cache_ttl_seconds", "WEB_SEARCH_CACHE_TTL", 300)

    @property
    def web_page_fetch_enabled(self) -> bool:
        return _setting_bool("search.web_page_fetch_enabled", "WEB_PAGE_FETCH_ENABLED", False)

    @property
    def web_page_max_bytes(self) -> int:
        return _setting_int("search.web_page_max_bytes", "WEB_PAGE_MAX_BYTES", 2 * 1024 * 1024)

    @property
    def web_page_max_redirects(self) -> int:
        return _setting_int("search.web_page_max_redirects", "WEB_PAGE_MAX_REDIRECTS", 3)

    @property
    def citation_relevance_enabled(self) -> bool:
        return _setting_bool("citation.relevance_enabled", "CITATION_RELEVANCE_ENABLED", True)

    @property
    def citation_relevance_threshold(self) -> float:
        return _setting_float("citation.relevance_threshold", "CITATION_RELEVANCE_THRESHOLD", 0.5)

    @property
    def opendataloader_hybrid_timeout_ms(self) -> int:
        return _setting_int(
            "ingestion.opendataloader_hybrid_timeout_ms",
            "OPENDATALOADER_HYBRID_TIMEOUT_MS",
            3_600_000,
        )

    @property
    def max_file_size(self) -> int:
        return _setting_int("files.max_size_bytes", "MAX_FILE_SIZE", 512 * 1024**2)

    @property
    def temp_dir(self) -> Path:
        return Path(_setting_str("files.temp_dir", "TEMP_DIR", "/tmp/vault_temp"))

    @property
    def ingestion_temp_root(self) -> Path:
        return Path(
            _setting_str(
                "files.ingestion_temp_root",
                "INGESTION_TEMP_ROOT",
                str(self.temp_dir),
            )
        )

    @property
    def encryption_script(self) -> Path:
        return Path(_setting_str("encryption.script_path", "ENCRYPTION_SCRIPT", "/app/enc/encryption.py"))

    @property
    def public_key(self) -> Path:
        return Path(_setting_str("encryption.public_key_path", "PUBLIC_KEY", "/app/enc/public_4096.pem"))

    @property
    def private_key(self) -> Path:
        indirection = _setting_str("encryption.private_key_env_var_name", "PRIVATE_KEY_ENV_VAR_NAME", "")
        if indirection and os.environ.get(indirection):
            return Path(os.environ[indirection])
        return Path(_setting_str("encryption.private_key_path", "PRIVATE_KEY", "/run/secrets/private_key"))

    @property
    def algorithm(self) -> str:
        value = _setting_str("security.algorithm", "ALGORITHM", "HS256")
        if value not in {"HS256", "HS384", "HS512"}:
            raise ValueError("ALGORITHM must be an HMAC SHA-2 JWT algorithm")
        return value

    @property
    def access_token_expire_minutes(self) -> int:
        # A saved admin timeout drives both lifetimes; otherwise deployment.
        from app.services.model_config import get_saved_session_timeout_minutes

        saved = get_saved_session_timeout_minutes()
        if saved is not None:
            return saved
        return _setting_int("security.access_token_expire_minutes", "ACCESS_TOKEN_EXPIRE_MINUTES", 120)

    @property
    def session_idle_timeout_minutes(self) -> int:
        from app.services.model_config import get_saved_session_timeout_minutes

        saved = get_saved_session_timeout_minutes()
        if saved is not None:
            return saved
        value = _setting_int("security.session_idle_timeout_minutes", "SESSION_IDLE_TIMEOUT_MINUTES", 120)
        if value <= 0:
            raise ValueError("SESSION_IDLE_TIMEOUT_MINUTES must be greater than zero")
        return value

    @property
    def refresh_token_expire_days(self) -> int:
        return _setting_int("security.refresh_token_expire_days", "REFRESH_TOKEN_EXPIRE_DAYS", 7)

    @property
    def max_active_sessions_per_user(self) -> int:
        return _setting_int("security.max_active_sessions_per_user", "MAX_ACTIVE_SESSIONS", 5)

    @property
    def registration_enabled(self) -> bool:
        # A saved admin toggle takes effect immediately with no restart;
        # otherwise the deployment configuration rules.
        from app.services.model_config import get_saved_registration_enabled

        saved = get_saved_registration_enabled()
        if saved is not None:
            return saved
        return _setting_bool(
            "security.registration_enabled",
            "REGISTRATION_ENABLED",
            not self.is_production,
        )

    @property
    def auth_rate_limit_fail_closed(self) -> bool:
        return _setting_bool(
            "security.auth_rate_limit_fail_closed",
            "AUTH_RATE_LIMIT_FAIL_CLOSED",
            self.is_production,
        )

    @property
    def validate_preview_ip(self) -> bool:
        return _setting_bool("security.validate_preview_ip", "VALIDATE_PREVIEW_IP", True)

    @property
    def allowed_origins(self) -> list[str]:
        return [
            value.strip()
            for value in _setting_str(
                "security.allowed_origins", "ALLOWED_ORIGINS", "http://localhost:3005"
            ).split(",")
            if value.strip()
        ]

    @property
    def google_client_id(self) -> str:
        return _setting_str("google_oauth.client_id", "GOOGLE_CLIENT_ID")

    @property
    def google_client_secret(self) -> str:
        return _setting_str("google_oauth.client_secret", "GOOGLE_CLIENT_SECRET")

    @property
    def google_redirect_uri(self) -> str:
        return _setting_str("google_oauth.redirect_uri", "GOOGLE_REDIRECT_URI")


settings = Settings()
