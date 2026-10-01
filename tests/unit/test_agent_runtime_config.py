import dataclasses

import pytest
from pydantic import ValidationError

from agent_runtime.config import (
    DEFAULT_MODEL,
    ConfigurationError,
    ModelNotAllowedError,
    RuntimeSettings,
)
from agent_runtime.schemas import RunRequest


def test_default_is_local_llama_and_allowlisted():
    settings = RuntimeSettings.from_env({})

    assert settings.default_model == "llama3.2:3b"
    assert settings.allowed_models == frozenset({DEFAULT_MODEL})
    assert settings.tool_gateway_url == "http://api:8080/api/internal/agent"
    assert settings.ollama_num_predict == 2048
    assert settings.queue_wait_timeout_seconds == 30
    assert settings.run_timeout_seconds == 420
    assert settings.cuga_max_steps == 6
    assert settings.resolve_model(None) == DEFAULT_MODEL


def test_model_override_requires_explicit_allowlist():
    settings = RuntimeSettings.from_env(
        {
            "LAVIX_AGENT_MODEL": "local/default",
            "LAVIX_AGENT_ALLOWED_MODELS": "local/default,local/approved",
            "LAVIX_AGENT_QUEUE_WAIT_TIMEOUT_SECONDS": "17",
        }
    )

    assert settings.resolve_model("local/approved") == "local/approved"
    assert settings.queue_wait_timeout_seconds == 17
    with pytest.raises(ModelNotAllowedError):
        settings.resolve_model("openrouter/auto")


def test_default_model_must_be_in_allowlist():
    with pytest.raises(ConfigurationError, match="default_model"):
        RuntimeSettings(
            default_model="local/default",
            allowed_models=frozenset({"local/other"}),
        )


def test_runtime_has_no_database_or_object_storage_configuration():
    field_names = {field.name.lower() for field in dataclasses.fields(RuntimeSettings)}
    forbidden_fragments = {"database", "postgres", "redis", "s3", "minio", "secret_key"}

    assert not any(fragment in field_name for fragment in forbidden_fragments for field_name in field_names)


def test_request_rejects_unknown_fields_and_requires_final_user_message():
    with pytest.raises(ValidationError):
        RunRequest.model_validate(
            {
                "messages": [{"role": "assistant", "content": "done"}],
                "provider": "openrouter",
            }
        )


def test_urls_reject_embedded_credentials():
    with pytest.raises(ConfigurationError, match="credentials"):
        RuntimeSettings(ollama_base_url="http://user:pass@ollama:11434")


def test_web_rewrite_defaults_are_web_leg_only():
    settings = RuntimeSettings.from_env({})

    assert settings.web_rewrite_enabled is True
    assert settings.web_rewrite_model is None
    assert settings.resolved_rewrite_model == DEFAULT_MODEL
    assert settings.web_rewrite_timeout_seconds == 20.0
    assert settings.web_rewrite_confidence_min == 0.5
    assert settings.web_rewrite_history_messages == 6
    assert settings.web_rewrite_history_token_cap == 1500
    assert settings.web_rewrite_assistant_trim_chars == 300
    assert settings.web_planner_timeout_seconds == 20.0


def test_web_rewrite_env_override_and_model_allowlist():
    settings = RuntimeSettings.from_env(
        {
            "LAVIX_AGENT_MODEL": "local/default",
            "LAVIX_AGENT_ALLOWED_MODELS": "local/default,local/rewrite",
            "LAVIX_WEB_REWRITE_ENABLED": "false",
            "LAVIX_WEB_REWRITE_MODEL": "local/rewrite",
            "LAVIX_WEB_REWRITE_TIMEOUT_SECONDS": "5.5",
            "LAVIX_WEB_REWRITE_CONFIDENCE_MIN": "0.7",
            "LAVIX_WEB_REWRITE_HISTORY_MESSAGES": "4",
            "LAVIX_WEB_REWRITE_HISTORY_TOKEN_CAP": "900",
            "LAVIX_WEB_REWRITE_ASSISTANT_TRIM_CHARS": "120",
            "LAVIX_WEB_PLANNER_TIMEOUT_SECONDS": "25.5",
        }
    )

    assert settings.web_rewrite_enabled is False
    assert settings.resolved_rewrite_model == "local/rewrite"
    assert settings.web_rewrite_timeout_seconds == 5.5
    assert settings.web_rewrite_confidence_min == 0.7
    assert settings.web_rewrite_history_messages == 4
    assert settings.web_rewrite_history_token_cap == 900
    assert settings.web_rewrite_assistant_trim_chars == 120
    assert settings.web_planner_timeout_seconds == 25.5


@pytest.mark.parametrize(
    "env",
    [
        {"LAVIX_WEB_REWRITE_MODEL": "local/unlisted"},
        {"LAVIX_WEB_REWRITE_TIMEOUT_SECONDS": "0"},
        {"LAVIX_WEB_REWRITE_TIMEOUT_SECONDS": "soon"},
        {"LAVIX_WEB_REWRITE_CONFIDENCE_MIN": "1.5"},
        {"LAVIX_WEB_REWRITE_HISTORY_MESSAGES": "0"},
        {"LAVIX_WEB_REWRITE_HISTORY_TOKEN_CAP": "-3"},
        {"LAVIX_WEB_PLANNER_TIMEOUT_SECONDS": "0"},
        {"LAVIX_WEB_PLANNER_TIMEOUT_SECONDS": "soon"},
    ],
)
def test_web_rewrite_rejects_bad_values(env):
    with pytest.raises(ConfigurationError):
        RuntimeSettings.from_env(env)


def test_request_allowlist_overrides_static_env_allowlist():
    settings = RuntimeSettings.from_env(
        {
            "LAVIX_AGENT_MODEL": "local/default",
            "LAVIX_AGENT_ALLOWED_MODELS": "local/default",
        }
    )

    assert settings.resolve_model("per-request/model", ["per-request/model"]) == "per-request/model"
    with pytest.raises(ModelNotAllowedError):
        settings.resolve_model("per-request/other", ["per-request/model"])
    # Absent override preserves the static env allowlist.
    assert settings.resolve_model("local/default", None) == "local/default"
    with pytest.raises(ModelNotAllowedError):
        settings.resolve_model("per-request/model", None)


def test_run_options_accepts_per_request_allowlist():
    from agent_runtime.schemas import RunOptions

    assert RunOptions().allowed_models is None
    assert RunOptions(allowed_models=["a:latest"]).allowed_models == ["a:latest"]


def test_model_max_num_ctx_parses_from_env():
    settings = RuntimeSettings.from_env(
        {
            "LAVIX_AGENT_MODEL": "local/default",
            "LAVIX_AGENT_MAX_NUM_CTX": "65536",
        }
    )

    assert settings.model_max_num_ctx == 65536
    assert RuntimeSettings.from_env({}).model_max_num_ctx == 32768
