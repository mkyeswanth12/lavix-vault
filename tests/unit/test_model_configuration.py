from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import admin
from app.services.model_config import (
    ChatModelConfiguration,
    ModelConfigurationConflict,
    ModelConfigurationRepository,
    OptionalModelConfiguration,
    SystemAIConfiguration,
    SystemAIConfigurationWrite,
    clock_timezone_source,
    resolve_account_chat_model,
    resolve_clock_timezone,
)
from app.services.model_config import (
    _read_saved_preferences as _real_read_saved_preferences,
)
from app.services.model_service import ModelService


class QueueCursor:
    def __init__(self, rows: list[dict[str, Any] | None]) -> None:
        self.rows = list(rows)
        self.executed: list[tuple[str, object]] = []

    def execute(self, sql: str, params: object = None) -> None:
        self.executed.append((" ".join(sql.lower().split()), params))

    def fetchone(self) -> dict[str, Any] | None:
        return self.rows.pop(0) if self.rows else None


class QueueConnection:
    def __init__(self, rows: list[dict[str, Any] | None]) -> None:
        self.cursor_value = QueueCursor(rows)

    def cursor(self) -> QueueCursor:
        return self.cursor_value


def database_context(connection: object):
    @contextmanager
    def get_db() -> Iterator[object]:
        yield connection

    return get_db


@pytest.fixture(autouse=True)
def _reset_preferences_cache():
    """The preferences TTL cache is process-global: never leak values between tests."""

    from app.services import model_config

    model_config.invalidate_preferences_cache()
    yield
    model_config.invalidate_preferences_cache()


def stored_config_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "revision": 4,
        "chat_enabled": True,
        "chat_default_model": "reasoner:latest",
        "chat_allowed_models": ["chat:latest", "reasoner:latest"],
        "vision_enabled": True,
        "vision_model": "vision:latest",
        "intelligence_enabled": True,
        "intelligence_model": "intelligence:latest",
        "memory_extraction_enabled": False,
        "memory_extraction_model": "reasoner:latest",
        "reranker_enabled": False,
        "clock_timezone": None,
        "updated_by": 8,
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def write_config() -> SystemAIConfigurationWrite:
    return SystemAIConfigurationWrite(
        chat=ChatModelConfiguration(
            enabled=True,
            default_model="reasoner:latest",
            allowed_models=("chat:latest", "reasoner:latest"),
        ),
        vision=OptionalModelConfiguration(enabled=True, model="vision:latest"),
        intelligence=OptionalModelConfiguration(enabled=True, model="intelligence:latest"),
        memory_extraction=OptionalModelConfiguration(enabled=False, model="reasoner:latest"),
        reranker_enabled=False,
        clock_timezone=None,
    )


def test_repository_uses_environment_until_the_singleton_is_saved(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "chat:latest")
    monkeypatch.setenv("ALLOWED_CHAT_MODELS", "chat:latest,reasoner:latest")
    monkeypatch.setenv("VISION_MODEL", "vision:latest")
    monkeypatch.setenv("INTELLIGENCE_MODEL", "intelligence:latest")
    monkeypatch.setenv("ENABLE_RERANKING", "true")
    connection = QueueConnection([None])

    config = ModelConfigurationRepository(connection).get()

    assert config.revision == 0
    assert config.source == "environment"
    assert config.chat == ChatModelConfiguration(
        enabled=True,
        default_model="chat:latest",
        allowed_models=("chat:latest", "reasoner:latest"),
    )
    assert config.vision == OptionalModelConfiguration(enabled=True, model="vision:latest")
    assert config.intelligence.model == "intelligence:latest"
    assert config.memory_extraction.model == "intelligence:latest"
    assert config.reranker_enabled is True


def test_account_chat_resolution_rejects_stale_or_uninstalled_preferences() -> None:
    config = SystemAIConfiguration(
        revision=4,
        source="database",
        chat=ChatModelConfiguration(
            enabled=True,
            default_model="chat:latest",
            allowed_models=("chat:latest", "reasoner:latest"),
        ),
        vision=OptionalModelConfiguration(enabled=False, model=None),
        intelligence=OptionalModelConfiguration(enabled=False, model=None),
        memory_extraction=OptionalModelConfiguration(enabled=False, model=None),
        reranker_enabled=False,
    )

    selected = resolve_account_chat_model(
        config,
        "reasoner:latest",
        ("chat:latest", "reasoner:latest"),
    )
    removed = resolve_account_chat_model(
        config,
        "reasoner:latest",
        ("chat:latest",),
    )

    assert selected.preferred_model == "reasoner:latest"
    assert selected.active_model == "reasoner:latest"
    assert selected.available is True
    assert removed.preferred_model is None
    assert removed.active_model == "chat:latest"
    assert removed.available is True


def test_account_chat_resolution_keeps_an_unavailable_default_visible() -> None:
    config = SystemAIConfiguration(
        revision=4,
        source="database",
        chat=ChatModelConfiguration(
            enabled=True,
            default_model="chat:latest",
            allowed_models=("chat:latest",),
        ),
        vision=OptionalModelConfiguration(enabled=False, model=None),
        intelligence=OptionalModelConfiguration(enabled=False, model=None),
        memory_extraction=OptionalModelConfiguration(enabled=False, model=None),
        reranker_enabled=False,
    )

    resolution = resolve_account_chat_model(config, None, ())

    assert resolution.preferred_model is None
    assert resolution.active_model == "chat:latest"
    assert resolution.available is False


def test_repository_resolves_nullable_columns_against_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "chat:latest")
    monkeypatch.setenv("ALLOWED_CHAT_MODELS", "chat:latest")
    monkeypatch.setenv("VISION_MODEL", "env-vision:latest")
    connection = QueueConnection(
        [
            stored_config_row(
                revision=7,
                chat_enabled=False,
                vision_enabled=False,
                vision_model=None,
                intelligence_model=None,
                reranker_enabled=None,
            )
        ]
    )

    config = ModelConfigurationRepository(connection).get()

    assert config.revision == 7
    assert config.source == "database"
    assert config.chat.enabled is False
    assert config.vision == OptionalModelConfiguration(enabled=False, model="env-vision:latest")
    assert config.intelligence.model is not None
    assert config.reranker_enabled is True


def test_repository_replaces_one_revision_and_clears_disallowed_preferences() -> None:
    connection = QueueConnection([stored_config_row(revision=5)])

    config = ModelConfigurationRepository(connection).replace(
        write_config(),
        expected_revision=4,
        updated_by=8,
    )

    assert config.revision == 5
    update, cleanup = connection.cursor_value.executed
    assert "update system_ai_configuration" in update[0]
    assert "where singleton_id = 1 and revision = %s" in update[0]
    assert update[1][-1] == 4
    assert cleanup == (
        "update users set preferred_chat_model = null where preferred_chat_model is not null "
        "and not (preferred_chat_model = any(%s::text[]))",
        (["chat:latest", "reasoner:latest"],),
    )


def test_repository_creates_only_the_absent_revision_zero_singleton() -> None:
    connection = QueueConnection([None, stored_config_row(revision=1)])

    config = ModelConfigurationRepository(connection).replace(
        write_config(),
        expected_revision=0,
        updated_by=8,
    )

    assert config.revision == 1
    update, insert, cleanup = connection.cursor_value.executed
    assert update[1][-1] == 0
    assert "insert into system_ai_configuration" in insert[0]
    assert "on conflict (singleton_id) do nothing" in insert[0]
    assert insert[1][-1] == 8
    assert cleanup[0].startswith("update users set preferred_chat_model = null")


def test_repository_reports_the_current_revision_after_a_stale_write() -> None:
    connection = QueueConnection([None, {"revision": 9}])

    try:
        ModelConfigurationRepository(connection).replace(
            write_config(),
            expected_revision=4,
            updated_by=8,
        )
    except ModelConfigurationConflict as exc:
        assert exc.expected_revision == 4
        assert exc.current_revision == 9
    else:
        raise AssertionError("stale replace must fail")


class LocalModelService:
    def get_available_models(self) -> list[dict[str, Any]]:
        return [
            {"name": "chat:latest", "size": 1},
            {"name": "reasoner:latest", "size": 2},
            {"name": "vision:latest", "size": 3, "modalities": ["text", "image"]},
            {"name": "intelligence:latest", "size": 4},
            {"name": "embed:latest", "size": 5},
        ]

    @staticmethod
    def get_model_capabilities(model: str) -> frozenset[str]:
        if model == "vision:latest":
            return frozenset({"completion", "vision"})
        return frozenset({"completion"})

    @staticmethod
    def is_service_available(_url: str) -> bool:
        return True


class StubRepository:
    current = SystemAIConfiguration(
        revision=4,
        source="database",
        chat=ChatModelConfiguration(
            enabled=True,
            default_model="chat:latest",
            allowed_models=("chat:latest", "reasoner:latest"),
        ),
        vision=OptionalModelConfiguration(enabled=True, model="vision:latest"),
        intelligence=OptionalModelConfiguration(enabled=True, model="intelligence:latest"),
        memory_extraction=OptionalModelConfiguration(enabled=False, model="reasoner:latest"),
        reranker_enabled=True,
    )
    replace_calls: list[tuple[SystemAIConfigurationWrite, int, int]] = []

    def __init__(self, _connection: object) -> None:
        pass

    def get(self) -> SystemAIConfiguration:
        return self.current

    def replace(
        self,
        value: SystemAIConfigurationWrite,
        *,
        expected_revision: int,
        updated_by: int,
    ) -> SystemAIConfiguration:
        self.replace_calls.append((value, expected_revision, updated_by))
        return SystemAIConfiguration(
            revision=expected_revision + 1,
            source="database",
            chat=value.chat,
            vision=value.vision,
            intelligence=value.intelligence,
            memory_extraction=value.memory_extraction,
            reranker_enabled=value.reranker_enabled,
            clock_timezone=value.clock_timezone,
            model_max_num_ctx=value.model_max_num_ctx,
            fallback_chat_model=value.fallback_chat_model,
            file_scope=value.file_scope,
            top_k=value.top_k,
            search_depth=value.search_depth,
            session_timeout_minutes=value.session_timeout_minutes,
            registration_enabled=value.registration_enabled,
            updated_by=updated_by,
        )


def admin_client(user: dict[str, Any]) -> TestClient:
    app = FastAPI()
    app.include_router(admin.router, prefix="/api/admin")
    app.dependency_overrides[admin.get_current_user] = lambda: user
    return TestClient(app)


def admin_update_payload() -> dict[str, Any]:
    return {
        "expected_revision": 4,
        "chat": {
            "enabled": True,
            "default_model": "reasoner:latest",
            "allowed_models": ["chat:latest", "reasoner:latest"],
        },
        "vision": {"enabled": True, "model": "vision:latest"},
        "intelligence": {"enabled": True, "model": "intelligence:latest"},
        "memory_extraction": {"enabled": False, "model": "reasoner:latest"},
        "reranker": {"enabled": False},
    }


def configure_admin_route_fakes(monkeypatch) -> None:
    StubRepository.replace_calls = []
    monkeypatch.setattr(admin, "get_db", database_context(object()))
    monkeypatch.setattr(admin, "get_model_service", LocalModelService)
    monkeypatch.setattr(admin, "ModelConfigurationRepository", StubRepository)
    monkeypatch.setenv("ALLOWED_CHAT_MODELS", "chat:latest,reasoner:latest")
    monkeypatch.setenv("LLM_MODEL", "chat:latest")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "embed:latest")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")


def test_admin_model_configuration_is_admin_only_and_reports_read_only_roles(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)

    denied = admin_client({"id": 4, "is_admin": False}).get("/api/admin/ai-models")
    response = admin_client({"id": 8, "is_admin": True}).get("/api/admin/ai-models")

    assert denied.status_code == 403
    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"] == 4
    assert payload["deployment_allowed_chat_models"] == ["chat:latest", "reasoner:latest"]
    assert payload["embedding"] == {
        "model": "embed:latest",
        "dimension": 1024,
        "configured": True,
        "available": True,
        "optional": False,
        "read_only": True,
    }
    assert payload["reranker"]["read_only_model"] is True
    assert payload["clock"] == {
        "timezone": None,
        "effective_timezone": "Asia/Kolkata",
        "source": "environment",
    }
    assert "openrouter" not in str(payload).lower()


def test_admin_model_configuration_replaces_the_complete_document(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)

    response = admin_client({"id": 8, "is_admin": True}).put(
        "/api/admin/ai-models",
        json=admin_update_payload(),
    )

    assert response.status_code == 200
    assert response.json()["revision"] == 5
    assert response.json()["chat"]["default_model"] == "reasoner:latest"
    assert response.json()["reranker"]["enabled"] is False
    assert len(StubRepository.replace_calls) == 1
    write, expected_revision, updated_by = StubRepository.replace_calls[0]
    assert write.chat.allowed_models == ("chat:latest", "reasoner:latest")
    # Payloads predating the clock override omit it: no override, not an error.
    assert write.clock_timezone is None
    assert expected_revision == 4
    assert updated_by == 8


def environment_config() -> SystemAIConfiguration:
    return ModelConfigurationRepository(QueueConnection([None])).get()


def test_clock_timezone_override_resolves_against_environment() -> None:
    assert clock_timezone_source(environment_config()) == "environment"
    assert resolve_clock_timezone(environment_config()) == "Asia/Kolkata"

    configured = ModelConfigurationRepository(
        QueueConnection([stored_config_row(clock_timezone="America/New_York")])
    ).get()
    assert configured.clock_timezone == "America/New_York"
    assert clock_timezone_source(configured) == "database"
    assert resolve_clock_timezone(configured) == "America/New_York"

    blanked = ModelConfigurationRepository(
        QueueConnection([stored_config_row(clock_timezone="  ")])
    ).get()
    assert blanked.clock_timezone is None
    assert resolve_clock_timezone(blanked) == "Asia/Kolkata"


def test_admin_clock_override_accepts_valid_zone_and_rejects_garbage(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    client = admin_client({"id": 8, "is_admin": True})

    override = admin_update_payload()
    override["clock"] = {"timezone": "America/New_York"}
    response = client.put("/api/admin/ai-models", json=override)

    assert response.status_code == 200
    assert response.json()["clock"] == {
        "timezone": "America/New_York",
        "effective_timezone": "America/New_York",
        "source": "database",
    }
    write = StubRepository.replace_calls[0][0]
    assert write.clock_timezone == "America/New_York"

    StubRepository.replace_calls = []
    garbage = admin_update_payload()
    garbage["clock"] = {"timezone": "Not/AZone"}
    assert client.put("/api/admin/ai-models", json=garbage).status_code == 422
    assert StubRepository.replace_calls == []

    reset = admin_update_payload()
    reset["clock"] = {"timezone": "   "}
    reset_response = client.put("/api/admin/ai-models", json=reset)
    assert reset_response.status_code == 200
    assert reset_response.json()["clock"]["timezone"] is None
    assert reset_response.json()["clock"]["source"] == "environment"
    assert StubRepository.replace_calls[0][0].clock_timezone is None


def test_admin_model_configuration_rejects_ceiling_violations_and_non_vlm(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    client = admin_client({"id": 8, "is_admin": True})
    outside_ceiling = admin_update_payload()
    outside_ceiling["chat"]["default_model"] = "intelligence:latest"
    outside_ceiling["chat"]["allowed_models"] = ["intelligence:latest"]
    not_a_vlm = admin_update_payload()
    not_a_vlm["vision"]["model"] = "reasoner:latest"

    ceiling_response = client.put("/api/admin/ai-models", json=outside_ceiling)
    vision_response = client.put("/api/admin/ai-models", json=not_a_vlm)

    assert ceiling_response.status_code == 422
    assert "ALLOWED_CHAT_MODELS" in ceiling_response.json()["detail"]
    assert vision_response.status_code == 422
    assert "not vision-capable" in vision_response.json()["detail"]
    assert StubRepository.replace_calls == []


def test_admin_model_save_succeeds_without_ceiling_for_installed_models(
    monkeypatch,
) -> None:
    configure_admin_route_fakes(monkeypatch)
    monkeypatch.delenv("ALLOWED_CHAT_MODELS", raising=False)
    client = admin_client({"id": 8, "is_admin": True})
    payload = admin_update_payload()
    payload["chat"]["default_model"] = "intelligence:latest"
    payload["chat"]["allowed_models"] = ["intelligence:latest"]

    response = client.put("/api/admin/ai-models", json=payload)

    assert response.status_code == 200
    assert response.json()["chat"]["default_model"] == "intelligence:latest"
    assert len(StubRepository.replace_calls) == 1


def test_admin_model_save_still_rejects_uninstalled_models_without_ceiling(
    monkeypatch,
) -> None:
    configure_admin_route_fakes(monkeypatch)
    monkeypatch.delenv("ALLOWED_CHAT_MODELS", raising=False)
    client = admin_client({"id": 8, "is_admin": True})
    payload = admin_update_payload()
    payload["chat"]["default_model"] = "ghost:latest"
    payload["chat"]["allowed_models"] = ["ghost:latest"]

    response = client.put("/api/admin/ai-models", json=payload)

    assert response.status_code == 422
    assert "not installed" in response.json()["detail"]
    assert StubRepository.replace_calls == []


def test_admin_can_disable_unavailable_optional_roles_without_a_vision_probe(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    payload = admin_update_payload()
    for role in ("vision", "intelligence", "memory_extraction"):
        payload[role] = {"enabled": False, "model": f"missing-{role}:latest"}

    def unexpected_capability_probe(_model: str) -> frozenset[str]:
        raise AssertionError("disabled Vision must not be capability-probed")

    monkeypatch.setattr(
        LocalModelService,
        "get_model_capabilities",
        staticmethod(unexpected_capability_probe),
    )
    response = admin_client({"id": 8, "is_admin": True}).put(
        "/api/admin/ai-models",
        json=payload,
    )

    assert response.status_code == 200
    for role in ("vision", "intelligence", "memory_extraction"):
        assert response.json()[role] == {
            "enabled": False,
            "model": f"missing-{role}:latest",
            "configured": True,
            "available": False,
            "optional": True,
        }
    write = StubRepository.replace_calls[0][0]
    assert write.vision == OptionalModelConfiguration(
        enabled=False,
        model="missing-vision:latest",
    )


def test_admin_model_configuration_rejects_stale_revisions_with_a_409(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)

    def stale_replace(
        _self,
        _value,
        *,
        expected_revision: int,
        updated_by: int,
    ) -> SystemAIConfiguration:
        del updated_by
        raise ModelConfigurationConflict(
            expected_revision=expected_revision,
            current_revision=7,
        )

    monkeypatch.setattr(StubRepository, "replace", stale_replace)
    response = admin_client({"id": 8, "is_admin": True}).put(
        "/api/admin/ai-models",
        json=admin_update_payload(),
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "model_configuration_conflict",
        "expected_revision": 4,
        "current_revision": 7,
    }


def test_admin_model_configuration_does_not_accept_embedding_or_provider_mutations(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    payload = admin_update_payload()
    payload["embedding"] = {"model": "other", "dimension": 384}
    payload["provider"] = "openrouter"

    response = admin_client({"id": 8, "is_admin": True}).put(
        "/api/admin/ai-models",
        json=payload,
    )

    assert response.status_code == 422
    assert StubRepository.replace_calls == []


def test_model_service_reads_and_caches_declared_ollama_capabilities(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, list[str]]:
            return {"capabilities": ["completion", "vision"]}

    def post(url: str, *, json: dict[str, str], timeout: int):
        assert timeout == 5
        calls.append((url, json))
        return Response()

    monkeypatch.setattr("app.services.model_service.requests.post", post)
    service = ModelService("http://ollama.test")

    first = service.get_model_capabilities(" vision:latest ")
    second = service.get_model_capabilities("vision:latest")

    assert first == second == frozenset({"completion", "vision"})
    assert calls == [
        (
            "http://ollama.test/api/show",
            {"model": "vision:latest"},
        )
    ]


def test_model_max_num_ctx_defaults_when_row_predates_column() -> None:
    config = ModelConfigurationRepository(QueueConnection([stored_config_row()])).get()

    assert config.model_max_num_ctx == 16384


def test_model_max_num_ctx_round_trips_stored_value() -> None:
    row = stored_config_row(model_max_num_ctx=65536)
    config = ModelConfigurationRepository(QueueConnection([row])).get()

    assert config.model_max_num_ctx == 65536


def test_admin_model_max_num_ctx_accepts_allowlist_and_rejects_other() -> None:
    from app.routers import admin as admin_module

    assert admin_module._normalize_max_num_ctx(32768) == 32768
    assert admin_module._normalize_max_num_ctx(None) is None
    import pytest

    with pytest.raises(ValueError, match="must be one of"):
        admin_module._normalize_max_num_ctx(20000)
    with pytest.raises(ValueError, match="must be one of"):
        admin_module._normalize_max_num_ctx(0)


def test_admin_put_persists_model_max_num_ctx(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    client = admin_client({"id": 8, "is_admin": True})
    payload = admin_update_payload()
    payload["model_max_num_ctx"] = 65536

    response = client.put("/api/admin/ai-models", json=payload)

    assert response.status_code == 200
    assert response.json()["model_max_num_ctx"] == 65536
    assert response.json()["model_max_num_ctx_options"] == [16384, 32768, 65536, 131072, 262144]
    write = StubRepository.replace_calls[-1][0]
    assert write.model_max_num_ctx == 65536


def test_admin_put_without_cap_preserves_stored_value(monkeypatch) -> None:
    import dataclasses

    configure_admin_route_fakes(monkeypatch)
    monkeypatch.setattr(
        StubRepository,
        "current",
        dataclasses.replace(StubRepository.current, model_max_num_ctx=131072),
    )
    client = admin_client({"id": 8, "is_admin": True})

    response = client.put("/api/admin/ai-models", json=admin_update_payload())

    assert response.status_code == 200
    write = StubRepository.replace_calls[-1][0]
    assert write.model_max_num_ctx == 131072


def _fallback_config() -> SystemAIConfiguration:
    return SystemAIConfiguration(
        revision=4,
        source="database",
        chat=ChatModelConfiguration(
            enabled=True,
            default_model="chat:latest",
            allowed_models=("chat:latest", "reasoner:latest", "fallback:latest"),
        ),
        vision=OptionalModelConfiguration(enabled=False, model=None),
        intelligence=OptionalModelConfiguration(enabled=False, model=None),
        memory_extraction=OptionalModelConfiguration(enabled=False, model=None),
        reranker_enabled=False,
        fallback_chat_model="fallback:latest",
    )


def test_account_chat_resolution_serves_fallback_when_default_unusable() -> None:
    resolution = resolve_account_chat_model(
        _fallback_config(),
        "reasoner:latest",
        ("fallback:latest",),
    )

    assert resolution.preferred_model is None
    assert resolution.active_model == "fallback:latest"
    assert resolution.available is True
    assert resolution.fallback_used is True


def test_account_chat_resolution_prefers_default_over_fallback() -> None:
    resolution = resolve_account_chat_model(
        _fallback_config(),
        None,
        ("chat:latest", "fallback:latest"),
    )

    assert resolution.active_model == "chat:latest"
    assert resolution.fallback_used is False


def test_account_chat_resolution_ignores_stale_fallback() -> None:
    resolution = resolve_account_chat_model(
        _fallback_config(),
        None,
        ("other:latest",),
    )

    assert resolution.active_model == "chat:latest"
    assert resolution.available is False
    assert resolution.fallback_used is False


def test_admin_rejects_fallback_outside_allowed_or_uninstalled(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    monkeypatch.delenv("ALLOWED_CHAT_MODELS", raising=False)
    client = admin_client({"id": 8, "is_admin": True})

    outside = admin_update_payload()
    outside["fallback_chat_model"] = "ghost:latest"
    response = client.put("/api/admin/ai-models", json=outside)
    assert response.status_code == 422
    assert "Fallback model" in response.json()["detail"]

    not_installed = admin_update_payload()
    not_installed["chat"]["default_model"] = "chat:latest"
    not_installed["chat"]["allowed_models"] = ["chat:latest", "ghost:latest"]
    not_installed["fallback_chat_model"] = "ghost:latest"
    response = client.put("/api/admin/ai-models", json=not_installed)
    assert response.status_code == 422
    assert "not installed" in response.json()["detail"]


def test_admin_put_persists_valid_fallback(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    client = admin_client({"id": 8, "is_admin": True})
    payload = admin_update_payload()
    payload["fallback_chat_model"] = "reasoner:latest"

    response = client.put("/api/admin/ai-models", json=payload)

    assert response.status_code == 200
    assert response.json()["chat"]["fallback_chat_model"] == "reasoner:latest"
    assert StubRepository.replace_calls[-1][0].fallback_chat_model == "reasoner:latest"


def test_rag_retrieval_settings_default_when_row_predates_columns() -> None:
    config = ModelConfigurationRepository(QueueConnection([stored_config_row()])).get()

    assert config.file_scope == 5
    assert config.top_k == 20


def test_rag_retrieval_settings_round_trip_stored_values() -> None:
    row = stored_config_row(file_scope=3, top_k=50)
    config = ModelConfigurationRepository(QueueConnection([row])).get()

    assert config.file_scope == 3
    assert config.top_k == 50


def test_admin_file_scope_accepts_range_and_rejects_other() -> None:
    from app.routers import admin as admin_module

    assert admin_module._normalize_file_scope(1) == 1
    assert admin_module._normalize_file_scope(5) == 5
    assert admin_module._normalize_file_scope(100) == 100
    assert admin_module._normalize_file_scope(None) is None

    with pytest.raises(ValueError, match="between 1 and 100"):
        admin_module._normalize_file_scope(0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        admin_module._normalize_file_scope(-3)
    with pytest.raises(ValueError, match="between 1 and 100"):
        admin_module._normalize_file_scope(101)
    with pytest.raises(ValueError, match="must be an integer"):
        admin_module._normalize_file_scope("5")
    with pytest.raises(ValueError, match="must be an integer"):
        admin_module._normalize_file_scope(True)


@pytest.mark.parametrize("value", [1, 20, 50, 100, 200, 500])
def test_admin_top_k_accepts_range(value: int) -> None:
    from app.routers import admin as admin_module

    assert admin_module._normalize_top_k(value) == value


def test_admin_top_k_rejects_out_of_range() -> None:
    import pytest

    from app.routers import admin as admin_module

    assert admin_module._normalize_top_k(None) is None
    with pytest.raises(ValueError, match="between 1 and 500"):
        admin_module._normalize_top_k(0)
    with pytest.raises(ValueError, match="between 1 and 500"):
        admin_module._normalize_top_k(-1)
    with pytest.raises(ValueError, match="between 1 and 500"):
        admin_module._normalize_top_k(501)
    with pytest.raises(ValueError, match="must be an integer"):
        admin_module._normalize_top_k(2.5)
    with pytest.raises(ValueError, match="must be an integer"):
        admin_module._normalize_top_k("20")


def test_admin_put_persists_rag_settings(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    client = admin_client({"id": 8, "is_admin": True})
    payload = admin_update_payload()
    payload["file_scope"] = 3
    payload["top_k"] = 10

    response = client.put("/api/admin/ai-models", json=payload)

    assert response.status_code == 200
    assert response.json()["file_scope"] == 3
    assert response.json()["top_k"] == 10
    write = StubRepository.replace_calls[-1][0]
    assert write.file_scope == 3
    assert write.top_k == 10


def test_admin_put_without_rag_settings_preserves_stored_values(monkeypatch) -> None:
    import dataclasses

    configure_admin_route_fakes(monkeypatch)
    monkeypatch.setattr(
        StubRepository,
        "current",
        dataclasses.replace(StubRepository.current, file_scope=7, top_k=42),
    )
    client = admin_client({"id": 8, "is_admin": True})

    response = client.put("/api/admin/ai-models", json=admin_update_payload())

    assert response.status_code == 200
    write = StubRepository.replace_calls[-1][0]
    assert write.file_scope == 7
    assert write.top_k == 42


def test_admin_rag_settings_reject_non_admin(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    client = admin_client({"id": 4, "is_admin": False})
    payload = admin_update_payload()
    payload["file_scope"] = 3
    payload["top_k"] = 10

    response = client.put("/api/admin/ai-models", json=payload)

    assert response.status_code == 403


def test_search_depth_defaults_when_row_predates_column() -> None:
    config = ModelConfigurationRepository(QueueConnection([stored_config_row()])).get()

    assert config.search_depth == "conservative"


def test_search_depth_round_trips_stored_value() -> None:
    row = stored_config_row(search_depth="deep")
    config = ModelConfigurationRepository(QueueConnection([row])).get()

    assert config.search_depth == "deep"


def test_search_depth_sanitizes_unknown_stored_value() -> None:
    row = stored_config_row(search_depth="ultra")
    config = ModelConfigurationRepository(QueueConnection([row])).get()

    assert config.search_depth == "conservative"


def test_admin_search_depth_accepts_tiers_and_rejects_other() -> None:
    from app.routers import admin as admin_module

    assert admin_module._normalize_search_depth("conservative") == "conservative"
    assert admin_module._normalize_search_depth("deep") == "deep"
    assert admin_module._normalize_search_depth("pro") == "pro"
    assert admin_module._normalize_search_depth(" Balanced ") == "balanced"
    assert admin_module._normalize_search_depth(None) is None

    with pytest.raises(ValueError, match="search_depth must be one of"):
        admin_module._normalize_search_depth("ultra")
    with pytest.raises(ValueError, match="search_depth must be one of"):
        admin_module._normalize_search_depth("")


@pytest.mark.parametrize("tier", ["conservative", "balanced", "deep", "pro"])
def test_admin_put_persists_search_depth(monkeypatch, tier: str) -> None:
    configure_admin_route_fakes(monkeypatch)
    client = admin_client({"id": 8, "is_admin": True})
    payload = admin_update_payload()
    payload["search_depth"] = tier

    response = client.put("/api/admin/ai-models", json=payload)

    assert response.status_code == 200
    assert response.json()["search_depth"] == tier
    assert response.json()["search_depth_options"] == ["conservative", "balanced", "deep", "pro"]
    write = StubRepository.replace_calls[-1][0]
    assert write.search_depth == tier


def test_admin_put_without_search_depth_preserves_stored_value(monkeypatch) -> None:
    import dataclasses

    configure_admin_route_fakes(monkeypatch)
    monkeypatch.setattr(
        StubRepository,
        "current",
        dataclasses.replace(StubRepository.current, search_depth="deep"),
    )
    client = admin_client({"id": 8, "is_admin": True})

    # Old/cached frontend omits search_depth entirely: an unrelated save
    # must NOT reset the stored tier.
    response = client.put("/api/admin/ai-models", json=admin_update_payload())

    assert response.status_code == 200
    write = StubRepository.replace_calls[-1][0]
    assert write.search_depth == "deep"
    assert response.json()["search_depth"] == "deep"


def test_admin_get_reports_stored_search_depth(monkeypatch) -> None:
    import dataclasses

    configure_admin_route_fakes(monkeypatch)
    monkeypatch.setattr(
        StubRepository,
        "current",
        dataclasses.replace(StubRepository.current, search_depth="balanced"),
    )
    client = admin_client({"id": 8, "is_admin": True})

    response = client.get("/api/admin/ai-models")

    assert response.status_code == 200
    assert response.json()["search_depth"] == "balanced"
    assert response.json()["search_depth_options"] == ["conservative", "balanced", "deep", "pro"]


def test_admin_put_rejects_unknown_search_depth(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    client = admin_client({"id": 8, "is_admin": True})
    payload = admin_update_payload()
    payload["search_depth"] = "ultra"

    response = client.put("/api/admin/ai-models", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("search_depth must be one of" in error.get("msg", "") for error in detail)


# ── ADMIN PREFERENCES (session timeout + registration toggle) ───────────────


def preferences_row(**overrides: Any) -> dict[str, Any]:
    row = stored_config_row()
    row["session_timeout_minutes"] = None
    row["registration_enabled"] = None
    row.update(overrides)
    return row


def test_preferences_resolve_saved_values_and_fall_back(monkeypatch) -> None:
    from app.services import model_config

    model_config.invalidate_preferences_cache()

    saved = ModelConfigurationRepository(
        QueueConnection([preferences_row(session_timeout_minutes=30, registration_enabled=True)])
    ).get()
    assert saved.session_timeout_minutes == 30
    assert saved.registration_enabled is True

    unset = ModelConfigurationRepository(QueueConnection([preferences_row()])).get()
    assert unset.session_timeout_minutes is None
    assert unset.registration_enabled is None

    dirty = ModelConfigurationRepository(
        QueueConnection([preferences_row(session_timeout_minutes=999, registration_enabled="bogus")])
    ).get()
    assert dirty.session_timeout_minutes is None
    assert dirty.registration_enabled is None


def test_preferences_cache_serves_ttl_and_invalidate_resets(monkeypatch) -> None:
    import app.database
    from app.services import model_config

    calls: list[str] = []

    @contextmanager
    def fake_db() -> Iterator[object]:
        calls.append("db")
        yield QueueConnection(
            [preferences_row(session_timeout_minutes=60, registration_enabled=False)]
        )

    monkeypatch.setattr(app.database, "get_db", fake_db)
    monkeypatch.setattr(
        model_config, "_read_saved_preferences", _real_read_saved_preferences
    )
    model_config.invalidate_preferences_cache()

    assert model_config.get_saved_session_timeout_minutes() == 60
    assert model_config.get_saved_registration_enabled() is False
    assert calls == ["db"]

    # Within TTL no second round trip happens.
    assert model_config.get_saved_session_timeout_minutes() == 60
    assert calls == ["db"]

    model_config.invalidate_preferences_cache()
    assert model_config.get_saved_session_timeout_minutes() == 60
    assert calls == ["db", "db"]

    # Expired entries re-read (multi-worker visibility bound).
    model_config._preferences_cache["expires_at"] = 0.0
    assert model_config.get_saved_registration_enabled() is False
    assert calls == ["db", "db", "db"]


def test_preferences_fall_back_when_database_is_unavailable(monkeypatch) -> None:
    import app.database
    from app.config import settings
    from app.services import model_config

    @contextmanager
    def broken_db() -> Iterator[object]:
        raise RuntimeError("database down")

    monkeypatch.setattr(app.database, "get_db", broken_db)
    monkeypatch.setattr(
        model_config, "_read_saved_preferences", _real_read_saved_preferences
    )
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("REGISTRATION_ENABLED", raising=False)
    monkeypatch.delenv("SESSION_IDLE_TIMEOUT_MINUTES", raising=False)
    monkeypatch.delenv("ACCESS_TOKEN_EXPIRE_MINUTES", raising=False)
    model_config.invalidate_preferences_cache()

    assert model_config.get_saved_session_timeout_minutes() is None
    assert model_config.get_saved_registration_enabled() is None
    assert settings.session_idle_timeout_minutes == 120
    assert settings.access_token_expire_minutes == 120
    assert settings.registration_enabled is False


def test_admin_preferences_is_admin_only(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)

    denied_get = admin_client({"id": 4, "is_admin": False}).get("/api/admin/preferences")
    denied_put = admin_client({"id": 4, "is_admin": False}).put(
        "/api/admin/preferences",
        json={"session_timeout_minutes": 30, "expected_revision": 4},
    )
    assert denied_get.status_code == 403
    assert denied_put.status_code == 403


def test_admin_preferences_reports_effective_values_and_saved_flags(monkeypatch) -> None:
    from dataclasses import replace

    from app.services import model_config

    configure_admin_route_fakes(monkeypatch)
    # The stubbed repository stands in for the committed row.
    monkeypatch.setattr(
        StubRepository,
        "current",
        replace(
            StubRepository.current,
            session_timeout_minutes=30,
            registration_enabled=True,
        ),
    )

    # Effective values resolve through settings: serve saved values directly.
    monkeypatch.setattr(model_config, "_read_saved_preferences", lambda: (30, True))

    response = admin_client({"id": 8, "is_admin": True}).get("/api/admin/preferences")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_timeout_minutes"] == 30
    assert payload["session_timeout_saved"] is True
    assert payload["registration_enabled"] is True
    assert payload["registration_saved"] is True
    assert payload["expected_revision"] == 4


def test_admin_preferences_saves_timeout_and_toggle(monkeypatch) -> None:
    from app.services import model_config

    configure_admin_route_fakes(monkeypatch)
    StubRepository.replace_calls = []

    # Effective values in the response resolve through settings: serve the
    # just-saved values as the committed database would on the next read.
    monkeypatch.setattr(model_config, "_read_saved_preferences", lambda: (30, False))
    client = admin_client({"id": 8, "is_admin": True})

    response = client.put(
        "/api/admin/preferences",
        json={
            "session_timeout_minutes": 30,
            "registration_enabled": False,
            "expected_revision": 4,
        },
    )

    assert response.status_code == 200
    assert len(StubRepository.replace_calls) == 1
    write, expected_revision, updated_by = StubRepository.replace_calls[0]
    assert write.session_timeout_minutes == 30
    assert write.registration_enabled is False
    # Model fields ride through untouched.
    assert write.chat.default_model == "chat:latest"
    assert expected_revision == 4
    assert updated_by == 8
    payload = response.json()
    assert payload["session_timeout_minutes"] == 30
    assert payload["registration_enabled"] is False


def test_admin_preferences_rejects_invalid_values(monkeypatch) -> None:
    configure_admin_route_fakes(monkeypatch)
    StubRepository.replace_calls = []
    client = admin_client({"id": 8, "is_admin": True})

    assert (
        client.put(
            "/api/admin/preferences",
            json={"session_timeout_minutes": 999, "expected_revision": 4},
        ).status_code
        == 422
    )
    assert (
        client.put(
            "/api/admin/preferences",
            json={"session_timeout_minutes": 30, "registration_enabled": "yes", "expected_revision": 4},
        ).status_code
        == 422
    )
    assert (
        client.put("/api/admin/preferences", json={"expected_revision": 4}).status_code == 422
    )
    assert StubRepository.replace_calls == []


def test_registration_toggle_blocks_and_allows_register(monkeypatch) -> None:
    from app.routers import auth as auth_routes
    from app.services import model_config

    def register_client() -> TestClient:
        app = FastAPI()
        app.include_router(auth_routes.router, prefix="/api/auth")
        return TestClient(app)

    monotone_body = {
        "username": "toggle-user",
        "email": "toggle@example.com",
        "password": "strong-password",
    }

    # Toggle off: 403 before any database user lookup happens.
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("REGISTRATION_ENABLED", raising=False)
    monkeypatch.setattr(model_config, "_read_saved_preferences", lambda: (None, False))
    closed = register_client().post("/api/auth/register", json=monotone_body)
    assert closed.status_code == 403

    # Toggle on: registration proceeds (dev env keeps the rate limiter open).
    monkeypatch.setenv("APP_ENV", "dev")
    open_connection = QueueConnection([None, None, {"id": 7}])

    @contextmanager
    def open_db() -> Iterator[object]:
        yield open_connection

    monkeypatch.setattr(auth_routes, "get_db", open_db)
    monkeypatch.setattr(model_config, "_read_saved_preferences", lambda: (None, True))
    created = register_client().post("/api/auth/register", json=monotone_body)
    assert created.status_code == 200
    assert created.json()["user_id"] == 7


def test_saved_timeout_is_honored_by_new_session_lifetimes(monkeypatch) -> None:
    from app.config import settings
    from app.services import model_config

    monkeypatch.setattr(model_config, "_read_saved_preferences", lambda: (30, True))

    assert settings.session_idle_timeout_minutes == 30
    assert settings.access_token_expire_minutes == 30
