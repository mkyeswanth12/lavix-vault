from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.db.readiness import SchemaHeadStatus, SchemaNotReadyError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeConnection:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.closed = False

    def close(self) -> None:
        self.events.append("close")
        self.closed = True

    def cursor(self):
        return FakeCursor(self.events)


class FakeCursor:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def execute(self, *args, **kwargs):
        return None

    def fetchone(self):
        return None

    def close(self) -> None:
        return None


def load_main(monkeypatch, *, connection: FakeConnection, schema_check):
    settings = SimpleNamespace(
        app_name="Lavix Test API",
        app_version="0.9.49-beta",
        allowed_origins=["https://ui.test"],
        ollama_base_url="http://ollama:11434",
        openrouter_api_key="",
        openrouter_base_url="https://openrouter.test/api/v1",
        openrouter_model="open/free",
        embedding_model_name="snowflake-arctic-embed2:cpu",
        embedding_dimension=1024,
        embedding_api_url="http://ollama:11434/v1/embeddings",
        validate_runtime_secrets=lambda: connection.events.append("secrets"),
        validate_config_file=lambda: connection.events.append("config"),
    )
    config_module = types.ModuleType("app.config")
    config_module.settings = settings

    database_module = types.ModuleType("app.database")

    class DatabaseUnavailableError(RuntimeError):
        pass

    def get_connection():
        connection.events.append("connect")
        return connection

    from contextlib import contextmanager

    @contextmanager
    def get_db():
        connection.events.append("get_db")
        yield connection

    database_module.DatabaseUnavailableError = DatabaseUnavailableError
    database_module.get_connection = get_connection
    database_module.get_db = get_db
    database_module.close_pool = lambda: None
    database_module.put_connection = lambda conn: conn.close() if hasattr(conn, 'close') else None
    database_module._pool = None

    routers_module = types.ModuleType("app.routers")
    routers_module.__path__ = []
    for router_name in ("admin", "ai_endpoints", "auth", "file_upload", "files", "folders"):
        router_module = types.ModuleType(f"app.routers.{router_name}")
        router_module.router = APIRouter()
        setattr(routers_module, router_name, router_module)
        monkeypatch.setitem(sys.modules, f"app.routers.{router_name}", router_module)

    agent_module = types.ModuleType("app.agent")
    agent_module.__path__ = []
    agent_gateway_module = types.ModuleType("app.agent.gateway")
    agent_gateway_module.router = APIRouter()

    model_service_module = types.ModuleType("app.services.model_service")
    model_service_module.calls = []

    def init_model_service(*args, **kwargs):
        connection.events.append("model")
        model_service_module.calls.append((args, kwargs))

    model_service_module.init_model_service = init_model_service
    services_module = types.ModuleType("app.services")
    services_module.__path__ = []

    psutil_module = types.ModuleType("psutil")
    psutil_module.virtual_memory = lambda: SimpleNamespace(available=1, total=1)

    import app.db.readiness as readiness

    monkeypatch.setattr(readiness, "require_schema_head", schema_check)
    monkeypatch.setitem(sys.modules, "app.config", config_module)
    monkeypatch.setitem(sys.modules, "app.database", database_module)
    monkeypatch.setitem(sys.modules, "app.routers", routers_module)
    monkeypatch.setitem(sys.modules, "app.agent", agent_module)
    monkeypatch.setitem(sys.modules, "app.agent.gateway", agent_gateway_module)
    monkeypatch.setitem(sys.modules, "app.services", services_module)
    monkeypatch.setitem(sys.modules, "app.services.model_service", model_service_module)
    monkeypatch.setitem(sys.modules, "psutil", psutil_module)

    module_name = f"test_main_startup_{id(connection)}"
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / "app/main.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    module._test_model_service = model_service_module
    return module


def test_startup_checks_schema_before_initializing_services_and_closes_connection(monkeypatch):
    events: list[str] = []
    connection = FakeConnection(events)

    def schema_check(candidate):
        assert candidate is connection
        events.append("schema")
        return SchemaHeadStatus(True, "ready", 1, 1)

    main = load_main(monkeypatch, connection=connection, schema_check=schema_check)

    asyncio.run(main.startup_event())

    assert events == ["config", "secrets", "connect", "schema", "close", "model", "get_db"]
    assert connection.closed is True
    assert main._test_model_service.calls == [(("http://ollama:11434",), {})]


def test_startup_fails_before_service_initialization_when_schema_is_stale(monkeypatch):
    events: list[str] = []
    connection = FakeConnection(events)
    stale = SchemaHeadStatus(False, "pending_migrations", 0, 1, (1,))

    def schema_check(candidate):
        assert candidate is connection
        events.append("schema")
        raise SchemaNotReadyError(stale)

    main = load_main(monkeypatch, connection=connection, schema_check=schema_check)

    with pytest.raises(SchemaNotReadyError) as exc_info:
        asyncio.run(main.startup_event())

    assert exc_info.value.status is stale
    assert events == ["config", "secrets", "connect", "schema", "close"]
    assert connection.closed is True
    assert main._test_model_service.calls == []


def test_api_startup_contains_no_schema_or_data_mutations():
    source = (PROJECT_ROOT / "app/main.py").read_text(encoding="utf-8").upper()

    for mutation in (
        "CREATE TABLE",
        "CREATE INDEX",
        "ALTER TABLE",
        "DROP CONSTRAINT",
        "INSERT INTO",
        "UPDATE USERS",
        ".COMMIT(",
    ):
        assert mutation not in source


def test_validation_response_does_not_echo_sensitive_input(monkeypatch):
    connection = FakeConnection([])
    main = load_main(
        monkeypatch,
        connection=connection,
        schema_check=lambda _connection: SchemaHeadStatus(True, "ready", 1, 1),
    )
    error = RequestValidationError(
        [
            {
                "type": "string_too_short",
                "loc": ("body", "password"),
                "msg": "String should have at least 8 characters",
                "input": "secret-password-value",
                "ctx": {"min_length": 8},
            }
        ]
    )
    response = asyncio.run(main.validation_exception_handler(object(), error))
    payload = json.loads(response.body)

    assert response.status_code == 422
    assert payload["detail"][0]["location"] == ["body", "password"]
    assert "secret-password-value" not in response.body.decode()


def test_http_error_response_keeps_the_standard_mirrored_envelope(monkeypatch):
    connection = FakeConnection([])
    main = load_main(
        monkeypatch,
        connection=connection,
        schema_check=lambda _connection: SchemaHeadStatus(True, "ready", 1, 1),
    )
    request = SimpleNamespace(method="GET", url=SimpleNamespace(path="/api/auth/me"))
    detail = {
        "code": "session_idle_timeout",
        "message": "Session locked after 120 minutes of inactivity",
        "idle_timeout_seconds": 7_200,
    }

    response = asyncio.run(
        main.http_exception_handler(
            request,
            StarletteHTTPException(status_code=401, detail=detail),
        )
    )

    assert response.status_code == 401
    assert json.loads(response.body) == {"detail": detail, "error": detail, "code": 401}


def test_database_unavailable_response_is_retryable_and_sanitized(monkeypatch):
    connection = FakeConnection([])
    main = load_main(
        monkeypatch,
        connection=connection,
        schema_check=lambda _connection: SchemaHeadStatus(True, "ready", 1, 1),
    )
    request = SimpleNamespace(method="GET", url=SimpleNamespace(path="/api/files"))
    response = asyncio.run(
        main.database_unavailable_handler(
            request,
            main.DatabaseUnavailableError("private database diagnostic"),
        )
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert json.loads(response.body) == {
        "detail": "Service temporarily unavailable",
        "error": "Service temporarily unavailable",
        "code": 503,
    }
    assert "private" not in response.body.decode()
