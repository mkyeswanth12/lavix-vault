from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import database


@pytest.fixture
def db_settings(monkeypatch):
    value = SimpleNamespace(
        db_host="postgres",
        db_port=5432,
        db_name="vault",
        db_user="vault",
        db_pass="do-not-leak",
        secret_key="unit-test-secret",
    )
    monkeypatch.setattr(database, "settings", value)
    database._ENC_KEY = None
    return value


@pytest.fixture(autouse=True)
def _reset_pool():
    """Ensure each test starts with no pool."""
    database._pool = None
    yield
    database._pool = None


def test_connection_uses_bounded_timeout_and_dict_rows(monkeypatch, db_settings) -> None:
    sentinel = object()
    created_pool = MagicMock()
    created_pool.getconn.return_value = sentinel

    with patch.object(database, "ConnectionPool") as MockPool:
        MockPool.return_value = created_pool
        result = database.get_connection()
        assert result is sentinel
        pool_kwargs = MockPool.call_args
        assert pool_kwargs.kwargs["kwargs"]["connect_timeout"] == 2
        assert pool_kwargs.kwargs["kwargs"]["row_factory"] is database.dict_row


def test_connection_error_hides_password(monkeypatch, db_settings) -> None:
    created_pool = MagicMock()
    created_pool.getconn.side_effect = RuntimeError("driver included secret")

    with patch.object(database, "ConnectionPool") as MockPool:
        MockPool.return_value = created_pool
        with pytest.raises(database.DatabaseUnavailableError) as caught:
            database.get_connection()
        assert db_settings.db_pass not in str(caught.value)
        assert isinstance(caught.value.__cause__, RuntimeError)


class FakeConnection:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def test_get_db_commits_or_rolls_back_and_always_closes(monkeypatch) -> None:
    success = FakeConnection()
    monkeypatch.setattr(database, "get_connection", lambda: success)
    monkeypatch.setattr(database, "put_connection", lambda conn: conn.close())
    with database.get_db() as connection:
        assert connection is success
    assert (success.commits, success.rollbacks, success.closed) == (1, 0, True)

    failure = FakeConnection()
    monkeypatch.setattr(database, "get_connection", lambda: failure)
    monkeypatch.setattr(database, "put_connection", lambda conn: conn.close())
    with pytest.raises(ValueError), database.get_db():
        raise ValueError("abort")
    assert (failure.commits, failure.rollbacks, failure.closed) == (0, 1, True)


def test_secret_round_trip_and_tamper_failure(db_settings) -> None:
    ciphertext = database.encrypt_secret("archived-provider-key")
    assert ciphertext != "archived-provider-key"
    assert database.decrypt_secret(ciphertext) == "archived-provider-key"
    assert database.decrypt_secret(ciphertext[:-3] + "abc") == ""
