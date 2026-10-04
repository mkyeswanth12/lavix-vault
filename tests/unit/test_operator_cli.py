"""Unit tests for the promote-admin operator CLI (fake connections only)."""

from __future__ import annotations

from app import cli


class FakeCursor:
    def __init__(self, row):
        self.row = row
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.commit()
        return None


def _patch_db(monkeypatch, row):
    conn = FakeConnection(FakeCursor(row))
    monkeypatch.setattr(cli, "get_db", lambda: conn)
    return conn


def test_promote_admin_updates_one_user(monkeypatch, capsys):
    conn = _patch_db(
        monkeypatch,
        {"id": 7, "username": "owner", "is_admin": False, "is_active": True, "perm_delete": False},
    )
    assert cli.main(["promote-admin", "owner"]) == 0
    assert conn.committed
    assert any(
        "UPDATE users SET is_admin = TRUE, perm_delete = TRUE" in sql
        for sql, _ in conn._cursor.statements
    )
    out = capsys.readouterr().out
    assert "promoted" in out
    assert "delete" in out


def test_promote_admin_unknown_user_fails(monkeypatch, capsys):
    _patch_db(monkeypatch, None)
    assert cli.main(["promote-admin", "ghost"]) == 1
    assert "no such user" in capsys.readouterr().err


def test_promote_admin_already_admin_is_noop(monkeypatch, capsys):
    conn = _patch_db(
        monkeypatch,
        {"id": 7, "username": "owner", "is_admin": True, "is_active": True, "perm_delete": True},
    )
    assert cli.main(["promote-admin", "owner"]) == 0
    assert not any("UPDATE" in sql for sql, _ in conn._cursor.statements)
    assert "already an admin" in capsys.readouterr().out


def test_promote_admin_already_admin_backfills_missing_delete(monkeypatch, capsys):
    conn = _patch_db(
        monkeypatch,
        {"id": 7, "username": "owner", "is_admin": True, "is_active": True, "perm_delete": False},
    )
    assert cli.main(["promote-admin", "owner"]) == 0
    assert any(
        "UPDATE users SET perm_delete = TRUE" in sql for sql, _ in conn._cursor.statements
    )
    assert not any("is_admin = TRUE" in sql for sql, _ in conn._cursor.statements)
    out = capsys.readouterr().out
    assert "already an admin" in out
    assert "delete permission granted" in out


def test_promote_admin_rejects_empty_username(monkeypatch):
    _patch_db(monkeypatch, None)
    assert cli.main(["promote-admin", "   "]) == 2


def test_promote_admin_rejects_deactivated_user(monkeypatch, capsys):
    _patch_db(
        monkeypatch,
        {"id": 9, "username": "old", "is_admin": False, "is_active": False, "perm_delete": False},
    )
    assert cli.main(["promote-admin", "old"]) == 1
    assert "deactivated" in capsys.readouterr().err
