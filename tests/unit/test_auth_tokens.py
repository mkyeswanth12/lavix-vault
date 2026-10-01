from __future__ import annotations

from app import auth


def test_refresh_hash_uses_the_entire_token_and_constant_format(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "unit-test-refresh-hash-secret")
    shared_prefix = "x" * 96
    first = f"{shared_prefix}-first"
    second = f"{shared_prefix}-second"

    first_hash = auth.hash_refresh_token(first)
    second_hash = auth.hash_refresh_token(second)

    assert first_hash.startswith("hmac-sha256:")
    assert len(first_hash) == len("hmac-sha256:") + 64
    assert first_hash != second_hash
    assert auth.verify_refresh_token(first, first_hash)
    assert not auth.verify_refresh_token(second, first_hash)


def test_created_refresh_token_is_stored_with_hmac_hash(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "unit-test-refresh-storage-secret")
    captured = {}

    def capture(_connection, **values) -> None:
        captured.update(values)

    monkeypatch.setattr(auth, "_store_refresh_token", capture)
    token = auth.create_refresh_token(
        7,
        "84fd8bdb-fae6-4f14-9657-4bc042c64a2d",
        connection=object(),
    )

    assert captured["user_id"] == 7
    assert captured["session_id"] == "84fd8bdb-fae6-4f14-9657-4bc042c64a2d"
    assert auth.verify_refresh_token(token, captured["token_hash"])
