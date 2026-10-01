"""Small HMAC capabilities for read-only agent tool calls.

The CUGA process receives no database or object-storage credentials.  The API
mints one short-lived token per run, and remains the authority for user, file,
and web-search scope when the runtime calls back into its tool gateway.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any
from uuid import UUID


class CapabilityError(ValueError):
    """A capability is malformed, expired, forged, or outside its run scope."""


@dataclass(frozen=True, slots=True)
class CapabilityScope:
    run_id: str
    user_id: int
    file_ids: tuple[int, ...] | None
    web_search_enabled: bool
    deep_search: bool
    issued_at: int
    expires_at: int
    chat_mode: str | None = None


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise CapabilityError("invalid capability encoding") from exc


def _file_ids(values: Iterable[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    result: list[int] = []
    for raw in values:
        if isinstance(raw, bool):
            raise CapabilityError("invalid file scope")
        value = int(raw)
        if value <= 0:
            raise CapabilityError("invalid file scope")
        if value not in result:
            result.append(value)
    if len(result) > 50:
        raise CapabilityError("file scope is too large")
    return tuple(result)


def _chat_mode(value: Any) -> str | None:
    if value is None:
        return None
    if value in ("casual", "expert"):
        return value
    raise CapabilityError("invalid chat mode")


class CapabilitySigner:
    """Mint and verify compact, dependency-free, versioned capabilities."""

    def __init__(
        self,
        secret: str | bytes,
        *,
        ttl_seconds: int = 300,
        max_ttl_seconds: int = 300,
        clock: Callable[[], float] = time.time,
    ) -> None:
        key = secret.encode() if isinstance(secret, str) else secret
        if len(key) < 32:
            raise ValueError("agent capability secret must contain at least 32 bytes")
        if not 1 <= ttl_seconds <= max_ttl_seconds <= 900:
            raise ValueError("invalid capability lifetime")
        self._key = key
        self._ttl = ttl_seconds
        self._max_ttl = max_ttl_seconds
        self._clock = clock

    def mint(
        self,
        *,
        run_id: str,
        user_id: int,
        file_ids: Iterable[int] | None,
        web_search_enabled: bool,
        deep_search: bool,
        chat_mode: str | None = None,
    ) -> str:
        normalized_run_id = str(UUID(str(run_id)))
        if isinstance(user_id, bool) or int(user_id) <= 0:
            raise CapabilityError("invalid capability subject")
        now = int(self._clock())
        payload = {
            "v": 1,
            "run_id": normalized_run_id,
            "user_id": int(user_id),
            "file_ids": list(_file_ids(file_ids)) if file_ids is not None else None,
            "web": bool(web_search_enabled),
            "deep": bool(deep_search),
            "mode": _chat_mode(chat_mode),
            "iat": now,
            "exp": now + self._ttl,
        }
        encoded = _encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        signature = _encode(hmac.digest(self._key, encoded.encode("ascii"), hashlib.sha256))
        return f"{encoded}.{signature}"

    def verify(self, token: str, *, expected_run_id: str | None = None) -> CapabilityScope:
        if not token or len(token) > 8_192 or token.count(".") != 1:
            raise CapabilityError("invalid capability")
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = hmac.digest(self._key, encoded.encode("ascii"), hashlib.sha256)
        if not hmac.compare_digest(expected_signature, _decode(supplied_signature)):
            raise CapabilityError("invalid capability signature")
        try:
            payload: Any = json.loads(_decode(encoded))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CapabilityError("invalid capability payload") from exc
        if not isinstance(payload, dict) or payload.get("v") != 1:
            raise CapabilityError("unsupported capability")

        try:
            run_id = str(UUID(payload["run_id"]))
            user_id = payload["user_id"]
            issued_at = payload["iat"]
            expires_at = payload["exp"]
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise CapabilityError("invalid capability claims") from exc
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (user_id, issued_at, expires_at)
        ):
            raise CapabilityError("invalid capability claims")
        if user_id <= 0:
            raise CapabilityError("invalid capability subject")
        if not isinstance(payload.get("web"), bool) or not isinstance(payload.get("deep"), bool):
            raise CapabilityError("invalid capability permissions")

        now = int(self._clock())
        if issued_at > now + 30 or expires_at <= now:
            raise CapabilityError("capability expired or not yet valid")
        if expires_at <= issued_at or expires_at - issued_at > self._max_ttl:
            raise CapabilityError("invalid capability lifetime")
        if expected_run_id is not None and run_id != str(UUID(str(expected_run_id))):
            raise CapabilityError("capability run mismatch")

        raw_ids = payload.get("file_ids")
        if raw_ids is not None and not isinstance(raw_ids, list):
            raise CapabilityError("invalid file scope")
        return CapabilityScope(
            run_id=run_id,
            user_id=user_id,
            file_ids=_file_ids(raw_ids),
            web_search_enabled=payload["web"],
            deep_search=payload["deep"],
            issued_at=issued_at,
            expires_at=expires_at,
            chat_mode=_chat_mode(payload.get("mode")),
        )
