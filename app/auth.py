"""Password, JWT, and authenticated-session helpers."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from uuid import UUID, uuid4

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import settings
from app.database import get_db

security = HTTPBearer()
BearerCredentials = Annotated[HTTPAuthorizationCredentials, Depends(security)]
_REFRESH_HASH_PREFIX = "hmac-sha256:"
BCRYPT_PASSWORD_MAX_BYTES = 72


@dataclass(frozen=True, slots=True)
class SessionActivity:
    """Server-clock view of one authenticated browser session."""

    last_activity_at: datetime
    server_now: datetime
    idle_expires_at: datetime

    @property
    def is_idle(self) -> bool:
        return self.server_now >= self.idle_expires_at


def session_idle_timeout_seconds() -> int:
    return settings.session_idle_timeout_minutes * 60


def session_idle_deadline(last_activity_at: datetime) -> datetime:
    return last_activity_at + timedelta(minutes=settings.session_idle_timeout_minutes)


def session_idle_timeout_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "code": "session_idle_timeout",
            "message": (
                f"Session locked after {settings.session_idle_timeout_minutes} minutes of inactivity"
            ),
            "idle_timeout_seconds": session_idle_timeout_seconds(),
        },
    )


def get_session_activity(cursor: Any, user_id: int, session_id: str) -> SessionActivity | None:
    """Read, but never extend, the current idle deadline for a live session."""

    cursor.execute(
        """
        SELECT MAX(last_activity_at) AS last_activity_at, NOW() AS server_now
        FROM refresh_tokens
        WHERE user_id = %s
          AND session_id = %s
          AND is_revoked = FALSE
          AND expires_at > NOW()
        """,
        (user_id, session_id),
    )
    row = cursor.fetchone()
    if not row or row["last_activity_at"] is None:
        return None
    last_activity_at = row["last_activity_at"]
    return SessionActivity(
        last_activity_at=last_activity_at,
        server_now=row["server_now"],
        idle_expires_at=session_idle_deadline(last_activity_at),
    )


def revoke_idle_session(cursor: Any, user_id: int, session_id: str) -> None:
    """Revoke exactly one timed-out session, leaving the user's other sessions intact."""

    cursor.execute(
        """
        UPDATE refresh_tokens
        SET is_revoked = TRUE
        WHERE user_id = %s AND session_id = %s AND is_revoked = FALSE
        """,
        (user_id, session_id),
    )


def validate_bcrypt_password(password: str) -> str:
    """Reject values bcrypt would silently truncate after 72 UTF-8 bytes."""

    if len(password.encode("utf-8")) > BCRYPT_PASSWORD_MAX_BYTES:
        raise ValueError("Password must not exceed 72 UTF-8 bytes")
    return password


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        password_bytes = validate_bcrypt_password(plain_password).encode("utf-8")
        return bcrypt.checkpw(password_bytes, hashed_password.encode("ascii"))
    except (TypeError, ValueError):
        return False


def get_password_hash(password: str) -> str:
    password_bytes = validate_bcrypt_password(password).encode("utf-8")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("ascii")


def hash_refresh_token(token: str) -> str:
    """Hash a signed, high-entropy refresh JWT without bcrypt truncation."""

    digest = hmac.new(
        settings.secret_key.encode("utf-8"),
        token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{_REFRESH_HASH_PREFIX}{digest}"


def verify_refresh_token(token: str, stored_hash: str) -> bool:
    """Verify current HMAC hashes and allow one-time rotation of legacy bcrypt rows."""

    if stored_hash.startswith(_REFRESH_HASH_PREFIX):
        return hmac.compare_digest(hash_refresh_token(token), stored_hash)
    try:
        # Legacy refresh rows used bcrypt over a JWT longer than 72 bytes. Keep
        # their historical truncated verification only for one-time rotation;
        # user passwords always go through the strict helper above.
        legacy_token = token.encode("utf-8")[:BCRYPT_PASSWORD_MAX_BYTES]
        return bcrypt.checkpw(legacy_token, stored_hash.encode("ascii"))
    except (TypeError, ValueError):
        return False


def create_access_token(
    user_id: int,
    session_id: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a short-lived access token bound to a server-side session."""
    expires_at = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload = {
        "sub": str(user_id),
        "sid": session_id,
        "type": "access",
        "exp": expires_at,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def _store_refresh_token(
    connection: Any,
    *,
    user_id: int,
    session_id: str,
    token_hash: str,
    expires_at: datetime,
    last_activity_at: datetime | None,
) -> None:
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT session_id
        FROM refresh_tokens
        WHERE user_id = %s AND is_revoked = FALSE AND expires_at > NOW()
        GROUP BY session_id
        ORDER BY MIN(created_at) ASC
        """,
        (user_id,),
    )
    active_sessions = cursor.fetchall()
    known_session_ids = {str(row["session_id"]) for row in active_sessions}
    if len(active_sessions) >= settings.max_active_sessions_per_user and session_id not in known_session_ids:
        cursor.execute(
            "UPDATE refresh_tokens SET is_revoked = TRUE WHERE session_id = %s AND user_id = %s",
            (active_sessions[0]["session_id"], user_id),
        )

    cursor.execute(
        """
        INSERT INTO refresh_tokens (
            user_id, session_id, token_hash, expires_at, last_activity_at
        )
        VALUES (%s, %s, %s, %s, COALESCE(%s, NOW()))
        """,
        (user_id, session_id, token_hash, expires_at, last_activity_at),
    )


def create_refresh_token(
    user_id: int,
    session_id: str,
    *,
    connection: Any | None = None,
    last_activity_at: datetime | None = None,
) -> str:
    """Create and persist a long-lived refresh token."""
    expires_at = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": str(user_id),
        "sid": session_id,
        "jti": str(uuid4()),
        "type": "refresh",
        "exp": expires_at,
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
    token_hash = hash_refresh_token(token)

    if connection is not None:
        _store_refresh_token(
            connection,
            user_id=user_id,
            session_id=session_id,
            token_hash=token_hash,
            expires_at=expires_at,
            last_activity_at=last_activity_at,
        )
    else:
        with get_db() as owned_connection:
            _store_refresh_token(
                owned_connection,
                user_id=user_id,
                session_id=session_id,
                token_hash=token_hash,
                expires_at=expires_at,
                last_activity_at=last_activity_at,
            )
    return token


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
            options={"verify_exp": True},
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid",
        ) from None


def _token_identity(payload: dict[str, Any], *, expected_type: str) -> tuple[int, str]:
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=401, detail="Invalid token type")
    subject = payload.get("sub")
    session_id = payload.get("sid")
    try:
        user_id = int(subject)
        session_id = str(UUID(str(session_id)))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token claims") from None
    return user_id, session_id


def get_current_user(credentials: BearerCredentials) -> dict[str, Any]:
    payload = decode_token(credentials.credentials)
    user_id, session_id = _token_identity(payload, expected_type="access")

    session_state = "active"
    user = None
    with get_db() as connection:
        cursor = connection.cursor()
        activity = get_session_activity(cursor, user_id, session_id)
        if activity is None:
            session_state = "revoked"
        elif activity.is_idle:
            revoke_idle_session(cursor, user_id, session_id)
            session_state = "idle"
        else:
            cursor.execute(
                """
                SELECT id, username, email, is_active, persona_prompt, is_admin,
                       perm_upload, perm_download, perm_delete, perm_ai, perm_share,
                       perm_folders, perm_rename, avatar_data, memory_enabled
                FROM users
                WHERE id = %s
                """,
                (user_id,),
            )
            user = cursor.fetchone()

    if session_state == "idle":
        raise session_idle_timeout_exception()
    if session_state == "revoked":
        raise HTTPException(status_code=401, detail="Session revoked or logged out")

    if not user or not user["is_active"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user


def require_ai_permission(
    user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    """Fail closed when the authenticated user is not allowed to use AI."""

    if user.get("perm_ai") is not True:
        raise HTTPException(status_code=403, detail="AI access permission denied")
    return user


def authenticate_user(
    username: str,
    password: str,
    client_ip: str = "unknown",
) -> dict[str, Any] | None:
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, username, email, password_hash, is_active
            FROM users
            WHERE username = %s OR email = %s
            """,
            (username, username),
        )
        user = cursor.fetchone()
        if not user or not verify_password(password, user["password_hash"]):
            return None

        cursor.execute(
            "UPDATE users SET last_login = NOW(), last_login_ip = %s WHERE id = %s",
            (client_ip, user["id"]),
        )
    return user
