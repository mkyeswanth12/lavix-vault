"""Authentication, profile, session, and provider-settings API."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any
from urllib.parse import urlencode, urlparse

import redis as redis_lib
import requests
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import AfterValidator, BaseModel, EmailStr, Field

from app.auth import (
    BearerCredentials,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    get_password_hash,
    get_session_activity,
    require_ai_permission,
    revoke_idle_session,
    session_idle_deadline,
    session_idle_timeout_exception,
    session_idle_timeout_seconds,
    validate_bcrypt_password,
    verify_password,
    verify_refresh_token,
)
from app.config import settings
from app.database import get_db
from app.services.model_config import (
    ModelConfigurationRepository,
    resolve_account_chat_model,
)
from app.services.model_service import get_model_service

logger = logging.getLogger(__name__)
router = APIRouter()
CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]
CurrentAIUser = Annotated[dict[str, Any], Depends(require_ai_permission)]

_rl_client: redis_lib.Redis | None = None
_USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
_PROVIDERS = frozenset({"", "auto", "ollama", "openrouter"})
BcryptPassword = Annotated[
    str,
    Field(min_length=1, max_length=72),
    AfterValidator(validate_bcrypt_password),
]


def _get_rl_client() -> redis_lib.Redis:
    global _rl_client
    if _rl_client is None:
        _rl_client = redis_lib.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
    return _rl_client


def _rate_limit(request: Request, key_prefix: str, max_hits: int, window: int) -> None:
    """Enforce a fixed-window per-peer limit."""
    ip_address = request.client.host if request.client else "unknown"
    key = f"rl:{key_prefix}:{ip_address}"
    try:
        client = _get_rl_client()
        count = client.incr(key)
        if count == 1:
            client.expire(key, window)
        if count > max_hits:
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests — retry after {window}s.",
                headers={"Retry-After": str(window)},
            )
    except HTTPException:
        raise
    except Exception:
        logger.warning("Authentication rate limiter unavailable", exc_info=True)
        if settings.auth_rate_limit_fail_closed:
            raise HTTPException(status_code=503, detail="Authentication throttling unavailable") from None


def _token_identity(payload: dict[str, Any], expected_type: str) -> tuple[int, str]:
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=401, detail="Invalid token type")
    try:
        return int(payload["sub"]), str(uuid.UUID(str(payload["sid"])))
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token claims") from None


def _unique_username(cursor: Any, base: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9._-]", "_", base).strip("._-") or "user"
    candidate = base[:50]
    suffix = 1
    while True:
        cursor.execute("SELECT 1 FROM users WHERE username = %s", (candidate,))
        if not cursor.fetchone():
            return candidate
        tail = f"_{suffix}"
        candidate = f"{base[: 50 - len(tail)]}{tail}"
        suffix += 1


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    email: EmailStr = Field(max_length=254)
    password: BcryptPassword


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=254)
    password: BcryptPassword


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=4_096)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    session_id: str
    last_activity_at: datetime
    idle_expires_at: datetime
    idle_timeout_seconds: int


class ActivityResponse(BaseModel):
    last_activity_at: datetime
    idle_expires_at: datetime
    idle_timeout_seconds: int


class UsernameUpdateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)


class ChatModelPreferenceRequest(BaseModel):
    model: str | None = Field(default=None, max_length=200)


@router.get("/capabilities")
def authentication_capabilities() -> dict[str, bool]:
    return {
        "registration_enabled": settings.registration_enabled,
        "google_oauth_enabled": bool(
            settings.google_client_id and settings.google_client_secret and _google_redirect_uris()
        ),
    }


@router.post("/register")
def register(req: RegisterRequest, request: Request) -> dict[str, Any]:
    if not settings.registration_enabled:
        raise HTTPException(status_code=403, detail="Account registration is disabled")
    _rate_limit(request, "register", max_hits=3, window=60)
    if len(req.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters",
        )
    username = req.username.strip()
    if not 3 <= len(username) <= 50 or not _USERNAME_PATTERN.fullmatch(username):
        raise HTTPException(status_code=400, detail="Invalid username")

    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")
        cursor.execute("SELECT id FROM users WHERE email = %s", (str(req.email),))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (username, str(req.email), get_password_hash(req.password)),
        )
        user_id = cursor.fetchone()["id"]
    return {"message": "User created successfully", "user_id": user_id}


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, request: Request) -> TokenResponse:
    _rate_limit(request, "login", max_hits=100, window=60)
    client_ip = request.client.host if request.client else "unknown"
    user = authenticate_user(req.username, req.password, client_ip)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account is inactive")

    session_id = str(uuid.uuid4())
    last_activity_at = datetime.now(UTC)
    return TokenResponse(
        access_token=create_access_token(user["id"], session_id),
        refresh_token=create_refresh_token(
            user["id"],
            session_id,
            last_activity_at=last_activity_at,
        ),
        session_id=session_id,
        last_activity_at=last_activity_at,
        idle_expires_at=session_idle_deadline(last_activity_at),
        idle_timeout_seconds=session_idle_timeout_seconds(),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(req: RefreshRequest, request: Request) -> TokenResponse:
    _rate_limit(request, "refresh", max_hits=10, window=60)
    payload = decode_token(req.refresh_token)
    user_id, session_id = _token_identity(payload, "refresh")

    token_reuse_detected = False
    idle_timeout_detected = False
    new_access_token = ""
    new_refresh_token = ""
    last_activity_at: datetime | None = None
    try:
        with get_db() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT id, token_hash, is_revoked, created_at, last_activity_at, NOW() AS server_now
                FROM refresh_tokens
                WHERE user_id = %s AND session_id = %s AND expires_at > NOW()
                ORDER BY created_at DESC, id DESC
                FOR UPDATE
                """,
                (user_id, session_id),
            )
            matching_token = next(
                (
                    row
                    for row in cursor.fetchall()
                    if verify_refresh_token(req.refresh_token, row["token_hash"])
                ),
                None,
            )
            if not matching_token:
                raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
            if matching_token["is_revoked"]:
                # Idle detection first: an idle-revoked session has no
                # successor token by definition.
                revoked_last_activity = matching_token.get("last_activity_at")
                server_now = matching_token.get("server_now")
                was_idle = (
                    revoked_last_activity is not None
                    and server_now is not None
                    and server_now >= session_idle_deadline(revoked_last_activity)
                )
                if was_idle:
                    revoke_idle_session(cursor, user_id, session_id)
                    idle_timeout_detected = True
                else:
                    # A successor token for the same session means another tab
                    # already rotated this exact row (benign concurrent
                    # refresh). Only a replay with NO successor is theft.
                    cursor.execute(
                        """
                        SELECT 1 FROM refresh_tokens
                        WHERE user_id = %s AND session_id = %s
                          AND is_revoked = FALSE AND expires_at > NOW()
                          AND created_at >= %s AND id <> %s
                        LIMIT 1
                        """,
                        (
                            user_id,
                            session_id,
                            matching_token.get("created_at"),
                            matching_token["id"],
                        ),
                    )
                    if cursor.fetchone():
                        raise HTTPException(
                            status_code=401,
                            detail="Refresh token was already rotated",
                        )
                    cursor.execute(
                        "UPDATE refresh_tokens SET is_revoked = TRUE WHERE user_id = %s",
                        (user_id,),
                    )
                    token_reuse_detected = True
            else:
                activity = get_session_activity(cursor, user_id, session_id)
                if activity is None:
                    raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
                if activity.is_idle:
                    revoke_idle_session(cursor, user_id, session_id)
                    idle_timeout_detected = True
                else:
                    last_activity_at = activity.last_activity_at
                    cursor.execute(
                        "UPDATE refresh_tokens SET is_revoked = TRUE WHERE id = %s",
                        (matching_token["id"],),
                    )
                    new_access_token = create_access_token(user_id, session_id)
                    new_refresh_token = create_refresh_token(
                        user_id,
                        session_id,
                        connection=connection,
                        last_activity_at=last_activity_at,
                    )
    except HTTPException:
        raise
    except Exception:
        logger.warning("Refresh token rotation failed", exc_info=True)
        raise HTTPException(status_code=401, detail="Refresh failed") from None

    if token_reuse_detected:
        raise HTTPException(
            status_code=401,
            detail="Security alert: Token reuse detected. All sessions invalidated.",
        )
    if idle_timeout_detected:
        raise session_idle_timeout_exception()
    if last_activity_at is None:
        raise HTTPException(status_code=401, detail="Refresh failed")

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        session_id=session_id,
        last_activity_at=last_activity_at,
        idle_expires_at=session_idle_deadline(last_activity_at),
        idle_timeout_seconds=session_idle_timeout_seconds(),
    )


@router.post("/activity", response_model=ActivityResponse)
def record_activity(user: CurrentUser, credentials: BearerCredentials) -> ActivityResponse:
    """Extend the current session only after an explicit human-activity heartbeat."""

    user_id, session_id = _token_identity(decode_token(credentials.credentials), "access")
    if user_id != int(user["id"]):
        raise HTTPException(status_code=401, detail="Invalid token claims")

    session_state = "active"
    last_activity_at: datetime | None = None
    with get_db() as connection:
        cursor = connection.cursor()
        activity = get_session_activity(cursor, user_id, session_id)
        if activity is None:
            session_state = "revoked"
        elif activity.is_idle:
            revoke_idle_session(cursor, user_id, session_id)
            session_state = "idle"
        else:
            # Update every non-expired token row in the session, including
            # rotated rows, so replay detection can distinguish an active
            # compromised session from a session that genuinely timed out.
            cursor.execute(
                """
                UPDATE refresh_tokens
                SET last_activity_at = NOW()
                WHERE user_id = %s AND session_id = %s AND expires_at > NOW()
                RETURNING last_activity_at
                """,
                (user_id, session_id),
            )
            rows = cursor.fetchall()
            if rows:
                last_activity_at = max(row["last_activity_at"] for row in rows)
            else:
                session_state = "revoked"

    if session_state == "idle":
        raise session_idle_timeout_exception()
    if session_state == "revoked" or last_activity_at is None:
        raise HTTPException(status_code=401, detail="Session revoked or logged out")
    return ActivityResponse(
        last_activity_at=last_activity_at,
        idle_expires_at=session_idle_deadline(last_activity_at),
        idle_timeout_seconds=session_idle_timeout_seconds(),
    )


@router.get("/me")
def get_me(user: CurrentUser) -> dict[str, Any]:
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "is_active": user["is_active"],
        "is_admin": user.get("is_admin", False),
        "persona_prompt": user.get("persona_prompt"),
        "avatar_data": user.get("avatar_data"),
    }


class PersonaUpdateRequest(BaseModel):
    persona_prompt: str = Field(max_length=4_000)


@router.put("/persona")
def update_persona(req: PersonaUpdateRequest, user: CurrentUser) -> dict[str, str]:
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE users SET persona_prompt = %s WHERE id = %s",
            (req.persona_prompt, user["id"]),
        )
    return {"message": "Persona updated"}


class AvatarUpdateRequest(BaseModel):
    avatar_data: str = Field(max_length=70_000)


@router.put("/avatar")
def update_avatar(req: AvatarUpdateRequest, user: CurrentUser) -> dict[str, str]:
    if len(req.avatar_data) > 70_000:
        raise HTTPException(status_code=400, detail="Avatar too large (max ~50KB)")
    if req.avatar_data and not req.avatar_data.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE users SET avatar_data = %s WHERE id = %s",
            (req.avatar_data or None, user["id"]),
        )
    return {"message": "Avatar updated"}


class ChangePasswordRequest(BaseModel):
    current_password: BcryptPassword
    new_password: BcryptPassword


@router.post("/change-password")
def change_password(req: ChangePasswordRequest, user: CurrentUser) -> dict[str, str]:
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE id = %s FOR UPDATE", (user["id"],))
        row = cursor.fetchone()
        if not row or not verify_password(req.current_password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        cursor.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (get_password_hash(req.new_password), user["id"]),
        )
        cursor.execute(
            "UPDATE refresh_tokens SET is_revoked = TRUE WHERE user_id = %s",
            (user["id"],),
        )
    return {"message": "Password changed successfully"}


@router.get("/sessions")
def list_sessions(user: CurrentUser) -> list[dict[str, str | int]]:
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT DISTINCT ON (session_id)
                   session_id, created_at, last_activity_at
            FROM refresh_tokens
            WHERE user_id = %s
              AND is_revoked = FALSE
              AND expires_at > NOW()
              AND last_activity_at > NOW() - (%s * INTERVAL '1 minute')
            ORDER BY session_id, created_at DESC
            """,
            (user["id"], settings.session_idle_timeout_minutes),
        )
        rows = cursor.fetchall()
    return [
        {
            "session_id": str(row["session_id"]),
            "created_at": row["created_at"].isoformat(),
            "last_activity_at": row["last_activity_at"].isoformat(),
            "idle_expires_at": session_idle_deadline(row["last_activity_at"]).isoformat(),
            "idle_timeout_seconds": session_idle_timeout_seconds(),
        }
        for row in rows
    ]


@router.delete("/sessions/{session_id}")
def revoke_session(session_id: str, user: CurrentUser) -> dict[str, str]:
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            UPDATE refresh_tokens
            SET is_revoked = TRUE
            WHERE session_id = %s AND user_id = %s
            """,
            (session_id, user["id"]),
        )
    return {"message": "Session revoked"}


@router.post("/logout")
def logout(user: CurrentUser, credentials: BearerCredentials) -> dict[str, str]:
    _, session_id = _token_identity(decode_token(credentials.credentials), "access")
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            UPDATE refresh_tokens
            SET is_revoked = TRUE
            WHERE session_id = %s AND user_id = %s
            """,
            (session_id, user["id"]),
        )
    return {"message": "Successfully logged out from session"}


_OAUTH_STATE_COOKIE = "lavix_oauth_state"
_OAUTH_STATE_MAX_AGE_SECONDS = 600


def _make_google_state() -> str:
    nonce = secrets.token_urlsafe(16)
    signature = hmac.new(
        settings.secret_key.encode(),
        nonce.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{nonce}.{signature}"


def _oauth_cookie_secure(request: Request) -> bool:
    proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
    return (proto or request.url.scheme) == "https"


def _verify_google_state(state_value: str | None, request: Request) -> bool:
    """State must carry a valid signature AND match the initiating browser's cookie.

    A statically verifiable state alone enables login CSRF: anyone can mint
    valid states by hitting /google/login and redirect a victim through an
    attacker-account flow. Binding the value to an HttpOnly cookie makes the
    check per-browser, and consuming the cookie makes it single-use.
    """
    if not state_value:
        return False
    try:
        nonce, signature = state_value.split(".", maxsplit=1)
    except ValueError:
        return False
    expected = hmac.new(
        settings.secret_key.encode(),
        nonce.encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return False
    cookie_state = request.cookies.get(_OAUTH_STATE_COOKIE)
    return isinstance(cookie_state, str) and bool(cookie_state) and hmac.compare_digest(
        cookie_state, state_value
    )


def _google_redirect_uris() -> list[str]:
    return [uri.strip() for uri in settings.google_redirect_uri.split(",") if uri.strip()]


def _get_google_redirect_uri(request: Request) -> str:
    redirect_uris = _google_redirect_uris()
    if not redirect_uris:
        return ""
    request_host = (request.headers.get("x-forwarded-host") or request.url.hostname or "").split(":")[0]
    for uri in redirect_uris:
        if urlparse(uri).hostname == request_host:
            return uri
    return redirect_uris[0]


@router.get("/google/login")
def google_login(request: Request) -> RedirectResponse:
    redirect_uri = _get_google_redirect_uri(request)
    if not settings.google_client_id or not settings.google_client_secret or not redirect_uri:
        raise HTTPException(503, "Google OAuth not configured")
    state = _make_google_state()
    params = urlencode(
        {
            "client_id": settings.google_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
        }
    )
    response = RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")
    response.set_cookie(
        _OAUTH_STATE_COOKIE,
        state,
        max_age=_OAUTH_STATE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=_oauth_cookie_secure(request),
        path="/",
    )
    return response


@router.get("/google/callback")
def google_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    redirect_uri = _get_google_redirect_uri(request)
    parsed_redirect = urlparse(redirect_uri)
    frontend = f"{parsed_redirect.scheme}://{parsed_redirect.netloc}"
    if not redirect_uri or not frontend:
        raise HTTPException(503, "Google OAuth not configured")

    def _finish(path: str) -> RedirectResponse:
        # The state cookie is single-use: consumed on every callback outcome.
        response = RedirectResponse(f"{frontend}{path}")
        response.delete_cookie(_OAUTH_STATE_COOKIE, path="/")
        return response

    if error or not code:
        return _finish("/?google_error=cancelled")
    if not _verify_google_state(state, request):
        return _finish("/?google_error=invalid_state")

    try:
        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
        token_response.raise_for_status()
        google_access_token = token_response.json().get("access_token")
        if not google_access_token:
            return _finish("/?google_error=invalid_token")
        userinfo_response = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {google_access_token}"},
            timeout=10,
        )
        userinfo_response.raise_for_status()
        payload = userinfo_response.json()
    except (requests.RequestException, ValueError):
        logger.info("Google OAuth exchange failed", exc_info=True)
        return _finish("/?google_error=token_exchange_failed")

    google_user_id = payload.get("sub")
    if not isinstance(google_user_id, str) or not google_user_id:
        return _finish("/?google_error=invalid_token")
    email = str(payload.get("email") or "").strip()

    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, is_active FROM users WHERE google_user_id = %s",
            (google_user_id,),
        )
        user = cursor.fetchone()
        if not user and email:
            cursor.execute("SELECT id, is_active FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
            if user:
                cursor.execute(
                    "UPDATE users SET google_user_id = %s WHERE id = %s",
                    (google_user_id, user["id"]),
                )

        is_new_user = not user
        if not user:
            if not settings.registration_enabled:
                return _finish("/?google_error=registration_disabled")
            base = email.split("@", maxsplit=1)[0] if email else f"user_{google_user_id[-8:]}"
            username = _unique_username(cursor, base)
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, google_user_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id, is_active
                """,
                (
                    username,
                    email or f"{google_user_id}@google.local",
                    get_password_hash(secrets.token_urlsafe(32)),
                    google_user_id,
                ),
            )
            user = cursor.fetchone()
        if not user["is_active"]:
            return _finish("/?google_error=account_inactive")

        cursor.execute(
            "UPDATE users SET last_login = NOW(), last_login_ip = %s WHERE id = %s",
            (request.client.host if request.client else "unknown", user["id"]),
        )
        cursor.execute("SELECT username FROM users WHERE id = %s", (user["id"],))
        stored_username = cursor.fetchone()["username"]
        user_id = int(user["id"])

    session_id = str(uuid.uuid4())
    last_activity_at = datetime.now(UTC)
    exchange_key = secrets.token_urlsafe(24)
    _get_rl_client().setex(
        f"google_auth:{exchange_key}",
        60,
        json.dumps(
            {
                "access_token": create_access_token(user_id, session_id),
                "refresh_token": create_refresh_token(
                    user_id,
                    session_id,
                    last_activity_at=last_activity_at,
                ),
                "session_id": session_id,
                "last_activity_at": last_activity_at.isoformat(),
                "idle_expires_at": session_idle_deadline(last_activity_at).isoformat(),
                "idle_timeout_seconds": session_idle_timeout_seconds(),
                "is_new_user": is_new_user,
                "username": stored_username,
            }
        ),
    )
    return _finish(f"/?google_auth={exchange_key}")


@router.get("/google/token")
def google_get_token(key: str) -> dict[str, Any]:
    client = _get_rl_client()
    raw_payload = client.getdel(f"google_auth:{key}")
    if not raw_payload:
        raise HTTPException(400, "Invalid or expired Google auth key")
    try:
        data = json.loads(raw_payload)
        return {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "token_type": "bearer",
            "session_id": data["session_id"],
            "last_activity_at": data["last_activity_at"],
            "idle_expires_at": data["idle_expires_at"],
            "idle_timeout_seconds": data["idle_timeout_seconds"],
            "is_new_user": data.get("is_new_user", False),
            "username": data.get("username", ""),
        }
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "Invalid or expired Google auth key") from None


@router.put("/username")
def update_username(body: UsernameUpdateRequest, user: CurrentUser) -> dict[str, str]:
    new_username = body.username.strip()
    if not 3 <= len(new_username) <= 50:
        raise HTTPException(400, "Username must be 3–50 characters")
    if not _USERNAME_PATTERN.fullmatch(new_username):
        raise HTTPException(400, "Username may only contain letters, numbers, '.', '_', '-'")
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT 1 FROM users WHERE username = %s AND id != %s",
            (new_username, user["id"]),
        )
        if cursor.fetchone():
            raise HTTPException(409, "Username already taken")
        cursor.execute(
            "UPDATE users SET username = %s WHERE id = %s",
            (new_username, user["id"]),
        )
    return {"message": "Username updated", "username": new_username}


@router.get("/models")
def get_available_models(_user: CurrentAIUser) -> dict[str, list[dict[str, Any]]]:
    return {"models": get_model_service().get_available_models()}


@router.put("/chat-model")
def update_chat_model(
    body: ChatModelPreferenceRequest,
    user: CurrentAIUser,
) -> dict[str, Any]:
    """Persist one account-wide local chat model, or follow the system default."""

    preferred_model = body.model.strip() if isinstance(body.model, str) else None
    preferred_model = preferred_model or None
    installed_names = {
        str(model["name"])
        for model in get_model_service().get_available_models()
        if model.get("name")
    }
    with get_db() as connection:
        system_config = ModelConfigurationRepository(connection).get()
        allowed_models = list(system_config.chat.allowed_models)
        if preferred_model is not None and not system_config.chat.enabled:
            raise HTTPException(
                status_code=409,
                detail={"code": "ai_chat_disabled", "message": "AI Chat is disabled"},
            )
        if preferred_model is not None and preferred_model not in allowed_models:
            raise HTTPException(status_code=422, detail="Requested model is not allowlisted")
        if preferred_model is not None and preferred_model not in installed_names:
            raise HTTPException(status_code=422, detail="Requested model is not installed")
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE users SET preferred_chat_model = %s WHERE id = %s",
            (preferred_model, user["id"]),
        )
    resolution = resolve_account_chat_model(
        system_config,
        preferred_model,
        tuple(installed_names),
    )
    return {
        "preferred_chat_model": resolution.preferred_model,
        "active_chat_model": resolution.active_model,
        "allowed_chat_models": allowed_models,
        "chat_enabled": system_config.chat.enabled,
        "revision": system_config.revision,
    }


@router.get("/model-config")
def get_model_config(user: CurrentAIUser, provider: str = "") -> dict[str, Any]:
    if provider not in _PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported model provider")
    model_service = get_model_service()
    preferred_chat_model: str | None = None
    with get_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT preferred_chat_model FROM users WHERE id = %s",
            (user["id"],),
        )
        row = cursor.fetchone()
        if row:
            preferred_chat_model = str(row.get("preferred_chat_model") or "").strip() or None
        system_config = ModelConfigurationRepository(connection).get()

    # Ollama is the only active chat provider; OpenRouter support was removed.
    effective_provider = "ollama" if provider in ("", "auto", "ollama") else provider
    installed_models = model_service.get_available_models()
    available_models = [str(model["name"]) for model in installed_models if model.get("name")]
    installed_names = set(available_models)
    allowed_models = list(system_config.chat.allowed_models)
    chat_resolution = resolve_account_chat_model(
        system_config,
        preferred_chat_model,
        available_models,
    )
    preferred_chat_model = chat_resolution.preferred_model
    active_model = chat_resolution.active_model
    vision_model = system_config.vision.model or ""
    intelligence_model = system_config.intelligence.model or ""
    memory_extraction_model = system_config.memory_extraction.model or ""
    embedding_model = settings.embedding_model_name.strip()
    reranker_configured = bool(settings.rerank_base_url.strip() and settings.rerank_model.strip())
    health_probe = getattr(model_service, "is_service_available", None)
    reranker_available = bool(
        reranker_configured
        and system_config.reranker_enabled
        and callable(health_probe)
        and health_probe(f"{settings.rerank_base_url.rstrip('/')}/health")
    )
    result: dict[str, Any] = {
        "revision": system_config.revision,
        "configuration_source": system_config.source,
        "model": active_model or "",
        "active_chat_model": active_model,
        "preferred_chat_model": preferred_chat_model,
        "allowed_chat_models": allowed_models,
        "provider": effective_provider,
        "available_models": available_models,
        "openrouter_models": [],
        "openrouter_configured": False,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_available": bool(available_models),
        "embedding_model": embedding_model,
        "embedding_api_url": settings.embedding_api_url,
        "reranker_configured": reranker_configured,
        "reranker_model": settings.rerank_model if settings.rerank_base_url else "",
        "reranker_url": settings.rerank_base_url,
        "enable_reranking": system_config.reranker_enabled,
        "chat": {
            "model": active_model,
            "preferred_model": preferred_chat_model,
            "allowed_models": allowed_models,
            "default_model": system_config.chat.default_model,
            "enabled": system_config.chat.enabled,
            "configured": bool(system_config.chat.default_model),
            "available": chat_resolution.available,
            "optional": False,
        },
        "vision": {
            "model": vision_model,
            "configured": bool(vision_model),
            "enabled": system_config.vision.enabled,
            "available": bool(vision_model and vision_model in installed_names),
            "optional": True,
        },
        "intelligence": {
            "model": intelligence_model,
            "configured": bool(intelligence_model),
            "enabled": system_config.intelligence.enabled,
            "available": bool(intelligence_model and intelligence_model in installed_names),
            "optional": True,
        },
        "memory_extraction": {
            "model": memory_extraction_model,
            "configured": bool(memory_extraction_model),
            "enabled": system_config.memory_extraction.enabled,
            "available": bool(
                memory_extraction_model and memory_extraction_model in installed_names
            ),
            "optional": True,
        },
        "embedding": {
            "model": embedding_model,
            "dimension": settings.embedding_dimension,
            "url": settings.embedding_api_url,
            "configured": bool(embedding_model and settings.embedding_api_url.strip()),
            "available": bool(embedding_model and embedding_model in installed_names),
            "optional": False,
        },
        "reranker": {
            "model": settings.rerank_model,
            "url": settings.rerank_base_url,
            "configured": reranker_configured,
            "enabled": system_config.reranker_enabled,
            "available": reranker_available,
            "optional": True,
        },
    }
    return result
