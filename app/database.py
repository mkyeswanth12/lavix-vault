#!/usr/bin/env python3
"""Database connection pool + simple encryption for secrets"""

import base64
import logging
import os
from contextlib import contextmanager

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseUnavailableError(RuntimeError):
    """Raised without leaking connection credentials to callers or logs."""


_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=(
                f"host={settings.db_host} "
                f"port={settings.db_port} "
                f"dbname={settings.db_name} "
                f"user={settings.db_user}"
            ),
            kwargs={
                "password": settings.db_pass,
                "row_factory": dict_row,
                "connect_timeout": 2,
            },
            min_size=2,
            max_size=10,
            check=ConnectionPool.check_connection,
        )
        logger.info("Database connection pool created (min=2, max=10)")
    return _pool


def get_connection():
    try:
        return _get_pool().getconn()
    except Exception as exc:
        raise DatabaseUnavailableError("database unavailable") from exc


def put_connection(conn):
    """Return a connection to the pool. Safe to call with non-pool connections."""
    if conn is not None and _pool is not None:
        try:
            _pool.putconn(conn)
        except Exception:
            logger.debug("Could not return connection to pool (may not be pool-owned)")


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        put_connection(conn)


def close_pool():
    """Shut down the connection pool gracefully."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("Database connection pool closed")


# ── Simple Encryption for User Secrets (OpenRouter API Key) ─────────────
_ENC_KEY: bytes | None = None


def _get_encryption_key() -> bytes:
    """Derive encryption key from SECRET_KEY (32 bytes for AES-256-GCM)"""
    global _ENC_KEY
    if _ENC_KEY is None:
        import hashlib

        _ENC_KEY = hashlib.sha256(settings.secret_key.encode()).digest()
    return _ENC_KEY


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret (API key) and return base64 encoded ciphertext"""
    if not plaintext:
        return ""
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.b64encode(nonce + ct).decode()


def decrypt_secret(ciphertext_b64: str) -> str:
    """Decrypt a base64 encoded ciphertext"""
    if not ciphertext_b64:
        return ""
    try:
        key = _get_encryption_key()
        aesgcm = AESGCM(key)
        data = base64.b64decode(ciphertext_b64)
        nonce = data[:12]
        ct = data[12:]
        pt = aesgcm.decrypt(nonce, ct, None)
        return pt.decode()
    except Exception:
        logger.warning("Failed to decrypt stored provider secret")
        return ""
