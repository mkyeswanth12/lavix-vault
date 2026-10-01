"""Shared credential redaction (stdlib only — importable anywhere).

Values are replaced with [REDACTED]; labels and surrounding prose are kept.
Usernames alone are never redacted. See also the memory validator's
detection patterns (app.graph_memory.validation), which answer a different
question (store-or-reject); this module answers redact-in-place.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

_LABEL_VALUE = re.compile(
    r"(?i)\b(password|passcode|pin|api[ _-]?key|access[ _-]?token|refresh[ _-]?token|"
    r"secret|private[ _-]?key|bearer|auth[ _-]?token|token)"
    r"(?:\s*(?:is|=|:)\s*|\s+)([A-Za-z0-9_./+@=~$-]{4,})"
)
_USERINFO = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*)://([^/\s:@]+):([^/\s@]+)@"
)
_BEARER_JWT = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_.\-~+/=]+")
_JWT_SHAPE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_.\-/+=]+")
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:PRIVATE KEY|RSA PRIVATE KEY|OPENSSH PRIVATE KEY|DSA PRIVATE KEY|EC PRIVATE KEY)-----"
    r".*?"
    r"-----END (?:PRIVATE KEY|RSA PRIVATE KEY|OPENSSH PRIVATE KEY|DSA PRIVATE KEY|EC PRIVATE KEY)-----",
    re.IGNORECASE | re.DOTALL,
)
_SSH_KEY_LINE = re.compile(
    r"(?im)^.*\bssh-(?:rsa|ed25519|ecdsa|dss)\s+[A-Za-z0-9+/=]+.*$"
)


_PASSWORD_FOR_VALUE = re.compile(
    r"(?i)\bpassword\s+for\s+\S+\s*:\s+(\S+)"
)


def _scrub_userinfo(match: re.Match[str]) -> str:
    return f"{match.group(1)}://{match.group(2)}:{REDACTED}@"


def _scrub_labeled(match: re.Match[str]) -> str:
    text = match.group(0)
    value = match.group(2)
    return text[: -len(value)] + REDACTED if value else text


def redact_credentials(text: object) -> str:
    """Replace credential values with [REDACTED], keep everything else."""
    out = str(text or "")
    if not out:
        return ""
    out = _PRIVATE_KEY_BLOCK.sub(REDACTED, out)
    out = _SSH_KEY_LINE.sub(REDACTED, out)
    out = _USERINFO.sub(_scrub_userinfo, out)
    out = _BEARER_JWT.sub("Bearer " + REDACTED, out)
    out = _JWT_SHAPE.sub(REDACTED, out)
    out = _PASSWORD_FOR_VALUE.sub(
        lambda match: match.group(0)[: -len(match.group(1))] + REDACTED, out
    )
    out = _LABEL_VALUE.sub(_scrub_labeled, out)
    return out


_LABEL_ONLY = re.compile(
    r"(?i)(?:password|passcode|pin|api[ _-]?key|access[ _-]?token|"
    r"refresh[ _-]?token|secret|private[ _-]?key|bearer|auth[ _-]?token)"
    r"(?:\s+for\b[^\n]{0,120})?\s*:?\s*$"
    r"|\bssh(?:[ _-]?(?:key|fingerprint|passphrase))?\b"
    r"|\bgit[ _-]?credential\b"
    r"|^(?:token|secret|api[ _-]?key)\s*$",
)


def contains_credentials(text: object) -> bool:
    """True when text carries a credential value or a credential label."""
    value = str(text or "")
    if not value.strip():
        return False
    if (
        _PRIVATE_KEY_BLOCK.search(value)
        or _SSH_KEY_LINE.search(value)
        or _USERINFO.search(value)
        or _BEARER_JWT.search(value)
        or _JWT_SHAPE.search(value)
    ):
        return True
    if _LABEL_VALUE.search(value):
        return True
    return _LABEL_ONLY.search(value) is not None


# Labels that start a credential span in streamed text. When one appears,
# output holds from the label onward until the value completes, so a
# partial secret never reaches the wire.
_STREAM_LABEL = re.compile(
    r"(?i)(password|passcode|pin|api[ _-]?key|access[ _-]?token|"
    r"refresh[ _-]?token|secret|private[ _-]?key|bearer|auth[ _-]?token|token)\b"
)
# Single words that can begin (or extend) a label; a trailing buffer
# fragment matching either direction holds output until more text arrives.
# Holding too much only costs latency — finalize() always scrubs.
_STREAM_LABEL_WORDS = frozenset({
    "password", "passcode", "pin", "api", "key", "access", "refresh",
    "token", "secret", "private", "bearer", "auth",
})


def _stream_hold_at(text: str) -> int:
    """Earliest index to hold from: complete label match, or a trailing
    fragment that could complete into (or extend) a label word."""
    hold_at = len(text)
    for match in _STREAM_LABEL.finditer(text):
        if match.start() < hold_at:
            hold_at = match.start()
    tail = re.search(r"[A-Za-z]{2,}$", text)
    if tail is not None:
        fragment = tail.group(0).casefold()
        for word in _STREAM_LABEL_WORDS:
            if word.startswith(fragment) or fragment.startswith(word):
                position = tail.start()
                if position < hold_at:
                    hold_at = position
                break
    return hold_at


class CredentialStreamScrubber:
    """Hold-back scrubber for token deltas.

    Clean text streams immediately; from the last credential label onward
    output is held until finalize(), which scrubs the remainder in full.
    Invariant: "".join(feeds) + finalize() == redact_credentials(full).
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: object) -> str:
        """Return the emittable clean prefix for one delta (may be "")."""
        self._buffer += str(delta or "")
        cut = _stream_hold_at(self._buffer)
        emit, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return emit

    def finalize(self) -> str:
        """Scrub and release whatever is held. Always call at stream end."""
        held, self._buffer = self._buffer, ""
        return redact_credentials(held) if held else ""


class RedactingFilter:
    """Logging filter: scrub credential values from rendered records.

    Attach to handlers in main entrypoints. Operates on the formatted
    message only; args, levels, and logger names pass through untouched.
    """

    def filter(self, record) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        cleaned = redact_credentials(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True
