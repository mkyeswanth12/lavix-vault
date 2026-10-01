"""Bounded per-run evidence registry used to preserve the public SSE contract."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _RunEvidence:
    expires_at: float
    entries: list[dict[str, Any]] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=lambda: {"V": 0, "W": 0})


class RunEvidenceStore:
    def __init__(self, *, ttl_seconds: int = 600, max_entries: int = 40) -> None:
        if ttl_seconds < 1 or max_entries < 1:
            raise ValueError("evidence limits must be positive")
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._runs: dict[str, _RunEvidence] = {}
        self._lock = asyncio.Lock()

    async def record(
        self,
        run_id: str,
        kind: str,
        evidence: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prefix = "V" if kind == "vault" else "W" if kind == "web" else None
        if prefix is None:
            raise ValueError("unknown evidence kind")
        now = time.monotonic()
        async with self._lock:
            self._prune(now)
            run = self._runs.setdefault(run_id, _RunEvidence(expires_at=now + self._ttl))
            run.expires_at = now + self._ttl
            recorded: list[dict[str, Any]] = []
            for raw in evidence:
                key = self._key(prefix, raw)
                existing_id = run.keys.get(key)
                if existing_id is not None:
                    existing = next(item for item in run.entries if item["id"] == existing_id)
                    # Tie-break (mirrors the prefetch merge in
                    # agent_runtime/cuga_adapter.py): a later leg's
                    # relevant=True outranks an earlier relevant=False for
                    # the same URL, regardless of arrival order. Items
                    # without a relevant flag (vault evidence) keep
                    # first-wins behavior exactly as before.
                    if raw.get("relevant") is True and existing.get("relevant") is not True:
                        upgraded = deepcopy(raw)
                        upgraded["id"] = existing["id"]
                        upgraded["untrusted"] = True
                        run.entries[run.entries.index(existing)] = upgraded
                        logger.warning(
                            "evidence_tiebreak run=%s url=%.80s before=%s after=True",
                            run_id,
                            str(raw.get("url") or ""),
                            existing.get("relevant"),
                        )
                        recorded.append(deepcopy(upgraded))
                    else:
                        recorded.append(deepcopy(existing))
                    continue
                if len(run.entries) >= self._max_entries:
                    break
                run.counts[prefix] += 1
                item = deepcopy(raw)
                item["id"] = f"{prefix}{run.counts[prefix]}"
                item["untrusted"] = True
                run.keys[key] = item["id"]
                run.entries.append(item)
                recorded.append(deepcopy(item))
            return recorded

    async def get(self, run_id: str) -> list[dict[str, Any]]:
        now = time.monotonic()
        async with self._lock:
            self._prune(now)
            run = self._runs.get(run_id)
            return deepcopy(run.entries) if run else []

    async def discard(self, run_id: str) -> None:
        async with self._lock:
            self._runs.pop(run_id, None)

    def _prune(self, now: float) -> None:
        for run_id in [key for key, value in self._runs.items() if value.expires_at <= now]:
            self._runs.pop(run_id, None)

    @staticmethod
    def _key(prefix: str, value: dict[str, Any]) -> str:
        if prefix == "V":
            return f"V:{value.get('file_id')}:{value.get('revision')}:{value.get('chunk_id')}"
        return f"W:{value.get('url')}"


run_evidence_store = RunEvidenceStore()
