"""Parser contracts and shared identity helpers."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from ..models import CanonicalDocument


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ParseRequest:
    path: Path
    source_name: str
    media_type: str
    source_sha256: str = ""
    parser_chain: tuple[str, ...] = ()

    def with_path(self, path: Path, *, parser: str) -> ParseRequest:
        return replace(self, path=path, parser_chain=(*self.parser_chain, parser))

    def resolved_sha256(self) -> str:
        return self.source_sha256 or sha256_file(self.path)


class DocumentParser(Protocol):
    async def parse(
        self,
        request: ParseRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> CanonicalDocument: ...
