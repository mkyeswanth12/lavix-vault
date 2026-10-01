"""Durable, parser-agnostic ingestion primitives for Lavix Vault.

This package is intentionally isolated from the existing request handlers.  Heavy
document libraries are imported only inside their adapters so importing
``app.ingestion`` remains cheap and testable.
"""

from .models import (
    BoundingBox,
    CanonicalChunk,
    CanonicalDocument,
    CanonicalElement,
    CoordinateSystem,
    ElementType,
    IngestionJob,
    JobState,
    Provenance,
    PublicationResult,
)
from .routing import ParserKind, ParserRoute, route_file

__all__ = [
    "BoundingBox",
    "CanonicalChunk",
    "CanonicalDocument",
    "CanonicalElement",
    "CoordinateSystem",
    "ElementType",
    "IngestionJob",
    "JobState",
    "ParserKind",
    "ParserRoute",
    "PublicationResult",
    "Provenance",
    "route_file",
]
