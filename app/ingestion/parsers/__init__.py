"""Lazy parser adapters and their registry."""

from .base import DocumentParser, ParseRequest
from .registry import ParserRegistry, ParserSettings

__all__ = ["DocumentParser", "ParseRequest", "ParserRegistry", "ParserSettings"]
