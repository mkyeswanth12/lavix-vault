from pathlib import Path

from app.ingestion.parsers.docling import DoclingParser
from app.ingestion.parsers.fastpdf import FastPdfFirstParser, FastPdfParser
from app.ingestion.parsers.opendataloader import (
    OpenDataLoaderEmptyFallbackParser,
    OpenDataLoaderPdfParser,
)
from app.ingestion.parsers.registry import ParserRegistry, ParserSettings
from app.ingestion.routing import route_file


def test_pdf_route_keeps_opendataloader_primary_with_one_local_ocr_fallback(tmp_path: Path) -> None:
    registry = ParserRegistry(ParserSettings(temp_root=tmp_path))

    parser = registry.get(route_file("scan.pdf", "application/pdf"))

    # Fast digital-PDF extraction runs first by default, then the ODL chain.
    assert isinstance(parser, FastPdfFirstParser)
    assert isinstance(parser.fast, FastPdfParser)
    fallback = parser.fallback
    assert isinstance(fallback, OpenDataLoaderEmptyFallbackParser)
    assert isinstance(fallback.primary, OpenDataLoaderPdfParser)
    assert fallback.primary.settings.hybrid_timeout_ms == 3_600_000
    assert fallback.primary.settings.process_timeout_seconds == 7_200.0
    assert fallback.primary.settings.hybrid_timeout_ms < fallback.primary.settings.process_timeout_seconds * 1_000
    assert isinstance(fallback.fallback, DoclingParser)
    assert fallback.fallback is registry.get(route_file("report.docx", "application/octet-stream"))
