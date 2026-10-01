"""Fast-path digital PDF extraction with thin-fallback to the OCR chain."""

from pathlib import Path

import pytest

from app.ingestion.errors import EmptyDocumentError
from app.ingestion.parsers.base import ParseRequest
from app.ingestion.parsers.fastpdf import (
    FastPdfFirstParser,
    FastPdfParser,
    FastPdfSettings,
    FastPdfThinError,
)


def _request(tmp_path: Path, name: str = "doc.pdf") -> ParseRequest:
    target = tmp_path / name
    target.write_bytes(b"%PDF-1.4 fake")
    return ParseRequest(
        path=target,
        source_name=name,
        media_type="application/pdf",
        source_sha256="b" * 64,
    )


class _PagesParser(FastPdfParser):
    def __init__(self, pages: list[str], **kwargs):
        super().__init__(FastPdfSettings(**kwargs) if kwargs else FastPdfSettings())
        self._pages = pages

    async def parse(self, request, *, cancel_event=None):  # type: ignore[override]
        import app.ingestion.parsers.fastpdf as module

        original = module._extract_pages
        module._extract_pages = lambda path, event: list(self._pages)  # noqa: E731
        try:
            return await super().parse(request, cancel_event=cancel_event)
        finally:
            module._extract_pages = original


def test_accepts_digital_pdf(tmp_path):
    body = "Digital content paragraph. " * 40
    parser = _PagesParser([body, body])
    document = _run(parser, _request(tmp_path))
    assert document.parser_fingerprint.split("+")[-1].startswith("fastpdf:")
    assert len(document.elements) == 2
    assert document.elements[0].provenance[0].page_number == 1
    assert document.elements[1].provenance[0].page_number == 2
    assert all(e.text for e in document.elements)


def test_fingerprint_pins_pdfium_version(tmp_path):
    parser = _PagesParser(["Digital content paragraph. " * 40])
    document = _run(parser, _request(tmp_path))
    fingerprint = document.parser_fingerprint.split("+")[-1]
    assert fingerprint.startswith("fastpdf:")
    assert fingerprint != "fastpdf:unknown"


def test_thin_pdf_falls_through(tmp_path):
    parser = _PagesParser(["scan", ""])
    with pytest.raises(FastPdfThinError):
        _run(parser, _request(tmp_path))


def test_empty_document(tmp_path):
    target = tmp_path / "empty.pdf"
    target.write_bytes(b"")
    parser = FastPdfParser()
    with pytest.raises(EmptyDocumentError):
        _run(parser, ParseRequest(path=target, source_name="empty.pdf", media_type="application/pdf"))


def test_first_parser_chains_to_fallback(tmp_path):
    seen = {}

    class _Fallback:
        async def parse(self, request, *, cancel_event=None):
            seen["chain"] = request.parser_chain
            seen["called"] = True
            return "FALLBACK-DOC"

    chained = FastPdfFirstParser(_PagesParser(["scan"]), _Fallback())
    assert _run(chained, _request(tmp_path)) == "FALLBACK-DOC"
    assert seen["called"] is True
    assert any("fastpdf" in part and "thin" in part for part in seen["chain"])


def test_first_parser_returns_fast_result(tmp_path):
    body = "Digital content paragraph. " * 40

    class _Fallback:
        async def parse(self, request, *, cancel_event=None):  # pragma: no cover
            raise AssertionError("fallback must not run for digital PDFs")

    chained = FastPdfFirstParser(_PagesParser([body]), _Fallback())
    document = _run(chained, _request(tmp_path))
    assert len(document.elements) == 1


def _run(parser, request):
    import asyncio

    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        parser.parse(request)
    )
