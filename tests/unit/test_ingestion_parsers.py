import asyncio
import base64
import json
import tempfile
import unittest
from os import environ
from pathlib import Path
from unittest.mock import patch

import httpx

from app.graph_memory.inference_priority import BackgroundInferenceDeferred
from app.ingestion.errors import (
    CorruptDocumentError,
    EmptyDocumentError,
    IngestionCancelled,
    ParseError,
    ParserUnavailableError,
    PasswordRequiredError,
    ProcessExecutionError,
)
from app.ingestion.models import CanonicalDocument, CanonicalElement, ElementType
from app.ingestion.parsers.base import ParseRequest
from app.ingestion.parsers.docling import (
    DoclingParser,
    DoclingSettings,
    _scoped_docling_temp,
    docling_dict_to_elements,
)
from app.ingestion.parsers.opendataloader import (
    OpenDataLoaderEmptyFallbackParser,
    OpenDataLoaderNoElementsError,
    OpenDataLoaderPdfParser,
    OpenDataLoaderSettings,
    _only_scanner_watermark_text,
    odl_json_to_elements,
)
from app.ingestion.parsers.registry import ParserRegistry, ParserSettings
from app.ingestion.parsers.text import DeterministicTextParser
from app.ingestion.parsers.vision import (
    OllamaVisionParser,
    VisionFallbackParser,
    VisionSettings,
    _bounded_image_size,
    _has_usable_ocr,
    _usable_vision_summary,
    _vision_text,
)
from app.ingestion.routing import ParserKind, ParserRoute

SOURCE_HASH = "b" * 64


class IngestionParserTests(unittest.IsolatedAsyncioTestCase):
    def test_docling_uses_only_the_corpus_ocr_languages(self):
        self.assertEqual(DoclingSettings().tesseract_languages, ("eng", "kan", "ara"))
        self.assertEqual(DoclingSettings().pdf_ocr_psm, 6)

    def test_pdf_timeouts_default_to_one_hour_per_hybrid_call_with_two_hours_overall(self):
        with patch.dict(environ, {}, clear=True):
            settings = ParserSettings.from_environment()

        self.assertEqual(settings.opendataloader_hybrid_timeout_ms, 3_600_000)
        self.assertEqual(settings.parser_timeout_seconds, 7_200.0)
        self.assertLess(
            settings.opendataloader_hybrid_timeout_ms,
            settings.parser_timeout_seconds * 1_000,
        )

    def test_parser_timeout_can_be_overridden(self):
        with patch.dict(
            environ,
            {
                "INGESTION_PARSER_TIMEOUT_SECONDS": "1500",
                "OPENDATALOADER_HYBRID_TIMEOUT_MS": "600000",
            },
            clear=True,
        ):
            settings = ParserSettings.from_environment()

        self.assertEqual(settings.parser_timeout_seconds, 1_500.0)
        self.assertEqual(settings.opendataloader_hybrid_timeout_ms, 600_000)

    def test_docling_temp_scope_removes_third_party_leaks_and_restores_global_setting(self):
        original = tempfile.tempdir
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "parser-temp"
            with _scoped_docling_temp(root):
                leaked = Path(tempfile.mkdtemp()) / "drawing_only.docx"
                leaked.write_bytes(b"temporary")
                self.assertTrue(leaked.exists())
            self.assertEqual(list(root.iterdir()), [])
        self.assertEqual(tempfile.tempdir, original)

    async def test_registry_normalizes_trusted_docling_extension_without_changing_source(self):
        class RecordingParser:
            observed_suffix = None
            observed_content = None
            observed_path = None

            async def parse(self, request, *, cancel_event=None):
                self.observed_suffix = request.path.suffix
                self.observed_content = request.path.read_bytes()
                self.observed_path = request.path
                return "parsed"

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "misleading.bin"
            source.write_bytes(b"trusted content")
            parser = RecordingParser()
            registry = ParserRegistry(ParserSettings(temp_root=root / "parser-temp"))
            route = ParserRoute(ParserKind.DOCLING, ".jpg")
            request = ParseRequest(source, source.name, "image/jpeg", SOURCE_HASH)
            with patch.object(registry, "get", return_value=parser):
                result = await registry.parse(route, request)

            self.assertEqual(result, "parsed")
            self.assertEqual(parser.observed_suffix, ".jpg")
            self.assertEqual(parser.observed_content, b"trusted content")
            self.assertEqual(source.read_bytes(), b"trusted content")
            self.assertFalse(parser.observed_path.exists())

    async def test_text_parser_is_deterministic_with_line_provenance(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "notes.txt"
            path.write_bytes(b"first line\nsecond line\n\nthird\n")
            parser = DeterministicTextParser()
            request = ParseRequest(
                path=path,
                source_name=path.name,
                media_type="text/plain",
                source_sha256=SOURCE_HASH,
            )
            first = await parser.parse(request)
            second = await parser.parse(request)
        self.assertEqual(first, second)
        self.assertEqual(len(first.elements), 2)
        self.assertEqual(first.elements[0].provenance[0].line_start, 1)
        self.assertEqual(first.elements[0].provenance[0].line_end, 2)
        self.assertEqual(first.elements[1].provenance[0].line_start, 4)

    async def test_text_parser_observes_preexisting_cancellation(self):
        event = asyncio.Event()
        event.set()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "notes.txt"
            path.write_text("content", encoding="utf-8")
            request = ParseRequest(path, path.name, "text/plain", SOURCE_HASH)
            with self.assertRaisesRegex(Exception, "cancelled"):
                await DeterministicTextParser().parse(request, cancel_event=event)

    async def test_local_vision_builds_a_searchable_image_element(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "diagram.png"
            path.write_bytes(b"synthetic-image")
            parser = OllamaVisionParser(VisionSettings(model="qwen-test"))
            request = ParseRequest(path, path.name, "image/png", SOURCE_HASH)
            with (
                patch.object(
                    parser,
                    "_prepare_image",
                    return_value=b"bounded-image",
                ),
                patch.object(
                    parser,
                    "_post",
                    return_value={"summary": "Revenue 2026. A bar chart comparing quarterly revenue."},
                ),
            ):
                document = await parser.parse(request)

        self.assertEqual(document.elements[0].element_type, ElementType.IMAGE)
        self.assertIn("Revenue 2026", document.elements[0].text)
        self.assertIn("bar chart", document.elements[0].text)
        self.assertIn("qwen-test", document.parser_fingerprint)
        self.assertIn("ollama-vision:v8", document.parser_fingerprint)

    async def test_blank_optional_vision_model_fails_closed_without_a_request(self):
        parser = OllamaVisionParser(VisionSettings(model=""))
        request = ParseRequest(Path("unused.png"), "unused.png", "image/png", SOURCE_HASH)

        with (
            patch.object(parser, "_prepare_image") as prepare_image,
            self.assertRaisesRegex(ParserUnavailableError, "vision is disabled"),
        ):
            await parser.parse(request)

        prepare_image.assert_not_called()

    async def test_local_vision_defers_before_model_call_when_chat_has_priority(self):
        async def blocked():
            return False

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "diagram.png"
            path.write_bytes(b"synthetic-image")
            parser = OllamaVisionParser(
                VisionSettings(model="qwen-test"),
                priority_gate=blocked,
                priority_timeout_seconds=0.1,
            )
            request = ParseRequest(path, path.name, "image/png", SOURCE_HASH)
            with (
                patch.object(parser, "_prepare_image", return_value=b"bounded-image"),
                patch.object(parser, "_post") as model_call,
                self.assertRaises(BackgroundInferenceDeferred),
            ):
                await parser.parse(request)

        model_call.assert_not_called()

    async def test_local_vision_cancellation_closes_the_inflight_request(self):
        request_started = asyncio.Event()
        request_closed = asyncio.Event()
        never_finishes = asyncio.Event()

        async def handler(_request):
            request_started.set()
            try:
                await never_finishes.wait()
            finally:
                request_closed.set()
            raise AssertionError("cancelled request unexpectedly resumed")

        parser = OllamaVisionParser(
            VisionSettings(model="qwen-test"),
            transport=httpx.MockTransport(handler),
        )
        cancelled = asyncio.Event()
        task = asyncio.create_task(
            parser._await_model_call(b"bounded-image", cancelled)
        )
        await asyncio.wait_for(request_started.wait(), timeout=0.2)
        cancelled.set()
        with self.assertRaises(IngestionCancelled):
            await asyncio.wait_for(task, timeout=0.2)
        await asyncio.wait_for(request_closed.wait(), timeout=0.2)

    def test_local_vision_bounds_large_images_without_cropping(self):
        self.assertEqual(
            _bounded_image_size(2160, 3840, max_pixels=896 * 896, max_edge=1280),
            (672, 1194),
        )
        self.assertEqual(
            _bounded_image_size(1596, 968, max_pixels=896 * 896, max_edge=1280),
            (1150, 697),
        )
        self.assertEqual(
            _bounded_image_size(320, 200, max_pixels=896 * 896, max_edge=1280),
            (320, 200),
        )

    def test_local_vision_normalizes_before_encoding(self):
        class Image:
            shape = (3840, 2160, 3)

        class Encoded:
            @staticmethod
            def tobytes():
                return b"normalized-png"

        class Cv2:
            IMREAD_COLOR = 1
            INTER_AREA = 3
            IMWRITE_PNG_COMPRESSION = 16
            resized_to = None

            @staticmethod
            def imread(_path, _mode):
                return Image()

            @classmethod
            def resize(cls, _image, dimensions, *, interpolation):
                cls.resized_to = (dimensions, interpolation)
                return Image()

            @staticmethod
            def imencode(extension, _image, options):
                assert extension == ".png"
                assert options == [16, 6]
                return True, Encoded()

        parser = OllamaVisionParser(VisionSettings())
        with patch("app.ingestion.parsers.vision._load_cv2", return_value=Cv2):
            payload = parser._prepare_image(Path("unused.webp"))

        self.assertEqual(payload, b"normalized-png")
        self.assertEqual(Cv2.resized_to, ((672, 1194), Cv2.INTER_AREA))

    async def test_local_vision_request_uses_bounded_payload_and_context(self):
        observed = {}

        async def handler(request):
            observed["request"] = request
            observed["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                content=b'{"message":{"content":"{\\"summary\\":\\"Key text. A concise image summary.\\"}"}}',
            )

        parser = OllamaVisionParser(
            VisionSettings(model="qwen-test"),
            transport=httpx.MockTransport(handler),
        )
        result = await parser._post(b"bounded-image")

        request = observed["request"]
        body = observed["body"]
        self.assertEqual(result, '{"summary":"Key text. A concise image summary."}')
        self.assertEqual(base64.b64decode(body["messages"][0]["images"][0]), b"bounded-image")
        self.assertEqual(parser.settings.timeout_seconds, 300)
        self.assertEqual(request.headers["accept-encoding"], "identity")
        self.assertEqual(body["options"]["num_ctx"], 16384)
        self.assertEqual(body["options"]["num_gpu"], 0)
        self.assertEqual(body["options"]["num_predict"], 256)
        self.assertEqual(body["format"]["required"], ["summary"])
        self.assertEqual(body["format"]["properties"]["summary"]["maxLength"], 600)
        self.assertEqual(set(body["format"]["properties"]), {"summary"})
        self.assertIn("one factual plain-text", body["messages"][0]["content"])
        self.assertIn("main header or top-center box", body["messages"][0]["content"])
        self.assertIn("do not transcribe or guess", body["messages"][0]["content"])

    async def test_local_vision_recovers_truncated_summary_at_output_token_limit(self):
        async def handler(_request):
            return httpx.Response(
                200,
                content=b'{"done_reason":"length","message":{"content":"{\\"summary\\":\\"Useful partial summary\\nwith detail"}}',
            )

        parser = OllamaVisionParser(
            VisionSettings(),
            transport=httpx.MockTransport(handler),
        )
        self.assertEqual(
            _vision_text(await parser._post(b"bounded-image")),
            "Useful partial summary\nwith detail",
        )

    def test_local_vision_extracts_valid_dict_and_json_string(self):
        self.assertEqual(_vision_text({"summary": "Useful dict summary"}), "Useful dict summary")
        self.assertEqual(
            _vision_text('{"summary":"Useful string summary"}'),
            "Useful string summary",
        )

    def test_local_vision_rejects_empty_or_structural_content(self):
        for payload in (None, {}, "", {"summary": ""}, '{"summary":""}', '{"other":"raw"}'):
            with self.subTest(payload=payload):
                self.assertEqual(_vision_text(payload), "")

    def test_local_vision_rejects_markup_and_numeric_ocr_hallucinations(self):
        for summary in (
            "Urban road page </broken> with invented browser text and buildings.",
            "A city scene allegedly identified by tracking number 340058760000000.",
        ):
            with self.subTest(summary=summary):
                self.assertEqual(_vision_text({"summary": summary}), "")

    async def test_vision_fallback_runs_only_for_empty_ocr(self):
        class Parser:
            def __init__(self, result=None, error=None):
                self.result = result
                self.error = error
                self.calls = 0

            async def parse(self, _request, *, cancel_event=None):
                self.calls += 1
                if self.error:
                    raise self.error
                return self.result

        request = ParseRequest(Path("unused.png"), "unused.png", "image/png", SOURCE_HASH)
        primary = Parser(error=EmptyDocumentError())
        fallback = Parser(result="vision-document")
        parser = VisionFallbackParser(primary, fallback)
        self.assertEqual(await parser.parse(request), "vision-document")
        self.assertEqual((primary.calls, fallback.calls), (1, 1))

        primary = Parser(error=ParseError("corrupt"))
        fallback = Parser(result="must-not-run")
        with self.assertRaises(ParseError):
            await VisionFallbackParser(primary, fallback).parse(request)
        self.assertEqual(fallback.calls, 0)

    async def test_vision_replaces_weak_ocr_but_preserves_it_when_vision_is_unavailable(self):
        class Parser:
            def __init__(self, result=None, error=None):
                self.result = result
                self.error = error
                self.calls = 0

            async def parse(self, _request, *, cancel_event=None):
                self.calls += 1
                if self.error:
                    raise self.error
                return self.result

        def document(text):
            return CanonicalDocument(
                source_name="scan.png",
                source_sha256=SOURCE_HASH,
                media_type="image/png",
                parser_fingerprint="ocr",
                elements=(CanonicalElement("el-1", ElementType.TEXT, text, ()),),
            )

        request = ParseRequest(Path("unused.png"), "unused.png", "image/png", SOURCE_HASH)
        weak = document("x 7 |")
        primary = Parser(result=weak)
        vision = Parser(result="vision-document")
        self.assertEqual(
            await VisionFallbackParser(primary, vision).parse(request),
            "vision-document",
        )
        self.assertEqual((primary.calls, vision.calls), (1, 1))

        unavailable = Parser(error=ParserUnavailableError("offline"))
        with self.assertRaises(ParserUnavailableError):
            # A transient vision outage must stay retryable upstream instead
            # of permanently baking an "unreadable" placeholder into the index.
            await VisionFallbackParser(Parser(result=weak), unavailable).parse(request)
        self.assertEqual(unavailable.calls, 1)

        empty_vision = Parser(error=EmptyDocumentError("no text"))
        empty_document = await VisionFallbackParser(Parser(result=weak), empty_vision).parse(request)
        self.assertIn("No reliable text", empty_document.elements[0].text)
        self.assertIn("image-description-unavailable:v1", empty_document.parser_fingerprint)

        usable = document("Invoice number 1042 total amount due")
        unused_vision = Parser(result="must-not-run")
        self.assertIs(
            await VisionFallbackParser(Parser(result=usable), unused_vision).parse(request),
            usable,
        )
        self.assertEqual(unused_vision.calls, 0)

    def test_mixed_script_ocr_noise_is_routed_to_vision(self):
        def document(text):
            return CanonicalDocument(
                source_name="scan.jpg",
                source_sha256=SOURCE_HASH,
                media_type="image/jpeg",
                parser_fingerprint="ocr",
                elements=(CanonicalElement("el-1", ElementType.TEXT, text, ()),),
            )

        self.assertFalse(_has_usable_ocr(document("NID TITS WOdEWHOIqyy 30010 ಊರ ಗಾರ್ ತ್ರ ಸಂಖ್ಯೆ ದಾಖಲೆ ರ್ಮ")))
        self.assertTrue(_has_usable_ocr(document("ಕರ್ನಾಟಕ ಕಂದಾಯ ಭೂಮಿ ದಾಖಲೆ ಮತ್ತು ತಾಲೂಕು ಗ್ರಾಮ ದಾಖಲೆ ಮಾಹಿತಿ")))
        self.assertTrue(_has_usable_ocr(document("ಕರ್ನಾಟಕ ಸರ್ಕಾರ Revenue Department ಗ್ರಾಮ ದಾಖಲೆ Land Record")))
        self.assertFalse(_has_usable_ocr(document("<html><body>broken OCR text</body></html>")))

    def test_vision_summary_rejects_markup_but_allows_comparisons(self):
        self.assertFalse(_usable_vision_summary("The image contains <html> broken markup."))
        self.assertTrue(_usable_vision_summary("The chart shows revenue > expenses for the quarter."))
        self.assertEqual(
            _vision_text({"summary": "This is an L.P.O. (Letter of Purchase) for spray paint."}),
            "This is an L.P.O. for spray paint.",
        )
        self.assertEqual(
            _vision_text(
                {
                    "summary": (
                        "Document type: Parking Sign\n"
                        "Issuing Organization: Unknown\n"
                        "Broad Line-Item Subject: Street scene with parked cars and buildings."
                    )
                }
            ),
            "Street scene with parked cars and buildings.",
        )

    async def test_pdf_ocr_fallback_runs_only_for_odl_empty_and_records_the_attempt(self):
        class Parser:
            def __init__(self, *, result=None, error=None, settings=None):
                self.result = result
                self.error = error
                self.settings = settings
                self.calls = 0
                self.request = None
                self.cancel_event = None

            async def parse(self, request, *, cancel_event=None):
                self.calls += 1
                self.request = request
                self.cancel_event = cancel_event
                if self.error:
                    raise self.error
                return self.result

        settings = OpenDataLoaderSettings(hybrid="off")
        request = ParseRequest(Path("scan.pdf"), "scan.pdf", "application/pdf", SOURCE_HASH)
        cancel_event = asyncio.Event()
        primary = Parser(error=OpenDataLoaderNoElementsError("no elements"), settings=settings)
        fallback = Parser(result="ocr-document")

        parser = OpenDataLoaderEmptyFallbackParser(primary, fallback)
        self.assertEqual(
            await parser.parse(request, cancel_event=cancel_event),
            "ocr-document",
        )
        self.assertEqual((primary.calls, fallback.calls), (1, 1))
        self.assertIs(fallback.cancel_event, cancel_event)
        self.assertEqual(
            fallback.request.parser_chain,
            (f"{settings.fingerprint}:result=empty",),
        )

        cancelled = asyncio.Event()
        cancelled.set()
        primary = Parser(error=OpenDataLoaderNoElementsError("no elements"), settings=settings)
        fallback = Parser(result="must-not-run")
        with self.assertRaises(IngestionCancelled):
            await OpenDataLoaderEmptyFallbackParser(primary, fallback).parse(
                request,
                cancel_event=cancelled,
            )
        self.assertEqual(fallback.calls, 0)

        for error in (
            EmptyDocumentError("empty source"),
            ParseError("parse"),
            PasswordRequiredError("password"),
            CorruptDocumentError("corrupt"),
            ParserUnavailableError("unavailable"),
            IngestionCancelled(),
        ):
            with self.subTest(error=type(error).__name__):
                primary = Parser(error=error, settings=settings)
                fallback = Parser(result="must-not-run")
                with self.assertRaises(type(error)):
                    await OpenDataLoaderEmptyFallbackParser(primary, fallback).parse(request)
                self.assertEqual(fallback.calls, 0)

    async def test_pdf_cli_crash_falls_through_to_docling(self):
        class Parser:
            def __init__(self, *, result=None, error=None, settings=None):
                self.result = result
                self.error = error
                self.settings = settings
                self.calls = 0
                self.request = None

            async def parse(self, request, *, cancel_event=None):
                self.calls += 1
                self.request = request
                if self.error:
                    raise self.error
                return self.result

        settings = OpenDataLoaderSettings(hybrid="off")
        request = ParseRequest(Path("cert.pdf"), "cert.pdf", "application/pdf", SOURCE_HASH)
        primary = Parser(
            error=ProcessExecutionError("ODL exited", detail="boom"),
            settings=settings,
        )
        fallback = Parser(result="docling-document")
        self.assertEqual(
            await OpenDataLoaderEmptyFallbackParser(primary, fallback).parse(request),
            "docling-document",
        )
        self.assertEqual((primary.calls, fallback.calls), (1, 1))
        self.assertEqual(
            fallback.request.parser_chain,
            (f"{settings.fingerprint}:result=crashed",),
        )

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            empty_pdf = root / "empty.pdf"
            empty_pdf.write_bytes(b"")
            fallback = Parser(result="must-not-run")
            parser = OpenDataLoaderEmptyFallbackParser(
                OpenDataLoaderPdfParser(temp_root=root / "parser-temp"),
                fallback,
            )
            with self.assertRaises(EmptyDocumentError):
                await parser.parse(ParseRequest(empty_pdf, empty_pdf.name, "application/pdf", SOURCE_HASH))
            self.assertEqual(fallback.calls, 0)

    async def test_opendataloader_unknown_exit_is_retryable_with_safe_stderr_detail(self):
        class Runner:
            result = None
            args = None

            async def run(self, args, **kwargs):
                self.args = args
                source_path = args[-1]
                temporary_path = kwargs["cwd"]

                class Result:
                    returncode = 1
                    stdout_text = "document output must not become an operator diagnostic"
                    stderr_text = (
                        "x" * 1_800 + f"\n\x1b[31mbackend panic reading {source_path} from {temporary_path} "
                        "token=supersecret final worker failure\x1b[0m\n"
                    )

                self.result = Result()
                return self.result

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "large-private.pdf"
            source.write_bytes(b"%PDF synthetic large fixture")
            runner = Runner()
            parser = OpenDataLoaderPdfParser(
                temp_root=root / "parser-temp",
                runner=runner,
                settings=OpenDataLoaderSettings(),
            )

            with self.assertRaises(ProcessExecutionError) as raised:
                await parser.parse(ParseRequest(source, source.name, "application/pdf", SOURCE_HASH))

        error = raised.exception
        self.assertTrue(error.retryable)
        self.assertEqual(error.code, "process_failed")
        self.assertIs(error.result, runner.result)
        self.assertEqual(
            runner.args[runner.args.index("--hybrid-timeout") + 1],
            "3600000",
        )
        self.assertLessEqual(len(error.detail), 1_700)
        self.assertIn("backend panic", error.detail)
        self.assertIn("final worker failure", error.detail)
        self.assertIn("token=<redacted>", error.detail)
        self.assertNotIn("supersecret", error.detail)
        self.assertNotIn(str(source), error.detail)
        self.assertNotIn("\x1b", error.detail)
        self.assertNotIn("document output", error.detail)

    async def test_opendataloader_password_and_corrupt_exits_remain_terminal(self):
        class Runner:
            def __init__(self, diagnostic):
                self.diagnostic = diagnostic

            async def run(self, _args, **_kwargs):
                class Result:
                    returncode = 1
                    stdout_text = ""
                    stderr_text = self.diagnostic

                return Result()

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "document.pdf"
            source.write_bytes(b"%PDF synthetic fixture")
            request = ParseRequest(source, source.name, "application/pdf", SOURCE_HASH)
            cases = (
                ("input is encrypted", PasswordRequiredError),
                ("malformed cross-reference table", CorruptDocumentError),
            )
            for diagnostic, expected in cases:
                with self.subTest(diagnostic=diagnostic):
                    parser = OpenDataLoaderPdfParser(
                        temp_root=root / "parser-temp",
                        runner=Runner(diagnostic),
                        settings=OpenDataLoaderSettings(hybrid="off"),
                    )
                    with self.assertRaises(expected) as raised:
                        await parser.parse(request)
                    self.assertFalse(raised.exception.retryable)

    async def test_opendataloader_watermark_only_json_raises_the_typed_fallback_boundary(self):
        class Runner:
            async def run(self, args, **_kwargs):
                output_dir = Path(args[args.index("--output-dir") + 1])
                (output_dir / "result.json").write_text(
                    json.dumps({"children": [{"type": "paragraph", "content": "CamScanner"}]}),
                    encoding="utf-8",
                )

                class Result:
                    returncode = 0
                    stdout_text = ""
                    stderr_text = ""

                return Result()

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "scan.pdf"
            source.write_bytes(b"%PDF scanner-watermark fixture")
            parser = OpenDataLoaderPdfParser(
                temp_root=root / "parser-temp",
                runner=Runner(),
                settings=OpenDataLoaderSettings(hybrid="off"),
            )

            with self.assertRaisesRegex(OpenDataLoaderNoElementsError, "no useful"):
                await parser.parse(ParseRequest(source, source.name, "application/pdf", SOURCE_HASH))

    async def test_docling_pdf_output_fingerprint_includes_local_tesseract(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "scan.pdf"
            path.write_bytes(b"%PDF synthetic scan")
            parser = DoclingParser()
            request = ParseRequest(
                path,
                path.name,
                "application/pdf",
                SOURCE_HASH,
                parser_chain=("opendataloader:test:result=empty",),
            )
            with (
                patch.object(parser, "_ocr_fingerprint", return_value="ocr:tesseract-cli:test:lang=eng"),
                patch.object(
                    parser,
                    "_convert",
                    return_value={
                        "texts": [
                            {
                                "self_ref": "#/texts/0",
                                "label": "paragraph",
                                "text": "Text recovered from the scanned page",
                                "prov": [{"page_no": 1}],
                            }
                        ],
                        "body": {"children": [{"$ref": "#/texts/0"}]},
                    },
                ),
            ):
                document = await parser.parse(request)

        self.assertEqual(document.elements[0].provenance[0].page_number, 1)
        self.assertIn("opendataloader:test:result=empty", document.parser_fingerprint)
        self.assertIn("docling:2.112.0:targeted:local", document.parser_fingerprint)
        self.assertIn("ocr:tesseract-cli:test:lang=eng:psm=6", document.parser_fingerprint)

    def test_opendataloader_mapping_keeps_page_bbox_and_heading(self):
        payload = {
            "children": [
                {
                    "id": "h1",
                    "type": "heading",
                    "heading_level": 1,
                    "content": "Overview",
                    "page_number": 1,
                    "bounding_box": [10, 20, 110, 80],
                },
                {
                    "id": "p1",
                    "type": "paragraph",
                    "content": "Body text",
                    "page_number": 2,
                },
            ]
        }
        elements = odl_json_to_elements(payload, SOURCE_HASH, "odl:test")
        self.assertEqual(
            [item.element_type for item in elements],
            [
                ElementType.HEADING,
                ElementType.PARAGRAPH,
            ],
        )
        self.assertEqual(elements[0].provenance[0].bbox.left, 10)
        self.assertEqual(elements[1].provenance[0].page_number, 2)
        self.assertEqual(elements[1].provenance[0].section_path, ("Overview",))

    def test_opendataloader_treats_a_lone_scanner_watermark_as_no_useful_text(self):
        watermark = odl_json_to_elements(
            {"children": [{"type": "paragraph", "content": "CamScanner"}]},
            SOURCE_HASH,
            "odl:test",
        )
        useful = odl_json_to_elements(
            {
                "children": [
                    {"type": "heading", "content": "Purchase order"},
                    {"type": "paragraph", "content": "CamScanner"},
                ]
            },
            SOURCE_HASH,
            "odl:test",
        )

        self.assertTrue(_only_scanner_watermark_text(watermark))
        self.assertFalse(_only_scanner_watermark_text(useful))

    def test_opendataloader_table_rows_keep_header_with_values(self):
        payload = {
            "children": [
                {
                    "id": "t1",
                    "type": "table",
                    "page_number": 4,
                    "bounding_box": [10, 20, 400, 200],
                    "number of rows": 3,
                    "number of columns": 3,
                    "rows": [
                        {
                            "type": "table row",
                            "row number": 1,
                            "cells": [
                                {"type": "table cell", "column number": 1, "column span": 1,
                                 "page number": 4, "kids": [{"type": "paragraph", "text": "Round"}]},
                                {"type": "table cell", "column number": 2, "column span": 1,
                                 "page number": 4, "kids": [{"type": "paragraph", "text": "Amount"}]},
                                {"type": "table cell", "column number": 3, "column span": 1,
                                 "page number": 4, "kids": [{"type": "paragraph", "text": "Lead"}]},
                            ],
                        },
                        {
                            "type": "table row",
                            "row number": 2,
                            "cells": [
                                {"type": "table cell", "column number": 1, "column span": 1,
                                 "page number": 4, "kids": [{"type": "paragraph", "text": "Seed-Q"}]},
                                {"type": "table cell", "column number": 2, "column span": 1,
                                 "page number": 4, "kids": [{"type": "paragraph", "text": "$11.11M"}]},
                                {"type": "table cell", "column number": 3, "column span": 1,
                                 "page number": 4, "kids": [{"type": "paragraph", "text": "BlueFinch"}]},
                            ],
                        },
                        {
                            "type": "table row",
                            "row number": 3,
                            "cells": [
                                {"type": "table cell", "column number": 1, "column span": 1,
                                 "page number": 4, "kids": [{"type": "paragraph", "text": "Alpha-X"}]},
                                {"type": "table cell", "column number": 2, "column span": 1,
                                 "page number": 4, "kids": [{"type": "paragraph", "text": "$22.22M"}]},
                                {"type": "table cell", "column number": 3, "column span": 1,
                                 "page number": 4, "kids": [{"type": "paragraph", "text": "RedKite"}]},
                            ],
                        },
                    ],
                }
            ]
        }
        elements = odl_json_to_elements(payload, SOURCE_HASH, "odl:test")
        self.assertEqual(len(elements), 2)
        self.assertTrue(all(item.element_type is ElementType.TABLE for item in elements))
        self.assertEqual(
            elements[0].text, "Round: Seed-Q | Amount: $11.11M | Lead: BlueFinch"
        )
        self.assertEqual(
            elements[1].text, "Round: Alpha-X | Amount: $22.22M | Lead: RedKite"
        )
        self.assertEqual(elements[0].provenance[0].page_number, 4)
        self.assertEqual(elements[0].metadata["table_row"], 2)

    def test_opendataloader_table_merged_header_and_empty_cells(self):
        payload = {
            "children": [
                {
                    "id": "t2",
                    "type": "table",
                    "page number": 5,
                    "rows": [
                        {
                            "row number": 1,
                            "cells": [
                                {"column number": 1, "column span": 1, "page number": 5,
                                 "kids": [{"type": "paragraph", "text": "Round"}]},
                                {"column number": 2, "column span": 2, "page number": 5,
                                 "kids": [{"type": "paragraph", "text": "Amount"}]},
                            ],
                        },
                        {
                            "row number": 2,
                            "cells": [
                                {"column number": 1, "column span": 1, "page number": 5,
                                 "kids": [{"type": "paragraph", "text": "Gamma-W"}]},
                                {"column number": 2, "column span": 1, "page number": 5,
                                 "kids": [{"type": "paragraph", "text": "$44.44M"}]},
                                {"column number": 3, "column span": 1, "page number": 5,
                                 "kids": []},
                            ],
                        },
                    ],
                }
            ]
        }
        elements = odl_json_to_elements(payload, SOURCE_HASH, "odl:test")
        self.assertEqual(len(elements), 1)
        # merged header covers both amount columns; the empty cell is skipped
        self.assertEqual(elements[0].text, "Round: Gamma-W | Amount: $44.44M")

    def test_opendataloader_table_rows_keep_their_own_pages(self):
        payload = {
            "children": [
                {
                    "id": "t3",
                    "type": "table",
                    "page_number": 6,
                    "rows": [
                        {
                            "row number": 1,
                            "cells": [
                                {"column number": 1, "page number": 6,
                                 "kids": [{"type": "paragraph", "text": "Year"}]},
                                {"column number": 2, "page number": 6,
                                 "kids": [{"type": "paragraph", "text": "Revenue"}]},
                            ],
                        },
                        {
                            "row number": 2,
                            "cells": [
                                {"column number": 1, "page number": 6,
                                 "kids": [{"type": "paragraph", "text": "2040"}]},
                                {"column number": 2, "page number": 6,
                                 "kids": [{"type": "paragraph", "text": "$1.00M"}]},
                            ],
                        },
                        {
                            "row number": 3,
                            "cells": [
                                {"column number": 1, "page number": 7,
                                 "kids": [{"type": "paragraph", "text": "2041"}]},
                                {"column number": 2, "page number": 7,
                                 "kids": [{"type": "paragraph", "text": "$2.00M"}]},
                            ],
                        },
                    ],
                }
            ]
        }
        elements = odl_json_to_elements(payload, SOURCE_HASH, "odl:test")
        self.assertEqual(len(elements), 2)
        self.assertEqual(elements[0].provenance[0].page_number, 6)
        self.assertEqual(elements[1].provenance[0].page_number, 7)
        self.assertIn("Year: 2041", elements[1].text)

    def test_opendataloader_empty_table_warns_but_keeps_surrounding_text(self):
        payload = {
            "children": [
                {"type": "paragraph", "content": "Before", "page_number": 1},
                {"id": "t4", "type": "table", "page_number": 2, "rows": [
                    {"row number": 1, "cells": [
                        {"column number": 1, "page number": 2, "kids": []},
                        {"column number": 2, "page number": 2, "kids": []},
                    ]},
                ]},
                {"type": "paragraph", "content": "After", "page_number": 3},
            ]
        }
        with self.assertLogs("app.ingestion.parsers.opendataloader", level="WARNING") as logs:
            elements = odl_json_to_elements(payload, SOURCE_HASH, "odl:test")
        self.assertEqual([item.text for item in elements], ["Before", "After"])
        self.assertTrue(any("yielded no text" in line for line in logs.output))

    def test_opendataloader_table_ragged_row_gets_column_fallback(self):
        payload = {
            "children": [
                {"id": "t6", "type": "table", "page_number": 9, "rows": [
                    {"row number": 1, "cells": [
                        {"column number": 1, "page number": 9,
                         "kids": [{"type": "paragraph", "text": "Metric"}]},
                    ]},
                    {"row number": 2, "cells": [
                        {"column number": 1, "page number": 9,
                         "kids": [{"type": "paragraph", "text": "Uptime"}]},
                        {"column number": 2, "page number": 9,
                         "kids": [{"type": "paragraph", "text": "extra"}]},
                    ]},
                ]}
            ]
        }
        elements = odl_json_to_elements(payload, SOURCE_HASH, "odl:test")
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].text, "Metric: Uptime | Column 2: extra")

    def test_opendataloader_warns_once_for_pages_with_zero_text_units(self):
        payload = {
            "number of pages": 3,
            "children": [
                {"type": "paragraph", "content": "First", "page_number": 1},
                {"type": "list", "page_number": 2},
                {"type": "paragraph", "content": "Third", "page_number": 3},
            ],
        }
        with self.assertLogs("app.ingestion.parsers.opendataloader", level="WARNING") as logs:
            elements = odl_json_to_elements(payload, SOURCE_HASH, "odl:test")
        self.assertEqual([item.text for item in elements], ["First", "Third"])
        warnings = [line for line in logs.output if "zero text units" in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn("[2]", warnings[0])

    def test_opendataloader_table_never_stringifies_structure(self):
        payload = {
            "children": [
                {"id": "t5", "type": "table", "page_number": 8, "rows": [
                    {"row number": 1, "cells": [
                        {"column number": 1, "page number": 8,
                         "kids": [{"type": "paragraph", "text": "Metric"}]},
                        {"column number": 2, "page number": 8,
                         "kids": [{"type": "paragraph", "text": "Value"}]},
                    ]},
                    {"row number": 2, "cells": [
                                {"column number": 1, "page number": 8,
                                 "kids": [{"type": "paragraph",
                                           "kids": [{"type": "text", "text": "Uptime"}]}]},
                        {"column number": 2, "page number": 8,
                         "kids": [{"type": "paragraph", "text": "99.9%"}]},
                    ]},
                ]}
            ]
        }
        elements = odl_json_to_elements(payload, SOURCE_HASH, "odl:test")
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].text, "Metric: Uptime | Value: 99.9%")
        self.assertNotIn("{", elements[0].text)

    def test_docling_mapping_honors_body_order_tables_and_slides(self):
        payload = {
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "label": "paragraph",
                    "text": "After table",
                    "prov": [{"page_no": 3}],
                }
            ],
            "tables": [
                {
                    "self_ref": "#/tables/0",
                    "label": "table",
                    "data": {
                        "table_cells": [
                            {"row": 0, "column": 0, "text": "A"},
                            {"row": 0, "column": 1, "text": "B"},
                        ]
                    },
                    "prov": [{"page_no": 2}],
                }
            ],
            "body": {
                "children": [
                    {"$ref": "#/tables/0"},
                    {"$ref": "#/texts/0"},
                ]
            },
        }
        elements = docling_dict_to_elements(payload, SOURCE_HASH, "docling:test", source_suffix=".pptx")
        self.assertEqual(elements[0].element_type, ElementType.TABLE)
        self.assertEqual(elements[0].text, "A | B")
        self.assertEqual(elements[0].provenance[0].slide_number, 2)
        self.assertIsNone(elements[0].provenance[0].page_number)
        self.assertEqual(elements[1].text, "After table")


if __name__ == "__main__":
    unittest.main()
