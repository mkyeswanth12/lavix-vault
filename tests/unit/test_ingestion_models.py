import unittest

from app.ingestion.chunking import CanonicalChunker, ChunkingPolicy
from app.ingestion.models import (
    CanonicalDocument,
    CanonicalElement,
    ElementType,
    JobState,
    Provenance,
    build_element_id,
    validate_job_transition,
)
from app.ingestion.routing import ParserKind, route_file

SOURCE_HASH = "a" * 64
PARSER = "parser:test"


def element(ordinal: int, text: str, *, page: int, kind=ElementType.PARAGRAPH):
    provenance = (Provenance(page_number=page),)
    return CanonicalElement(
        element_id=build_element_id(
            source_sha256=SOURCE_HASH,
            parser_fingerprint=PARSER,
            element_type=kind,
            text=text,
            provenance=provenance,
            ordinal=ordinal,
        ),
        element_type=kind,
        text=text,
        provenance=provenance,
    )


class IngestionModelTests(unittest.TestCase):
    def test_job_state_machine_rejects_skips(self):
        validate_job_transition(JobState.DECRYPTING, JobState.PARSING)
        validate_job_transition(JobState.PUBLISHING, JobState.READY)
        with self.assertRaises(ValueError):
            validate_job_transition(JobState.DECRYPTING, JobState.READY)
        with self.assertRaises(ValueError):
            validate_job_transition(JobState.READY, JobState.QUEUED)

    def test_routing_uses_media_type_before_extension(self):
        cases = (
            ("report.pdf", "application/pdf", ParserKind.OPEN_DATALOADER_PDF),
            ("report.docx", "application/octet-stream", ParserKind.DOCLING),
            ("legacy.doc", "application/msword", ParserKind.LIBREOFFICE_DOCLING),
            ("notes.txt", "text/plain", ParserKind.TEXT),
            ("config.yml", "text/plain", ParserKind.TEXT),
            ("payload.json", "application/json", ParserKind.TEXT),
            ("analysis.py", "text/x-python", ParserKind.TEXT),
            ("table.csv", "text/plain", ParserKind.DOCLING),
            ("scan.tiff", "image/tiff", ParserKind.DOCLING),
            ("fake.pdf", "video/mp4", ParserKind.UNSUPPORTED),
            ("vector.svg", "image/svg+xml", ParserKind.UNSUPPORTED),
        )
        for name, media_type, expected in cases:
            with self.subTest(name=name, media_type=media_type):
                self.assertIs(route_file(name, media_type).parser, expected)

        jpeg = route_file("misleading.bin", "image/jpeg")
        self.assertIs(jpeg.parser, ParserKind.DOCLING)
        self.assertEqual(jpeg.normalized_extension, ".jpg")
        docx = route_file(
            "misleading.bin",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertIs(docx.parser, ParserKind.DOCLING)
        self.assertEqual(docx.normalized_extension, ".docx")

    def test_cfb_container_mimes_fall_back_to_extension_for_password_diagnostic(self):
        # Password-protected OOXML and some legacy Office saves sniff as CFB
        # container MIMEs. The extension must still route the file so
        # LibreOffice can raise its typed password error instead of a
        # misleading terminal "unsupported".
        for sniffed in (
            "application/x-ole-storage",
            "application/x-cfb",
            "application/cdfv2",
            "application/cdfv2-corrupt",
            "application/vnd.ms-office",
        ):
            for name, expected in (
                ("locked.docx", ParserKind.DOCLING),
                ("locked.xlsx", ParserKind.DOCLING),
                ("legacy.doc", ParserKind.LIBREOFFICE_DOCLING),
            ):
                with self.subTest(sniffed=sniffed, name=name):
                    route = route_file(name, sniffed)
                    self.assertIs(route.parser, expected)

        # Without any Office extension the CFB blob stays unsupported.
        self.assertIs(route_file("mystery.bin", "application/x-ole-storage").parser, ParserKind.UNSUPPORTED)

    def test_chunk_ids_are_stable_and_page_boundaries_are_preserved(self):
        document = CanonicalDocument(
            source_name="report.pdf",
            source_sha256=SOURCE_HASH,
            media_type="application/pdf",
            parser_fingerprint=PARSER,
            elements=(
                element(0, "First page paragraph.", page=1),
                element(1, "Second page paragraph.", page=2),
            ),
        )
        chunker = CanonicalChunker(ChunkingPolicy(target_tokens=100, max_tokens=120, overlap_tokens=10))
        first = chunker.chunk(document)
        second = chunker.chunk(document)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0].provenance[0].page_number, 1)
        self.assertEqual(first[1].provenance[0].page_number, 2)
        self.assertRegex(first[0].chunk_id, r"^chk_[0-9a-f]{64}$")
        self.assertNotIn("First page", first[1].embedding_text)

    def test_element_identity_changes_with_content(self):
        before = element(0, "alpha", page=1)
        after = element(0, "beta", page=1)
        self.assertNotEqual(before.element_id, after.element_id)

    def test_oversized_table_rows_respect_hard_chunk_limit(self):
        table = element(
            0,
            "header\n" + " ".join(f"value-{index}" for index in range(100)),
            page=1,
            kind=ElementType.TABLE,
        )
        document = CanonicalDocument(
            source_name="sheet.xlsx",
            source_sha256=SOURCE_HASH,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            parser_fingerprint=PARSER,
            elements=(table,),
        )
        chunker = CanonicalChunker(ChunkingPolicy(target_tokens=20, max_tokens=25, overlap_tokens=3))
        chunks = chunker.chunk(document)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 25 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
