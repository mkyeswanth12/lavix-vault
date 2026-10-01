"""Type-aware routing inside CanonicalChunker (OG SmartChunker port)."""


from app.ingestion.chunking import CanonicalChunker, ChunkingPolicy
from app.ingestion.models import (
    CanonicalDocument,
    CanonicalElement,
    ElementType,
    Provenance,
)


def _el(i, element_type, text, **kwargs):
    return CanonicalElement(
        element_id=f"el{i}",
        element_type=element_type,
        text=text,
        provenance=(Provenance(page_number=1),),
        **kwargs,
    )


def _doc(name, elements, media="text/plain"):
    return CanonicalDocument(
        source_name=name,
        source_sha256="a" * 64,
        media_type=media,
        parser_fingerprint="test",
        elements=tuple(elements),
    )


def _headed_elements():
    return [
        _el(0, ElementType.TITLE, "Intro", heading_level=1),
        _el(1, ElementType.PARAGRAPH, "Body text here. " * 30),
        _el(2, ElementType.HEADING, "Next", heading_level=1),
        _el(3, ElementType.PARAGRAPH, "More body. " * 30),
    ]


def test_fingerprint_is_v3():
    assert ChunkingPolicy().fingerprint.startswith("canonical-v3:")


def test_shape_from_extension():
    chunker = CanonicalChunker()
    assert chunker._detect_shape(_doc("s.py", [_el(0, ElementType.PARAGRAPH, "hello")])) == "code"
    assert chunker._detect_shape(_doc("d.md", [_el(0, ElementType.PARAGRAPH, "hello")])) == "headed"
    assert chunker._detect_shape(_doc("n.txt", [_el(0, ElementType.PARAGRAPH, "hello")])) == "prose"


def test_shape_from_code_fraction():
    chunker = CanonicalChunker()
    doc = _doc(
        "n.txt",
        [
            _el(0, ElementType.CODE, "x = 1\n" * 100),
            _el(1, ElementType.PARAGRAPH, "short note"),
        ],
    )
    assert chunker._detect_shape(doc) == "code"


def test_shape_from_headings_and_markdown_regex():
    chunker = CanonicalChunker()
    doc = _doc("n.txt", _headed_elements())
    assert chunker._detect_shape(doc) == "headed"
    md_text = "# A\n\n- one\n- two\n- three\n\n[link](http://x)\n\n```\ncode\n```\n"
    doc2 = _doc("n.txt", [_el(0, ElementType.PARAGRAPH, md_text)])
    assert chunker._detect_shape(doc2) == "headed"


def test_headed_sections_stay_whole():
    chunks = CanonicalChunker().chunk(_doc("d.md", _headed_elements(), "text/markdown"))
    assert len(chunks) == 2
    assert "Intro" in chunks[0].text
    assert "Next" in chunks[1].text
    assert chunks[0].provenance and chunks[1].provenance


def test_code_units_separated_from_prose():
    chunks = CanonicalChunker().chunk(
        _doc(
            "s.py",
            [
                _el(0, ElementType.PARAGRAPH, "Module docs. " * 20),
                _el(1, ElementType.CODE, "def foo():\n    return 1\n" * 20),
                _el(2, ElementType.CODE, "def bar():\n    return 2\n" * 20),
            ],
            "text/x-python",
        )
    )
    # OG strict: every code block is its own chunk, prose stays apart.
    assert len(chunks) == 3
    assert "Module docs" in chunks[0].text
    assert "def foo" in chunks[1].text and "def bar" not in chunks[1].text
    assert "def bar" in chunks[2].text
    assert [c.chunk_type for c in chunks] == ["paragraph", "code_block", "code_block"]


def test_headed_section_stays_whole_past_target():
    paras = [_el(0, ElementType.TITLE, "Big section", heading_level=1)]
    paras += [_el(i + 1, ElementType.PARAGRAPH, f"Section body {i}. " * 30) for i in range(4)]
    chunks = CanonicalChunker().chunk(_doc("d.md", paras, "text/markdown"))
    # OG markdown: one chunk per section unless max forces a cut.
    # 4 paras x ~150 tokens = ~600: past the 450 target, under the 700 max.
    assert chunks[0].token_count > 450
    assert len(chunks) == 1
    assert "Big section" in chunks[0].text
    assert chunks[0].token_count <= 700
    assert chunks[0].chunk_type == "mixed"


def test_oversize_prose_splits_on_sentence_boundaries():
    text = "".join(f"Sentence number {i} ends here. " for i in range(200))
    chunks = CanonicalChunker().chunk(
        _doc("n.txt", [_el(0, ElementType.PARAGRAPH, text)])
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 700
        assert not chunk.text.rstrip().endswith((" and", " the", " or"))


def test_tiny_document_is_single():
    chunks = CanonicalChunker().chunk(
        _doc("n.txt", [_el(0, ElementType.PARAGRAPH, "hello world")])
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "single"


def test_prose_grouping_unchanged():
    paras = [_el(i, ElementType.PARAGRAPH, f"Para {i}. " * 40) for i in range(4)]
    chunks = CanonicalChunker().chunk(_doc("n.txt", paras))
    assert len(chunks) == 1
    assert chunks[0].token_count <= 700
    assert all(chunks[0].chunk_id for _ in [0])


def test_oversize_table_splits_by_rows_with_header_repeated():
    header = "Round: r | Amount: a | Lead: l"
    rows = "\n".join(f"Round: zeta-{i} | Amount: ${i}.00M | Lead: Finch-{i}" for i in range(60))
    chunks = CanonicalChunker().chunk(_doc("d.pdf", [_el(0, ElementType.TABLE, header + "\n" + rows)]))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 700
        assert chunk.chunk_type == "table"
        lines = [line for line in chunk.text.splitlines() if line.strip()]
        # every piece re-opens with the header, and no row is cut mid-row
        assert lines[0] == header
        for line in lines[1:]:
            assert line.startswith("Round: zeta-")
