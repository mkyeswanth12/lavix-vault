from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest

from app.graph_memory import inference_priority
from app.graph_memory.inference_priority import BackgroundInferenceDeferred
from app.ingestion.errors import IngestionCancelled
from app.ingestion.intelligence import (
    DOCUMENT_TYPES,
    ConfiguredDocumentIntelligenceService,
    DocumentIntelligenceService,
    IntelligenceSettings,
    deterministic_intelligence,
    deterministic_tags,
    normalize_tags,
    representative_text,
)
from app.ingestion.models import CanonicalChunk, CanonicalDocument, Provenance


def _document(
    name: str = "tax-invoice.pdf",
    media_type: str = "application/pdf",
    parser_fingerprint: str = "test:parser",
) -> CanonicalDocument:
    return CanonicalDocument(
        source_name=name,
        source_sha256="a" * 64,
        media_type=media_type,
        parser_fingerprint=parser_fingerprint,
        elements=(),
    )


def _chunk(ordinal: int, text: str) -> CanonicalChunk:
    return CanonicalChunk(
        chunk_id=f"chk_{ordinal:064x}",
        ordinal=ordinal,
        text=text,
        embedding_text=text,
        element_ids=(f"el_{ordinal:064x}",),
        provenance=(Provenance(page_number=ordinal + 1),),
        token_count=max(1, len(text.split())),
    )


def test_deterministic_fallback_is_meaningful_and_controlled() -> None:
    chunks = (
        _chunk(0, "Tax Invoice. Invoice number 1042. Bill to Example Company."),
        _chunk(1, "Consulting services total amount due is 1250 dollars."),
    )
    result = deterministic_intelligence(_document(), chunks)

    assert result.doc_type == "invoice"
    assert result.doc_type in DOCUMENT_TYPES
    assert result.status == "fallback"
    assert "Invoice number 1042" in result.summary
    assert result.tags == ()


def test_deterministic_fallback_never_expands_past_three_complete_sentences() -> None:
    result = deterministic_intelligence(
        _document("fragmented-notes.txt", "text/plain"),
        (
            _chunk(
                0,
                "One. Two. Three. The fourth sentence contains the useful project context. "
                "The fifth sentence records the implementation decision. "
                "The sixth sentence records the validation result.",
            ),
        ),
    )

    sentences = [
        part
        for part in re.split(r"(?<=[.!?])\s+", result.summary)
        if part
    ]
    assert 1 <= len(sentences) <= 3
    assert len(result.summary) <= 700
    assert "fourth sentence" in result.summary


def test_deterministic_fallback_builds_grounded_repeatable_semantic_tags() -> None:
    chunks = (
        _chunk(
            0,
            "Network security controls protect cloud workloads. "
            "Network security controls require access reviews.",
        ),
        _chunk(
            1,
            "Cloud workloads use encryption. Network security controls are audited.",
        ),
    )

    result = deterministic_intelligence(_document("security-notes.txt", "text/plain"), chunks)

    assert "network security controls" in result.tags
    assert all(tag in " ".join(chunk.text for chunk in chunks).casefold() for tag in result.tags)
    assert "document" not in result.tags


def test_deterministic_tags_count_repeated_office_table_cells_independently() -> None:
    evidence = (
        "Department Name | Clearance Item Name | Clearance Incharge Name\n"
        "Dues Clearance | Dues Clearance | sandra.h; Ravi_P; karthik_k36; bojamma.a01"
    )

    result = deterministic_intelligence(
        _document("PendingClearanceItem-1237431.xls", "application/vnd.ms-excel"),
        (_chunk(0, evidence),),
    )

    assert result.tags == ("clearance",)
    assert not {"sandra", "ravi", "karthik", "bojamma", "department name"}.intersection(
        result.tags
    )


def test_sparse_two_row_table_uses_its_only_grounded_data_topic() -> None:
    evidence = "project | code\nCSV proof | CSV-8192"

    result = deterministic_intelligence(
        _document("format-proof.csv", "text/csv"),
        (_chunk(0, evidence),),
    )

    assert result.tags == ("proof",)
    assert not {"csv", "project", "code", "8192"}.intersection(result.tags)


def test_short_raster_diagram_keeps_one_grounded_multiword_label() -> None:
    evidence = (
        "User\n\nUser\n\nmetadata\n\nAgent\n\nDatabase\n\nTool\n\nFetch\n\n"
        "event/movie\n\ndata\n\nRAG\n\nTool\n\nResponse"
    )

    result = deterministic_intelligence(
        _document(
            "case_4.tiff",
            "image/tiff",
            "docling:2.112.0:targeted:local+ocr:tesseract-cli:tesseract_5.5.2",
        ),
        (_chunk(0, evidence),),
    )

    assert result.tags == ("event movie",)


def test_sparse_vision_subject_omits_document_type_and_people() -> None:
    evidence = (
        "Document type: Tax Invoice\n"
        "Issuing Organization: HR HOMES\n"
        "Subject: Purchase of bags by Mr. Rajesh Maheshwari and Mrs. Nancy Jain."
    )

    result = deterministic_intelligence(
        _document(
            "1715773827.webp",
            "image/webp",
            "ollama-vision:v8:qwen2.5vl:3b-q6-16k",
        ),
        (_chunk(0, evidence),),
    )

    assert result.tags == ("purchase of bags",)
    assert not {"tax invoice", "hr homes", "rajesh maheshwari", "nancy jain"}.intersection(
        result.tags
    )


@pytest.mark.asyncio
async def test_empty_model_tags_keep_grounded_sparse_vision_subject(monkeypatch) -> None:
    evidence = (
        "Document type: Tax Invoice\n"
        "Issuing Organization: HR HOMES\n"
        "Subject: Purchase of bags by Mr. Rajesh Maheshwari and Mrs. Nancy Jain."
    )
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "invoice",
            "summary": "The Tax Invoice concerns the purchase of bags from HR HOMES.",
            "tags": [],
        },
    )

    result = await service.analyze(
        _document(
            "1715773827.webp",
            "image/webp",
            "ollama-vision:v8:qwen2.5vl:3b-q6-16k",
        ),
        (_chunk(0, evidence),),
    )

    assert result.status == "model"
    assert result.tags == ("purchase of bags",)


@pytest.mark.asyncio
async def test_grounded_model_summary_recovers_repeated_ocr_topic(monkeypatch) -> None:
    evidence = (
        "HR HOMES\n\nTAX INVOICE\n\nWarranty related Terms & conditions\n\n"
        "An Invoice Must accompany products returned for warranty.\n\n"
        "Goods damaged During transit voids warranty.\n\n"
        "90 days limited warranty unless otherwise stated.\n\n"
        "All items carry MFG Warranty only No return or exchange."
    )
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "invoice",
            "summary": (
                "This tax invoice covers goods purchased from HR HOMES and includes "
                "product warranty terms."
            ),
            "tags": ["product_details", "warranty_terms"],
        },
    )

    result = await service.analyze(
        _document(
            "1715773827.webp",
            "image/webp",
            "docling:2.112.0:targeted:local+ocr:tesseract-cli:tesseract_5.5.2",
        ),
        (_chunk(0, evidence),),
    )

    assert result.status == "model"
    assert result.tags == ("warranty",)


@pytest.mark.asyncio
async def test_rejected_model_table_headers_do_not_erase_grounded_fallback(monkeypatch) -> None:
    evidence = (
        "Department Name | Clearance Item Name | Clearance Incharge Name\n"
        "Dues Clearance | Dues Clearance | sandra.h; Ravi_P; karthik_k36; bojamma.a01"
    )
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "spreadsheet",
            "summary": "Dues Clearance is the listed clearance item for the department.",
            "tags": ["department name", "clearance item name", "incharge names"],
        },
    )

    result = await service.analyze(
        _document("PendingClearanceItem-1237431.xls", "application/vnd.ms-excel"),
        (_chunk(0, evidence),),
    )

    assert result.tags == ("clearance",)


def test_deterministic_tags_require_topical_summary_support_and_reject_ocr_filler() -> None:
    text = (
        "The features are for the user and tool. Dep artm agers exponen ally. "
        "The features were for the user and tool. Dep artm agers exponen ally. "
        "The features are for the user and tool. Dep artm agers exponen ally. "
        "The psychology of money explains behavior. "
        "The psychology of money shapes decisions. "
        "The psychology of money affects saving."
    )

    tags = deterministic_tags(
        text,
        source_name="book.pdf",
        parser_fingerprint="text:v1",
        summary=("The psychology of money explains how money shapes financial behavior and saving."),
    )

    assert tags == ("psychology of money",)
    assert not {"the", "and", "was", "for", "user", "tool", "dep artm", "agers"}.intersection(tags)


def test_deterministic_tags_keep_one_complete_entity_not_overlapping_fragments() -> None:
    evidence = (
        "Amazon Seller Services Private Limited issues marketplace invoices. "
        "Amazon Seller Services Private Limited processes seller payments. "
        "Amazon Seller Services Private Limited operates the marketplace."
    )

    tags = deterministic_tags(
        evidence,
        source_name="invoice.pdf",
        parser_fingerprint="text:v1",
        summary=(
            "Amazon Seller Services Private Limited issues marketplace invoices and processes "
            "seller payments."
        ),
    )

    assert tags == ("amazon seller services private limited",)


def test_deterministic_tags_fail_closed_for_pervasively_split_text_layers() -> None:
    fragmented = (
        "M K Yes w anth E ngineering M a n age m ent. "
        "Organizations are syste m s created to achieve co m m on go als. "
        "People-to-w ork relatio nships are linked to the extern al environ m ent."
    )

    assert (
        deterministic_tags(
            fragmented,
            source_name="course.pdf",
            parser_fingerprint="pdf:text:v1",
            summary=fragmented,
        )
        == ()
    )


def test_tag_contract_rejects_filler_addresses_and_repeated_ocr_tokens() -> None:
    evidence = (
        "the and was you for features user tool need for 1st Main Road "
        "10c 10c whiterose dep artm exponen ally money Tel Aviv cloud security"
    )

    assert normalize_tags(
        [
            "the",
            "and",
            "was",
            "you",
            "for",
            "features",
            "user",
            "tool",
            "need for",
            "1st Main Road",
            "10c 10c whiterose",
            "dep artm",
            "exponen ally",
            "money",
            "tel aviv",
            "cloud-security",
        ],
        evidence_text=evidence,
        source_name="topics.pdf",
    ) == ("money", "tel aviv", "cloud security")


def test_tag_contract_keeps_company_entity_but_not_address_or_name_fragments() -> None:
    evidence = (
        "Sold By: Amazon Seller Services Private Limited. "
        "Billing Address: MK yeswanth friends hotspot PG, 10c, 10c, whiterose layout, "
        "1st Main Road."
    )

    assert normalize_tags(
        [
            "amazon seller services private limited",
            "seller services private",
            "services private limited",
            "yeswanth friends hotspot",
            "10c 10c whiterose",
            "1st main road",
        ],
        evidence_text=evidence,
        source_name="invoice.pdf",
    ) == ("amazon seller services private limited",)


def test_tag_contract_rejects_long_flattened_address_values_and_form_labels() -> None:
    evidence = (
        "Sold By:\nAmazon Seller Services Private Limited.\n"
        "Billing Address:\nMK Yeswanth Friends Hotspot PG, 10c, Whiterose Layout,\n"
        "1st Main Road, Pattandur Agrahara 2nd, Bengaluru, Karnataka, 560066 IN\n"
        "State/UT Code:\n29.\nPlace of delivery:\nKarnataka.\n"
        "Registered Office: 8th Floor, Dr Rajkumar Road, Malleshwaram West, 560055."
    )

    assert normalize_tags(
        [
            "amazon seller services private limited",
            "mk yeswanth friends hotspot pg",
            "pattandur agrahara",
            "bengaluru karnataka",
            "state ut",
            "ut code",
            "order date",
            "main",
            "karnataka",
            "malleshwaram west",
            "billing address",
            "date",
            "code",
        ],
        evidence_text=evidence,
        source_name="invoice.pdf",
    ) == ("amazon seller services private limited",)


def test_existing_tag_cleanup_preserves_vlm_topics_and_short_acronyms() -> None:
    assert normalize_tags(
        ["wet street", "headlights on", "vcp dcv", "gas supply", "us", "the", "tool"],
        source_name="document.pdf",
        preserve_existing=True,
    ) == ("wet street", "headlights on", "vcp dcv", "gas supply", "us")


def test_existing_tag_cleanup_keeps_broad_and_specific_search_facets() -> None:
    assert normalize_tags(
        ["software engineer", "junior software engineer"],
        source_name="resume.pdf",
        preserve_existing=True,
    ) == ("software engineer", "junior software engineer")

    assert normalize_tags(
        ["unstructured data"],
        source_name="assistant.docx",
        preserve_existing=True,
    ) == ("unstructured data",)


def test_model_tag_cleanup_rejects_broad_ai_but_keeps_named_protocol() -> None:
    evidence = (
        "The Agent2Agent Protocol enables decentralized communication between AI agents. "
        "Dynamic discovery connects compatible agents."
    )

    assert normalize_tags(
        ["AI", "Agent2Agent Protocol", "Dynamic discovery"],
        evidence_text=evidence,
        source_name="Agent2Agent (A2A) Protocol.pptx",
    ) == ("agent2agent protocol", "dynamic discovery")


@pytest.mark.asyncio
async def test_configured_intelligence_honours_disable_without_calling_model(monkeypatch) -> None:
    service = ConfiguredDocumentIntelligenceService(
        lambda: (False, "should-not-run:latest"),
        IntelligenceSettings(model="environment:latest"),
    )
    monkeypatch.setattr(
        DocumentIntelligenceService,
        "_post",
        lambda *_args: pytest.fail("disabled intelligence must not call Ollama"),
    )

    result = await service.analyze(
        _document(),
        (_chunk(0, "Tax Invoice. Invoice number 1042."),),
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "intelligence_disabled"


@pytest.mark.asyncio
async def test_configured_intelligence_uses_current_role_model(monkeypatch) -> None:
    current_model = ["first:latest"]
    observed: list[str] = []

    def fake_post(service, _document, _chunks):
        observed.append(service.settings.model)
        return {
            "doc_type": "invoice",
            "summary": "Tax invoice 1042 records consulting services billed to Example Company.",
            "tags": ["consulting services"],
        }

    monkeypatch.setattr(DocumentIntelligenceService, "_post", fake_post)
    service = ConfiguredDocumentIntelligenceService(
        lambda: (True, current_model[0]),
        IntelligenceSettings(model="environment:latest"),
    )
    chunks = (
        _chunk(
            0,
            "Tax invoice 1042 records consulting services billed to Example Company.",
        ),
    )

    await service.analyze(_document(), chunks)
    current_model[0] = "second:latest"
    await service.analyze(_document(), chunks)

    assert observed == ["first:latest", "second:latest"]


@pytest.mark.asyncio
async def test_configured_intelligence_discards_output_from_replaced_role(monkeypatch) -> None:
    roles = iter(((True, "first:latest"), (True, "second:latest")))
    monkeypatch.setattr(
        DocumentIntelligenceService,
        "_post",
        lambda *_args: {
            "doc_type": "invoice",
            "summary": "Tax invoice 1042 records consulting services billed to Example Company.",
            "tags": ["consulting services"],
        },
    )
    service = ConfiguredDocumentIntelligenceService(
        lambda: next(roles),
        IntelligenceSettings(model="environment:latest"),
    )

    with pytest.raises(BackgroundInferenceDeferred, match="role changed"):
        await service.analyze(
            _document(),
            (_chunk(0, "Tax invoice 1042 records consulting services billed to Example Company."),),
        )


@pytest.mark.asyncio
async def test_configured_intelligence_waits_for_foreground_inference(monkeypatch) -> None:
    gate_results = iter((False, False, True))
    gate_calls = 0

    async def priority_gate() -> bool:
        nonlocal gate_calls
        gate_calls += 1
        return next(gate_results)

    monkeypatch.setattr(
        DocumentIntelligenceService,
        "_post",
        lambda *_args: {
            "doc_type": "invoice",
            "summary": "Tax invoice 1042 records consulting services billed to Example Company.",
            "tags": ["consulting services"],
        },
    )
    service = ConfiguredDocumentIntelligenceService(
        lambda: (True, "intelligence:latest"),
        IntelligenceSettings(model="environment:latest"),
        priority_gate=priority_gate,
        priority_poll_seconds=0.01,
    )

    result = await service.analyze(
        _document(),
        (_chunk(0, "Tax invoice 1042 records consulting services billed to Example Company."),),
    )

    assert gate_calls == 3
    assert result.status == "model"


@pytest.mark.asyncio
async def test_priority_wait_remains_cancellable() -> None:
    async def busy() -> bool:
        return False

    cancelled = asyncio.Event()
    service = DocumentIntelligenceService(
        IntelligenceSettings(model="intelligence:latest"),
        priority_gate=busy,
        priority_poll_seconds=0.01,
    )
    task = asyncio.create_task(
        service.analyze(
            _document(),
            (_chunk(0, "Tax invoice 1042."),),
            cancel_event=cancelled,
        )
    )
    await asyncio.sleep(0)
    cancelled.set()

    with pytest.raises(IngestionCancelled):
        await task


@pytest.mark.asyncio
async def test_task_cancellation_closes_an_inflight_intelligence_request() -> None:
    request_started = asyncio.Event()
    request_closed = asyncio.Event()
    never_finishes = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        request_started.set()
        try:
            await never_finishes.wait()
        finally:
            request_closed.set()
        raise AssertionError("cancelled request unexpectedly resumed")

    service = DocumentIntelligenceService(
        IntelligenceSettings(model="intelligence:latest"),
        transport=httpx.MockTransport(handler),
    )
    task = asyncio.create_task(
        service.analyze(
            _document(),
            (_chunk(0, "Tax invoice 1042."),),
        )
    )
    await asyncio.wait_for(request_started.wait(), timeout=0.2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.2)
    await asyncio.wait_for(request_closed.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_active_foreground_lease_defers_with_a_deadline_and_no_model_call(
    monkeypatch,
) -> None:
    model_calls = 0

    async def busy() -> bool:
        return False

    def model_call(*_args) -> None:
        nonlocal model_calls
        model_calls += 1

    service = DocumentIntelligenceService(
        IntelligenceSettings(
            model="intelligence:latest",
            priority_wait_seconds=0.03,
        ),
        priority_gate=busy,
        priority_poll_seconds=0.005,
    )
    monkeypatch.setattr(service, "_post", model_call)
    loop = asyncio.get_running_loop()
    started = loop.time()

    with pytest.raises(BackgroundInferenceDeferred):
        await service.analyze(
            _document(),
            (_chunk(0, "Tax invoice 1042."),),
        )

    assert loop.time() - started < 0.2
    assert model_calls == 0


@pytest.mark.asyncio
async def test_unavailable_redis_gate_defers_with_a_deadline_and_no_model_call(
    monkeypatch,
) -> None:
    def unavailable_redis():
        raise ConnectionError("isolated Redis outage")

    monkeypatch.setattr(inference_priority, "_redis_client", unavailable_redis)
    service = DocumentIntelligenceService(
        IntelligenceSettings(
            model="intelligence:latest",
            priority_wait_seconds=0.03,
        ),
        priority_gate=inference_priority.background_inference_allowed,
        priority_poll_seconds=0.005,
    )
    monkeypatch.setattr(
        service,
        "_post",
        lambda *_args: pytest.fail("a closed priority gate must not call the model"),
    )
    loop = asyncio.get_running_loop()
    started = loop.time()

    with pytest.raises(BackgroundInferenceDeferred):
        await service.analyze(
            _document(),
            (_chunk(0, "Tax invoice 1042."),),
        )

    assert loop.time() - started < 0.2


def test_persisted_tags_drop_document_type_variants_and_reporting_metadata() -> None:
    evidence = (
        "PepsiCo annual report discusses organic revenue growth, fiscal year results, "
        "and an annual meeting of shareholders."
    )

    assert normalize_tags(
        [
            "annual-report",
            "fiscal year",
            "organic revenue growth",
            "annual meeting of shareholders",
        ],
        evidence_text=evidence,
        summary=evidence,
        source_name="PEPSICO_2023_10K.pdf",
    ) == ("organic revenue growth", "annual meeting of shareholders")


def test_representative_text_samples_the_end_with_a_hard_limit() -> None:
    chunks = tuple(_chunk(index, f"section-{index} " + ("x" * 200)) for index in range(30))
    text = representative_text(chunks, 600)

    assert len(text) <= 600
    assert "section-0" in text
    assert "section-29" in text


@pytest.mark.asyncio
async def test_intelligence_request_keeps_multimodal_projector_off_the_gpu() -> None:
    observed: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed["request"] = request
        observed["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=(
                b'{"message":{"content":"{\\"doc_type\\":\\"report\\",'
                b'\\"summary\\":\\"A sufficiently detailed report summary.\\",'
                b'\\"tags\\":[\\"revenue\\",\\"growth\\",\\"quarterly\\"]}"}}'
            ),
        )

    service = DocumentIntelligenceService(
        IntelligenceSettings(model="qwen-test"),
        transport=httpx.MockTransport(handler),
    )
    await service._post(
        _document("quarterly-report.pdf"),
        (_chunk(0, "Revenue increased during the quarter."),),
    )

    request = observed["request"]
    body = observed["body"]
    assert isinstance(request, httpx.Request)
    assert isinstance(body, dict)
    assert service.settings.timeout_seconds == 300
    assert request.headers["accept-encoding"] == "identity"
    assert body["options"]["num_ctx"] == 16384
    assert body["options"]["num_gpu"] == 0


@pytest.mark.asyncio
async def test_model_result_is_normalized_without_padding_precise_tags(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "REPORT",
            "summary": "  A useful Q4 finance report summary with measured findings.  ",
            "tags": ["Finance", "finance", "Q4"],
        },
    )
    result = await service.analyze(
        _document("quarterly-report.pdf"),
        (_chunk(0, "Finance executive summary. Revenue increased during Q4."),),
    )

    assert result.doc_type == "report"
    assert result.status == "model"
    assert result.model == "local-test"
    assert result.summary.startswith("A useful")
    assert result.tags == ("finance", "q4")


@pytest.mark.asyncio
async def test_grounded_model_summary_is_capped_at_three_complete_sentences(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "presentation",
            "summary": (
                "Drishti predicts crowd flow from venue signals. "
                "It identifies safety risks before congestion grows. "
                "The system sends proactive alerts to operators. "
                "Its tools include Vertex AI Vision and geospatial intelligence."
            ),
            "tags": ["crowd flow", "safety risks"],
        },
    )
    evidence = (
        "Drishti predicts crowd flow from venue signals. "
        "It identifies safety risks before congestion grows. "
        "The system sends proactive alerts to operators. "
        "Its tools include Vertex AI Vision and geospatial intelligence."
    )

    result = await service.analyze(
        _document("Presentation.ppt", "application/vnd.ms-powerpoint"),
        (_chunk(0, evidence),),
    )

    assert result.status == "model"
    assert result.summary == (
        "Drishti predicts crowd flow from venue signals. "
        "It identifies safety risks before congestion grows. "
        "The system sends proactive alerts to operators."
    )
    assert len([part for part in re.split(r"(?<=[.!?])\s+", result.summary) if part]) == 3
    assert "geospatial intelligence" not in result.summary


@pytest.mark.asyncio
async def test_model_can_deliberately_return_no_tags_without_filename_padding(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "other",
            "summary": "The available text does not establish a reliable subject.",
            "tags": [],
        },
    )

    result = await service.analyze(
        _document("ambiguous-project-copy.pdf"),
        (_chunk(0, "ambiguous project copy ambiguous project copy"),),
    )

    assert result.status == "model"
    assert result.tags == ()


def test_deterministic_classifier_distinguishes_books_from_research_papers() -> None:
    book = deterministic_intelligence(
        _document("Site Reliability Engineering Book.pdf"),
        (_chunk(0, "Chapter 1. Published by O'Reilly Media. Second edition."),),
    )
    paper = deterministic_intelligence(
        _document("fault-tolerance-study.pdf"),
        (_chunk(0, "Abstract. Methodology. Experimental findings. DOI 10.1000/example."),),
    )

    assert book.doc_type == "book"
    assert paper.doc_type == "research_paper"


@pytest.mark.asyncio
async def test_valid_model_tags_are_not_padded_with_format_or_ocr_noise(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "other",
            "summary": "The image contains a short placeholder notice for a year field.",
            "tags": [
                "placeholder",
                "incomplete",
                "year",
                "mixed language",
                "informal",
                "unstructured",
                "contains symbols",
                "text",
                "PNG",
                "untrusted data",
                "filename",
                "media type",
                "doc type",
                "screenshot",
                "template1",
                "name",
                "copy",
                "details",
                "print",
                "digital copy",
                "scanned document",
                "other",
                "docker-compose.yml",
                "powerpoint",
                "presentation slides",
                "education",
                "projects",
                "skills",
                "contact info",
                "declaration",
                "scanned",
                "digital",
                "six years experience",
                "invoce",
            ],
        },
    )

    result = await service.analyze(
        _document("year.png"),
        (_chunk(0, "A short placeholder notice for a year field. placeholder incomplete goowd"),),
    )

    assert result.tags == ("placeholder",)
    assert "year" not in result.tags
    assert "png" not in result.tags
    assert "goowd" not in result.tags
    assert "untrusted data" not in result.tags
    assert "filename" not in result.tags
    assert "screenshot" not in result.tags
    assert "template1" not in result.tags
    assert "docker-compose.yml" not in result.tags
    assert "powerpoint" not in result.tags
    assert "presentation slides" not in result.tags
    assert "six years experience" not in result.tags


@pytest.mark.asyncio
async def test_evasive_ocr_metadata_is_replaced_with_provable_scan_evidence(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "other",
            "summary": (
                "The document appears to be a mix of text in multiple languages, including "
                "Kannada and English, but the content is not clearly structured or coherent. "
                "It may contain fragmented information or symbols that are not standard or "
                "easily interpretable."
            ),
            "tags": ["mixed language", "informal", "unstructured", "contains symbols"],
        },
    )
    fingerprint = (
        "opendataloader:2.4.7:result=empty+docling:2.112.0:targeted:local+"
        "ocr:tesseract-cli:tesseract_5.3.0:lang=eng,kan:psm=6"
    )
    chunks = (
        _chunk(0, "ಕರ್ನಾಟಕ ಕಂದಾಯ ಭೂಮಿ ದಾಖಲೆ ಕರ್ನಾಟಕ ಕಂದಾಯ ಭೂಮಿ ದಾಖಲೆ revenue stamp"),
        _chunk(1, "ತಾಲ್ಲೂಕು ಗ್ರಾಮ ಭೂಮಿ ದಾಖಲೆ ತಾಲ್ಲೂಕು ಗ್ರಾಮ ಭೂಮಿ ದಾಖಲೆ government record"),
    )

    result = await service.analyze(_document("AGC_2025.pdf", parser_fingerprint=fingerprint), chunks)

    assert result.status == "fallback"
    assert result.model is None
    assert result.doc_type == "other"
    assert "2 scanned pages" in result.summary
    assert "kannada script" in result.summary.lower()
    assert "OCR transcription errors may remain" in result.summary
    assert result.tags == ()
    assert not {"mixed language", "informal", "unstructured", "contains symbols"}.intersection(result.tags)
    assert "agc" not in result.tags


@pytest.mark.asyncio
async def test_evasive_raster_model_summary_falls_back_to_grounded_ocr(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "other",
            "summary": "The document appears to be a generic other type with no specific subject or content.",
            "tags": [],
        },
    )

    evidence = "User metadata Agent Database Tool Fetch event movie data RAG Tool Response"
    result = await service.analyze(
        _document("agent-diagram.png", "image/png"),
        (_chunk(0, evidence),),
    )

    assert result.status == "fallback"
    assert result.summary == f"{evidence}."
    assert len(result.summary) <= 700
    assert result.summary.endswith(".")


def test_raster_ocr_fallback_is_grounded_useful_and_bounded() -> None:
    evidence = (
        "PFU Business report covers new customer development and product sales. "
        "The report describes scanner products and document management services. "
        "It records privacy controls and business risk management. "
        "This fourth sentence must not be included in the summary."
    )

    result = deterministic_intelligence(
        _document("scanned_pdf.jpg", "image/jpeg"),
        (_chunk(0, evidence),),
    )

    assert result.status == "fallback"
    assert result.summary == (
        "PFU Business report covers new customer development and product sales. "
        "The report describes scanner products and document management services. "
        "It records privacy controls and business risk management."
    )
    assert "Automatic summary unavailable" not in result.summary
    assert len(result.summary) <= 700
    assert len([part for part in re.split(r"(?<=[.!?])\s+", result.summary) if part]) == 3


@pytest.mark.asyncio
async def test_specific_unicode_model_tags_are_preserved(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "form",
            "summary": "A ಕನ್ನಡ ಭೂ ದಾಖಲೆ land-record form lists village and property details.",
            "tags": ["ಕನ್ನಡ", "ಭೂ ದಾಖಲೆ", "land record"],
        },
    )

    result = await service.analyze(
        _document("land-record.pdf"),
        (_chunk(0, "ಕನ್ನಡ ಭೂ ದಾಖಲೆ form with village property details"),),
    )

    assert result.tags == ("ಕನ್ನಡ", "ಭೂ ದಾಖಲೆ")


def test_image_fallback_never_turns_filename_or_ocr_noise_into_tags() -> None:
    result = deterministic_intelligence(
        _document("waterfall-city-night.jpg", "image/jpeg"),
        (_chunk(0, "oar bua fig beb wine"),),
    )

    assert result.tags == ()
    assert "oar" not in result.tags
    assert result.summary == (
        "Automatic summary unavailable for this image; its extracted text remains searchable."
    )


@pytest.mark.asyncio
async def test_model_contact_details_and_urls_never_become_tags(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "resume",
            "summary": (
                "A software engineering resume describing Python and FastAPI deployment "
                "under ISO 27001 controls."
            ),
            "tags": [
                "person@example.com",
                "+91 9848883742",
                "87-879 1 Telecom Nagar Kurnool",
                "https://github.com/example",
                "python",
                "fastapi",
                "iso-27001",
                "cricket-world-cup-2023",
            ],
        },
    )

    result = await service.analyze(
        _document("resume.pdf"),
        (
            _chunk(
                0,
                "Software engineering resume describing Python and FastAPI deployment "
                "under ISO 27001 controls and related experience.",
            ),
        ),
    )

    assert result.tags == ("python", "fastapi", "iso 27001")


@pytest.mark.asyncio
async def test_tags_require_contiguous_evidence_and_reject_metadata_dates_and_people(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "policy",
            "summary": "The cloud security policy defines access controls for production systems.",
            "tags": [
                "cloud-policy",
                "cloud security",
                "research-paper",
                "blue_background",
                "2025",
                "john doe",
                "controls",
            ],
        },
    )

    result = await service.analyze(
        _document("controls.pdf"),
        (
            _chunk(
                0,
                "The cloud security policy defines access controls for production systems. "
                "Prepared by John Doe in 2025 with a blue background.",
            ),
        ),
    )

    assert result.tags == ("cloud security",)


@pytest.mark.asyncio
async def test_capitalized_entity_in_the_middle_must_exist_in_source(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    summary = "The policy protects Zephyria systems and describes access controls."
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {"doc_type": "policy", "summary": summary, "tags": []},
    )

    result = await service.analyze(
        _document("policy.pdf"),
        (_chunk(0, "The policy protects systems and describes access controls."),),
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "ungrounded_summary"
    assert result.summary != summary


@pytest.mark.asyncio
async def test_invalid_summary_does_not_discard_independently_grounded_tags(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "policy",
            "summary": "The policy protects Zephyria systems and describes access controls.",
            "tags": ["cloud security", "access controls", "author"],
        },
    )

    result = await service.analyze(
        _document("policy.pdf"),
        (_chunk(0, "The cloud security policy defines access controls for production systems."),),
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "ungrounded_summary"
    assert result.tags == ("cloud security", "access controls")


@pytest.mark.asyncio
async def test_purchase_order_type_and_tags_are_evidence_fenced(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "agreement",
            "summary": (
                "CATCO Star Trading issued a purchase order for five spray paint lines. "
                "The order includes VAT and a stated total."
            ),
            "tags": [
                "purchase order",
                "spray paints",
                "catco star trading",
                "mr.vijay",
                "u.a.e",
                "total price",
            ],
        },
    )

    result = await service.analyze(
        _document("catco.pdf"),
        (
            _chunk(
                0,
                "CATCO STAR TRADING PURCHASE ORDER A-PO-000009. Five spray paint colours, "
                "VAT AED 67.68 and total AED 1421.28. MR.VIJAY U.A.E.",
            ),
        ),
    )

    assert result.doc_type == "purchase_order"
    assert result.tags == ("spray paints", "catco star trading")


@pytest.mark.asyncio
async def test_raster_cannot_claim_research_paper_without_scholarly_evidence(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "research_paper",
            "summary": "An application screenshot shows viable server configuration status.",
            "tags": ["research paper", "viable", "server configuration"],
        },
    )

    result = await service.analyze(
        _document("server-screenshot.png", "image/png"),
        (_chunk(0, "Application server configuration status is viable."),),
    )

    assert result.doc_type == "image"
    assert result.tags == ("server configuration",)


@pytest.mark.asyncio
async def test_raster_report_and_pitchbook_types_require_strong_visible_cues(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))

    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "image",
            "summary": "The PFU business report covers scanner sales and business risks.",
            "tags": ["scanner sales", "business risks"],
        },
    )
    report = await service.analyze(
        _document("scan.jpg", "image/jpeg"),
        (_chunk(0, "PFU Business report covering scanner sales and business risks."),),
    )
    assert report.doc_type == "report"

    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "research_paper",
            "summary": "The Ultimate SpeedCloud Pitchbook by NxtGen introduces SpeedCloud.",
            "tags": ["speedcloud", "nxtgen"],
        },
    )
    pitchbook = await service.analyze(
        _document("cover.jpg", "image/jpeg"),
        (_chunk(0, "SpeedCloud by NxtGen. The Ultimate SpeedCloud Pitchbook."),),
    )
    assert pitchbook.doc_type == "presentation"
    assert pitchbook.status == "model"


@pytest.mark.asyncio
async def test_raster_lpo_is_classified_as_purchase_order(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "image",
            "summary": "The L.P.O. orders black, white, and red spray paint.",
            "tags": ["spray paint"],
        },
    )

    result = await service.analyze(
        _document("order.jpg", "image/jpeg", "ollama-vision:v8:test"),
        (_chunk(0, "L.P.O No. 9340. Black, white and red spray paint."),),
    )

    assert result.doc_type == "purchase_order"
    assert result.status == "model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("summary", "evidence"),
    [
        (
            "The graphic is a generic placeholder without a clear topic.",
            "User metadata Agent Database Tool Fetch event movie data RAG Tool Response",
        ),
        (
            "rag means reinforcement agent graph and fetches movie data.",
            "User metadata Agent Database Tool Fetch event movie data RAG Tool Response",
        ),
        (
            "TechGuruPlus issued the tax invoice for freight.",
            "Billing Address TechGuruPlus. Freight Charges 500. "
            "For Adventure Ranz Pvt Ltd Authorized Signatory.",
        ),
        (
            "Goods shipped on 20-Dec-20.",
            "Invoice Date 20-Dec-20. Delivery Note Date. State Name Karnataka.",
        ),
        (
            "The total is 1,450 including GST.",
            "Grand Total 1,450. Grand Total (Including Tax) 1,712.",
        ),
    ],
)
async def test_rephrased_unsupported_claims_are_rejected(monkeypatch, summary, evidence) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "image",
            "summary": summary,
            "tags": [],
        },
    )

    result = await service.analyze(
        _document("scan.jpg", "image/jpeg"),
        (_chunk(0, evidence),),
    )

    assert result.status == "fallback"
    assert result.summary != summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("summary", "evidence"),
    [
        (
            "The diagram uses RAG (Reinforcement Agent Graph) to fetch movie data.",
            "User metadata Agent Database Tool Fetch event movie data RAG Tool Response",
        ),
        (
            "This is a purchase order for a successful payment to Google India.",
            "Transaction Successful. Paid to Google India Digital Services.",
        ),
        (
            "The invoice lists HS Code 42MM10057101 and an amount of 4,130.",
            "Tax Invoice. HSN SAC 42MM 1005 7101. Amount 4,130.",
        ),
        (
            "Goods were delivered on December 20, 2020.",
            "Invoice Date 20-Dec-20. Delivery Note Date. State Name Karnataka.",
        ),
    ],
)
async def test_unsupported_summary_claims_fall_back(monkeypatch, summary, evidence) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "image",
            "summary": summary,
            "tags": [],
        },
    )

    result = await service.analyze(
        _document("scan.jpg", "image/jpeg"),
        (_chunk(0, evidence),),
    )

    assert result.status == "fallback"
    assert result.model is None
    assert result.summary != summary


@pytest.mark.asyncio
async def test_invoice_role_and_tax_total_must_match_labeled_evidence(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings(model="local-test"))
    monkeypatch.setattr(
        service,
        "_post",
        lambda _document, _chunks: {
            "doc_type": "invoice",
            "summary": ("This is a tax invoice from TechGuruPlus for Rs. 1,450 including freight and GST."),
            "tags": ["techguruplus", "freight"],
        },
    )
    evidence = (
        "Billing Address TechGuruPlus. Freight Charges 500. Grand Total 1,450. "
        "Grand Total (Including Tax) 1,712. Please make the cheque in favour of "
        "Adventure Ranz Pvt Ltd. for Adventure Ranz Pvt Ltd Authorized Signatory."
    )

    result = await service.analyze(
        _document("invoice.png", "image/png"),
        (_chunk(0, evidence),),
    )

    assert result.status == "fallback"
    assert result.doc_type == "invoice"


@pytest.mark.asyncio
async def test_model_failure_falls_back_but_cancellation_does_not(monkeypatch) -> None:
    service = DocumentIntelligenceService(IntelligenceSettings())

    def fail(_document, _chunks):
        raise TimeoutError

    monkeypatch.setattr(service, "_post", fail)
    chunks = (_chunk(0, "Receipt. Payment received in full."),)
    result = await service.analyze(_document("receipt.pdf"), chunks)
    assert result.status == "fallback"
    assert result.doc_type == "receipt"
    assert result.fallback_reason == "model_timeout"
    assert result.metadata()["fallback_reason"] == "model_timeout"

    cancelled = asyncio.Event()
    cancelled.set()
    with pytest.raises(IngestionCancelled):
        await service.analyze(_document(), chunks, cancel_event=cancelled)


def test_recover_truncated_object_salvages_cut_off_json() -> None:
    from app.ingestion.intelligence import _recover_truncated_object

    truncated = (
        '{"doc_type": "research_paper", "summary": "The Meridian Trade Accord '
        'was signed in Lisbon.", "tags": ["meridian trade accord", "tariffs"'
    )
    recovered = _recover_truncated_object(truncated)
    assert recovered is not None
    assert recovered["doc_type"] == "research_paper"
    assert "Lisbon" in recovered["summary"]
    assert recovered["tags"] == ["meridian trade accord", "tariffs"]


def test_recover_truncated_object_extracts_embedded_complete_object() -> None:
    from app.ingestion.intelligence import _recover_truncated_object

    payload = (
        'Here is the analysis:\n{"doc_type": "invoice", "summary": " Paid in full.", '
        '"tags": []}\nHope this helps.'
    )
    recovered = _recover_truncated_object(payload)
    assert recovered is not None
    assert recovered["doc_type"] == "invoice"


def test_recover_truncated_object_rejects_garbage() -> None:
    from app.ingestion.intelligence import _recover_truncated_object

    assert _recover_truncated_object("no braces here") is None
    assert _recover_truncated_object('{"summary": "never closes') is None
    assert _recover_truncated_object('["not", "a", "dict"]') is None
    assert _recover_truncated_object("") is None


def test_clean_summary_keeps_honorific_name_whole() -> None:
    from app.ingestion.intelligence import _clean_summary

    summary = _clean_summary(
        "The accord created a joint oversight board chaired by Dr. Elena Vasquez, "
        "with its secretariat in Porto. Ratification requires approval soon. "
        "Critics remain skeptical of enforcement."
    )
    assert "Dr. Elena Vasquez" in summary
    assert not summary.rstrip().endswith("Dr.")


def test_post_salvages_truncated_model_json() -> None:
    from app.ingestion.intelligence import DocumentIntelligenceService, IntelligenceSettings

    truncated = (
        '{"doc_type": "research_paper", "summary": "The Meridian Trade Accord '
        'was signed in Lisbon on June 14, 2023.", "tags": ["meridian accord"'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": truncated}})

    service = DocumentIntelligenceService(
        IntelligenceSettings(model="salvage-test"),
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(
        service._post(_document(), (_chunk(0, "The Meridian Trade Accord was signed in Lisbon."),))
    )
    assert result["doc_type"] == "research_paper"
    assert "Lisbon" in result["summary"]


def test_post_still_raises_on_unsalvageable_json() -> None:
    from app.ingestion.intelligence import DocumentIntelligenceService, IntelligenceSettings

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "not json at all"}})

    service = DocumentIntelligenceService(
        IntelligenceSettings(model="salvage-test"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError):
        asyncio.run(
            service._post(
                _document(), (_chunk(0, "The Meridian Trade Accord was signed in Lisbon."),)
            )
        )
