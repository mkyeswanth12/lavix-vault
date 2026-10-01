"""Cross-person attribution: retrieval may return multiple people's resumes,
but synthesis must never transfer facts between their source blocks.

Fixtures are synthetic and in-memory; no production or private resumes are
used. These tests pin the synthesis identity boundary (SOURCE labels +
prompt rule) and the fixture-grounded attribution contract. They do not run
an LLM: live-model verification reuses _assert_person_bound() against real
answers in QA.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

from agent_runtime.config import RuntimeSettings
from agent_runtime.cuga_adapter import (
    AUTHORITATIVE_SYNTHESIS_PROMPT,
    CugaAdapter,
    _synthesis_evidence,
    _synthesis_evidence_ids,
)
from agent_runtime.schemas import RunRequest

# ---------------------------------------------------------------------------
# Synthetic fixtures (nearest-neighbor-shaped, distinct people).
# ---------------------------------------------------------------------------

GAGENDRA_RESUME = """Gagendra Kumar
Total Experience: 8 years
Company A — Software Engineer, 2020-2022
Company B — Senior Software Engineer, 2022-present
Skills: Python, PostgreSQL"""

YESWANTH_RESUME = """M.K. Yeswanth
Total Experience: 11 years
Company C — Analyst, 2020-2023
Infosys — Consultant, 2023-2025
Skills: Java, SAP"""

# Shared employer, different periods (Test E).
GAGENDRA_SHARED = """Gagendra Kumar
Total Experience: 8 years
Company X — Engineer, 2020-2022"""

YESWANTH_SHARED = """M.K. Yeswanth
Total Experience: 11 years
Company X — Manager, 2023-2025"""

GAGENDRA_OWN_FACTS = ("Company A", "Company B", "8 years", "Python")
YESWANTH_ONLY_FACTS = ("Infosys",)

GOOD_GAGENDRA_ANSWER = (
    "Gagendra Kumar has 8 years of total experience, working at Company A "
    "from 2020-2022 and at Company B since 2022."
)
# The exact reported failure mode.
BAD_GAGENDRA_ANSWER = (
    "Gagendra Kumar has 8 years of total experience, working at Infosys "
    "from 2023-2025."
)


def _vault_result(*items: tuple[str, str, str]) -> dict[str, Any]:
    """Build a gateway-shaped vault result: (id, filename, content)."""
    return {
        "ok": True,
        "evidence": [
            {"id": item_id, "filename": filename, "section_path": "Experience", "content": content}
            for item_id, filename, content in items
        ],
        "count": len(items),
    }


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _vault_items_in_body(body: str) -> list[dict[str, Any]]:
    """Parse the Execution-output vault array out of assembled content.

    The blob is compact JSON (newlines/unicode escaped); parsing beats
    substring splitting because each item's `source` label trails its
    content inside the same dict.
    """
    start = body.index('{"vault":')
    payload, _ = json.JSONDecoder().raw_decode(body[start:])
    items = payload["vault"]
    assert isinstance(items, list) and all(isinstance(item, dict) for item in items)
    return items


def _assert_person_bound(
    answer: str,
    *,
    person: str,
    own_facts: tuple[str, ...],
    alien_facts: tuple[str, ...],
    own_chunks: tuple[str, ...],
) -> None:
    """Fixture-grounded attribution contract used by tests A/C/D/E.

    Every sentence naming `person` must not contain a fact that appears
    only outside that person's fixture chunks. This asserts binding, not
    mere word absence: the BAD bug-report answer fails while the GOOD one
    passes even though both mention employers and dates.
    """
    person_first = person.split()[0].casefold()
    for sentence in _split_sentences(answer):
        words = set(re.findall(r"[A-Za-z][\w']*", sentence.casefold()))
        if person_first not in words and person.casefold() not in sentence.casefold():
            continue
        for alien in alien_facts:
            assert alien.casefold() not in sentence.casefold(), (
                f"cross-person attribution: {alien!r} bound to {person!r} in: {sentence!r}"
            )
            assert not any(
                alien.casefold() in chunk.casefold() for chunk in own_chunks
            ), f"fixture error: {alien!r} is not alien to {person!r}"
    assert any(
        fact.casefold() in answer.casefold() for fact in own_facts
    ), f"answer lost {person!r} own facts: {answer!r}"


def _adapter() -> CugaAdapter:
    return CugaAdapter(RuntimeSettings(), gateway=None)


def _messages_backend() -> Any:
    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content

    return SimpleNamespace(HumanMessage=_Message, AIMessage=_Message)


def _fresh_request(text: str) -> RunRequest:
    """Genuinely new chat: single user message, no history (mirrors
    orchestrator _prepare with chat_id=None → history=[])."""
    return RunRequest.model_validate(
        {
            "run_id": "00000000-0000-0000-0000-00000000c001",
            "user_query": text,
            "messages": [{"role": "user", "content": text}],
            "options": {"web_search_enabled": False, "deep_search": True},
        }
    )


# ---------------------------------------------------------------------------
# Projection compatibility (I) + label format.
# ---------------------------------------------------------------------------


def test_source_labels_use_real_ids_and_keep_existing_fields():
    result = _vault_result(
        ("V1", "Gagendra Kumar Resume.pdf", GAGENDRA_RESUME),
        ("V2", "M.K. Yeswanth resume.pdf", YESWANTH_RESUME),
    )
    projected = _synthesis_evidence(result, kind="vault")

    assert [item["source"] for item in projected] == [
        '[SOURCE V1 · file "Gagendra Kumar Resume.pdf"]',
        '[SOURCE V2 · file "M.K. Yeswanth resume.pdf"]',
    ]
    # Existing fields byte-identical to the pre-patch shape.
    assert [item["filename"] for item in projected] == [
        "Gagendra Kumar Resume.pdf",
        "M.K. Yeswanth resume.pdf",
    ]
    assert [item["content"] for item in projected] == [GAGENDRA_RESUME, YESWANTH_RESUME]
    assert [item["section_path"] for item in projected] == ["Experience", "Experience"]
    assert all(set(item) == {"filename", "section_path", "content", "source"} for item in projected)


def test_source_label_falls_back_without_id():
    result = {
        "ok": True,
        "evidence": [{"filename": "resume.pdf", "content": "text"}],
        "count": 1,
    }
    (item,) = _synthesis_evidence(result, kind="vault")
    assert item["source"] == '[SOURCE V1 · file "resume.pdf"]'


def test_no_person_metadata_invented():
    # Filename carries no person name here: the label must not gain one.
    result = {
        "ok": True,
        "evidence": [{"id": "V7", "filename": "resume.pdf", "content": GAGENDRA_RESUME}],
        "count": 1,
    }
    (item,) = _synthesis_evidence(result, kind="vault")
    assert item["source"] == '[SOURCE V7 · file "resume.pdf"]'
    assert "Gagendra" not in item["source"]
    for forbidden in ("person_id", "entity_id", "owner", "author", "subject", "entity"):
        assert forbidden not in item


def test_prompt_binds_facts_to_source_blocks():
    assert "same labeled block" in AUTHORITATIVE_SYNTHESIS_PROMPT
    assert "Never transfer employers" in AUTHORITATIVE_SYNTHESIS_PROMPT
    assert "never print" in AUTHORITATIVE_SYNTHESIS_PROMPT.lower()
    # Pre-existing rules intact (no replacement/weakening).
    assert "not found in context" in AUTHORITATIVE_SYNTHESIS_PROMPT
    assert "Do not include citation IDs" in AUTHORITATIVE_SYNTHESIS_PROMPT


# ---------------------------------------------------------------------------
# A. Fresh chat, both resumes: boundary present in model input.
# ---------------------------------------------------------------------------


def test_fresh_chat_history_is_empty_and_blocks_stay_labeled():
    request = _fresh_request("What is Gagendra Kumar's total experience?")
    assert [message.role for message in request.messages] == ["user"]

    adapter = _adapter()
    messages = adapter._to_backend_messages(
        request,
        _messages_backend(),
        vault_result=_vault_result(
            ("V1", "Gagendra Kumar Resume.pdf", GAGENDRA_RESUME),
            ("V2", "M.K. Yeswanth resume.pdf", YESWANTH_RESUME),
        ),
    )
    body = messages[-1].content
    assert "What is Gagendra Kumar's total experience?" in body
    assert "Conversation history" not in body
    assert "Execution output:" in body
    # Labels ride inside compact JSON (non-ASCII escaped); assert the
    # stable prefixes — exact label text is pinned at projection level.
    assert "[SOURCE V1" in body
    assert "[SOURCE V2" in body
    # Contents survive assembly (single-line fragments; raw newlines are
    # JSON-escaped in transit).
    for fragment in ("Company A", "Company B", "8 years", "Infosys", "11 years"):
        assert fragment in body, fragment


def test_attribution_contract_catches_reported_failure():
    _assert_person_bound(
        GOOD_GAGENDRA_ANSWER,
        person="Gagendra Kumar",
        own_facts=GAGENDRA_OWN_FACTS,
        alien_facts=YESWANTH_ONLY_FACTS,
        own_chunks=(GAGENDRA_RESUME,),
    )
    try:
        _assert_person_bound(
            BAD_GAGENDRA_ANSWER,
            person="Gagendra Kumar",
            own_facts=GAGENDRA_OWN_FACTS,
            alien_facts=YESWANTH_ONLY_FACTS,
            own_chunks=(GAGENDRA_RESUME,),
        )
    except AssertionError as exc:
        assert "cross-person attribution" in str(exc)
    else:
        raise AssertionError("contract failed to catch the reported Gagendra+Infosys fusion")


# ---------------------------------------------------------------------------
# B. Existing chat with Yeswanth history: window unchanged, no inheritance.
# ---------------------------------------------------------------------------


def test_existing_yeswanth_history_stays_context_not_evidence():
    request = RunRequest.model_validate(
        {
            "run_id": "00000000-0000-0000-0000-00000000c002",
            "user_query": "What is Gagendra Kumar's total experience?",
            "messages": [
                {"role": "user", "content": "Where did Yeswanth work?"},
                {"role": "assistant", "content": "M.K. Yeswanth worked at Infosys."},
                {"role": "user", "content": "What is Gagendra Kumar's total experience?"},
            ],
            "options": {"web_search_enabled": False, "deep_search": True},
        }
    )
    adapter = _adapter()
    messages = adapter._to_backend_messages(
        request,
        _messages_backend(),
        vault_result=_vault_result(
            ("V1", "Gagendra Kumar Resume.pdf", GAGENDRA_RESUME),
            ("V2", "M.K. Yeswanth resume.pdf", YESWANTH_RESUME),
        ),
    )
    body = messages[-1].content
    prior = [message.content for message in messages if "Prior assistant reply" in message.content]
    # History window behavior unchanged: prior assistant turn marked context-only.
    assert len(prior) == 1
    assert "not verified evidence" in prior[0]
    assert "M.K. Yeswanth worked at Infosys." in prior[0]
    # Vault blocks still labeled; history text never gains a SOURCE label.
    assert body.count("[SOURCE V") == 2
    _assert_person_bound(
        GOOD_GAGENDRA_ANSWER,
        person="Gagendra Kumar",
        own_facts=GAGENDRA_OWN_FACTS,
        alien_facts=YESWANTH_ONLY_FACTS,
        own_chunks=(GAGENDRA_RESUME,),
    )


# ---------------------------------------------------------------------------
# C. User-memory contamination stays out of vault blocks.
# ---------------------------------------------------------------------------


def test_memory_block_stays_separate_from_labeled_vault_blocks():
    request = _fresh_request("What is Gagendra Kumar's total experience?")
    adapter = _adapter()
    messages = adapter._to_backend_messages(
        request,
        _messages_backend(),
        vault_result=_vault_result(
            ("V1", "Gagendra Kumar Resume.pdf", GAGENDRA_RESUME),
        ),
        graph_result={
            "ok": True,
            "memories": [
                {
                    "subject": "I",
                    "predicate": "worked_at",
                    "object_value": "Infosys (M.K. Yeswanth context)",
                }
            ],
        },
    )
    body = messages[-1].content
    # Memory rides in its own personalization section, never inside a vault
    # SOURCE block; production memory retrieval itself is unchanged.
    assert "Personalization" in body
    assert "Infosys (M.K. Yeswanth context)" in body
    vault_section = body.split("Execution output:")[1].split("Personalization context:")[0]
    assert "Infosys (M.K. Yeswanth context)" not in vault_section
    assert "[SOURCE V1" in vault_section


# ---------------------------------------------------------------------------
# D. Both resumes valid: Yeswanth facts answerable, not suppressed.
# ---------------------------------------------------------------------------


def test_yeswanth_facts_remain_answerable():
    request = _fresh_request("Where did M.K. Yeswanth work?")
    adapter = _adapter()
    messages = adapter._to_backend_messages(
        request,
        _messages_backend(),
        vault_result=_vault_result(
            ("V1", "Gagendra Kumar Resume.pdf", GAGENDRA_RESUME),
            ("V2", "M.K. Yeswanth resume.pdf", YESWANTH_RESUME),
        ),
    )
    body = messages[-1].content
    # Separation cuts both ways: Yeswanth's own block is fully present.
    assert "Infosys" in body and "11 years" in body and "Company C" in body
    assert "[SOURCE V2" in body
    _assert_person_bound(
        "M.K. Yeswanth worked at Infosys from 2023-2025.",
        person="M.K. Yeswanth",
        own_facts=("Infosys", "2023-2025"),
        alien_facts=("Company A",),
        own_chunks=(YESWANTH_RESUME,),
    )


# ---------------------------------------------------------------------------
# E. Shared employer, different dates stay file-bound.
# ---------------------------------------------------------------------------


def test_shared_employer_dates_stay_bound_to_source_blocks():
    adapter = _adapter()
    messages = adapter._to_backend_messages(
        _fresh_request("Where did Gagendra Kumar work?"),
        _messages_backend(),
        vault_result=_vault_result(
            ("V1", "Gagendra Resume.pdf", GAGENDRA_SHARED),
            ("V2", "Yeswanth Resume.pdf", YESWANTH_SHARED),
        ),
    )
    items = _vault_items_in_body(messages[-1].content)
    assert [item["source"] for item in items] == [
        '[SOURCE V1 · file "Gagendra Resume.pdf"]',
        '[SOURCE V2 · file "Yeswanth Resume.pdf"]',
    ]
    assert "2020-2022" in items[0]["content"] and "2023-2025" not in items[0]["content"]
    assert "2023-2025" in items[1]["content"] and "2020-2022" not in items[1]["content"]


# ---------------------------------------------------------------------------
# F. Citation/evidence safety: IDs, membership, cards unchanged.
# ---------------------------------------------------------------------------


def test_evidence_ids_and_membership_unchanged():
    result = _vault_result(
        ("V1", "Gagendra Kumar Resume.pdf", GAGENDRA_RESUME),
        ("V2", "M.K. Yeswanth resume.pdf", YESWANTH_RESUME),
    )
    ids = _synthesis_evidence_ids(result, None)
    assert ids == {"vault": ["V1", "V2"], "web": []}
    # Labels align 1:1 with the card-membership IDs (same order, same ids).
    projected = _synthesis_evidence(result, kind="vault")
    for item, want_id in zip(projected, ids["vault"], strict=True):
        assert f"[SOURCE {want_id} ·" in item["source"]
