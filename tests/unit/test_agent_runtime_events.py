import json

import pytest

from agent_runtime.events import AnswerDeltaProjector, StateUpdateNormalizer, project_answer_text


def test_tuple_state_extracts_final_but_suppresses_code_and_reasoning():
    normalizer = StateUpdateNormalizer()
    raw = (
        ("CugaLiteSubgraph:abc",),
        {
            "call_model": {
                "private_state": {"script": "print('TOP SECRET GENERATED CODE')"},
                "chat_messages": [{"content": "private chain of thought"}],
                "final_answer": "Safe answer",
                "execution_complete": True,
                # The terminal CUGA update clears its generated script.
                # A nested historical script above remains non-public state.
                "script": None,
            }
        },
    )

    events = normalizer.feed(raw)
    serialized = json.dumps(events)

    assert {"type": "final", "answer": "Safe answer"} in events
    assert "TOP SECRET" not in serialized
    assert "chain of thought" not in serialized


def test_root_and_subgraph_final_answer_is_emitted_only_once():
    normalizer = StateUpdateNormalizer()

    first = normalizer.feed(
        (
            ("subgraph",),
            {
                "call_model": {
                    "final_answer": "One answer",
                    "execution_complete": True,
                    "script": None,
                }
            },
        )
    )
    duplicate = normalizer.feed(
        {
            "call_model": {
                "final_answer": "One answer",
                "execution_complete": True,
                "script": None,
            }
        }
    )
    later_variant = normalizer.feed({"SDKCallback": {"final_answer": "A second internal rendering"}})

    assert [event for event in first if event["type"] == "final"] == [
        {"type": "final", "answer": "One answer"}
    ]
    assert not [event for event in duplicate if event["type"] == "final"]
    assert not [event for event in later_variant if event["type"] == "final"]


def test_only_direct_known_terminal_nodes_can_publish_final_answer():
    cases = [
        {"sandbox": {"final_answer": "sandbox error payload", "execution_complete": True}},
        {"SDKCallback": {"final_answer": "callback rendering"}},
        {"call_model": {"final_answer": "nonterminal planning", "script": None}},
        {
            "call_model": {
                "final_answer": "code-generation pass",
                "execution_complete": True,
                "script": "result = await search_vault(query='private')",
            }
        },
        {"call_model": {"tool": {"final_answer": "nested tool result"}}},
        {"FinalAnswerAgent": {"sandbox": {"final_answer": "nested sandbox result"}}},
        {"FinalAnswerAgent": {"final_answer": "parent-graph wrapper answer"}},
    ]

    for raw in cases:
        normalizer = StateUpdateNormalizer()
        assert not [event for event in normalizer.feed(raw) if event["type"] == "final"]


def test_pinned_terminal_state_is_observed_even_when_its_prose_projects_empty():
    normalizer = StateUpdateNormalizer()

    events = normalizer.feed(
        {
            "call_model": {
                "final_answer": "Execution output: private tool envelope",
                "execution_complete": True,
                "script": None,
            }
        }
    )

    assert normalizer.has_terminal_state is True
    assert normalizer.has_final_answer is False
    assert not [event for event in events if event["type"] == "final"]


def test_internal_error_text_is_replaced_with_safe_error():
    normalizer = StateUpdateNormalizer()

    events = normalizer.feed(
        {
            "sandbox": {
                "error_message": "postgresql://vault:password@database/private",
                "script": "open('/run/secrets/key')",
            }
        }
    )

    error = next(event for event in events if event["type"] == "error")
    assert error == {
        "type": "error",
        "code": "agent_execution_failed",
        "message": "Agent execution failed",
    }
    assert "password" not in json.dumps(events)


def test_non_mapping_updates_are_ignored():
    normalizer = StateUpdateNormalizer()

    assert normalizer.feed("raw token text must not pass through") == []
    assert normalizer.feed(("bad", "shape")) == []


def test_incremental_answer_projection_handles_every_protocol_boundary():
    raw = 'The supported answer [W1].\n<LAVIX_FOLLOWUPS>{"followups":["What changed?"]}</LAVIX_FOLLOWUPS>'

    for split in range(len(raw) + 1):
        projector = AnswerDeltaProjector()
        deltas = [
            projector.feed(raw[:split]),
            projector.feed(raw[split:]),
            projector.feed("", final=True),
        ]
        assert "".join(deltas) == "The supported answer."
        assert "W1" not in "".join(deltas)
        assert "LAVIX" not in "".join(deltas)


def test_lavix_answer_wrapper_tags_never_stream_but_preserve_the_answer():
    raw = "<LAVIX_ANSWER>The supported answer.</LAVIX_ANSWER>"

    for split in range(len(raw) + 1):
        projector = AnswerDeltaProjector()
        deltas = [
            projector.feed(raw[:split]),
            projector.feed(raw[split:]),
            projector.feed("", final=True),
        ]
        assert "".join(deltas) == "The supported answer."
        assert "lavix" not in "".join(deltas).casefold()
        assert projector.public_text == "The supported answer."

    projector = AnswerDeltaProjector()
    deltas = [projector.feed(character) for character in raw]
    deltas.append(projector.feed("", final=True))
    assert "".join(deltas) == "The supported answer."


def test_unfinished_lavix_answer_wrapper_is_held_and_similar_html_is_public():
    for raw in ("<LAVIX_ANSWER", "</LAVIX_ANSWER", "< LAVIX_ANSWER data-id='protocol'"):
        projector = AnswerDeltaProjector()
        assert "".join(projector.feed(character) for character in raw) == ""
        assert projector.feed("", final=True) == ""
        assert projector.public_text == ""

    ordinary = "The <lavix_answering> element is ordinary prose."
    projector = AnswerDeltaProjector()
    deltas = [projector.feed(character) for character in ordinary]
    deltas.append(projector.feed("", final=True))
    assert "".join(deltas) == ordinary
    assert projector.public_text == ordinary


def test_completed_projection_drops_internal_lines_and_keeps_normal_code():
    answer = project_answer_text(
        "Useful response.\n"
        "Evidence: [V1]\n"
        '```json\n{"evidence":[{"provenance":"private"}]}\n```\n'
        "```python\nprint(2 + 2)\n```"
    )

    # Once an operational marker starts, the remainder is private even if it
    # later happens to contain otherwise displayable Markdown.
    assert answer == "Useful response."


def test_final_event_uses_the_same_projection_and_separates_followups():
    normalizer = StateUpdateNormalizer()

    events = normalizer.feed(
        {
            "call_model": {
                "final_answer": (
                    "Public answer [W1].\n"
                    '<LAVIX_FOLLOWUPS>{"followups":["What changed next?"]}'
                    "</LAVIX_FOLLOWUPS>\nTool output: private"
                ),
                "execution_complete": True,
                "script": None,
            }
        }
    )

    assert next(event for event in events if event["type"] == "final") == {
        "type": "final",
        "answer": "Public answer.",
        "followups": ["What changed next?"],
    }


def test_adversarial_stream_never_emits_inline_or_dangling_private_remainders():
    cases = {
        "Message only. Tool output: secret log": "Message only.",
        "Message only Tool output: secret log": "Message only",
        "Message only. Logs: password=secret": "Message only.",
        "Message only. Evidence: [W1] raw payload": "Message only.",
        "Message only [W1].": "Message only.",
        "Message only. <tool_output>secret": "Message only.",
        "Message only. <lavix_runtime>secret": "Message only.",
        "Message only. Tool out": "Message only.",
        "Message only. [W": "Message only.",
        "Message only. Let me search the vault": "Message only.",
        "Message only. **Evidence:** private": "Message only.",
        "Message only. Tool    output: private": "Message only.",
        "Message only.\n### **Sources**\nprivate": "Message only.",
        "Message only.\n**Logs**\nprivate": "Message only.",
        "Message only.\n2026-07-16T01:11:19Z [INFO] private": "Message only.",
        "Message only.\n[2026-07-16 01:11:19] ERROR private": "Message only.",
        "Message only.\n\nanswer_id": "Message only.",
        "Message only.\nanswer_id=12345": "Message only.",
        "Message only.\nanswer_id: 20250720_123": "Message only.",
    }

    forbidden = (
        "tool output",
        "logs:",
        "evidence:",
        "[w",
        "lavix",
        "let me search",
        "answer_id",
    )
    for raw, expected in cases.items():
        projector = AnswerDeltaProjector()
        deltas: list[str] = []
        for character in raw:
            delta = projector.feed(character)
            if delta:
                deltas.append(delta)
                public_so_far = "".join(deltas).casefold()
                assert not any(marker in public_so_far for marker in forbidden)
        deltas.append(projector.feed("", final=True))
        assert "".join(deltas) == expected
        assert project_answer_text(raw) == expected


def test_projection_does_not_treat_normal_comparison_as_an_internal_tag():
    assert project_answer_text("The value is 2 < 3.") == "The value is 2 < 3."
    assert project_answer_text("This is a source") == "This is a source"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Isolated trailing dateline echoes are stripped ...
        ("Elon Musk is CEO of Tesla.\n\n2026-09-05", "Elon Musk is CEO of Tesla."),
        ("Done.\n\n2026-09-05 14:30:00", "Done."),
        ("Done.\n2026-09-05T11:14:46+00:00", "Done."),
        # ... but genuine dates stay, mid-sentence, sentence-final, or alone.
        (
            "The event was held on 2026-09-05 as planned.",
            "The event was held on 2026-09-05 as planned.",
        ),
        ("The event was held on 2026-09-05", "The event was held on 2026-09-05"),
        ("2026-09-05", "2026-09-05"),
        ("Met on Sept 5, 2026 in Paris.", "Met on Sept 5, 2026 in Paris."),
    ],
)
def test_projection_strips_trailing_dateline_echo_only(raw, expected):
    assert project_answer_text(raw) == expected


def test_streaming_preserves_ordinary_comparisons_html_and_similar_tag_names():
    cases = (
        "The value is 2 < 3 and x < y.",
        "Use <strong data-id='public'>bold</strong> HTML.",
        "The <analysisfoo> element is ordinary prose.",
        "The <tool_calling> element is also ordinary.",
    )

    for raw in cases:
        projector = AnswerDeltaProjector()
        deltas = [projector.feed(character) for character in raw]
        deltas.append(projector.feed("", final=True))
        assert "".join(deltas) == raw
        assert projector.public_text == raw


def test_extended_private_tag_artifact_never_emits_and_is_dropped_at_eof():
    projector = AnswerDeltaProjector()
    artifact_chunks = [
        "<lavix_runtime_un",
        "trusted",
        "_",
        "cuga",
        "_draft",
    ]

    assert [projector.feed(chunk) for chunk in artifact_chunks] == ["", "", "", "", ""]
    assert projector.feed("", final=True) == ""
    assert projector.public_text == ""


def test_all_protected_internal_tag_roots_hold_until_the_opening_tag_closes():
    for root in ("runtime", "evidence", "tool", "execution"):
        for opening in (f"<lavix_{root}_private", f"< lavix_{root}_private"):
            projector = AnswerDeltaProjector()

            assert projector.feed(opening) == ""
            assert projector.feed(" data-id='private'") == ""
            assert projector.feed("", final=True) == ""
            assert projector.public_text == ""


def test_every_private_tag_name_holds_incomplete_opening_and_closing_fragments():
    names = (
        "untrusted_response_style_preference",
        "untrusted_user_preferences",
        "analysis",
        "thinking",
        "think",
        "reasoning",
        "private",
        "internal",
        "system",
        "developer",
        "tool_call",
        "tool_result",
        "tool_output",
        "lavix_runtime_private",
        "lavix_runtime_private-",
        "lavix_evidence_private",
        "lavix_tool_private",
        "lavix_execution_private",
    )

    for name in names:
        for raw in (f"<{name} data-id='private'", f"<\n/\n{name}"):
            for split in range(len(raw) + 1):
                projector = AnswerDeltaProjector()
                deltas = [
                    projector.feed(raw[:split]),
                    projector.feed(raw[split:]),
                    projector.feed("", final=True),
                ]
                assert "".join(deltas) == ""
                assert projector.public_text == ""

            projector = AnswerDeltaProjector()
            assert "".join(projector.feed(character) for character in raw) == ""
            assert projector.feed("", final=True) == ""

        for raw in (f"<{name} data-id='private'>", f"</{name} data-id='private'>"):
            projector = AnswerDeltaProjector()
            assert "".join(projector.feed(character) for character in raw) == ""
            assert projector.feed("", final=True) == ""
            assert projector.public_text == ""


def test_private_tag_prefix_with_newline_delimiters_never_leaks_before_recognition():
    raw = "<\n/\nlavix_runtime_untrusted_cuga_draft"
    projector = AnswerDeltaProjector()

    assert "".join(projector.feed(character) for character in raw) == ""
    assert projector.feed("", final=True) == ""
    assert projector.public_text == ""


def test_complete_private_block_can_be_followed_by_streamed_safe_answer():
    projector = AnswerDeltaProjector()
    chunks = [
        "<lavix_runtime_untrusted_cuga_draft>",
        '{"answer":"private"}',
        "</lavix_runtime_untrusted_cuga_draft>",
        "The safe answer is 4.",
    ]

    deltas = [projector.feed(chunk) for chunk in chunks]
    deltas.append(projector.feed("", final=True))

    assert "".join(deltas) == "The safe answer is 4."
    assert projector.public_text == "The safe answer is 4."


def test_projection_preserves_inline_source_prose_but_drops_source_headings():
    assert project_answer_text("The source: an archived report.") == ("The source: an archived report.")
    assert project_answer_text("Source: private execution detail") == ""
    assert project_answer_text("Answer. Sources: private execution detail") == "Answer."


def test_ordinary_iso_date_answer_streams_before_finalization():
    raw = "2026-07-16 is Thursday."
    projector = AnswerDeltaProjector()
    deltas: list[str] = []
    first_delta_at: int | None = None
    for index, character in enumerate(raw):
        delta = projector.feed(character)
        if delta:
            first_delta_at = index if first_delta_at is None else first_delta_at
            deltas.append(delta)
    deltas.append(projector.feed("", final=True))

    assert first_delta_at is not None and first_delta_at < len(raw) - 1
    assert "".join(deltas) == raw


def test_unclosed_public_fence_renders_instead_of_vanishing():
    raw = "Here is the script:\n```bash\n#!/bin/bash\necho hello\nuptime\n"
    assert project_answer_text(raw) == raw.rstrip("\n")
    projector = AnswerDeltaProjector()
    deltas = [projector.feed(raw[:20]), projector.feed(raw[20:]), projector.feed("", final=True)]
    assert "".join(deltas) == raw.rstrip("\n")


def test_unclosed_public_fence_streams_live_before_close():
    projector = AnswerDeltaProjector()
    first = projector.feed("Intro:\n```python\nprint(")
    assert "print(" in "".join([first, projector.feed("1)\n"), projector.feed("", final=True)])


def test_unclosed_json_evidence_fence_still_drops():
    raw = 'Intro:\n```json\n{"evidence":[{"provenance":"private"}]}\n'
    assert project_answer_text(raw) == "Intro:"
    projector = AnswerDeltaProjector()
    assert projector.feed(raw) == "Intro:"
    assert projector.feed("", final=True) == ""


def test_unclosed_unknown_language_fence_still_drops():
    raw = "Intro:\n```mysterylang\nsome text here\n"
    assert project_answer_text(raw) == "Intro:"


def test_unclosed_public_fence_with_protocol_body_still_drops():
    raw = "Intro:\n```bash\nresult = await search_vault(query=\"x\")\n"
    assert project_answer_text(raw) == "Intro:"


def test_bare_trailing_fence_marker_still_drops():
    assert project_answer_text("Intro text```") == "Intro text"


def test_attributed_source_backstop_strips_domains_not_prose():
    assert (
        project_answer_text("According to sportingnews.com, the Aussies won.")
        == "the Aussies won."
    )
    assert (
        project_answer_text("See https://example.com/x. According to bbc.com, rain fell.")
        == "See https://example.com/x. rain fell."
    )
    # Generic prose attributions are legitimate content and stay.
    assert (
        project_answer_text("According to the report, rain fell.")
        == "According to the report, rain fell."
    )
    assert (
        project_answer_text("According to experts, rain fell.")
        == "According to experts, rain fell."
    )
    # Model-written brand attributions are never legitimate prose, however
    # well-known the name: citations render from metadata, not model text.
    assert (
        project_answer_text("See https://example.com/x. According to BBC News, rain fell.")
        == "See https://example.com/x. rain fell."
    )
    assert (
        project_answer_text("The iPhone 17 costs Rs 79900, according to Flipkart, with 256GB storage.")
        == "The iPhone 17 costs Rs 79900, with 256GB storage."
    )


def test_bare_trailing_metadata_key_never_streams_nor_persists():
    projector = AnswerDeltaProjector()
    deltas = [
        projector.feed("GST is charged by businesses.\n\nMeta"),
        projector.feed("data: answer_id-09-10_01"),
        projector.feed("", final=True),
    ]
    joined = "".join(deltas)
    assert "Metadata" not in joined
    assert joined == project_answer_text(
        "GST is charged by businesses.\n\nMetadata: answer_id-09-10_01"
    )


def test_metadata_tag_tail_never_streams_nor_persists():
    # Exact failure from prod run 8ddf2a29: model footer leaked into the
    # stream (<metadata> + answer_id JSON) but was stripped in final,
    # tripping the stream/final mismatch guard and suppressing the answer.
    projector = AnswerDeltaProjector()
    deltas = [
        projector.feed("Prices varied by storage."),
        projector.feed("\n<metadata>"),
        projector.feed('\n  {"answer_id": "2026-09-11_14-30-12", "timestamp": "2026-09-11T14:30:12Z"}'),
        projector.feed("", final=True),
    ]
    joined = "".join(deltas)
    assert "<metadata" not in joined.casefold()
    assert "answer_id" not in joined.casefold()
    assert joined == project_answer_text(
        'Prices varied by storage.\n<metadata>\n  {"answer_id": "2026-09-11_14-30-12", "timestamp": "2026-09-11T14:30:12Z"}'
    )
    assert joined == "Prices varied by storage."


def test_backtick_metadata_footer_stream_final_agree():
    # Run e2174b97 shape: model emits "`Metadata: ..." (backtick, colon).
    # Streaming must hold it and final must strip it identically —
    # otherwise the mismatch guard suppresses the whole answer.
    raw = 'Paris.\n\n`Metadata: answer_id="2026-09-11_14-30-12"'
    projector = AnswerDeltaProjector()
    streamed = ""
    for index in range(0, len(raw), 4):
        streamed += projector.feed(raw[index : index + 4])
    streamed += projector.feed("", final=True)
    assert streamed == project_answer_text(raw) == "Paris."


def test_brand_attribution_never_streams_nor_persists():
    # Prod "iPhone 17 price" shape: the model names a retailer with no
    # Sources chip. The attribution must neither stream (mismatch guard
    # would suppress the answer) nor persist.
    raw = "The iPhone 17 costs Rs 79900, according to Flipkart, with 256GB storage."
    projector = AnswerDeltaProjector()
    streamed = ""
    for index in range(0, len(raw), 6):
        streamed += projector.feed(raw[index : index + 6])
    streamed += projector.feed("", final=True)
    final = project_answer_text(raw)
    assert "Flipkart" not in streamed
    assert streamed == final == "The iPhone 17 costs Rs 79900, with 256GB storage."


def test_generic_attribution_streams_and_persists():
    # "According to the report/experts" is prose, not sourcing: it must
    # stream incrementally like any other content (no EOF-only delay).
    raw = "According to the report, rain fell."
    projector = AnswerDeltaProjector()
    first = projector.feed(raw)
    assert first == raw
    assert projector.feed("", final=True) == ""
    assert project_answer_text(raw) == raw


def test_project_answer_text_strips_null_answer_id_and_metadata_block():
    from agent_runtime.events import project_answer_text

    raw = (
        "[Support hotline number: 1-800-123-4567]\n\nMetadata\n"
        '{ "answer_id": null, "timestamp": null }'
    )
    cleaned = project_answer_text(raw)
    assert "answer_id" not in cleaned
    assert "Metadata" not in cleaned
    assert "null" not in cleaned


def test_metadata_header_json_block_stream_final_agree():
    # Fix 3 shape: model emits "Metadata" header line (no colon) plus a
    # null-valued answer_id JSON block. Streaming must hold from the header
    # and final must strip both identically.
    raw = (
        "[Support hotline number: 1-800-123-4567]\n\nMetadata\n"
        '{ "answer_id": null, "timestamp": null }'
    )
    projector = AnswerDeltaProjector()
    streamed = ""
    for index in range(0, len(raw), 4):
        streamed += projector.feed(raw[index : index + 4])
    streamed += projector.feed("", final=True)
    assert streamed == project_answer_text(raw)
    assert "answer_id" not in streamed
    assert "Metadata" not in streamed


def test_project_answer_text_strips_markdown_followup_block() -> None:
    body = (
        "Indirect talks persist but face little progress.\n"
        "\n"
        "Followup questions:\n"
        "\n"
        "  How does Iran's rejection shape current negotiations?\n"
        "  What role do regional allies play in mediating these talks?\n"
        "\n"
        "Note: Followup could not be verified against retrieved\n"
        "sources."
    )

    cleaned = project_answer_text(body)

    assert cleaned == "Indirect talks persist but face little progress."
    assert "Followup" not in cleaned


def test_project_answer_text_strips_bold_followups_heading() -> None:
    body = (
        "They rely on moralizing narratives.\n"
        "\n"
        "Followups:\n"
        "\n"
        "  How does projection tie into tribalism in virtual spaces?\n"
        "  What modern behaviors does Greene link to these biases?\n"
        "\n"
        "Note: Followups could not be verified against retrieved sources."
    )

    cleaned = project_answer_text(body)

    assert cleaned == "They rely on moralizing narratives."
    assert "Followup" not in cleaned


def test_project_answer_text_preserves_genuine_caveats_and_topic() -> None:
    genuine = (
        "The total is $4,820 for 40 kits.\n"
        "\n"
        "Note: August, 25 could not be verified against retrieved sources."
    )
    assert "August, 25" in project_answer_text(genuine)

    discussion = (
        "Researchers study how followup questions shape interviews. "
        "Good followup questions are open-ended and specific."
    )
    assert project_answer_text(discussion) == discussion


def test_project_answer_text_strips_lone_trailing_followup_note() -> None:
    body = "The talks continue.\n\nNote: Followups could not be verified."

    assert project_answer_text(body) == "The talks continue."


def test_project_answer_text_strips_followup_block_keeping_genuine_note() -> None:
    body = (
        "The invoice is from Initech Supplies.\n"
        "\n"
        "*Followups:*\n"
        "- Was this invoice for sensor kits?\n"
        "- What total amount did it cover\n"
        "\n"
        "Note: August, 21, 2026 could not be verified against retrieved sources."
    )

    cleaned = project_answer_text(body)

    assert cleaned == (
        "The invoice is from Initech Supplies.\n"
        "\n"
        "Note: August, 21, 2026 could not be verified against retrieved sources."
    )
    assert "Followup" not in cleaned.split("Note:")[0]
