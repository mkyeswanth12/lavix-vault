"""Regression tests for BUG-003: multi-file source attribution.

Deterministic layers only (no model): identity projection, per-file
envelopes, per-file number/date attribution, vault card pruning, and
file-aware pairing. Live end-to-end verification is recorded separately.
"""
from agent_runtime.cuga_adapter import (
    _pairing_mismatches,
    _synthesis_evidence,
    _unattributed_file_numbers,
    _vault_card_atom_coverage,
)

A_SELLER = "Seller = Company X. Contact seller@example.com for orders."
B_PROJECT = "Project = Project Y. Kickoff in March."
A_TOTAL = "Invoice total 7,799 rupees. Tax 1,189 rupees."
B_GROWTH = "Growth averaged 4 percent. Vision 2030 guides planning."


def vault(*items):
    return {"evidence": [{"id": f"V{i+1}", **item} for i, item in enumerate(items)]}


def item(file_id, filename, content):
    return {"file_id": file_id, "filename": filename, "content": content}


def test_1_identity_survives_projection():
    out = _synthesis_evidence(
        vault(item(101, "a.pdf", "x"), item(102, "b.pdf", "y")), kind="vault"
    )
    assert "file_id 101" in out[0]["source"]
    assert "file_id 102" in out[1]["source"]
    assert out[0]["file_id"] == 101 and out[1]["file_id"] == 102


def test_2_reverse_order_keeps_attribution():
    fwd = _synthesis_evidence(vault(item(101, "a.pdf", "x"), item(102, "b.pdf", "y")), kind="vault")
    rev = _synthesis_evidence(vault(item(102, "b.pdf", "y"), item(101, "a.pdf", "x")), kind="vault")
    assert {i["file_id"] for i in fwd} == {i["file_id"] for i in rev} == {101, 102}
    # envelopes: same-file chunks stay adjacent
    multi = _synthesis_evidence(
        vault(
            item(101, "a.pdf", "x1"),
            item(102, "b.pdf", "y1"),
            item(101, "a.pdf", "x2"),
        ),
        kind="vault",
    )
    ids = [i["file_id"] for i in multi]
    assert ids == [101, 101, 102]
    assert "chunk 1 of 2" in multi[0]["source"]
    assert "chunk 2 of 2" in multi[1]["source"]


def test_3_irrelevant_file_does_not_break_attribution():
    res = vault(
        item(101, "a.pdf", A_SELLER),
        item(102, "b.pdf", B_PROJECT),
        item(103, "c.pdf", "Unrelated weather discussion with no figures."),
    )
    bad = _unattributed_file_numbers("Seller = Company X. Project = Project Y.", res)
    assert bad == []


def test_4_numbers_stay_with_their_file():
    res = vault(item(101, "a.pdf", A_TOTAL), item(102, "b.pdf", B_GROWTH))
    assert _unattributed_file_numbers("The total is 7,799 rupees.", res) == []
    assert _unattributed_file_numbers("Growth averaged 4 percent.", res) == []


def test_5_dates_stay_with_their_file():
    res = vault(
        item(101, "a.pdf", "Contract signed March 3, 2026."),
        item(102, "b.pdf", "Renewal due June 9, 2026."),
    )
    assert _unattributed_file_numbers("Signed March 3, 2026.", res) == []
    assert _unattributed_file_numbers("Due June 9, 2026.", res) == []


def test_6_cross_file_binding_is_flagged():
    res = vault(
        item(101, "a.pdf", "Seller = Company X. Total 100."),
        item(102, "b.pdf", "Project = Project Y. Budget 200."),
    )
    # 200 belongs to B's budget; binding it to A's seller must not pass.
    bad = _unattributed_file_numbers("Seller Company X paid 200.", res)
    assert bad != []


def test_7_revoked_or_deleted_means_no_block():
    # Excluded files never reach vault_result: single remaining file
    # keeps the legacy pooled path (helper returns []).
    res = vault(item(101, "a.pdf", A_TOTAL))
    assert _unattributed_file_numbers("The total is 7,799 rupees.", res) == []


def test_9_three_files_associate():
    res = vault(
        item(101, "a.pdf", A_SELLER),
        item(102, "b.pdf", B_PROJECT),
        item(103, "c.pdf", "Owner = Priya. Since 2021."),
    )
    answer = "Seller = Company X. Project = Project Y. Owner Priya since 2021."
    assert _unattributed_file_numbers(answer, res) == []


def test_10_rename_keeps_file_id_attribution():
    before = _synthesis_evidence(vault(item(101, "invoice.pdf", A_TOTAL)), kind="vault")
    after = _synthesis_evidence(vault(item(101, "renamed.pdf", A_TOTAL)), kind="vault")
    assert "file_id 101" in before[0]["source"]
    assert "file_id 101" in after[0]["source"]
    assert _unattributed_file_numbers("The total is 7,799 rupees.", vault(item(101, "renamed.pdf", A_TOTAL))) == []


def test_11_name_numbers_do_not_trip_attribution():
    res = vault(
        item(101, "a.pdf", "Qatar National Vision 2030 report."),
        item(102, "b.pdf", "Model X100 review. QAT 2026 roadmap. Company X 2025 results."),
    )
    assert _unattributed_file_numbers("Vision 2030 guides planning.", res) == []
    assert _unattributed_file_numbers("Model X100 review notes.", res) == []


def test_12_true_contradiction_still_refuses():
    draft = "Team A scored 120 points."
    pooled = ["Team A scored 95 points in the final."]
    assert _pairing_mismatches(draft, pooled) != []
    per_file_ok = [(101, "Team A scored 120 points.")]
    assert _pairing_mismatches(draft, per_file_ok) == []
    per_file_clash = [(101, "Team A scored 95 points.")]
    assert _pairing_mismatches(draft, per_file_clash) != []
    cross = [(101, "Team A scored 95 points."), (102, "Team A scored 120 points.")]
    # Same-file contradiction in 101 still fires despite 102 agreeing.
    assert _pairing_mismatches(draft, cross) != []


def test_vault_pruning_keeps_covering_cards_only():
    res = vault(item(101, "a.pdf", A_SELLER), item(102, "b.pdf", B_PROJECT))
    cov = _vault_card_atom_coverage(res, ["Company X"])
    assert cov["V1"] != [] and cov["V2"] == []


def _res2():
    return {
        "evidence": [
            {"id": "V1", "file_id": 101, "filename": "invoice.pdf",
             "content": "Seller = Company X. Tax invoices from Amazon Seller Services."},
            {"id": "V2", "file_id": 102, "filename": "qatar.pdf",
             "content": "Qatar energy overview. LNG export capacity."},
        ]
    }


def test_false_denial_fires_when_entity_present():
    from agent_runtime.cuga_adapter import _false_denial_entity

    assert _false_denial_entity(
        "The seller on the invoice is not explicitly mentioned.", _res2()
    ) == "seller"
    assert _false_denial_entity(
        "There is no information about the seller.", _res2()
    ) == "seller"


def test_true_denial_passes_when_entity_absent():
    from agent_runtime.cuga_adapter import _false_denial_entity

    assert _false_denial_entity(
        "Solar panel prices are not mentioned in these documents.", _res2()
    ) == ""
    assert _false_denial_entity(
        "The invoice does not mention solar panel prices.", _res2()
    ) == ""
    assert _false_denial_entity("The seller is Company X.", _res2()) == ""


def test_denial_guard_needs_two_files():
    from agent_runtime.cuga_adapter import _false_denial_entity

    single = {"evidence": [_res2()["evidence"][0]]}
    assert _false_denial_entity("The seller is not mentioned.", single) == ""


def _res_partial():
    # Only file 102 retrieved; 101 was requested but yielded nothing.
    return {
        "evidence": [
            {"id": "V1", "file_id": 102, "filename": "qatar.pdf",
             "content": "Qatar energy overview. LNG export capacity."},
        ]
    }


def test_denial_about_unevidenced_file_refuses():
    from agent_runtime.cuga_adapter import _false_denial_entity

    answer = "The seller on the invoice is not explicitly mentioned."
    assert _false_denial_entity(answer, _res_partial(), [101, 102], "Name the seller on the invoice") != ""


def test_full_pool_denial_without_contradiction_passes():
    from agent_runtime.cuga_adapter import _false_denial_entity

    assert _false_denial_entity(
        "Solar panel prices are not mentioned.", _res2(), [101, 102], "What are prices?"
    ) == ""
    # Answer grounded in the evidenced file, no denial at all.
    assert _false_denial_entity(
        "The seller is Company X.", _res2(), [101, 102], "Who is the seller?"
    ) == ""


def _raw_chunk(cid, fid, text):
    return {"id": cid, "file_id": fid, "filename": f"f{fid}.pdf",
            "section_path": [], "content": text, "relevant": True}


def test_compaction_keeps_stranded_fairness_file():
    from agent_runtime.gateway import _compact_tool_result

    evidence = [_raw_chunk(f"V{i+1}", 310, f"qatar chunk {i}") for i in range(8)]
    evidence.append(_raw_chunk("V9", 306, "Tax Invoice Seller = Company X"))
    result = _compact_tool_result({"ok": True, "evidence": evidence, "count": 9}, "seller invoice", kind="vault")
    kept = result["evidence"]
    assert len(kept) == 6
    assert {item["file_id"] for item in kept} == {310, 306}
    assert kept[-1]["file_id"] == 306
    # rank order otherwise preserved
    assert [item["id"] for item in kept[:5]] == ["V1", "V2", "V3", "V4", "V5"]


def test_compaction_unchanged_without_stranding():
    from agent_runtime.gateway import _compact_tool_result

    evidence = [_raw_chunk(f"V{i+1}", 310, f"chunk {i}") for i in range(8)]
    result = _compact_tool_result({"ok": True, "evidence": evidence, "count": 8}, "q", kind="vault")
    assert [item["id"] for item in result["evidence"]] == ["V1", "V2", "V3", "V4", "V5", "V6"]


def test_pairing_is_sentence_local_not_row_global():
    from agent_runtime.cuga_adapter import _pairing_mismatches

    # One long row: qatar and 2 live sentences apart — must not pair.
    row = "Qatar National Vision 2030 guides planning. The program rests on 2 pillars."
    assert _pairing_mismatches("Qatar 2030 plan.", [(101, row)]) == []
    # Same sentence: genuine binding still checked.
    row2 = "Team A scored 95 points in the final game."
    assert _pairing_mismatches("Team A scored 120 points.", [(101, row2)]) != []


def test_evidence_name_numbers_skip_nomenclature():
    from agent_runtime.cuga_adapter import _pairing_mismatches

    evidence = [(310, "Qatar National Vision 2030 report. Qatar produces 8000 barrels daily."),
                (306, "Seller = Company X. Total 7799.")]
    # Draft restating the name: no refusal.
    assert _pairing_mismatches("Qatar Vision 2030 plan.", evidence) == []
    # Genuine contradiction on a non-name number still refuses.
    assert _pairing_mismatches("Qatar produces 9000 barrels daily.", evidence) != []


def test_bare_counts_do_not_pair_but_qualified_values_do():
    from agent_runtime.cuga_adapter import _answer_number_pairs, _pairing_mismatches

    assert _answer_number_pairs("2 Middle East Economic Survey notes growth.") == []
    assert ("qatar", "4") in [
        (e, n) for e, n in _answer_number_pairs("Qatar growth of 4% annually.")
    ]
    ev = [(310, "2 Middle East Economic Survey. Qatar growth of 4% annually.")]
    assert _pairing_mismatches("Qatar growth of 4% annually.", ev) == []
    assert _pairing_mismatches("Qatar growth of 9% annually.", ev) != []


def test_excerpt_keeps_sentence_integrity():
    from agent_runtime.gateway import _evidence_excerpt

    text = ("Qatar growth outlook. The strategy targets growth of 4% across Qatar's "
            "non-energy sectors next year. Other news. ") + "x" * 2100
    out = _evidence_excerpt("growth Qatar outlook", text)
    assert "growth of 4%" in out
    assert "Qatar" in out.split("growth of 4%")[0][-120:]


def _cards():
    return {
        "evidence": [
            {"id": "V1", "file_id": 310, "filename": "qatar.pdf",
             "content": "Qatar energy overview. LNG export capacity. National Vision 2030."},
        ]
    }


def _kw_cards():
    return [
        "Qatar energy overview. LNG export capacity. National Vision 2030.",
    ]


def test_uncited_keyword_refuses_denial_plus_speculation():
    from agent_runtime.cuga_adapter import _uncited_question_keywords

    q = "Name the seller on the invoice and one energy topic from the Qatar document"
    bad = ("The seller on the invoice is not explicitly mentioned. However, it can "
           "be inferred that the seller is likely related to Qatar energy sector.")
    assert _uncited_question_keywords(bad, q, _kw_cards()) != ""


def test_uncited_keyword_passes_synonym_mention():
    from agent_runtime.cuga_adapter import _uncited_question_keywords

    q = "Which renewable sources does the Qatar document mention"
    good = "The Qatar document mentions renewable sources such as solar and biomass energy."
    cited = ["Qatar energy overview. Solar and biomass energy. LNG export capacity."]
    assert _uncited_question_keywords(good, q, cited) == ""


def test_cited_keywords_pass():
    from agent_runtime.cuga_adapter import _uncited_question_keywords

    q = "Name the seller on the invoice and one energy topic from the Qatar document"
    good = ("The seller is Amazon Seller Services. Qatar grows LNG exports. "
            "National Vision 2030 diversifies the economy.")
    cited = _kw_cards() + ["Seller = Company X. Amazon Seller Services invoice. Tax total 7799."]
    assert _uncited_question_keywords(good, q, cited) == ""


def test_undiscussed_topics_pass():
    from agent_runtime.cuga_adapter import _uncited_question_keywords

    assert _uncited_question_keywords("Qatar grows LNG exports.", "Tell me about Qatar", _kw_cards()) == ""
    assert _uncited_question_keywords("Hello!", "Hello", []) == ""


def test_word_numbers_ground_in_digit_evidence():
    from agent_runtime.cuga_adapter import _grounding_gap

    vault = {"evidence": [
        {"id": "V1", "file_id": 306, "filename": "invoice.pdf",
         "content": "Shipping Charges 50. Tax rate 18 IGST."},
    ]}
    # Single-word figures ground in digit evidence; fused phrases with
    # mismatched wording ("Tax Eighteen" vs "Tax rate 18") stay gapped by
    # design — refusing an awkwardly phrased truth beats shipping it blind.
    assert _grounding_gap("Shipping is Fifty.", vault, None) == []
    assert "Tax Eighteen" in _grounding_gap("Tax Eighteen percent.", vault, None)
    assert "Fifty" in _grounding_gap("Shipping is Fifty rupees.", {"evidence": [
        {"id": "V1", "file_id": 306, "filename": "x.pdf", "content": "No figures here."}]}, None)
