"""Tests for the admin-configurable web search depth (search_depth tier).

Covers the tier -> budget resolution, the RunOptions wire field, and the
end-to-end threading through _prefetch_web_evidence: planner leg budget,
per-leg results, excerpt chars, and the multi-hop evidence pool cap.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_runtime import cuga_adapter
from agent_runtime.config import RuntimeSettings
from agent_runtime.cuga_adapter import SearchDepthSpec, SearchPlan, _validate_search_plan
from agent_runtime.schemas import RunOptions
from agent_runtime.scope import RunScope, bind_run_scope


class DepthFakeGateway:
    """Records every search_web call with its tier-driven knobs."""

    def __init__(self, web_result: dict[str, Any] | None = None) -> None:
        self.web_calls: list[tuple[str, int, int | None]] = []
        self.web_result = web_result or {
            "ok": True,
            "evidence": [
                {
                    "id": "W1",
                    "title": "Current source",
                    "url": "https://example.test/current",
                    "content": "bounded current-web evidence snippet passage with topical sentences. " * 6,
                    "untrusted": True,
                }
            ],
            "count": 1,
        }

    async def search_web(self, query, max_results=3, excerpt_chars=None):
        self.web_calls.append((query, max_results, excerpt_chars))
        return self.web_result

    async def close(self):
        pass


def make_adapter(gateway: DepthFakeGateway):
    return cuga_adapter.CugaAdapter(
        RuntimeSettings(),
        gateway,
        backend_loader=lambda: None,
        run_semaphore=asyncio.Semaphore(1),
    )


def planner_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "normalized_question": "What are G7, G20 and G2 in geopolitics?",
        "intent": "define geopolitical groupings",
        "entities": ["G7", "G20", "G2"],
        "sub_questions": ["What is the G7?", "What is the G20?"],
        "search_queries": [
            "G7 Group of Seven geopolitical grouping",
            "G20 Group of Twenty geopolitical forum",
        ],
        "source_preferences": ["britannica.com"],
        "ambiguities": [],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Tier -> budget resolution
# ---------------------------------------------------------------------------


def test_search_depth_spec_tiers() -> None:
    conservative = cuga_adapter.get_search_depth_spec("conservative")
    balanced = cuga_adapter.get_search_depth_spec("balanced")
    deep = cuga_adapter.get_search_depth_spec("deep")
    pro = cuga_adapter.get_search_depth_spec("pro")

    assert conservative == SearchDepthSpec(
        search_legs=3, evidence_pool_cap=6, results_per_leg=5, excerpt_chars=2000
    )
    assert balanced == SearchDepthSpec(
        search_legs=4, evidence_pool_cap=8, results_per_leg=6, excerpt_chars=3000
    )
    assert deep == SearchDepthSpec(
        search_legs=5, evidence_pool_cap=10, results_per_leg=8, excerpt_chars=4000
    )
    assert pro == SearchDepthSpec(
        search_legs=7, evidence_pool_cap=100, results_per_leg=12, excerpt_chars=6000
    )


@pytest.mark.parametrize("depth", [None, "", "ultra", "PRO "])
def test_search_depth_spec_falls_back_to_conservative(depth: str | None) -> None:
    spec = cuga_adapter.get_search_depth_spec(depth)

    assert spec.results_per_leg == 5
    assert spec.search_legs == 3
    assert spec.evidence_pool_cap == 6
    assert spec.excerpt_chars == 2000


# ---------------------------------------------------------------------------
# RunOptions wire field
# ---------------------------------------------------------------------------


def test_run_options_accepts_the_four_tiers() -> None:
    assert RunOptions().search_depth is None
    assert RunOptions(search_depth="conservative").search_depth == "conservative"
    assert RunOptions(search_depth="balanced").search_depth == "balanced"
    assert RunOptions(search_depth="deep").search_depth == "deep"
    assert RunOptions(search_depth="pro").search_depth == "pro"


@pytest.mark.parametrize("value", ["ultra", "Conservative", 5, True])
def test_run_options_rejects_non_tier_values(value: Any) -> None:
    with pytest.raises(ValueError):
        RunOptions(search_depth=value)


# ---------------------------------------------------------------------------
# Planner validation threading
# ---------------------------------------------------------------------------


def test_validate_search_plan_enforces_tier_leg_budget() -> None:
    five_queries = dict(
        planner_payload(),
        search_queries=[f"G7 geopolitical grouping fact {i}" for i in range(5)],
    )

    plan, reason = _validate_search_plan(
        five_queries, user_query="q", max_search_queries=5
    )
    assert reason == "", reason
    assert plan is not None
    assert len(plan.search_queries) == 5

    plan, reason = _validate_search_plan(
        five_queries, user_query="q", max_search_queries=3
    )
    assert plan is None
    assert reason == "too-many-queries"


def test_validate_search_plan_defaults_to_conservative_budget() -> None:
    four_queries = dict(
        planner_payload(),
        search_queries=[f"G7 geopolitical grouping fact {i}" for i in range(4)],
    )

    # No explicit budget: the conservative tier's 3-leg cap applies, so a
    # 4-query plan is rejected exactly as before this feature existed.
    plan, reason = _validate_search_plan(four_queries, user_query="q")
    assert plan is None
    assert reason == "too-many-queries"


# ---------------------------------------------------------------------------
# _execute_planner_legs threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_planner_legs_respects_tier_budget() -> None:
    plan = SearchPlan(
        normalized_question="What are G7, G20 and G2?",
        intent="define geopolitical groupings",
        entities=("G7", "G20", "G2"),
        sub_questions=("What is the G7?",),
        search_queries=tuple(f"G7 geopolitical grouping fact {i}" for i in range(5)),
        source_preferences=(),
        ambiguities=(),
    )
    adapter = make_adapter(DepthFakeGateway())

    scope = RunScope(run_id="00000000-0000-0000-0000-0000000000d1", capability_token="opaque-d1")
    with bind_run_scope(scope):
        await adapter._execute_planner_legs(
            plan, "what are g7 g20 and g2", spec=cuga_adapter.get_search_depth_spec("deep")
        )
    # Deep tier: all 5 legs execute with 8 results/leg and 4000-char excerpts.
    assert len(adapter.gateway.web_calls) == 5
    assert all(call[1] == 8 and call[2] == 4000 for call in adapter.gateway.web_calls)

    adapter = make_adapter(DepthFakeGateway())
    scope = RunScope(run_id="00000000-0000-0000-0000-0000000000d2", capability_token="opaque-d2")
    with bind_run_scope(scope):
        await adapter._execute_planner_legs(
            plan, "what are g7 g20 and g2", spec=cuga_adapter.get_search_depth_spec("conservative")
        )
    # Conservative tier: the 5-query plan is capped at 3 legs.
    assert len(adapter.gateway.web_calls) == 3
    assert all(call[1] == 5 and call[2] == 2000 for call in adapter.gateway.web_calls)


# ---------------------------------------------------------------------------
# _prefetch_web_evidence end-to-end threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_web_evidence_default_is_conservative() -> None:
    gateway = DepthFakeGateway()
    adapter = make_adapter(gateway)
    scope = RunScope(run_id="00000000-0000-0000-0000-0000000000c1", capability_token="opaque-c1")
    with bind_run_scope(scope):
        result = await adapter._prefetch_web_evidence("What is the current stable release?")

    assert result is not None
    # Deterministic single-hop path: one call with the conservative budget.
    assert len(gateway.web_calls) == 1
    query, max_results, excerpt_chars = gateway.web_calls[0]
    assert query
    assert max_results == 5
    assert excerpt_chars == 2000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tier, results_per_leg, excerpt_chars",
    [("conservative", 5, 2000), ("balanced", 6, 3000), ("deep", 8, 4000)],
)
async def test_prefetch_web_evidence_per_tier_knobs(
    tier: str, results_per_leg: int, excerpt_chars: int
) -> None:
    gateway = DepthFakeGateway()
    adapter = make_adapter(gateway)
    scope = RunScope(run_id="00000000-0000-0000-0000-0000000000c2", capability_token="opaque-c2")
    with bind_run_scope(scope):
        result = await adapter._prefetch_web_evidence(
            "What is the current stable release?", search_depth=tier
        )

    assert result is not None
    assert len(gateway.web_calls) == 1
    _, max_results, chars = gateway.web_calls[0]
    assert max_results == results_per_leg
    assert chars == excerpt_chars


@pytest.mark.asyncio
async def test_prefetch_web_evidence_planner_legs_and_leg_queries_map(monkeypatch) -> None:
    five_query_plan = SearchPlan(
        normalized_question="What are G7, G20 and G2?",
        intent="define geopolitical groupings",
        entities=("G7", "G20", "G2"),
        sub_questions=("What is the G7?",),
        search_queries=tuple(f"G7 geopolitical grouping fact {i}" for i in range(5)),
        source_preferences=(),
        ambiguities=(),
    )

    async def fake_plan(*args: Any, **kwargs: Any) -> tuple[SearchPlan | None, str]:
        return five_query_plan, ""

    monkeypatch.setattr(cuga_adapter, "_plan_search_queries", fake_plan)

    gateway = DepthFakeGateway()
    adapter = make_adapter(gateway)
    scope = RunScope(run_id="00000000-0000-0000-0000-0000000000c3", capability_token="opaque-c3")
    with bind_run_scope(scope):
        result = await adapter._prefetch_web_evidence(
            "What are G7, G20 and G2?", search_depth="deep"
        )

    assert result is not None
    # Deep tier: all 5 planned legs execute and all 5 leg_queries entries
    # (leg-specific semantic verification) survive.
    assert len(gateway.web_calls) == 5
    assert len(result["leg_queries"]) == 5
    assert set(result["leg_queries"]) == {"leg_1", "leg_2", "leg_3", "leg_4", "leg_5"}
    assert all(call[1] == 8 and call[2] == 4000 for call in gateway.web_calls)


@pytest.mark.asyncio
async def test_prefetch_web_evidence_planner_legs_capped_conservative(monkeypatch) -> None:
    five_query_plan = SearchPlan(
        normalized_question="What are G7, G20 and G2?",
        intent="define geopolitical groupings",
        entities=("G7", "G20", "G2"),
        sub_questions=("What is the G7?",),
        search_queries=tuple(f"G7 geopolitical grouping fact {i}" for i in range(5)),
        source_preferences=(),
        ambiguities=(),
    )

    async def fake_plan(*args: Any, **kwargs: Any) -> tuple[SearchPlan | None, str]:
        return five_query_plan, ""

    monkeypatch.setattr(cuga_adapter, "_plan_search_queries", fake_plan)

    gateway = DepthFakeGateway()
    adapter = make_adapter(gateway)
    scope = RunScope(run_id="00000000-0000-0000-0000-0000000000c4", capability_token="opaque-c4")
    with bind_run_scope(scope):
        result = await adapter._prefetch_web_evidence(
            "What are G7, G20 and G2?", search_depth="conservative"
        )

    assert result is not None
    assert len(gateway.web_calls) == 3
    assert len(result["leg_queries"]) == 3
    assert set(result["leg_queries"]) == {"leg_1", "leg_2", "leg_3"}
    assert all(call[1] == 5 and call[2] == 2000 for call in gateway.web_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tier, pool_cap",
    [("conservative", 6), ("balanced", 8), ("deep", 10)],
)
async def test_prefetch_web_evidence_pool_cap_per_tier(tier: str, pool_cap: int) -> None:
    # Five unflagged results per hop, two hops -> a 10-item raw pool. The
    # multi-hop relevance gate passes everything (unflagged items are kept),
    # so the tier's evidence_pool_cap is the only bound on the forwarded pool.
    many_results = {
        "ok": True,
        "evidence": [
            {
                "id": f"W{i}",
                "title": f"Source {i}",
                "url": f"https://example.test/{i}",
                "content": "facts",
                "untrusted": True,
            }
            for i in range(5)
        ],
        "count": 5,
    }
    gateway = DepthFakeGateway(many_results)
    adapter = make_adapter(gateway)
    scope = RunScope(run_id="00000000-0000-0000-0000-0000000000c5", capability_token="opaque-c5")
    with bind_run_scope(scope):
        result = await adapter._prefetch_web_evidence(
            "What is the Group of Seven? And what is the Group of Twenty?",
            search_depth=tier,
        )

    assert result is not None
    # Two decomposed hops executed, each with the tier's per-leg budget.
    assert len(gateway.web_calls) == 2
    assert all(call[1] == cuga_adapter.get_search_depth_spec(tier).results_per_leg for call in gateway.web_calls)
    assert result["count"] == pool_cap
