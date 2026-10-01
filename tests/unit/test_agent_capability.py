import asyncio
from uuid import uuid4

import pytest

from app.agent.capability import CapabilityError, CapabilitySigner
from app.agent.evidence import RunEvidenceStore

SECRET = "s" * 32


def test_default_capability_lifetime_covers_agent_window():
    signer = CapabilitySigner(SECRET, clock=lambda: 1_000.0)
    run_id = str(uuid4())

    token = signer.mint(
        run_id=run_id,
        user_id=1,
        file_ids=None,
        web_search_enabled=False,
        deep_search=False,
    )

    scope = signer.verify(token, expected_run_id=run_id)
    assert scope.expires_at - scope.issued_at == 300


def test_capability_round_trip_preserves_authoritative_scope():
    now = [1_000.0]
    signer = CapabilitySigner(SECRET, ttl_seconds=60, clock=lambda: now[0])
    run_id = str(uuid4())

    token = signer.mint(
        run_id=run_id,
        user_id=17,
        file_ids=[9, 3, 9],
        web_search_enabled=False,
        deep_search=True,
    )
    scope = signer.verify(token, expected_run_id=run_id)

    assert scope.user_id == 17
    assert scope.file_ids == (9, 3)
    assert scope.web_search_enabled is False
    assert scope.deep_search is True
    assert scope.expires_at == 1_060


def test_capability_rejects_tampering_expiry_and_run_mismatch():
    now = [1_000.0]
    signer = CapabilitySigner(SECRET, ttl_seconds=10, clock=lambda: now[0])
    run_id = str(uuid4())
    token = signer.mint(
        run_id=run_id,
        user_id=1,
        file_ids=None,
        web_search_enabled=True,
        deep_search=False,
    )

    payload, signature = token.split(".")
    with pytest.raises(CapabilityError, match="signature"):
        signer.verify(f"{payload[:-1]}A.{signature}")
    with pytest.raises(CapabilityError, match="run mismatch"):
        signer.verify(token, expected_run_id=str(uuid4()))
    now[0] = 1_011
    with pytest.raises(CapabilityError, match="expired"):
        signer.verify(token)


def test_capability_validates_secret_subject_and_file_scope():
    with pytest.raises(ValueError, match="32 bytes"):
        CapabilitySigner("short")
    signer = CapabilitySigner(SECRET)
    with pytest.raises(CapabilityError, match="subject"):
        signer.mint(
            run_id=str(uuid4()),
            user_id=0,
            file_ids=None,
            web_search_enabled=False,
            deep_search=False,
        )
    with pytest.raises(CapabilityError, match="file scope"):
        signer.mint(
            run_id=str(uuid4()),
            user_id=1,
            file_ids=[-1],
            web_search_enabled=False,
            deep_search=False,
        )


def test_evidence_store_assigns_stable_deduplicated_ids():
    async def scenario():
        store = RunEvidenceStore(max_entries=3)
        first = await store.record(
            "run-1",
            "vault",
            [{"file_id": 2, "revision": 1, "chunk_id": "c1", "content": "first"}],
        )
        again = await store.record(
            "run-1",
            "vault",
            [
                {"file_id": 2, "revision": 1, "chunk_id": "c1", "content": "duplicate"},
                {"file_id": 2, "revision": 1, "chunk_id": "c2", "content": "second"},
            ],
        )
        web = await store.record("run-1", "web", [{"url": "https://example.test"}])
        return first, again, web, await store.get("run-1")

    first, again, web, all_evidence = asyncio.run(scenario())
    assert first[0]["id"] == "V1"
    assert [item["id"] for item in again] == ["V1", "V2"]
    assert web[0]["id"] == "W1"
    assert len(all_evidence) == 3
    assert all(item["untrusted"] for item in all_evidence)


def test_capability_round_trip_preserves_chat_mode():
    signer = CapabilitySigner(SECRET, clock=lambda: 1_000.0)
    run_id = str(uuid4())
    token = signer.mint(
        run_id=run_id,
        user_id=1,
        file_ids=None,
        web_search_enabled=False,
        deep_search=False,
        chat_mode="expert",
    )
    assert signer.verify(token, expected_run_id=run_id).chat_mode == "expert"


def test_capability_mode_defaults_none_and_rejects_garbage():
    signer = CapabilitySigner(SECRET, clock=lambda: 1_000.0)
    run_id = str(uuid4())
    token = signer.mint(
        run_id=run_id,
        user_id=1,
        file_ids=None,
        web_search_enabled=False,
        deep_search=False,
    )
    assert signer.verify(token, expected_run_id=run_id).chat_mode is None
    with pytest.raises(CapabilityError):
        signer.mint(
            run_id=run_id,
            user_id=1,
            file_ids=None,
            web_search_enabled=False,
            deep_search=False,
            chat_mode="verbose",
        )


def test_capability_token_survives_prefetch_plus_tool_call_reuse():
    """One run calls search-vault twice (prefetch + model tool call)."""
    signer = CapabilitySigner(SECRET, clock=lambda: 1_000.0)
    run_id = str(uuid4())
    token = signer.mint(
        run_id=run_id,
        user_id=1,
        file_ids=[1],
        web_search_enabled=False,
        deep_search=False,
    )
    first = signer.verify(token, expected_run_id=run_id)
    second = signer.verify(token, expected_run_id=run_id)
    assert first.run_id == second.run_id == run_id
    assert first.file_ids == second.file_ids == (1,)


def test_capability_foreign_project_secret_rejected_as_signature():
    """Cross-project alias collision: another stack's secret must fail closed."""
    mine = CapabilitySigner(SECRET, clock=lambda: 1_000.0)
    theirs = CapabilitySigner("t" * 32, clock=lambda: 1_000.0)
    run_id = str(uuid4())
    token = mine.mint(
        run_id=run_id,
        user_id=1,
        file_ids=[1],
        web_search_enabled=False,
        deep_search=False,
    )
    with pytest.raises(CapabilityError, match="signature"):
        theirs.verify(token, expected_run_id=run_id)


def test_capability_concurrent_runs_verify_independently():
    """Distinct runs must never validate against each other's run id."""
    import threading

    signer = CapabilitySigner(SECRET, clock=lambda: 1_000.0)
    errors: list[str] = []

    def one(_: int) -> None:
        try:
            run_id = str(uuid4())
            token = signer.mint(
                run_id=run_id,
                user_id=1,
                file_ids=None,
                web_search_enabled=False,
                deep_search=False,
            )
            scope = signer.verify(token, expected_run_id=run_id)
            assert scope.run_id == run_id
            with pytest.raises(CapabilityError, match="run mismatch"):
                signer.verify(token, expected_run_id=str(uuid4()))
        except Exception as exc:  # noqa: BLE001 - collected, asserted below
            errors.append(str(exc))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_capability_expiry_reason_is_explicit_for_logging():
    """The denial reason must distinguish expiry (for secret-free log lines)."""
    now = [1_000.0]
    signer = CapabilitySigner(SECRET, ttl_seconds=10, clock=lambda: now[0])
    run_id = str(uuid4())
    token = signer.mint(
        run_id=run_id,
        user_id=1,
        file_ids=None,
        web_search_enabled=False,
        deep_search=False,
    )
    now[0] = 1_011.0
    with pytest.raises(CapabilityError, match="expired"):
        signer.verify(token, expected_run_id=run_id)
