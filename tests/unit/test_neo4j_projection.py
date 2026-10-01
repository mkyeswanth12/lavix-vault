"""Phase 2 Neo4j projection: PG-canonical, tenant-scoped, fail-open.

RED-first: neo4j_projection module, service wiring, and repository helpers
do not exist yet. All driver interaction goes through an injected fake;
no test requires a live Neo4j.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.graph_memory.models import MemoryKind, MemoryStatus
from app.graph_memory.neo4j_projection import (
    Neo4jProjection,
    Neo4jProjectionRefused,
    Neo4jUnavailable,
    ProjectableItem,
)


def active_item(**overrides):
    base = {
        "user_id": 41,
        "tenant_uuid": "11111111-1111-4111-8111-111111111111",
        "fingerprint": "a" * 64,
        "kind": "fact",
        "subject": "I",
        "predicate": "name",
        "object_value": "mk",
        "status": "active",
    }
    base.update(overrides)
    return ProjectableItem(**base)


class RecordingSession:
    def __init__(self):
        self.runs = []

    def run(self, cypher, params=None):
        self.runs.append((str(cypher), dict(params or {})))
        return []

    def close(self):
        pass


class RecordingDriver:
    def __init__(self, session=None):
        self.session_obj = session or RecordingSession()
        self.sessions_opened = 0
        self.closed = False

    def session(self, **kwargs):
        self.sessions_opened += 1
        return self.session_obj

    def close(self):
        self.closed = True


def projection(session=None):
    return Neo4jProjection(
        uri="bolt://neo4j:7687",
        user="neo4j",
        password="test-only",
        driver=RecordingDriver(session),
    )


def queries(proj):
    return proj._driver.session_obj.runs


# ---------------------------------------------------------------------------
# Upsert: idempotent, tenant-scoped, ACTIVE-only
# ---------------------------------------------------------------------------


def test_upsert_merges_user_memory_and_edges_keyed_by_fingerprint():
    proj = projection()
    proj.upsert_item(active_item())
    runs = queries(proj)
    assert runs, "upsert must emit Cypher"
    blob = "\n".join(cypher for cypher, _ in runs)
    assert "MERGE" in blob
    assert "a" * 64 in str(runs), "fingerprint must key the MERGE"
    for _, params in runs:
        assert params.get("user_id") == 41, f"every query carries the tenant user id: {params}"
    assert any("HAS_MEMORY" in cypher for cypher, _ in runs)


def test_upsert_twice_emits_same_idempotent_shape():
    proj = projection()
    proj.upsert_item(active_item())
    first = [c for c, _ in queries(proj)]
    proj.upsert_item(active_item())
    second = [c for c, _ in queries(proj)][len(first):]
    assert second == first


def test_upsert_refuses_non_active_items():
    proj = projection()
    for status in ("pending", "rejected", "expired"):
        with pytest.raises(Neo4jProjectionRefused):
            proj.upsert_item(active_item(status=status))
    assert queries(proj) == [], "refused items must never reach the driver"


def test_upsert_refuses_secret_bearing_text():
    proj = projection()
    with pytest.raises(Neo4jProjectionRefused):
        proj.upsert_item(active_item(object_value="hunter2 my password is hunter2"))
    assert queries(proj) == []


def test_relationship_item_projects_about_edges():
    proj = projection()
    proj.upsert_item(
        active_item(kind="relationship", subject="mk", predicate="knows", object_value="sam")
    )
    blob = "\n".join(cypher for cypher, _ in queries(proj))
    assert "ABOUT" in blob


# ---------------------------------------------------------------------------
# Deletes: tenant-scoped, edges go with the node
# ---------------------------------------------------------------------------


def test_delete_item_detaches_scoped_node():
    proj = projection()
    proj.delete_item(
        tenant_uuid="11111111-1111-4111-8111-111111111111",
        user_id=41,
        fingerprint="a" * 64,
    )
    runs = queries(proj)
    assert runs
    cypher, params = runs[0]
    assert "DETACH DELETE" in cypher
    assert params["user_id"] == 41
    assert params["fingerprint"] == "a" * 64
    assert params["tenant_uuid"] == "11111111-1111-4111-8111-111111111111"


def test_delete_tenant_is_user_scoped():
    proj = projection()
    proj.delete_tenant(
        tenant_uuid="11111111-1111-4111-8111-111111111111", user_id=41
    )
    runs = queries(proj)
    assert runs
    cypher, params = runs[0]
    assert "DETACH DELETE" in cypher
    assert params["user_id"] == 41
    assert params["tenant_uuid"] == "11111111-1111-4111-8111-111111111111"


# ---------------------------------------------------------------------------
# Constraints + reader timeout
# ---------------------------------------------------------------------------


def test_ensure_constraints_pins_user_and_fingerprint_uniqueness():
    proj = projection()
    proj.ensure_constraints()
    blob = "\n".join(cypher for cypher, _ in queries(proj))
    assert "CONSTRAINT" in blob
    assert "u.user_id" in blob
    # Fingerprint uniqueness per user rides the composite m.key
    # (user_id:fingerprint); version-proof across Neo4j 5.x.
    assert "m.key" in blob


class HangingSession(RecordingSession):
    def run(self, cypher, params=None):
        import time

        time.sleep(5)
        return []


def test_slow_driver_surfaces_unavailable_for_fail_open():
    proj = Neo4jProjection(
        uri="bolt://neo4j:7687",
        user="neo4j",
        password="test-only",
        driver=RecordingDriver(HangingSession()),
        timeout_seconds=0.05,
    )
    with pytest.raises(Neo4jUnavailable):
        proj.read_related(
            user_id=41,
            tenant_uuid="11111111-1111-4111-8111-111111111111",
            terms=["mk"],
            limit=4,
        )


# ---------------------------------------------------------------------------
# Service wiring: state truly moves only on driver results
# ---------------------------------------------------------------------------


def _tenant(generation=1):
    return SimpleNamespace(
        user_id=41,
        tenant_uuid="11111111-1111-4111-8111-111111111111",
        generation=generation,
        purge_state="ready",
    )


def _mutation(status="active"):
    from app.graph_memory.models import GraphMemoryItem

    row = GraphMemoryItem.model_validate(
        {
            "id": uuid4(),
            "kind": "fact",
            "subject": "I",
            "predicate": "name",
            "object_value": "mk",
            "confidence": 0.95,
            "status": status,
            "source_excerpt": "my name is mk",
            "last_confirmed_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2027-01-01T00:00:00+00:00",
            "revision": 1,
            "projection_state": "pending",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    )
    return SimpleNamespace(tenant=_tenant(), item=row, job_id=uuid4())


def test_service_marks_projected_only_after_driver_success():
    from app.graph_memory.service import GraphMemoryService

    proj = projection()
    completed = []

    class Repo:
        def get_tenant(self, user_id):
            return _tenant()

        def get_projectable_item(self, user_id, memory_id):
            return active_item()

        def mark_item_projection_complete(self, *args, **kwargs):
            completed.append(True)
            return _tenant()

    service = GraphMemoryService(repository=Repo())  # type: ignore[arg-type]
    service.projection = proj
    service.project_item(_mutation())
    assert completed, "driver success must complete the projection"
    assert queries(proj), "driver must have been called"


def test_service_never_projects_pending_items():
    from app.graph_memory.service import GraphMemoryService

    proj = projection()
    completed = []

    class Repo:
        def get_tenant(self, user_id):
            return _tenant()

        def get_projectable_item(self, user_id, memory_id):
            return active_item(status="pending")

        def mark_item_projection_complete(self, *args, **kwargs):
            completed.append(True)
            return _tenant()

    service = GraphMemoryService(repository=Repo())  # type: ignore[arg-type]
    service.projection = proj
    service.project_item(_mutation(status="pending"))
    assert completed, "job still completes"
    assert queries(proj) == [], "pending rows must never reach Neo4j"


def test_service_driver_failure_propagates_for_backoff_retry():
    from app.graph_memory.service import GraphMemoryService

    class FailingDriver(RecordingDriver):
        def session(self, **kwargs):
            raise ConnectionError("neo4j down")

    proj = Neo4jProjection(
        uri="bolt://neo4j:7687",
        user="neo4j",
        password="test-only",
        driver=FailingDriver(),
    )
    completed = []

    class Repo:
        def get_tenant(self, user_id):
            return _tenant()

        def get_projectable_item(self, user_id, memory_id):
            return active_item()

        def mark_item_projection_complete(self, *args, **kwargs):
            completed.append(True)
            return _tenant()

    service = GraphMemoryService(repository=Repo())  # type: ignore[arg-type]
    service.projection = proj
    with pytest.raises(Neo4jUnavailable):
        service.project_item(_mutation())
    assert completed == [], "failed projection must not complete"


def test_recall_falls_back_to_postgres_when_graph_is_down():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from app.graph_memory.service import GraphMemoryService

    remembered = SimpleNamespace(
        id=uuid4(),
        kind=MemoryKind.FACT,
        subject="I",
        predicate="name",
        object_value="mk",
        confidence=0.95,
        status=MemoryStatus.ACTIVE,
        expires_at=datetime(2027, 1, 1, tzinfo=ZoneInfo("UTC")),
    )

    class FailingDriver(RecordingDriver):
        def session(self, **kwargs):
            raise ConnectionError("neo4j down")

    service = GraphMemoryService(
        repository=SimpleNamespace(
            active_items_for_profile=lambda user_id, *, limit=100: (remembered,),
            fetch_active_by_fingerprints=lambda user_id, fps: (),
        )
    )  # type: ignore[arg-type]
    service.projection = Neo4jProjection(
        uri="bolt://neo4j:7687",
        user="neo4j",
        password="test-only",
        driver=FailingDriver(),
    )
    assert [r.object_value for r in service.recall(41, "what is my name?")] == ["mk"]


# ---------------------------------------------------------------------------
# Drift report: counts both directions, heals only via the upsert path
# ---------------------------------------------------------------------------


def test_report_drift_counts_and_heals_via_upsert_jobs():
    from app.graph_memory.repository import ProjectionSnapshotRow
    from app.graph_memory.worker import GraphProjectionReconciler

    tenant_uuid = uuid4()
    queued = []

    class Repo:
        def get_tenant(self, uid):
            return SimpleNamespace(user_id=uid, tenant_uuid=tenant_uuid)

        def active_projection_snapshot(self, uid):
            return {
                "a" * 64: ProjectionSnapshotRow(uuid4(), "I", "name", "mk"),
                "b" * 64: ProjectionSnapshotRow(uuid4(), "I", "city", "Hyd"),
                "s" * 64: ProjectionSnapshotRow(
                    uuid4(), "I", "password", "x", "secret_suspect"
                ),
            }

        def queue_projection_job(self, uid, memory_id, operation="upsert"):
            assert operation == "upsert"
            queued.append(memory_id)
            return uuid4()

    class Proj:
        def fingerprint_set(self, *, user_id, tenant_uuid):
            return {"a" * 64, "z" * 64}

        def fetch_node(self, *, user_id, tenant_uuid, fingerprint):
            return {"subject": "I", "predicate": "name", "object_value": "WRONG"}

    class Control:
        def tenant_user_ids(self, uid):
            return (41,)

    service = SimpleNamespace(repository=Repo(), projection=Proj())
    report = GraphProjectionReconciler(service, control=Control()).report_drift()
    assert report["tenants"] == 1
    assert report["missing_in_neo4j"] == 1
    assert report["neo4j_orphans"] == 1
    assert report["diverged"] == 1
    assert report["healed"] == 2
    assert report["skipped_secret"] == 1
    assert len(queued) == 2
    assert report["failures"] == []


def test_report_drift_without_projection_refuses():
    from app.graph_memory.worker import GraphProjectionReconciler

    service = SimpleNamespace(repository=SimpleNamespace(), projection=None)
    with pytest.raises(RuntimeError, match="not configured"):
        GraphProjectionReconciler(service).report_drift()


def test_service_delete_uses_payload_fingerprint_when_pg_row_is_gone():
    from app.graph_memory.repository import GraphMemoryNotFoundError
    from app.graph_memory.service import GraphMemoryService

    proj = projection()
    completed = []

    class Repo:
        def get_tenant(self, user_id):
            return _tenant()

        def get_projectable_item(self, user_id, memory_id):
            return None  # API already deleted the canonical row

        def delete_item(self, user_id, memory_id):
            raise GraphMemoryNotFoundError("memory_item_not_found")

        def mark_item_projection_complete(self, *args, **kwargs):
            completed.append(kwargs)
            return _tenant()

    service = GraphMemoryService(repository=Repo())  # type: ignore[arg-type]
    service.projection = proj
    mutation = _mutation()
    mutation.item = SimpleNamespace(id=mutation.item.id, fingerprint="a" * 64)
    service.project_item(mutation, deleted=True)
    assert completed and completed[0].get("deleted") is True
    cypher, params = queries(proj)[0]
    assert "DETACH DELETE" in cypher
    assert params["fingerprint"] == "a" * 64


def test_recheck_attaches_late_projection(monkeypatch):
    from types import SimpleNamespace

    from app.graph_memory import runtime as rt

    monkeypatch.setenv("GRAPH_MEMORY_NEO4J_ENABLED", "true")
    fake = object()
    monkeypatch.setattr(rt, "build_neo4j_projection", lambda: fake)
    service = SimpleNamespace(projection=None)
    rt.recheck_neo4j_projection(service)
    assert service.projection is fake


def test_recheck_noop_when_flag_off_or_attached(monkeypatch):
    from types import SimpleNamespace

    from app.graph_memory import runtime as rt

    monkeypatch.delenv("GRAPH_MEMORY_NEO4J_ENABLED", raising=False)
    service = SimpleNamespace(projection=None)
    rt.recheck_neo4j_projection(service)
    assert service.projection is None
    attached = object()
    service2 = SimpleNamespace(projection=attached)
    monkeypatch.setenv("GRAPH_MEMORY_NEO4J_ENABLED", "true")
    rt.recheck_neo4j_projection(service2)
    assert service2.projection is attached


def test_build_defaults_to_disabled_without_env(monkeypatch):
    from app.graph_memory import runtime as rt

    monkeypatch.delenv("GRAPH_MEMORY_NEO4J_ENABLED", raising=False)
    assert rt.build_neo4j_projection() is None


# ---------------------------------------------------------------------------
# Phase 5: question topics (flag-gated, Neo4j-only, TTL, synthesis-only)
# ---------------------------------------------------------------------------


def test_question_topic_extraction_shapes():
    from app.graph_memory.question_topics import extract_question_topics

    assert extract_question_topics("Who designed the Eiffel Tower?") == ("Eiffel Tower",)
    assert extract_question_topics("What is the ICC ranking process?") == ("ICC",)
    assert "blorpt-gate" in extract_question_topics('is the "blorpt-gate" pushed?')
    assert extract_question_topics("hi") == ()
    assert extract_question_topics("my password is hunter2?") == ()
    assert extract_question_topics("is my back pain serious?") == ()
    assert extract_question_topics("does Alexei work at Initech?") == ()


def test_question_topics_skipped_when_flag_off(monkeypatch):
    from types import SimpleNamespace

    from app.graph_memory import service as service_module

    monkeypatch.delenv("GRAPH_QUESTION_MEMORY_ENABLED", raising=False)
    stored = []

    class Proj:
        def store_question_topics(self, **kwargs):
            stored.append(kwargs)

    service = service_module.GraphMemoryService(
        repository=SimpleNamespace(
            get_tenant=lambda uid: SimpleNamespace(
                user_id=uid,
                tenant_uuid="11111111-1111-4111-8111-111111111111",
                enabled=True,
            )
        ),
        projection=Proj(),
    )
    assert service.record_question_topics(41, "Who designed the Eiffel Tower?") == ()
    assert stored == []


def test_question_topics_require_consent_and_store_with_ttl(monkeypatch):
    from types import SimpleNamespace

    from app.graph_memory import service as service_module

    monkeypatch.setenv("GRAPH_QUESTION_MEMORY_ENABLED", "true")
    stored = []

    class Proj:
        def store_question_topics(self, *, user_id, tenant_uuid, topics):
            stored.append((user_id, tenant_uuid, topics))

    service = service_module.GraphMemoryService(
        repository=SimpleNamespace(
            get_tenant=lambda uid: SimpleNamespace(
                user_id=uid,
                tenant_uuid="11111111-1111-4111-8111-111111111111",
                enabled=True,
            )
        ),
        projection=Proj(),
    )
    assert service.record_question_topics(41, "Who designed the Eiffel Tower?") == (
        "Eiffel Tower",
    )
    assert stored and stored[0][2] == ["Eiffel Tower"]

    off = SimpleNamespace(
        user_id=41,
        tenant_uuid="11111111-1111-4111-8111-111111111111",
        enabled=False,
    )
    service2 = service_module.GraphMemoryService(
        repository=SimpleNamespace(get_tenant=lambda uid: off),
        projection=Proj(),
    )
    assert service2.record_question_topics(41, "Who designed the Eiffel Tower?") == ()


def test_topic_store_and_delete_cypher_shape():
    proj = projection()
    proj.store_question_topics(
        user_id=41,
        tenant_uuid="11111111-1111-4111-8111-111111111111",
        topics=("Eiffel Tower",),
    )
    blob = "\n".join(c for c, _ in queries(proj))
    assert "Topic" in blob
    assert "HAS_TOPIC" in blob
    assert "expires_at" in blob
    for _, params in queries(proj):
        assert params.get("user_id") == 41
    proj2 = projection()
    proj2.delete_topic(user_id=41, name="Eiffel Tower")
    cypher, params = queries(proj2)[0]
    assert "DETACH DELETE" in cypher
    assert params["user_id"] == 41


def test_graph_memory_settings_read_yml_when_env_absent(monkeypatch, tmp_path):
    from app import config as config_module

    cfg = tmp_path / "config.yml"
    cfg.write_text(
        "graph_memory:\n"
        "  neo4j_enabled: true\n"
        "  neo4j_uri: bolt://custom:7687\n"
        "  question_memory_enabled: true\n"
    )
    monkeypatch.setenv("LAVIX_CONFIG_FILE", str(cfg))
    for var in (
        "GRAPH_MEMORY_NEO4J_ENABLED",
        "GRAPH_MEMORY_NEO4J_URI",
        "GRAPH_MEMORY_NEO4J_USER",
        "GRAPH_MEMORY_NEO4J_PASSWORD_FILE",
        "GRAPH_MEMORY_NEO4J_DATABASE",
        "GRAPH_MEMORY_NEO4J_TIMEOUT_SECONDS",
        "GRAPH_QUESTION_MEMORY_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    config_module.reload_config()
    try:
        settings = config_module.Settings()
        assert settings.graph_memory_neo4j_enabled is True
        assert settings.graph_memory_neo4j_uri == "bolt://custom:7687"
        assert settings.graph_memory_neo4j_user == "neo4j"
        assert settings.graph_question_memory_enabled is True
    finally:
        config_module.reload_config()


def test_graph_memory_settings_default_off(monkeypatch):
    from app import config as config_module

    monkeypatch.setenv("LAVIX_CONFIG_FILE", "/nonexistent/lavix-config.yml")
    for var in ("GRAPH_MEMORY_NEO4J_ENABLED", "GRAPH_QUESTION_MEMORY_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    config_module.reload_config()
    try:
        settings = config_module.Settings()
        assert settings.graph_memory_neo4j_enabled is False
        assert settings.graph_question_memory_enabled is False
    finally:
        config_module.reload_config()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("neo4j/s3cr3t-value", "s3cr3t-value"),
        ("neo4j/a=b/c", "a=b/c"),
        ("bare-password", "bare-password"),
        ("", ""),
        ("neo4j/", ""),
    ],
)
def test_auth_flat_password_split(raw, expected):
    from app.graph_memory.runtime import _password_from_auth_value

    assert _password_from_auth_value(raw) == expected
