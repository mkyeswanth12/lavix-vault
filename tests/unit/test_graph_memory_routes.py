from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.graph_memory import routes
from app.graph_memory.models import (
    ClearMemoryResponse,
    GraphMemoryItem,
    GraphMemoryItemPage,
    GraphMemoryStatusResponse,
)
from app.graph_memory.repository import GraphMemoryConflictError, ItemMutation, TenantRecord
from app.graph_memory.service import deterministic_about_me


def graph_item(memory_id: UUID | None = None) -> GraphMemoryItem:
    now = datetime(2026, 7, 16, tzinfo=UTC)
    return GraphMemoryItem(
        id=memory_id or uuid4(),
        kind="fact",
        subject="I",
        predicate="live_in",
        object_value="Pune",
        confidence=0.95,
        status="active",
        source_excerpt="I live in Pune.",
        last_confirmed_at=now,
        expires_at=now + timedelta(days=90),
        revision=2,
        projection_state="projected",
        created_at=now,
        updated_at=now,
    )


class FakeService:
    def __init__(self):
        self.user_ids = []
        self.item = graph_item()
        self.tenant = TenantRecord(
            user_id=72,
            tenant_uuid=uuid4(),
            enabled=True,
            retention_days=90,
            generation=4,
            revision=3,
            graph_revision=5,
            purge_state="ready",
            last_learned_at=None,
        )

    def get_status(self, user_id):
        self.user_ids.append(user_id)
        return GraphMemoryStatusResponse(
            enabled=True,
            retention_days=90,
            revision=3,
            generation=4,
            purge_state="ready",
            last_learned_at=None,
            next_expiry_at=self.item.expires_at,
            counts={"active": 1, "pending": 0, "expired": 0},
            about_me=deterministic_about_me([self.item], graph_revision=5),
        )

    def update_settings(self, user_id, **_values):
        return self.get_status(user_id)

    def list_items(self, user_id, **values):
        self.user_ids.append(user_id)
        return GraphMemoryItemPage(
            items=[self.item], total=1, limit=values["limit"], offset=values["offset"]
        )

    def edit_item(self, user_id, memory_id, **_values):
        self.user_ids.append(user_id)
        self.item = self.item.model_copy(update={"id": memory_id})
        return ItemMutation(self.tenant, self.item)

    approve_item = edit_item
    renew_item = edit_item

    def delete_item(self, user_id, memory_id):
        return self.edit_item(user_id, memory_id)

    def clear_relationship_memory(self, user_id):
        self.user_ids.append(user_id)
        return ClearMemoryResponse(graph_deleted_count=1, generation=5, purge_state="pending")

    def clear_personal_memory(self, user_id):
        self.user_ids.append(user_id)
        return ClearMemoryResponse(
            graph_deleted_count=1,
            saved_preferences_deleted_count=2,
            generation=5,
            purge_state="pending",
        )


def client_for(service: FakeService, *, perm_ai: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/ai")
    app.dependency_overrides[routes.get_current_user] = lambda: {"id": 72, "perm_ai": perm_ai}
    app.dependency_overrides[routes.get_graph_memory_service] = lambda: service
    return TestClient(app)


def test_graph_memory_routes_are_authenticated_tenant_owned_and_documented() -> None:
    service = FakeService()
    client = client_for(service)

    status = client.get("/api/ai/graph-memory")
    assert status.status_code == 200
    assert status.json()["retention_days"] == 90
    assert status.json()["about_me"]["summary"]

    listing = client.get("/api/ai/graph-memory/items?status=active&kind=fact&limit=10")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    cleared = client.delete("/api/ai/personal-memory")
    assert cleared.json()["saved_preferences_deleted_count"] == 2
    assert service.user_ids == [72, 72, 72]

    paths = client.app.openapi()["paths"]
    assert {"get", "put", "delete"}.issubset(paths["/api/ai/graph-memory"])
    assert {"get"}.issubset(paths["/api/ai/graph-memory/items"])
    assert {"patch", "delete"}.issubset(paths["/api/ai/graph-memory/items/{memory_id}"])
    assert "post" in paths["/api/ai/graph-memory/items/{memory_id}/approve"]
    assert "post" in paths["/api/ai/graph-memory/items/{memory_id}/renew"]
    assert "delete" in paths["/api/ai/personal-memory"]


def test_settings_revision_conflict_is_a_stable_409() -> None:
    class ConflictService(FakeService):
        def update_settings(self, user_id, **_values):
            raise GraphMemoryConflictError("settings_revision_conflict")

    response = client_for(ConflictService()).put(
        "/api/ai/graph-memory",
        json={"enabled": True, "retention_days": 90, "expected_revision": 1},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "settings_revision_conflict"


def test_invalid_retention_and_item_page_are_rejected_before_service() -> None:
    client = client_for(FakeService())
    assert (
        client.put("/api/ai/graph-memory", json={"enabled": True, "retention_days": 60}).status_code
        == 422
    )
    assert client.get("/api/ai/graph-memory/items?limit=101").status_code == 422


def test_revoked_ai_permission_cannot_enable_but_can_clear_private_memory() -> None:
    client = client_for(FakeService(), perm_ai=False)
    denied = client.put(
        "/api/ai/graph-memory",
        json={"enabled": True, "retention_days": 90},
    )
    assert denied.status_code == 403
    assert client.delete("/api/ai/graph-memory").status_code == 200
    assert client.delete("/api/ai/personal-memory").status_code == 200


def test_graph_setting_repository_synchronizes_the_single_memory_consent() -> None:
    source = (Path(__file__).parents[2] / "app/graph_memory/repository.py").read_text()
    assert "UPDATE users SET memory_enabled = %s WHERE id = %s" in source
