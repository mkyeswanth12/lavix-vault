"""Authenticated public API for user-owned relationship memory."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import get_current_user

from .models import (
    MAX_ITEM_PAGE_SIZE,
    ClearMemoryResponse,
    DeleteItemResponse,
    GraphMemoryItemEditRequest,
    GraphMemoryItemPage,
    GraphMemorySettingsRequest,
    GraphMemoryStatusResponse,
    MemoryKind,
    MemoryStatus,
    MutationResponse,
)
from .repository import (
    GraphMemoryConflictError,
    GraphMemoryDisabledError,
    GraphMemoryNotFoundError,
    GraphMemoryValidationError,
)
from .service import GraphMemoryService

router = APIRouter()

CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]


def get_graph_memory_service() -> GraphMemoryService:
    # Process-shared runtime service: projection + embedder attached when
    # configured. Route handlers only write PostgreSQL and queue jobs
    # (the worker projects), so sharing is safe; topic controls need the
    # live projection in the request path.
    from .runtime import graph_memory_runtime_service

    return graph_memory_runtime_service()


MemoryService = Annotated[GraphMemoryService, Depends(get_graph_memory_service)]


def _raise_public_error(exc: Exception) -> None:
    if isinstance(exc, GraphMemoryNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (GraphMemoryConflictError, GraphMemoryDisabledError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, GraphMemoryValidationError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


@router.get("/graph-memory", response_model=GraphMemoryStatusResponse)
def get_graph_memory(user: CurrentUser, service: MemoryService) -> GraphMemoryStatusResponse:
    return service.get_status(int(user["id"]))


@router.put("/graph-memory", response_model=GraphMemoryStatusResponse)
def update_graph_memory(
    body: GraphMemorySettingsRequest,
    user: CurrentUser,
    service: MemoryService,
) -> GraphMemoryStatusResponse:
    if body.enabled and user.get("perm_ai") is not True:
        raise HTTPException(status_code=403, detail="AI access permission denied")
    try:
        return service.update_settings(
            int(user["id"]),
            enabled=body.enabled,
            retention_days=body.retention_days,
            expected_revision=body.expected_revision,
        )
    except Exception as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable") from exc


@router.delete("/graph-memory", response_model=ClearMemoryResponse)
def clear_graph_memory(user: CurrentUser, service: MemoryService) -> ClearMemoryResponse:
    try:
        return service.clear_relationship_memory(int(user["id"]))
    except Exception as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable") from exc


@router.get("/graph-memory/items", response_model=GraphMemoryItemPage)
def list_graph_memory_items(
    user: CurrentUser,
    service: MemoryService,
    status: Annotated[MemoryStatus | None, Query()] = None,
    kind: Annotated[MemoryKind | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_ITEM_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> GraphMemoryItemPage:
    return service.list_items(
        int(user["id"]),
        status=status,
        kind=kind,
        limit=limit,
        offset=offset,
    )


@router.patch("/graph-memory/items/{memory_id}", response_model=MutationResponse)
def edit_graph_memory_item(
    memory_id: UUID,
    body: GraphMemoryItemEditRequest,
    user: CurrentUser,
    service: MemoryService,
) -> MutationResponse:
    try:
        mutation = service.edit_item(
            int(user["id"]),
            memory_id,
            subject=body.subject,
            predicate=body.predicate,
            object_value=body.object_value,
            expected_revision=body.expected_revision,
        )
        return MutationResponse(item=mutation.item)
    except Exception as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable") from exc


@router.post("/graph-memory/items/{memory_id}/approve", response_model=MutationResponse)
def approve_graph_memory_item(
    memory_id: UUID,
    user: CurrentUser,
    service: MemoryService,
) -> MutationResponse:
    try:
        return MutationResponse(item=service.approve_item(int(user["id"]), memory_id).item)
    except Exception as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable") from exc


@router.get("/graph-memory/topics")
def list_graph_memory_topics(
    user: CurrentUser,
    service: MemoryService,
) -> dict[str, Any]:
    """Phase 5 user control: view live question topics (flag-gated)."""
    return {"topics": service.list_question_topics(int(user["id"]))}


@router.delete("/graph-memory/topics/{name}")
def delete_graph_memory_topic(
    name: str,
    user: CurrentUser,
    service: MemoryService,
) -> dict[str, bool]:
    """Phase 5 user control: delete one question topic."""
    if not service.delete_question_topic(int(user["id"]), name):
        raise HTTPException(status_code=404, detail="Topic not found or disabled")
    return {"deleted": True}


@router.post("/graph-memory/items/{memory_id}/renew", response_model=MutationResponse)
def renew_graph_memory_item(
    memory_id: UUID,
    user: CurrentUser,
    service: MemoryService,
) -> MutationResponse:
    try:
        return MutationResponse(item=service.renew_item(int(user["id"]), memory_id).item)
    except Exception as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable") from exc


@router.delete("/graph-memory/items/{memory_id}", response_model=DeleteItemResponse)
def delete_graph_memory_item(
    memory_id: UUID,
    user: CurrentUser,
    service: MemoryService,
) -> DeleteItemResponse:
    try:
        mutation = service.delete_item(int(user["id"]), memory_id)
        return DeleteItemResponse(
            deleted_id=mutation.item.id,
            purge_state=mutation.tenant.purge_state,
        )
    except Exception as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable") from exc


@router.delete("/personal-memory", response_model=ClearMemoryResponse)
def clear_all_personal_memory(user: CurrentUser, service: MemoryService) -> ClearMemoryResponse:
    """Clear graph memory and manual Saved Preferences in one transaction."""

    try:
        return service.clear_personal_memory(int(user["id"]))
    except Exception as exc:
        _raise_public_error(exc)
        raise AssertionError("unreachable") from exc
