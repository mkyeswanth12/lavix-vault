"""Domain and wire models for conversation relationship memory."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

DEFAULT_RETENTION_DAYS = 90
PENDING_RETENTION_DAYS = 14
MAX_ITEM_PAGE_SIZE = 100
MAX_RECALL_RESULTS = 8
MAX_RECALL_CHARS = 1_600


class MemoryKind(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"


class MemoryStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    REJECTED = "rejected"


class ProjectionState(StrEnum):
    PENDING = "pending"
    PROJECTED = "projected"
    DELETE_PENDING = "delete_pending"
    FAILED = "failed"
    STALE = "stale"


class GraphMemorySettingsRequest(BaseModel):
    enabled: bool
    retention_days: Literal[30, 90, 365] = DEFAULT_RETENTION_DAYS
    expected_revision: int | None = Field(default=None, ge=0)


class GraphMemoryItemEditRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=100)
    object_value: str = Field(min_length=1, max_length=500)
    expected_revision: int = Field(ge=1)

    @field_validator("subject", "predicate", "object_value", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        return value


class MemoryCandidate(BaseModel):
    """Strict boundary for model-produced extraction candidates.

    The worker must attach provenance from the saved user message instead of
    allowing the model to choose a tenant or source message.
    """

    kind: MemoryKind
    subject: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=100)
    object_value: str = Field(min_length=1, max_length=500)
    confidence: Annotated[float, Field(ge=0, le=1)]
    source_excerpt: str = Field(min_length=1, max_length=500)
    source_role: Literal["user"] = "user"
    explicit_user_assertion: bool
    contains_sensitive_data: bool = False

    @field_validator("subject", "predicate", "object_value", "source_excerpt", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        return value


class GraphMemoryItem(BaseModel):
    id: UUID
    kind: MemoryKind
    subject: str
    predicate: str
    object_value: str
    confidence: float
    status: MemoryStatus
    source_chat_id: UUID | None = None
    source_message_id: UUID | None = None
    source_excerpt: str
    source_created_at: datetime | None = None
    last_confirmed_at: datetime
    expires_at: datetime
    revision: int
    projection_state: ProjectionState
    created_at: datetime
    updated_at: datetime


class AboutMeProfile(BaseModel):
    status: Literal["empty", "ready"]
    summary: str


class GraphMemoryStatusResponse(BaseModel):
    enabled: bool
    retention_days: Literal[30, 90, 365]
    revision: int
    generation: int
    purge_state: Literal["ready", "pending", "failed"]
    last_learned_at: datetime | None
    next_expiry_at: datetime | None
    counts: dict[str, int]
    about_me: AboutMeProfile


class GraphMemoryItemPage(BaseModel):
    items: list[GraphMemoryItem]
    total: int
    limit: int
    offset: int


class ClearMemoryResponse(BaseModel):
    graph_deleted_count: int
    saved_preferences_deleted_count: int = 0
    generation: int
    purge_state: Literal["ready", "pending", "failed"]


class MutationResponse(BaseModel):
    item: GraphMemoryItem


class DeleteItemResponse(BaseModel):
    deleted_id: UUID
    purge_state: Literal["ready", "pending", "failed"]


class RecallRecord(BaseModel):
    id: UUID
    kind: MemoryKind
    subject: str
    predicate: str
    object_value: str
    confidence: float
    expires_at: datetime
