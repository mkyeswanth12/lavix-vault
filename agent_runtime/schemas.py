"""Wire models for the private runtime API."""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    web_search_enabled: bool = True
    deep_search: bool = False
    requested_file_ids: list[int] | None = Field(default=None, max_length=50)
    chat_mode: Literal["casual", "expert"] | None = Field(default=None)
    # Per-request allowlist from the API's DB-backed model config. When
    # present it governs model validation (admin changes apply without
    # agent restarts); absent preserves the static env allowlist.
    allowed_models: list[str] | None = Field(default=None, max_length=32)
    # Per-request context ceiling from the admin DB config (same pattern:
    # present governs, absent falls back to the agent env default).
    model_max_num_ctx: int | None = Field(default=None, ge=1024, le=1048576)
    # Per-request retrieval candidate count from the admin DB config (same
    # pattern: present governs, absent falls back to the agent default).
    retrieval_top_k: int | None = Field(default=None, ge=1, le=500)
    # Per-request web search depth tier from the admin DB config (same
    # pattern: present governs, absent falls back to the conservative
    # tier). Resolved to its retrieval budget by the runtime.
    search_depth: Literal["conservative", "balanced", "deep", "pro"] | None = Field(default=None)
    # Whether the user has opted into relationship memory. Gates the
    # personal-declaration acknowledgment: only an opted-in user hears a
    # persistence promise (the extraction worker can only learn for them).
    # Omitted when False so older agent builds, which forbid unknown
    # options fields, keep working (their default is False too).
    memory_opted_in: bool = False

    @field_validator("requested_file_ids")
    @classmethod
    def validate_file_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(file_id <= 0 for file_id in value):
            raise ValueError("requested_file_ids must contain positive integers")
        return list(dict.fromkeys(value))


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID = Field(default_factory=uuid4)
    user_query: str = Field(min_length=1, max_length=20_000)
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    model: str | None = Field(default=None, max_length=200)
    options: RunOptions = Field(default_factory=RunOptions)
    # Rewriter-only wider history window (up to 6 msgs / 3 pairs). The
    # synthesis window (`messages`, max 3) is untouched by this field.
    rewrite_context: list[ChatMessage] | None = Field(default=None, max_length=12)
    # Per-session entity carry-forward, maintained API-side in
    # chat_messages.content_json ("resolved_entities"). Merged with each
    # rewrite's output and returned on the final event for write-back.
    resolved_entities: dict[str, str] | None = Field(default=None, max_length=20)
    # Commit-2 signal: True when this message answers a pending
    # clarification turn (session flag `clarification_pending`, API-side).
    # Accepted now so the wire contract is stable; unused until commit 2.
    answers_clarification: bool = False
    # Scoped-attachment text for query grounding (filename + vision
    # summary + tags, API-assembled, capped). Lets the rewrite validator
    # tell attachment-anchored terms apart from invented topics.
    attachment_text: str = Field(default="", max_length=2000)
    attachment_count: int = Field(default=0, ge=0, le=50)

    @field_validator("user_query")
    @classmethod
    def user_query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("user_query must not be blank")
        return value

    @field_validator("resolved_entities")
    @classmethod
    def resolved_entities_must_be_bounded(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return None
        cleaned = {str(k).strip()[:64]: str(v).strip()[:200] for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if k and v}

    @model_validator(mode="after")
    def validate_conversation(self) -> RunRequest:
        if self.messages[-1].role != "user":
            raise ValueError("the final message must have role 'user'")
        if sum(len(message.content) for message in self.messages) > 100_000:
            raise ValueError("conversation history is too large")
        return self


class InvokeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    model: str
    answer: str
