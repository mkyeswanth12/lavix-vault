from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StrictInt, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # The selected model is resolved server-side from the account preference
    # under the current administrator allowlist.  Keeping it as private
    # coordinator state prevents old or hand-written clients from bypassing
    # the Chat/Settings control plane with a per-request override.
    _resolved_model: str | None = PrivateAttr(default=None)
    _allowed_models: list[str] | None = PrivateAttr(default=None)
    _model_max_num_ctx: int | None = PrivateAttr(default=None)
    _retrieval_top_k: int | None = PrivateAttr(default=None)
    _search_depth: str | None = PrivateAttr(default=None)
    _fallback_used: bool = PrivateAttr(default=False)

    message: str = Field(min_length=1, max_length=20_000)
    provider: str | None = Field(default="", max_length=30)
    file_ids: list[int] | None = Field(default=None, max_length=50)
    chat_upload_ids: list[int] | None = Field(default=None, max_length=50)
    folder_ids: list[int] | None = Field(default=None, max_length=20)
    chat_id: str | None = Field(default=None, max_length=36)
    # Explicit depth signal for the active chatbox mode. Style still comes
    # from persona_prompt (untrusted); mode is trusted and only sets depth.
    mode: Literal["casual", "expert"] | None = Field(default=None)
    deep_search: bool = False
    # Web search defaults ON to match the runtime sidecar (RunOptions) and
    # the UI toggle default. Users opt out per request; sensitive material
    # should still be asked about with the toggle explicitly OFF since
    # queries leave the deployment through public engines.
    web_search_enabled: bool = True
    images: list[str] | None = Field(default=None, max_length=4)
    persona_prompt: str | None = Field(default=None, max_length=4_000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @field_validator("file_ids", "chat_upload_ids")
    @classmethod
    def valid_file_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(file_id <= 0 for file_id in value):
            raise ValueError("file IDs must be positive")
        return list(dict.fromkeys(value))

    @field_validator("folder_ids")
    @classmethod
    def valid_folder_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(folder_id <= 0 for folder_id in value):
            raise ValueError("folder IDs must be positive")
        return list(dict.fromkeys(value))


class SemanticSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1_000)
    max_results: int | None = Field(default=10, ge=1, le=20)


class ChatCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default="New Chat", max_length=200)


class ChatPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)
    messages: list[dict] | None = Field(default=None, max_length=1_000)
    pinned: bool | None = None


class ChatScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Each key is optional: absent keys are left alone, explicit [] clears.
    file_ids: list[StrictInt] | None = Field(default=None, max_length=50)
    folder_ids: list[StrictInt] | None = Field(default=None, max_length=20)

    @field_validator("file_ids")
    @classmethod
    def valid_scope_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(file_id <= 0 for file_id in value):
            raise ValueError("file IDs must be positive")
        return list(dict.fromkeys(value))

    @field_validator("folder_ids")
    @classmethod
    def valid_scope_folder_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(folder_id <= 0 for folder_id in value):
            raise ValueError("folder IDs must be positive")
        return list(dict.fromkeys(value))
