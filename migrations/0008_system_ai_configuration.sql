-- Revisioned, singleton control plane for local AI roles.  The table is
-- intentionally empty after migration: NULL/absent values resolve through the
-- deployment environment, preserving the pre-migration runtime exactly until
-- an administrator performs the first save.

CREATE TABLE IF NOT EXISTS system_ai_configuration (
    singleton_id SMALLINT PRIMARY KEY,
    revision BIGINT NOT NULL,
    chat_enabled BOOLEAN,
    chat_default_model VARCHAR(200),
    chat_allowed_models TEXT[],
    vision_enabled BOOLEAN,
    vision_model VARCHAR(200),
    intelligence_enabled BOOLEAN,
    intelligence_model VARCHAR(200),
    memory_extraction_enabled BOOLEAN,
    memory_extraction_model VARCHAR(200),
    reranker_enabled BOOLEAN,
    updated_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT system_ai_configuration_singleton_check
        CHECK (singleton_id = 1),
    CONSTRAINT system_ai_configuration_revision_check
        CHECK (revision >= 1),
    CONSTRAINT system_ai_configuration_chat_models_check
        CHECK (chat_allowed_models IS NULL OR CARDINALITY(chat_allowed_models) > 0),
    CONSTRAINT system_ai_configuration_chat_default_check
        CHECK (
            chat_default_model IS NULL
            OR chat_allowed_models IS NULL
            OR chat_default_model = ANY(chat_allowed_models)
        )
);
