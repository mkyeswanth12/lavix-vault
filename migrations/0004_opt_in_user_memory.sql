-- Explicit, tenant-owned chat preferences. Memory is disabled by default and
-- populated only through authenticated user actions; no implicit extraction.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS memory_enabled BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS user_memories (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_text VARCHAR(500) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT user_memories_text_length_check
        CHECK (CHAR_LENGTH(memory_text) BETWEEN 1 AND 500),
    CONSTRAINT user_memories_user_text_unique UNIQUE (user_id, memory_text)
);

CREATE INDEX IF NOT EXISTS idx_user_memories_user_recent
    ON user_memories (user_id, updated_at DESC, id DESC);
