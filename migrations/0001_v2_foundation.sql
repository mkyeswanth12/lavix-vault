-- Lavix Vault v2 baseline and additive upgrade foundation.
--
-- Every object required by the API is created here so a blank pgvector
-- database and an existing Lavix v1 database follow the same migration path.
-- Existing v1-only physical objects are left untouched during the additive
-- upgrade, but v2 code does not read them. No sample or administrator is seeded.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_login TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_login_ip VARCHAR(50),
    storage_quota_bytes BIGINT NOT NULL DEFAULT 10737418240,
    storage_used_bytes BIGINT NOT NULL DEFAULT 0,
    persona_prompt TEXT,
    google_user_id TEXT UNIQUE,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    perm_upload BOOLEAN NOT NULL DEFAULT TRUE,
    perm_download BOOLEAN NOT NULL DEFAULT TRUE,
    perm_delete BOOLEAN NOT NULL DEFAULT TRUE,
    perm_ai BOOLEAN NOT NULL DEFAULT TRUE,
    perm_share BOOLEAN NOT NULL DEFAULT TRUE,
    perm_folders BOOLEAN NOT NULL DEFAULT TRUE,
    perm_rename BOOLEAN NOT NULL DEFAULT TRUE,
    avatar_data TEXT,
    openrouter_api_key TEXT,
    openrouter_provider VARCHAR(20) NOT NULL DEFAULT 'auto'
);

ALTER TABLE users ADD COLUMN IF NOT EXISTS persona_prompt TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS google_user_id TEXT UNIQUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS perm_upload BOOLEAN DEFAULT TRUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS perm_download BOOLEAN DEFAULT TRUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS perm_delete BOOLEAN DEFAULT TRUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS perm_ai BOOLEAN DEFAULT TRUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS perm_share BOOLEAN DEFAULT TRUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS perm_folders BOOLEAN DEFAULT TRUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS perm_rename BOOLEAN DEFAULT TRUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_data TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS openrouter_api_key TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS openrouter_provider VARCHAR(20) DEFAULT 'auto';

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_revoked BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens (user_id);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_session_id ON refresh_tokens (session_id);

CREATE TABLE IF NOT EXISTS folders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    parent_id INTEGER REFERENCES folders(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_at TIMESTAMP
);

ALTER TABLE folders ADD COLUMN IF NOT EXISTS parent_id INTEGER REFERENCES folders(id) ON DELETE CASCADE;
ALTER TABLE folders ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE;
ALTER TABLE folders ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP;
ALTER TABLE folders DROP CONSTRAINT IF EXISTS folders_user_id_name_key;
DROP INDEX IF EXISTS idx_folders_unique_per_parent;
CREATE UNIQUE INDEX idx_folders_unique_per_parent
    ON folders (user_id, COALESCE(parent_id, -1), name)
    WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_folders_user_id ON folders (user_id);
CREATE INDEX IF NOT EXISTS idx_folders_parent_id ON folders (parent_id);
CREATE INDEX IF NOT EXISTS idx_folders_is_deleted ON folders (is_deleted)
    WHERE is_deleted = FALSE;

CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    uuid UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    display_name TEXT NOT NULL,
    mime_type VARCHAR(255),
    file_size_bytes BIGINT NOT NULL,
    encrypted_filename TEXT NOT NULL,
    rsa_key_filename TEXT NOT NULL,
    sha256_hash CHAR(64) NOT NULL,
    s3_bucket_name VARCHAR(255) NOT NULL,
    s3_path TEXT NOT NULL,
    s3_key_path TEXT NOT NULL,
    quick_tags TEXT[],
    quick_summary TEXT,
    user_granted_ai_access BOOLEAN NOT NULL DEFAULT FALSE,
    uploaded_at TIMESTAMP NOT NULL DEFAULT NOW(),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_at TIMESTAMP,
    folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL
);

ALTER TABLE files ADD COLUMN IF NOT EXISTS folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_files_user_id ON files (user_id);
CREATE INDEX IF NOT EXISTS idx_files_uploaded_at ON files (uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_files_is_deleted ON files (is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_files_user_deleted_date
    ON files (user_id, is_deleted, uploaded_at DESC);

CREATE TABLE IF NOT EXISTS chats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New Chat',
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pinned BOOLEAN NOT NULL DEFAULT FALSE,
    archived BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats (user_id);
CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats (updated_at DESC);

CREATE TABLE IF NOT EXISTS decryption_sessions (
    id SERIAL PRIMARY KEY,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    purpose VARCHAR(50) NOT NULL,
    client_ip VARCHAR(45),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '15 minutes'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE decryption_sessions ADD COLUMN IF NOT EXISTS client_ip VARCHAR(45);
CREATE INDEX IF NOT EXISTS idx_decryption_sessions_expiry ON decryption_sessions (expires_at);

CREATE TABLE IF NOT EXISTS user_policy_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    perm_upload BOOLEAN NOT NULL DEFAULT TRUE,
    perm_download BOOLEAN NOT NULL DEFAULT TRUE,
    perm_delete BOOLEAN NOT NULL DEFAULT TRUE,
    perm_ai BOOLEAN NOT NULL DEFAULT TRUE,
    perm_share BOOLEAN NOT NULL DEFAULT TRUE,
    perm_folders BOOLEAN NOT NULL DEFAULT TRUE,
    perm_rename BOOLEAN NOT NULL DEFAULT TRUE,
    is_builtin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

INSERT INTO user_policy_templates (
    name, perm_upload, perm_download, perm_delete, perm_ai,
    perm_share, perm_folders, perm_rename, is_builtin
)
VALUES
    ('Full Access', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE),
    ('Read Only', FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE),
    ('Basic User', TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, TRUE),
    ('Power User', TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)
ON CONFLICT (name) DO NOTHING;

ALTER TABLE files
    ADD COLUMN IF NOT EXISTS desired_revision INTEGER NOT NULL DEFAULT 0;

ALTER TABLE files
    ADD COLUMN IF NOT EXISTS current_revision INTEGER;

ALTER TABLE files
    ADD COLUMN IF NOT EXISTS ai_status TEXT;

ALTER TABLE files
    ADD COLUMN IF NOT EXISTS ai_error_code TEXT;

ALTER TABLE files
    ADD COLUMN IF NOT EXISTS ai_error_detail TEXT;

-- Legacy embedding flags described the retired vector store and cannot prove
-- that a file has a published pgvector revision. Require explicit consent and
-- re-indexing instead of exposing stale or cross-generation derived data.
UPDATE files
SET user_granted_ai_access = FALSE,
    current_revision = NULL,
    ai_status = 'not_granted',
    ai_error_code = NULL,
    ai_error_detail = NULL
WHERE ai_status IS NULL;

ALTER TABLE files
    ALTER COLUMN ai_status SET DEFAULT 'not_granted',
    ALTER COLUMN ai_status SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'files_revision_order_check'
          AND conrelid = 'files'::regclass
    ) THEN
        ALTER TABLE files
            ADD CONSTRAINT files_revision_order_check
            CHECK (
                desired_revision >= 0
                AND (
                    current_revision IS NULL
                    OR (
                        current_revision >= 1
                        AND current_revision <= desired_revision
                    )
                )
            );
    END IF;
END
$$;

-- Replace the legacy quota trigger with contribution-based accounting. This
-- covers delete/restore transitions as well as ownership and file-size edits.
CREATE OR REPLACE FUNCTION update_user_storage()
RETURNS TRIGGER AS $$
DECLARE
    storage_delta BIGINT;
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NOT COALESCE(NEW.is_deleted, FALSE) AND NEW.user_id IS NOT NULL THEN
            UPDATE users
            SET storage_used_bytes = COALESCE(storage_used_bytes, 0) + NEW.file_size_bytes
            WHERE id = NEW.user_id;
        END IF;
        RETURN NEW;
    END IF;

    IF TG_OP = 'DELETE' THEN
        IF NOT COALESCE(OLD.is_deleted, FALSE) AND OLD.user_id IS NOT NULL THEN
            UPDATE users
            SET storage_used_bytes = COALESCE(storage_used_bytes, 0) - OLD.file_size_bytes
            WHERE id = OLD.user_id;
        END IF;
        RETURN OLD;
    END IF;

    IF OLD.user_id IS NOT DISTINCT FROM NEW.user_id THEN
        storage_delta :=
            CASE WHEN NOT COALESCE(NEW.is_deleted, FALSE) THEN NEW.file_size_bytes ELSE 0 END
            - CASE WHEN NOT COALESCE(OLD.is_deleted, FALSE) THEN OLD.file_size_bytes ELSE 0 END;

        IF storage_delta <> 0 AND NEW.user_id IS NOT NULL THEN
            UPDATE users
            SET storage_used_bytes = COALESCE(storage_used_bytes, 0) + storage_delta
            WHERE id = NEW.user_id;
        END IF;
    ELSE
        IF NOT COALESCE(OLD.is_deleted, FALSE) AND OLD.user_id IS NOT NULL THEN
            UPDATE users
            SET storage_used_bytes = COALESCE(storage_used_bytes, 0) - OLD.file_size_bytes
            WHERE id = OLD.user_id;
        END IF;

        IF NOT COALESCE(NEW.is_deleted, FALSE) AND NEW.user_id IS NOT NULL THEN
            UPDATE users
            SET storage_used_bytes = COALESCE(storage_used_bytes, 0) + NEW.file_size_bytes
            WHERE id = NEW.user_id;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_user_storage ON files;

CREATE TRIGGER trigger_update_user_storage
AFTER INSERT OR DELETE OR UPDATE OF is_deleted, file_size_bytes, user_id
ON files
FOR EACH ROW
EXECUTE FUNCTION update_user_storage();

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'files_ai_status_check'
          AND conrelid = 'files'::regclass
    ) THEN
        ALTER TABLE files
            ADD CONSTRAINT files_ai_status_check
            CHECK (
                ai_status IN (
                    'not_granted', 'queued', 'decrypting', 'converting',
                    'parsing', 'chunking', 'embedding', 'publishing', 'ready',
                    'failed', 'cancelled', 'unsupported', 'password_required'
                )
            );
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    priority INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_owner TEXT,
    lease_expires_at TIMESTAMPTZ,
    cancel_requested_at TIMESTAMPTZ,
    progress_current INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER,
    parser_fingerprint TEXT,
    error_code TEXT,
    error_detail TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ingestion_jobs_state_check
        CHECK (state IN (
            'queued', 'decrypting', 'converting', 'parsing', 'chunking',
            'embedding', 'publishing', 'ready', 'failed', 'cancelled',
            'unsupported', 'password_required'
        )),
    CONSTRAINT ingestion_jobs_revision_check
        CHECK (revision >= 1),
    CONSTRAINT ingestion_jobs_attempts_check
        CHECK (attempts >= 0 AND max_attempts > 0 AND attempts <= max_attempts),
    CONSTRAINT ingestion_jobs_progress_check
        CHECK (
            progress_current >= 0
            AND (progress_total IS NULL OR progress_total >= progress_current)
        ),
    CONSTRAINT ingestion_jobs_lease_check
        CHECK (
            (lease_owner IS NULL AND lease_expires_at IS NULL)
            OR (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
        ),
    CONSTRAINT ingestion_jobs_file_revision_unique
        UNIQUE (file_id, revision),
    CONSTRAINT ingestion_jobs_composite_identity_unique
        UNIQUE (id, user_id, file_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_claim
    ON ingestion_jobs (priority DESC, available_at, created_at)
    WHERE state = 'queued' AND cancel_requested_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user_state
    ON ingestion_jobs (user_id, state, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_expired_lease
    ON ingestion_jobs (lease_expires_at)
    WHERE lease_expires_at IS NOT NULL;


CREATE TABLE IF NOT EXISTS document_revisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    job_id UUID NOT NULL,
    source_sha256 CHAR(64) NOT NULL,
    parser_fingerprint TEXT NOT NULL,
    chunker_fingerprint TEXT NOT NULL,
    embedding_fingerprint TEXT NOT NULL,
    embedding_dimension SMALLINT NOT NULL DEFAULT 1024,
    status TEXT NOT NULL DEFAULT 'building',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    source_size_bytes BIGINT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    CONSTRAINT document_revisions_job_fkey
        FOREIGN KEY (job_id, user_id, file_id, revision)
        REFERENCES ingestion_jobs (id, user_id, file_id, revision)
        ON DELETE CASCADE,
    CONSTRAINT document_revisions_revision_check
        CHECK (revision >= 1),
    CONSTRAINT document_revisions_source_sha256_check
        CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT document_revisions_dimension_check
        CHECK (embedding_dimension = 1024),
    CONSTRAINT document_revisions_status_check
        CHECK (status IN ('building', 'current', 'superseded', 'failed', 'deleted')),
    CONSTRAINT document_revisions_chunk_count_check
        CHECK (chunk_count >= 0),
    CONSTRAINT document_revisions_source_size_check
        CHECK (source_size_bytes IS NULL OR source_size_bytes >= 0),
    CONSTRAINT document_revisions_identity_unique
        UNIQUE (file_id, revision),
    CONSTRAINT document_revisions_job_unique
        UNIQUE (job_id),
    CONSTRAINT document_revisions_composite_identity_unique
        UNIQUE (id, user_id, file_id, revision)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_document_revisions_current_file
    ON document_revisions (file_id)
    WHERE status = 'current';

CREATE INDEX IF NOT EXISTS idx_document_revisions_user_file
    ON document_revisions (user_id, file_id, revision DESC);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    element_ids TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    provenance JSONB NOT NULL DEFAULT '[]'::jsonb,
    section_path TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    token_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1024),
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(content, ''))
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT document_chunks_pkey
        PRIMARY KEY (file_id, revision, chunk_id),
    CONSTRAINT document_chunks_revision_fkey
        FOREIGN KEY (file_id, revision)
        REFERENCES document_revisions (file_id, revision)
        ON DELETE CASCADE,
    CONSTRAINT document_chunks_id_check
        CHECK (chunk_id ~ '^chk_[0-9a-f]{64}$'),
    CONSTRAINT document_chunks_revision_check
        CHECK (revision >= 1),
    CONSTRAINT document_chunks_ordinal_check
        CHECK (ordinal >= 0),
    CONSTRAINT document_chunks_content_check
        CHECK (length(content) > 0 AND length(embedding_text) > 0),
    CONSTRAINT document_chunks_token_count_check
        CHECK (token_count >= 0),
    CONSTRAINT document_chunks_revision_ordinal_unique
        UNIQUE (file_id, revision, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_user_file
    ON document_chunks (user_id, file_id, revision DESC, ordinal);

CREATE INDEX IF NOT EXISTS idx_document_chunks_search
    ON document_chunks USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
    ON document_chunks USING HNSW (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;


CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sequence_number INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    content_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    provider TEXT,
    model TEXT,
    tool_name TEXT,
    tool_call_id TEXT,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chat_messages_sequence_check
        CHECK (sequence_number >= 0),
    CONSTRAINT chat_messages_role_check
        CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    CONSTRAINT chat_messages_prompt_tokens_check
        CHECK (prompt_tokens IS NULL OR prompt_tokens >= 0),
    CONSTRAINT chat_messages_completion_tokens_check
        CHECK (completion_tokens IS NULL OR completion_tokens >= 0),
    CONSTRAINT chat_messages_chat_sequence_unique
        UNIQUE (chat_id, sequence_number)
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_user_chat
    ON chat_messages (user_id, chat_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_chat_messages_created
    ON chat_messages (chat_id, created_at);
