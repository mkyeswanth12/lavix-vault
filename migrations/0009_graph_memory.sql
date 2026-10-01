-- Opt-in conversation relationship memory.
--
-- PostgreSQL is authoritative for consent, lifecycle, provenance, expiry and
-- deletion. Neo4j is a rebuildable projection; no Neo4j internal identifier is
-- stored or exposed by the application.

CREATE TABLE IF NOT EXISTS graph_memory_tenants (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    tenant_uuid UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    retention_days SMALLINT NOT NULL DEFAULT 90,
    generation BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL DEFAULT 0,
    graph_revision BIGINT NOT NULL DEFAULT 0,
    purge_state TEXT NOT NULL DEFAULT 'ready',
    purge_requested_at TIMESTAMPTZ,
    last_learned_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT graph_memory_tenants_retention_check
        CHECK (retention_days IN (30, 90, 365)),
    CONSTRAINT graph_memory_tenants_generation_check CHECK (generation >= 0),
    CONSTRAINT graph_memory_tenants_settings_revision_check CHECK (revision >= 0),
    CONSTRAINT graph_memory_tenants_revision_check CHECK (graph_revision >= 0),
    CONSTRAINT graph_memory_tenants_purge_state_check
        CHECK (purge_state IN ('ready', 'pending', 'failed')),
    CONSTRAINT graph_memory_tenants_identity_unique UNIQUE (user_id, tenant_uuid)
);

CREATE TABLE IF NOT EXISTS graph_memory_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL,
    tenant_uuid UUID NOT NULL,
    generation BIGINT NOT NULL,
    kind TEXT NOT NULL,
    subject VARCHAR(200) NOT NULL,
    predicate VARCHAR(100) NOT NULL,
    object_value VARCHAR(500) NOT NULL,
    normalized_fingerprint CHAR(64) NOT NULL,
    confidence NUMERIC(4, 3) NOT NULL,
    status TEXT NOT NULL,
    source_chat_id UUID REFERENCES chats(id) ON DELETE SET NULL,
    source_message_id UUID REFERENCES chat_messages(id) ON DELETE SET NULL,
    source_excerpt VARCHAR(500) NOT NULL,
    last_confirmed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    approved_at TIMESTAMPTZ,
    revision INTEGER NOT NULL DEFAULT 1,
    projection_state TEXT NOT NULL DEFAULT 'pending',
    projection_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT graph_memory_items_tenant_fk
        FOREIGN KEY (user_id, tenant_uuid)
        REFERENCES graph_memory_tenants(user_id, tenant_uuid)
        ON DELETE CASCADE,
    CONSTRAINT graph_memory_items_generation_check CHECK (generation >= 0),
    CONSTRAINT graph_memory_items_kind_check
        CHECK (kind IN ('fact', 'preference', 'entity', 'relationship')),
    CONSTRAINT graph_memory_items_status_check
        CHECK (status IN ('pending', 'active', 'rejected')),
    CONSTRAINT graph_memory_items_confidence_check
        CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT graph_memory_items_revision_check CHECK (revision >= 1),
    CONSTRAINT graph_memory_items_projection_state_check
        CHECK (projection_state IN ('pending', 'projected', 'delete_pending', 'failed')),
    CONSTRAINT graph_memory_items_subject_length_check
        CHECK (CHAR_LENGTH(BTRIM(subject)) BETWEEN 1 AND 200),
    CONSTRAINT graph_memory_items_predicate_length_check
        CHECK (CHAR_LENGTH(BTRIM(predicate)) BETWEEN 1 AND 100),
    CONSTRAINT graph_memory_items_object_length_check
        CHECK (CHAR_LENGTH(BTRIM(object_value)) BETWEEN 1 AND 500),
    CONSTRAINT graph_memory_items_excerpt_length_check
        CHECK (CHAR_LENGTH(BTRIM(source_excerpt)) BETWEEN 1 AND 500),
    CONSTRAINT graph_memory_items_fingerprint_format_check
        CHECK (normalized_fingerprint ~ '^[0-9a-f]{64}$')
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_memory_items_live_fingerprint
    ON graph_memory_items (user_id, normalized_fingerprint)
    WHERE status IN ('pending', 'active');

CREATE INDEX IF NOT EXISTS idx_graph_memory_items_user_status_expiry
    ON graph_memory_items (user_id, status, expires_at, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_graph_memory_items_projection
    ON graph_memory_items (projection_state, updated_at)
    WHERE projection_state IN ('pending', 'delete_pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_graph_memory_items_source_message
    ON graph_memory_items (user_id, source_message_id)
    WHERE source_message_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS graph_memory_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL,
    tenant_uuid UUID NOT NULL,
    generation BIGINT NOT NULL,
    job_type TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    source_chat_id UUID REFERENCES chats(id) ON DELETE SET NULL,
    source_message_id UUID REFERENCES chat_messages(id) ON DELETE SET NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_owner TEXT,
    lease_expires_at TIMESTAMPTZ,
    error_code TEXT,
    error_detail TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT graph_memory_jobs_tenant_fk
        FOREIGN KEY (user_id, tenant_uuid)
        REFERENCES graph_memory_tenants(user_id, tenant_uuid)
        ON DELETE CASCADE,
    CONSTRAINT graph_memory_jobs_generation_check CHECK (generation >= 0),
    CONSTRAINT graph_memory_jobs_type_check
        CHECK (job_type IN ('extract', 'project', 'expire', 'summarize', 'purge')),
    CONSTRAINT graph_memory_jobs_state_check
        CHECK (state IN ('queued', 'running', 'complete', 'failed', 'cancelled')),
    CONSTRAINT graph_memory_jobs_attempts_check
        CHECK (attempts >= 0 AND max_attempts > 0 AND attempts <= max_attempts),
    CONSTRAINT graph_memory_jobs_payload_object_check
        CHECK (jsonb_typeof(payload) = 'object'),
    CONSTRAINT graph_memory_jobs_lease_check
        CHECK (
            (lease_owner IS NULL AND lease_expires_at IS NULL)
            OR (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
        )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_memory_extract_job_source
    ON graph_memory_jobs (user_id, job_type, source_message_id)
    WHERE job_type = 'extract' AND source_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_graph_memory_jobs_claim
    ON graph_memory_jobs (available_at, created_at)
    WHERE state = 'queued';

CREATE INDEX IF NOT EXISTS idx_graph_memory_jobs_user_state
    ON graph_memory_jobs (user_id, state, updated_at DESC);

CREATE TABLE IF NOT EXISTS graph_memory_profiles (
    user_id INTEGER PRIMARY KEY,
    tenant_uuid UUID NOT NULL,
    generation BIGINT NOT NULL,
    graph_revision BIGINT NOT NULL,
    status TEXT NOT NULL DEFAULT 'stale',
    summary TEXT NOT NULL DEFAULT '',
    profile_groups JSONB NOT NULL DEFAULT '{}'::jsonb,
    backing_memory_ids UUID[] NOT NULL DEFAULT '{}'::uuid[],
    error_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT graph_memory_profiles_tenant_fk
        FOREIGN KEY (user_id, tenant_uuid)
        REFERENCES graph_memory_tenants(user_id, tenant_uuid)
        ON DELETE CASCADE,
    CONSTRAINT graph_memory_profiles_generation_check CHECK (generation >= 0),
    CONSTRAINT graph_memory_profiles_revision_check CHECK (graph_revision >= 0),
    CONSTRAINT graph_memory_profiles_status_check
        CHECK (status IN ('ready', 'stale', 'failed')),
    CONSTRAINT graph_memory_profiles_groups_object_check
        CHECK (jsonb_typeof(profile_groups) = 'object')
);
