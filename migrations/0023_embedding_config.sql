-- Embedding deployment metadata + relaxed revision dimension check.
--
-- 0001/0014 created 1024-dim vector columns; those stay byte-for-byte
-- untouched so existing installs migrate with no data change. This migration
-- only (a) records which model/dimensions a database was built for, and
-- (b) widens the per-revision dimension CHECK from "= 1024" to the pgvector
-- HNSW-indexable range. Fresh installs with a non-1024 EMBEDDING_DIMENSIONS
-- get correctly-sized columns from the migrator wrapper (Python side, only
-- when the tables are still empty); dimension changes on populated
-- databases go through `python -m app.cli reembed`, never this migration.

CREATE TABLE IF NOT EXISTS embedding_config (
    singleton_id SMALLINT PRIMARY KEY DEFAULT 1,
    model TEXT NOT NULL,
    dimensions SMALLINT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    target_model TEXT,
    target_dimensions SMALLINT,
    backfilled_chunks BIGINT NOT NULL DEFAULT 0,
    verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT embedding_config_singleton_check CHECK (singleton_id = 1),
    CONSTRAINT embedding_config_dimensions_check
        CHECK (dimensions BETWEEN 1 AND 4000),
    CONSTRAINT embedding_config_status_check
        CHECK (status IN ('active', 'reembedding'))
);

ALTER TABLE document_revisions DROP CONSTRAINT IF EXISTS document_revisions_dimension_check;
ALTER TABLE document_revisions
    ADD CONSTRAINT document_revisions_dimension_check
        CHECK (embedding_dimension BETWEEN 1 AND 4000);
