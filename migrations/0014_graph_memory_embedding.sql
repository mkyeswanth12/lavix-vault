-- Phase 4 semantic recall: optional vector similarity over memory rows.
--
-- embedding is NULL until the worker embeds the row (new upserts embed on
-- the project path; drift-healed rows backfill naturally). NULL rows stay
-- keyword-only. pgvector extension already exists (foundation migration).

ALTER TABLE IF EXISTS graph_memory_items
    ADD COLUMN IF NOT EXISTS embedding vector(1024);

CREATE INDEX IF NOT EXISTS idx_graph_memory_items_embedding
    ON graph_memory_items USING hnsw (embedding vector_cosine_ops);
