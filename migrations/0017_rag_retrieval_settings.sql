-- Admin-configurable RAG retrieval scope settings.
--
-- file_scope caps the maximum number of DISTINCT FILES that may contribute
-- to the final RAG context in discovery/unscoped retrieval. Explicitly
-- tagged files are always fully represented (up to the per-request scope
-- ceiling) and are never truncated by this setting. Default 5 preserves
-- the historical effective behavior; range 1-100.
--
-- top_k controls how many candidate chunks the vector retrieval stage
-- returns per query before reranking. Default 20; range 1-500. Downstream
-- stages (rerank input, evidence selection, synthesis context, timeouts)
-- keep their own independent safety limits. One file may contribute
-- multiple chunks, so top_k is not a file count.

ALTER TABLE IF EXISTS system_ai_configuration
    ADD COLUMN IF NOT EXISTS file_scope INTEGER NOT NULL DEFAULT 5;

ALTER TABLE IF EXISTS system_ai_configuration
    DROP CONSTRAINT IF EXISTS system_ai_configuration_file_scope_check;

ALTER TABLE IF EXISTS system_ai_configuration
    ADD CONSTRAINT system_ai_configuration_file_scope_check
    CHECK (file_scope BETWEEN 1 AND 100);

ALTER TABLE IF EXISTS system_ai_configuration
    ADD COLUMN IF NOT EXISTS top_k INTEGER NOT NULL DEFAULT 20;

ALTER TABLE IF EXISTS system_ai_configuration
    DROP CONSTRAINT IF EXISTS system_ai_configuration_top_k_check;

ALTER TABLE IF EXISTS system_ai_configuration
    ADD CONSTRAINT system_ai_configuration_top_k_check
    CHECK (top_k BETWEEN 1 AND 500);
