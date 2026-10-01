-- Admin-configurable web search depth.
--
-- search_depth controls the thoroughness and context budget of web search RAG:
--   - conservative (default): 3 search legs, 6 pool cap, 3 results/leg, 2000 excerpt chars
--   - balanced: 4 search legs, 8 pool cap, 5 results/leg, 3000 excerpt chars
--   - deep: 5 search legs, 10 pool cap, 6 results/leg, 4000 excerpt chars
--
-- Governs both planner leg execution and gateway compaction.

ALTER TABLE IF EXISTS system_ai_configuration
    ADD COLUMN IF NOT EXISTS search_depth TEXT NOT NULL DEFAULT 'conservative';

ALTER TABLE IF EXISTS system_ai_configuration
    DROP CONSTRAINT IF EXISTS system_ai_configuration_search_depth_check;

ALTER TABLE IF EXISTS system_ai_configuration
    ADD CONSTRAINT system_ai_configuration_search_depth_check
    CHECK (search_depth IN ('conservative', 'balanced', 'deep'));
