-- Phase 2 Neo4j projection: the 'stale' projection state.
--
-- 'stale' = row content diverged from its Neo4j node (or the node is
-- missing) and a healing upsert job is queued. The worker re-projects
-- stale rows through the same idempotent upsert path as 'pending'.

ALTER TABLE IF EXISTS graph_memory_items
    DROP CONSTRAINT IF EXISTS graph_memory_items_projection_state_check;

ALTER TABLE IF EXISTS graph_memory_items
    ADD CONSTRAINT graph_memory_items_projection_state_check
        CHECK (projection_state IN ('pending', 'projected', 'delete_pending', 'failed', 'stale'));

DROP INDEX IF EXISTS idx_graph_memory_items_projection;

CREATE INDEX IF NOT EXISTS idx_graph_memory_items_projection
    ON graph_memory_items (projection_state, updated_at)
    WHERE projection_state IN ('pending', 'delete_pending', 'failed', 'stale');
