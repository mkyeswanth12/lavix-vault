-- Grounded About Me synthesis cache.
--
-- One row per user holding the last VALID LLM-synthesized (or deterministic
-- fallback) About Me summary. The row is revision-pinned: graph_revision
-- records the tenant graph_revision the summary was built from. Any memory
-- mutation bumps graph_revision, which makes the cached row stale and
-- triggers a background 'summarize' job. Readers always serve the stored
-- row (stale-while-revalidate) — a stale or running synthesis never blanks
-- the About Me card, and a failed synthesis never overwrites the last
-- valid summary.

CREATE TABLE IF NOT EXISTS memory_summaries (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    source_memory_ids UUID[] NOT NULL DEFAULT '{}',
    graph_revision BIGINT NOT NULL DEFAULT 0,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model TEXT NOT NULL DEFAULT '',
    is_fallback BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT memory_summaries_revision_check CHECK (graph_revision >= 0)
);

-- Single-flight guard for background synthesis: at most one queued or
-- running 'summarize' job per user. Finished jobs no longer match the
-- predicate, so regeneration after completion is always allowed.
-- 'summarize' is already a permitted job_type (see 0009); no DDL needed
-- on graph_memory_jobs itself.

CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_memory_summarize_single_flight
    ON graph_memory_jobs (user_id)
    WHERE job_type = 'summarize' AND state IN ('queued', 'running');
