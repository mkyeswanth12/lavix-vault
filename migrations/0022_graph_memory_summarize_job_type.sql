-- Re-allow the 'summarize' background job type.
--
-- History: 0009 permitted
-- ('extract','project','expire','summarize','purge'). 0010 then removed
-- 'summarize' (deleted summarize jobs, re-added the CHECK without it)
-- because no model-generated cache existed at the time. 0018 reintroduced
-- the summarize pipeline (memory_summaries table, _enqueue_summarize,
-- worker _summarize handler, summarize single-flight index) but its
-- comment wrongly claimed 'summarize' was already permitted, so no DDL
-- touched graph_memory_jobs. Every successful memory save enqueues a
-- summarize job in the same transaction, so the CHECK violation rolled
-- back the primary item write and nothing was ever stored.
--
-- This migration restores 'summarize' to the allowed set. 'expire' is
-- kept: the worker dispatches on it even though nothing enqueues it
-- today (expiry runs inline via expire_due).

ALTER TABLE IF EXISTS graph_memory_jobs
    DROP CONSTRAINT IF EXISTS graph_memory_jobs_type_check;

ALTER TABLE IF EXISTS graph_memory_jobs
    ADD CONSTRAINT graph_memory_jobs_type_check
        CHECK (job_type IN ('extract', 'project', 'expire', 'summarize', 'purge'));
