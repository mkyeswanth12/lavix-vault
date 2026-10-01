-- Make Safe Automatic consent authoritative and remove the unused generated
-- profile cache. Migration 0009 is immutable and may already be installed.

-- Before this release, PUT /api/ai/memory could update only the legacy flag,
-- but there is no audit record proving which side of an existing mismatch was
-- the user's last choice. Fail closed without deleting memory: disable both
-- values and advance the graph settings revision. Subsequent writes update both
-- values in one PostgreSQL transaction.
UPDATE graph_memory_tenants AS tenant
SET enabled = FALSE,
    revision = tenant.revision + 1,
    updated_at = NOW()
FROM users
WHERE users.id = tenant.user_id
  AND users.memory_enabled IS DISTINCT FROM tenant.enabled;

UPDATE users
SET memory_enabled = FALSE
FROM graph_memory_tenants AS tenant
WHERE users.id = tenant.user_id
  AND users.memory_enabled IS DISTINCT FROM tenant.enabled;

-- When no graph tenant exists, the legacy flag is the only persisted consent
-- record. Seed it once while preserving the 90-day default. New users created
-- after this migration keep both schema defaults disabled.
INSERT INTO graph_memory_tenants (user_id, enabled, retention_days, revision)
SELECT
    users.id,
    users.memory_enabled,
    90,
    CASE WHEN users.memory_enabled THEN 1 ELSE 0 END
FROM users
ON CONFLICT (user_id) DO NOTHING;

COMMENT ON COLUMN users.memory_enabled IS
    'Compatibility mirror of graph_memory_tenants.enabled; update both atomically';

-- About Me is now derived synchronously and deterministically from the bounded
-- active-memory query. No model-generated cache or no-op summarize job remains.
DELETE FROM graph_memory_jobs WHERE job_type = 'summarize';

ALTER TABLE graph_memory_jobs
    DROP CONSTRAINT IF EXISTS graph_memory_jobs_type_check;

ALTER TABLE graph_memory_jobs
    ADD CONSTRAINT graph_memory_jobs_type_check
    CHECK (job_type IN ('extract', 'project', 'expire', 'purge'));

DROP TABLE IF EXISTS graph_memory_profiles;
