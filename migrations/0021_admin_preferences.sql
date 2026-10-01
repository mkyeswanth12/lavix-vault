-- Admin Preferences overrides. Both columns are NULL by default, which means
-- the deployment configuration (config.yml / environment) applies. A stored
-- value takes precedence at request time. Values are validated against the
-- allowlists below by the admin API before they are written.
--
-- session_timeout_minutes drives BOTH session_idle_timeout_minutes and
-- access_token_expire_minutes. registration_enabled gates /api/auth/register
-- immediately with no restart.
ALTER TABLE system_ai_configuration
    ADD COLUMN IF NOT EXISTS session_timeout_minutes INTEGER;

ALTER TABLE system_ai_configuration
    ADD COLUMN IF NOT EXISTS registration_enabled BOOLEAN;

ALTER TABLE IF EXISTS system_ai_configuration
    DROP CONSTRAINT IF EXISTS system_ai_configuration_session_timeout_check;

ALTER TABLE IF EXISTS system_ai_configuration
    ADD CONSTRAINT system_ai_configuration_session_timeout_check
    CHECK (
        session_timeout_minutes IS NULL
        OR session_timeout_minutes IN (15, 30, 60, 120, 240, 480, 1440)
    );
