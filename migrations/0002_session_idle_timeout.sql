-- Add a server-enforced inactivity deadline to each refresh-token session.
-- Existing live sessions receive a fresh grace period when this migration runs;
-- revoked or naturally expired history retains its original creation time.

ALTER TABLE refresh_tokens
    ADD COLUMN IF NOT EXISTS last_activity_at TIMESTAMPTZ;

UPDATE refresh_tokens
SET last_activity_at = CASE
    WHEN is_revoked = FALSE AND expires_at > NOW() THEN NOW()
    ELSE created_at
END
WHERE last_activity_at IS NULL;

ALTER TABLE refresh_tokens
    ALTER COLUMN last_activity_at SET DEFAULT NOW();

ALTER TABLE refresh_tokens
    ALTER COLUMN last_activity_at SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_refresh_tokens_active_session_activity
    ON refresh_tokens (user_id, session_id, last_activity_at DESC)
    WHERE is_revoked = FALSE;
