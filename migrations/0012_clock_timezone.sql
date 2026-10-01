-- Optional chat-clock override. NULL means the deployment default from
-- LAVIX_TIMEZONE / app.timezone applies. A stored IANA name takes
-- precedence per chat request. Values are validated as IANA timezone
-- names by the admin API before they are written.
ALTER TABLE system_ai_configuration
    ADD COLUMN IF NOT EXISTS clock_timezone VARCHAR(64);
