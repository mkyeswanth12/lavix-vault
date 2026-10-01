-- Account-wide local chat-model preference.  NULL follows the configured
-- system default so deployment model changes do not require rewriting users.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS preferred_chat_model VARCHAR(200);
