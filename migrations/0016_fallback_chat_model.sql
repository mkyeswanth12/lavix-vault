-- Explicit fallback chat model (admin-chosen, nullable).
--
-- Previously the fallback was implicit (system default, then failure).
-- NULL means "no fallback": an unusable active model errors honestly
-- instead of substituting. A set value must name an allowed model;
-- enforced at the API layer (this CHECK only guards type/shape).
-- Stale values (model later removed from allowed) self-heal to NULL-like
-- behavior at resolution time, same as personal overrides.

ALTER TABLE IF EXISTS system_ai_configuration
    ADD COLUMN IF NOT EXISTS fallback_chat_model TEXT;
