-- Add the 'pro' web search depth tier.
--
-- pro is the top tier: 7 search legs, 100-item evidence pool cap (effectively
-- uncapped), 12 results per leg, 6000-char excerpts. Existing rows are
-- unaffected; the constraint only validates new writes.

ALTER TABLE IF EXISTS system_ai_configuration
    DROP CONSTRAINT IF EXISTS system_ai_configuration_search_depth_check;

ALTER TABLE IF EXISTS system_ai_configuration
    ADD CONSTRAINT system_ai_configuration_search_depth_check
    CHECK (search_depth IN ('conservative', 'balanced', 'deep', 'pro'));
