-- Admin-configurable ceiling for auto-detected model context lengths.
--
-- Detection may report very large native windows (e.g. 262144 via rope
-- scaling) that exceed what the hardware serves (OOM-killed llama-server
-- observed at 262144). The ceiling caps detection; default 16384 matches
-- the historical hardcoded num_ctx. Admin picks from a fixed allowlist
-- (16k/32k/64k/128k/256k) in Settings; arbitrary values are rejected at
-- the API layer so only this CHECK plus the allowlist guard the range.

ALTER TABLE IF EXISTS system_ai_configuration
    ADD COLUMN IF NOT EXISTS model_max_num_ctx INTEGER NOT NULL DEFAULT 16384;

ALTER TABLE IF EXISTS system_ai_configuration
    DROP CONSTRAINT IF EXISTS system_ai_configuration_model_max_num_ctx_check;

ALTER TABLE IF EXISTS system_ai_configuration
    ADD CONSTRAINT system_ai_configuration_model_max_num_ctx_check
    CHECK (model_max_num_ctx IN (16384, 32768, 65536, 131072, 262144));
