-- Change perm_delete default from TRUE to FALSE for new users.
-- Hard-delete is destructive and irreversible; new accounts should
-- require explicit admin opt-in rather than having it enabled by default.
-- Existing non-admin users who still have the legacy TRUE default are
-- also set to FALSE so the policy is consistent across the fleet.

ALTER TABLE users
    ALTER COLUMN perm_delete SET DEFAULT FALSE;

UPDATE users
    SET perm_delete = FALSE
    WHERE perm_delete = TRUE
      AND is_admin = FALSE;
