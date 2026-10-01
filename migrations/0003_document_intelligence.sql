-- Revision-aligned document type and intelligence generation state.

ALTER TABLE files
    ADD COLUMN IF NOT EXISTS doc_type TEXT;

ALTER TABLE files
    ADD COLUMN IF NOT EXISTS intelligence_status TEXT NOT NULL DEFAULT 'pending';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'files_doc_type_check'
          AND conrelid = 'files'::regclass
    ) THEN
        ALTER TABLE files
            ADD CONSTRAINT files_doc_type_check
            CHECK (doc_type IS NULL OR doc_type IN (
                'agreement', 'brochure', 'certificate', 'contract', 'form',
                'identity_document', 'image', 'invoice', 'letter', 'manual',
                'meeting_notes', 'policy', 'presentation', 'proposal', 'receipt',
                'report', 'research_paper', 'resume', 'spreadsheet', 'statement',
                'other'
            ));
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'files_intelligence_status_check'
          AND conrelid = 'files'::regclass
    ) THEN
        ALTER TABLE files
            ADD CONSTRAINT files_intelligence_status_check
            CHECK (intelligence_status IN ('pending', 'model', 'fallback'));
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_files_user_doc_type
    ON files (user_id, doc_type)
    WHERE is_deleted = FALSE AND doc_type IS NOT NULL;
