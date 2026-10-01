-- Keep buyer-issued purchase orders distinct from agreements and invoices.

ALTER TABLE files
    DROP CONSTRAINT IF EXISTS files_doc_type_check;

ALTER TABLE files
    ADD CONSTRAINT files_doc_type_check
    CHECK (doc_type IS NULL OR doc_type IN (
        'agreement', 'book', 'brochure', 'certificate', 'contract', 'form',
        'identity_document', 'image', 'invoice', 'letter', 'manual',
        'meeting_notes', 'policy', 'presentation', 'proposal', 'purchase_order',
        'receipt', 'report', 'research_paper', 'resume', 'spreadsheet',
        'statement', 'other'
    ));
