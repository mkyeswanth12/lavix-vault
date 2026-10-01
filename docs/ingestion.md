# Ingestion

Indexing is explicit and reversible. Uploading stores an encrypted source but leaves `user_granted_ai_access=false` and `ai_status=not_granted`. Only a grant command queues parsing. Revocation cancels active work and physically removes derived revisions and chunks while preserving the encrypted source file.

## Routing matrix

Routing uses the trusted/sniffed MIME type first and a sanitized filename extension only when MIME is absent, `application/octet-stream`, or `application/zip`. One deliberate exception accepts a `.csv` extension with sniffed `text/plain`, because libmagic commonly classifies ordinary CSV that way.

| Formats | Parser | Notes |
| --- | --- | --- |
| PDF | OpenDataLoader PDF, then local Docling/Tesseract only for a typed no-useful-text result | OpenDataLoader keeps JSON output, `xycut` reading order, one CLI thread, image output disabled, and the `docling-fast` hybrid URL. A zero-element result or a result containing only a known scan-app watermark enters OCR. Password-protected and corrupt inputs are terminal; timeouts, unavailable dependencies, and otherwise-unclassified nonzero OpenDataLoader/hybrid exits are retryable. None of those failures enters OCR. |
| DOCX, XLSX, PPTX | Docling | Local `DocumentConverter`; preserves available sections, pages/slides, sheets/cells, tables, and boxes in canonical provenance |
| HTML, HTM, CSV | Docling | Routed through the same canonical adapter |
| PNG, JPG/JPEG, TIF/TIFF, BMP, WebP | Docling with local Tesseract CLI OCR (`eng`, `kan`, `ara`), then local Qwen-VL only when OCR is empty or demonstrably malformed | Supported raster MIME/extension routes only; no runtime OCR-engine download |
| DOC, XLS, PPT | LibreOffice then Docling | Headless safe-mode conversion with an isolated temporary user profile to DOCX/XLSX/PPTX |
| TXT and safe text/code formats (Markdown, JSON/JSONL, YAML, TOML, XML, Python, JavaScript/TypeScript, shell, SQL, CSS, and common compiled-language source) | Text parser | UTF BOM, UTF-8, then CP-1252 decoding; paragraph chunks retain line ranges |

An audio or video MIME type always wins over a misleading extension and is rejected. Other formats return `415` when AI access is requested. Password-protected inputs end in `password_required`; unsupported worker results end in `unsupported`; typed parser or integrity failures end in `failed` unless classified retryable.

The worker installs only Docling's pinned conversion, local-model, PDF-backend, Office, and HTML extras. Docling raster OCR explicitly uses Tesseract CLI with the bounded English, Kannada, and Arabic language data required by the corpus, and disables TableFormer. OpenDataLoader's `auto` hybrid triage sends backend-routed PDF pages to Docling in ordered, bounded internal batches; repeated POSTs for one large PDF are page batches, not worker retries. When OpenDataLoader successfully produces JSON but maps to no elements or only a known scanner watermark, the same local Docling adapter forces full-page Tesseract OCR for that PDF with page-segmentation mode 6, still with TableFormer disabled. The final parser fingerprint records the no-useful-text OpenDataLoader attempt, Docling version, Tesseract CLI version, OCR languages, and PDF OCR mode. The pinned headless OpenCV package remains present because OpenDataLoader's hybrid service imports `cv2`; it is not used as an alternate OCR engine. Normal PDF tables remain OpenDataLoader-owned, while Office tables use Docling's native backends. The worker image bakes the Docling layout and TableFormer models revision- and hash-pinned, so parsing runs offline with no runtime model downloads.

Renaming changes only the display name. Parser routing continues to use the original stored filename and detected MIME type.

## State machine

The exact durable states are:

```text
not_granted
queued -> decrypting -> [converting ->] parsing -> chunking
       -> embedding -> publishing -> ready
```

Terminal alternatives are `failed`, `cancelled`, `unsupported`, and `password_required`. Retryable service/process failures return the same durable job to `queued` with a delayed `available_at` until `max_attempts` is exhausted. The default schema allows three attempts.

An unclassified nonzero OpenDataLoader/hybrid process exit uses the stable `process_failed` code. The worker retains a single-line, 2,000-character maximum diagnostic derived from stderr in the queued/failed job and its warning log. ANSI control sequences, the source and temporary paths, and common secret assignments are removed. OpenDataLoader stdout is deliberately excluded because it can contain converted document output. Known password and corrupt-document signatures keep their non-retryable classifications.

The file API exposes the exact value as `state` or `ingestion_state`. During the compatibility window it also returns `status` or `ai_status`: active states collapse to `processing` and `unsupported` maps to `not_supported`. A cancelled first revision remains `cancelled`; cancellation does not revoke consent.

Poll `GET /api/files/ai-status/{file_id}` for the latest desired job state. A consented file remains searchable whenever `current_revision` is non-null, including while a replacement is queued/running or after that replacement fails; the current revision changes only during atomic publication. The response also includes the desired revision, job UUID, current chunk count, and stable error code/detail.

## Job and consent semantics

- Grant accepts an `Idempotency-Key`. Repeating the key returns the original job; a new key does not create another revision when the current desired revision is already queued, active, or ready.
- Bulk grant locks files in ID order, skips unsupported formats, and returns created/reused jobs.
- Workers claim with `SKIP LOCKED`, heartbeat their lease, and poll for cancellation.
- A transient database or network failure in the outer claim loop does not terminate the worker; it waits before retrying, while an already-claimed job remains recoverable through its lease.
- Every state transition checks user ownership, non-deleted source, active consent, desired revision, lease owner, lease expiry, and cancellation.
- Publication rechecks those fences inside the transaction. A superseded or revoked job cannot become current.
- Explicit per-file **Revoke AI access** cancels active work and deletes `document_revisions`/`document_chunks`; MinIO source objects remain encrypted and available to the user.
- Cancel-indexing stops unfinished work without deleting chunks, revisions, or consent. Files with an older current revision return to `ready`; first-time unfinished files become `cancelled` and can be enqueued again.
- The legacy `disable-all-embeddings` endpoint is a deprecated compatibility alias for the same non-destructive bulk cancellation. It must not erase published embeddings or intelligence.
- Soft file/folder trash follows the same non-destructive cancellation rule. Restore reconciles the status to the preserved current revision. Hard delete, empty trash, and revoke remain destructive.

## Canonical data and chunking

Parser-specific output becomes a shared `CanonicalDocument` containing normalized elements and source locators. Provenance can contain page, slide, sheet, cell range, line range, character range, section path, block ID, and bounding box. Missing fields remain null rather than being invented.

Chunking stays within a citeable source boundary such as a page, slide, sheet, or section. Stable element and chunk IDs are SHA-256-derived from normalized content and fingerprints. `embedding_text` adds bounded structural context while `content` remains the text returned as evidence.

The default embedding model is `snowflake-arctic-embed2:cpu`. The API speaks an OpenAI-compatible `/v1/embeddings` protocol to Ollama, batches eight texts with a 120-second request timeout, retries transient failures up to three times, rejects non-finite values, and requires exactly 1,024 dimensions. The database schema intentionally enforces that dimension.

## Document intelligence

Every newly published revision gets an independently validated controlled
`doc_type`, a summary of at most three complete sentences, and up to six
specific content-derived tags. The worker applies the sentence and 700-character
bound before grounding and never cuts a model or extractive summary mid-sentence.
A bad summary must not discard a valid type or valid tags, and a bad tag must
not discard a grounded summary. Authored books and buyer-issued purchase orders
have distinct `book` and `purchase_order` types; `research_paper` is reserved
for multiple scholarly cues such as an abstract, methodology, references,
citations, or a DOI. Strong visible report and pitchbook titles can classify
raster sources as `report` and `presentation` instead of the generic `image`
type. Model types are checked against source evidence before publication.

A multi-word tag must occur as one normalized contiguous phrase in sampled
source evidence; it does not have to be repeated in the generated summary.
Hyphens, underscores, and repeated spaces normalize to spaces. Person names,
controlled document-type labels, dates/years, visual/layout descriptions,
field/section names, contact data, filenames/formats, and generic metadata are
rejected. Deterministic fallback may retain only tags that pass the same source
grounding rules; an honest empty list is preferred to filename or OCR guesses.
The WebUI performs normalized exact matching for `#tag` searches so one tag does
not accidentally match a longer unrelated tag.

The worker makes at most one bounded local JSON call to `INTELLIGENCE_MODEL`
after embedding and before the atomic publication transaction. Summary
validation checks each sentence and every capitalized entity, then rejects
unsupported numbers, fused identifiers, acronym expansions, document-type
claims, mismatched delivery dates, issuer roles, and tax-inclusive totals. A
timeout, malformed response, evasive or ungrounded field, or unavailable model
cannot fail indexing. A safe extractive fallback is built from source evidence
when possible, and the bounded reason is stored under the current revision's
`metadata.intelligence.fallback_reason`; it contains neither the model response
nor source text. Validated vision descriptions remain visible, but rejected
raster OCR is never copied into `quick_summary`. Scanned-PDF fallback reports
only provable page/script evidence rather than inventing a topic. A fully valid
model result uses `intelligence_status=model`. Filename tokens are not stored as
pretend AI tags at upload time.

The model sees at most 12,000 representative characters in a 16,384-token context.
This worker-only CPU request has a 300-second cold-call timeout; it does not alter
the model's global Ollama placement used by interactive chat.

The public file row stores `doc_type`, `quick_summary`, `quick_tags`, and
`intelligence_status`; the same result is included in the current revision
metadata. The UI presents the physical extension as **Format** and this
controlled `doc_type` as **File type**, never the raw OpenXML MIME string.
Publication updates all intelligence fields with the revision pointer in one
transaction. Reindexing preserves the published revision and its intelligence
until the replacement publishes, so retrieval and type/tag filters remain
usable during processing or after a replacement failure. Per-file revoke
removes them with the derived chunks, while cancel, the deprecated
`disable-all-embeddings` compatibility alias, and soft trash preserve the last
current revision and its intelligence.

Existing current revisions can be classified without decryption, parsing, embedding, or revision changes. The command is dry-run by default:

```bash
docker compose exec api \
  python -m app.ingestion.intelligence_backfill --limit 5

docker compose exec api \
  python -m app.ingestion.intelligence_backfill --apply

docker compose exec api \
  python -m app.ingestion.intelligence_backfill --normalize-tags-only --apply

docker compose exec api \
  python -m app.ingestion.intelligence_backfill --missing-tags-only --limit 10
```

Each dry-run reports changed counts by controlled file type and bounded fallback reason. Full metadata repair deliberately skips raster files because it cannot inspect their decrypted pixels; preserve good VLM metadata and use a targeted reindex for a raster file that genuinely needs repair. Tag-only repair is revision-fenced and applies only the shared safety/normalization policy to existing tags. It preserves otherwise valid tags without treating a representative text sample as complete evidence. `--missing-tags-only` selects only current indexed non-raster revisions whose tag array is empty, runs the configured Document Intelligence role through the foreground-priority gate, and proposes only evidence-grounded tags. Its write fence refuses to overwrite tags that appeared after selection and preserves the summary, controlled type, intelligence status, revision pointer, chunks, and embeddings. Missing-tag raster revisions are reported as targeted-reindex candidates because this command never decrypts source pixels. Neither mode parses sources, changes revision IDs, or writes embeddings.

Full metadata repair and `--missing-tags-only` use the same Redis foreground-priority gate as normal
Document Intelligence publication and defers model calls while chat owns the
lease. `--normalize-tags-only` makes no model call and therefore does not wait
for an inference slot.

## Resource bounds

- Global upload limit: 512 MiB by default, additionally bounded by the user's remaining quota and nginx's 520 MiB request ceiling.
- Docling input limit: 512 MiB; the ODL hybrid service is also configured for 512 MiB.
- Text parser limit: 64 MiB.
- Qwen-VL fallback input limit: 20 MiB per textless or demonstrably weak-OCR raster image, normalized without
  cropping to at most 802,816 pixels and a 1,280-pixel edge. Worker-only vision calls
  use CPU placement, a 16,384-token context, and a 300-second cold-call timeout. Output
  uses one bounded `summary` field containing only key visible text and a concise factual
  description; a safely recoverable partial summary remains usable if its JSON is truncated.
  Markup-like output, implausible long numbers, or structurally noisy text is rejected. If the
  optional vision call is unavailable or invalid, the worker publishes an honest
  `image-description-unavailable` marker instead of making unreliable OCR searchable.
- Raster and scanned-document OCR installs only the corpus languages English, Kannada, and Arabic.
- OpenDataLoader JSON output limit: 128 MiB.
- ODL hybrid request timeout: 3,600,000 ms (one hour); overall ODL parser timeout: 7,200 seconds; LibreOffice conversion timeout: 300 seconds. A slow layout batch can use the larger inner allowance, while the separate parent ceiling still bounds the complete multi-batch parser process and cleanup.
- Scanned-PDF OCR is attempted only after OpenDataLoader's typed no-useful-text result (zero mapped elements or only a known scanner watermark). The worker readiness probe rejects that fallback path, so OCR success cannot hide a broken OpenDataLoader service.
  Cancellation still terminates the OpenDataLoader process group immediately, while the
  finite two-hour ceiling gives large OCR/layout-heavy PDFs bounded completion headroom,
  including the largest thousand-page-plus PDF fixtures.
- Decrypted plaintext exists only within a private temporary lease and is removed after parsing, before chunking and embedding.

Direct Docling conversion currently runs in a worker thread and has no killable hard timeout. Process-isolating that call with an enforceable deadline remains production hardening.

These are correctness and safety limits, not performance guarantees. In particular, DOCX/PPTX/XLSX throughput depends on document structure and available CPU/RAM; benchmark the actual corpus rather than assuming a fixed Docling rate.
