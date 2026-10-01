# Testing

The repository separates fast code tests from disposable-stack checks. No test command should use normal deployment volumes or mutate the source corpus.

## Local checks

Install the locked application and development dependencies:

```bash
uv sync --frozen --all-groups
```

Run the current Python suite and static checks:

```bash
uv run pytest tests/unit tests/contract -q
uv run ruff check app agent_runtime tests
```

The release unit suite must cover capability signing, CUGA event filtering and model
isolation, final-node delta projection, public answer/source/follow-up contracts,
gateway scope intersection, chat persistence/streaming, revisioned AI-model
configuration, tenant-scoped manual and relationship memory, pgvector query
fences, parser routing/adapters, independently
validated intelligence fields, exact tag matching, stable models/chunks, worker
leases/transitions/retries, non-destructive cancellation, atomic publication,
source integrity, encrypted storage compensation, preview/file/folder routes,
migrations, startup readiness, web-search SSRF controls, synchronized recovery
manifest capture/restore verification, and Compose contracts.

The focused recovery-contract tests require no live service and exercise exact
stream hashing, the 2 GiB object fence, response cleanup, object-generation
change detection, final database stability, verifier-compatible v2 encoding,
mode-`0600` atomic no-replace publication, and redacted receipts:

```bash
uv run pytest \
  tests/unit/test_recovery_manifest_capture.py \
  tests/unit/test_recovery_verifier.py -q
```

These fakes prove the code contract, not backup validity. Release evidence still
requires a synchronized capture with stopped writers and a manifest-bound
rehearsal against separate PostgreSQL/MinIO endpoints.

Contract tests exercise durable chat/session compatibility. Tests use fakes for external services unless marked otherwise; a green unit suite is not evidence that local Ollama, OpenDataLoader, LibreOffice, MinIO, or CUGA can process a real corpus.

The frontend has a direct workflow test, production build, and production-dependency audit:

```bash
cd webui
npm ci
npm test
npm run build
npm audit --omit=dev
```

The workflow test uses JSDOM. It is valuable for helpers, client contracts, and
regressions. Its behavioral PDF coverage includes per-file generation fencing,
same-file request coalescing, cancellation/supersession of hung renderer work,
detachment of a hung session request, retry-cooldown reset, and cache-clear
invalidation. Existing source-contract assertions also confirm that the React
UI wires the pre-redesign control, generated/invalidated events, and a 45-second
deadline; they do not mount that UI or advance the deadline. Deadline
feedback, decoded canvas/image output, responsive hit targets, and visible
cross-instance state belong to the authenticated real-browser gate.

Run the dependency-free anonymous Chromium baseline against a started WebUI:

```bash
node scripts/test-browser-smoke.mjs http://127.0.0.1:3005/
```

The script uses an isolated temporary browser profile, supplies no credentials,
and performs no application writes. It fails on an uncaught JavaScript error,
the visible application error boundary, a failed static browser request, an
empty React root, or an unreachable/mis-typed bundled PDF.js module worker. Set
`CHROME_BIN` when Chrome or Chromium is outside the usual system paths. This is
a fast deployment smoke check only; it does not replace the authenticated
real-browser acceptance gate below, which must render a real PDF, inspect
responsive geometry, and observe live chat DOM updates.

The disposable `smoke` path additionally exercised an authenticated Chromium
gate through a random loopback-only WebUI port against a throwaway project.
That gate is currently manual: create a normal user through the rendered
registration form, then verify login, upload, chat streaming, and Settings
reads by hand. Passwords are generated in memory and are never printed or
written to the repository.

The authenticated gate is required to prove all of the following in real
Chromium. The harness implements these assertions and their artifact outputs,
but capability is not release evidence: only a passing final manifest-bound run
against one immutable source identity can close this gate:

- registration and normal/admin login through the rendered forms, the exact
  120-minute server deadline, and proof that background API polling does not
  extend it;
- a real encrypted two-page PDF, first-page thumbnail, and one quiet 40-by-40
  rotate-icon action in the details Preview header and full-viewer header. Both
  use the accessible name **Regenerate thumbnail** and the same generation path.
  Mouse and Enter activation must expose loading/completion states, await a
  decoded image, avoid unintended viewer/collapse changes, and synchronize
  mounted list, grid, and detail instances.
  The 45-second deadline must be accelerated deterministically, and a late valid
  result from work that ignores cancellation must remain fenced from the cache
  and every mounted instance;
- one shared injected thumbnail failure with exactly one header recovery action
  and one status message in details/viewer, plus one **Try again** control in
  list and grid, followed by successful recovery; every action must remain
  unobstructed and at least 40px with no panel/page overflow at 800px and 390px;
- mounted PPT and PPTX previews that retain their friendly format fallback
  instead of entering an erroneous thumbnail-retry state, plus a failed
  direct-image preview whose retry starts a fresh preview session and recovers;
- an uppercase `.PDF` discoverable through `original_filename` while stored with
  generic MIME, a non-empty PDF.js canvas,
  next/previous navigation, injected full-viewer failure with visible retry, and
  an injected download failure with an actionable retry. Viewer and error/retry
  controls must remain unobstructed and at least 40px at 800px and 390px;
- overflowing document-type chips with external left/right controls, no visible
  scrollbar, no button overlap, 40px arrow targets, no page-level horizontal
  overflow, and a selected chip that remains reachable after narrowing to
  390px before the list state is restored;
- at least two distinct, growing assistant DOM states before the same response
  completes, followed by exact PostgreSQL-backed restoration after page reload;
- sanitized assistant text, de-duplicated percentage source cards, and exact
  agreement between persisted and rendered zero-to-two concise, topic-specific
  follow-ups, with missing/invalid candidates left empty and generic/source-title
  templates rejected;
- a deterministic arithmetic request that reaches CUGA's bounded calculator
  stage, returns `745`, and persists without exposing execution protocol;
- six synchronized model roles, non-admin read-only presentation, an admin-only
  optional-VLM revision observed by the normal user, and restoration of the
  original setting;
- Safe Automatic memory, 30/90/365-day choices with a 30-to-90 round trip,
  About Me and relationship-memory transparency, saved-preference creation,
  relationship-only clear that preserves the preference, and all-personal
  clear that removes it; and
- a healthy or busy OpenDataLoader card described as private-network
  infrastructure, never as an isolated failure.

To force chip overflow without twelve redundant model classifications, the gate
uploads test-owned encrypted files and assigns their display-only document-type
fixture metadata directly in the disposable database. It verifies the
container's Compose project/service labels before that write. Real type
generation remains covered by the corpus profile; this browser assertion claims
only rendered geometry for its fixture.

The current browser script proved the old wide regeneration pill, its geometry,
real PDF canvas/worker behavior, uppercase display-filename fallback, deadline
fencing, cross-mount recovery, document-chip layout, Office fallbacks, and
direct-image retry in historical run `515127-a5fa1e78`. It also emitted six
narrow/desktop state screenshots. The harness must now be adapted for
`original_filename`, the compact details/full-viewer action, and the no-duplicate
failure contract. Only a new passing manifest-bound run closes the gate.

The gate does not validate Google OAuth, a public TLS/reverse-proxy path,
password entry for protected documents, assistive-technology behavior, or a
production user's existing data. Those require separately authorized
environments and are not inferred from isolated browser evidence.

## Disposable verification

There is no committed disposable-stack harness: verification runs use a
throwaway Compose project (`docker compose -p lavix-verify-<id>`) with
shifted published ports against a copy of `docker-compose.yaml` filled with
fresh `gen-passwords.sh` credentials. Teardown (`docker compose -p
<project> down --volumes`) must remove only that project's resources — never
Ollama data, model tags, or normal Lavix volumes.

## Release verification checklist

Before a release claim, verify each of the following against a throwaway
project (never the live stack). On 2026-07-16 an isolated drill passed the
recovery subset: two enabled tenants, active and pending expiry,
edit/renew/approve/delete, paging and tenant fences, safe-empty recall with a
persisted ordinary answer, post-restart recall, and generation-fenced clear.

- API/WebUI, Ollama, reranker, and graph-worker health; refresh
  rotation/logout/120-minute idle contract; encrypted text round trip;
  pgvector retrieval/reranking; true CUGA token deltas and sanitized durable
  answer; de-duplicated percentage sources; meaningful follow-ups and bounded
  calculator; admin model revision and Chat/Settings sync; tenant fences;
  preview/file/folder/trash lifecycle; non-destructive cancel versus
  destructive revoke; Safe Automatic memory lifecycle; real-browser
  PDF/stream/layout/Settings pass.
- Stratified real text plus a generated, verified text-layer-free PDF that
  must cross ODL's typed-empty boundary into bounded local OCR; DOC/DOCX,
  PPT/PPTX, XLS/XLSX, JPEG/PNG/TIFF/WebP/BMP OCR, HTML, and CSV; canonical
  chunks, 1,024-dimensional vectors, useful independently validated
  summary/type/tags (and no invented metadata for sparse OCR), exact
  filters/retrieval; unsupported video, password-locked PDF, and corrupt
  Office failure states.
- Full orchestration through real Ollama and CUGA, including more than one
  observable final-answer token delta when the model emits them, no private
  protocol in stream or history, returned vault evidence, and
  PostgreSQL-persisted user/assistant messages. A vault-only assertion
  disables web search so SearXNG cannot mask failed retrieval; a separate
  assertion tests live opted-in web evidence.
- A live inference-priority drill: stage API-created Document Intelligence
  and graph Memory Extraction jobs behind a temporary Redis scheduling fence,
  start overlapping authenticated CUGA streams, remove only that fence, and
  prove both real workers claim and defer their jobs while the CUGA leases
  remain active. Both deferrals must refund their attempt, publish nothing
  early, and later finish through their configured models with exactly one
  charged attempt.

### Live web eval and refusal monitoring

`tests/eval/run_web_eval.py` scores fixed retrieval + faithfulness question
sets against the running stack (grades A≥90/B≥75/C≥60; refusals count as
failures, 0 fabrications is the hard requirement). It needs an operator
login token:

```bash
python tests/eval/run_web_eval.py --token <JWT> --base http://localhost:9999
```

Re-rate after any retrieval change. Between evals, watch refusal calibration
in production logs — per-class codes exist for tuning the parametric block:

```bash
docker compose logs agent-runtime | grep -E "refused=no_evidence_|unverified=explanatory"
```

The format corpus below needs a read-only fixture directory for manual
verification runs (PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX, JPEG/PNG/TIFF/WebP/BMP,
HTML, CSV, plus video/protected/corrupt failure states). Every successful
format case must have a current revision, non-empty chunks, 1,024-dimensional
embeddings, and independently grounded intelligence.

## Authoritative release acceptance

The whole-product implementation order, open items, browser requirements,
standalone Ollama/GPU checks, memory-lifecycle drill, cleanup rules, and final
evidence manifest live only in the
the internal release plan. This
document defines how individual test profiles run; it is not a second release
plan.

## Coverage boundaries

HTTP-level checks cannot by itself prove browser PDF.js rendering, thumbnail canvas
output, chip geometry, or visible token cadence; use the anonymous Chromium
smoke script plus a recorded manual pass for those. Password entry for protected documents,
Google OAuth, TLS/egress
controls, parser adversarial fuzzing, disaster recovery, and long-duration load
remain separate gates unless a dated artifact explicitly proves them.
OpenRouter support was removed entirely; chat is Ollama-only through the
isolated CUGA sidecar, and no provider compatibility surface remains to test.

The dated recovery drill proves relationship-memory lifecycle behavior
only. The synchronized PostgreSQL/MinIO/RSA capture, isolated restore,
and digest-bound all-file decrypt/hash verification have not been rehearsed and
remain a release blocker under the unified plan.

Passing throwaway-project checks does not update production. The unified plan's
promotion gate separately verifies that the public deployment runs the exact
tested image IDs, serves the expected hashed WebUI asset, migrates to the
expected schema, and passes authenticated live smoke. Restarting an older baked
WebUI container is not deployment of newer source.
