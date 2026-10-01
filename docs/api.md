# API

The public base path is `/api`. Interactive OpenAPI is served at `/docs`. Except for root/health, auth bootstrap, OAuth callbacks, and the tokenized preview URL, user routes require `Authorization: Bearer <access_token>`. Admin routes additionally require `is_admin=true`.

## Minimal upload-to-chat workflow

Register and sign in:

```bash
API=http://localhost:9999/api
curl -sS -X POST "$API/auth/register" \
  -H 'Content-Type: application/json' \
  -d '{"username":"test-user","email":"test@example.com","password":"change-me-123"}'

curl -sS -X POST "$API/auth/login" \
  -H 'Content-Type: application/json' \
  -d '{"username":"test-user","password":"change-me-123"}'
```

Registration must be explicitly enabled by an operator; it is disabled by default in production.

Use the returned access token for subsequent calls:

```bash
TOKEN='<access_token>'
curl -sS -X POST "$API/files/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@/path/report.pdf' \
  -F 'replace=false'

FILE_ID='<file_id>'
curl -sS -X POST "$API/files/grant-ai-access/$FILE_ID" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Idempotency-Key: example-report-v1'

curl -sS "$API/files/ai-status/$FILE_ID" \
  -H "Authorization: Bearer $TOKEN"
```

After `state` becomes `ready`, create a durable chat and stream a scoped answer:

```bash
curl -sS -X POST "$API/ai/chats" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"title":"Report review"}'

CHAT_ID='<chat UUID>'
curl -N -X POST "$API/ai/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"message\":\"Summarize the findings and cite the source.\",\"provider\":\"ollama\",\"chat_id\":\"$CHAT_ID\",\"file_ids\":[$FILE_ID],\"web_search_enabled\":false}"
```

The chat stream is SSE. It emits JSON `status` events, zero or more incremental
`token` deltas, one `sources` event, optional `usage` and `followups`, optional
safe `error` events, and `data: [DONE]`. Clients append each token exactly once;
they must not add a second simulated typewriter delay. The `sources` event
carries `unverified: true` when the answer is background knowledge with no
retrievable evidence (amber banner); absence means evidence-backed.

## Route inventory

### Health and discovery

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/`, `/api` | Version and API discovery |
| GET | `/health`, `/api/health` | Available system/container memory check; returns `503` below 7% |
| GET | `/docs`, `/openapi.json`, `/redoc` | Generated API reference |

### Authentication and account

Every path in this table is below `/api/auth`.

| Method | Paths |
| --- | --- |
| GET | `/capabilities`, `/me`, `/sessions`, `/models`, `/model-config`, `/settings` |
| POST | `/register`, `/login`, `/refresh`, `/activity`, `/logout`, `/change-password` |
| PUT | `/persona`, `/avatar`, `/username`, `/chat-model`, `/settings` |
| DELETE | `/sessions/{session_id}` |
| GET | `/google/login`, `/google/callback`, `/google/token` |

`GET /api/auth/capabilities` exposes only whether registration and Google OAuth are available, allowing the login UI to hide disabled flows. Registration requires `REGISTRATION_ENABLED=true`. Usernames are 3–50 characters using letters, numbers, `.`, `_`, or `-`; passwords are at least eight characters and at most 72 UTF-8 bytes because bcrypt enforces that ceiling. Login, refresh, and Google token exchange return access and refresh JWTs, a session UUID, `last_activity_at`, `idle_expires_at`, and `idle_timeout_seconds`. Refresh tokens are hashed in PostgreSQL and can be revoked by session or logout.

`POST /api/auth/activity` records genuine browser activity for the authenticated session. Other authenticated calls and refresh rotation enforce but do not extend the 120-minute inactivity deadline. At the exact deadline the server revokes only the current session and returns `401` with `detail.code=session_idle_timeout`. `GET /api/auth/sessions` includes creation, last-activity, and idle-expiry timestamps. Google routes require the optional corresponding environment configuration; new OAuth users are also blocked while registration is disabled.

`GET /api/auth/model-config` is the read-only role map used by Settings and
Chat. It includes the revision, active and preferred chat models, system
allowlist/default, and the availability/configuration state for Chat, Vision,
Document Intelligence, Memory Extraction, Embedding, and Reranker.
`PUT /api/auth/chat-model` changes one user's account-wide preference or clears
it to follow the system default. The selected model must be enabled, installed,
and in the system allowlist; both Chat and Settings must reflect the result.

### Files and ingestion

Every path in this table is below `/api/files`.

| Method | Paths |
| --- | --- |
| POST | `/upload`, `/search`, `/grant-ai-access/{file_id}`, `/revoke-ai-access/{file_id}`, `/enable-all-embeddings`, `/disable-all-embeddings`, `/cancel-indexing`, `/cancel-indexing/{file_id}`, `/create-session/{file_id}`, `/copy/{file_id}`, `/restore/{file_id}` |
| GET | `/list`, `/ai-status/{file_id}`, `/download/{file_id}`, `/preview/{token}` |
| PATCH | `/rename/{file_id}`, `/move/{file_id}` |
| DELETE | `/delete/{file_id}`, `/hard-delete/{file_id}`, `/trash/empty` |

Upload is multipart with `file`, optional `folder_id`, and optional `replace`. It stores the encrypted source only; it does not auto-grant AI access. `files/search` searches filename, quick summary, and tags, while `ai/search` searches indexed chunk content. A `files/search` query beginning with `#` matches one complete semantic tag case-insensitively; it never widens `#cloud` to `cloud security`.

The two cancel-indexing routes stop unfinished work while preserving consent and any published revision, embeddings, summary, tags, and type. For safety with cached older clients, `disable-all-embeddings` is a deprecated alias for the same non-destructive bulk cancellation and reports `/api/files/cancel-indexing` as its replacement. Only the explicitly named per-file `revoke-ai-access/{file_id}` operation removes derived index data.

Default permission policy: newly registered accounts receive every
permission flag **except** `perm_delete` (migration `0011` made file
deletion an explicit admin opt-in). A file trash / hard-delete / empty-trash
call without the grant returns `403` with an actionable message naming the
Admin → Users → Permissions grant path. Folder trash/restore instead gate on
`perm_folders` and cascade to contained files; administrators grant or revoke
either flag per user (`PATCH /api/admin/users/{user_id}/permissions`) or by
policy template (`POST /api/admin/users/{user_id}/apply-policy`).

`GET /api/ai/stats` retains the compatibility fields `total`, `tagged`, `ai_ready`, and `failed`, reports exact parser-supported, active, queued, searchable, cancelled, password-required, model-summary, and extractive-fallback-summary counts, and returns `indexable`—consented files whose format the parser supports—as the honest progress denominator alongside the legacy `supported` count.

`create-session/{file_id}` creates a short-lived preview token. Preview tokens are hardcoded to the `preview` purpose, expire in PostgreSQL, and by default are bound to the API's immediate peer address. Direct download still requires a bearer token. The WebUI feeds PDF bytes from this route into its bundled PDF.js viewer; token failure, unsupported bytes, and renderer failure must surface as explicit UI errors rather than a blank successful frame.

### Folders

| Method | Paths |
| --- | --- |
| GET | `/api/folders`, `/api/folders/trash` |
| POST | `/api/folders`, `/api/folders/{folder_id}/restore` |
| PUT | `/api/folders/{folder_id}` |
| PATCH | `/api/folders/{folder_id}/move` |
| DELETE | `/api/folders/{folder_id}`, `/api/folders/{folder_id}/hard` |

Folders are tenant-scoped and hierarchical. Moves reject self/descendant cycles and active sibling-name conflicts. Trashing a folder recursively trashes files and cancels unfinished jobs while preserving published revisions and consent. Restoring one folder does not implicitly restore its deleted descendants or files.

### AI and chat

Every path in this table is below `/api/ai`.

| Method | Paths |
| --- | --- |
| POST | `/chat`, `/chats`, `/search`, `/memory`, `/chat/revoke/{session_id}`, `/graph-memory/items/{id}/approve`, `/graph-memory/items/{id}/renew` |
| GET | `/chats`, `/chats/{chat_id}`, `/chat/sessions`, `/chat/history`, `/memory`, `/graph-memory`, `/graph-memory/items`, `/tags`, `/stats` |
| PUT | `/memory`, `/graph-memory` |
| PATCH | `/chats/{chat_id}`, `/graph-memory/items/{id}` |
| DELETE | `/chats/{chat_id}`, `/chat/history`, `/chat/history/{session_id}`, `/memory`, `/memory/{memory_id}`, `/graph-memory`, `/graph-memory/items/{id}`, `/personal-memory` |

All AI/chat/model utility routes require `perm_ai`. `/api/ai/chat` accepts at most 50 selected file IDs, validates them against caller ownership, consent, and a current ready revision, and accepts only provider `ollama`. It does not accept a per-request model: the server resolves the account-wide Chat/Settings preference under the administrator allowlist, falling back to the administrator default when necessary. Chat-upload selections wait up to 30 seconds for publication and then return the typed SSE error `file_still_processing`; other non-ready selections return `selected_file_not_ready` instead of silently running without evidence. Image-chat input is rejected. Single-file deep questions use scoped `POST /api/ai/chat` with one `file_id` (and optional `deep_search`); the legacy `POST /api/ai/deep-read` route was removed in `0.9.49-beta` — it was a thin wrapper over the same scoped-chat path. The user-facing flow is AI-Ready → Activate (`grant-ai-access`) → indexing → scoped chat/RAG.

`web_search_enabled` forces evidence-backed search (except pure math), overriding
content-gating and identity skips. Factual questions with no usable evidence are
refused with `no_verifiable_evidence` (reason-coded per class in server logs);
generic-conceptual answers without evidence carry `unverified: true` instead of
refusing. Stored assistant messages persist the flag in `content_json`, and
history reloads project it back through the same public contract.

The chat SSE contract emits `status`, incremental final-answer `token` deltas,
`sources`, optional `usage`, optional `followups`, errors, and `[DONE]`. Planning
tokens are never streamed. The answer and durable history are passed through
the same fail-closed public projection, which removes `[V1]`/`[W1]`,
`Evidence:`/`Sources:` lines, private follow-up tags, generated code, tool
payloads, and runtime logs.

Public sources use this lean shape: `id`, `kind`, `match_percentage`, and only
the applicable `file_id`, `filename`, `title`, `mime_type`, or safe HTTP(S)
`url`. Vault hits are grouped by `file_id`; web hits are grouped by canonical
URL after fragments and tracking parameters are removed. The earliest group
position is kept with its strongest match. The WebUI does not render internal
citation IDs or numeric source ordinals. `match_percentage` is an integer from
0 through 100; raw reranker/SearXNG scores and evidence text are not persisted
in the public object. It is a ranking aid, not temperature or truth confidence.

Prompt and completion counts are captured from Ollama callbacks and persisted
with the assistant message. Up to two follow-up questions are generated in the
same final model call, removed from the visible answer, limited to 64 characters
and 10 words, checked for topical overlap, and persisted in `content_json`.
Invalid, generic, source-title-heavy, echoed, or absent suggestions are simply
omitted.

`/api/ai/search` performs tenant-fenced hybrid retrieval without agent generation. It returns at most the requested number of distinct files in ranked order with `match_percentage`, summary, tags, and document type when available. Chunk text and revision/provenance internals remain server-side until the user selects a file. Multiple high-ranked chunks from one file never crowd other file suggestions out of the response.

`/api/ai/memory` is the manual Saved Preferences store. `POST` saves one
user-authored preference, `GET` returns the compatibility consent mirror and
saved items, and `DELETE` clears all or one item. Its legacy `PUT` delegates to
the same atomic Safe Automatic settings transaction; it cannot independently
enable only one memory store. It stores at most 20 items of 500 characters each
in tenant-scoped PostgreSQL and never mines chat. When the shared consent is
enabled, at most six recent items within a fixed character budget are appended
to the runtime prompt as delimited untrusted preferences; they cannot authorize
tools, files, web access, or replace retrieved evidence.

`/api/ai/graph-memory` is the authoritative Safe Automatic relationship-memory
settings surface. `PUT` updates the shared consent and accepts only 30, 90, or
365 retention days plus an optional optimistic `expected_revision`. `GET`
returns counts, expiry,
generation/purge state, and the provenance-backed About Me view. Item routes
list, edit, approve, renew, and delete tenant-owned memories. Pending candidates
have a fixed 14-day lifetime. `DELETE /graph-memory` clears relationship memory;
`DELETE /personal-memory` also clears manual Saved Preferences. Neither route
touches files, embeddings, chats, or account records. See
[Conversation memory](conversation-memory.md) for the full lifecycle.

### Administration

Every path in this table is below `/api/admin`.

| Method | Paths |
| --- | --- |
| GET | `/ai-models`, `/users`, `/storage`, `/system`, `/users/{user_id}/permissions`, `/policies` |
| PUT | `/ai-models` |
| PATCH | `/users/{user_id}`, `/users/{user_id}/permissions` |
| POST | `/users/{user_id}/reset-password`, `/policies`, `/users/{user_id}/apply-policy` |

`/api/admin/system` reports service reachability and PostgreSQL counts for users, vectors, queued jobs, and active jobs. It does not expose credentials.

`GET /api/admin/ai-models` returns the singleton revisioned role configuration,
deployment Chat ceiling, installed Ollama metadata, and availability for Chat,
Vision, Document Intelligence, Memory Extraction, Embedding, and Reranker.
`PUT /api/admin/ai-models` atomically replaces the writable roles using
`expected_revision`; stale writes return `409`. The optional `clock` section
(`{"timezone": "<IANA name>" | null}`) overrides the deployment chat-clock
zone; omitted or null means the `LAVIX_TIMEZONE` default. Unknown zones
return `422`. Only administrators can call
these routes. Chat must keep a non-empty allowlist and a default inside it;
selected tags must exist locally, and the Chat allowlist cannot exceed
`models.allowed_chat_models`. Vision is optional but, when configured, must pass the
Ollama modality probe. Embedding model/dimension and the reranker model are
read-only because they require deployment/index changes rather than an ordinary
settings edit.

## Private agent APIs

The API's `/api/internal/agent/search-vault`, `/search-web`, and `/recall-graph`
routes are excluded from OpenAPI and are not end-user endpoints. They require
both `X-Lavix-Capability` and `X-Lavix-Run-ID`; the capability is minted by the
API for one CUGA run. Graph recall accepts only a bounded query/result count,
derives the user from the capability, and never records memory as public
evidence.

The sidecar exposes `/internal/v1/invoke`, `/internal/v1/stream`, and internal live/ready health routes only on its Compose network. Do not publish these ports.

## Compatibility and errors

The v2 file responses retain legacy `status`/`ai_status` fields while exposing exact `state`/`ingestion_state` fields. New clients should use the exact fields.

HTTP failures use normal status codes. The global FastAPI handler generally returns `{"error": ..., "code": ...}`; validation and some compatibility paths may include additional detail. Clients should branch on the HTTP status and treat error text as display information, not a stable machine contract. Ingestion's `error_code` is the stable programmatic failure identifier.
