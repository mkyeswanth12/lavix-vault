# Security

Lavix Vault uses application-level file encryption and explicit AI consent, but its security depends on correct secret handling and deployment network controls. Read this before using real data.

## Data protection

- Each upload receives a random 256-bit AES key and nonce base. File chunks are encrypted with AES-256-GCM.
- The per-file AES material is wrapped with RSA-4096 and stored as a separate MinIO object. MinIO object names are derived from a fresh file UUID, not the user-supplied filename.
- The plaintext SHA-256 is recorded. Downloads and worker ingestion decrypt into a mode-`0700` temporary lease, verify the hash, return/move only a mode-`0600` response file, and clean it after use.
- Uploads are streamed and bounded by global size, quota, duplicate-name/content, and folder-ownership checks. Partial object-store failures trigger best-effort immediate deletion; a durable orphan-cleanup queue is not yet implemented, so operators must reconcile MinIO against PostgreSQL before production claims of guaranteed compensation.
- MinIO anonymous access is private. Root credentials stay in the server/init job; API and worker receive a bucket-scoped account. The v2 API and worker run non-root, read-only, without Linux capabilities, and receive the private key as a file-backed Compose secret.

Encryption at the application layer does not replace TLS. Terminate HTTPS in front of WebUI/API, protect MinIO/PostgreSQL traffic according to the deployment environment, and encrypt backups.

## Authorization and consent

All ordinary file, folder, retrieval, chat, and memory SQL is tenant-filtered;
explicitly authorized administrator routes can act across users. Permission
flags gate upload, download, delete, AI access, folder operations, and rename.
File deletion (`perm_delete`) is denied by default for new accounts and must be granted explicitly by an administrator; folder operations gate on `perm_folders` instead. File preview requires a short-lived PostgreSQL token and by default is bound to
the API's immediate client address. In the supplied WebUI proxy path that
address is the proxy, not a cryptographically verified originating IP. Preview
responses disable caching and sniffing, apply a CSP sandbox, and downgrade
active HTML/XML/SVG media to plain text. PDF.js renders tokenized PDF bytes in
the trusted WebUI; do not weaken the API CSP to make a browser PDF plugin work.

AI retrieval additionally requires a non-deleted file,
`user_granted_ai_access=true`, a non-null `current_revision`, and the matching
owned `document_revisions.status=current` row. A queued, running, or failed
replacement may change `ai_status` while that older current revision remains
searchable. Revocation cancels workers and deletes derived chunks/revisions.
Revision, consent, and lease fences are checked again at publication time.

Refresh tokens are stored as keyed hashes and rotated after use. PostgreSQL also records session-level human activity. Protected API calls and refreshes enforce a 120-minute idle deadline without extending it; only the authenticated activity endpoint advances that timestamp. Idle expiry revokes the current session UUID, while replay of a rotated token from a still-active session retains the existing all-session reuse response.

## Agent boundary

The CUGA container has no database, Redis, MinIO, private-key, or user-JWT
settings. For each run the API signs a capability with a separate secret. It
expires after 300 seconds, is bound to a run UUID and user ID, and optionally
restricts file IDs, web search, and deep search. Gateway requests intersect
rather than expand file scope. The extra lifetime covers the bounded 30-second
queue wait and 180-second admitted run without turning the capability into a
long-lived credential.

CUGA receives three capability-authenticated read-only tools (`search_vault`,
`search_web`, and `recall_graph`) plus bounded local arithmetic and date/time
utilities. Vault/web content is marked untrusted; evidence is bounded and
assigned server-side IDs. Relationship memories are also untrusted but are not
evidence. CUGA graph/model callbacks cannot cross the streaming boundary. A
dedicated answer-only invocation owns both provenance-marked deltas and the
matching final answer; the API rejects other delta provenance and never appends
a replacement answer. The public projection removes generated code, capability
values, raw tool state, logs, evidence headings, internal citation IDs, and
private follow-up metadata.

Manual Saved Preferences are explicit PostgreSQL tenant records. They share one
atomic consent setting with Safe Automatic Relationship Memory, which can
extract only grounded, non-sensitive assertions from the user's own messages.
Deterministic validation
rejects questions, uncertainty, inferred/non-personal traits, secrets, tokens,
contact identifiers, and low-confidence candidates. PostgreSQL is the sole
authority for consent, provenance, retention, and clear generations. A recalled
set is bounded,
delimited as untrusted personalization, excluded from Sources, and cannot
expand file, tool, web-search, or evidence authority.

Clearing relationship memory increments a generation fence so late jobs cannot
repopulate cleared rows. **Clear all personal memory** also deletes manual
Saved Preferences but never files, embeddings, chat history, or account data.

Important residual limits:

- CUGA executes its generated Python inside its own process. Shell/filesystem CUGA features are disabled and the container is read-only/non-root, but this is still not equivalent to a dedicated code sandbox.
- The Compose `agent-tools` network is not `internal: true`; it separates CUGA from database/storage service membership but does not enforce an outbound network allowlist. Add host/firewall/container egress policy in production.
- The ingestion worker holds database, MinIO, and decryption credentials while running Docling and LibreOffice against untrusted documents. Container hardening limits impact, but parser execution is not yet a credential-free sandbox. Use an additional parser isolation boundary before accepting hostile public uploads.
- The graph-memory worker handles model-produced candidates. Prompt rules are
  not the security boundary: retain deterministic grounding, sensitive-data
  rejection, tenant filters, optimistic revisions, generation fences, and
  PostgreSQL re-authorization during recall.
- One process semaphore serializes CUGA runs because upstream tracking is process-global. Queue admission is bounded and returns `agent_busy` after 30 seconds. Scale with isolated replicas only after preserving per-process separation.
- Authentication throttles currently key on the API's immediate peer. Behind a shared reverse proxy this is a coarse shared limit, not reliable end-client attribution; enforce per-client rate limiting at the trusted edge.
- Registration starts enabled so the first account can be created; turn it off in Admin > Preferences after promoting the owner. Production auth throttling fails closed when Redis is unavailable.

## Web search

SearXNG queries are server-side but its enabled engines are public Internet services. When a user enables web search, CUGA-generated terms—including terms derived from the question or document context—can leave the deployment. Make that disclosure visible to users and disable web search for sensitive material. Live engines are `bing`, `bing news`, and `mwmbl`; Wikipedia and Presearch are disabled, and DuckDuckGo/Startpage are never requested. Optional page fetching accepts HTTP(S), rejects credentials in URLs, resolves and blocks loopback/private/link-local/reserved targets, limits redirects and bytes, and revalidates each redirect. E2E disables page fetching. Keep it disabled unless full-page content is required.

Treat all web and document evidence as untrusted model input. Citation IDs show which evidence was supplied; they do not prove that the source itself is true.

## Secrets

Use independent high-entropy values for database, MinIO, JWT, and agent capabilities. Never place real values in committed templates such as `app/config/config.example.yml`, images, logs, or test artifacts.

| Secret | Rotation effect |
| --- | --- |
| `DB_PASSWORD` | Coordinate API/worker/migration and PostgreSQL |
| MinIO credentials | Coordinate API, worker, init job, and MinIO |
| `SECRET_KEY` | Invalidates JWTs; rotating it signs out every active session |
| `AGENT_CAPABILITY_SECRET` | Invalidates in-flight CUGA capabilities; rotate separately from JWTs |
| RSA key pair | Existing wrapped AES keys cannot be opened after an uncoordinated switch |

`encryption.public_key_path` and `encryption.private_key_path` in
`app/config/config.yml` must name one matched pair. Both files live in the
gitignored `./secrets` directory: the private key enters containers as a Compose
secret mounted at `/run/secrets/private_key`, and the public key binds read-only
at `/app/enc/public_4096.pem`. The private-key file must be owned by UID `10001`
and should be mode `0400` or `0600` with no group/world bits; the public key may
be mode `0644`. Back up the private key in a dedicated secret system, separately
from encrypted data.

## Credential-shaped answers (refusal-wins contract)

When a final answer trips the credential check, the refusal REPLACES it —
there is no redacted variant:

- No raw value appears anywhere: stream deltas, final text, persisted rows,
  history, or logs. The stream projector holds credential-shaped spans, so
  at most a benign text prefix streams before the refusal lands.
- The persisted row and history hold the refusal text itself (it is what the
  user saw); there is no separate `[REDACTED]` persisted form.
- User log lines carry length + hash only (`qlog`), never raw query text.
- Outgoing SearXNG queries pass the credential scrubber; bare non-shaped
  tokens are not scrubbed (documented limit — see B8).

## Required RSA-key remediation

`enc/unlock/private_4096.pem` has existed in repository history and its working-tree mode has been too permissive. Treat that key as compromised even if repository access was limited. The current v2 Compose requires both key files outside the repository, but merely copying the historical pair elsewhere does not rotate it.

A production rotation must be coordinated:

1. Inventory every MinIO wrapped-key object and take a tested backup of PostgreSQL, MinIO, and the currently usable key.
2. Generate a new RSA pair in a protected environment.
3. While the old key is still available, unwrap and rewrap every stored AES-key record with the new public key, or decrypt and re-encrypt the source object. Verify every file hash.
4. Place the new pair in the gitignored `./secrets` directory and point
   `encryption.public_key_path`/`encryption.private_key_path` in `config.yml`
   at the new files.
5. Stop writers for the final cutover, verify all objects with the new key, switch services, and retain the old key only in a controlled recovery escrow until the migration is proven.
6. Remove the private key from the repository and rewrite/purge repository history, then invalidate every clone/cache that retained it.

The repository does not currently provide an automated bulk rewrap command. Do not delete or replace the old key before a migration tool and rollback plan exist; doing so would make existing vault objects unrecoverable.

## Deployment checklist

- Rotate the historically tracked RSA key before production.
- Add durable encrypted-object orphan reconciliation before treating storage compensation as guaranteed.
- Use HTTPS and restrictive CORS origins; do not expose CUGA, MinIO, PostgreSQL, Redis, ODL, Infinity, or SearXNG ports publicly.
- Add an enforceable CUGA/worker egress policy appropriate to the deployment.
- Use separate secrets and volumes for production, staging, and E2E.
- Back up and restore-test PostgreSQL, MinIO, migration history, and key material together.
- Review logs and `artifacts/` before sharing; test logs can contain filenames and operational metadata.
