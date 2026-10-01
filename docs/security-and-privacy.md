# Security and privacy

Who this page is for: anyone asking "who can read my files?".

## What is encrypted and how

Every uploaded file gets its own random 32-byte key. The file is scrambled
with AES-256-GCM (in 64 MB pieces), and that per-file key is locked with your
RSA-4096 public key. Result per file: `.enc` (contents), `.key.rsa4096`
(locked key), `.sha256` (integrity). Decryption reverses it with
`secrets/private_4096.pem`. Database provider secrets use a separate
`AESGCM(SHA256(SECRET_KEY))` envelope — not the RSA keys.

## Who can read what

- Your files: only you (and admins' metadata view: names, sizes, counts —
  never contents or private chats).
- The AI: only files you granted AI access to, only while granted. Revoking
  deletes the index rows.
- Nobody else: containers talk on private Docker networks; only the web UI
  (3005) and API (9999) are published.

## What never leaves the machine (and the one exception)

Nothing leaves except: (1) prompts/chunks/embeddings sent to **your own**
Ollama server, and (2) the search query text sent to SearXNG **only when you
enable Web Search** (page fetching is off). No telemetry flags exist in the
code. Downloads use `no-store` headers; previews are sandboxed.

## Networks, rotation, backups, limits

- Networks: `frontend` (browser-facing), `backend` (data), `agent-tools`
  (chat orchestrator, no secrets), `document-processing` (PDF parsing).
  SearXNG has no published port. Neo4j (optional profile) sits on `backend`
  only, with no published port.
- Rotate every credential: regenerate in `docker-compose.yaml` (passwords via
  script — including `NEO4J_PASSWORD` when graph memory is on — RSA via
  `generate-keys.sh --force` — old files become unreadable
  with a new keypair, so re-upload), `ALTER USER vault WITH PASSWORD` for
  `DB_PASSWORD`, restart. Token reuse invalidates all sessions by design.
- Protect and back up: `docker-compose.yaml` (uncommitted),
  `secrets/private_4096.pem` (offline copy — loss is permanent), database +
  MinIO dumps ([Operations](operations.md)).
- Honest limits: this does not protect against a compromised host or Docker
  daemon (root there can read everything), weak passwords, or screenshots.
  AGPL pieces (MinIO, SearXNG) are used unmodified as separate services.
- Indexed text and search vectors in Postgres are **not** application-encrypted
  (only access-controlled): use host full-disk encryption, and encrypted swap
  since RAM-backed temporary files can reach disk through host swapping.

Next steps: [Operations](operations.md), [Reference](reference.md).
