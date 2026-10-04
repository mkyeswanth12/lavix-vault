# Reference

Who this page is for: anyone looking something up. Tables, no prose.

## Services and ports

| Service | Image | Ports | Networks | Limits |
|---|---|---|---|---|
| preflight | busybox:1.37.0 | none | backend | none |
| postgres | pgvector/pgvector:pg15 | none | backend | none |
| redis | redis:7.4-alpine | none | backend | none |
| minio | quay.io/minio/minio:RELEASE.2025-09-07 | none | backend | none |
| minio-init | minio/mc:RELEASE.2025-08-13 | none | backend | one-shot |
| migrate | lavix-vault:1.0.0 | none | backend | one-shot |
| api | lavix-vault:1.0.0 | 9999:8080 | frontend, backend, agent-tools | 4g, 2.0 CPU |
| webui | lavix-vault-webui:1.0.0 | 3005:8080 | frontend | none |
| ingestion-worker | lavix-vault-worker:1.0.0 | none | backend, document-processing | 8g, 4.0 CPU |
| graph-memory-worker | lavix-vault:1.0.0 | none | backend | none |
| agent-runtime | lavix-vault-agent:1.0.0 | none | agent-tools | 3g, 2.0 CPU |
| odl-hybrid | lavix-vault-worker:1.0.0 | none | document-processing | 8g, 4.0 CPU |
| reranker | lavix-vault-reranker:1.0.0 | none | backend | 3g, 2.0 CPU |
| neo4j | neo4j:5-community | none | backend only | 2g, 1.0 CPU |
| searxng | searxng/searxng | none (internal) | backend | none |

## Scripts and flags

| Script | Flags | Purpose |
|---|---|---|
| `scripts/gen-passwords.sh` | `[--write]` | Fill secrets, including `NEO4J_PASSWORD` (`--write` only replaces bare `CHANGE_ME`) |
| `scripts/neo4j-start.sh` | none (container entrypoint) | Refuses placeholder/weak `NEO4J_PASSWORD`, then execs stock Neo4j |
| `scripts/generate-keys.sh` | `[--force]` | RSA-4096 keypair |
| `scripts/preflight.sh` | `[--models]` | Human-readable host checks |
| `scripts/preflight-gate.sh` | env only | In-compose enforcing gate (no flags) |
| `scripts/minio-init.sh` | env only | Bucket + app user (one-shot) |
| `scripts/capture_recovery_manifest.py` | `--output --recovery-id --source-revision --confirm-writers-stopped` | Read-only recovery manifest |
| `scripts/verify_restored_vault.py` | `--manifest --manifest-sha256 --recovery-id --observed-source-revision --confirm-isolated-restore` | Verify a restore |
| `scripts/eval_live_rag.py` | `[--base-url] [--bank] --output` | Live RAG eval (needs token) |
| `scripts/validate_live_rag.py` | `[--base-url] --questions --output` | Repeatable RAG evidence |
| `scripts/recover-graph-jobs.py` (Advanced) | `(--user-id ID\|--all) [--apply] [--print-sql] [--dsn]` | Re-queue graph jobs (dry-run default) |
| `scripts/scrub_credentials.py` (Advanced) | `[--user-id] [--chat-id] [--file-id] [--all] [--apply] [--yes-really] [--confirm-phrase]` | Redact credentials (dry-run default) |
| `scripts/test-browser-smoke.mjs` (Advanced) | `[url]` | Anonymous browser smoke test |

## CLI and entry points

| Command | Purpose |
|---|---|
| `docker compose exec api python -m app.cli promote-admin <username>` | Make a user admin and grant file deletion |
| `docker compose run --rm api python -m app.cli reembed --model <n> --dimensions <d> [--dry-run]` | Migrate vectors (stop workers first) |
| `docker compose run --rm migrate python -m app.db.migrate [up\|status\|verify]` | Migrations |
| `python -m app.ingestion.worker [--factory MOD:fn] [--idle-seconds F]` | Ingestion worker |
| `python -m app.graph_memory.worker [run\|reconcile\|drift] [--idle-seconds F] [--user-id ID]` | Memory worker |
| `python -m app.ingestion.intelligence_backfill [--apply] [--user-id ID] [--limit N] [--normalize-tags-only\|--missing-tags-only]` (Advanced) | Backfill tags (dry-run default) |
| `python -m agent_runtime` | Agent API on 0.0.0.0:8090 |

## Settings, limits, file types, URLs

Settings: see [Configuration](configuration.md) (14 EDIT-block keys). Timeouts: capability 300 s, run 420 s, tool 45 s, read 450 s,
web 12 s. Auth: access 120 min, refresh 7 days, 5 sessions, idle 120 min.
Limits: file 512 MB, avatar ~50 KB, register 3/60 s, login 100/60 s, refresh
10/60 s, query 1000 chars, memory expiry 30/90/365 d (default 90).
File types: PDF (tables), DOCX/XLSX/PPTX/HTML/CSV (Docling), DOC/XLS/PPT
(LibreOffice), common images (vision/OCR), TXT + 33 code/text extensions;
audio/video stored but never indexed; password-protected waits for password.
URLs: UI `http://localhost:3005`, API `http://localhost:9999`
(OpenAPI at `/docs`), API routes under `/api/auth`, `/api/files`,
`/api/folders`, `/api/ai`, `/api/admin` (agent-internal
`/api/internal/agent` is not for browsers).

## Glossary

Admin, AI access (grant/revoke), AI-Ready box, capability (short-lived signed
agent token), chunk, embedding, idle lock, ODL-hybrid, persona, project
(Compose `-p` name), quota, reranker, revision (one indexing pass per file),
scope (`@` file/folder selection), trash, vector, volume: all defined in
[Concepts](concepts.md) and used consistently with those meanings.

Next steps: [Development](development.md), [Troubleshooting](troubleshooting.md).
