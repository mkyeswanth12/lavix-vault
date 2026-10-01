# Troubleshooting

Who this page is for: anyone staring at an error. Find what you see, read why,
do the fix. Run commands from the repo folder.

## Startup: preflight messages

Every `PREFLIGHT FAIL: …` line names the bad setting. The exact templates
(the gate fills in the setting name) and their fixes:

| Exact message (setting name filled in) | Why | Fix |
|---|---|---|
| `<VAR> is empty — set it in docker-compose.yaml (EDIT THIS SECTION) or run ./scripts/gen-passwords.sh --write` | Skipped value | Edit it or run `--write` |
| `<VAR> is still CHANGE_ME — set it in docker-compose.yaml (EDIT THIS SECTION) or run ./scripts/gen-passwords.sh --write` | Placeholder left | Edit it or run `--write` |
| `<VAR> still contains a CHANGE_ME placeholder` | Partial placeholder | Replace the whole value |
| `<VAR> contains characters outside [A-Za-z0-9_-] — regenerate with ./scripts/gen-passwords.sh --write` | Symbols break connection strings | Re-type without symbols |
| `<VAR> is too short (N chars, need >= M) — regenerate with ./scripts/gen-passwords.sh --write` | Below minimum (16; 32 for SECRET_KEY, AGENT_CAPABILITY_SECRET, SEARXNG_SECRET) | Regenerate |
| `duplicate secret values (each must be unique): … — regenerate with ./scripts/gen-passwords.sh --write` | Two secrets share a value | Regenerate with `--write` |
| `SECRET_KEY and AGENT_CAPABILITY_SECRET must be independent` | Same value twice | Make them differ |
| `MINIO_ROOT_PASSWORD and MINIO_APP_SECRET_KEY must differ` | Same value twice | Make them differ |
| `OLLAMA_URL is empty — set it to http://IP:PORT of your Ollama server` | Not set | Set `http://IP:PORT` by hand (never `--write`) |
| `OLLAMA_URL is still CHANGE_ME — set it to http://IP:PORT of your Ollama server` | Placeholder left | Same as above |
| `OLLAMA_URL must start with http:// or https://` | Bad URL shape | Same as above |
| `OLLAMA_URL must include a port` / `OLLAMA_URL port must be numeric` | Missing/bad port | Same as above |
| `RERANKER_MODE is empty — set local, external or off in docker-compose.yaml (x-settings)` | Not set | Use one of the three words |
| `RERANKER_MODE must be local, external or off` | Typo | Use one of the three words |
| `RERANKER_MODEL must be a baked preset (got '…'). Supported presets: BAAI/bge-reranker-base, cross-encoder/ms-marco-MiniLM-L-6-v2` | Unknown model | Use a supported preset, then `docker compose build reranker` |
| `GRAPH_MEMORY must be true or false` | Typo in the flag | Use `"true"` or `"false"` (quoted) |
| `NEO4J_PASSWORD is still CHANGE_ME` / `is empty` (gate or preflight) | Password needed but not set | Set a real 16+ character password or run `./scripts/gen-passwords.sh --write` |
| `NEO4J_PASSWORD is too short` / `contains characters outside [A-Za-z0-9_-]` | Weak/illegal password | Regenerate with `--write` |
| `neo4j: set a real NEO4J_PASSWORD (16+ characters, letters/digits/_/- only)` / `is still CHANGE_ME` / `is too short` / `contains characters outside [A-Za-z0-9_-]` (container log, exit 3) | The Neo4j start script (`scripts/neo4j-start.sh`) refused | Same fix: real password in `docker-compose.yaml`, then `docker compose up -d` |
| `RERANKER_URL is empty — RERANKER_MODE=external needs an http(s)://host:port endpoint` / `must start with http:// or https://` / `must include a port` / `port must be numeric` | External URL missing/malformed | Set a valid URL, or switch mode |
| `secrets/private_4096.pem is missing` | Keys never created | `./scripts/generate-keys.sh` |
| `... not readable by container uid 10001` | Host permissions | `setfacl -m u:10001:r-- secrets/private_4096.pem` |
| `... is a directory ...` (key path) | A bind mount auto-created a folder where a file belongs | `rmdir` it, regenerate with `--force` |
| `.env exists` (warning) | Stray file | Delete it — the stack reads no `.env` |

## Ports, Ollama, models

- **Port already in use (9999, 3005):** warning only. Stop the other service
  or change the published port.
- **Ollama is unreachable:** the URL must work *from inside containers*, not
  just your browser. If Ollama binds loopback-only (`127.0.0.1`), containers
  cannot reach it: make Ollama listen on the LAN/host-gateway interface and
  use that IP (or `http://host.docker.internal:11434` where wired). Then
  confirm tags: `curl -fsS http://IP:PORT/api/tags`.
- **Missing models / wrong dimensions:** pull the exact tags in
  [Install](install.md); `EMBEDDING_DIMS` must equal the model's output
  (1024 for the default). A mismatch fails indexing closed — fix via
  `reembed` ([Configuration](configuration.md)).
- **Slow first start/build:** normal (model layers). Watch
  `docker compose logs odl-hybrid` and `-f ingestion-worker`.

## Login and session problems

- **Wrong password / inactive account:** check caps; ask an admin for Users >
  Reset PW (no self-service reset exists).
- **Session Timed Out / Suspended:** the timeout (default 2 h) locked you out.
  Sign in again; uploads and chats are preserved. Admins change it in
  Preferences > Session timeout.
- **401 bursts on many endpoints at once:** expired token or idle lock; the app
  refreshes transparently once, then asks you to sign in. Files are never lost
  (uploads pause and resume).

## VAULT SEARCH FAILED banner

The card reads "VAULT SEARCH FAILED — Document search hit an internal error.
Retry; if it persists, re-index the file." It means the agent's vault lookup
returned no usable result. Known causes, in order:

1. **Two stacks, one network:** another Lavix project shares an external
   Docker network, so `api` resolves to the wrong stack and its capability is
   rejected (403). Fix with unique network aliases + explicit `LAVIX_AGENT_TOOL_GATEWAY_URL` /
   `AGENT_RUNTIME_URL` (see [Operations](operations.md#two-stacks-uninstall-resources)).
   Retrying usually succeeds — the failure is per lookup, not per file.
2. **File not ready:** still `INDEXING`, `failed`, or access revoked. Wait,
   retry the row, or revoke + grant AI access again.
3. **Nothing matches:** the question needs terms the files do not contain —
   this is correct behavior, not a bug. Rephrase, tag files with `@`, or turn
   Web Search on.
4. **Reranker/embedding outage:** check `docker compose ps` (reranker healthy?)
   and Ollama (embedding model present?).

## Uploads, memory, misc

- **Upload stuck in processing:** watch worker logs; out-of-memory kills look
  like exit 137 — add RAM or scale workers down.
- **`Still indexing — your message is kept`:** the file is not searchable yet.
  Wait for `AI-READY`, then send again.
- **Blank PDF preview / render error:** use Retry/download; very large scans
  need patience (OCR is CPU-bound).
- **Backend Not Responding + low RAM:** wait for the 30 s auto-retry or
  `Retry Now`; free host memory.

## How to report a bug (no secrets)

```bash
docker compose ps
docker compose logs --since 30m api agent-runtime ingestion-worker > lavix-logs.txt
docker compose config | sed 's/CHANGE_ME/REDACTED/g' > lavix-config.txt
```

Send `lavix-logs.txt` + `lavix-config.txt` + what you clicked. Never send
`docker-compose.yaml` unredacted, `secrets/`, tokens, or passwords.

Next steps: [Operations](operations.md), [Security and privacy](security-and-privacy.md).
