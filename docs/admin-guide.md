# Admin guide

Who this page is for: Lavix Vault admins (the "ADMIN" tab in Settings).

## Users and roles

Settings > ADMIN > Users. Search users; each row shows status (Admin/Inactive
dot), usage (`X / Y · N files`). Buttons per user: Permissions (expand to
toggle Upload/Download/Delete/AI/Copy/Folders/Rename), Apply Policy (template
dropdown), Storage quota (GB input + Set quota — `0 blocks all uploads`),
Reset PW (modal, min 8 chars), Deactivate/Activate, Make/Revoke Admin.

What admins can and cannot see: admins manage accounts, quotas, models, and
system state. They cannot read other users' file contents or private chats —
only metadata (names, sizes, counts) and system health.

## Registration

Preferences > Public registration toggle. Leave it ON only while people sign
up; turn it OFF afterwards (with it off, `/api/auth/register` returns 403).
First user: register, then run
`docker compose exec api python -m app.cli promote-admin <username>`
(`ok: promoted user … to admin with file delete permission`), then disable registration.
Promotion also grants file deletion (denied by default for new accounts);
re-running it for an already-admin account tops up a missing delete grant.
To take deletion away again, toggle Delete off for that user in Permissions.

## Session timeout and passwords

Preferences > Session timeout: 15 min / 30 min / 1 h / 2 h / 4 h / 8 h / 24 h
(default 2 h). It controls both the login token and idle lock: after the
timeout without activity the session locks (`Session Timed Out`) and sign-in
is required. Password reset is admin-only (Users > Reset PW); there is no
self-service "forgot password". If you are locked out as the only admin, see
[Troubleshooting](troubleshooting.md#login-and-session-problems).

## AI models settings

AI Models tab. Provider is locked to `Ollama (Local)`. Per role (Chat, Vision,
Document intelligence, Memory extraction, Embedding, Reranker): enable toggle,
default-model dropdown, allowed-model fallback dropdown. Max context
(16k recommended; larger needs more RAM — 262k risks out-of-memory). RAG
Scope (File Scope 1–100 files, Top-K 5–500 candidates) and Web Search depth
(Conservative/Balanced/Deep/Pro). Saved model choices override the compose
file. Changing the embedding model later requires
`python -m app.cli reembed` (see [Configuration](configuration.md)).

## Files, access, memory, system

- Bulk Enable/Revoke AI, cancel indexing, trash management per
  [User guide](user-guide.md); `Clear All Embeddings` (System tab, confirmed)
  wipes vectors — files must be re-indexed afterwards.
- Graph memory: user opt-in items are listed under each user's AI Memory;
  admins see counts and retention, not private contents. The Neo4j
  projection needs no extra step (runs by default; see
  [Operations](operations.md#graph-memory)); without it, memory is
  PostgreSQL-only.
- Storage tab: per-user bars (`used/quota`, file counts). System tab: service
  dots + readiness, app version, vector index (`N vectors · X queued`),
  total users. Service state refreshes every 15 s (`Refresh now` forces it).

Next steps: [Operations](operations.md), [Security and privacy](security-and-privacy.md).
