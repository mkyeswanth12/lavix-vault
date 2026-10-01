# Security Policy

## Reporting a vulnerability

Please do NOT open a public GitHub issue for security reports.
Email **security@example.com** with details (affected component, reproduction
steps, impact). You will receive an acknowledgment within 72 hours.

## Deployment secrets — what lives where

| Artifact | Committed? | Purpose |
|---|---|---|
| `docker-compose.yaml` (EDIT THIS SECTION) | **no** (local edits never committed) | Every secret + `OLLAMA_URL`; services read them as environment variables |
| `app/config/config.example.yml` | yes | Optional-fallback template (advanced non-secret knobs only) |
| `secrets/` | **no** (gitignored) | RSA-4096 signing keypair |

Credentials sit in `docker-compose.yaml` **and** in container environments
(visible via `docker inspect` to anyone with Docker/host access), so protect
the file and the host. The `preflight` service (and `./scripts/preflight.sh`)
refuse startup while any value is `CHANGE_ME`, empty, too short, charset-dirty
(`[^A-Za-z0-9_-]`), or reused across services.

## Credential rotation guide

Rotate any value by editing the EDIT THIS SECTION block in
`docker-compose.yaml`, then restarting the affected services.

| Credential | Rotation effect | Steps |
|---|---|---|
| `SECRET_KEY` | Signs out all sessions | Edit compose → `docker compose up -d api` |
| `AGENT_CAPABILITY_SECRET` | Invalidates in-flight agent capabilities | Edit compose → restart api + agent-runtime |
| `DB_PASSWORD` | Requires coordinated DB change | `ALTER USER vault WITH PASSWORD ...` in Postgres, then compose, restart api/workers |
| MinIO root / app keys | Object-store auth only (file contents stay readable) | Update compose, restart minio/minio-init/api/workers |
| `SEARXNG_SECRET` | Search cache isolation only | Update compose, restart searxng |
| RSA-4096 keypair | **Re-wrap required.** Old ciphertext needs the old private key | Follow docs/security.md rewrap procedure; never delete the old key first |

## Historical exposure note

Before the public release, development history contained fallback signing
keys, MinIO credentials, and a historical RSA keypair
(`enc/unlock/private_4096.pem`). The public repository starts from a fresh
commit history that excludes them, but anyone who ever pulled the private
development tree should treat those values as burned and rotate their own
deployment accordingly.
