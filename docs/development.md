# Development

Who this page is for: contributors. Users can stop at [Reference](reference.md).

## Setup

```bash
uv sync --frozen --all-groups
```

## Running tests and lint

```bash
uv run pytest tests/unit tests/contract -q
uv run ruff check app agent_runtime tests
```

Web UI (from `webui/`): `npm ci`, `npm test` (`node --test
src/frontend-workflows.test.mjs`), `npm run build`, `npm audit --omit=dev`.

## Code layout

`app/` (API, ingestion, auth, agent gateway), `agent_runtime/` (isolated chat
orchestrator, no app secrets), `webui/src` (`App.jsx` UI, `vault-client.js`
API/auth), `migrations/` (numbered SQL, checksummed — never edit an applied
one), `reranker/` + `agent_runtime/Dockerfile` (own images), `scripts/`
(operator tooling), `tests/unit` + `tests/contract` (+ `tests/integration`
for maintainers).

## Migrations rules

New numbered migration per change; the runner records SHA-256 and rejects
edits to applied history. Never edit `0001`/`0014` or any shipped file.

## Building from source

`docker compose build` (runtime, worker with baked models, agent, reranker,
webui). Model bakes are revision-pinned and hash-verified — a hash mismatch
fails the build closed.

## Adding a reranker preset

Add the repo + revision + per-file SHA-256 to `reranker/Dockerfile` presets
(besides `BAAI/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-6-v2`),
rebuild, and extend the contract test allowlist.

## Release checklist

Update version strings, `CHANGELOG.md`, docs screenshots-of-behavior (playwright smoke:
`node scripts/test-browser-smoke.mjs`), full gate green, squash to one orphan
commit, tag, never push secrets (`gitleaks detect --source . --config
.gitleaks.toml`).

Next steps: [Architecture](architecture.md), [Reference](reference.md).
