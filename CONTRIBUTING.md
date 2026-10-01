# Contributing to Lavix Vault

Thanks for your interest in improving Lavix Vault!

## Development setup

```bash
git clone <your-fork-url> && cd Lavix-Vault
./scripts/generate-keys.sh         # RSA-4096 keys into ./secrets (gitignored)
./scripts/gen-passwords.sh --write # fills CHANGE_ME passwords in docker-compose.yaml
# edit docker-compose.yaml: set OLLAMA_URL to http://IP:PORT
uv sync --frozen --all-groups         # Python 3.12
docker compose up --build -d          # full stack (needs Docker + Ollama)
curl -fsS http://localhost:9999/api/health/ready
```

## Checks before opening a PR

```bash
uv run pytest tests/unit tests/contract -q     # backend suites must be green
uv run ruff check app agent_runtime tests      # lint
cd webui && npm ci && npm test && npm run build
./scripts/preflight.sh                         # credentials + keys + Ollama
```

## Ground rules

- **No secrets in the repo.** Real values belong only in your local
  `docker-compose.yaml` (EDIT THIS SECTION) and `secrets/`, both
  gitignored and never committed. New configuration knobs get an environment
  variable read in `app/config.py` (env first, optional `config.yml`
  fallback); secret values never gain committed defaults.
- **PostgreSQL is authoritative.** Optional stores (Redis, object storage) are
  caches or blobs, never the source of truth.
- **Fail closed.** Missing configuration or secrets must stop startup with an
  actionable error, never fall back to permissive defaults.
- Tests pin contracts on purpose (`tests/unit/test_container_contract.py`,
  `frontend-workflows.test.mjs`). If you change behavior, update the contract
  deliberately and say why in the PR.

## Reporting security issues

See SECURITY.md — please do not open public issues for vulnerabilities.
