# Publishing to GitHub (fresh history)

The development history of this project contains pre-release credentials and
thousands of WIP commits. The public repository must start from a **fresh,
curated history**. Never push the local `main` branch as-is.

## Pre-flight checklist

1. All gates green:
   ```bash
   uv run pytest tests/unit tests/contract -q
   uv run ruff check app agent_runtime tests
   cd webui && npm ci && npm test && npm run build
   ./scripts/preflight.sh
   ```
2. Secrets sweep of the tree that will be published:
   ```bash
   git grep -nE "[0-9a-f]{40,}" -- docker-compose.yaml app/config/config.example.yml || true
   git ls-files | grep -i '\.env' && echo "MUST print nothing"
   ```
3. Confirm gitignored paths are not tracked: `git ls-files | grep -E 'config.yml|secrets|\.pem'`
   → only `enc/encryption.py` may appear; no `.pem`, no `config.yml`, no `secrets/`.
4. Confirm the published `docker-compose.yaml` still contains only `CHANGE_ME`
   in its EDIT THIS SECTION block (never your edited values).

## Create the public tree

```bash
# From the repo root — creates an orphan branch with one clean commit.
git checkout --orphan public-release
git add -A
git commit -m "Lavix Vault: encrypted document vault with consent-controlled agentic RAG"

# Optional smoke check of what is being published:
git ls-files | less
```

## Push

```bash
gh repo create <name> --public --source . --remote public --push \
  --branch public-release
```

Or manually:

```bash
git remote add public git@github.com:<you>/<name>.git
git push public public-release:main
```

## After publishing

- Rotate every credential listed in SECURITY.md for any deployment that ever
  used them (they were exposed in private history).
- Keep your edited `docker-compose.yaml` and `secrets/` local forever
  (`git update-index --assume-unchanged docker-compose.yaml` hides the local
  edits from git status).
- Tag releases from `public-release` only; never merge old local branches in.
