# Install

Who this page is for: anyone installing Lavix Vault for the first time.
Run every command from the folder you cloned the repo into (it contains
`docker-compose.yaml`).

## 1. Prerequisites

**Hardware.** 16 GB RAM minimum (32 GB recommended), 4+ CPUs, 30 GB free disk.
The PDF parser and reranker are heavy on first start; 8 GB machines will
struggle or run out of memory.

**Docker.** You need Docker 24+ with the Compose v2 plugin.

- Linux: install [Docker Engine](https://docs.docker.com/engine/install/)
  for your distro, then check:

```bash
docker --version
docker compose version
```

What you should see: `Docker version 24.x` (or newer) and
`Docker Compose version v2.x`.

- macOS: install [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/).
  The same two commands must work in Terminal.
- Windows: install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
  plus [Docker Desktop with WSL2 backend](https://docs.docker.com/desktop/install/windows-install/).
  Run every command below inside the Linux (Ubuntu) terminal, not PowerShell.

**Ollama.** Install [Ollama](https://ollama.com/download) where it can stay
running, then pull the three required models (names are the code defaults;
custom tags are not needed):

```bash
ollama pull llama3.2:3b                # chat + agent helpers (always required)
ollama pull qwen2.5vl:3b                   # vision and document intelligence
ollama pull snowflake-arctic-embed2:cpu    # embeddings (1024 dims)
```

What you should see: each pull ends with `success`. Check with
`curl -fsS http://IP:PORT/api/tags`.

Important: `llama3.2:3b` is always required — the agent's rewrite and planner
steps use it even when Admin Settings selects another chat model. And Ollama
must be reachable **from inside Docker containers**: if Ollama runs on the
Docker host itself, use `http://host.docker.internal:11434` and make sure
Ollama listens on that interface, not only on loopback (see
[Troubleshooting](troubleshooting.md#ollama-is-unreachable)).

**Small tools.** `openssl`, `python3`, `curl`. Check with
`openssl version && python3 --version && curl --version`.

## 2. Get the code

```bash
git clone <this-repo> lavix-vault
cd lavix-vault
```

## 3. Create the keys

```bash
./scripts/generate-keys.sh
```

What you should see: `Keys written (gitignored, never commit them).`
This creates `secrets/private_4096.pem` (mode 0600, readable by container
uid 10001) and `secrets/public_4096.pem` (mode 0644). These two files are
git-ignored and must never be committed. Back up the private key somewhere
offline now — losing it makes encrypted documents **unrecoverable**.
To regenerate, run with `--force`.

## 4. Fill the secrets

```bash
./scripts/gen-passwords.sh --write
```

What you should see: a line per secret confirming it was filled.
This replaces every bare `CHANGE_ME` in `docker-compose.yaml` (it never
touches `OLLAMA_URL` — you set that by hand next).

## 5. Point at Ollama and start

Edit `docker-compose.yaml` (EDIT THIS SECTION at the top): set `OLLAMA_URL`
to `http://IP:PORT` of your Ollama server, reachable from Docker.

```bash
./scripts/preflight.sh
docker compose up -d --build
```

What you should see: preflight prints checks (credentials, keys, Ollama,
ports); then containers start. First start takes a while (image builds plus
baked model layers). Check health:

```bash
curl -fsS http://localhost:9999/api/health/ready
docker compose ps
```

What you should see: `{"status":"healthy",...}` and every service `Up`.

## 6. First login, admin, lockdown

1. Open `http://localhost:3005` and register the first user.
2. Promote it to admin:

```bash
docker compose exec api python -m app.cli promote-admin <username>
```

What you should see: `ok: promoted user '<username>' ... to admin with file delete permission`.

3. As admin, open Settings and turn registration off (Admin Preferences >
   Public registration). Leave it on only if you want strangers to sign up.

## 7. First upload and first question

1. Open Files > Upload, add a PDF, and click **Enable AI** (grant AI access).
   The AI-Ready box shows `INDEXING`; when it flips to `AI-READY`, ask in AI Chat:

```text
What is this document about?
```

What you should see: an answer with a `Sources` row (clickable `V1` card).

Optional: graph memory (Neo4j relationship projection) — see the operations
guide. Skip it unless you know you want it.

Next steps: [Configuration](configuration.md) for every setting,
[User guide](user-guide.md) for daily use.
