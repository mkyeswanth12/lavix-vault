# Configuration

Who this page is for: whoever edits `docker-compose.yaml` (EDIT THIS SECTION
at the top). After changing a secret, recreate the containers
(`docker compose up -d`). Never commit the edited file; there is no `.env`
file anywhere — settings live in `docker-compose.yaml`, saved Admin
Preferences, or the optional `config.yml` fallback (advanced; the default
flow never needs it).

## Settings, one by one

| Setting | Allowed values | Default | When to change | Danger if wrong |
|---|---|---|---|---|
| `OLLAMA_URL` | `http(s)://host:port` | `CHANGE_ME` (you must set it) | Always, at install | Containers cannot reach models; preflight refuses to start |
| `EMBEDDING_MODEL` | an Ollama embedding model tag | `snowflake-arctic-embed2:cpu` | Only before first use (see below) | New vectors mismatch old ones; search breaks until `reembed` |
| `EMBEDDING_DIMS` | number matching the model output | `"1024"` | With the model above | Dimension mismatch fails indexing closed |
| `RERANKER_MODE` | `local`, `external`, `off` | `local` | `external`: you run your own reranker; `off`: skip reranking | `external` needs `RERANKER_URL`; `off` lowers answer quality |
| `RERANKER_URL` | `http(s)://host:port` | `http://reranker:7997` | Only with `RERANKER_MODE=external` | Ignored otherwise |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-6-v2` | `BAAI/bge-reranker-base` | To switch presets | Changing it needs `docker compose build reranker` |
| `DB_PASSWORD` | 16+ chars, letters/digits/`_`- only, unique | `CHANGE_ME` | At install | Preflight refuses; duplicates rejected |
| `MINIO_ROOT_PASSWORD` | 16+ chars, unique, must differ from app key | `CHANGE_ME` | At install | Same as above |
| `MINIO_APP_SECRET_KEY` | 16+ chars, unique | `CHANGE_ME` | At install | Same as above |
| `SECRET_KEY` | 32+ chars, must differ from capability secret | `CHANGE_ME` | At install | Logins cannot be signed |
| `AGENT_CAPABILITY_SECRET` | 32+ chars, api only | `CHANGE_ME` | At install | Agent tool calls denied (403) |
| `SEARXNG_SECRET` | 32+ chars | `CHANGE_ME` | At install | Search service refuses to start |
| `GRAPH_MEMORY` | `"true"`, `"false"` | `"true"` | Only to disable the Neo4j projection (with `--scale neo4j=0`) | Anything else fails the gate |
| `NEO4J_PASSWORD` | 16+ chars, letters/digits/`_`- only, unique | `CHANGE_ME` | At install; always checked | Weak/placeholder password fails the gate and Neo4j refuses to start |

> **Do I need this?** Graph memory is a relationship-memory projection in
> Neo4j, on by default. PostgreSQL works without it: memory, recall, and
> About Me all function PostgreSQL-only. Turn it off only if you want a
> lighter stack (see [Operations](operations.md#graph-memory)).

Example (fragments, not a whole file):

```yaml
x-settings:
  OLLAMA_URL: http://192.168.1.10:11434
  EMBEDDING_MODEL: snowflake-arctic-embed2:cpu
  EMBEDDING_DIMS: "1024"
  RERANKER_MODE: local
```

## Choosing models, in simple terms

- **Chat** (answers your questions) and **vision** (reads images): change any
  time in Admin Settings > AI Models. `llama3.2:3b` stays required underneath
  for agent helper steps.
- **Embeddings** (turns text into numbers for search): fixed the first time you
  index anything. The database columns are sized for that model's dimensions.
- **Reranker** (picks the best passages): baked presets only (see table);
  switching presets needs a rebuild.

Warning: changing the embedding model later without migrating breaks search.
The safe path is:

```bash
docker compose run --rm api python -m app.cli reembed --model <name> --dimensions <n> --dry-run
docker compose run --rm api python -m app.cli reembed --model <name> --dimensions <n>
```

Run `--dry-run` first (it prints the plan and changes nothing). Stop the
workers while it runs. This rewrites stored vectors; it does not touch your
files.

## Admin-saved settings (override files)

Session timeout and registration live in Admin Preferences (saved in the
database) and beat the file/env values. Chat/vision model choices in Admin
Settings likewise win. See [Admin guide](admin-guide.md).

Next steps: [User guide](user-guide.md), [Operations](operations.md).
