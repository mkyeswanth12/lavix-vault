# Operations

Who this page is for: whoever keeps the stack running. Run commands from the
folder containing `docker-compose.yaml`.

## Start, stop, restart, status

```bash
docker compose up -d --build   # start (builds images)
docker compose ps              # status: every service should be Up
docker compose stop            # stop (keeps data)
docker compose up -d           # start again
docker compose restart api    # restart one service
```

Health: `curl -fsS http://localhost:9999/api/health/ready` should print
`{"status":"healthy",...}`. The web UI also has Settings > ADMIN > System
(service dots, readiness, vector counts; refreshes every 15 s).

Logs: `docker compose logs api` (add `--since 30m`, `--tail 100`).
Follow live: `docker compose logs -f ingestion-worker`.

## Updating to a new version

1. Back up (below). 2. `git pull`. 3. `docker compose up -d --build`.
4. Watch `docker compose logs migrate` once (migrations run automatically;
   applied migrations are checksummed — never edit old migration files).

## Backup and restore

Back up three things: the database, object storage, and `secrets/`. If you
use graph memory, also back up the `neo4j-data` volume (it is a disposable
projection — you can also rebuild it from PostgreSQL instead).

```bash
docker compose exec postgres pg_dump -U vault vault > backup-vault.sql
docker compose exec minio mc mirror --overwrite minio/lavix-vault ./backup-minio
cp -r secrets backup-secrets   # includes private_4096.pem (0600)
```

Restore means: fresh volumes, restore the SQL dump, mirror the bucket back,
put the keys in `secrets/`, then `up -d`. Two helper scripts exist for
evidence-grade recovery (see `--help` for exact flags):
`scripts/capture_recovery_manifest.py` (read-only manifest capture) and
`scripts/verify_restored_vault.py` (manifest-bound verification).

The recovery ownership boundary is deliberate:

| Asset | Recovery treatment |
| --- | --- |
| PostgreSQL + MinIO + RSA/secret references | Canonical Lavix recovery set; capture together and restore-test |
| Pre-excision Neo4j volume | Excised system; never attach to v2, operator-retained rollback asset only |
| Projection Neo4j volume (neo4j-data) | Disposable projection; excluded from the canonical recovery set; rebuild from PostgreSQL with the reconcile/drift job and re-verify counts before use |
| Historical Neo4j volume | Operator-retained rollback/quarantine asset only |
| Standalone Ollama model stores | Owned by the Ollama operator; Lavix records only read-only tag/ID/availability evidence and never backs up, starts, stops or recreates Ollama |
| Worker/model caches | Rebuildable deployment artifacts |

The capture utility (`scripts/capture_recovery_manifest.py`, flags `--output
--recovery-id --source-revision --confirm-writers-stopped`, all required)
opens a repeatable-read, read-only PostgreSQL transaction, streams and hashes
the exact encrypted payload/key bytes, and publishes a `lavix.recovery-manifest.v2`
document binding artifacts to a recovery ID, source revision, schema head,
counts, hashes, and UTC interval. Rows carry a redacted stable-identity SHA-256
over the database user plus a SHA-256 of `to_jsonb(files_row)::text`, covering
every physical column. Verification (`scripts/verify_restored_vault.py`, flags
`--manifest --manifest-sha256 --recovery-id --observed-source-revision
--confirm-isolated-restore`) checks the manifest digest, source revision,
tenant set, every referenced object, plaintext hashes, retrieval, chat, and
relationship-memory expiry. Test restores into a different Compose project with
isolated storage — never production volumes or credentials.

Warning: losing `secrets/private_4096.pem` makes encrypted documents
**unrecoverable**. Back it up offline.

## Disk, workers, re-indexing

- Disk: `docker system df`; volumes hold Postgres, MinIO, model cache.
  `docker compose down -v` deletes **everything** (see Uninstall).
- Faster bulk uploads: `docker compose up -d --scale ingestion-worker=2`.
- Re-index one file (new parser, fixed OCR): Files > revoke AI access, then
  grant AI access again. The old index rows are deleted and a new revision is
  parsed, chunked, and embedded from scratch. Bulk: repeat per file (or the
  bulk grant/revoke endpoints). Graph memory is untouched by file re-indexing.
- Neo4j graph projection: see [Graph memory](#graph-memory).
  Postgres stays canonical.

## Graph memory

Graph memory projects relationship memory into Neo4j. It runs by default;
PostgreSQL alone still handles memory, recall, and About Me if you opt out.

### Using it (default)

Nothing to do: plain `docker compose up -d` starts Neo4j (`GRAPH_MEMORY`
`"true"`, real `NEO4J_PASSWORD` required — `gen-passwords.sh --write` fills
it). `docker compose ps` shows `neo4j` healthy.

### Turning it off

```bash
# 1. Set GRAPH_MEMORY to "false" in docker-compose.yaml.
# 2. Start without Neo4j:
docker compose up -d --scale neo4j=0
```

Memory keeps working PostgreSQL-only. The `neo4j-data` volume keeps the old
graph. To delete it for good: `docker compose down -v` removes **all**
volumes (database and files too), or remove just that volume by name while
the stack is stopped.

## Two stacks, uninstall, resources

Two stacks on one host are safe with different project names
(`docker compose -p other up -d`) — but never join them to one shared external
network (see [Troubleshooting](troubleshooting.md#two-stacks-one-network)).

Uninstall completely: `docker compose down -v` deletes containers, networks,
and **all volumes** (database, files, settings, `neo4j-data` if created, keys
in volumes — the `secrets/` folder and repo stay). Also remove the folder
to finish.

Resource use (examples from a 58-page PDF run, `docker stats`, 5 s samples):
odl-hybrid peaks ~4.2 CPU / 3.5 GiB; ingestion-worker ~4.1 CPU / 2.1 GiB;
reranker ~2.0 CPU / 1.6 GiB. First build needs substantial disk/RAM for model
layers. Treat these as examples, not guarantees.

Next steps: [Troubleshooting](troubleshooting.md), [Reference](reference.md).
