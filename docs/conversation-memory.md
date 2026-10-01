# Conversation memory

Lavix Vault supports two user-owned personalization stores. **Saved
Preferences** are manual PostgreSQL notes retained for compatibility.
**Relationship Memory** is the Safe Automatic, expiring memory system stored
canonically in PostgreSQL, with a Neo4j projection (on by default)
and optional question-topic nodes (disabled by default). Neither store
is document evidence, and neither can widen an agent capability.

## Safe Automatic policy

Relationship Memory is disabled by default. Enabling it is an explicit user
choice and requires AI permission. Only assertions from the user's own saved
chat messages are eligible for extraction. Assistant messages, retrieved
documents, web pages, questions, uncertain statements, inferred traits, and
tool output are never authoritative memory sources.

The extraction boundary rejects third-party assertions, secrets and sensitive
content, including passwords, tokens, private keys, contact/government/payment
identifiers, health conditions, income or wealth, and sensitive personal traits.
It requires an explicit personal assertion and deterministic grounding. Subject
is always normalized to `"I"` and the full user message is used as the source
excerpt because the 3B extraction model cannot reliably follow the exact-excerpt
prompt. Model-supplied safety booleans cannot bypass these checks.

The graph worker resolves the current administrator-selected Memory Extraction
role for each job and accepts only a strict bounded JSON schema. In the release
contract, extraction explicitly sends `num_gpu=0` so this low-priority work
cannot consume the constrained GPU. Interactive chat advertises a short
foreground-inference lease in Redis. The background worker defers its Ollama
call while that lease is active—and also defers when it cannot prove the
priority gate is available—without spending a model attempt. Expiry and
clear-fence work do not require that extraction model call. The
acceptance run must prove the chat lease and worker deferral together.

Eligible items use two confidence bands:

- confidence at or above `0.88` becomes active automatically;
- confidence from `0.55` through `0.879` becomes pending review;
- lower-confidence or policy-rejected candidates are not retained as memories.

Pending review items expire after 14 days. Active items use the user's selected
30-, 90-, or 365-day retention, with 90 days as the default. Editing, approving,
renewing, disabling, expiring, deleting, and clearing are explicit lifecycle
events. A longer retention setting must not resurrect an already expired item.

## Authority and storage

PostgreSQL is the sole authority for relationship memory: consent, tenant
identity, source provenance, status, expiry, revisions, clear generations,
embeddings, and job state all live in canonical rows. Neo4j is a rebuildable
projection only — no Neo4j internal identifier is stored or exposed:

```text
saved user message
    -> bounded extraction job
    -> deterministic safety/grounding checks
    -> PostgreSQL graph_memory_items (canonical, projection_state=pending)
    -> project job: Neo4j upsert + pgvector embed (fail-open each)
    -> projection_state=projected | failed | stale
```

`pending` means awaiting projection; `projected` is written only after a
real driver success; `failed` covers terminal job failures and the
deterministic secret backstop (`secret_suspect`, no retry loop); `stale`
means the projected content diverged (or the node is missing) and a healing
upsert job is queued. With no projector configured, rows stay `pending`
and jobs complete PostgreSQL-only (loud warning, never a false
`projected`).

Every lifecycle event—edit, approve, renew, expire, delete, and clear—flows
through project jobs, so Neo4j nodes and edges follow: deletes carry the
fingerprint in the job payload (the canonical row is already gone when the
worker runs), expiry cleans up post-commit, and clear/purge is
generation-fenced (a stale purge never wipes a newer cycle's nodes). A
`drift` CLI (`python -m app.graph_memory.worker drift`) reports
missing/orphan/diverged counts and heals via the upsert path only; orphans
are reported, never auto-deleted outside the tenant fence. Full rebuilds go
through the fenced `reconcile` command.

Every lifecycle event—edit, approve, renew, expire, delete, and clear—updates
canonical rows directly. A generation counter fences work that was queued
before a clear operation, so a late worker cannot repopulate cleared memory.
Failures leave PostgreSQL authoritative and retryable; they never convert stale
state into valid recall.

The long-running worker also survives transient PostgreSQL or Docker DNS
failures. Its claim loop retries with stop-aware exponential backoff capped at
30 seconds; durable leases and generation fences remain the recovery authority,
so a dependency outage never authorizes unfenced writes.

## Recall in CUGA

`recall_graph` is a private, capability-authenticated tool. CUGA supplies only a
query and a bounded result count. The API derives the user from the signed run
capability and re-authorizes every candidate against active, unexpired
PostgreSQL rows before returning anything.

The tool queries PostgreSQL keyword recall first (deterministic,
case-insensitive, active-only), then tops up with pgvector cosine similarity
(`embedding vector(1024)`, snowflake model; NULL rows stay keyword-only),
then with Neo4j graph-traversed facts re-resolved to live active rows.
Question topics (Phase 5, flag-gated) expand the graph traversal terms.
Total recall stays capped (`MAX_RECALL_RESULTS`, default 8) for the
synthesis token budget. Recalled values are delimited as untrusted
personalization: they cannot grant file access, enable web search, change
tool permissions, or override system instructions. They are deliberately
excluded from the evidence store and public Sources cards, and they never
enter query rewriting, planning, or web queries (pinned by
`test_recall_memories_never_alter_web_queries`).

If memory is disabled or clearing is pending, recall fails closed to no
personalization. Vault/web retrieval and chat remain available unless they have
an independent failure.

Recalled personal facts answer personal questions directly: memory recall works
for names, hardware, projects, and preferences.

## About Me

The API returns a human-readable **About Me** profile grouped into facts,
interests/preferences, projects/work, people/entities, and relationships. It is
derived deterministically on each status request from at most 12 current,
active, unexpired items as a single summary paragraph. A copy button in the
Settings surface lets the user copy the summary text.

The Settings surface renders the profile as a summary paragraph. It also shows active and pending items,
source provenance, status and expiry, total/paging, and edit, approve,
renew, and delete controls. The implementation must still pass the real-browser
gate in the the internal release plan.

Deleting or expiring an item removes it from the next derived profile. An empty
profile states that no relationship memories exist; it never invents a user
description from chat history.

## User controls and API

All routes are under `/api/ai` and are authenticated/tenant-scoped.

| Method | Path | Effect |
| --- | --- | --- |
| GET | `/graph-memory` | Settings, counts, expiry, purge state, and About Me |
| PUT | `/graph-memory` | Enable/disable and set 30/90/365-day retention using an expected revision |
| GET | `/graph-memory/items` | Page/filter active or pending facts, preferences, entities, and relationships |
| PATCH | `/graph-memory/items/{id}` | Edit a user-owned item using its expected revision |
| POST | `/graph-memory/items/{id}/approve` | Approve a pending item |
| POST | `/graph-memory/items/{id}/renew` | Refresh an active item's retention |
| DELETE | `/graph-memory/items/{id}` | Delete one relationship-memory item |
| DELETE | `/graph-memory` | Clear Relationship Memory only |
| DELETE | `/personal-memory` | Clear Relationship Memory and manual Saved Preferences |
| GET | `/graph-memory/topics` | View live question topics (flag-gated) |
| DELETE | `/graph-memory/topics/{name}` | Delete one question topic |

The existing `/api/ai/memory` routes continue to manage manual Saved
Preferences. Its legacy consent `PUT` delegates to the same revisioned graph
settings transaction, preserving the selected retention while atomically
mirroring consent in `users.memory_enabled`. Migration 0010 seeds missing graph
tenants from that sole legacy consent record. If both records already exist but
disagree, it fails closed by disabling both and incrementing the graph settings
revision; Saved Preferences and graph items are not deleted. New users remain
disabled by default.
Settings compares both read contracts and fails closed with an explicit repair
warning if a mismatch is ever observed. It exposes both clear operations using
the explicit labels above; a generic “clear memory” label would be too ambiguous.

## Operator lifecycle

PostgreSQL holds every relationship-memory record; back it up as part of the
canonical recovery set. The `graph-memory-worker` container performs lifecycle
work—extraction-job claiming, projection upserts, expiry, TTL cleanup, and
clear-fence enforcement—against PostgreSQL and, when enabled, Neo4j.

Flags (all default OFF in code; ON only in isolated verification stacks):

| Flag | Effect |
| --- | --- |
| `GRAPH_MEMORY_NEO4J_ENABLED` | Attach the Neo4j projection (reads password from `NEO4J_PASSWORD`, falling back to `GRAPH_MEMORY_NEO4J_PASSWORD_FILE`). Startup retries constraints 3x; the worker maintenance loop re-attaches later. Down/slow = memory works via PostgreSQL alone. Never enable on live without Postgres + Neo4j volume backups first (manual rollout step). |
| `GRAPH_QUESTION_MEMORY_ENABLED` | Record deterministic question topics as `(:Topic)` nodes (30-day TTL, per user). Recall boost + follow-up help as synthesis context only. |

The Neo4j service runs by default in `docker-compose.yaml`
(disable with `GRAPH_MEMORY "false"` plus `--scale neo4j=0`). Its `neo4j-data` volume is a disposable projection — excluded
from the canonical recovery set; rebuild from PostgreSQL with
`reconcile`/`drift` and re-verify counts (see `docs/operations.md`). Never
attach a pre-excision graph volume to any stack. The Neo4j password lives
only in the secrets flat (mode 644 so the container uid can read it, matching
sibling flats) and the operator's shell for `MEM_NEO4J_PASSWORD`; it is never
logged, printed, or committed.

An acceptance run must prove tenant separation, opt-in/off behavior, candidate
rejection, pending and active expiry, edit/approve/renew/delete, generation-fenced
clear, bounded recall, About Me provenance, and failure-open
chat behavior. See the
the internal release plan. A
passing unit suite alone does not prove the live background worker.
