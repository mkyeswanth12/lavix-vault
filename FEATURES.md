# Lavix Vault — memory subsystem features (v0.9.49-beta → v1.0.0 candidate)

Status key: TESTED = covered by unit tests AND live-verified on an isolated
stack (separate `-p` project + shifted ports). UNIT = unit tests only.
MANUAL = by-hand retest still required on the real stack.

## Relationship memory (Safe Automatic)

| Feature | Status | Evidence |
| --- | --- | --- |
| Chat declaration learning ("my name is X", "call me Y" ack + extract) | TESTED | 15 declaration unit tests; Phase-3 F1/F2 live |
| Name correction supersede (old expires, node removed, Sam only) | TESTED | Phase-3 F3 live |
| Instruction-shaped input never stored ("ask me questions") | TESTED | Phase-3 F4 live |
| Secrets/valueless prompts never stored or projected | TESTED | 19 redaction unit tests; secret backstop unit; Phase-3 F5 live |
| Other-person / health stays pending/rejected, never projected | TESTED | Phase-3 F6 live |
| Approve / edit / renew / delete / clear routes | TESTED | Phase-4 route script 13/13 live |
| Clear generation fencing (stale purge cannot wipe new nodes) | TESTED | unit fence test; Phase-4 clear live |
| Expiry + worker maintenance | TESTED | unit; Phase-3 F3 old-node removal live |
| GPU-lease deferral (Redis foreground lease, fail-closed) | UNIT | `test_worker_defers_model_call_while_foreground_inference_is_active`; contention trigger not forced live |
| Consent switch + dual-flag repair (fails closed on mismatch) | TESTED | Phase-4 consent script live |
| Recall: keyword → pgvector → Neo4j traversal, capped, fail-open | TESTED | 4 vector unit tests; vague probe flipped live ("how should you address me?" → KING) |
| Recall isolation (never R1/planner/web) | TESTED | `test_recall_memories_never_alter_web_queries` re-run; flag-ON canary live (zero Eiffel leak) |
| R1 rewrite window scrubs user-pasted secrets (persist untouched) | TESTED | 2 unit tests |
| About Me deterministic summary + copy | UNIT | pre-existing tests; real-browser gate still pending (see release plan) |

## Neo4j projection (optional, default OFF)

| Feature | Status | Evidence |
| --- | --- | --- |
| Idempotent MERGE upserts, fingerprint key, tenant-scoped | TESTED | 23 projection unit tests; 23/23 isolated-stack checks |
| Pending never projected; state truly moves (projected/failed/stale) | TESTED | unit + live |
| Delete/clear/expire/supersede remove nodes + edges (+orphan entities) | TESTED | unit + live |
| Secret backstop (`secret_suspect`, no retry; drift skips) | TESTED | unit + live |
| Reader 3s timeout, fail-open to PostgreSQL | TESTED | unit + Neo4j-down live run |
| Reconcile (fenced rebuild) + drift report/heal via upsert path | TESTED | unit + live drift CLI |
| Startup retry + maintenance re-attach | TESTED | code + live recovery observed |
| Profile-gated compose service, digest pin, env password from the credentials block | TESTED | container-contract test |

## Question topics (optional, default OFF)

| Feature | Status | Evidence |
| --- | --- | --- |
| Deterministic extraction + strict exclusions | TESTED | shape unit tests |
| Flag/consent gates, 30-day TTL, user controls (view/delete/clear) | TESTED | unit + live (store/list/delete/clear-wipe) |
| Synthesis-only use; canary + isolation with flag ON | TESTED | live canary; isolation pin |

## Saved Preferences (manual)

| Feature | Status | Evidence |
| --- | --- | --- |
| 20-item cap, 500-char limit, consent gating, dedupe | TESTED | Phase-4 script live |

## UNTESTED / manual-only

- Real-browser Settings gate (approve/edit/delete/renew clicks, About Me copy) — see unified release plan.
- Live-stack rollout of either flag (requires Postgres + Neo4j volume backups first).
- `mk`, KING, and drift-canary retests on the real stack (user by-hand step).
- Embedding backfill for pre-pgvector rows (heals naturally on re-projection; no batch job).
- Neo4j nodes for deleted users (no tenant row to scope by; operator-only gap).
