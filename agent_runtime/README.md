# Lavix CUGA runtime

`agent_runtime` is a private, Ollama-only IBM CUGA sidecar. It is the dedicated chat orchestrator; LangChain's `ChatOllama`, message types, and `StructuredTool` are implementation adapters loaded lazily on the first run.

The dependency set is separate from the API, CPU-only, hash locked, and pins CUGA to commit `ef3eac315d20bb88064755bd87cc6d434b90f8a0`.

## Boundary

The runtime settings deliberately contain no PostgreSQL, Redis, MinIO/S3,
Neo4j, encryption-key, JWT, or end-user identity credentials. Before importing
CUGA, the adapter removes inherited provider/storage credentials and forces
CUGA knowledge, skills, policies, supervisor, reflection, shell, filesystem,
tracing, and external sandbox features off.

For each request the API supplies an opaque, short-lived `X-Lavix-Capability`. CUGA can call only:

- `search_vault`: API-owned pgvector/full-text retrieval within the capability's tenant/file scope.
- `search_web`: API-owned SearXNG search when the capability permits it.
- `recall_graph`: API-owned, PostgreSQL-re-authorized relationship-memory recall when the user has opted in.
- `calculate`: bounded arithmetic with no names, functions, code, shell, or filesystem access.
- `current_datetime`: trusted current time at a bounded fixed UTC offset.

The runtime cannot expand file scope; both its client and the API gateway
intersect requested IDs with the signed scope. Vault/web evidence remains
untrusted input and receives IDs in the API, not in CUGA. Graph recall is also
untrusted personalization, but it is never assigned an evidence ID or shown as
a Source.

## Private HTTP surface

- `GET /internal/v1/health/live`: process liveness.
- `GET /internal/v1/health/ready`: configuration/lazy-initialization state only; it does not invoke Ollama or prove CUGA generation.
- `POST /internal/v1/invoke`: bounded complete JSON response.
- `POST /internal/v1/stream`: newline-delimited safe `status`, incremental `answer_delta`, `final`, `usage`, `error`, and `done` events.

Do not publish this port. CUGA graph/model callbacks never produce public
`answer_delta` events. Plain chat uses one dedicated answer-only streaming
invocation. Calculator/clock requests execute inside CUGA, but only matching
results recorded by the capability-bound tool closures after the pinned
terminal state reach a narrow exact formatter. Vault/web/deep RAG executes the
private CUGA graph, then passes only bounded verified evidence and an optional
sanitized, 8,000-character CUGA draft to the answer-only invocation. Both
authoritative producers carry one fixed internal provenance value; the API
accepts only that provenance and requires the streamed projection to equal the
final answer.
There is no browser replacement/rewind protocol. An incremental projector
holds ambiguous marker/tag prefixes and strips citations, evidence headings,
generated code, tool payloads, logs, and follow-up metadata before a delta
crosses the boundary. The public API converts NDJSON to SSE and owns durable
messages and de-duplicated sources.

## Configuration

Only `LAVIX_AGENT_*` variables are consumed:

| Variable | Default |
| --- | --- |
| `LAVIX_AGENT_OLLAMA_BASE_URL` | `OLLAMA_URL` from `docker-compose.yaml` (EDIT THIS SECTION) |
| `LAVIX_AGENT_MODEL` | `llama3.2:3b` (code default; compose sets no model — the per-request model from Admin Settings governs) |
| `LAVIX_AGENT_ALLOWED_MODELS` | the default model only (per-request admin allowlist governs when present) |
| `LAVIX_AGENT_TOOL_GATEWAY_URL` | `http://api:8080/api/internal/agent` (code default) |
| `LAVIX_AGENT_OLLAMA_NUM_CTX` | `16384` |
| `LAVIX_AGENT_OLLAMA_NUM_PREDICT` | `2048` |
| `LAVIX_AGENT_TOOL_TIMEOUT_SECONDS` | `45` |
| `LAVIX_AGENT_QUEUE_WAIT_TIMEOUT_SECONDS` | `30` |
| `LAVIX_AGENT_RUN_TIMEOUT_SECONDS` | `420` |
| `LAVIX_AGENT_CUGA_MAX_STEPS` | `6` |
| `LAVIX_AGENT_MAX_INPUT_CHARS` | `20000` |

The default model must be in the effective, revisioned system allowlist and the
immutable deployment ceiling; a request-level override is rejected otherwise.
The Chat selector and Settings active-model view persist the same account-wide
preference. OpenRouter and other remote model providers are not supported by
this service.

## Execution limits

One request runs at a time per process because CUGA currently uses
process-global mutable tracking. A request waits at most 30 seconds for that
slot and then receives the stable `agent_busy` error; an admitted run has its
own 180-second limit. The adapter attempts to delete the per-run checkpointer
thread, bounds input and steps, releases the slot during every exit path, and
filters internal script/graph state from events; cleanup failure is logged and
does not replace the answer. Follow-up candidates are generated in the same
final call and travel only in the final metadata block; no second model request
is made merely to populate suggestions.

The Compose container runs as UID `10002`, has a read-only root filesystem, drops all capabilities, sets `no-new-privileges`, and receives no host mounts or Docker socket. Generated Python still executes in this process, so those controls are defense in depth rather than a complete code sandbox. The `agent-tools` Docker network is not an egress firewall; production should restrict outbound traffic to the API gateway and Ollama with an enforceable network policy.
