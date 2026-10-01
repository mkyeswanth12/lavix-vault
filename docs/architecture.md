# Architecture

Who this page is for: curious users and contributors. One-line descriptions.

```mermaid
flowchart TB
    user([You, in a browser])
    subgraph stack["Your server: Docker Compose"]
        web["webui<br>port 3005"]
        api["api<br>port 9999<br>login, consent, files, chat"]
        agent["agent-runtime<br>IBM CUGA, sandboxed<br>no keys, no passwords"]
        iw["ingestion-worker<br>decrypt, parse, index"]
        mw["graph-memory-worker"]
        odl["odl-hybrid<br>PDF and OCR parsing"]
        rr["reranker"]
        sx["searxng<br>private web search"]
        pg[("postgres + pgvector")]
        mn[("minio<br>encrypted files")]
        rd[("redis")]
        n4[("neo4j<br>optional")]
    end
    ol[("Ollama<br>outside the stack")]
    user --> web --> api
    api --> pg
    api --> mn
    api --> rd
    api --> rr
    api --> sx
    api -- "signed short-lived pass" --> agent
    agent -- "asks for evidence" --> api
    agent --> ol
    api --> ol
    iw --> pg
    iw --> mn
    iw --> odl
    iw --> ol
    mw --> pg
    mw --> ol
    mw -.-> n4
    api -.-> n4
```
Only `webui` (3005) and `api` (9999) publish ports. `preflight`, `migrate`,
and `minio-init` run once per start and exit.

- `preflight`: refuses startup until credentials are set.
- `migrate` / `minio-init`: one-shot setup (tables; bucket + user).
- `webui` / `api`: what you see / what enforces the rules.
- `ingestion-worker` / `odl-hybrid`: upload-to-ready pipeline (below).
- `agent-runtime`: ask-a-question pipeline (below); holds no secrets, calls
  back with short-lived signed tokens. Built on IBM's open-source CUGA agent
  framework (pinned commit; LangChain only as Ollama adapters). One run at a
  time: a second question waits about 30 seconds, then gets an honest
  "agent busy" message. Each run gets a 300-second signed pass listing
  exactly its files; per-run evidence is deleted afterwards.
- `postgres` / `minio` / `redis`: system of record, bytes, scratch.
- `reranker` / `searxng`: answer quality / optional web.
- `graph-memory-worker`: opt-in preference learning. `neo4j` holds the
  relationship projection the worker maintains: graph-traversed recall on
  top, but PostgreSQL stays canonical — memory works fully without it, and
  the projection rebuilds from Postgres. Opt out with `GRAPH_MEMORY "false"`
  plus `--scale neo4j=0`.

Upload to ready:

```mermaid
flowchart LR
    UP[Upload] --> ENC[Encrypt with RSA-wrapped key]
    ENC --> PARSE[Parse: FastPDF text, else ODL+hybrid tables]
    PARSE --> CHUNK[Chunk: rows keep headers]
    CHUNK --> EMB[Embed via Ollama]
    EMB --> READY[AI-READY]
```

Ask a question:

```mermaid
flowchart LR
    Q[Your question + tagged files] --> RW[Rewrite into search queries]
    RW --> VS[Vector search in your files]
    VS --> RR[Rerank top passages]
    RR --> SYN[Answer from passages only + source cards]
```

Next steps: [Reference](reference.md), [Development](development.md).
