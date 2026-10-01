# Changelog — v0.9.49-beta

What this release contains (from the shipped tree):

- Segmented Docker networks, non-root containers, digest-pinned images.
- Stock-model policy: chat/vision via Ollama, baked local reranker presets.
- Guarded embeddings: dimension/model checks that fail closed, `reembed` path.
- Baked reranker; worker bakes Docling layout + TableFormer models (pinned,
  hash-verified, fully offline parsing).
- ODL table rows as `Header: value` units with per-row pages; zero-text-pages
  warning; no silent node drops.
- Agent capability denial logs a secret-free reason code; contract test bans
  external networks and colliding aliases.
- Admin: AI model roles, session timeout, registration toggle, quotas, user
  permissions, storage/system views.
- Memory: opt-in relationship memory with expiry, About Me, preferences.
- One compose file: the Neo4j overlay is merged in as an optional `graph`
  profile (`GRAPH_MEMORY` + `NEO4J_PASSWORD`); default install unchanged.
