# Third-party software

Lavix Vault builds on the following third-party components. All container
base images are digest-pinned in their Dockerfiles; Python/Node
dependencies are hash-pinned (`uv.lock`, `package-lock.json`).

## Container base images

| Image | License | Purpose |
|---|---|---|
| `python:3.12-slim-bookworm` (Docker, PSF) | PSF License | API / workers / agent runtime |
| `ghcr.io/astral-sh/uv` | MIT | Build-time dependency installer |
| `node:22-alpine` | MIT (Node.js) | WebUI build stage |
| `nginxinc/nginx-unprivileged` | BSD-2-Clause (nginx) | WebUI static server |
| `pgvector/pgvector:pg15` | PostgreSQL License | Vector database |
| `redis:7.4-alpine` | BSD-3-Clause (Redis, pre-relicense line) | Queue / cache |
| `quay.io/minio/minio` + `minio/mc` | AGPL-3.0 | Object storage. Note: MinIO server is AGPL; using the official image unmodified over its S3 API does not impose AGPL on Lavix Vault itself, but redistributing a modified MinIO build would. |
| `searxng/searxng` | AGPL-3.0 | Web search (same note as MinIO: used unmodified as a separate service) |
| `michaelf34/infinity` | MIT | Reranker server |
| `neo4j:5-community` (optional profile) | GPL-3.0 (Community Edition) | Optional memory-graph projection, off by default |

## ML models (downloaded at build/run time, not shipped in git)

| Model | License | Purpose |
|---|---|---|
| `BAAI/bge-reranker-base` | MIT | Search reranking (default preset) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 | Search reranking (small preset) |
| `docling-project/docling-layout-heron` (baked, rev `8f39ad3c`) | Apache-2.0 (per model card) | PDF layout detection (worker image) |
| `docling-project/docling-models` v2.3.0 TableFormer-accurate (baked, sha `fc0f2d45`) | CDLA-Permissive-2.0 + Apache-2.0 (per model card; both permit commercial use) | PDF table-structure extraction (worker image) |
| `nomic-embed-text` (nomic-ai/nomic-embed-text-v1) | Apache-2.0 | Optional 768-dim embeddings |
| Ollama models you pull yourself (e.g. llama/qwen families, `snowflake-arctic-embed2`) | Per-model licenses on ollama.com / Hugging Face — check before commercial use | Chat, vision, embeddings |

## Agent framework (bundled in the agent image, not shipped in git)

| Component | License | Purpose |
|---|---|---|
| `cuga-project/cuga-agent` @ `ef3eac3` (commit-pinned URL) | Apache-2.0 (from the pinned tarball's LICENSE) | Agent orchestration sidecar |
| `langchain-ollama` 1.1.0 | MIT (from wheel metadata) | Ollama chat/message/tool adapters only |
| `langgraph` 1.2.9 | MIT (from wheel metadata) | Agent dependency |
| `neo4j` Python driver 5.28.2 | Apache-2.0 (from installed metadata) | Optional graph projection client (server image row above states its license as previously recorded) |

## Notable runtime dependencies

- Tesseract OCR (`Apache-2.0`), LibreOffice (`MPL-2.0`/`LGPL`), Poppler (`GPL-2.0/3.0`, used unmodified as CLI tools inside the worker image)
- PostgreSQL `pgvector` extension (PostgreSQL License)
- See `uv.lock` and `webui/package-lock.json` for the full pinned dependency trees and their licenses.

If you redistribute Lavix Vault (especially the MinIO/SearXNG/Neo4j/Poppler pieces or modified builds of them), review the AGPL/GPL obligations for your use case.
