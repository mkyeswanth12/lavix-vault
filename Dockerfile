# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
FROM ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a AS uv

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS runtime-deps

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       libmagic1 \
       openssl \
       tini \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 lavix \
    && useradd --uid 10001 --gid lavix --create-home --home-dir /home/lavix lavix \
    && mkdir -p /app/temp /home/lavix/.cache \
    && chown lavix:lavix /app/temp /home/lavix/.cache

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:${PATH}" \
    TEMP_DIR=/app/temp \
    PUBLIC_KEY=/app/enc/public_4096.pem \
    PRIVATE_KEY=/run/secrets/private_key

USER 10001:10001
ENTRYPOINT ["/usr/bin/tini", "--"]

FROM runtime-deps AS runtime
COPY --chown=lavix:lavix app ./app
COPY --chown=lavix:lavix agent_runtime/__init__.py agent_runtime/events.py ./agent_runtime/
COPY --chown=lavix:lavix migrations ./migrations
COPY --chown=lavix:lavix enc/encryption.py ./enc/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

FROM runtime-deps AS worker-deps
USER root
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    apt-get update \
    && apt-get install -y --no-install-recommends \
       libreoffice-calc \
       libreoffice-impress \
       libreoffice-writer \
       openjdk-17-jre-headless \
       tesseract-ocr \
       tesseract-ocr-ara \
       tesseract-ocr-eng \
       tesseract-ocr-kan \
    && rm -rf /var/lib/apt/lists/* \
    && uv sync --frozen --no-dev --group worker --no-install-project
USER 10001:10001

FROM worker-deps AS worker
USER root
# Bake the Docling layout + table-structure models so PDFs parse fully
# offline. Both are pinned by revision and hash-verified, like the reranker
# presets. The table bake covers the accurate TableFormer only (the pipeline
# default); fast mode would download on first use (see README).
ENV HF_HOME=/home/lavix/.cache/huggingface
RUN --mount=type=cache,target=/tmp/hf-xet,sharing=locked \
    HF_XET_CACHE=/tmp/hf-xet python - <<'PY'
import hashlib
import pathlib
from huggingface_hub import snapshot_download

REVISION = "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
FILES = {
    "config.json": "fdea30805ce2f5666b147fca941dcdd27ad468e27d6ed21902207d3da056a97d",
    "model.safetensors": "00333a43451945aaf89db8ca9c0a17e75d1537c17db60fdb91aa95f4c7929e0c",
    "preprocessor_config.json": "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
}
local_dir = pathlib.Path("/home/lavix/.cache/huggingface/hub/models--docling-project--docling-layout-heron/snapshots") / REVISION
snapshot_download(
    repo_id="docling-project/docling-layout-heron",
    revision=REVISION,
    local_dir=str(local_dir),
    allow_patterns=list(FILES),
    max_workers=1,
)
for filename, expected in FILES.items():
    digest = hashlib.sha256((local_dir / filename).read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"Hash mismatch for {filename}: got {digest}, want {expected}")
# Offline resolution needs the ref pointer (transformers asks for "main").
refs = local_dir.parents[1] / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(REVISION)
print(f"baked docling-layout-heron at {REVISION} (hashes verified)")

# TableFormer (accurate) for table-structure extraction. The pipeline asks
# for tag "v2.3.0", which must resolve offline via the refs pointer.
TABLE_TAG = "v2.3.0"
TABLE_SHA = "fc0f2d45e2218ea24bce5045f58a389aed16dc23"
TABLE_FILES = {
    "model_artifacts/tableformer/accurate/tableformer_accurate.safetensors": "2a7d6c924b3cd12fb99a09280ca9c33a89c5d60b93253617d2e088c1a40374d9",
    "model_artifacts/tableformer/accurate/tm_config.json": "984e122ceb8ccf84d84c9d2882f6f2302a44b4f1e577babd6289892c36f3cffd",
}
table_dir = pathlib.Path("/home/lavix/.cache/huggingface/hub/models--docling-project--docling-models/snapshots") / TABLE_SHA
snapshot_download(
    repo_id="docling-project/docling-models",
    revision=TABLE_TAG,
    local_dir=str(table_dir),
    allow_patterns=list(TABLE_FILES),
    max_workers=1,
)
for filename, expected in TABLE_FILES.items():
    digest = hashlib.sha256((table_dir / filename).read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"Hash mismatch for {filename}: got {digest}, want {expected}")
table_refs = table_dir.parents[1] / "refs"
table_refs.mkdir(parents=True, exist_ok=True)
(table_refs / TABLE_TAG).write_text(TABLE_SHA)
print(f"baked docling-models tableformer-accurate at {TABLE_TAG} ({TABLE_SHA}) (hashes verified)")
PY
RUN chown -R lavix:lavix /home/lavix/.cache/huggingface
USER 10001:10001
# Offline by design now that the layout and table-structure models are
# baked: never phone home at parse time (README).
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1
COPY --chown=lavix:lavix app ./app
COPY --chown=lavix:lavix agent_runtime/__init__.py agent_runtime/events.py ./agent_runtime/
COPY --chown=lavix:lavix migrations ./migrations
COPY --chown=lavix:lavix enc/encryption.py ./enc/
CMD ["python", "-m", "app.ingestion.worker"]

FROM runtime-deps AS test-deps
USER root
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    uv sync --frozen --group dev --no-install-project
USER 10001:10001

FROM test-deps AS test
COPY --chown=lavix:lavix app ./app
COPY --chown=lavix:lavix agent_runtime/__init__.py agent_runtime/events.py ./agent_runtime/
COPY --chown=lavix:lavix migrations ./migrations
COPY --chown=lavix:lavix enc/encryption.py ./enc/
COPY --chown=lavix:lavix tests/conftest.py ./tests/conftest.py
COPY --chown=lavix:lavix tests/integration ./tests/integration
COPY --chown=lavix:lavix \
    tests/unit/__init__.py \
    tests/unit/test_foreground_inference_priority_integration.py \
    ./tests/unit/
CMD ["python", "-m", "pytest", "-p", "no:cacheprovider", "tests/integration", "--strict-markers", "-q"]
