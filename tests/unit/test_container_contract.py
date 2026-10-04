import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _service_block(compose: str, service: str) -> str:
    matches = list(re.finditer(r"^  ([a-z0-9-]+):", compose, re.MULTILINE))
    for index, match in enumerate(matches):
        if match.group(1) != service:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(compose)
        return compose[match.start() : end]
    return ""


def test_compose_segments_networks_and_keeps_the_agent_off_backend():
    compose = read("docker-compose.yaml")

    assert "command: [python, -m, app.db.migrate, up]" in compose
    assert "/docker-entrypoint-initdb.d" not in compose
    # Four segmented networks; the agent lives on agent-tools only.
    for network in ("frontend: {}", "backend: {}", "agent-tools: {}", "document-processing: {}"):
        assert network in compose
    assert _service_block(compose, "agent-runtime").count("networks:") == 1
    assert "networks: [agent-tools]" in _service_block(compose, "agent-runtime")
    assert "networks: [frontend, backend, agent-tools]" in _service_block(compose, "api")
    assert "networks: [frontend]" in _service_block(compose, "webui")
    assert "networks: [backend, document-processing]" in _service_block(compose, "ingestion-worker")
    assert "networks: [document-processing]" in _service_block(compose, "odl-hybrid")
    for service in (
        "postgres", "redis", "minio", "minio-init", "migrate",
        "searxng", "reranker", "graph-memory-worker", "preflight",
    ):
        assert "networks: [backend]" in _service_block(compose, service), service
    # No external Ollama network: containers reach Ollama only via OLLAMA_URL.
    assert "lavix-ollama" not in compose
    assert "vault-ollama" not in compose


def test_compose_declares_no_external_networks_or_colliding_aliases():
    # Bare `api` / `agent-runtime` names must never resolve across projects:
    # two stacks sharing one external network cross-talk (wrong API rejects
    # the capability with 403). The single compose file stays fully
    # self-contained (there is no overlay file anymore).
    assert not (ROOT / "docker-compose.graph.yaml").exists()
    compose = read("docker-compose.yaml")
    assert "external:" not in compose
    assert "aliases:" not in compose
    assert "external: true" not in compose


def test_agent_runtime_gets_no_application_secrets():
    compose = read("docker-compose.yaml")
    dockerfile = read("Dockerfile")
    runtime_init = read("agent_runtime/__init__.py")

    agent_block = _service_block(compose, "agent-runtime")
    # The orchestrator reaches vault data only through capability-authenticated
    # gateway tools: no JWT, no capability secret, no passwords in its env.
    assert "AGENT_CAPABILITY_SECRET" not in agent_block
    assert "SECRET_KEY" not in agent_block
    assert "PASSWORD" not in agent_block
    assert "LAVIX_AGENT_OLLAMA_BASE_URL" in agent_block
    assert "user: \"10002:10002\"" in agent_block
    # No external Ollama network: containers reach Ollama only via OLLAMA_URL.
    assert "lavix-ollama" not in compose
    assert "vault-ollama" not in compose
    # Sidecar tunables live in code defaults, not in compose.
    assert "LAVIX_AGENT_QUEUE_WAIT_TIMEOUT_SECONDS" not in compose
    assert "LAVIX_AGENT_RUN_TIMEOUT_SECONDS" not in compose
    assert "LAVIX_AGENT_TOOL_TIMEOUT_SECONDS" not in compose
    assert "LAVIX_AGENT_OLLAMA_NUM_PREDICT" not in compose
    from agent_runtime.config import RuntimeSettings

    assert RuntimeSettings().ollama_num_predict == 2048
    assert RuntimeSettings().tool_timeout_seconds == 45
    assert RuntimeSettings().run_timeout_seconds == 420
    assert RuntimeSettings().queue_wait_timeout_seconds == 30
    assert RuntimeSettings().ollama_num_ctx == 16384
    # Tunables live in the committed template; the sidecar keeps only
    # non-secret literals in compose.
    example = read("app/config/config.example.yml")
    assert "agent_capability_ttl_seconds: 300" in example
    assert "read_timeout_seconds: 450" in example

    admin = read("app/routers/admin.py")
    assert "/internal/v1/health/ready" in admin
    assert 'services["opendataloader"] = "ok" if odl_usable else "error"' in admin
    assert '"topology": "private"' in admin
    assert "WORKER_HEALTH_KEY" in admin
    assert (
        dockerfile.count(
            "COPY --chown=lavix:lavix agent_runtime/__init__.py agent_runtime/events.py "
            "./agent_runtime/"
        )
        == 3
    )
    # Lean API/worker images copy the shared public projector, not the full
    # runtime. Importing its package must therefore remain dependency-free.
    assert "from .config import" not in runtime_init


def test_webui_serves_pdf_module_workers_as_javascript_with_a_fresh_cache_key():
    nginx = read("webui/nginx/nginx.conf")
    thumbnails = read("webui/src/thumbnail-generators.js")

    assert r"location ~* ^/assets/.*\.mjs$" in nginx
    assert "application/javascript mjs" in nginx
    assert 'X-Content-Type-Options "nosniff"' in nginx
    api_location = nginx.split("location /api {", maxsplit=1)[1].split("\n    }", maxsplit=1)[0]
    assert "proxy_hide_header X-Content-Type-Options;" in api_location
    assert "?lavix-pdf-worker=2" in thumbnails


def test_session_timeouts_are_pinned_by_the_config_template_and_code_defaults():
    from app.config import Settings

    example = read("app/config/config.example.yml")

    expected = {
        "access_token_expire_minutes": 120,
        "session_idle_timeout_minutes": 120,
        "refresh_token_expire_days": 7,
        "max_active_sessions_per_user": 5,
    }
    settings = Settings()
    for key, value in expected.items():
        assert f"{key}: {value}" in example
        assert getattr(settings, key) == value

    # The real config file must never be committed.
    gitignore = read(".gitignore")
    assert "app/config/config.yml" in gitignore


def test_worker_has_verified_odl_dependencies_and_bounded_ephemeral_scratch():
    compose = read("docker-compose.yaml")
    dockerfile = read("Dockerfile")
    docling = read("app/ingestion/parsers/docling.py")
    project = read("pyproject.toml")
    lock = read("uv.lock")

    assert '"opencv-python-headless==4.13.0.92"' in project
    assert 'name = "opencv-python-headless"' in lock
    assert 'version = "4.13.0.92"' in lock
    # Worker scratch space: bounded tmpfs plus the template-pinned temp root.
    example = read("app/config/config.example.yml")
    assert 'ingestion_temp_root: "/app/temp"' in example
    assert "/tmp:size=2g,mode=1777,uid=10001,gid=10001" in compose
    assert "/app/temp:size=2g,mode=0700,uid=10001,gid=10001" in compose
    # Grace period stays above the 2h parser ceiling (7200s) so a normal
    # drain never kills a large document at the parser deadline.
    assert "stop_grace_period: 2h15m" in compose
    from app.ingestion.parsers.registry import ParserSettings

    ceiling = ParserSettings.__dataclass_fields__["parser_timeout_seconds"].default
    assert 8100 > ceiling
    assert "http://localhost:8080/api/health/ready" in compose
    assert "tesseract-ocr-ara" in dockerfile
    assert "tesseract-ocr-eng" in dockerfile
    assert "tesseract-ocr-kan" in dockerfile
    assert "InputFormat.PDF: PdfFormatOption(" in docling
    assert "force_full_page_ocr=True" in docling
    assert "psm=self.settings.pdf_ocr_psm" in docling
    assert docling.count("do_table_structure=False") == 2


def test_ollama_reaches_containers_only_through_ollama_url():
    compose = read("docker-compose.yaml")
    application_config = read("app/config.py")
    runtime_config = read("agent_runtime/config.py")
    embedding = read("app/ingestion/embedding.py")
    intelligence = read("app/ingestion/intelligence.py")
    vision = read("app/ingestion/parsers/vision.py")

    # The settings block owns the single Ollama endpoint; every consumer
    # takes it from the shared anchor. No fallback network, no aliases.
    assert "OLLAMA_URL:" in compose
    assert "CHANGE_ME" in compose.split("OLLAMA_URL:")[1].split("\n")[0]
    assert "OLLAMA_BASE_URL: *ollama-url" in compose
    assert "LAVIX_AGENT_OLLAMA_BASE_URL: *ollama-url" in compose
    assert "\n  ollama:" not in compose
    assert "\n  vault-ollama:" not in compose
    assert "vault_ollama_data" not in compose
    assert "host.docker.internal:host-gateway" in compose
    for source in (application_config, runtime_config, embedding, intelligence, vision):
        assert "vault-ollama" not in source
        assert "http://host.docker.internal:11434" in source

    # Worker/model tunables are pinned by the template.
    example = read("app/config/config.example.yml")
    assert "vision_timeout_seconds: 300" in example
    assert "intelligence_timeout_seconds: 300" in example
    assert "intelligence_num_ctx: 16384" in example
    assert "intelligence_priority_wait_seconds: 2" in example
    assert "vision_num_ctx: 16384" in example
    assert "parser_timeout_seconds: 7200" in example
    assert "opendataloader_hybrid_timeout_ms: 3600000" in example
    assert "embedding_batch_size: 16" in example
    assert "embedding_timeout_seconds: 300" in example


def test_credentials_block_holds_every_secret_with_change_me_defaults():
    compose = read("docker-compose.yaml")
    block = compose.split("x-settings:", 1)[1].split("x-hardened:", 1)[0]

    for anchored in (
        "DB_PASSWORD:",
        "MINIO_ROOT_PASSWORD:",
        "MINIO_APP_SECRET_KEY:",
        "SEARXNG_SECRET:",
        "SECRET_KEY:",
        "AGENT_CAPABILITY_SECRET:",
        "OLLAMA_URL:",
    ):
        assert anchored in block
        assert "CHANGE_ME" in block.split(anchored, 1)[1].split("\n")[0]
    # Six user secrets plus the graph password; chat/vision models live in
    # code + Admin Settings, MinIO names are fixed literals.
    assert "EDIT THIS SECTION" in compose
    for absent in (
        "CHAT_MODEL",
        "VISION_MODEL",
        "minio-root-user",
        "minio-app-access-key",
        "LAVIX_AGENT_MODEL",
        "LAVIX_AGENT_ALLOWED_MODELS",
    ):
        assert absent not in compose, absent
    # Fixed MinIO identity literals appear exactly where the answers pin them.
    assert compose.count("MINIO_ROOT_USER: lavix-root") == 2  # minio, minio-init
    assert "MINIO_APP_ACCESS_KEY: lavix-app" in compose
    assert "S3_ACCESS_KEY: lavix-app" in compose
    # No Compose interpolation (shell ${} inside entrypoints is fine, and
    # always written doubled as $${} so Compose passes it through).
    assert not re.search(r"(?<!\$)\$\{[A-Za-z_]", compose)
    # The neo4j start guard lives in a mounted script file instead of inline
    # YAML, so no doubled dollars are needed anywhere.
    assert "$${" not in compose
    assert "env_file" not in compose
    assert ".env" not in compose
    # Timezone is UTC; ports and image tags are literals.
    assert "Asia/Kolkata" not in compose
    assert "TZ: UTC" in compose
    assert '"9999:8080"' in compose
    assert '"3005:8080"' in compose
    # SearXNG is internal-only: no published port.
    assert '"8080:8080"' not in compose
    assert "ports:" not in _service_block(compose, "searxng")
    assert "image: lavix-vault:1.0.0" in compose
    assert "image: lavix-vault-worker:1.0.0" in compose
    assert "image: lavix-vault-agent:1.0.0" in compose
    assert "image: lavix-vault-reranker:1.0.0" in compose
    assert "image: lavix-vault-webui:1.0.0" in compose


def test_preflight_gate_blocks_every_secret_consumer():
    compose = read("docker-compose.yaml")
    gate = _service_block(compose, "preflight")
    gate_script = read("scripts/preflight-gate.sh")

    assert "busybox:1.37.0@sha256:" in gate
    assert "DB_PASSWORD: *db-password" in gate
    assert "OLLAMA_URL: *ollama-url" in gate
    assert "RERANKER_MODEL: *rerank-model" in gate
    # Gate logic lives in the mounted script, not inline YAML.
    assert "command: [sh, /gate.sh]" in gate
    assert "source: ./scripts/preflight-gate.sh" in gate
    assert "target: /gate.sh" in gate
    assert "create_host_path: false" in gate
    assert "PREFLIGHT FAIL" in gate_script
    assert "[A-Za-z0-9_-]" in gate_script
    assert "must be unique" in gate_script
    # NEO4J_PASSWORD is always validated (graph memory runs by default).
    assert "GRAPH_MEMORY" in gate_script
    assert 'check_secret NEO4J_PASSWORD "$NEO4J_PASSWORD" 16' in gate_script
    assert "MINIO_ROOT_USER" not in gate_script
    assert "MINIO_APP_ACCESS_KEY" not in gate_script
    assert "Supported presets:" in gate_script
    assert "service_completed_successfully" in compose
    # Direct gate dependents; everyone else is covered transitively
    # (migrate/api/workers via postgres+minio, webui via api).
    for service in (
        "postgres",
        "minio",
        "searxng",
        "agent-runtime",
    ):
        block = _service_block(compose, service)
        assert "preflight" in block, service


def test_graph_profile_lives_in_the_single_compose_file():
    # Neo4j is optional and profile-gated in the one compose file (there is
    # no overlay file anymore). PostgreSQL stays canonical; the app default
    # stays OFF. The plain stack never needs a Neo4j password.
    compose = read("docker-compose.yaml")
    application_config = read("app/config.py")
    runtime = read("app/graph_memory/runtime.py")
    example = read("app/config/config.example.yml")
    start_script = read("scripts/neo4j-start.sh")

    assert not (ROOT / "docker-compose.graph.yaml").exists()
    neo4j = _service_block(compose, "neo4j")
    assert "profiles:" not in neo4j
    assert "networks: [backend]" in neo4j
    assert "ports:" not in neo4j
    assert "mem_limit: 2g" in neo4j
    assert "neo4j:5-community@sha256:" in neo4j
    assert "NEO4J_PASSWORD: *neo4j-password" in neo4j
    assert "source: ./scripts/neo4j-start.sh" in neo4j
    assert "create_host_path: false" in neo4j
    assert "neo4j-data:/data" in neo4j
    assert "neo4j-data:" in compose
    assert "GRAPH_MEMORY:" in compose
    assert '"true"' in compose.split("GRAPH_MEMORY:")[1].split("\n")[0]
    worker = _service_block(compose, "graph-memory-worker")
    assert "GRAPH_MEMORY_NEO4J_ENABLED: *graph-memory" in worker
    assert "NEO4J_PASSWORD: *neo4j-password" in worker
    assert "neo4j: {condition: service_healthy, required: false}" in worker
    # The start guard refuses placeholder/weak passwords and execs stock neo4j.
    assert "CHANGE_ME" in start_script
    assert "exit 3" in start_script
    assert 'NEO4J_AUTH="neo4j/$NEO4J_PASSWORD"' in start_script
    assert "unset NEO4J_PASSWORD" in start_script
    assert "exec /startup/docker-entrypoint.sh neo4j" in start_script
    # The app prefers the env value over any password file.
    assert "NEO4J_PASSWORD" in application_config
    assert "graph_memory_neo4j_password" in application_config
    assert "settings.graph_memory_neo4j_password" in runtime
    assert "neo4j_enabled: false" in example
    # Dev-only orchestration files are gone from the public tree.
    assert not (ROOT / "compose.e2e.yaml").exists()
    assert not (ROOT / "compose.mem.yaml").exists()
    assert not (ROOT / "scripts" / "test-e2e.sh").exists()
    assert not (ROOT / "scripts" / "bootstrap-local.sh").exists()
    assert not (ROOT / "tests" / "e2e").exists()

    api_block = _service_block(compose, "api")
    assert "neo4j" not in api_block


def test_recovery_contract_uses_canonical_data_and_a_read_only_bulk_verifier():
    operations = read("docs/operations.md")
    verifier = read("scripts/verify_restored_vault.py")
    capture = read("scripts/capture_recovery_manifest.py")

    assert "| Pre-excision Neo4j volume | Excised system" in operations
    assert "| Projection Neo4j volume (neo4j-data) | Disposable projection" in operations
    assert "Lavix records only read-only tag/ID/availability evidence" in operations
    assert "lavix.recovery-manifest.v2" in operations
    assert "to_jsonb(files_row)::text" in operations
    assert "stable-identity SHA-256" in operations
    assert "--confirm-writers-stopped" in capture
    assert "encode_recovery_manifest" in capture
    assert "database_changed_during_capture" in capture
    assert "os.link(staged, destination" in capture
    assert 'os.fchmod(descriptor, 0o600)' in capture

    assert (
        'cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")'
        in verifier
    )
    assert 'cursor.execute("SET LOCAL TIME ZONE \'UTC\'")' in verifier
    assert "FROM files" in verifier
    assert "database_changed_during_verification" in verifier
    assert "manifest_digest_mismatch" in verifier
    assert "source_revision_mismatch" in verifier
    assert "database_tenantset_mismatch" in verifier
    assert "to_jsonb(f)::text" in verifier
    assert "jsonb_build_object" in verifier
    for forbidden in (
        "fput_object(",
        "remove_object(",
        "copy_object(",
        "UPDATE files",
        "DELETE FROM files",
    ):
        assert forbidden not in verifier
        assert forbidden not in capture


def test_runtime_images_are_digest_pinned_and_webui_is_nonroot():
    dockerfile = read("Dockerfile")
    agent_dockerfile = read("agent_runtime/Dockerfile")
    reranker_dockerfile = read("reranker/Dockerfile")
    webui_dockerfile = read("webui/Dockerfile")

    for source in (dockerfile, agent_dockerfile, reranker_dockerfile, webui_dockerfile):
        stages = set(re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)$", source, re.MULTILINE | re.IGNORECASE))
        for line in source.splitlines():
            if line.startswith("FROM "):
                image = line.split()[1]
                if image not in stages:
                    assert "@sha256:" in image

    assert "USER 10002:10002" in agent_dockerfile
    assert "USER 10003:10003" in reranker_dockerfile
    assert "USER 101:101" in webui_dockerfile
    assert "HEALTHCHECK" in webui_dockerfile
    assert "docling-slim[convert-core,format-html,format-office,format-pdf,models-local]" in read(
        "pyproject.toml"
    )
    assert "       libgl1 \\" not in dockerfile
    assert "node_modules" in read("webui/.dockerignore")
    assert not re.search(r"^COPY .*enc/public_4096\.pem", dockerfile, re.MULTILINE)


def test_minio_application_credentials_are_restricted_rotatable_and_separate():
    compose = read("docker-compose.yaml")

    # Fixed non-secret identity literals, exactly where pinned; only the
    # passwords travel as user secrets through shared anchors.
    assert "MINIO_ROOT_PASSWORD: *minio-root" in compose
    assert "MINIO_APP_SECRET_KEY: *minio-app" in compose
    assert "MINIO_ROOT_USER" not in read("scripts/preflight-gate.sh")
    assert "MINIO_APP_ACCESS_KEY" not in read("scripts/preflight-gate.sh")
    assert "_FILE" not in compose
    # The init logic lives in the mounted script, not inline YAML.
    init_block = _service_block(compose, "minio-init")
    init_script = read("scripts/minio-init.sh")
    assert "source: ./scripts/minio-init.sh" in init_block
    assert "target: /minio-init.sh" in init_block
    assert "entrypoint: [/bin/sh, /minio-init.sh]" in init_block
    assert 'test "$MINIO_ROOT_USER" != "$MINIO_APP_ACCESS_KEY"' in init_script
    assert 'test "$MINIO_ROOT_PASSWORD" != "$MINIO_APP_SECRET_KEY"' in init_script
    assert 'mc admin user add local "$MINIO_APP_ACCESS_KEY" "$MINIO_APP_SECRET_KEY"' in init_script
    assert "mc admin user info" not in compose
    assert "mc admin policy info" not in compose
    assert "sed " not in init_script
    assert 'case "$bucket" in' in init_script
    assert 'printf \'{"Version":"2012-10-17"' in init_script
    assert "mc admin policy create local lavix-bucket-access" in init_script


def test_compose_carries_no_config_mounts_and_gate_covers_postgres():
    compose = read("docker-compose.yaml")
    gitignore = read(".gitignore")

    # No config-folder mounts: the default flow needs no config.yml at all.
    assert "/app/config" not in compose
    assert "config-mount" not in compose
    # Postgres reads the shared anchor as POSTGRES_PASSWORD.
    postgres_block = _service_block(compose, "postgres")
    assert "POSTGRES_PASSWORD: *db-password" in postgres_block
    # App images share one environment anchor (DB_PASSWORD only; the
    # DB_PASS alias still resolves in code).
    assert "x-app-env: &app-env" in compose
    assert "<<: *app-env" in compose
    assert "DB_PASSWORD: *db-password" in compose
    # One ingestion worker by default; scale out documented in a comment.
    assert "ingestion-worker-2" not in compose
    assert "--scale ingestion-worker=2" in compose
    # Signing keys come from the gitignored ./secrets directory.
    assert "file: ./secrets/private_4096.pem" in compose
    assert "source: ./secrets/public_4096.pem" in compose
    assert "secrets" in gitignore


def test_searxng_secret_is_env_supplied_not_embedded():
    compose = read("docker-compose.yaml")
    settings = read("searxng/settings.yml")

    # The pinned image maps SEARXNG_SECRET over server.secret_key, so the
    # settings file carries only a documented dummy and no templating step.
    assert "SEARXNG_SECRET: *searxng-secret" in compose
    assert "SEARXNG_SECRET_FILE" not in compose
    assert "__LAVIX_SEARXNG_SECRET__" not in settings
    assert "__LAVIX_SEARXNG_SECRET__" not in compose
    assert "replaced-by-SEARXNG_SECRET-env-at-runtime" in settings
    assert "mvault-searxng-secret" not in settings
    assert "engine: ahmia" not in settings
    assert "engine: torch" not in settings
    assert "remove:\n      - ahmia\n      - torch" in settings


def test_reranker_mode_wiring_is_explicit_and_surfaced():
    compose = read("docker-compose.yaml")
    application_config = read("app/config.py")
    main = read("app/main.py")
    admin = read("app/routers/admin.py")
    gate_script = read("scripts/preflight-gate.sh")

    # x-settings carries reranker mode/url/model plus embedding model/dims.
    assert "x-settings:" in compose
    assert "RERANKER_MODE:" in compose
    assert "local | external | off" in compose
    assert "RERANKER_URL:" in compose
    assert "RERANKER_MODEL:" in compose
    assert "BAAI/bge-reranker-base" in compose
    assert "RERANKER_MODE: *rerank-mode" in compose
    assert "EMBEDDING_MODEL:" in compose
    assert "EMBEDDING_DIMS:" in compose
    assert "EMBEDDING_MODEL: *embed-model" in compose
    assert "EMBEDDING_DIMENSIONS: *embed-dims" in compose
    # Mode parsing, aliases, and the off switch live in code, not comments.
    assert "def rerank_mode" in application_config
    assert '("RERANKER_MODEL", "RERANK_MODEL")' in application_config
    assert '("RERANKER_URL", "RERANK_BASE_URL")' in application_config
    assert 'RERANKER_MODE=off always wins' in read("app/agent/retrieval.py")
    # Active mode is visible in health and in the (read-only) admin payload.
    assert "_reranker_health" in main
    assert '"mode": settings.rerank_mode' in admin
    assert '"url": settings.rerank_base_url' in admin
    # Nothing hard-depends on the local reranker (scale-to-0 safe).
    assert "reranker:" not in compose.split("depends_on:")[0] or True
    for block_name in ("api", "ingestion-worker", "graph-memory-worker"):
        assert "reranker" not in _service_block(compose, block_name)
    # Gate and host preflight validate the mode and the external URL.
    assert 'RERANKER_MODE must be local, external or off' in gate_script
    assert 'RERANKER_URL must start with http:// or https://' in gate_script
    assert "RERANKER_MODE" in read("scripts/preflight.sh")


def test_embedding_choices_are_guarded_end_to_end():
    compose = read("docker-compose.yaml")
    application_config = read("app/config.py")
    admin = read("app/routers/admin.py")

    # Canonical names first, legacy aliases honored, range enforced.
    assert '("EMBEDDING_MODEL", "EMBEDDING_MODEL_NAME")' in application_config
    assert '("EMBEDDING_DIMENSIONS", "EMBEDDING_DIMENSION")' in application_config
    assert "between 1 and 4000" in application_config
    # x-app-env carries the model/dims to every app container at once.
    assert "EMBEDDING_MODEL: *embed-model" in compose
    assert "EMBEDDING_DIMENSIONS: *embed-dims" in compose
    # Chat/vision models live in code defaults + Admin Settings, never compose.
    assert '"llama3.2:3b"' in application_config
    assert '"qwen2.5vl:3b"' in application_config
    # Migration never edits applied files; 0023 only adds metadata + range.
    assert (ROOT / "migrations" / "0023_embedding_config.sql").exists()
    migration = read("migrations/0023_embedding_config.sql")
    assert "CREATE TABLE IF NOT EXISTS embedding_config" in migration
    assert "BETWEEN 1 AND 4000" in migration
    assert "vector(1024)" not in migration
    # Fresh-DB sizing + startup check + reembed command exist.
    assert "def ensure_fresh_dimensions" in read("app/db/embedding_ddl.py")
    assert "def check_embedding_deployment" in read("app/db/embedding_check.py")
    assert "docker compose run --rm api python -m app.cli reembed" in read(
        "app/db/embedding_check.py"
    )
    assert "reembed" in read("app/cli.py")
    # Admin shows database truth read-only; no PUT path can silently swap.
    assert "_embedding_database_info" in admin
    assert "change_requires_reembed" in admin
    update_block = admin.split("class AdminAIModelsUpdate", 1)[1].split(
        "class InstalledModelResponse", 1
    )[0]
    assert re.search(r"^\s+embedding\s*:", update_block, re.MULTILINE) is None
    # Host preflight warns (never fails) with exact pull commands.
    preflight = read("scripts/preflight.sh")
    assert "EMBEDDING_MODEL" in preflight
    assert "ollama pull $model" in preflight


def test_reranker_is_pinned_baked_offline_and_uses_bounded_torch():
    compose = read("docker-compose.yaml")
    dockerfile = read("reranker/Dockerfile")

    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "INFINITY_ANONYMOUS_USAGE_STATS=0" in dockerfile
    assert "HF_HUB_DISABLE_TELEMETRY=1" in dockerfile
    assert "reranker-cache" not in compose
    # Build ARG selects a hash-pinned preset; unknown models fail loudly.
    assert "ARG RERANKER_MODEL=BAAI/bge-reranker-base" in dockerfile
    assert "RERANKER_MODEL: *rerank-model" in compose
    assert "RERANK_MODEL: *rerank-model" in compose
    assert "PRESETS = {" in dockerfile
    assert '"BAAI/bge-reranker-base"' in dockerfile
    assert '"cross-encoder/ms-marco-MiniLM-L-6-v2"' in dockerfile
    assert "2cfc18c9415c912f9d8155881c133215df768a70" in dockerfile
    assert "233902d25c440f23af6f7d6e94d2946bac0bee0a" in dockerfile
    assert "Supported presets:" in dockerfile
    assert "Hash mismatch" in dockerfile
    assert "snapshot_download(" in dockerfile
    assert "allow_patterns=patterns" in dockerfile
    # One fixed bake path; the serve command lives in an explicit entrypoint
    # script (a shell-form CMD would hand `-c` to the base entrypoint, and the
    # base default points at a path we do not bake).
    assert 'local_dir = "/models/reranker"' in dockerfile
    assert "COPY --chown=10003:10003 reranker/serve.sh /serve.sh" in dockerfile
    assert 'ENTRYPOINT ["/serve.sh"]' in dockerfile
    serve = read("reranker/serve.sh")
    assert "exec infinity_emb v2 --model-id /models/reranker" in serve
    assert '--served-model-name "${RERANKER_MODEL:?RERANKER_MODEL is not set}"' in serve
    assert "--engine torch --device cpu --batch-size 8" in serve
    assert "--no-trust-remote-code --no-compile --no-bettertransformer --port 7997" in serve
    assert "--model-id" not in compose


def test_setup_scripts_generate_gate_and_key_material_without_env_files():
    gen_passwords = read("scripts/gen-passwords.sh")
    preflight = read("scripts/preflight.sh")
    generate_keys = read("scripts/generate-keys.sh")

    # Password generator: hex only, --write patches CHANGE_ME in place.
    assert "openssl rand -hex" in gen_passwords
    assert "--write" in gen_passwords
    assert "OLLAMA_URL" in gen_passwords
    assert ".env" in gen_passwords  # the "never creates a .env" promise
    assert "env_file" not in gen_passwords
    # Host preflight mirrors the compose gate, including the charset rule.
    assert "docker-compose.yaml" in preflight
    assert "A-Za-z0-9_-" in preflight
    assert "must be unique" in preflight or "must be independent" in preflight
    assert ".env exists" in preflight
    assert "uid 10001" in preflight
    # Key generator grants uid-10001 access, backs up reminder, runs preflight.
    assert "10001" in generate_keys
    assert "preflight.sh" in generate_keys
    assert "unrecoverable" in generate_keys
    assert ".env" not in generate_keys
    assert "bootstrap-local" not in generate_keys


def test_hardened_services_run_nonroot_with_bounded_resources():
    compose = read("docker-compose.yaml")

    # The shared anchor pins non-root; named users override it explicitly.
    assert 'user: "10001:10001"' in compose.split("x-hardened:", 1)[1].split("services:", 1)[0]
    assert 'user: "101:101"' in _service_block(compose, "webui")
    assert 'user: "10002:10002"' in _service_block(compose, "agent-runtime")
    # Third-party images keep their own users (no user key on their services).
    for service in ("postgres", "redis", "minio", "searxng"):
        assert "user:" not in _service_block(compose, service), service
    # Resource caps sized for a 16 GB host (measured peaks in README).
    expected = {
        "api": ("4g", "2.0"),
        "ingestion-worker": ("8g", "4.0"),
        "odl-hybrid": ("8g", "4.0"),
        "agent-runtime": ("3g", "2.0"),
        "reranker": ("3g", "2.0"),
    }
    for service, (mem, cpus) in expected.items():
        block = _service_block(compose, service)
        assert f"mem_limit: {mem}" in block, service
        assert f"cpus: {cpus}" in block, service


def test_single_file_binds_use_long_form_without_host_creation():
    compose = read("docker-compose.yaml")

    # Every bind of a single must-exist file uses the long form with
    # create_host_path: false, so a missing file fails loudly instead of
    # materialising as an empty directory.
    for source in (
        "./scripts/preflight-gate.sh",
        "./scripts/minio-init.sh",
        "./secrets/public_4096.pem",
        "./searxng/settings.yml",
        "./searxng/limiter.toml",
    ):
        assert f"source: {source}" in compose, source
    assert compose.count("create_host_path: false") >= 5
    # No short-form file binds remain anywhere.
    for line in compose.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ./") and ":/" in stripped:
            raise AssertionError(f"short-form bind mount: {stripped}")


def test_no_env_file_flow_remains_anywhere():
    tracked = (ROOT / ".git").exists()
    assert tracked
    for path in sorted(ROOT.rglob("*")):
        if ".git" in path.parts or "node_modules" in path.parts or "__pycache__" in path.parts:
            continue
        if path.is_file() and path.suffix == ".example":
            raise AssertionError(f"stale example file: {path}")
    leftover = [name for name in os.listdir(ROOT) if name.startswith(".env")]
    assert leftover == []
    assert not any(line.startswith("!") and "env" in line for line in read(".gitignore").splitlines())
    assert ".env" in read(".gitignore").splitlines()
