from __future__ import annotations

from pathlib import Path

import pytest

from app.db.migrations import (
    DEFAULT_MIGRATIONS_DIR,
    MigrationChecksumError,
    MigrationDiscoveryError,
    apply_migrations,
    discover_migrations,
)
from app.db.readiness import SchemaNotReadyError, check_schema_head, require_schema_head
from app.ingestion.models import JobState


class FakeConnection:
    def __init__(self, *, table_exists: bool = False, history=None, fail_on: str | None = None):
        self.table_exists = table_exists
        self.history = list(history or [])
        self.fail_on = fail_on
        self.executed_migrations: list[str] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self._snapshot = (table_exists, list(self.history))

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1
        self._snapshot = (self.table_exists, list(self.history))

    def rollback(self):
        self.rollbacks += 1
        self.table_exists, history = self._snapshot
        self.history = list(history)

    def close(self):
        self.closed = True


class FakeCursor:
    def __init__(self, connection: FakeConnection):
        self.connection = connection
        self._one = None
        self._many = []
        self.closed = False

    def execute(self, sql: str, params=None):
        normalized = " ".join(sql.split()).lower()
        if self.connection.fail_on and self.connection.fail_on.lower() in sql.lower():
            raise RuntimeError("synthetic migration failure")

        if "pg_advisory_xact_lock" in normalized:
            self._one = {"pg_advisory_xact_lock": None}
        elif normalized.startswith("create table if not exists public.schema_migrations"):
            self.connection.table_exists = True
        elif normalized.startswith("select to_regclass"):
            self._one = {"table_name": "schema_migrations" if self.connection.table_exists else None}
        elif "from public.schema_migrations" in normalized:
            self._many = [
                dict(row) for row in sorted(self.connection.history, key=lambda row: row["version"])
            ]
        elif normalized.startswith("insert into public.schema_migrations"):
            version, name, checksum = params
            self.connection.history.append(
                {
                    "version": version,
                    "name": name,
                    "checksum": checksum,
                    "applied_at": None,
                }
            )
        else:
            self.connection.executed_migrations.append(sql)
        return self

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._many)

    def close(self):
        self.closed = True


def test_foundation_migration_declares_all_runtime_tables_and_indexes():
    migrations = discover_migrations(DEFAULT_MIGRATIONS_DIR)

    assert [migration.version for migration in migrations] == list(range(1, 24))
    migration = migrations[0]
    assert migration.name == "v2_foundation"
    assert len(migration.checksum) == 64

    sql = migration.sql.lower()
    for table in (
        "users",
        "refresh_tokens",
        "folders",
        "files",
        "chats",
        "decryption_sessions",
        "user_policy_templates",
        "ingestion_jobs",
        "document_revisions",
        "document_chunks",
        "chat_messages",
    ):
        assert f"create table if not exists {table}" in sql

    assert "create extension if not exists vector" in sql
    assert "embedding vector(1024)" in sql
    assert "search_vector tsvector generated always" in sql
    assert "using hnsw" in sql

    for column in (
        "desired_revision integer not null default 0",
        "current_revision integer",
        "ai_status text",
        "ai_error_code text",
        "ai_error_detail text",
    ):
        assert column in sql

    assert "desired_index_revision" not in sql
    assert "current_index_revision" not in sql
    assert "object_uuid" not in sql
    assert "doc_type" not in sql
    assert "memory_items" not in sql
    assert "embeddings_generated" not in sql
    assert "ai_access_granted_at" not in sql
    assert "last_ai_access" not in sql
    assert "model_preferences" not in sql
    assert "clerk_user_id" not in sql
    assert "create table if not exists model_perf" not in sql
    assert "encryption_algorithm" not in sql
    assert "chunk_size" not in sql
    assert "total_chunks" not in sql
    assert "last_accessed" not in sql
    assert "idx_refresh_tokens_token_hash" not in sql
    assert "idx_files_uuid" not in sql
    assert "idx_decryption_sessions_token" not in sql
    assert "where ai_status is null" in sql
    assert "set user_granted_ai_access = false" in sql
    assert "alter column ai_status set not null" in sql
    assert "create or replace function update_user_storage()" in sql
    assert "update of is_deleted, file_size_bytes, user_id" in sql
    assert "embedding_fingerprint text not null" in sql
    assert "primary key (file_id, revision, chunk_id)" in sql
    assert "foreign key (file_id, revision)" in sql
    assert "content text not null" in sql

    for state in JobState:
        assert f"'{state.value}'" in sql

    idle_migration = migrations[1]
    assert idle_migration.name == "session_idle_timeout"
    idle_sql = idle_migration.sql.lower()
    assert "add column if not exists last_activity_at timestamptz" in idle_sql
    assert "when is_revoked = false and expires_at > now() then now()" in idle_sql
    assert "alter column last_activity_at set not null" in idle_sql
    assert "idx_refresh_tokens_active_session_activity" in idle_sql

    intelligence_migration = migrations[2]
    assert intelligence_migration.name == "document_intelligence"
    intelligence_sql = intelligence_migration.sql.lower()
    assert "add column if not exists doc_type text" in intelligence_sql
    assert "add column if not exists intelligence_status text" in intelligence_sql
    assert "idx_files_user_doc_type" in intelligence_sql

    memory_migration = migrations[3]
    assert memory_migration.name == "opt_in_user_memory"
    memory_sql = memory_migration.sql.lower()
    assert "add column if not exists memory_enabled boolean not null default false" in memory_sql
    assert "create table if not exists user_memories" in memory_sql
    assert "references users(id) on delete cascade" in memory_sql
    assert "char_length(memory_text) between 1 and 500" in memory_sql
    assert "unique (user_id, memory_text)" in memory_sql
    assert "idx_user_memories_user_recent" in memory_sql

    book_type_migration = migrations[4]
    assert book_type_migration.name == "book_document_type"
    book_type_sql = book_type_migration.sql.lower()
    assert "drop constraint if exists files_doc_type_check" in book_type_sql
    assert "'agreement', 'book', 'brochure'" in book_type_sql

    purchase_order_migration = migrations[5]
    assert purchase_order_migration.name == "purchase_order_document_type"
    purchase_order_sql = purchase_order_migration.sql.lower()
    assert "drop constraint if exists files_doc_type_check" in purchase_order_sql
    assert "'proposal', 'purchase_order'" in purchase_order_sql

    preference_migration = migrations[6]
    assert preference_migration.name == "preferred_chat_model"
    preference_sql = preference_migration.sql.lower()
    assert "add column if not exists preferred_chat_model varchar(200)" in preference_sql

    model_config_migration = migrations[7]
    assert model_config_migration.name == "system_ai_configuration"
    model_config_sql = model_config_migration.sql.lower()
    assert "create table if not exists system_ai_configuration" in model_config_sql
    assert "singleton_id smallint primary key" in model_config_sql
    assert "check (singleton_id = 1)" in model_config_sql
    assert "revision bigint not null" in model_config_sql
    assert "chat_allowed_models text[]" in model_config_sql
    assert "memory_extraction_model varchar(200)" in model_config_sql
    assert "references users(id) on delete set null" in model_config_sql


def test_apply_migrations_records_checksum_and_is_idempotent():
    connection = FakeConnection()

    first = apply_migrations(connection)
    assert first.applied_versions == tuple(range(1, 24))
    assert first.previous_version == 0
    assert first.current_version == 23
    assert connection.table_exists is True
    assert len(connection.history) == 23
    assert [row["checksum"] for row in connection.history] == [
        migration.checksum for migration in discover_migrations()
    ]
    assert connection.commits == 1
    assert connection.rollbacks == 0

    migration_sql_count = len(connection.executed_migrations)
    second = apply_migrations(connection)
    assert second.applied_versions == ()
    assert second.previous_version == 23
    assert len(connection.executed_migrations) == migration_sql_count
    assert len(connection.history) == 23
    assert connection.commits == 2


def test_apply_rejects_changed_checksum_without_running_sql():
    migration = discover_migrations()[0]
    connection = FakeConnection(
        table_exists=True,
        history=[
            {
                "version": migration.version,
                "name": migration.name,
                "checksum": "0" * 64,
                "applied_at": None,
            }
        ],
    )

    with pytest.raises(MigrationChecksumError):
        apply_migrations(connection)

    assert connection.executed_migrations == []
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_apply_rolls_back_failed_migration(tmp_path: Path):
    migration_file = tmp_path / "0001_broken.sql"
    migration_file.write_text("SELECT synthetic_failure;\n", encoding="utf-8")
    connection = FakeConnection(fail_on="synthetic_failure")

    with pytest.raises(RuntimeError, match="synthetic migration failure"):
        apply_migrations(connection, tmp_path)

    assert connection.table_exists is False
    assert connection.history == []
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_readiness_is_non_mutating_when_history_table_is_missing():
    connection = FakeConnection(table_exists=False)

    status = check_schema_head(connection)

    assert status.ready is False
    assert status.reason == "migration_table_missing"
    assert status.current_version == 0
    assert status.head_version == 23
    assert connection.table_exists is False
    assert connection.commits == 0
    assert connection.rollbacks == 0


def test_readiness_reports_pending_then_ready_after_apply():
    connection = FakeConnection(table_exists=True)

    pending = check_schema_head(connection)
    assert pending.ready is False
    assert pending.reason == "pending_migrations"
    assert pending.pending_versions == tuple(range(1, 24))

    apply_migrations(connection)
    ready = require_schema_head(connection)
    assert ready.ready is True
    assert ready.reason == "ready"
    assert ready.current_version == ready.head_version == 23


def test_require_schema_head_raises_stable_error():
    connection = FakeConnection(table_exists=False)

    with pytest.raises(SchemaNotReadyError) as exc_info:
        require_schema_head(connection)

    assert exc_info.value.status.reason == "migration_table_missing"
    assert "current=0, head=23" in str(exc_info.value)


def test_discovery_rejects_malformed_filename(tmp_path: Path):
    (tmp_path / "migration.sql").write_text("SELECT 1;\n", encoding="utf-8")

    with pytest.raises(MigrationDiscoveryError, match="Invalid migration filename"):
        discover_migrations(tmp_path)


def test_migration_0015_adds_guarded_context_ceiling():
    migrations = {migration.version: migration for migration in discover_migrations(DEFAULT_MIGRATIONS_DIR)}

    assert 15 in migrations
    sql = migrations[15].sql
    assert "model_max_num_ctx" in sql
    assert "16384" in sql and "262144" in sql


def test_migration_0016_adds_nullable_fallback_model():
    migrations = {migration.version: migration for migration in discover_migrations(DEFAULT_MIGRATIONS_DIR)}

    assert 16 in migrations
    sql = migrations[16].sql
    assert "fallback_chat_model" in sql


def test_migration_0017_adds_guarded_rag_retrieval_settings():
    migrations = {migration.version: migration for migration in discover_migrations(DEFAULT_MIGRATIONS_DIR)}

    assert 17 in migrations
    sql = migrations[17].sql
    assert "file_scope" in sql
    assert "top_k" in sql
    assert "DEFAULT 5" in sql
    assert "DEFAULT 20" in sql
    assert "BETWEEN 1 AND 100" in sql
    assert "BETWEEN 1 AND 500" in sql


def test_migration_0019_adds_guarded_search_depth():
    migrations = {migration.version: migration for migration in discover_migrations(DEFAULT_MIGRATIONS_DIR)}

    assert 19 in migrations
    sql = migrations[19].sql
    assert "search_depth" in sql
    assert "ADD COLUMN IF NOT EXISTS search_depth TEXT NOT NULL DEFAULT 'conservative'" in sql
    assert "DROP CONSTRAINT IF EXISTS system_ai_configuration_search_depth_check" in sql
    assert "ADD CONSTRAINT system_ai_configuration_search_depth_check" in sql
    assert "CHECK (search_depth IN ('conservative', 'balanced', 'deep'))" in sql


def test_migration_0020_adds_pro_search_depth_tier():
    migrations = {migration.version: migration for migration in discover_migrations(DEFAULT_MIGRATIONS_DIR)}

    assert 20 in migrations
    sql = migrations[20].sql
    assert "DROP CONSTRAINT IF EXISTS system_ai_configuration_search_depth_check" in sql
    assert "ADD CONSTRAINT system_ai_configuration_search_depth_check" in sql
    assert "CHECK (search_depth IN ('conservative', 'balanced', 'deep', 'pro'))" in sql
