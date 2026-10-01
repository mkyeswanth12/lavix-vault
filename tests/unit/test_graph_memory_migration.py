from __future__ import annotations

from app.db.migrations import discover_migrations


def test_graph_memory_migration_is_canonical_tenant_scoped_and_expiring() -> None:
    migration = next(item for item in discover_migrations() if item.version == 9)
    assert migration.name == "graph_memory"
    sql = " ".join(migration.sql.lower().split())

    for table in (
        "graph_memory_tenants",
        "graph_memory_items",
        "graph_memory_jobs",
        "graph_memory_profiles",
    ):
        assert f"create table if not exists {table}" in sql

    assert "enabled boolean not null default false" in sql
    assert "retention_days smallint not null default 90" in sql
    assert "retention_days in (30, 90, 365)" in sql
    assert "generation bigint not null default 0" in sql
    assert "purge_state in ('ready', 'pending', 'failed')" in sql
    assert sql.count("foreign key (user_id, tenant_uuid)") == 3
    assert sql.count("references graph_memory_tenants(user_id, tenant_uuid)") == 3

    assert "source_message_id uuid references chat_messages(id) on delete set null" in sql
    assert "last_confirmed_at timestamptz not null default now()" in sql
    assert "expires_at timestamptz not null" in sql
    assert "normalized_fingerprint char(64) not null" in sql
    assert "where status in ('pending', 'active')" in sql
    assert "backing_memory_ids uuid[] not null" in sql
    assert "job_type in ('extract', 'project', 'expire', 'summarize', 'purge')" in sql


def test_graph_memory_migration_has_no_legacy_graph_or_dynamic_schema_objects() -> None:
    migration = next(item for item in discover_migrations() if item.version == 9)
    sql = migration.sql.casefold()
    assert "legacy" not in sql
    assert "apoc" not in sql
    assert "cypher" not in sql
    assert "lavix-vault_vault_neo4j_data" not in sql


def test_memory_consent_reconciliation_is_atomic_and_removes_dead_profile_cache() -> None:
    migration = next(item for item in discover_migrations() if item.version == 10)
    assert migration.name == "memory_consent_and_profile_cleanup"
    sql = " ".join(migration.sql.lower().split())

    assert "update graph_memory_tenants as tenant set enabled = false" in sql
    assert "users.memory_enabled is distinct from tenant.enabled" in sql
    assert "update users set memory_enabled = false" in sql
    assert "insert into graph_memory_tenants" in sql
    assert "users.memory_enabled" in sql
    assert "on conflict (user_id) do nothing" in sql
    assert "compatibility mirror of graph_memory_tenants.enabled" in sql
    assert "delete from graph_memory_jobs where job_type = 'summarize'" in sql
    assert "job_type in ('extract', 'project', 'expire', 'purge')" in sql
    assert "drop table if exists graph_memory_profiles" in sql
    assert "delete from graph_memory_items" not in sql
    assert "delete from user_memories" not in sql
    assert "drop table if exists graph_memory_items" not in sql
