"""Contract: every job_type the code can enqueue must be DB-legal.

Regression test for the outage where _enqueue_summarize INSERTed
job_type='summarize' while graph_memory_jobs_type_check only allowed
extract/project/expire/purge, rolling back every memory save.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.db.migrations import discover_migrations
from app.graph_memory.repository import JOB_TYPES

REPO_PATH = Path(__file__).resolve().parents[2] / "app" / "graph_memory" / "repository.py"


def _job_type_literals_in_inserts() -> set[str]:
    """job_type string literals used in INSERT statements in repository.py."""
    values: set[str] = set()
    for match in re.finditer(
        r"'(extract|project|expire|purge|summarize)'", REPO_PATH.read_text(encoding="utf-8")
    ):
        # Only count literals on lines that enqueue jobs (INSERT context or
        # the canonical JOB_TYPES set); WHERE-clause readers are irrelevant
        # but harmless to include since they must be legal values too.
        values.add(match.group(1))
    return values


def test_canonical_job_types_cover_every_enqueued_literal() -> None:
    assert {"extract", "project", "expire", "purge", "summarize"} <= JOB_TYPES
    assert _job_type_literals_in_inserts() <= JOB_TYPES


def test_migration_0022_permits_every_canonical_job_type() -> None:
    migration = next(item for item in discover_migrations() if item.version == 22)
    assert migration.name == "graph_memory_summarize_job_type"
    sql = " ".join(migration.sql.lower().split())
    for job_type in sorted(JOB_TYPES):
        assert f"'{job_type}'" in sql, job_type
    # Correction only: the only DDL targets graph_memory_jobs.
    assert sql.count("alter table") == 2
    assert sql.count("graph_memory_jobs") >= 2
    for other in ("graph_memory_items (", "graph_memory_tenants (", "memory_summaries ("):
        assert other not in sql


def test_head_check_constraint_still_matches_history() -> None:
    """0010 narrowed the CHECK; 0022 must be a strict superset of it."""
    by_version = {item.version: item for item in discover_migrations()}
    old = " ".join(by_version[10].sql.lower().split())
    assert "check (job_type in ('extract', 'project', 'expire', 'purge'))" in old
    new = " ".join(by_version[22].sql.lower().split())
    for job_type in ("extract", "project", "expire", "purge", "summarize"):
        assert f"'{job_type}'" in new, job_type
