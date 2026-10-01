"""Unit tests for the recovery-script scope predicate (no database)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "recover-graph-jobs.py"
_spec = importlib.util.spec_from_file_location("recover_graph_jobs", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules["recover_graph_jobs"] = _module
_spec.loader.exec_module(_module)

build_sql = _module.build_sql
is_recoverable = _module.is_recoverable


def test_only_failed_constraint_jobs_are_recoverable() -> None:
    assert is_recoverable({"state": "failed", "error_code": "graph_job_failed"}) is True
    assert is_recoverable({"state": "complete", "error_code": "graph_job_failed"}) is False
    assert is_recoverable({"state": "cancelled", "error_code": "graph_job_failed"}) is False
    assert is_recoverable({"state": "failed", "error_code": "worker_lease_expired"}) is False
    assert is_recoverable({"state": "failed", "error_code": None}) is False
    assert is_recoverable({"state": "queued", "error_code": None}) is False


def test_scope_sql_is_user_bound_by_default() -> None:
    list_sql, reset_sql, params = build_sql(41)
    assert "user_id = %s" in list_sql
    assert "user_id = %s" in reset_sql
    assert params == (41,)
    assert "failed" in reset_sql and "graph_job_failed" in reset_sql
    assert "state = 'queued'" in reset_sql
    assert "attempts = 0" in reset_sql


def test_all_scope_has_no_user_filter() -> None:
    _, _, params = build_sql(None)
    assert params == ()
