"""Shared test configuration.

Tests use explicit dependency injection and local fakes.  Do not globally replace
application or third-party modules here: doing so makes unrelated tests exercise
MagicMock objects instead of production code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _isolate_deployment_config(monkeypatch, tmp_path):
    """Keep the developer's real app/config/config.yml out of every test.

    Deterministic unit/contract behavior requires the YAML layer to be empty
    unless a test explicitly targets it via LAVIX_CONFIG_FILE.
    """

    from app.config import reload_config

    monkeypatch.setenv("LAVIX_CONFIG_FILE", str(tmp_path / "absent-config.yml"))
    monkeypatch.delenv("APP_ENV", raising=False)
    reload_config()
    yield
    reload_config()


@pytest.fixture(autouse=True)
def _neutralize_saved_admin_preferences(monkeypatch):
    """Report no saved admin preferences unless a test opts in.

    settings.* reads saved preferences through this hook on every access; a
    live lookup would open a database pool (and block) in tests without one.
    Returning (None, None) reproduces the exact pre-override behavior:
    deployment configuration rules. Tests that need saved values override
    this hook explicitly or drive the real reader with a fake database.
    """

    from app.services import model_config

    monkeypatch.setattr(
        model_config, "_read_saved_preferences", lambda: (None, None)
    )
