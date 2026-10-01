"""Credential redaction for storage, answers, logs, and queries.

Mock secrets only — no test in this file may contain a real credential.
"""
from app.security.redact import contains_credentials, redact_credentials


def test_labeled_values_redacted_label_kept():
    assert (
        redact_credentials("my password is hunter2")
        == "my password is [REDACTED]"
    )
    assert (
        redact_credentials("api key: abc123xyz")
        == "api key: [REDACTED]"
    )
    assert redact_credentials("token=xyz789") == "token=[REDACTED]"


def test_valueless_prompts_keep_labels():
    assert (
        redact_credentials("Password for http://example.test:1:")
        == "Password for http://example.test:1:"
    )
    assert redact_credentials("password:") == "password:"


def test_userinfo_url_redacts_pass_keeps_user():
    assert (
        redact_credentials("push to http://operator-1:s3cret@example.test/x")
        == "push to http://operator-1:[REDACTED]@example.test/x"
    )


def test_bearer_and_jwt_redacted():
    assert (
        redact_credentials("auth Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig")
        == "auth Bearer [REDACTED]"
    )


def test_private_key_block_redacted():
    text = "key below\n-----BEGIN PRIVATE KEY-----\nABCDEF\n-----END PRIVATE KEY-----\ndone"
    out = redact_credentials(text)
    assert "ABCDEF" not in out
    assert "[REDACTED]" in out
    assert out.startswith("key below")
    assert out.endswith("done")


def test_username_alone_and_prose_untouched():
    assert redact_credentials("operator-1 pushed the branch") == "operator-1 pushed the branch"
    assert (
        redact_credentials("the secret to good dosa is patience")
        == "the secret to good dosa is patience"
    )
    assert redact_credentials("tokenize the output") == "tokenize the output"


def test_contains_credentials_predicate():
    assert contains_credentials("my password is hunter2") is True
    assert contains_credentials("nothing sensitive here") is False
    assert contains_credentials("Password for http://example.test:1:") is True


class TestCredentialStreamScrubber:
    def _scrubber(self):
        from app.security.redact import CredentialStreamScrubber

        return CredentialStreamScrubber()

    def test_clean_deltas_pass_through_immediately(self):
        scrubber = self._scrubber()
        assert scrubber.feed("Hello ") == "Hello "
        assert scrubber.feed("world") == "world"
        assert scrubber.finalize() == ""

    def test_split_secret_never_partially_emitted(self):
        scrubber = self._scrubber()
        out = ""
        out += scrubber.feed("my pass")
        out += scrubber.feed("word is hun")
        assert "hun" not in out
        out += scrubber.feed("ter2 end")
        out += scrubber.finalize()
        assert out == "my password is [REDACTED] end"

    def test_stream_equals_final_exactly(self):
        from app.security.redact import redact_credentials

        full = "user operator-1 token abc123 done"
        scrubber = self._scrubber()
        streamed = "".join(scrubber.feed(full[i : i + 3]) for i in range(0, len(full), 3))
        streamed += scrubber.finalize()
        assert streamed == redact_credentials(full)


def test_chat_content_redaction_assistant_only():
    from app.routers.chat.orchestrator import redact_chat_content

    assert (
        redact_chat_content("assistant", 'the password is hunter2 ok')
        == "the password is [REDACTED] ok"
    )
    assert (
        redact_chat_content("user", "my password is hunter2")
        == "my password is hunter2"
    )


def test_history_assembly_redacts_assistant_rows():
    from app.routers.chat.orchestrator import _bounded_recent_history

    rows = [
        {"role": "user", "content": "my password is hunter2"},
        {"role": "assistant", "content": "the password is hunter2 ok"},
    ]
    bounded = _bounded_recent_history(rows, max_chars=10000, max_messages=10)
    by_role = {row["role"]: row["content"] for row in bounded}
    assert "[REDACTED]" in by_role["assistant"]
    assert "hunter2" not in by_role["assistant"]
    assert by_role["user"] == "my password is hunter2"


def test_vision_document_redaction():
    from app.ingestion.chunking import CanonicalDocument, CanonicalElement, ElementType
    from app.ingestion.worker import redact_document_text
    document = CanonicalDocument(
        source_name="shot.png",
        source_sha256="0" * 64,
        media_type="image/png",
        parser_fingerprint="vision",
        elements=(
            CanonicalElement(
                element_id="e1",
                element_type=ElementType.TEXT,
                text="Password for http://example.test:1: hunter2 end",
                provenance=(),
                heading_level=None,
            ),
        ),
        metadata={},
    )
    cleaned = redact_document_text(document)
    assert "hunter2" not in cleaned.elements[0].text
    assert "Password for" in cleaned.elements[0].text
    assert cleaned.source_sha256 == "0" * 64


def test_final_post_check_refuses_on_leak():
    from app.routers.chat.orchestrator import _safe_public_final

    refused, replacement = _safe_public_final('the password is hunter2 ok')
    assert refused is True
    assert "hunter2" not in replacement

    refused, text = _safe_public_final("Paris is the capital of France.")
    assert refused is False
    assert text == "Paris is the capital of France."


def test_log_filter_redacts_bodies():
    import logging

    from app.security.redact import RedactingFilter

    records = []

    class Sink(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("test-redact-filter-probe")
    logger.addFilter(RedactingFilter())
    logger.addHandler(Sink())
    logger.setLevel(logging.INFO)
    logger.warning("query=%s", "my password is hunter2")
    logger.warning("plain line with number %d", 42)
    assert "hunter2" not in records[0]
    assert records[1] == "plain line with number 42"


def test_agent_log_filter_matches_app_filter_on_secrets():
    from agent_runtime.cuga_adapter import _RedactingLogFilter

    vectors = [
        "query=my password is hunter2 ok",
        "push to http://operator-1:s3cret@example.test/x",
        "auth Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig end",
        "-----BEGIN PRIVATE KEY-----\nABCDEF\n-----END PRIVATE KEY-----",
    ]
    assert len(vectors) == 4
    import logging

    class Sink(logging.Handler):
        def __init__(self, store):
            super().__init__()
            self.store = store

        def emit(self, record):
            self.store.append(record.getMessage())

    for text in vectors:
        for factory in (
            __import__("app.security.redact", fromlist=["RedactingFilter"]).RedactingFilter,
            _RedactingLogFilter,
        ):
            records = []
            sink = Sink(records)
            logger = logging.getLogger("test-parity-probe")
            logger.addFilter(factory())
            logger.addHandler(sink)
            logger.setLevel(logging.INFO)
            logger.warning("%s", text)
            logger.removeHandler(sink)
            assert "hunter2" not in records[-1]
            assert "s3cret" not in records[-1]
            assert "ABCDEF" not in records[-1]
            assert "eyJhbGci" not in records[-1]


def _script_factory(rows):
    writes = []

    class FakeCursor:
        def __init__(self):
            self._rows = []

        def execute(self, sql, params=None):
            normalized = " ".join(str(sql).split())
            if normalized.startswith("SELECT"):
                table = "chat_messages" if "chat_messages" in normalized else (
                    "document_chunks" if "document_chunks" in normalized else "files"
                )
                self._rows = rows.get(table, [])
            else:
                writes.append(normalized.split()[0])
                self._rows = []

        def fetchall(self):
            return self._rows

        def fetchone(self):
            return self._rows[0] if self._rows else None

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return FakeCursor()

        def commit(self):
            writes.append("COMMIT")

    def factory():
        return FakeConnection()

    factory.writes = writes
    return factory


def test_cleanup_dry_run_writes_nothing(capsys):
    import sys

    sys.path.insert(0, "scripts")
    from scrub_credentials import main

    rows = {
        "chat_messages": [{"id": 1, "content": "my password is hunter2"}],
        "document_chunks": [],
        "files": [],
    }
    factory = _script_factory(rows)
    assert main(["--user-id", "2"], connection_factory=factory) == 0
    assert factory.writes == []
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "hunter2" not in out


def test_cleanup_apply_redacts_and_counts_only(capsys):
    import sys

    sys.path.insert(0, "scripts")
    from scrub_credentials import main

    updated = {}

    class FakeCursor:
        def execute(self, sql, params=None):
            normalized = " ".join(str(sql).split())
            if normalized.startswith("SELECT"):
                if "chat_messages" in normalized:
                    self._rows = [{"id": 1, "content": "my password is hunter2"}]
                elif "document_chunks" in normalized:
                    self._rows = [{"chunk_id": "c1", "content": "token abc123 end"}]
                else:
                    self._rows = [{"id": 9, "quick_summary": "API key xyz789 here"}]
            else:
                updated[normalized.split()[0]] = params

        def fetchall(self):
            return getattr(self, "_rows", [])

        def fetchone(self):
            rows = getattr(self, "_rows", [])
            return rows[0] if rows else None

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return FakeCursor()

        def commit(self):
            updated["COMMIT"] = True

    def factory():
        return FakeConnection()

    assert main(["--user-id", "2", "--apply"], connection_factory=factory) == 0
    out = capsys.readouterr().out
    assert "hunter2" not in out and "abc123" not in out and "xyz" not in out
    assert "chat_messages=1 document_chunks=1 files=1" in out
    for params in updated.values():
        if isinstance(params, tuple):
            assert not any(
                secret in str(value)
                for value in params
                for secret in ("hunter2", "abc123")
            )


def test_cleanup_all_apply_requires_double_confirmation(capsys):
    import sys

    sys.path.insert(0, "scripts")
    from scrub_credentials import main

    def factory():
        raise AssertionError("must not connect without confirmation")

    import pytest

    with pytest.raises(SystemExit):
        main(["--all", "--apply"], connection_factory=factory)
    assert main(
        ["--all", "--apply", "--yes-really"], connection_factory=_script_factory({})
    ) == 0
    assert main(
        ["--all", "--apply", "--confirm-phrase", "SCRUB SECRETS"],
        connection_factory=_script_factory({}),
    ) == 0
    with pytest.raises(SystemExit):
        main([], connection_factory=_script_factory({}))


def test_refusal_wins_contract_stream_and_persist():
    # FIX 6 contract: refusal replaces the draft wholesale. Streamed prefix
    # (if any) is always credential-free; persisted row holds the refusal.
    from agent_runtime.events import AnswerDeltaProjector
    from app.routers.chat.orchestrator import _safe_public_final
    from app.security.redact import CredentialStreamScrubber

    draft = "The test API key is MOCK-KEY-0000-AAAA for staging."
    scrubber = CredentialStreamScrubber()
    projector = AnswerDeltaProjector()
    streamed = ""
    # Realistic tokenizer deltas (leading spaces, as the model emits them).
    deltas = ["The", " test", " API", " key", " is", " MOCK-KEY-0000-AAAA", " for", " staging."]
    for delta in deltas:
        cleaned = scrubber.feed(delta)
        public = projector.feed(cleaned) if cleaned else ""
        streamed += public
    streamed += projector.feed(scrubber.finalize(), final=True)
    assert "MOCK-KEY-0000-AAAA" not in streamed
    refused, replacement = _safe_public_final(draft)
    assert refused is True
    assert "MOCK-KEY-0000-AAAA" not in replacement
    assert "can't share" in replacement
