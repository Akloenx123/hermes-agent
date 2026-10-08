"""A session that fails in a background thread leaves its traceback in the log.

The deferred agent build and the cold-resume hydration both run off the JSON-RPC path and hand the
client only ``str(exc)``. Neither logged the exception, so a Desktop ``agent_init_failed`` with
"maximum recursion depth exceeded" (#134890) left no frame in errors.log, agent.log or gui.log to
say where the recursion was.
"""

import logging
import sqlite3
import threading

from agent.auxiliary_unavailable import ProviderNotConfiguredError
from tui_gateway import server


def _error_records(caplog):
    return [r for r in caplog.records if r.name == server.logger.name and r.levelno >= logging.ERROR]


def _fail_build(monkeypatch, exc):
    """Run the deferred agent build against a registered session, failing at ``_make_agent``."""

    def _raise(*_a, **_kw):
        raise exc

    monkeypatch.setattr(server, "_emit", lambda *a, **kw: True)
    monkeypatch.setattr(server, "_make_agent", _raise)
    monkeypatch.setattr(server, "_deferred_build_agent_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr(server, "_await_resume_history", lambda *a, **kw: True)
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_clear_session_context", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_bind_build_profile_scopes", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_finish_agent_build", lambda *a, **kw: None)
    session = {"session_key": "key", "agent_ready": threading.Event()}
    monkeypatch.setitem(server._sessions, "sid-build", session)
    server._start_agent_build("sid-build", session)
    session["_agent_build_thread"].join(timeout=30)
    return session


def test_a_failed_agent_build_logs_its_traceback(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=server.logger.name)

    session = _fail_build(monkeypatch, RecursionError("maximum recursion depth exceeded"))

    assert session["agent_error"] == "maximum recursion depth exceeded"
    [record] = _error_records(caplog)
    assert "sid-build" in record.getMessage()
    assert record.exc_info and record.exc_info[0] is RecursionError


def test_a_missing_provider_is_logged_without_a_stack(monkeypatch, caplog):
    """No provider is a setup state the client routes to onboarding, not a code fault."""
    caplog.set_level(logging.ERROR, logger=server.logger.name)

    _fail_build(monkeypatch, ProviderNotConfiguredError("No LLM provider configured."))

    [record] = _error_records(caplog)
    assert "No LLM provider configured" in record.getMessage()
    assert not record.exc_info


def test_a_failed_resume_hydration_logs_its_traceback(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=server.logger.name)
    session = {"history": [], "history_lock": threading.RLock(), "resume_hydrating": True,
               "resume_history_ready": threading.Event(), "agent_ready": threading.Event()}
    monkeypatch.setitem(server._sessions, "sid-resume", session)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: True)

    def _malformed(*_a, **_kw):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(server, "_load_resume_transcript", _malformed)

    server._schedule_resume_hydration("sid-resume", "stored-id", object())

    assert session["agent_ready"].wait(5)
    assert "malformed" in session["resume_history_error"]
    [record] = _error_records(caplog)
    assert "sid-resume" in record.getMessage() and "stored-id" in record.getMessage()
    assert record.exc_info and record.exc_info[0] is sqlite3.DatabaseError
