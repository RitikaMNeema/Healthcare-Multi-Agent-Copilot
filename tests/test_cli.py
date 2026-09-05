"""The CLI has no resolved identity or reviewer-role checks (it talks to the
graph and its SQLite files directly - see cli.py's module docstring), so its
approval-deciding commands (`approve`, `pending`) are gated behind an
explicit opt-in env var instead. This isn't real authorization - anyone who
can run this CLI already has direct DB access - it just means a container
that never sets the var (see Dockerfile) can't casually run these commands.
"""
import sys

import pytest

from copilot import cli


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["copilot", *argv])
    cli.main()


def test_approve_refuses_without_env_var(monkeypatch):
    monkeypatch.delenv("COPILOT_ALLOW_CLI_APPROVALS", raising=False)
    with pytest.raises(SystemExit, match="Refusing to run"):
        _run_cli(monkeypatch, ["approve", "--request-id", "r1", "--approver", "someone", "--decision", "approve"])


def test_pending_refuses_without_env_var(monkeypatch):
    monkeypatch.delenv("COPILOT_ALLOW_CLI_APPROVALS", raising=False)
    with pytest.raises(SystemExit, match="Refusing to run"):
        _run_cli(monkeypatch, ["pending"])


def test_approve_refuses_with_falsy_env_var(monkeypatch):
    monkeypatch.setenv("COPILOT_ALLOW_CLI_APPROVALS", "0")
    with pytest.raises(SystemExit, match="Refusing to run"):
        _run_cli(monkeypatch, ["pending"])


def test_pending_runs_when_env_var_enabled(monkeypatch, capsys):
    monkeypatch.setenv("COPILOT_ALLOW_CLI_APPROVALS", "1")
    _run_cli(monkeypatch, ["pending"])  # must not raise
    capsys.readouterr()  # nothing pending yet - just confirming it ran


def test_chat_does_not_require_the_env_var(monkeypatch):
    monkeypatch.delenv("COPILOT_ALLOW_CLI_APPROVALS", raising=False)
    _run_cli(monkeypatch, ["chat", "--query", "What is the timely filing deadline for Medicare claims?",
                            "--role", "viewer", "--user", "alice"])


def test_audit_does_not_require_the_env_var(monkeypatch):
    monkeypatch.delenv("COPILOT_ALLOW_CLI_APPROVALS", raising=False)
    _run_cli(monkeypatch, ["audit", "--request-id", "nonexistent"])  # empty trail, but must not refuse to run
