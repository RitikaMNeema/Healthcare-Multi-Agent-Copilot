import os

import pytest

# Force the deterministic mock LLM backend for the whole test session, even if
# a developer happens to have ANTHROPIC_API_KEY exported - tests must never
# make real, billed, non-deterministic API calls.
os.environ["COPILOT_LLM_BACKEND"] = "mock"


@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setenv("COPILOT_CHECKPOINT_DB", str(tmp_path / "checkpoints.db"))
    yield
