import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

PRIMARY_MODEL = os.environ.get("COPILOT_PRIMARY_MODEL", "claude-opus-5")
FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get("COPILOT_FALLBACK_MODELS", "claude-sonnet-5,claude-haiku-4-5").split(",")
    if m.strip()
]

KB_DIR = os.environ.get("COPILOT_KB_DIR", os.path.join(_REPO_ROOT, "data", "knowledge_base"))
CLAIMS_DB_PATH = os.environ.get("COPILOT_CLAIMS_DB", os.path.join(_REPO_ROOT, "data", "claims.db"))
MAX_TOOL_ITERATIONS = int(os.environ.get("COPILOT_MAX_TOOL_ITERATIONS", "4"))


def default_audit_db_path() -> str:
    return os.environ.get("COPILOT_AUDIT_DB", os.path.join(_REPO_ROOT, "data", "audit.db"))


def default_checkpoint_db_path() -> str:
    return os.environ.get("COPILOT_CHECKPOINT_DB", os.path.join(_REPO_ROOT, "data", "checkpoints.db"))


def llm_backend_name() -> str:
    explicit = os.environ.get("COPILOT_LLM_BACKEND")
    if explicit:
        return explicit
    return "claude" if os.environ.get("ANTHROPIC_API_KEY") else "mock"
