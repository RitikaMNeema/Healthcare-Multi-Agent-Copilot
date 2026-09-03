import os

from copilot.config import KB_DIR


class FileAccessError(Exception):
    pass


def read_kb_file(filename: str) -> str:
    """Read a file from the knowledge base directory only - basename + prefix
    checks prevent path traversal (`../../etc/passwd`, absolute paths, etc.)."""
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name in (".", ".."):
        raise FileAccessError(f"invalid filename: {filename!r}")

    kb_root = os.path.abspath(KB_DIR)
    path = os.path.abspath(os.path.join(kb_root, safe_name))
    if not path.startswith(kb_root + os.sep):
        raise FileAccessError("path traversal detected")
    if not os.path.isfile(path):
        raise FileAccessError(f"file not found: {safe_name}")

    with open(path, encoding="utf-8") as f:
        return f.read()
