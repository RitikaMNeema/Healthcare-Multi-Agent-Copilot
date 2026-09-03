import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

from copilot.config import default_audit_db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    ts REAL NOT NULL,
    user_id TEXT,
    role TEXT,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_request ON audit_events(request_id);
"""


class AuditLog:
    """Append-only audit trail of every governance-relevant decision the copilot makes."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or default_audit_db_path()
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log(self, *, request_id: str, event_type: str, payload: dict | None = None,
             user_id: str | None = None, role: str | None = None) -> str:
        event_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_events (id, request_id, ts, user_id, role, event_type, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_id, request_id, time.time(), user_id, role, event_type,
                 json.dumps(payload or {}, default=str)),
            )
        return event_id

    def trail_for(self, request_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, request_id, ts, user_id, role, event_type, payload "
                "FROM audit_events WHERE request_id = ? ORDER BY ts",
                (request_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def recent(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, request_id, ts, user_id, role, event_type, payload "
                "FROM audit_events ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]


def _row_to_dict(row) -> dict:
    keys = ["id", "request_id", "ts", "user_id", "role", "event_type", "payload"]
    record = dict(zip(keys, row))
    record["payload"] = json.loads(record["payload"])
    return record
