import os
import sqlite3
import time
from contextlib import contextmanager

from copilot.config import default_audit_db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_approvals (
    request_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    summary TEXT NOT NULL,
    risk TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    decided_by TEXT,
    decided_at REAL
);
"""


class ApprovalQueue:
    """Durable, queryable record of human-in-the-loop approval requests.

    This is deliberately independent of the LangGraph checkpointer: the checkpointer
    resumes the *graph*, this table lets a compliance dashboard or another process
    see what's pending without touching graph internals.
    """

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

    def submit(self, request_id: str, *, summary: str, risk: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pending_approvals "
                "(request_id, created_at, summary, risk, status, decided_by, decided_at) "
                "VALUES (?, ?, ?, ?, 'pending', NULL, NULL)",
                (request_id, time.time(), summary, risk),
            )

    def decide(self, request_id: str, *, approver: str | None, approved: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE pending_approvals SET status = ?, decided_by = ?, decided_at = ? WHERE request_id = ?",
                ("approved" if approved else "rejected", approver, time.time(), request_id),
            )

    def get(self, request_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT request_id, created_at, summary, risk, status, decided_by, decided_at "
                "FROM pending_approvals WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def list_pending(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT request_id, created_at, summary, risk, status, decided_by, decided_at "
                "FROM pending_approvals WHERE status = 'pending' ORDER BY created_at",
            ).fetchall()
        return [_row_to_dict(row) for row in rows]


def _row_to_dict(row) -> dict:
    keys = ["request_id", "created_at", "summary", "risk", "status", "decided_by", "decided_at"]
    return dict(zip(keys, row))
