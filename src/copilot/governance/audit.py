import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager

from copilot.config import default_audit_db_path
from copilot.sqlite_utils import connect as sqlite_connect

GENESIS_HASH = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    ts REAL NOT NULL,
    user_id TEXT,
    role TEXT,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_request ON audit_events(request_id);
"""


def _compute_entry_hash(prev_hash: str, *, event_id: str, request_id: str, ts: float,
                         user_id: str | None, role: str | None, event_type: str, payload_json: str) -> str:
    canonical = json.dumps(
        {"id": event_id, "request_id": request_id, "ts": ts, "user_id": user_id,
         "role": role, "event_type": event_type, "payload": payload_json},
        sort_keys=True,
    )
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only, hash-chained audit trail of every governance-relevant
    decision the copilot makes - each entry's hash covers the previous
    entry's hash plus its own fields (the same construction as a git commit
    chain), so `verify_chain()` can detect a row edited or deleted after the
    fact. This does not stop someone with direct DB access from rewriting the
    whole chain from the tampered row forward - no local hash chain can, that
    needs an external anchor (e.g. periodically publishing the latest hash
    somewhere append-only) - but it does mean tampering can't hide, only be
    made total, which is the property that actually matters for an audit
    trail: silent, partial edits become detectable.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or default_audit_db_path()
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite_connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log(self, *, request_id: str, event_type: str, payload: dict | None = None,
             user_id: str | None = None, role: str | None = None) -> str:
        event_id = str(uuid.uuid4())
        ts = time.time()
        payload_json = json.dumps(payload or {}, default=str)

        with self._connect() as conn:
            last_hash_row = conn.execute("SELECT entry_hash FROM audit_events ORDER BY rowid DESC LIMIT 1").fetchone()
            prev_hash = last_hash_row[0] if last_hash_row else GENESIS_HASH
            entry_hash = _compute_entry_hash(
                prev_hash, event_id=event_id, request_id=request_id, ts=ts,
                user_id=user_id, role=role, event_type=event_type, payload_json=payload_json,
            )
            conn.execute(
                "INSERT INTO audit_events (id, request_id, ts, user_id, role, event_type, payload, prev_hash, entry_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, request_id, ts, user_id, role, event_type, payload_json, prev_hash, entry_hash),
            )
        return event_id

    def trail_for(self, request_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, request_id, ts, user_id, role, event_type, payload "
                "FROM audit_events WHERE request_id = ? ORDER BY rowid",
                (request_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def recent(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, request_id, ts, user_id, role, event_type, payload "
                "FROM audit_events ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def verify_chain(self) -> tuple[bool, str | None]:
        """Recomputes every entry's hash from its stored fields and the
        previous entry's stored hash. Returns (True, None) if the chain is
        intact, or (False, id_of_first_broken_entry) at the first mismatch -
        everything after that point is unverifiable regardless of whether it
        was itself altered."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, request_id, ts, user_id, role, event_type, payload, prev_hash, entry_hash "
                "FROM audit_events ORDER BY rowid",
            ).fetchall()

        expected_prev = GENESIS_HASH
        for row in rows:
            event_id, request_id, ts, user_id, role, event_type, payload_json, stored_prev, stored_entry = row
            if stored_prev != expected_prev:
                return False, event_id
            recomputed = _compute_entry_hash(
                stored_prev, event_id=event_id, request_id=request_id, ts=ts,
                user_id=user_id, role=role, event_type=event_type, payload_json=payload_json,
            )
            if recomputed != stored_entry:
                return False, event_id
            expected_prev = stored_entry
        return True, None


def _row_to_dict(row) -> dict:
    keys = ["id", "request_id", "ts", "user_id", "role", "event_type", "payload"]
    record = dict(zip(keys, row))
    record["payload"] = json.loads(record["payload"])
    return record
