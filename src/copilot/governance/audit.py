import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager

from copilot.config import default_audit_db_path
from copilot.governance.migrations import apply_migrations
from copilot.sqlite_utils import connect as sqlite_connect

GENESIS_HASH = "0" * 64


def _compute_entry_hash_v1(prev_hash: str, *, event_id: str, request_id: str, ts: float,
                            user_id: str | None, role: str | None,
                            event_type: str, payload_json: str) -> str:
    """The pre-tenancy hash algorithm - canonical JSON has no tenant_id key
    at all (not even null). Only ever used to *verify* rows written before
    migration 2 added tenant_id; never used for new writes. Do not add
    fields here - a row's original hash is fixed forever once written, and
    this function exists solely to reproduce it."""
    canonical = json.dumps(
        {"id": event_id, "request_id": request_id, "ts": ts, "user_id": user_id,
         "role": role, "event_type": event_type, "payload": payload_json},
        sort_keys=True,
    )
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


def _compute_entry_hash_v2(prev_hash: str, *, event_id: str, request_id: str, ts: float,
                            user_id: str | None, role: str | None, tenant_id: str | None,
                            event_type: str, payload_json: str) -> str:
    canonical = json.dumps(
        {"id": event_id, "request_id": request_id, "ts": ts, "user_id": user_id,
         "role": role, "tenant_id": tenant_id, "event_type": event_type, "payload": payload_json},
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

    Payloads passed to `log()` are the caller's responsibility to minimize -
    see `governance/redaction.py` - this class stores whatever it's given
    verbatim (hash-chained, not encrypted at the row level) and does not
    itself inspect payload content.

    `log()` wraps its read-last-hash-then-insert in a single `BEGIN
    IMMEDIATE` transaction so two concurrent writers can't both read the same
    "last hash" and each compute a chain entry against it - see
    `tests/test_security.py`'s concurrent-writer test.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or default_audit_db_path()
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            apply_migrations(conn)

    @contextmanager
    def _connect(self):
        conn = sqlite_connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log(self, *, request_id: str, event_type: str, payload: dict | None = None,
             user_id: str | None = None, role: str | None = None, tenant_id: str | None = None) -> str:
        event_id = str(uuid.uuid4())
        ts = time.time()
        payload_json = json.dumps(payload or {}, default=str)

        # Autocommit mode (isolation_level=None) so our own explicit BEGIN
        # IMMEDIATE takes effect instead of sqlite3's implicit deferred
        # transaction - IMMEDIATE grabs the write lock before the SELECT,
        # so no other writer's INSERT can land between "read last hash" and
        # "insert this entry".
        conn = sqlite_connect(self.db_path, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            last_hash_row = conn.execute("SELECT entry_hash FROM audit_events ORDER BY rowid DESC LIMIT 1").fetchone()
            prev_hash = last_hash_row[0] if last_hash_row else GENESIS_HASH
            entry_hash = _compute_entry_hash_v2(
                prev_hash, event_id=event_id, request_id=request_id, ts=ts,
                user_id=user_id, role=role, tenant_id=tenant_id, event_type=event_type, payload_json=payload_json,
            )
            conn.execute(
                "INSERT INTO audit_events "
                "(id, request_id, ts, user_id, role, tenant_id, event_type, payload, prev_hash, entry_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, request_id, ts, user_id, role, tenant_id, event_type, payload_json, prev_hash, entry_hash),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return event_id

    def trail_for(self, request_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, request_id, ts, user_id, role, tenant_id, event_type, payload "
                "FROM audit_events WHERE request_id = ? ORDER BY rowid",
                (request_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def recent(self, limit: int = 50, tenant_id: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if tenant_id is not None:
                rows = conn.execute(
                    "SELECT id, request_id, ts, user_id, role, tenant_id, event_type, payload "
                    "FROM audit_events WHERE tenant_id = ? ORDER BY rowid DESC LIMIT ?",
                    (tenant_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, request_id, ts, user_id, role, tenant_id, event_type, payload "
                    "FROM audit_events ORDER BY rowid DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def purge_older_than(self, cutoff_ts: float) -> int:
        """Retention enforcement: permanently deletes events older than
        `cutoff_ts`. This is the *only* sanctioned way to shorten the chain:
        it records what the new oldest surviving row's `prev_hash` legitimately
        is in `audit_chain_meta` before deleting, so `verify_chain()` still
        expects exactly that value afterward. An attacker deleting the same
        rows directly (bypassing this method) leaves the meta table
        unchanged, so verify_chain still catches it - only deletion *through
        this method* is trusted, not deletion of old-enough rows in general.
        Real deployments should archive purged rows (encrypted, access-logged)
        before deleting, per the retention policy documented in the README -
        this method only performs the deletion half."""
        with self._connect() as conn:
            new_first = conn.execute(
                "SELECT prev_hash FROM audit_events WHERE ts >= ? ORDER BY rowid LIMIT 1", (cutoff_ts,),
            ).fetchone()
            cursor = conn.execute("DELETE FROM audit_events WHERE ts < ?", (cutoff_ts,))
            if new_first is not None:
                conn.execute("UPDATE audit_chain_meta SET trusted_genesis_hash = ? WHERE id = 1", (new_first[0],))
            return cursor.rowcount

    def verify_chain(self) -> tuple[bool, str | None]:
        """Recomputes every entry's hash from its stored fields and the
        previous entry's stored hash. Returns (True, None) if the chain is
        intact, or (False, id_of_first_broken_entry) at the first mismatch -
        everything after that point is unverifiable regardless of whether it
        was itself altered.

        Rows at or before `legacy_hash_boundary_rowid` (set once, at
        migration time - see migrations.py) predate tenant_id and are
        verified with the original pre-tenancy algorithm; everything after
        uses the current one. Without this split, every database that has
        ever been migrated would recompute a different hash than what a
        legacy row was actually stored with and get flagged as 100% tampered
        - a false alarm, not a real one."""
        with self._connect() as conn:
            meta_row = conn.execute(
                "SELECT trusted_genesis_hash, legacy_hash_boundary_rowid FROM audit_chain_meta WHERE id = 1",
            ).fetchone()
            rows = conn.execute(
                "SELECT rowid, id, request_id, ts, user_id, role, tenant_id, event_type, payload, prev_hash, entry_hash "
                "FROM audit_events ORDER BY rowid",
            ).fetchall()

        expected_prev = meta_row[0] if meta_row else GENESIS_HASH
        legacy_boundary = meta_row[1] if meta_row else 0
        for row in rows:
            (rowid, event_id, request_id, ts, user_id, role, tenant_id, event_type, payload_json,
             stored_prev, stored_entry) = row
            if stored_prev != expected_prev:
                return False, event_id
            if rowid <= legacy_boundary:
                recomputed = _compute_entry_hash_v1(
                    stored_prev, event_id=event_id, request_id=request_id, ts=ts,
                    user_id=user_id, role=role, event_type=event_type, payload_json=payload_json,
                )
            else:
                recomputed = _compute_entry_hash_v2(
                    stored_prev, event_id=event_id, request_id=request_id, ts=ts,
                    user_id=user_id, role=role, tenant_id=tenant_id, event_type=event_type, payload_json=payload_json,
                )
            if recomputed != stored_entry:
                return False, event_id
            expected_prev = stored_entry
        return True, None


def _row_to_dict(row) -> dict:
    keys = ["id", "request_id", "ts", "user_id", "role", "tenant_id", "event_type", "payload"]
    record = dict(zip(keys, row))
    record["payload"] = json.loads(record["payload"])
    return record
