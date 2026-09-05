"""Reproduces the reported production bug: a database created by an older
version of this code (before `tenant_id` existed) must upgrade cleanly when
opened by the current code, not raise `OperationalError: no such column:
tenant_id`. `CREATE TABLE IF NOT EXISTS` alone can't do this - it's a no-op
against a table that already exists - hence the explicit, versioned
migrations in `governance/migrations.py`.
"""
import sqlite3
import time

from copilot.governance.approvals import ApprovalQueue
from copilot.governance.audit import AuditLog, _compute_entry_hash_v1
from copilot.governance.migrations import apply_migrations


def _create_pre_tenant_database(path: str) -> None:
    """Recreates exactly the schema this project shipped before tenant_id
    was added - no tenant_id column, no audit_chain_meta, no
    schema_migrations table - the real shape of an already-deployed database."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE audit_events (
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
        CREATE INDEX idx_audit_request ON audit_events(request_id);

        CREATE TABLE pending_approvals (
            request_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            requester_user_id TEXT,
            summary TEXT NOT NULL,
            risk TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            decided_by TEXT,
            decided_at REAL
        );
    """)
    # A real (not placeholder) hash, computed with the actual pre-tenancy
    # algorithm - so tests that call verify_chain() exercise the real
    # migration-boundary logic, not a fixture that would fail regardless.
    genesis = "0" * 64
    entry_hash = _compute_entry_hash_v1(
        genesis, event_id="e1", request_id="r1", ts=1.0, user_id="alice", role="operator",
        event_type="legacy_event", payload_json="{}",
    )
    conn.execute(
        "INSERT INTO audit_events (id, request_id, ts, user_id, role, event_type, payload, prev_hash, entry_hash) "
        "VALUES ('e1', 'r1', 1.0, 'alice', 'operator', 'legacy_event', '{}', ?, ?)",
        (genesis, entry_hash),
    )
    conn.commit()
    conn.close()


def test_auditlog_upgrades_a_pre_tenant_database_without_error(tmp_path):
    db_path = str(tmp_path / "legacy_audit.db")
    _create_pre_tenant_database(db_path)

    audit = AuditLog(db_path=db_path)  # must not raise OperationalError

    # The pre-existing row survives the migration untouched.
    trail = audit.trail_for("r1")
    assert len(trail) == 1
    assert trail[0]["event_type"] == "legacy_event"

    # tenant_id now exists and works for new writes/reads.
    audit.log(request_id="r2", event_type="new_event", tenant_id="tenant-a", payload={})
    assert audit.recent(tenant_id="tenant-a")[0]["request_id"] == "r2"

    # The chain verifies across the legacy/current boundary: the pre-migration
    # row (hashed without tenant_id) and the post-migration row (hashed with
    # it) are each checked with the algorithm that actually produced their
    # stored hash - migrating does not itself break tamper detection for a
    # database that was never tampered with.
    is_valid, _ = audit.verify_chain()
    assert is_valid is True


def test_tampering_a_legacy_row_is_still_detected_after_migration(tmp_path):
    # The legacy/current hash-algorithm split must not become a loophole - a
    # tampered legacy row still has to fail verification, just checked with
    # the algorithm that actually produced its original hash.
    db_path = str(tmp_path / "legacy_audit.db")
    _create_pre_tenant_database(db_path)
    audit = AuditLog(db_path=db_path)
    audit.log(request_id="r2", event_type="new_event", tenant_id="tenant-a", payload={})

    with audit._connect() as conn:
        conn.execute("UPDATE audit_events SET payload = '{\"tampered\": true}' WHERE id = 'e1'")

    is_valid, broken_id = audit.verify_chain()
    assert is_valid is False
    assert broken_id == "e1"


def test_purge_to_empty_then_new_write_is_not_misverified_as_legacy(tmp_path):
    # The exact reported edge case: SQLite reuses low rowids once a table is
    # completely emptied. If hash-algorithm choice were still inferred from
    # "rowid <= legacy_hash_boundary_rowid", a new v2-hashed row landing at a
    # reused low rowid would be wrongly checked with the v1 algorithm and
    # fail verification even though nothing was tampered. The explicit,
    # permanent hash_version column (migration 3) makes this impossible by
    # construction - a row's version is stamped once, at write time, not
    # inferred from its position.
    db_path = str(tmp_path / "legacy_audit.db")
    _create_pre_tenant_database(db_path)
    audit = AuditLog(db_path=db_path)  # migrates; legacy row 'e1' gets hash_version=1

    purged = audit.purge_older_than(cutoff_ts=time.time() + 1)  # purges every row, including 'e1'
    assert purged == 1

    with audit._connect() as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    assert remaining == 0  # table is now empty - next INSERT can reuse rowid 1

    audit.log(request_id="r2", event_type="new_event", tenant_id="tenant-a", payload={})

    is_valid, broken_id = audit.verify_chain()
    assert is_valid is True, f"false tamper report at {broken_id} - new row misverified with the wrong algorithm"

    with audit._connect() as conn:
        stored_version = conn.execute("SELECT hash_version FROM audit_events WHERE request_id = 'r2'").fetchone()[0]
    assert stored_version == 2


def test_approvalqueue_upgrades_a_pre_tenant_database_without_error(tmp_path):
    db_path = str(tmp_path / "legacy_approvals.db")
    _create_pre_tenant_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO pending_approvals (request_id, created_at, requester_user_id, summary, risk, status) "
        "VALUES ('r1', 1.0, 'bob', 'legacy summary', 'medium', 'pending')",
    )
    conn.commit()
    conn.close()

    queue = ApprovalQueue(db_path=db_path)  # must not raise OperationalError

    pending = queue.get("r1")
    assert pending is not None
    assert pending["summary"] == "legacy summary"
    assert pending["tenant_id"] is None  # legacy row predates tenancy - no data was fabricated

    queue.submit("r2", summary="new request", risk="high", requester_user_id="carol", tenant_id="tenant-a")
    assert queue.get("r2")["tenant_id"] == "tenant-a"


def test_apply_migrations_is_idempotent(tmp_path):
    db_path = str(tmp_path / "fresh.db")
    conn = sqlite3.connect(db_path)
    first = apply_migrations(conn)
    second = apply_migrations(conn)
    conn.close()
    assert first == [1, 2, 3]
    assert second == []  # nothing left to apply


def test_migrations_recorded_and_shared_across_auditlog_and_approvalqueue(tmp_path):
    # Both classes point at the same file by default in production - whichever
    # opens first should fully migrate it, and the second must see it as
    # already current rather than re-running (or conflicting with) migrations.
    db_path = str(tmp_path / "shared.db")
    AuditLog(db_path=db_path)
    conn = sqlite3.connect(db_path)
    applied_versions = {row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()}
    conn.close()
    assert applied_versions == {1, 2, 3}

    ApprovalQueue(db_path=db_path)  # must not raise, must not duplicate migration rows
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    conn.close()
    assert count == 3
