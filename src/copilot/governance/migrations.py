"""Explicit, versioned SQLite migrations for the audit/approvals database.

`CREATE TABLE IF NOT EXISTS` only helps a brand-new database - against an
*existing* table it is a no-op, so adding a column to an already-deployed
schema (like `tenant_id` on `audit_events`/`pending_approvals`) silently does
nothing, and the column is simply missing - surfacing later, in production,
as `OperationalError: no such column: tenant_id`. SQLite has no
`ADD COLUMN IF NOT EXISTS`, so "did this database already get this change"
has to be tracked explicitly.

This module tracks which migrations a database has applied (a
`schema_migrations` table) and applies whichever ones it's missing, in
version order, each in its own transaction - so a fresh database and an
old, already-deployed one converge on the same schema from wherever they
started, with no separate "initial schema" special case to keep in sync with
later ALTERs by hand.

Both `AuditLog` and `ApprovalQueue` call `apply_migrations()` against
whatever file they're given (by default, the same file) - each migration
touches only the table(s) it owns, and running the full set twice against an
already-migrated database is a no-op (every migration here is
idempotent on top of `schema_migrations` tracking, as a second line of
defense).
"""
import sqlite3
import time
from typing import Callable

_GENESIS_HASH = "0" * 64


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    if not _has_column(conn, table, column):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def _m1_initial_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
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

        CREATE TABLE IF NOT EXISTS pending_approvals (
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


def _m2_tenancy_and_chain_meta(conn: sqlite3.Connection) -> None:
    # Any audit_events row that already existed when this migration runs was
    # hash-chained under the pre-tenancy algorithm - its stored entry_hash
    # was computed from a canonical payload that never had a tenant_id field
    # at all, not even as null. Recomputing that row's hash with an algorithm
    # that includes tenant_id would never match, falsely flagging every
    # migrated database as 100% tampered. Recording this boundary rowid
    # before adding the column lets verify_chain() apply the correct
    # historical algorithm on either side of it - see audit.py's
    # _compute_entry_hash_v1/_v2.
    legacy_boundary_rowid = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM audit_events").fetchone()[0]

    _add_column_if_missing(conn, "audit_events", "tenant_id", "TEXT")
    _add_column_if_missing(conn, "pending_approvals", "tenant_id", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_events(tenant_id)")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS audit_chain_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            trusted_genesis_hash TEXT NOT NULL,
            legacy_hash_boundary_rowid INTEGER NOT NULL DEFAULT 0
        );
    """)
    # Defensive: in case audit_chain_meta already existed (e.g. from a
    # version of this code that created it inline before this migration
    # module existed) without this column - CREATE TABLE IF NOT EXISTS above
    # would otherwise silently skip adding it, the exact bug this migration
    # system exists to prevent.
    _add_column_if_missing(conn, "audit_chain_meta", "legacy_hash_boundary_rowid", "INTEGER NOT NULL DEFAULT 0")
    existing_meta = conn.execute("SELECT id FROM audit_chain_meta WHERE id = 1").fetchone()
    if existing_meta is None:
        conn.execute(
            "INSERT INTO audit_chain_meta (id, trusted_genesis_hash, legacy_hash_boundary_rowid) VALUES (1, ?, ?)",
            (_GENESIS_HASH, legacy_boundary_rowid),
        )
    else:
        # audit_chain_meta already existed (from a version of this code that
        # created it inline, before schema_migrations tracked anything) -
        # leave its trusted_genesis_hash alone, but the boundary must still
        # be set correctly or every pre-existing row would wrongly be
        # verified under the new (tenant_id-aware) hash algorithm.
        conn.execute(
            "UPDATE audit_chain_meta SET legacy_hash_boundary_rowid = ? WHERE id = 1", (legacy_boundary_rowid,),
        )


# (version, description, migration function) - append-only. Never edit an
# already-shipped entry's behavior after it's been applied anywhere; add a
# new, later-numbered migration instead, the same rule as any other
# migration system.
MIGRATIONS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "create audit_events and pending_approvals base tables", _m1_initial_schema),
    (2, "add tenant_id to audit_events/pending_approvals, add audit_chain_meta", _m2_tenancy_and_chain_meta),
]


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    """Applies every migration this connection's database hasn't recorded
    yet, in order. Returns the version numbers actually applied (empty if
    the database was already current)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL, description TEXT NOT NULL)",
    )
    already_applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()}

    newly_applied = []
    for version, description, migrate in MIGRATIONS:
        if version in already_applied:
            continue
        migrate(conn)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
            (version, time.time(), description),
        )
        newly_applied.append(version)
    conn.commit()
    return newly_applied
