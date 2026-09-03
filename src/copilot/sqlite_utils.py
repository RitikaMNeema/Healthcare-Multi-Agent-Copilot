"""Shared SQLite connection setup.

WAL (write-ahead log) mode lets readers proceed without blocking on a writer,
which is where SQLite's default rollback-journal mode serializes far more
than it needs to. `scripts/load_test.py` measured this directly: at
concurrency 50, p50 latency was 141ms and p99 was 1.6s in the default mode
purely from writers queuing behind each other - see the load-test section in
the README for the before/after numbers. `busy_timeout` makes a connection
wait for a lock instead of immediately raising "database is locked" under a
contention spike WAL doesn't fully absorb.
"""
import sqlite3


def connect(path: str, **kwargs) -> sqlite3.Connection:
    conn = sqlite3.connect(path, **kwargs)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
