import json
import threading
import time

import pytest

from copilot.governance.audit import AuditLog
from copilot.governance.identity import UnknownAPIKeyError, generate_api_key, resolve_identity
from copilot.governance.rate_limit import RateLimiter


def test_generate_and_resolve_api_key(tmp_path):
    path = str(tmp_path / "keys.json")
    raw_key = generate_api_key("newuser", "operator", path=path)

    identity = resolve_identity(raw_key, path=path)
    assert identity == {"user_id": "newuser", "role": "operator", "tenant_id": "default"}


def test_unknown_api_key_rejected(tmp_path):
    path = str(tmp_path / "keys.json")
    generate_api_key("someone", "viewer", path=path)
    with pytest.raises(UnknownAPIKeyError):
        resolve_identity("definitely-not-a-real-key", path=path)


def test_api_keys_are_stored_hashed_not_plaintext(tmp_path):
    path = str(tmp_path / "keys.json")
    raw_key = generate_api_key("someone", "admin", path=path)
    with open(path) as f:
        stored = json.load(f)
    assert raw_key not in json.dumps(stored)


def test_rate_limiter_allows_then_blocks():
    limiter = RateLimiter(max_requests=3, window_seconds=60.0)
    assert limiter.check("alice") is True
    assert limiter.check("alice") is True
    assert limiter.check("alice") is True
    assert limiter.check("alice") is False


def test_rate_limiter_tracks_keys_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    assert limiter.check("alice") is True
    assert limiter.check("bob") is True
    assert limiter.check("alice") is False


def test_rate_limiter_window_expires():
    limiter = RateLimiter(max_requests=1, window_seconds=0.05)
    assert limiter.check("alice") is True
    assert limiter.check("alice") is False
    time.sleep(0.06)
    assert limiter.check("alice") is True


def test_audit_chain_verifies_when_untampered():
    audit = AuditLog()
    audit.log(request_id="r1", event_type="a", payload={"x": 1})
    audit.log(request_id="r1", event_type="b", payload={"x": 2})
    is_valid, broken_id = audit.verify_chain()
    assert is_valid is True
    assert broken_id is None


def test_audit_chain_detects_tampering():
    audit = AuditLog()
    audit.log(request_id="r1", event_type="a", payload={"amount": 100})
    second_id = audit.log(request_id="r1", event_type="b", payload={"amount": 200})

    # Simulate an attacker editing a row directly in the database, bypassing `log()`.
    with audit._connect() as conn:
        conn.execute("UPDATE audit_events SET payload = ? WHERE id = ?", (json.dumps({"amount": 999999}), second_id))

    is_valid, broken_id = audit.verify_chain()
    assert is_valid is False
    assert broken_id == second_id


def test_audit_chain_detects_deleted_row():
    audit = AuditLog()
    first_id = audit.log(request_id="r1", event_type="a", payload={})
    second_id = audit.log(request_id="r1", event_type="b", payload={})
    audit.log(request_id="r1", event_type="c", payload={})

    with audit._connect() as conn:
        conn.execute("DELETE FROM audit_events WHERE id = ?", (first_id,))

    # Deleting the first row means the new first row's stored prev_hash still
    # points at the deleted row's hash, not the genesis value verify_chain
    # expects to see first - the gap is detected immediately.
    is_valid, broken_id = audit.verify_chain()
    assert is_valid is False
    assert broken_id == second_id


def test_audit_chain_survives_concurrent_writers(tmp_path):
    # Without BEGIN IMMEDIATE around read-last-hash-then-insert, two threads
    # can both read the same "last hash" and each compute a chain entry
    # against it, corrupting the total order. 8 threads x 25 writes each is
    # enough to reliably trigger that race on the old (unlocked) code path.
    audit = AuditLog(db_path=str(tmp_path / "concurrent_audit.db"))
    n_threads = 8
    writes_per_thread = 25
    errors: list[Exception] = []

    def writer(thread_id: int) -> None:
        try:
            for i in range(writes_per_thread):
                audit.log(request_id=f"r-{thread_id}", event_type="concurrent_write", payload={"i": i})
        except Exception as exc:  # noqa: BLE001 - capture for the main thread to assert on
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []

    is_valid, broken_id = audit.verify_chain()
    assert is_valid is True, f"chain broken at {broken_id}"

    all_events = audit.recent(limit=n_threads * writes_per_thread + 10)
    assert len(all_events) == n_threads * writes_per_thread
