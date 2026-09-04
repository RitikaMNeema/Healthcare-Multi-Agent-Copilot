"""API-key identity resolution for the HTTP surface.

The FastAPI layer must never trust a client-supplied `role`, `user_id`, or
`tenant_id` - a JSON field is just a claim, easily forged. An identity
resolved from a server-held key -> (user_id, role, tenant_id) mapping is a
fact instead. Keys are stored hashed (SHA-256), never in plaintext, so a
leaked copy of the identities file doesn't hand out working credentials.

`tenant_id` enforces that one organization's requests, approvals, and audit
records are invisible to another's callers, even an admin/compliance_officer
in a different tenant - see the tenant checks in `api/server.py`.

The committed `data/api_keys.json` holds a handful of fixed, publicly
documented demo keys (see README) for trying the project locally - clearly
not a real secret store. A real deployment points `COPILOT_API_KEYS_FILE` at
its own, non-committed file (via `.env`) and mints keys with
`generate_api_key()`, which returns the raw key exactly once and persists
only its hash.
"""
import hashlib
import json
import os
import secrets

from copilot.config import default_api_keys_path

DEFAULT_TENANT_ID = "default"


class UnknownAPIKeyError(Exception):
    pass


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def load_identities(path: str | None = None) -> dict:
    path = path or default_api_keys_path()
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_identity(raw_key: str, path: str | None = None) -> dict:
    """Returns {"user_id", "role", "tenant_id"} for a valid key, else raises."""
    identities = load_identities(path)
    identity = identities.get(_hash_key(raw_key))
    if identity is None:
        raise UnknownAPIKeyError("invalid or unknown API key")
    identity.setdefault("tenant_id", DEFAULT_TENANT_ID)
    return identity


def generate_api_key(user_id: str, role: str, tenant_id: str = DEFAULT_TENANT_ID, path: str | None = None) -> str:
    """Mints a new identity and returns the raw key - the only time it's ever
    visible. Only its hash is written to disk."""
    path = path or default_api_keys_path()
    identities = load_identities(path)
    raw_key = secrets.token_urlsafe(24)
    identities[_hash_key(raw_key)] = {"user_id": user_id, "role": role, "tenant_id": tenant_id}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(identities, f, indent=2)
    return raw_key
