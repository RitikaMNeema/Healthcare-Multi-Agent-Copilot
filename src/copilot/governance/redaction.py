"""Data minimization for anything written to the audit log or approval queue.

The audit log previously stored complete tool inputs/outputs and a slice of
the generated answer - workable for local debugging, indefensible as a real
audit trail for a system that touches PHI-adjacent data. What actually needs
to survive for an audit trail to do its job (prove what happened, when, and
under what policy decision) is *far* less than the content itself: which
tool ran, what it decided, how many records it touched, and a stable but
irreversible reference to any identifier involved - not the identifier, and
never patient-identifying text.

`hash_identifier` and `redact_text` are deliberately conservative pattern
matchers, not a full de-identification engine (real HIPAA Safe Harbor
de-identification needs healthcare-aware NER for names/addresses, which is
out of scope here - see the README's "what's still out of scope" note). They
catch the identifier shapes this project's own data actually produces
(patient IDs, SSNs, emails, phone numbers, MRNs, DOB mentions) so nothing
that shape reaches storage.
"""
import hashlib
import re

PATIENT_ID_RE = re.compile(r"\bPT-\d{4,}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
MRN_RE = re.compile(r"\bMRN[-:\s]?\d{5,10}\b", re.I)
DOB_RE = re.compile(r"\b(?:date of birth|dob|d\.o\.b\.|born)\b\s*(?:is|:|on)?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", re.I)
ADDRESS_HINT_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\s]{2,30}\b(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\b", re.I,
)


def hash_identifier(value: str) -> str:
    """Deterministic, irreversible - the same identifier always hashes the
    same way (so an auditor can still correlate events about one patient
    across log entries) without the hash revealing the identifier itself."""
    return "h_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def redact_text(text: str) -> str:
    if not text:
        return text
    text = PATIENT_ID_RE.sub(lambda m: hash_identifier(m.group()), text)
    text = MRN_RE.sub("[REDACTED-MRN]", text)
    text = SSN_RE.sub("[REDACTED-SSN]", text)
    text = EMAIL_RE.sub("[REDACTED-EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED-PHONE]", text)
    text = DOB_RE.sub("[REDACTED-DOB]", text)
    text = ADDRESS_HINT_RE.sub("[REDACTED-ADDRESS]", text)
    return text


def summarize_for_audit(*, tool_name: str, tool_input: dict, result) -> dict:
    """What actually gets written to the audit log for a tool call - never
    the raw input or result. `result_count` covers both single-record tools
    (analyze_denial -> 1) and multi-record ones (query_claims -> however many
    rows matched, from `total_matching_count` when the tool reports it)."""
    if isinstance(result, dict) and "total_matching_count" in result:
        result_count = result["total_matching_count"]
    elif isinstance(result, dict) and result.get("claim_id"):
        result_count = 1
    elif isinstance(result, list):
        result_count = len(result)
    elif isinstance(result, dict):
        result_count = 1
    else:
        result_count = 0

    input_hash = hash_identifier(repr(sorted(tool_input.items())))
    return {"tool": tool_name, "result_count": result_count, "input_hash": input_hash}
