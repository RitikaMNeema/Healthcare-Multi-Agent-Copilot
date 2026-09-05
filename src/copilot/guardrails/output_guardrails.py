import re

PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}

# Content that should never reach the user, full stop - not even after
# human review (see graph.py's "block" routing, which bypasses approval
# entirely). Keep this list to things where review genuinely can't help:
# instructions for the thing itself, not a topic that happens to be sensitive.
UNSAFE_INSTRUCTION_PHRASES = [
    "how to hack into",
    "how to make a bomb",
]

# High risk and always held for human review, but - unlike the phrases above
# - legitimately releasable once reviewed: explaining what a compliant bulk
# export *requires* is normal policy Q&A, not performing an unauthorized
# export. Matches the HIPAA policy's own wording for a bulk PHI export - a
# single claim/casework lookup never needs this phrase, so its presence is a
# reliable signal the answer is discussing a bulk export of patient-
# identifiable data rather than routine individual casework.
SENSITIVE_TOPIC_PHRASES = [
    "patient-identifiable claims data",
    "patient's social security number",
]


def scan_output(text: str) -> tuple[str, list[str]]:
    issues = []
    lowered = text.lower()

    for phrase in UNSAFE_INSTRUCTION_PHRASES:
        if phrase in lowered:
            issues.append(f"contains unsafe-instruction phrase: '{phrase}'")
    for phrase in SENSITIVE_TOPIC_PHRASES:
        if phrase in lowered:
            issues.append(f"contains sensitive-topic phrase: '{phrase}'")

    pii_hits = [name for name, pattern in PII_PATTERNS.items() if pattern.search(text)]
    if pii_hits:
        issues.append(f"possible PII detected: {', '.join(pii_hits)}")

    if any("unsafe-instruction phrase" in issue or "sensitive-topic phrase" in issue for issue in issues):
        risk = "high"
    elif issues:
        risk = "medium"
    else:
        risk = "low"

    return risk, issues
