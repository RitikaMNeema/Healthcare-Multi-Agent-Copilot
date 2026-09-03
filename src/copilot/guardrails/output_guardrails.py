import re

PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}

BANNED_PHRASES = [
    "how to hack into",
    "social security number",
    "credit card number",
    "how to make a bomb",
]


def scan_output(text: str) -> tuple[str, list[str]]:
    issues = []
    lowered = text.lower()

    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            issues.append(f"contains banned phrase: '{phrase}'")

    pii_hits = [name for name, pattern in PII_PATTERNS.items() if pattern.search(text)]
    if pii_hits:
        issues.append(f"possible PII detected: {', '.join(pii_hits)}")

    if any("banned phrase" in issue for issue in issues):
        risk = "high"
    elif issues:
        risk = "medium"
    else:
        risk = "low"

    return risk, issues
