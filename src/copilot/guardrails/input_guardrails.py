import re

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\b.{0,30}\binstructions\b", re.I),
    re.compile(r"disregard\b.{0,30}\b(guardrails|rules|policy)\b", re.I),
    re.compile(r"reveal\b.{0,20}\b(system prompt|instructions)\b", re.I),
    re.compile(r"act as (an? )?unrestricted", re.I),
    re.compile(r"pretend (you have|there is) no (safety|restrictions?|rules)", re.I),
]

DISALLOWED_TOPIC_PATTERNS = [
    re.compile(r"how to (make|build|synthesize) (a )?(bomb|explosive|weapon)", re.I),
    re.compile(r"undetectable malware", re.I),
    re.compile(r"bypass (antivirus|edr|security controls)", re.I),
]


def check_input(text: str) -> dict:
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return {"blocked": True, "reason": "possible prompt injection attempt"}
    for pattern in DISALLOWED_TOPIC_PATTERNS:
        if pattern.search(text):
            return {"blocked": True, "reason": "disallowed topic"}
    return {"blocked": False, "reason": ""}
