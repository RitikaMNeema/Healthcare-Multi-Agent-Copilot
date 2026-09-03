"""Claim-level and document-level citation verification.

A draft answer can cite a policy document (`[filename.md]`) or a specific
claim (`CLM-000123`). This checks every such citation actually corresponds to
something a tool call surfaced *this turn* - catching a hallucinated citation
(a plausible-looking claim ID or filename the model invented rather than
retrieved) before it reaches the guardrail risk score.
"""
import re

DOC_CITATION_RE = re.compile(r"\[([a-zA-Z0-9_\-]+\.md)\]")
CLAIM_CITATION_RE = re.compile(r"\b(CLM-\d{6})\b", re.I)


def verify_citations(answer_text: str, *, evidence_doc_sources: set[str], evidence_claim_ids: set[str]) -> list[str]:
    issues = []

    for doc in dict.fromkeys(DOC_CITATION_RE.findall(answer_text)):
        if doc not in evidence_doc_sources:
            issues.append(f"unverified citation: [{doc}] was not among the documents retrieved this turn")

    for claim_id in dict.fromkeys(m.upper() for m in CLAIM_CITATION_RE.findall(answer_text)):
        if claim_id not in evidence_claim_ids:
            issues.append(f"unverified citation: claim {claim_id} was not returned by any tool call this turn")

    return issues
