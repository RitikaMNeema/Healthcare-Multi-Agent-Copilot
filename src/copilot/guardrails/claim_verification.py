"""Claim-level grounding: does the answer say only what the evidence supports.

`citation_check.py` checks that a citation *token* the answer mentions (a doc
filename, a claim ID) was actually retrieved this turn - that's citation
provenance, not semantic verification. An answer can cite a real, retrieved
document while still asserting something that document doesn't say, or say
nothing citable at all while still making an unsupported factual claim. This
module is the actual "claim-level verification": it breaks the draft answer
into individual factual claims and classifies each one against the evidence
actually retrieved this turn.

This costs one extra LLM call per turn and is only as good as the model's
entailment judgment - it is a real grounding check, not a guarantee. Treat
"contradicted" and "insufficient_evidence" findings as a strong signal for
risk escalation and human review (see graph.py's node_critic), not as an
infallible filter that makes review unnecessary.
"""
from typing import Literal

from pydantic import BaseModel, Field

from copilot.config import FALLBACK_MODELS, PRIMARY_MODEL
from copilot.fallback import call_with_fallback

SYSTEM_PROMPT = """You are a claim-grounding checker for a governed healthcare
claims/denial-management copilot. You will be given the evidence actually retrieved
this turn (policy document excerpts and/or claim/metric records) and a draft answer.

Break the draft answer into its individual factual claims - ignore greetings, hedges,
and pure formatting. For each claim, decide exactly one of:
- "supported": the evidence directly states this or clearly implies it
- "contradicted": the evidence directly conflicts with this claim
- "insufficient_evidence": the evidence doesn't address this claim either way

Use ONLY the evidence given - never mark something "supported" from outside/general
medical, legal, or payer-policy knowledge, even if you believe it to be true. If no
evidence was retrieved at all, every genuine factual claim should be
"insufficient_evidence" (a claim that isn't a factual assertion at all, like
"I can help with that", should be omitted rather than force-classified)."""


class ClaimFinding(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "insufficient_evidence"]
    rationale: str


class ClaimVerificationResult(BaseModel):
    findings: list[ClaimFinding] = Field(default_factory=list)


def verify_claims(llm, *, draft_answer: str, evidence_text: str) -> tuple[ClaimVerificationResult, str]:
    def attempt(model: str) -> ClaimVerificationResult:
        content = (
            f"Evidence retrieved this turn:\n{evidence_text or '(no evidence was retrieved this turn)'}\n\n"
            f"Draft answer:\n{draft_answer}"
        )
        return llm.complete_structured(
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": content}],
            schema=ClaimVerificationResult, model=model,
        )

    return call_with_fallback(attempt, models=[PRIMARY_MODEL, *FALLBACK_MODELS])


def summarize(result: ClaimVerificationResult) -> dict:
    return {
        "supported_claims": [f.claim for f in result.findings if f.verdict == "supported"],
        "contradicted_claims": [f.claim for f in result.findings if f.verdict == "contradicted"],
        "insufficient_evidence_claims": [f.claim for f in result.findings if f.verdict == "insufficient_evidence"],
    }
