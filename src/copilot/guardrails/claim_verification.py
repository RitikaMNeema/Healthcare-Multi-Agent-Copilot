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
import re
from typing import Literal

from pydantic import BaseModel, Field

from copilot.config import FALLBACK_MODELS, PRIMARY_MODEL
from copilot.fallback import call_with_fallback

SYSTEM_PROMPT = """You are a claim-grounding checker for a governed healthcare
claims/denial-management copilot. You will be given the evidence actually retrieved
this turn (policy document excerpts and/or claim/metric records) and a draft answer.

Break the draft answer into its individual factual claims - ignore greetings, hedges,
and pure formatting. For each claim, decide exactly one of:
- "supported": the evidence directly states this or clearly implies it - list which
  specific piece(s) of evidence support it in evidence_refs (e.g. a document filename
  like "prior_authorization_policy.md", a claim ID like "CLM-000123", or a short
  distinguishing fragment of the evidence line itself if it carries no other identifier)
- "contradicted": the evidence directly conflicts with this claim - evidence_refs should
  identify the conflicting evidence the same way
- "insufficient_evidence": the evidence doesn't address this claim either way -
  evidence_refs should be empty, since there is nothing to cite

Use ONLY the evidence given - never mark something "supported" from outside/general
medical, legal, or payer-policy knowledge, even if you believe it to be true. If no
evidence was retrieved at all, every genuine factual claim should be
"insufficient_evidence" (a claim that isn't a factual assertion at all, like
"I can help with that", should be omitted rather than force-classified). If the draft
answer contains ANY genuine factual claim, findings must not be empty - an empty
findings list is only correct when the draft has nothing factual to check at all."""


class ClaimFinding(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "insufficient_evidence"]
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)


class ClaimVerificationResult(BaseModel):
    findings: list[ClaimFinding] = Field(default_factory=list)


_TRIVIAL_SENTENCE_RE = re.compile(
    r"^(hi|hello|hey|thanks?|thank you|ok|okay|sure|i can help|i'd be happy|i can't help|i cannot help)\b", re.I,
)


def _candidate_claim_sentences(text: str) -> list[str]:
    """A simple, deterministic, LLM-independent floor count of how many
    claim-shaped sentences a draft answer contains. This never judges
    support/contradiction (only an entailment-capable model can do that) -
    it exists solely so `has_coverage_gap` can catch the case where
    verification silently covered nothing at all, without trusting the same
    model whose output it's sanity-checking."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    return [s for s in sentences if len(s.split()) >= 4 and not _TRIVIAL_SENTENCE_RE.match(s)]


def has_coverage_gap(draft_answer: str, result: ClaimVerificationResult) -> bool:
    """True when the draft has at least one substantive, claim-shaped
    sentence but verification reported zero findings at all. An empty
    findings list is not the same thing as "everything is supported" - it
    can also mean verification failed to engage with the answer, and
    treating that as clean would let a completely unverified answer through
    with no signal at all. This is a floor check only: it flags *no*
    coverage, not partial coverage, to avoid false-positiving on answers
    that genuinely have only one or two claims."""
    return bool(_candidate_claim_sentences(draft_answer)) and len(result.findings) == 0


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
        "supported_claims": [
            {"claim": f.claim, "evidence_refs": f.evidence_refs}
            for f in result.findings if f.verdict == "supported"
        ],
        "contradicted_claims": [f.claim for f in result.findings if f.verdict == "contradicted"],
        "insufficient_evidence_claims": [f.claim for f in result.findings if f.verdict == "insufficient_evidence"],
    }
