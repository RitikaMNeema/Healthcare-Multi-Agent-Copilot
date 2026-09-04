from copilot.guardrails.claim_verification import ClaimFinding, ClaimVerificationResult, summarize, verify_claims
from copilot.llm import MockBackend


def test_summarize_buckets_by_verdict():
    result = ClaimVerificationResult(findings=[
        ClaimFinding(claim="a", verdict="supported", rationale="r"),
        ClaimFinding(claim="b", verdict="contradicted", rationale="r"),
        ClaimFinding(claim="c", verdict="insufficient_evidence", rationale="r"),
    ])
    summary = summarize(result)
    assert summary["supported_claims"] == ["a"]
    assert summary["contradicted_claims"] == ["b"]
    assert summary["insufficient_evidence_claims"] == ["c"]


def test_summarize_handles_no_findings():
    summary = summarize(ClaimVerificationResult(findings=[]))
    assert summary == {"supported_claims": [], "contradicted_claims": [], "insufficient_evidence_claims": []}


def test_verify_claims_with_no_evidence_marks_everything_insufficient():
    llm = MockBackend()
    result, _ = verify_claims(
        llm, draft_answer="Aetna requires prior authorization for CPT 97110 after two visits.", evidence_text="",
    )
    assert result.findings
    assert all(f.verdict == "insufficient_evidence" for f in result.findings)


def test_verify_claims_with_supporting_evidence_marks_supported():
    llm = MockBackend()
    evidence = "Aetna requires prior authorization for therapeutic exercise CPT 97110 beyond the first two visits."
    result, _ = verify_claims(
        llm,
        draft_answer="Aetna requires prior authorization for therapeutic exercise CPT 97110 beyond the first two visits.",
        evidence_text=evidence,
    )
    assert result.findings
    assert any(f.verdict == "supported" for f in result.findings)


def test_verify_claims_never_fabricates_contradicted_in_mock():
    # The mock is a keyword-overlap heuristic, not a real entailment model - it
    # can positively confirm support but should never claim to have detected a
    # contradiction, since that requires semantic judgment it doesn't have.
    llm = MockBackend()
    result, _ = verify_claims(
        llm, draft_answer="The sky is green and claims are denied for no reason.",
        evidence_text="Unrelated evidence about prior authorization timelines.",
    )
    assert all(f.verdict != "contradicted" for f in result.findings)
