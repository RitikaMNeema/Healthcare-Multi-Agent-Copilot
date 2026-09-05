from copilot.guardrails.claim_verification import (
    ClaimFinding,
    ClaimVerificationResult,
    has_coverage_gap,
    summarize,
    verify_claims,
)
from copilot.llm import MockBackend


def test_summarize_buckets_by_verdict():
    result = ClaimVerificationResult(findings=[
        ClaimFinding(claim="a", verdict="supported", rationale="r", evidence_refs=["source.md"]),
        ClaimFinding(claim="b", verdict="contradicted", rationale="r"),
        ClaimFinding(claim="c", verdict="insufficient_evidence", rationale="r"),
    ])
    summary = summarize(result)
    assert summary["supported_claims"] == [{"claim": "a", "evidence_refs": ["source.md"]}]
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


def test_mock_attaches_evidence_ref_to_supported_claims():
    llm = MockBackend()
    evidence = "[prior_authorization_policy.md]: Aetna requires prior authorization for CPT 97110 after two visits."
    result, _ = verify_claims(
        llm, draft_answer="Aetna requires prior authorization for CPT 97110 after two visits.",
        evidence_text=evidence,
    )
    supported = [f for f in result.findings if f.verdict == "supported"]
    assert supported
    assert supported[0].evidence_refs == ["prior_authorization_policy.md"]


def test_has_coverage_gap_true_for_substantive_answer_with_zero_findings():
    # The deterministic floor check: a real answer but the verifier reported
    # nothing at all - that's the verifier failing to engage, not "all clean".
    empty_result = ClaimVerificationResult(findings=[])
    assert has_coverage_gap("Aetna requires prior authorization for CPT 97110 after two visits.", empty_result) is True


def test_has_coverage_gap_false_when_findings_present():
    result = ClaimVerificationResult(findings=[ClaimFinding(claim="x", verdict="supported", rationale="ok")])
    assert has_coverage_gap("Aetna requires prior authorization for CPT 97110.", result) is False


def test_has_coverage_gap_false_for_trivial_non_factual_answer():
    # A short greeting/refusal with no findings is correctly empty, not a gap -
    # there's nothing factual in it to have skipped verifying.
    empty_result = ClaimVerificationResult(findings=[])
    assert has_coverage_gap("Hi! I can help with that.", empty_result) is False


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
