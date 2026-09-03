from copilot.tools.analyze_denial import analyze_denial
from copilot.tools.calculate_denial_metrics import calculate_denial_metrics
from copilot.tools.claims_db import ClaimNotFoundError
from copilot.tools.create_remediation_plan import create_remediation_plan
from copilot.tools.query_claims import query_claims


def test_query_claims_filters_by_payer_and_procedure():
    result = query_claims(payer="Aetna", denial_code=None, procedure_code="97110",
                           status=None, start_date=None, end_date=None, limit=5)
    assert result["total_matching_count"] == 17
    assert result["returned_count"] == 5
    assert all(c["payer"] == "Aetna" and c["procedure_code"] == "97110" for c in result["claims"])


def test_query_claims_limit_is_capped():
    result = query_claims(payer=None, denial_code=None, procedure_code=None, status=None,
                           start_date=None, end_date=None, limit=9999)
    assert result["returned_count"] <= 20


def test_analyze_denial_known_claim():
    result = analyze_denial("CLM-000039")
    assert result["denial_code"] == "CO-197"
    assert result["is_appealable"] is True
    assert result["appeal_filed"] is False


def test_analyze_denial_unknown_claim_raises():
    try:
        analyze_denial("CLM-999999")
        raised = False
    except ClaimNotFoundError:
        raised = True
    assert raised


def test_denial_rate_matches_known_ground_truth():
    result = calculate_denial_metrics(metric="denial_rate", payer="Aetna", procedure_code=None,
                                       denial_code=None, start_date=None, end_date=None)
    assert result["total_claims"] == 122
    assert result["denied_claims"] == 30
    assert result["denial_rate_pct"] == 24.6


def test_top_denial_codes_for_united_healthcare():
    result = calculate_denial_metrics(metric="top_denial_codes", payer="UnitedHealthcare", procedure_code=None,
                                       denial_code=None, start_date=None, end_date=None)
    assert result["results"][0]["denial_code"] == "CO-29"
    assert result["results"][0]["count"] == 22


def test_overturn_rate_for_co197():
    result = calculate_denial_metrics(metric="overturn_rate", payer=None, procedure_code=None,
                                       denial_code="CO-197", start_date=None, end_date=None)
    assert result["appealed_claims"] == 5
    assert result["overturned_claims"] == 4
    assert result["overturn_rate_pct"] == 80.0


def test_remediation_plan_identifies_dominant_pattern():
    plan = create_remediation_plan(payer="Aetna", denial_code=None, procedure_code=None)
    assert plan["pattern_summary"]["affected_claim_count"] == 30
    assert plan["denial_code_breakdown"][0]["denial_code"] == "CO-197"
    assert plan["recommended_actions"]
    assert plan["policy_references"]
