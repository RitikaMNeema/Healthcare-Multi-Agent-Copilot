import pytest

from copilot.governance import permissions
from copilot.governance.audit import AuditLog
from copilot.tools import registry


def test_viewer_cannot_query_claims():
    audit = AuditLog()
    with pytest.raises(registry.ToolPermissionDenied):
        registry.invoke_tool(
            "query_claims",
            {"payer": "Aetna", "denial_code": None, "procedure_code": None, "status": None,
             "start_date": None, "end_date": None, "limit": 5},
            role="viewer", user_id="u1", request_id="r1", audit=audit,
        )


def test_operator_cannot_create_remediation_plan():
    audit = AuditLog()
    with pytest.raises(registry.ToolPermissionDenied):
        registry.invoke_tool(
            "create_remediation_plan", {"payer": "Aetna", "denial_code": None, "procedure_code": None},
            role="operator", user_id="u1", request_id="r2", audit=audit,
        )


def test_denied_tool_call_is_audited():
    audit = AuditLog()
    with pytest.raises(registry.ToolPermissionDenied):
        registry.invoke_tool(
            "analyze_denial", {"claim_id": "CLM-000039"}, role="viewer", user_id="u1", request_id="r-denied", audit=audit,
        )
    trail = audit.trail_for("r-denied")
    assert any(event["event_type"] == "tool_denied" for event in trail)


def test_admin_can_use_all_tools():
    assert permissions.allowed_tools("admin") == {
        "search_payer_policy", "calculate_denial_metrics", "query_claims", "analyze_denial", "create_remediation_plan",
    }


def test_viewer_gets_policy_and_metrics_only():
    assert permissions.allowed_tools("viewer") == {"search_payer_policy", "calculate_denial_metrics"}


def test_unknown_role_raises():
    with pytest.raises(permissions.UnknownRoleError):
        permissions.allowed_tools("superuser")


def test_only_admin_and_compliance_officer_can_review_approvals():
    assert permissions.can_review_approvals("admin") is True
    assert permissions.can_review_approvals("compliance_officer") is True
    assert permissions.can_review_approvals("operator") is False
    assert permissions.can_review_approvals("viewer") is False


def test_compliance_officer_has_minimal_tool_access():
    assert permissions.allowed_tools("compliance_officer") == {"search_payer_policy"}


def test_tool_definitions_for_role_filters_correctly():
    viewer_tools = {spec["name"] for spec in registry.tool_definitions_for_role("viewer")}
    operator_tools = {spec["name"] for spec in registry.tool_definitions_for_role("operator")}
    admin_tools = {spec["name"] for spec in registry.tool_definitions_for_role("admin")}
    assert viewer_tools == {"search_payer_policy", "calculate_denial_metrics"}
    assert operator_tools == {"search_payer_policy", "calculate_denial_metrics", "query_claims", "analyze_denial"}
    assert admin_tools == {
        "search_payer_policy", "calculate_denial_metrics", "query_claims", "analyze_denial", "create_remediation_plan",
    }
