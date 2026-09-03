import pytest

from copilot.governance import permissions
from copilot.governance.audit import AuditLog
from copilot.tools import registry


def test_viewer_cannot_use_calculator():
    audit = AuditLog()
    with pytest.raises(registry.ToolPermissionDenied):
        registry.invoke_tool(
            "calculator", {"expression": "1+1"}, role="viewer", user_id="u1", request_id="r1", audit=audit,
        )


def test_denied_tool_call_is_audited():
    audit = AuditLog()
    with pytest.raises(registry.ToolPermissionDenied):
        registry.invoke_tool(
            "read_file", {"filename": "faq.md"}, role="operator", user_id="u1", request_id="r-denied", audit=audit,
        )
    trail = audit.trail_for("r-denied")
    assert any(event["event_type"] == "tool_denied" for event in trail)


def test_admin_can_use_all_tools():
    assert permissions.allowed_tools("admin") >= {"search_kb", "calculator", "read_file"}


def test_viewer_only_has_search():
    assert permissions.allowed_tools("viewer") == {"search_kb"}


def test_unknown_role_raises():
    with pytest.raises(permissions.UnknownRoleError):
        permissions.allowed_tools("superuser")


def test_only_admin_auto_approves():
    assert permissions.can_auto_approve("admin") is True
    assert permissions.can_auto_approve("operator") is False
    assert permissions.can_auto_approve("viewer") is False


def test_tool_definitions_for_role_filters_correctly():
    from copilot.tools.registry import tool_definitions_for_role

    viewer_tools = {spec["name"] for spec in tool_definitions_for_role("viewer")}
    admin_tools = {spec["name"] for spec in tool_definitions_for_role("admin")}
    assert viewer_tools == {"search_kb"}
    assert admin_tools == {"search_kb", "calculator", "read_file"}
