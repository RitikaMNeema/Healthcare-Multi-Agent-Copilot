from dataclasses import dataclass


@dataclass(frozen=True)
class RolePolicy:
    allowed_tools: frozenset[str]
    auto_approve: bool


ROLE_POLICIES: dict[str, RolePolicy] = {
    # HIPAA minimum-necessary tiers: viewers get policy text and aggregate metrics
    # only (no individual, PHI-adjacent claim records); operators can look up
    # individual claims/denials for casework; admins can additionally generate
    # cross-payer remediation plans and auto-approve their own high-risk requests.
    "viewer": RolePolicy(allowed_tools=frozenset({"search_payer_policy", "calculate_denial_metrics"}), auto_approve=False),
    "operator": RolePolicy(
        allowed_tools=frozenset({"search_payer_policy", "calculate_denial_metrics", "query_claims", "analyze_denial"}),
        auto_approve=False,
    ),
    "admin": RolePolicy(
        allowed_tools=frozenset({
            "search_payer_policy", "calculate_denial_metrics", "query_claims", "analyze_denial", "create_remediation_plan",
        }),
        auto_approve=True,
    ),
}


class UnknownRoleError(Exception):
    pass


def policy_for(role: str) -> RolePolicy:
    try:
        return ROLE_POLICIES[role]
    except KeyError:
        raise UnknownRoleError(f"unknown role: '{role}'. Known roles: {sorted(ROLE_POLICIES)}") from None


def allowed_tools(role: str) -> frozenset[str]:
    return policy_for(role).allowed_tools


def can_auto_approve(role: str) -> bool:
    return policy_for(role).auto_approve
