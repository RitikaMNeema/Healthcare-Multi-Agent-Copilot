from dataclasses import dataclass


@dataclass(frozen=True)
class RolePolicy:
    allowed_tools: frozenset[str]
    can_review_approvals: bool


ROLE_POLICIES: dict[str, RolePolicy] = {
    # HIPAA minimum-necessary tiers for tool access, and a *separate* axis for
    # who may review a pending approval. These used to be conflated (a role
    # flag let admins auto-approve their own high-risk requests) - they no
    # longer are: no role can approve its own request (see graph.py's
    # separation-of-duties check), and no role bypasses review for anything
    # above low risk (see route_after_critic) - can_review_approvals only
    # controls who is *eligible* to review someone else's request.
    "viewer": RolePolicy(
        allowed_tools=frozenset({"search_payer_policy", "calculate_denial_metrics"}),
        can_review_approvals=False,
    ),
    "operator": RolePolicy(
        allowed_tools=frozenset({"search_payer_policy", "calculate_denial_metrics", "query_claims", "analyze_denial"}),
        can_review_approvals=False,
    ),
    "admin": RolePolicy(
        allowed_tools=frozenset({
            "search_payer_policy", "calculate_denial_metrics", "query_claims", "analyze_denial", "create_remediation_plan",
        }),
        can_review_approvals=True,
    ),
    # Reviewers whose job is approvals/audit oversight - no claims-tool access
    # of their own, by design (a reviewer doesn't need casework tools to
    # review someone else's request, and not having them removes any
    # temptation to "just check the claim myself" outside the review flow).
    "compliance_officer": RolePolicy(
        allowed_tools=frozenset({"search_payer_policy"}),
        can_review_approvals=True,
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


def can_review_approvals(role: str) -> bool:
    return policy_for(role).can_review_approvals
