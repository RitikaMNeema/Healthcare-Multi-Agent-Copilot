from dataclasses import dataclass


@dataclass(frozen=True)
class RolePolicy:
    allowed_tools: frozenset[str]
    auto_approve: bool


ROLE_POLICIES: dict[str, RolePolicy] = {
    "viewer": RolePolicy(allowed_tools=frozenset({"search_kb"}), auto_approve=False),
    "operator": RolePolicy(allowed_tools=frozenset({"search_kb", "calculator"}), auto_approve=False),
    "admin": RolePolicy(allowed_tools=frozenset({"search_kb", "calculator", "read_file"}), auto_approve=True),
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
