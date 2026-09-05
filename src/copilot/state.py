from typing import TypedDict


class CopilotState(TypedDict, total=False):
    request_id: str
    user_id: str
    role: str
    tenant_id: str
    query: str

    blocked: bool
    block_reason: str

    task_type: str
    plan_steps: list[str]

    draft_answer: str
    used_model: str
    tools_used: list[str]
    evidence_claim_ids: list[str]
    evidence_doc_sources: list[str]
    evidence_text: list[str]
    plan_followed: bool | None

    guardrail_risk: str
    guardrail_issues: list[str]
    requires_approval: bool
    supported_claims: list[str]
    unsupported_claims: list[str]
    policy_violations: list[str]
    recommended_action: str
    revision_count: int

    approval_status: str
    final_answer: str
