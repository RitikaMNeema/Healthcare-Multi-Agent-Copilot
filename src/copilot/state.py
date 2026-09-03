from typing import TypedDict


class CopilotState(TypedDict, total=False):
    request_id: str
    user_id: str
    role: str
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

    guardrail_risk: str
    guardrail_issues: list[str]
    requires_approval: bool

    approval_status: str
    final_answer: str
