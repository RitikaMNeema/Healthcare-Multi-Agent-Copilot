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
    plan_needs_retrieval: bool

    retrieved_context: list[dict]

    draft_answer: str
    used_model: str

    guardrail_risk: str
    guardrail_issues: list[str]
    requires_approval: bool

    approval_status: str
    final_answer: str
