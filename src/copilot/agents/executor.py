from dataclasses import dataclass, field

from copilot.config import FALLBACK_MODELS, MAX_TOOL_ITERATIONS, PRIMARY_MODEL
from copilot.fallback import call_with_fallback
from copilot.tools import registry
from copilot.tools.registry import ToolPermissionDenied

SYSTEM_PROMPT = """You are the execution agent for a governed healthcare claims/denial-management copilot.

You have access to tools for policy search, claims lookup, denial analysis, aggregate
metrics, and (admin only) remediation planning. Choose the tool that matches what the
user actually needs - do not guess at claim data or policy facts you have not retrieved.
When citing a policy document in your answer, use the exact bracket form [filename.md].
When citing a specific claim, use its exact claim ID (e.g. CLM-000123). Never cite a
document or claim ID you did not actually receive from a tool result this turn. When you
have enough information, respond with a final text answer and stop calling tools. If a
tool call is denied for permission reasons, explain the limitation instead of retrying it."""


# The one tool each task_type is planned around, for the small set of task
# types that map cleanly to a single tool - used only to observe and audit
# whether the executor actually followed the plan, not to hard-restrict its
# tool choice (a real request can legitimately need a different tool than
# planned, e.g. a permission degradation - see plan_followed below).
_TASK_TYPE_EXPECTED_TOOL = {
    "policy_lookup": "search_payer_policy",
    "claims_query": "query_claims",
    "denial_analysis": "analyze_denial",
    "metrics": "calculate_denial_metrics",
    "remediation": "create_remediation_plan",
}


@dataclass
class ExecutionResult:
    answer: str
    used_model: str
    tools_used: list[str] = field(default_factory=list)
    evidence_claim_ids: list[str] = field(default_factory=list)
    evidence_doc_sources: list[str] = field(default_factory=list)
    plan_followed: bool | None = None  # None when task_type has no single expected tool (e.g. "general")


def execute(llm, *, query: str, role: str, request_id: str, user_id: str, audit, tracing=None,
            task_type: str | None = None, plan_steps: list[str] | None = None) -> ExecutionResult:
    tool_specs = registry.tool_definitions_for_role(role)

    # The plan is real input to this turn, not just something the planner
    # computed and the executor never sees - it's stated as context, then
    # `plan_followed` records (for audit/tests) whether the model actually
    # used the tool that plan called for.
    if task_type:
        steps_text = "; ".join(plan_steps or [])
        content = f"Planned approach (task_type={task_type}): {steps_text}\n\nRequest: {query}"
    else:
        content = query
    messages: list[dict] = [{"role": "user", "content": content}]
    result = ExecutionResult(answer="", used_model=PRIMARY_MODEL)

    for _ in range(MAX_TOOL_ITERATIONS):
        def attempt(model: str, _messages=messages):
            return llm.complete_with_tools(system=SYSTEM_PROMPT, messages=_messages, tools=tool_specs, model=model)

        response, used_model = call_with_fallback(attempt, models=[PRIMARY_MODEL, *FALLBACK_MODELS])
        result.used_model = used_model
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        if not tool_uses:
            result.answer = next((block.text for block in response.content if block.type == "text"), "")
            _set_plan_followed(result, task_type)
            return result

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_use in tool_uses:
            result.tools_used.append(tool_use.name)
            try:
                tool_result = registry.invoke_tool(
                    tool_use.name, tool_use.input,
                    role=role, user_id=user_id, request_id=request_id, audit=audit, tracing=tracing,
                )
                _record_evidence(result, tool_use.name, tool_result)
                tool_results.append({"type": "tool_result", "tool_use_id": tool_use.id, "content": str(tool_result)})
            except ToolPermissionDenied as exc:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tool_use.id, "content": str(exc), "is_error": True,
                })
            except Exception as exc:  # noqa: BLE001 - surface any tool failure back to the model as an error result
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tool_use.id, "content": f"tool error: {exc}", "is_error": True,
                })
        messages.append({"role": "user", "content": tool_results})

    result.answer = "I was unable to complete this within the allowed tool-call budget."
    _set_plan_followed(result, task_type)
    return result


def _set_plan_followed(result: ExecutionResult, task_type: str | None) -> None:
    expected_tool = _TASK_TYPE_EXPECTED_TOOL.get(task_type or "")
    if expected_tool is not None:
        result.plan_followed = expected_tool in result.tools_used


def _record_evidence(result: ExecutionResult, tool_name: str, tool_result: object) -> None:
    if tool_name in registry.CLAIM_LEVEL_TOOLS:
        result.evidence_claim_ids.extend(registry.extract_claim_ids(tool_name, tool_result))
    if tool_name == "search_payer_policy" and isinstance(tool_result, list):
        result.evidence_doc_sources.extend(hit["source"] for hit in tool_result if "source" in hit)
    if tool_name == "create_remediation_plan" and isinstance(tool_result, dict):
        result.evidence_doc_sources.extend(tool_result.get("policy_references", []))
