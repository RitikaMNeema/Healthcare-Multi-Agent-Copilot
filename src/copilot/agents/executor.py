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
    evidence_text: list[str] = field(default_factory=list)
    plan_followed: bool | None = None  # None when task_type has no single expected tool (e.g. "general")


def execute(llm, *, query: str, role: str, request_id: str, user_id: str, audit, tracing=None,
            task_type: str | None = None, plan_steps: list[str] | None = None) -> ExecutionResult:
    tool_specs = registry.tool_definitions_for_role(role)

    # The plan is real input to this turn, not just something the planner
    # computed and the executor never sees - it's stated as context, then
    # `plan_followed` records (for audit/tests) whether the model actually
    # used the tool that plan called for. This goes in the *system* prompt,
    # never prepended to the user message: a tool-calling model (and the mock
    # backend's naive text-extraction) can echo the literal user-message text
    # into a tool call's parameters - e.g. `search_payer_policy`'s `query` -
    # so plan-steering text living there would pollute retrieval with
    # unrelated terms instead of just steering which tool gets picked.
    system = SYSTEM_PROMPT
    if task_type:
        steps_text = "; ".join(plan_steps or [])
        system = f"{SYSTEM_PROMPT}\n\nPlanned approach for this request (task_type={task_type}): {steps_text}"
    messages: list[dict] = [{"role": "user", "content": query}]
    result = ExecutionResult(answer="", used_model=PRIMARY_MODEL)

    for _ in range(MAX_TOOL_ITERATIONS):
        def attempt(model: str, _messages=messages):
            return llm.complete_with_tools(system=system, messages=_messages, tools=tool_specs, model=model)

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

    snippet = _evidence_text_for(tool_name, tool_result)
    if snippet:
        result.evidence_text.append(snippet)


# The model already saw this exact content as a tool result this same turn
# (it's what the answer was generated from) - rendering it again here for the
# critic/claim-verifier isn't a new disclosure, just a second internal LLM
# call over the same information within the same request.
def _evidence_text_for(tool_name: str, tool_result: object) -> str | None:
    if tool_name == "search_payer_policy" and isinstance(tool_result, list):
        return "\n".join(f"[{hit['source']}]: {hit['text']}" for hit in tool_result if "source" in hit)

    if tool_name == "analyze_denial" and isinstance(tool_result, dict):
        if tool_result.get("denial_code") is None:
            return f"Claim {tool_result.get('claim_id')}: status={tool_result.get('status')}, not denied."
        return (
            f"Claim {tool_result.get('claim_id')}: payer={tool_result.get('payer')}, "
            f"procedure={tool_result.get('procedure_code')}, denial_code={tool_result.get('denial_code')} "
            f"({tool_result.get('denial_code_meaning')}), appealable={tool_result.get('is_appealable')}, "
            f"appeal_filed={tool_result.get('appeal_filed')}, appeal_outcome={tool_result.get('appeal_outcome')}, "
            f"recommended_actions={tool_result.get('recommended_actions')}"
        )

    if tool_name == "query_claims" and isinstance(tool_result, dict):
        rows = tool_result.get("claims", [])
        row_desc = "; ".join(
            f"{c.get('claim_id')} (status={c.get('status')}, denial_code={c.get('denial_code')})" for c in rows
        )
        return (
            f"query_claims: {tool_result.get('total_matching_count')} matching, "
            f"showing {tool_result.get('returned_count')}: {row_desc}"
        )

    if isinstance(tool_result, (dict, list)):
        return f"{tool_name}: {str(tool_result)[:800]}"
    return None


REVISE_SYSTEM_PROMPT = """You are revising a draft answer from a governed healthcare
claims/denial-management copilot, in response to specific problems a safety critic found.
Produce a corrected answer that fixes every listed problem while staying strictly grounded
in the evidence given - do not introduce any new claim the evidence doesn't support, and do
not re-word around a problem instead of actually fixing it. If a problem can't be fixed while
staying grounded in the evidence (e.g. the question simply can't be answered from what was
retrieved), say so plainly rather than fabricating a fix. Respond with the revised answer text
only - no preamble, no explanation of what you changed."""


def revise(llm, *, query: str, evidence_text: str, draft_answer: str, issues: list[str],
           policy_violations: list[str]) -> str:
    """A single, targeted rewrite pass - not a re-run of the tool-calling
    loop. The evidence is already retrieved; what needs fixing is how the
    answer uses it, not what was looked up. `graph.py`'s node_revise sends
    the result back through the full critic pipeline (including claim
    verification) rather than trusting the revision blindly."""
    problems = "\n".join(f"- {p}" for p in [*issues, *policy_violations]) or "(no specific problems listed)"
    content = (
        f"Original question:\n{query}\n\n"
        f"Evidence retrieved this turn:\n{evidence_text or '(no evidence was retrieved this turn)'}\n\n"
        f"Draft answer to revise:\n{draft_answer}\n\n"
        f"Problems to fix:\n{problems}"
    )

    def attempt(model: str) -> str:
        response = llm.complete_with_tools(
            system=REVISE_SYSTEM_PROMPT, messages=[{"role": "user", "content": content}], tools=[], model=model,
        )
        return next((block.text for block in response.content if block.type == "text"), draft_answer)

    revised, _ = call_with_fallback(attempt, models=[PRIMARY_MODEL, *FALLBACK_MODELS])
    return revised
