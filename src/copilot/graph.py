import os
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from copilot.agents import critic as critic_agent
from copilot.agents import executor as executor_agent
from copilot.agents import planner as planner_agent
from copilot.config import default_checkpoint_db_path
from copilot.governance.approvals import ApprovalQueue
from copilot.governance.audit import AuditLog
from copilot.governance.redaction import redact_text
from copilot.guardrails import claim_verification
from copilot.guardrails.citation_check import verify_citations
from copilot.guardrails.input_guardrails import check_input
from copilot.guardrails.output_guardrails import scan_output
from copilot.llm import get_backend
from copilot.observability.tracing import Tracing
from copilot.sqlite_utils import connect as sqlite_connect
from copilot.state import CopilotState

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _default_checkpointer() -> SqliteSaver:
    """Persist graph state to disk (not just in-process memory) so a paused,
    awaiting-approval request survives across separate CLI invocations or an
    API server restart - this is what makes human-in-the-loop resumption
    actually usable outside of a single long-lived process."""
    path = default_checkpoint_db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite_connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def build_graph(
    llm=None,
    audit: AuditLog | None = None,
    approvals: ApprovalQueue | None = None,
    checkpointer=None,
    tracing: Tracing | None = None,
):
    tracing = tracing or Tracing()
    llm = llm or get_backend(tracing=tracing)
    audit = audit or AuditLog()
    approvals = approvals or ApprovalQueue()
    checkpointer = checkpointer or _default_checkpointer()

    def node_input_guard(state: CopilotState) -> dict:
        with tracing.span("node.input_guard", request_id=state["request_id"], role=state["role"]):
            verdict = check_input(state["query"])
            audit.log(
                request_id=state["request_id"], event_type="input_guardrail",
                user_id=state["user_id"], role=state["role"], tenant_id=state.get("tenant_id"), payload=verdict,
            )
            return {"blocked": verdict["blocked"], "block_reason": verdict["reason"]}

    def route_after_input_guard(state: CopilotState) -> str:
        return "blocked" if state.get("blocked") else "plan"

    def node_blocked(state: CopilotState) -> dict:
        audit.log(
            request_id=state["request_id"], event_type="blocked",
            user_id=state["user_id"], role=state["role"], tenant_id=state.get("tenant_id"),
            payload={"reason": state.get("block_reason")},
        )
        return {"final_answer": f"I can't help with that request. Reason: {state.get('block_reason')}"}

    def node_plan(state: CopilotState) -> dict:
        with tracing.span("node.plan", request_id=state["request_id"], role=state["role"]):
            result, used_model = planner_agent.plan(llm, state["query"])
            audit.log(
                request_id=state["request_id"], event_type="plan_created",
                user_id=state["user_id"], role=state["role"], tenant_id=state.get("tenant_id"),
                payload={"task_type": result.task_type, "steps": result.steps, "model": used_model},
            )
            return {"task_type": result.task_type, "plan_steps": result.steps, "used_model": used_model}

    def node_execute(state: CopilotState) -> dict:
        with tracing.span("node.execute", request_id=state["request_id"], role=state["role"]):
            result = executor_agent.execute(
                llm, query=state["query"], role=state["role"], request_id=state["request_id"],
                user_id=state["user_id"], audit=audit, tracing=tracing,
                task_type=state.get("task_type"), plan_steps=state.get("plan_steps"),
            )
            audit.log(
                request_id=state["request_id"], event_type="draft_answer",
                user_id=state["user_id"], role=state["role"], tenant_id=state.get("tenant_id"),
                payload={"model": result.used_model, "tools_used": result.tools_used, "plan_followed": result.plan_followed},
            )
            return {
                "draft_answer": result.answer,
                "used_model": result.used_model,
                "tools_used": result.tools_used,
                "evidence_claim_ids": result.evidence_claim_ids,
                "evidence_doc_sources": result.evidence_doc_sources,
                "evidence_text": result.evidence_text,
                "plan_followed": result.plan_followed,
            }

    def node_critic(state: CopilotState) -> dict:
        with tracing.span("node.critic", request_id=state["request_id"], role=state["role"]):
            evidence_text = "\n\n".join(state.get("evidence_text", []))

            static_risk, static_issues = scan_output(state["draft_answer"])
            claim_result, _ = claim_verification.verify_claims(
                llm, draft_answer=state["draft_answer"], evidence_text=evidence_text,
            )
            claim_summary = claim_verification.summarize(claim_result)
            unsupported_claims = claim_summary["contradicted_claims"] + claim_summary["insufficient_evidence_claims"]

            verdict, _ = critic_agent.review(
                llm, question=state["query"], task_type=state.get("task_type"), plan_steps=state.get("plan_steps"),
                evidence_text=evidence_text, draft_answer=state["draft_answer"], claim_result=claim_result,
            )
            citation_issues = verify_citations(
                state["draft_answer"],
                evidence_doc_sources=set(state.get("evidence_doc_sources", [])),
                evidence_claim_ids=set(state.get("evidence_claim_ids", [])),
            )
            # An unverified citation, an unsupported claim, or a critic-recommended
            # block are each independently at least a medium-risk finding (block
            # forces high) - a plausible-looking but ungrounded answer is exactly
            # the kind of thing that should stop for human review even if the
            # critic's own `risk` field under-called it.
            risks = [static_risk, verdict.risk]
            if citation_issues or unsupported_claims:
                risks.append("medium")
            if verdict.recommended_action == "block":
                risks.append("high")
            final_risk = max(risks, key=lambda r: _RISK_ORDER[r])
            issues = list(dict.fromkeys(
                static_issues + verdict.issues + citation_issues
                + [f"unsupported claim: {c}" for c in unsupported_claims],
            ))  # de-dupe, preserve order
            requires_approval = final_risk != "low" or verdict.requires_approval
            # Issues/claims can quote fragments of the draft answer (e.g. "contains
            # banned phrase: ...") - redact before any of this reaches the audit
            # log. Graph *state* (returned below) deliberately keeps the raw text,
            # consistent with draft_answer/evidence_text already living there
            # unredacted - state is what an authorized reviewer reads in full via
            # the approval-detail endpoint, whereas the audit log is a durable,
            # more broadly-queryable record that data minimization applies to.
            audit.log(
                request_id=state["request_id"], event_type="guardrail_verdict",
                user_id=state["user_id"], role=state["role"], tenant_id=state.get("tenant_id"),
                payload={
                    "risk": final_risk, "issue_count": len(issues), "issues": [redact_text(i) for i in issues],
                    "requires_approval": requires_approval, "recommended_action": verdict.recommended_action,
                    "policy_violations": [redact_text(v) for v in verdict.policy_violations],
                    "supported_claim_count": len(claim_summary["supported_claims"]),
                    "unsupported_claim_count": len(unsupported_claims),
                },
            )
            return {
                "guardrail_risk": final_risk, "guardrail_issues": issues, "requires_approval": requires_approval,
                "supported_claims": claim_summary["supported_claims"], "unsupported_claims": unsupported_claims,
                "policy_violations": verdict.policy_violations, "recommended_action": verdict.recommended_action,
            }

    def route_after_critic(state: CopilotState) -> str:
        # No role bypasses review above low risk - not even admin. Auto-release
        # is reserved for content the guardrails classified as low risk;
        # everything else needs a reviewer who isn't the requester (enforced
        # in api/server.py, since that's where identity is actually verified).
        return "needs_approval" if state.get("requires_approval") else "finalize"

    def node_request_approval(state: CopilotState) -> dict:
        # Runs exactly once, before the pausing node - `interrupt()` re-executes
        # everything earlier in *its own* node on resume, so any one-time side
        # effect (submitting the pending-approval record, logging the request)
        # has to live here instead, or it would be double-recorded on resume.
        #
        # The stored summary is sanitized - never the draft answer or patient
        # data - since a reviewer's queue listing shouldn't leak sensitive
        # content to anyone who can merely list pending items. A reviewer who
        # is actually authorized to decide sees the real draft through the
        # graph's own checkpoint state (api/server.py's approval-detail
        # endpoint), not through this table.
        sanitized_summary = (
            f"{state.get('task_type', 'request')} via {', '.join(state.get('tools_used', []) or ['no tools'])} "
            f"- risk={state.get('guardrail_risk')}, {len(state.get('guardrail_issues', []))} guardrail issue(s)"
        )
        approvals.submit(
            state["request_id"], summary=sanitized_summary,
            risk=state.get("guardrail_risk", "medium"), requester_user_id=state["user_id"],
            tenant_id=state.get("tenant_id"),
        )
        audit.log(
            request_id=state["request_id"], event_type="approval_requested",
            user_id=state["user_id"], role=state["role"], tenant_id=state.get("tenant_id"),
            payload={"risk": state.get("guardrail_risk")},
        )
        return {}

    def node_await_approval(state: CopilotState) -> dict:
        with tracing.span("node.await_approval", request_id=state["request_id"], role=state["role"]):
            decision = interrupt({
                "request_id": state["request_id"],
                "draft_answer": state["draft_answer"],
                "risk": state.get("guardrail_risk"),
                "issues": state.get("guardrail_issues"),
            })
            approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
            approver = decision.get("approver") if isinstance(decision, dict) else None

            approvals.decide(state["request_id"], approver=approver, approved=approved)
            audit.log(
                request_id=state["request_id"], event_type="approval_decided",
                user_id=state["user_id"], role=state["role"], tenant_id=state.get("tenant_id"),
                payload={"approved": approved, "approver": approver},
            )

            if approved:
                return {"approval_status": "approved", "final_answer": state["draft_answer"]}
            return {
                "approval_status": "rejected",
                "final_answer": "This response was blocked by a human reviewer and cannot be released.",
            }

    def node_finalize(state: CopilotState) -> dict:
        final = state.get("final_answer") or state.get("draft_answer", "")
        audit.log(
            request_id=state["request_id"], event_type="finalized",
            user_id=state["user_id"], role=state["role"], tenant_id=state.get("tenant_id"),
            payload={"final_answer_length": len(final)},
        )
        return {"final_answer": final}

    graph = StateGraph(CopilotState)
    graph.add_node("input_guard", node_input_guard)
    graph.add_node("blocked", node_blocked)
    graph.add_node("plan", node_plan)
    graph.add_node("execute", node_execute)
    graph.add_node("critic", node_critic)
    graph.add_node("request_approval", node_request_approval)
    graph.add_node("await_approval", node_await_approval)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("input_guard")
    graph.add_conditional_edges("input_guard", route_after_input_guard, {"blocked": "blocked", "plan": "plan"})
    graph.add_edge("blocked", END)
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"needs_approval": "request_approval", "finalize": "finalize"})
    graph.add_edge("request_approval", "await_approval")
    graph.add_edge("await_approval", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def run_request(app, *, query: str, user_id: str, role: str, request_id: str | None = None,
                 tenant_id: str | None = None) -> tuple[str, dict]:
    request_id = request_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": request_id}}
    initial_state = {
        "request_id": request_id, "user_id": user_id, "role": role, "tenant_id": tenant_id, "query": query,
    }
    result = app.invoke(initial_state, config=config)
    return request_id, result


def resume_request(app, *, request_id: str, approved: bool, approver: str) -> dict:
    config = {"configurable": {"thread_id": request_id}}
    return app.invoke(Command(resume={"approved": approved, "approver": approver}), config=config)
