import os
import sqlite3
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from copilot.agents import critic as critic_agent
from copilot.agents import executor as executor_agent
from copilot.agents import planner as planner_agent
from copilot.config import default_checkpoint_db_path
from copilot.governance import permissions
from copilot.governance.approvals import ApprovalQueue
from copilot.governance.audit import AuditLog
from copilot.guardrails.citation_check import verify_citations
from copilot.guardrails.input_guardrails import check_input
from copilot.guardrails.output_guardrails import scan_output
from copilot.llm import get_backend
from copilot.observability.tracing import Tracing
from copilot.state import CopilotState

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _default_checkpointer() -> SqliteSaver:
    """Persist graph state to disk (not just in-process memory) so a paused,
    awaiting-approval request survives across separate CLI invocations or an
    API server restart - this is what makes human-in-the-loop resumption
    actually usable outside of a single long-lived process."""
    path = default_checkpoint_db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
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
                user_id=state["user_id"], role=state["role"], payload=verdict,
            )
            return {"blocked": verdict["blocked"], "block_reason": verdict["reason"]}

    def route_after_input_guard(state: CopilotState) -> str:
        return "blocked" if state.get("blocked") else "plan"

    def node_blocked(state: CopilotState) -> dict:
        audit.log(
            request_id=state["request_id"], event_type="blocked",
            user_id=state["user_id"], role=state["role"], payload={"reason": state.get("block_reason")},
        )
        return {"final_answer": f"I can't help with that request. Reason: {state.get('block_reason')}"}

    def node_plan(state: CopilotState) -> dict:
        with tracing.span("node.plan", request_id=state["request_id"], role=state["role"]):
            result, used_model = planner_agent.plan(llm, state["query"])
            audit.log(
                request_id=state["request_id"], event_type="plan_created",
                user_id=state["user_id"], role=state["role"],
                payload={"task_type": result.task_type, "steps": result.steps, "model": used_model},
            )
            return {"task_type": result.task_type, "plan_steps": result.steps, "used_model": used_model}

    def node_execute(state: CopilotState) -> dict:
        with tracing.span("node.execute", request_id=state["request_id"], role=state["role"]):
            result = executor_agent.execute(
                llm, query=state["query"], role=state["role"], request_id=state["request_id"],
                user_id=state["user_id"], audit=audit, tracing=tracing,
            )
            audit.log(
                request_id=state["request_id"], event_type="draft_answer",
                user_id=state["user_id"], role=state["role"],
                payload={"model": result.used_model, "tools_used": result.tools_used},
            )
            return {
                "draft_answer": result.answer,
                "used_model": result.used_model,
                "tools_used": result.tools_used,
                "evidence_claim_ids": result.evidence_claim_ids,
                "evidence_doc_sources": result.evidence_doc_sources,
            }

    def node_critic(state: CopilotState) -> dict:
        with tracing.span("node.critic", request_id=state["request_id"], role=state["role"]):
            static_risk, static_issues = scan_output(state["draft_answer"])
            verdict, _ = critic_agent.review(llm, state["draft_answer"])
            citation_issues = verify_citations(
                state["draft_answer"],
                evidence_doc_sources=set(state.get("evidence_doc_sources", [])),
                evidence_claim_ids=set(state.get("evidence_claim_ids", [])),
            )
            # An unverified citation is itself at least a medium-risk finding - a
            # plausible-looking but ungrounded document or claim reference is
            # exactly the kind of thing that should stop for human review.
            risks = [static_risk, verdict.risk] + (["medium"] if citation_issues else [])
            final_risk = max(risks, key=lambda r: _RISK_ORDER[r])
            issues = list(dict.fromkeys(static_issues + verdict.issues + citation_issues))  # de-dupe, preserve order
            requires_approval = final_risk != "low" or verdict.requires_approval
            audit.log(
                request_id=state["request_id"], event_type="guardrail_verdict",
                user_id=state["user_id"], role=state["role"],
                payload={"risk": final_risk, "issues": issues, "requires_approval": requires_approval},
            )
            return {"guardrail_risk": final_risk, "guardrail_issues": issues, "requires_approval": requires_approval}

    def route_after_critic(state: CopilotState) -> str:
        if state.get("requires_approval") and not permissions.can_auto_approve(state["role"]):
            return "needs_approval"
        return "finalize"

    def node_request_approval(state: CopilotState) -> dict:
        # Runs exactly once, before the pausing node - `interrupt()` re-executes
        # everything earlier in *its own* node on resume, so any one-time side
        # effect (submitting the pending-approval record, logging the request)
        # has to live here instead, or it would be double-recorded on resume.
        approvals.submit(
            state["request_id"], summary=state["draft_answer"][:280],
            risk=state.get("guardrail_risk", "medium"), requester_user_id=state["user_id"],
        )
        audit.log(
            request_id=state["request_id"], event_type="approval_requested",
            user_id=state["user_id"], role=state["role"], payload={"risk": state.get("guardrail_risk")},
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
                user_id=state["user_id"], role=state["role"], payload={"approved": approved, "approver": approver},
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
            user_id=state["user_id"], role=state["role"], payload={"final_answer_preview": final[:280]},
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


def run_request(app, *, query: str, user_id: str, role: str, request_id: str | None = None) -> tuple[str, dict]:
    request_id = request_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": request_id}}
    initial_state = {"request_id": request_id, "user_id": user_id, "role": role, "query": query}
    result = app.invoke(initial_state, config=config)
    return request_id, result


def resume_request(app, *, request_id: str, approved: bool, approver: str) -> dict:
    config = {"configurable": {"thread_id": request_id}}
    return app.invoke(Command(resume={"approved": approved, "approver": approver}), config=config)
