"""Production HTTP surface for the copilot.

Run with: uvicorn api.server:app --reload
"""
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from copilot.governance.approvals import ApprovalQueue
from copilot.governance.audit import AuditLog
from copilot.graph import build_graph, resume_request, run_request
from copilot.rag.retriever import get_retriever

app = FastAPI(title="Governed Multi-Agent Copilot", version="0.1.0")

_audit = AuditLog()
_approvals = ApprovalQueue()
_graph = build_graph(audit=_audit, approvals=_approvals)

# Loading the embedding model is a multi-second cold start (see
# data/observability_dashboard.html after any run - it shows up as the first
# search_payer_policy call). A long-lived server should pay that cost once at
# startup, not on whichever request happens to be first.
get_retriever()


class ChatRequest(BaseModel):
    query: str
    user_id: str
    role: str = "operator"


class ChatResponse(BaseModel):
    request_id: str
    status: str  # "completed" | "pending_approval" | "blocked"
    final_answer: str | None = None
    draft_answer: str | None = None
    risk: str | None = None
    issues: list[str] | None = None


class ApprovalDecisionRequest(BaseModel):
    approver: str
    approved: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    request_id = str(uuid.uuid4())
    _, result = run_request(_graph, query=req.query, user_id=req.user_id, role=req.role, request_id=request_id)

    if "__interrupt__" in result:
        return ChatResponse(
            request_id=request_id, status="pending_approval",
            draft_answer=result.get("draft_answer"), risk=result.get("guardrail_risk"),
            issues=result.get("guardrail_issues"),
        )
    if result.get("blocked"):
        return ChatResponse(request_id=request_id, status="blocked", final_answer=result.get("final_answer"))
    return ChatResponse(request_id=request_id, status="completed", final_answer=result.get("final_answer"))


@app.post("/approvals/{request_id}", response_model=ChatResponse)
def decide_approval(request_id: str, req: ApprovalDecisionRequest) -> ChatResponse:
    pending = _approvals.get(request_id)
    if not pending or pending["status"] != "pending":
        raise HTTPException(status_code=404, detail="no pending approval for this request_id")

    result = resume_request(_graph, request_id=request_id, approved=req.approved, approver=req.approver)
    return ChatResponse(request_id=request_id, status="completed", final_answer=result.get("final_answer"))


@app.get("/approvals")
def list_pending_approvals() -> list[dict]:
    return _approvals.list_pending()


@app.get("/audit/{request_id}")
def audit_trail(request_id: str) -> list[dict]:
    return _audit.trail_for(request_id)
