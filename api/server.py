"""Production HTTP surface for the copilot.

Run with: uvicorn api.server:app --reload

Every endpoint requires an `X-API-Key` header resolved server-side to a fixed
(user_id, role) via `governance/identity.py` - `role` and `user_id` are never
accepted as client-supplied request fields, because a JSON field is just a
claim (anyone could send `"role": "admin"`) while a resolved identity is a
fact. See README's demo API keys to try this locally.
"""
import uuid

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from copilot.governance.approvals import ApprovalQueue
from copilot.governance.audit import AuditLog
from copilot.governance.identity import UnknownAPIKeyError, resolve_identity
from copilot.governance.rate_limit import RateLimiter
from copilot.graph import build_graph, resume_request, run_request
from copilot.rag.retriever import get_retriever

app = FastAPI(title="Healthcare Multi-Agent Copilot", version="0.1.0")

_audit = AuditLog()
_approvals = ApprovalQueue()
_graph = build_graph(audit=_audit, approvals=_approvals)
_rate_limiter = RateLimiter(max_requests=30, window_seconds=60.0)

# Loading the embedding model is a multi-second cold start (see
# data/observability_dashboard.html after any run). A long-lived server pays
# that cost once, here, rather than on whichever request happens to be first.
get_retriever()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Identity(BaseModel):
    user_id: str
    role: str


def require_identity(api_key: str | None = Security(_api_key_header)) -> Identity:
    if not api_key:
        raise HTTPException(status_code=401, detail="missing X-API-Key header")
    try:
        identity = resolve_identity(api_key)
    except UnknownAPIKeyError:
        raise HTTPException(status_code=401, detail="invalid API key")
    if not _rate_limiter.check(identity["user_id"]):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    return Identity(**identity)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    request_id: str
    status: str  # "completed" | "pending_approval" | "blocked"
    final_answer: str | None = None
    draft_answer: str | None = None
    risk: str | None = None
    issues: list[str] | None = None


class ApprovalDecisionRequest(BaseModel):
    approved: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, identity: Identity = Depends(require_identity)) -> ChatResponse:
    request_id = str(uuid.uuid4())
    _, result = run_request(_graph, query=req.query, user_id=identity.user_id, role=identity.role, request_id=request_id)

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
def decide_approval(
    request_id: str, req: ApprovalDecisionRequest, identity: Identity = Depends(require_identity),
) -> ChatResponse:
    pending = _approvals.get(request_id)
    if not pending or pending["status"] != "pending":
        raise HTTPException(status_code=404, detail="no pending approval for this request_id")

    # Separation of duties: whoever triggered a high-risk request cannot also
    # be the one who approves or rejects it, even if their role would
    # otherwise be allowed to review approvals.
    if pending.get("requester_user_id") and pending["requester_user_id"] == identity.user_id:
        raise HTTPException(status_code=403, detail="the requester cannot approve their own request")

    result = resume_request(_graph, request_id=request_id, approved=req.approved, approver=identity.user_id)
    return ChatResponse(request_id=request_id, status="completed", final_answer=result.get("final_answer"))


@app.get("/approvals")
def list_pending_approvals(identity: Identity = Depends(require_identity)) -> list[dict]:
    return _approvals.list_pending()


@app.get("/audit/verify")
def audit_verify(identity: Identity = Depends(require_identity)) -> dict:
    if identity.role != "admin":
        raise HTTPException(status_code=403, detail="admin only")
    is_valid, broken_entry_id = _audit.verify_chain()
    return {"valid": is_valid, "first_broken_entry_id": broken_entry_id}


@app.get("/audit/{request_id}")
def audit_trail(request_id: str, identity: Identity = Depends(require_identity)) -> list[dict]:
    return _audit.trail_for(request_id)
