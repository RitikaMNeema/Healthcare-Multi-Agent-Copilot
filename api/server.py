"""Production HTTP surface for the copilot.

Run with: uvicorn api.server:app --reload

Every endpoint requires an `X-API-Key` header resolved server-side to a fixed
(user_id, role, tenant_id) via `governance/identity.py` - none of those are
ever accepted as client-supplied request fields, because a JSON field is
just a claim (anyone could send `"role": "admin"`) while a resolved identity
is a fact. See README's demo API keys to try this locally.
"""
import uuid

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from copilot.governance import permissions
from copilot.governance.approvals import AlreadyDecidedError, ApprovalQueue
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
    tenant_id: str


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


def require_reviewer(identity: Identity = Depends(require_identity)) -> Identity:
    """Gate for anything approval-related: only a role whose job is review
    (admin, compliance_officer) may even see the pending queue, let alone
    decide on it - a viewer/operator/requester gets 403, not just a
    same-identity check at decision time."""
    if not permissions.can_review_approvals(identity.role):
        raise HTTPException(status_code=403, detail="role is not authorized to review approvals")
    return identity


def require_audit_access(identity: Identity = Depends(require_identity)) -> Identity:
    if identity.role not in ("admin", "compliance_officer"):
        raise HTTPException(status_code=403, detail="role is not authorized to browse audit records")
    return identity


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    request_id: str
    status: str  # "completed" | "pending_approval" | "blocked"
    final_answer: str | None = None
    risk: str | None = None


class ApprovalDecisionRequest(BaseModel):
    approved: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, identity: Identity = Depends(require_identity)) -> ChatResponse:
    request_id = str(uuid.uuid4())
    _, result = run_request(
        _graph, query=req.query, user_id=identity.user_id, role=identity.role,
        tenant_id=identity.tenant_id, request_id=request_id,
    )

    if "__interrupt__" in result:
        # Deliberately no draft_answer/issues here - the requester sees only
        # that their request is pending, not its content (data minimization
        # applies to *them* too, not just other tenants/roles). A reviewer
        # reads the actual content through GET /approvals/{id}/detail, which
        # is access-controlled and logged separately.
        return ChatResponse(request_id=request_id, status="pending_approval", risk=result.get("guardrail_risk"))
    if result.get("blocked") or result.get("approval_status") == "blocked_by_critic":
        return ChatResponse(request_id=request_id, status="blocked", final_answer=result.get("final_answer"))
    return ChatResponse(request_id=request_id, status="completed", final_answer=result.get("final_answer"))


@app.get("/chat/{request_id}", response_model=ChatResponse)
def chat_status(request_id: str, identity: Identity = Depends(require_identity)) -> ChatResponse:
    """Lets the original requester (or a reviewer) poll a request that was
    held for approval - separate from the reviewer-only detail endpoint
    because the requester should see their own eventual answer without
    needing review permissions, but not the queue of *other* people's
    pending requests."""
    config = {"configurable": {"thread_id": request_id}}
    snapshot = _graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="no such request")

    is_owner = snapshot.values.get("user_id") == identity.user_id
    same_tenant = snapshot.values.get("tenant_id") == identity.tenant_id
    is_reviewer = permissions.can_review_approvals(identity.role)
    if not same_tenant or not (is_owner or is_reviewer):
        raise HTTPException(status_code=404, detail="no such request")

    if snapshot.next:  # graph still paused (e.g. awaiting approval)
        return ChatResponse(request_id=request_id, status="pending_approval", risk=snapshot.values.get("guardrail_risk"))
    if snapshot.values.get("blocked") or snapshot.values.get("approval_status") == "blocked_by_critic":
        return ChatResponse(request_id=request_id, status="blocked", final_answer=snapshot.values.get("final_answer"))
    return ChatResponse(request_id=request_id, status="completed", final_answer=snapshot.values.get("final_answer"))


@app.get("/approvals")
def list_pending_approvals(identity: Identity = Depends(require_reviewer)) -> list[dict]:
    return _approvals.list_pending(tenant_id=identity.tenant_id)


@app.get("/approvals/{request_id}/detail")
def approval_detail(request_id: str, identity: Identity = Depends(require_reviewer)) -> dict:
    pending = _approvals.get(request_id)
    if not pending or pending.get("tenant_id") != identity.tenant_id:
        raise HTTPException(status_code=404, detail="no such pending approval")

    config = {"configurable": {"thread_id": request_id}}
    snapshot = _graph.get_state(config)
    _audit.log(
        request_id=request_id, event_type="approval_detail_viewed",
        user_id=identity.user_id, role=identity.role, tenant_id=identity.tenant_id,
        payload={"viewer": identity.user_id},
    )
    return {
        "request_id": request_id,
        "draft_answer": snapshot.values.get("draft_answer"),
        "risk": snapshot.values.get("guardrail_risk"),
        "issues": snapshot.values.get("guardrail_issues"),
        "tools_used": snapshot.values.get("tools_used"),
        "recommended_action": snapshot.values.get("recommended_action"),
        "policy_violations": snapshot.values.get("policy_violations"),
        "unsupported_claims": snapshot.values.get("unsupported_claims"),
        "supported_claims": snapshot.values.get("supported_claims"),
    }


@app.post("/approvals/{request_id}", response_model=ChatResponse)
def decide_approval(
    request_id: str, req: ApprovalDecisionRequest, identity: Identity = Depends(require_reviewer),
) -> ChatResponse:
    pending = _approvals.get(request_id)
    if not pending or pending.get("tenant_id") != identity.tenant_id:
        raise HTTPException(status_code=404, detail="no pending approval for this request_id")
    if pending["status"] != "pending":
        raise HTTPException(status_code=409, detail="this request has already been decided")

    # Separation of duties, checked on both axes: role (require_reviewer,
    # above) and identity - whoever triggered a high-risk request cannot also
    # be the one who approves or rejects it, even an admin reviewing their
    # own submission.
    if pending.get("requester_user_id") and pending["requester_user_id"] == identity.user_id:
        raise HTTPException(status_code=403, detail="the requester cannot approve their own request")

    try:
        result = resume_request(_graph, request_id=request_id, approved=req.approved, approver=identity.user_id)
    except AlreadyDecidedError:
        # TOCTOU: another reviewer's decision landed between our pending-status
        # check above and the atomic UPDATE inside approvals.decide().
        raise HTTPException(status_code=409, detail="this request has already been decided")

    return ChatResponse(request_id=request_id, status="completed", final_answer=result.get("final_answer"))


@app.get("/audit/verify")
def audit_verify(identity: Identity = Depends(require_audit_access)) -> dict:
    is_valid, broken_entry_id = _audit.verify_chain()
    return {"valid": is_valid, "first_broken_entry_id": broken_entry_id}


@app.get("/audit/{request_id}")
def audit_trail(request_id: str, identity: Identity = Depends(require_identity)) -> list[dict]:
    trail = _audit.trail_for(request_id)
    if not trail:
        return []

    owns_request = any(event.get("user_id") == identity.user_id for event in trail)
    same_tenant = all(event.get("tenant_id") in (identity.tenant_id, None) for event in trail)
    is_reviewer = identity.role in ("admin", "compliance_officer")

    if not same_tenant:
        # Never reveal that a request from another tenant exists at all.
        raise HTTPException(status_code=404, detail="no such request")
    if not (owns_request or is_reviewer):
        raise HTTPException(status_code=403, detail="not authorized to view this request's audit trail")

    _audit.log(
        request_id=request_id, event_type="audit_viewed",
        user_id=identity.user_id, role=identity.role, tenant_id=identity.tenant_id,
        payload={"viewer": identity.user_id},
    )
    return trail
