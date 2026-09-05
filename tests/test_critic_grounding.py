"""Tests that the critic actually receives full-turn context (not just the
draft answer in isolation) and that the graph escalates risk from claim-level
grounding findings even when the critic's own verdict under-calls it - the
hard safety-net pattern already used for unverified citations.

Uses a hand-written stub LLM (not the shared MockBackend) so each schema's
response is fully controlled and deterministic, rather than depending on
heuristics that can't reliably produce every verdict (e.g. MockBackend's
claim verifier never fabricates "contradicted" - see test_claim_verification.py).
"""
from langgraph.checkpoint.memory import MemorySaver

from copilot.agents import critic
from copilot.graph import build_graph, run_request
from copilot.guardrails.claim_verification import ClaimFinding, ClaimVerificationResult
from copilot.guardrails.schemas import GuardrailVerdict, PlannerOutput
from copilot.llm import Block, FakeMessage


class StubLLM:
    """Fixed, schema-dispatched responses - no heuristics, no network."""

    def __init__(self, *, claim_findings, guardrail_verdict, draft_answer="A grounded-sounding answer."):
        self.claim_findings = claim_findings
        self.guardrail_verdict = guardrail_verdict
        self.draft_answer = draft_answer
        self.critic_calls: list[dict] = []

    def complete_structured(self, *, system, messages, schema, model=None):
        if schema is PlannerOutput:
            return PlannerOutput(task_type="policy_lookup", steps=["search policy", "answer"])
        if schema is ClaimVerificationResult:
            return ClaimVerificationResult(findings=self.claim_findings)
        if schema is GuardrailVerdict:
            self.critic_calls.append({"messages": messages})
            return self.guardrail_verdict
        raise NotImplementedError(schema)

    def complete_with_tools(self, *, system, messages, tools, max_tokens=1024, model=None):
        return FakeMessage(content=[Block(type="text", text=self.draft_answer)], stop_reason="end_turn")


def test_critic_receives_question_plan_evidence_and_claim_findings():
    stub = StubLLM(
        claim_findings=[ClaimFinding(claim="x", verdict="supported", rationale="matches evidence")],
        guardrail_verdict=GuardrailVerdict(risk="low", issues=[], requires_approval=False, rationale="fine"),
    )
    verdict, _ = critic.review(
        stub, question="What is Aetna's prior auth policy for 97110?", task_type="policy_lookup",
        plan_steps=["search policy", "answer"], evidence_text="Aetna requires prior auth for 97110.",
        draft_answer="Aetna requires prior authorization for 97110.",
        claim_result=ClaimVerificationResult(findings=[ClaimFinding(claim="x", verdict="supported", rationale="ok")]),
    )
    assert verdict.risk == "low"
    sent_content = stub.critic_calls[0]["messages"][0]["content"]
    assert "What is Aetna's prior auth policy for 97110?" in sent_content
    assert "policy_lookup" in sent_content
    assert "Aetna requires prior auth for 97110." in sent_content
    assert "[supported] x" in sent_content


def _fresh_app(llm):
    return build_graph(llm=llm, checkpointer=MemorySaver())


def test_unsupported_claims_force_review_even_when_critic_says_low_risk():
    # The critic's own verdict claims low risk / no approval needed, but the
    # automated claim-grounding pass found a contradicted claim - graph.py's
    # hard safety net must still force at least medium risk and approval,
    # exactly like it already does for an unverified citation.
    stub = StubLLM(
        claim_findings=[ClaimFinding(claim="Aetna covers this with no restrictions", verdict="contradicted",
                                      rationale="evidence says the opposite")],
        guardrail_verdict=GuardrailVerdict(risk="low", issues=[], requires_approval=False, rationale="looks fine"),
        draft_answer="Aetna covers this with no restrictions.",
    )
    app = _fresh_app(stub)
    _, result = run_request(app, query="Does Aetna cover this?", user_id="u1", role="operator")

    assert result["guardrail_risk"] != "low"
    assert result["requires_approval"] is True
    assert "Aetna covers this with no restrictions" in result["unsupported_claims"]
    assert "__interrupt__" in result


def test_critic_recommended_block_is_terminal_not_sent_for_approval():
    # This is the exact bug the routing redesign closes: recommended_action
    # "block" must never reach a human reviewer who could release it - it has
    # to be a dead end, full stop, regardless of what `risk`/`requires_approval`
    # the critic also reported.
    stub = StubLLM(
        claim_findings=[ClaimFinding(claim="x", verdict="supported", rationale="ok")],
        guardrail_verdict=GuardrailVerdict(
            risk="low", issues=[], requires_approval=False, rationale="benign on its face",
            recommended_action="block",
        ),
    )
    app = _fresh_app(stub)
    _, result = run_request(app, query="Some request", user_id="u1", role="operator")

    assert result["guardrail_risk"] == "high"
    assert result["approval_status"] == "blocked_by_critic"
    assert "__interrupt__" not in result  # never even offered to a human reviewer
    assert result["final_answer"] != result["draft_answer"]  # the draft itself is never released


def test_a_blocked_request_cannot_be_released_by_resuming_it():
    # Defense in depth: even if something tried to resume a blocked_by_critic
    # request as though it were pending approval, there is no await_approval
    # interrupt waiting on this thread to resume - the graph already ran to
    # completion at blocked_by_critic -> END.
    stub = StubLLM(
        claim_findings=[ClaimFinding(claim="x", verdict="supported", rationale="ok")],
        guardrail_verdict=GuardrailVerdict(
            risk="high", issues=["severe policy violation"], requires_approval=True,
            rationale="must not be released", recommended_action="block",
        ),
    )
    app = _fresh_app(stub)
    request_id, result = run_request(app, query="Some request", user_id="u1", role="operator")
    assert result["approval_status"] == "blocked_by_critic"

    snapshot = app.get_state({"configurable": {"thread_id": request_id}})
    assert not snapshot.next  # graph has no paused node left to resume


def test_revise_loops_back_through_critic_then_falls_back_to_approval():
    stub = StubLLM(
        claim_findings=[ClaimFinding(claim="x", verdict="supported", rationale="ok")],
        guardrail_verdict=GuardrailVerdict(
            risk="medium", issues=["awkward phrasing"], requires_approval=True,
            rationale="needs a rewrite", recommended_action="revise",
        ),
        draft_answer="An answer that keeps needing revision.",
    )
    app = _fresh_app(stub)
    _, result = run_request(app, query="Some request", user_id="u1", role="operator")

    # The stub's revise() call always returns the same draft, so the critic
    # keeps saying "revise" - MAX_REVISIONS caps the loop instead of spinning
    # forever, and the outcome falls back to human review rather than ever
    # silently releasing an answer the critic never approved.
    assert result["revision_count"] == 2
    assert "__interrupt__" in result


def test_fully_supported_low_risk_answer_does_not_require_approval():
    stub = StubLLM(
        claim_findings=[ClaimFinding(claim="Aetna requires prior auth for 97110", verdict="supported",
                                      rationale="matches evidence")],
        guardrail_verdict=GuardrailVerdict(risk="low", issues=[], requires_approval=False, rationale="grounded"),
        draft_answer="Aetna requires prior auth for 97110.",
    )
    app = _fresh_app(stub)
    _, result = run_request(app, query="What does Aetna require?", user_id="u1", role="operator")

    assert result["guardrail_risk"] == "low"
    assert result["requires_approval"] is False
    assert result["unsupported_claims"] == []
    assert "__interrupt__" not in result
