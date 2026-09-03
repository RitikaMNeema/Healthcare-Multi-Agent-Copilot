from copilot.config import FALLBACK_MODELS, PRIMARY_MODEL
from copilot.fallback import call_with_fallback
from copilot.guardrails.schemas import PlannerOutput

SYSTEM_PROMPT = """You are the planning agent for a governed healthcare claims/denial-management copilot.

Classify the user's request and produce a short step-by-step plan.
`task_type` must be exactly one of:
  - "policy_lookup": needs payer/claims/appeals/prior-authorization/HIPAA policy documentation.
  - "claims_query": needs specific claim records (filter by payer, denial code, procedure, date, status).
  - "denial_analysis": needs an explanation of why one specific claim was denied and next steps.
  - "metrics": needs an aggregate statistic (denial rate, overturn rate, top denial codes, claim volume).
  - "remediation": needs a remediation plan for a denial pattern across a payer/procedure/denial code.
  - "general": can be answered directly (greetings, meta-questions about the copilot itself)."""


def plan(llm, query: str) -> tuple[PlannerOutput, str]:
    def attempt(model: str) -> PlannerOutput:
        return llm.complete_structured(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
            schema=PlannerOutput,
            model=model,
        )

    return call_with_fallback(attempt, models=[PRIMARY_MODEL, *FALLBACK_MODELS])
