from copilot.config import FALLBACK_MODELS, PRIMARY_MODEL
from copilot.fallback import call_with_fallback
from copilot.guardrails.schemas import PlannerOutput

SYSTEM_PROMPT = """You are the planning agent for a governed enterprise copilot.

Classify the user's request and produce a short step-by-step plan.
`task_type` must be exactly one of:
  - "research": the request needs looking up policy or documentation.
  - "tool_task": the request needs a calculation or a specific file to be read.
  - "general": the request can be answered directly (greetings, meta-questions).

Set `needs_retrieval` to false only for pure "general" requests where knowledge-base
context would add nothing (e.g. small talk)."""


def plan(llm, query: str) -> tuple[PlannerOutput, str]:
    def attempt(model: str) -> PlannerOutput:
        return llm.complete_structured(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
            schema=PlannerOutput,
            model=model,
        )

    return call_with_fallback(attempt, models=[PRIMARY_MODEL, *FALLBACK_MODELS])
