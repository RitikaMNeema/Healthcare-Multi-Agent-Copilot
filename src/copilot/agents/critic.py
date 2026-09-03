from copilot.config import FALLBACK_MODELS, PRIMARY_MODEL
from copilot.fallback import call_with_fallback
from copilot.guardrails.schemas import GuardrailVerdict

SYSTEM_PROMPT = """You are the safety and compliance critic for a governed healthcare
claims/denial-management copilot.

Review the draft answer below and assess risk before it is released to the user.
Flag anything that exposes patient-identifiable information beyond what routine
individual casework requires (a single claim lookup is normal; a bulk export or list
of multiple patients' identifying details is not), gives unsafe instructions, or
contradicts internal payer/HIPAA policy. `risk` must be exactly one of "low", "medium",
"high". Set `requires_approval` to true whenever risk is not "low"."""


def review(llm, draft_answer: str) -> tuple[GuardrailVerdict, str]:
    def attempt(model: str) -> GuardrailVerdict:
        return llm.complete_structured(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Draft answer to review:\n\n{draft_answer}"}],
            schema=GuardrailVerdict,
            model=model,
        )

    return call_with_fallback(attempt, models=[PRIMARY_MODEL, *FALLBACK_MODELS])
