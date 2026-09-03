"""Per-model token pricing, so a traced LLM call can carry its own cost.

Prices are USD per 1M tokens (Anthropic first-party API rates). An unknown
model returns $0 rather than raising - cost tracking should never be why a
request fails.
"""

MODEL_PRICING_PER_MILLION_TOKENS = {
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


def estimate_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float:
    pricing = MODEL_PRICING_PER_MILLION_TOKENS.get(model)
    if pricing is None or input_tokens is None or output_tokens is None:
        return 0.0
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
