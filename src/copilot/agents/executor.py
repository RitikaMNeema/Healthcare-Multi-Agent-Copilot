from copilot.config import FALLBACK_MODELS, MAX_TOOL_ITERATIONS, PRIMARY_MODEL
from copilot.fallback import call_with_fallback
from copilot.tools import registry
from copilot.tools.registry import ToolPermissionDenied

SYSTEM_PROMPT = """You are the execution agent for a governed enterprise copilot.

Use the available tools when they help answer the request accurately - prefer a
tool over guessing when the answer depends on a calculation or an internal
document. When you have enough information, respond with a final text answer
and stop calling tools. If a tool call is denied for permission reasons,
explain the limitation to the user instead of retrying it."""


def execute(llm, *, query: str, role: str, request_id: str, user_id: str,
            retrieved_context: list[dict], audit) -> tuple[str, str]:
    tool_specs = registry.tool_definitions_for_role(role)
    context_block = (
        "\n\n".join(f"[{c['source']}] {c['text']}" for c in retrieved_context)
        if retrieved_context else "No relevant context found."
    )
    messages: list[dict] = [{"role": "user", "content": f"Context:\n{context_block}\n\nRequest: {query}"}]
    used_model = PRIMARY_MODEL

    for _ in range(MAX_TOOL_ITERATIONS):
        def attempt(model: str, _messages=messages):
            return llm.complete_with_tools(system=SYSTEM_PROMPT, messages=_messages, tools=tool_specs, model=model)

        response, used_model = call_with_fallback(attempt, models=[PRIMARY_MODEL, *FALLBACK_MODELS])
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        if not tool_uses:
            text = next((block.text for block in response.content if block.type == "text"), "")
            return text, used_model

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_use in tool_uses:
            try:
                result = registry.invoke_tool(
                    tool_use.name, tool_use.input,
                    role=role, user_id=user_id, request_id=request_id, audit=audit,
                )
                tool_results.append({"type": "tool_result", "tool_use_id": tool_use.id, "content": str(result)})
            except ToolPermissionDenied as exc:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tool_use.id, "content": str(exc), "is_error": True,
                })
            except Exception as exc:  # noqa: BLE001 - surface any tool failure back to the model as an error result
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tool_use.id, "content": f"tool error: {exc}", "is_error": True,
                })
        messages.append({"role": "user", "content": tool_results})

    return "I was unable to complete this within the allowed tool-call budget.", used_model
