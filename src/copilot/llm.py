"""LLM backend abstraction.

`ClaudeBackend` calls the real Anthropic API. `MockBackend` is a deterministic,
zero-dependency stand-in used by tests, the eval harness, and local demos when
no `ANTHROPIC_API_KEY` is configured - it makes the whole pipeline (planning,
tool routing, guardrail scoring, judging) runnable offline and reproducibly.

Both implement the same three-method surface, so every agent module is
written against the interface, never against `anthropic` directly.
"""
import re
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

from copilot.config import llm_backend_name
from copilot.rag.retriever import tokenize


class LLMBackend(Protocol):
    def complete_structured(self, *, system: str, messages: list[dict], schema: type[BaseModel],
                             model: str | None = None) -> BaseModel: ...

    def complete_with_tools(self, *, system: str, messages: list[dict], tools: list[dict],
                             max_tokens: int = 1024, model: str | None = None): ...


# --------------------------------------------------------------------------
# Claude backend
# --------------------------------------------------------------------------

class ClaudeBackend:
    def __init__(self, model: str | None = None):
        import anthropic

        from copilot.config import PRIMARY_MODEL

        self.default_model = model or PRIMARY_MODEL
        self.client = anthropic.Anthropic()

    def complete_structured(self, *, system, messages, schema, model=None):
        response = self.client.messages.parse(
            model=model or self.default_model,
            max_tokens=1024,
            system=system,
            messages=messages,
            output_format=schema,
        )
        return response.parsed_output

    def complete_with_tools(self, *, system, messages, tools, max_tokens=1024, model=None):
        return self.client.messages.create(
            model=model or self.default_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )


# --------------------------------------------------------------------------
# Mock backend - deterministic, offline, no network calls
# --------------------------------------------------------------------------

@dataclass
class Block:
    type: str
    text: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    id: str = ""


@dataclass
class FakeMessage:
    content: list[Block]
    stop_reason: str


_MATH_HINT_RE = re.compile(r"\d+\s*[-+*/]\s*\d+")
_MATH_EXPR_RE = re.compile(r"[-+]?\(?[\d.\s+\-*/()]{3,}\)?")
_FILE_RE = re.compile(r"[a-zA-Z0-9_\-]+\.md")
_GREETING_RE = re.compile(r"^\s*(hi|hello|hey)\b", re.I)


def _extract_text(messages: list[dict]) -> str:
    last = messages[-1]["content"]
    if isinstance(last, str):
        return last
    parts = []
    for block in last:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def _extract_context_and_query(user_text: str) -> tuple[str, str]:
    match = re.search(r"Context:\n(.*?)\n\nRequest: (.*)", user_text, re.S)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", user_text


def _extract_math_expression(text: str) -> str | None:
    if not _MATH_HINT_RE.search(text):
        return None
    for match in _MATH_EXPR_RE.finditer(text):
        candidate = match.group().strip()
        if any(op in candidate for op in "+-*/") and any(ch.isdigit() for ch in candidate):
            return candidate
    return None


def classify_task_type(query: str) -> str:
    lowered = query.lower()
    if _GREETING_RE.match(query):
        return "general"
    if _extract_math_expression(query) or "calculate" in lowered or "compute" in lowered:
        return "tool_task"
    if _FILE_RE.search(query) or "read the file" in lowered or "read file" in lowered:
        return "tool_task"
    question_starts = ("what", "how", "why", "who", "when", "where", "explain", "describe")
    if "?" in query or lowered.startswith(question_starts):
        return "research"
    return "general"


class MockBackend:
    def __init__(self, model: str | None = None):
        self.default_model = model or "mock"

    def complete_structured(self, *, system, messages, schema, model=None):
        name = schema.__name__
        text = _extract_text(messages)

        if name == "PlannerOutput":
            task_type = classify_task_type(text)
            steps = ["retrieve relevant context", "draft an answer", "run guardrail checks"]
            needs_retrieval = task_type != "general"
            return schema(task_type=task_type, steps=steps, needs_retrieval=needs_retrieval)

        if name == "GuardrailVerdict":
            from copilot.guardrails.output_guardrails import scan_output

            match = re.search(r"Draft answer to review:\n\n(.*)", text, re.S)
            draft = match.group(1) if match else text
            risk, issues = scan_output(draft)
            return schema(risk=risk, issues=issues, requires_approval=(risk != "low"),
                          rationale="mock backend: static keyword/PII heuristic scan")

        if name == "JudgeVerdict":
            match = re.search(r"Criteria for a good answer: (.*?)\n\nCopilot's answer:\n(.*)", text, re.S)
            criteria, answer = (match.group(1), match.group(2)) if match else ("", text)
            criteria_tokens = set(tokenize(criteria))
            answer_tokens = set(tokenize(answer))
            overlap = len(criteria_tokens & answer_tokens) / max(len(criteria_tokens), 1)
            if overlap > 0.5:
                score = 5
            elif overlap > 0.3:
                score = 4
            elif overlap > 0.15:
                score = 3
            elif answer.strip():
                score = 2
            else:
                score = 1
            verdict = "pass" if score >= 3 else "fail"
            return schema(score=score, verdict=verdict, rationale=f"mock backend: keyword overlap={overlap:.2f}")

        raise NotImplementedError(f"MockBackend cannot fabricate schema {name!r}")

    def complete_with_tools(self, *, system, messages, tools, max_tokens=1024, model=None):
        last_content = messages[-1]["content"]
        has_tool_result = isinstance(last_content, list) and any(
            isinstance(c, dict) and c.get("type") == "tool_result" for c in last_content
        )

        if has_tool_result:
            tool_result_text = next(
                c["content"] for c in last_content if isinstance(c, dict) and c.get("type") == "tool_result"
            )
            text = f"Based on the tool result, the answer is: {tool_result_text}"
            return FakeMessage(content=[Block(type="text", text=text)], stop_reason="end_turn")

        user_text = _extract_text(messages)
        context, query = _extract_context_and_query(user_text)
        tool_names = {t["name"] for t in tools}

        if _GREETING_RE.match(query):
            text = (
                "Hi! I can look up internal engineering policy in the knowledge base, run "
                "simple calculations, and - for admins - read specific knowledge-base files. "
                "Access is role-based: viewers can search and ask questions, operators can also "
                "use the calculator, and admins can read files and auto-approve their own "
                "high-risk requests. Anything flagged medium or high risk is held for human "
                "review unless your role is allowed to auto-approve it."
            )
            return FakeMessage(content=[Block(type="text", text=text)], stop_reason="end_turn")

        file_match = _FILE_RE.search(query)
        if file_match and "read_file" in tool_names:
            return FakeMessage(
                content=[Block(type="tool_use", name="read_file", input={"filename": file_match.group()}, id="mock_1")],
                stop_reason="tool_use",
            )

        expr = _extract_math_expression(query)
        if expr and "calculator" in tool_names:
            return FakeMessage(
                content=[Block(type="tool_use", name="calculator", input={"expression": expr}, id="mock_1")],
                stop_reason="tool_use",
            )

        if context and context != "No relevant context found.":
            text = f"Based on internal documentation: {context}"
        else:
            text = f"[mock] {query}"
        return FakeMessage(content=[Block(type="text", text=text)], stop_reason="end_turn")


def get_backend() -> LLMBackend:
    return MockBackend() if llm_backend_name() == "mock" else ClaudeBackend()
