"""LLM backend abstraction.

`ClaudeBackend` calls the real Anthropic API. `MockBackend` is a deterministic,
zero-dependency stand-in used by tests, the eval harness, and local demos when
no `ANTHROPIC_API_KEY` is configured - it makes the whole pipeline (planning,
tool routing, guardrail scoring, judging) runnable offline and reproducibly.

Both implement the same three-method surface, so every agent module is
written against the interface, never against `anthropic` directly.
"""
import ast
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

from pydantic import BaseModel

from copilot.config import llm_backend_name
from copilot.observability.cost import estimate_cost_usd
from copilot.rag.retriever import tokenize


@contextmanager
def _llm_span(tracing, span_name: str, model: str, **extra_attrs):
    """No-op if `tracing` is None, so every LLM call site can call this
    unconditionally. `record(usage)` sets token/cost attributes on the span
    if a real usage object was returned - the mock backend calls `record(None)`,
    which honestly reports no tokens/cost rather than fabricating numbers."""
    if tracing is None:
        yield lambda usage: None
        return
    with tracing.span(span_name, model=model, **extra_attrs) as span:
        def record(usage) -> None:
            if usage is None:
                return
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            if input_tokens is not None:
                span.set_attribute("input_tokens", input_tokens)
            if output_tokens is not None:
                span.set_attribute("output_tokens", output_tokens)
            if input_tokens is not None and output_tokens is not None:
                span.set_attribute("cost_usd", estimate_cost_usd(model, input_tokens, output_tokens))
        yield record


class LLMBackend(Protocol):
    def complete_structured(self, *, system: str, messages: list[dict], schema: type[BaseModel],
                             model: str | None = None) -> BaseModel: ...

    def complete_with_tools(self, *, system: str, messages: list[dict], tools: list[dict],
                             max_tokens: int = 1024, model: str | None = None): ...


# --------------------------------------------------------------------------
# Claude backend
# --------------------------------------------------------------------------

class ClaudeBackend:
    def __init__(self, model: str | None = None, tracing=None):
        import anthropic

        from copilot.config import PRIMARY_MODEL

        self.default_model = model or PRIMARY_MODEL
        self.client = anthropic.Anthropic()
        self.tracing = tracing

    def complete_structured(self, *, system, messages, schema, model=None):
        resolved_model = model or self.default_model
        with _llm_span(self.tracing, "llm.complete_structured", resolved_model, schema=schema.__name__) as record:
            response = self.client.messages.parse(
                model=resolved_model,
                max_tokens=1024,
                system=system,
                messages=messages,
                output_format=schema,
            )
            record(getattr(response, "usage", None))
            return response.parsed_output

    def complete_with_tools(self, *, system, messages, tools, max_tokens=1024, model=None):
        resolved_model = model or self.default_model
        with _llm_span(self.tracing, "llm.complete_with_tools", resolved_model) as record:
            response = self.client.messages.create(
                model=resolved_model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools,
            )
            record(getattr(response, "usage", None))
            return response


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


_GREETING_RE = re.compile(r"^\s*(hi|hello|hey)\b", re.I)

PAYER_NAMES = ["BlueCross BlueShield", "UnitedHealthcare", "Medicare", "Aetna"]
_PAYER_RE = re.compile("|".join(re.escape(p) for p in PAYER_NAMES), re.I)
_DENIAL_CODE_RE = re.compile(r"\b(CO|PR)-?(\d{1,3})\b", re.I)
_PROCEDURE_CODE_RE = re.compile(r"\b(\d{5}|[A-Z]\d{4})\b")
_CLAIM_ID_RE = re.compile(r"\bCLM-?(\d{6})\b", re.I)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_MONTH_YEAR_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.I)

REMEDIATION_KEYWORDS = ("remediation plan", "reduce denials", "reduce the denial", "fix this denial pattern", "root cause")
METRIC_KEYWORDS = ("denial rate", "overturn rate", "top denial code", "most common denial", "claim volume", "total billed")
CLAIMS_QUERY_KEYWORDS = (
    "list claims", "list the claims", "show me the claims", "show claims", "find claims",
    "how many claims", "look up claims", "look up the claims", "pull the claims",
)
POLICY_HINT_KEYWORDS = (
    "policy", "polic", "hipaa", "authorization", "appeal", "deadline", "requirement",
    "allowed", "rule", "timely filing", "minimum necessary", "breach notification",
)


def extract_payer(text: str) -> str | None:
    match = _PAYER_RE.search(text)
    if not match:
        return None
    matched = match.group()
    return next((p for p in PAYER_NAMES if p.lower() == matched.lower()), None)


def extract_denial_code(text: str) -> str | None:
    match = _DENIAL_CODE_RE.search(text)
    return f"{match.group(1).upper()}-{match.group(2)}" if match else None


def extract_procedure_code(text: str) -> str | None:
    match = _PROCEDURE_CODE_RE.search(text)
    return match.group(1) if match else None


def extract_claim_id(text: str) -> str | None:
    match = _CLAIM_ID_RE.search(text)
    return f"CLM-{match.group(1)}" if match else None


def extract_month_range(text: str) -> tuple[str | None, str | None]:
    match = _MONTH_YEAR_RE.search(text)
    if not match:
        return None, None
    month, year = _MONTHS[match.group(1).lower()], int(match.group(2))
    start = date(year, month, 1)
    end = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    return start.isoformat(), end.isoformat()


def classify_task_type(query: str) -> str:
    if _GREETING_RE.match(query):
        return "general"
    lowered = query.lower()
    if extract_claim_id(query):
        return "denial_analysis"
    if any(kw in lowered for kw in REMEDIATION_KEYWORDS):
        return "remediation"
    if any(kw in lowered for kw in METRIC_KEYWORDS):
        return "metrics"
    if any(kw in lowered for kw in CLAIMS_QUERY_KEYWORDS):
        return "claims_query"
    question_starts = ("what", "how", "why", "who", "when", "where", "explain", "describe", "is", "are", "does")
    if "?" in query or lowered.startswith(question_starts) or any(kw in lowered for kw in POLICY_HINT_KEYWORDS):
        return "policy_lookup"
    return "general"


def _decide_tool(query: str, tool_names: set[str]) -> tuple[str, dict] | None:
    claim_id = extract_claim_id(query)
    if claim_id and "analyze_denial" in tool_names:
        return "analyze_denial", {"claim_id": claim_id}

    lowered = query.lower()

    if any(kw in lowered for kw in REMEDIATION_KEYWORDS) and "create_remediation_plan" in tool_names:
        return "create_remediation_plan", {
            "payer": extract_payer(query), "denial_code": extract_denial_code(query),
            "procedure_code": extract_procedure_code(query),
        }

    metric = None
    if "denial rate" in lowered:
        metric = "denial_rate"
    elif "overturn rate" in lowered:
        metric = "overturn_rate"
    elif "top denial code" in lowered or "most common denial" in lowered:
        metric = "top_denial_codes"
    elif "claim volume" in lowered or "total billed" in lowered:
        metric = "claim_volume"
    if metric and "calculate_denial_metrics" in tool_names:
        start_date, end_date = extract_month_range(query)
        return "calculate_denial_metrics", {
            "metric": metric, "payer": extract_payer(query), "procedure_code": extract_procedure_code(query),
            "denial_code": extract_denial_code(query), "start_date": start_date, "end_date": end_date,
        }

    if any(kw in lowered for kw in CLAIMS_QUERY_KEYWORDS) and "query_claims" in tool_names:
        start_date, end_date = extract_month_range(query)
        status = next((s for s in ("denied", "paid", "appealed") if s in lowered), None)
        return "query_claims", {
            "payer": extract_payer(query), "denial_code": extract_denial_code(query),
            "procedure_code": extract_procedure_code(query), "status": status,
            "start_date": start_date, "end_date": end_date, "limit": 10,
        }

    if "search_payer_policy" in tool_names:
        return "search_payer_policy", {"query": query, "top_k": 3}

    return None


def _format_tool_result_answer(tool_name: str | None, tool_input: dict, tool_result_text: str) -> str:
    try:
        parsed = ast.literal_eval(tool_result_text)
    except (ValueError, SyntaxError):
        return f"Based on the tool result, the answer is: {tool_result_text}"

    if tool_name == "search_payer_policy" and isinstance(parsed, list):
        if not parsed:
            return "No relevant policy documentation was found."
        citations = "; ".join(f"[{hit['source']}] {hit['text']}" for hit in parsed)
        return f"Based on internal policy documentation: {citations}"

    if tool_name == "analyze_denial" and isinstance(parsed, dict):
        if parsed.get("denial_code") is None:
            return f"Claim {parsed['claim_id']} was not denied (status: {parsed.get('status')})."
        actions = " ".join(parsed.get("recommended_actions", []))
        return (
            f"Claim {parsed['claim_id']} ({parsed['payer']}, procedure {parsed['procedure_code']}) was denied "
            f"with code {parsed['denial_code']} ({parsed['denial_code_meaning']}). "
            f"Appealable: {parsed['is_appealable']}. Appeal filed: {parsed['appeal_filed']}, "
            f"outcome: {parsed.get('appeal_outcome')}. Recommended next steps: {actions}"
        )

    if tool_name == "query_claims" and isinstance(parsed, dict):
        claim_ids = ", ".join(c["claim_id"] for c in parsed.get("claims", []))
        return (
            f"Found {parsed['total_matching_count']} matching claims "
            f"(showing {parsed['returned_count']}: {claim_ids})."
        )

    if tool_name == "calculate_denial_metrics" and isinstance(parsed, dict):
        filters = {k: v for k, v in (tool_input or {}).items() if k != "metric" and v is not None}
        filter_desc = ", ".join(f"{k}={v}" for k, v in filters.items()) or "no filters"
        return f"Metric result for {filter_desc}: {parsed}"

    if tool_name == "create_remediation_plan" and isinstance(parsed, dict):
        refs = " ".join(f"[{r}]" for r in parsed.get("policy_references", []))
        actions = " ".join(parsed.get("recommended_actions", []))
        return (
            f"Remediation plan for {parsed.get('pattern_summary')}: "
            f"root-cause breakdown {parsed.get('denial_code_breakdown')}. "
            f"Recommended actions: {actions} Policy references: {refs}"
        )

    return f"Based on the tool result, the answer is: {tool_result_text}"


class MockBackend:
    def __init__(self, model: str | None = None, tracing=None):
        self.default_model = model or "mock"
        self.tracing = tracing

    def complete_structured(self, *, system, messages, schema, model=None):
        with _llm_span(self.tracing, "llm.complete_structured", self.default_model, schema=schema.__name__) as record:
            result = self._complete_structured_impl(system=system, messages=messages, schema=schema)
            record(None)  # mock: no real token usage to report
            return result

    def _complete_structured_impl(self, *, system, messages, schema):
        name = schema.__name__
        text = _extract_text(messages)

        if name == "PlannerOutput":
            task_type = classify_task_type(text)
            steps = {
                "policy_lookup": ["search payer/claims/appeals policy", "cite the relevant document"],
                "claims_query": ["query the claims database with the requested filters", "summarize matching claims"],
                "denial_analysis": ["look up the specific claim", "explain the denial and next steps"],
                "metrics": ["compute the requested aggregate metric", "report the result"],
                "remediation": ["aggregate the denial pattern", "draft recommended remediation actions"],
                "general": ["answer directly"],
            }[task_type]
            return schema(task_type=task_type, steps=steps)

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
        with _llm_span(self.tracing, "llm.complete_with_tools", self.default_model) as record:
            result = self._complete_with_tools_impl(system=system, messages=messages, tools=tools, max_tokens=max_tokens)
            record(None)  # mock: no real token usage to report
            return result

    def _complete_with_tools_impl(self, *, system, messages, tools, max_tokens=1024):
        last_content = messages[-1]["content"]
        has_tool_result = isinstance(last_content, list) and any(
            isinstance(c, dict) and c.get("type") == "tool_result" for c in last_content
        )

        if has_tool_result:
            prior_assistant = messages[-2]["content"]
            tool_use_block = next((b for b in prior_assistant if getattr(b, "type", None) == "tool_use"), None)
            tool_name = tool_use_block.name if tool_use_block else None
            tool_input = tool_use_block.input if tool_use_block else {}
            tool_result_text = next(
                c["content"] for c in last_content if isinstance(c, dict) and c.get("type") == "tool_result"
            )
            text = _format_tool_result_answer(tool_name, tool_input, tool_result_text)
            return FakeMessage(content=[Block(type="text", text=text)], stop_reason="end_turn")

        query = _extract_text(messages)
        tool_names = {t["name"] for t in tools}

        if _GREETING_RE.match(query):
            text = (
                "Hi! I can look up payer, claims, appeals, prior-authorization, and HIPAA privacy policy, "
                "query claims records, explain a specific claim's denial, compute aggregate denial metrics, "
                "and - for admins - build a remediation plan for a denial pattern. Access is role-based: "
                "viewers get policy search and aggregate metrics only, operators can also query individual "
                "claims and denials, and admins can additionally generate remediation plans and auto-approve "
                "their own high-risk requests. Anything flagged medium or high risk is held for human review "
                "unless your role is allowed to auto-approve it."
            )
            return FakeMessage(content=[Block(type="text", text=text)], stop_reason="end_turn")

        decision = _decide_tool(query, tool_names)
        if decision is None:
            return FakeMessage(content=[Block(type="text", text=f"[mock] {query}")], stop_reason="end_turn")

        tool_name, tool_input = decision
        return FakeMessage(
            content=[Block(type="tool_use", name=tool_name, input=tool_input, id="mock_1")],
            stop_reason="tool_use",
        )


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


def get_backend(tracing=None) -> LLMBackend:
    return MockBackend(tracing=tracing) if llm_backend_name() == "mock" else ClaudeBackend(tracing=tracing)
