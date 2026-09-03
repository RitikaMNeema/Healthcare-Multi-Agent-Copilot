"""Tool catalogue + the single choke point every tool call passes through.

Two layers of enforcement, deliberately redundant:
  1. `tool_definitions_for_role` means the model is never even offered a tool
     it isn't permitted to call.
  2. `invoke_tool` re-checks permission at execution time, so a bug in (1),
     a manually-crafted tool_use block, or a future caller that skips the
     role filter still can't execute an unauthorized tool.
Every invocation - permitted or denied - is written to the audit log.
"""
from copilot.governance import permissions
from copilot.tools.analyze_denial import analyze_denial
from copilot.tools.calculate_denial_metrics import VALID_METRICS, calculate_denial_metrics
from copilot.tools.claims_db import VALID_PAYERS
from copilot.tools.create_remediation_plan import create_remediation_plan
from copilot.tools.query_claims import query_claims
from copilot.tools.search_payer_policy import search_payer_policy

_PAYER_ENUM = sorted(VALID_PAYERS)
_DENIAL_CODE_ENUM = ["CO-16", "CO-50", "CO-97", "PR-1", "CO-197", "CO-29"]

TOOL_DEFINITIONS: dict[str, dict] = {
    "search_payer_policy": {
        "name": "search_payer_policy",
        "description": "Search internal payer, claims, appeals, prior-authorization, and HIPAA privacy policy documentation.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
    },
    "query_claims": {
        "name": "query_claims",
        "description": "Look up claims records with optional filters. Returns a total matching count plus up to `limit` rows.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "payer": {"type": ["string", "null"], "enum": [*_PAYER_ENUM, None]},
                "denial_code": {"type": ["string", "null"], "enum": [*_DENIAL_CODE_ENUM, None]},
                "procedure_code": {"type": ["string", "null"]},
                "status": {"type": ["string", "null"], "enum": ["paid", "denied", "appealed", None]},
                "start_date": {"type": ["string", "null"], "description": "YYYY-MM-DD, inclusive"},
                "end_date": {"type": ["string", "null"], "description": "YYYY-MM-DD, inclusive"},
                "limit": {"type": "integer"},
            },
            "required": ["payer", "denial_code", "procedure_code", "status", "start_date", "end_date", "limit"],
            "additionalProperties": False,
        },
    },
    "analyze_denial": {
        "name": "analyze_denial",
        "description": "Explain why a specific claim was denied, whether it's appealable, and recommended next steps.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"claim_id": {"type": "string"}},
            "required": ["claim_id"],
            "additionalProperties": False,
        },
    },
    "calculate_denial_metrics": {
        "name": "calculate_denial_metrics",
        "description": "Compute aggregate denial metrics: denial_rate, overturn_rate, top_denial_codes, or claim_volume, "
                        "optionally filtered by payer/procedure/denial code/date range.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": list(VALID_METRICS)},
                "payer": {"type": ["string", "null"], "enum": [*_PAYER_ENUM, None]},
                "procedure_code": {"type": ["string", "null"]},
                "denial_code": {"type": ["string", "null"], "enum": [*_DENIAL_CODE_ENUM, None]},
                "start_date": {"type": ["string", "null"]},
                "end_date": {"type": ["string", "null"]},
            },
            "required": ["metric", "payer", "procedure_code", "denial_code", "start_date", "end_date"],
            "additionalProperties": False,
        },
    },
    "create_remediation_plan": {
        "name": "create_remediation_plan",
        "description": "Build a remediation plan for a denial pattern (admin only): root-cause breakdown, "
                        "recommended actions, and the policy documents they're based on.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "payer": {"type": ["string", "null"], "enum": [*_PAYER_ENUM, None]},
                "denial_code": {"type": ["string", "null"], "enum": [*_DENIAL_CODE_ENUM, None]},
                "procedure_code": {"type": ["string", "null"]},
            },
            "required": ["payer", "denial_code", "procedure_code"],
            "additionalProperties": False,
        },
    },
}

TOOL_HANDLERS = {
    "search_payer_policy": lambda tool_input: search_payer_policy(tool_input["query"], tool_input.get("top_k", 3)),
    "query_claims": lambda tool_input: query_claims(**tool_input),
    "analyze_denial": lambda tool_input: analyze_denial(tool_input["claim_id"]),
    "calculate_denial_metrics": lambda tool_input: calculate_denial_metrics(**tool_input),
    "create_remediation_plan": lambda tool_input: create_remediation_plan(**tool_input),
}

# Tools that return individual claim-level data - used by the citation verifier
# to know which tool results a claim_id citation in the final answer could
# legitimately have come from.
CLAIM_LEVEL_TOOLS = frozenset({"query_claims", "analyze_denial"})


class ToolPermissionDenied(Exception):
    pass


class UnknownToolError(Exception):
    pass


def tool_definitions_for_role(role: str) -> list[dict]:
    allowed = permissions.allowed_tools(role)
    return [spec for name, spec in TOOL_DEFINITIONS.items() if name in allowed]


def invoke_tool(name: str, tool_input: dict, *, role: str, user_id: str, request_id: str, audit) -> object:
    if name not in TOOL_DEFINITIONS:
        raise UnknownToolError(f"no such tool: {name!r}")

    if name not in permissions.allowed_tools(role):
        audit.log(
            request_id=request_id, event_type="tool_denied", user_id=user_id, role=role,
            payload={"tool": name, "input": tool_input},
        )
        raise ToolPermissionDenied(f"role '{role}' is not permitted to use tool '{name}'")

    result = TOOL_HANDLERS[name](tool_input)
    audit.log(
        request_id=request_id, event_type="tool_invoked", user_id=user_id, role=role,
        payload={"tool": name, "input": tool_input, "result": result},
    )
    return result


def extract_claim_ids(tool_name: str, result: object) -> list[str]:
    """Pull every claim_id a tool result actually surfaced, for citation verification."""
    if tool_name == "analyze_denial" and isinstance(result, dict) and result.get("claim_id"):
        return [result["claim_id"]]
    if tool_name == "query_claims" and isinstance(result, dict):
        return [row["claim_id"] for row in result.get("claims", []) if "claim_id" in row]
    return []
