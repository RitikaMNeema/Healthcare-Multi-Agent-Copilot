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
from copilot.tools.calculator_tool import calculate
from copilot.tools.file_tool import read_kb_file
from copilot.tools.search_tool import search_kb

TOOL_DEFINITIONS: dict[str, dict] = {
    "search_kb": {
        "name": "search_kb",
        "description": "Search the internal knowledge base for relevant policy or documentation snippets.",
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
    "calculator": {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression (+, -, *, /, ** and parentheses).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    "read_file": {
        "name": "read_file",
        "description": "Read a file from the knowledge base directory by filename (admin only).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
}

TOOL_HANDLERS = {
    "search_kb": lambda tool_input: search_kb(tool_input["query"], tool_input.get("top_k", 3)),
    "calculator": lambda tool_input: calculate(tool_input["expression"]),
    "read_file": lambda tool_input: read_kb_file(tool_input["filename"]),
}


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
