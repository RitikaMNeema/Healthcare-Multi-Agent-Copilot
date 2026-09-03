# Copilot FAQ

This copilot can answer questions about internal engineering policy, run simple calculations, and look up documentation from the internal knowledge base. It cannot browse the live internet and cannot take any action outside of the tools it has been explicitly granted.

Access is role-based. Viewers can ask questions and search the knowledge base. Operators can additionally use the calculator tool. Admins can additionally read raw knowledge base files and can approve their own high-risk requests.

Any answer the copilot's guardrails flag as medium or high risk is held for human review before it is returned, unless the requester's role is allowed to auto-approve its own requests. Every request, tool call, guardrail verdict, and approval decision is written to an audit log that can be retrieved later.
