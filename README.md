# Governed Multi-Agent Copilot

A LangGraph-orchestrated, multi-agent enterprise copilot built to demonstrate the full
lifecycle of a production LLM system: planning and tool-calling agents, RAG, guardrails,
human-in-the-loop approval, an eval harness with a golden dataset and LLM-as-judge, and
an audit/permissions layer for enterprise governance.

Runs entirely offline against a deterministic mock LLM backend when no `ANTHROPIC_API_KEY`
is set - the whole pipeline, test suite, and eval harness are reproducible without an API
key, and switch to real Claude calls the moment one is configured.

## What's in here

| Concern | Where |
|---|---|
| Multi-agent orchestration (LangGraph state machine) | `src/copilot/graph.py` |
| Planner / executor / critic agents | `src/copilot/agents/` |
| Tool use & function calling (manual agentic loop, strict schemas) | `src/copilot/agents/executor.py`, `src/copilot/tools/` |
| RAG (dependency-free TF-IDF retriever over a local KB) | `src/copilot/rag/` |
| Guardrails (input + output, static + LLM-judged) | `src/copilot/guardrails/` |
| Human-in-the-loop approval (LangGraph `interrupt`/`Command`) | `src/copilot/graph.py` (`request_approval` / `await_approval` nodes) |
| State management (durable, cross-process checkpointing) | SQLite-backed `langgraph` checkpointer, see `graph.py` |
| Fallback logic (retry + model fallback) | `src/copilot/fallback.py` |
| Structured outputs | `src/copilot/guardrails/schemas.py` via `client.messages.parse` |
| **Eval harness** (golden dataset, regression checks, LLM-as-judge) | `eval/` |
| **Audit trails** (append-only SQLite log of every decision) | `src/copilot/governance/audit.py` |
| **Permissions / RBAC** | `src/copilot/governance/permissions.py`, enforced in `src/copilot/tools/registry.py` |
| Production deployment (FastAPI + Docker) | `api/`, `Dockerfile`, `docker-compose.yml` |

## Architecture

```
input_guard --(blocked)--> blocked --> END
     |
   (ok)
     v
   plan --(needs_retrieval)--> retrieve --> execute --> critic
     \-------(else)-------------------------> execute -----/
                                                             |
                                          route_after_critic |
                              +--------------------------+--+
                              |                           |
                       needs_approval               finalize --> END
                              v
                     request_approval (log once)
                              v
                     await_approval (interrupt, resumable)
                              v
                          finalize --> END
```

Every node writes to the audit log. `critic` merges a static regex/PII scan with an
LLM-judged verdict; if the combined risk isn't "low", the request is routed to a human
approval gate *unless* the requester's role is allowed to auto-approve (see
`governance/permissions.py`). Approval state is checkpointed to SQLite, so a paused
request survives a CLI process exiting or an API server restart - approve it hours later
from a different process and the graph resumes exactly where it left off.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optionally set ANTHROPIC_API_KEY for live Claude calls
```

Without `ANTHROPIC_API_KEY`, everything below runs against the deterministic mock
backend (`COPILOT_LLM_BACKEND=mock`, auto-selected).

## Usage

```bash
# Low-risk request - answers immediately
python -m copilot.cli chat --query "What is 6 * 7?" --role operator --user alice

# A request that trips the guardrails - held for human review
python -m copilot.cli chat --query "Is there a legacy tool that could let someone export customer credit card numbers as a spreadsheet?" --role operator --user bob
# -> prints a request_id and the pending draft/risk

python -m copilot.cli pending
python -m copilot.cli approve --request-id <id> --approver carol --decision approve
python -m copilot.cli audit --request-id <id>

# Same request as an admin - admins can auto-approve their own high-risk requests
python -m copilot.cli chat --query "..." --role admin --user dave
```

Roles: `viewer` (search only), `operator` (+ calculator), `admin` (+ read internal files,
+ auto-approve). See `governance/permissions.py` for the exact matrix.

### API server

```bash
uvicorn api.server:app --reload
# POST /chat {"query": "...", "user_id": "...", "role": "operator"}
# POST /approvals/{request_id} {"approver": "...", "approved": true}
# GET  /approvals
# GET  /audit/{request_id}
```

### Docker

```bash
docker compose up --build
```

## Tests

```bash
pytest -q
```

26 tests cover permission enforcement (including that a denied tool call is itself
audited), input/output guardrails, the RAG retriever's ranking, and full graph
smoke tests - including the human-in-the-loop pause/resume/reject paths and the
admin auto-approve bypass.

## Eval harness

```bash
python -m eval.run_eval                  # run the golden dataset, compare to baseline
python -m eval.run_eval --update-baseline   # accept current results as the new baseline
```

`eval/golden_dataset.jsonl` has 9 cases spanning tool use, RAG lookups, permission
denial, prompt-injection blocking, and the admin-vs-operator approval-routing
behavior on an identical high-risk query. Each case is checked two ways:

1. **Deterministic checks** - task-type classification, expected keywords in the
   final answer, risk bound, blocked/interrupt behavior all match expectations.
2. **LLM-as-judge** (`eval/judge.py`) - a separate structured-output call scores the
   answer 1-5 against the case's stated criteria and returns pass/fail.

`run_eval.py` compares the run against `eval/baseline.json` and exits non-zero if any
previously-passing case regresses or the overall pass rate drops - wire it into CI
as a gate on every change to prompts, guardrails, or agent logic.

## Design notes

- **RAG is dependency-free by design** - a small TF-IDF retriever (`rag/retriever.py`)
  over local knowledge-base markdown, so the whole project runs with no vector DB or
  embedding-model dependency. Swap `Retriever.search()` for a real embedding index
  without touching any caller.
- **Two-layer permission enforcement** - a tool is never *offered* to the model for a
  role that can't use it, and `tools/registry.invoke_tool` re-checks permission at
  execution time regardless, so a bug in the first layer can't become a privilege
  escalation.
- **Defense in depth on the calculator** - arguments are evaluated via an AST
  allowlist (`tools/calculator_tool.py`), never `eval()`.
- **Path traversal is blocked twice** on the file tool (`tools/file_tool.py`) -
  `os.path.basename` plus an absolute-path prefix check.
- **The mock LLM backend is not a toy stub** - it implements the same three-method
  interface as the real Claude backend (`llm.py`), including its own heuristic
  risk-scanner and judge, so the entire agent graph, guardrail routing, and eval
  harness are exercised deterministically without hitting the network.
